import copy
import collections
from typing import List, Union, Tuple, Dict

import torch
import torch.nn as nn
import torch.multiprocessing as mp

from slbd.server.server_model.manager.model_process import ComputeWorker, TaskType
from slbd.server.server_model.server_model import ServerModel
from slbd.server.server_model.torch_server_model import TorchServerModel
from slbd.server.server_model.manager import ServerModelManager
from slbd.common.typing import ServerModelEvaluateIns, ServerModelFitIns
from slbd.torch.model_utils import extract_submodel_weights, get_num_layers


class ModelProcessPool:
    def __init__(self, model, communication_type):
        assert communication_type in {"pipe", "queue"}
        self._use_pipe = communication_type == "pipe"
        self._idle_processes = []
        self._sid_to_active_process_mapping = {}
        self._model = model

    def reserve_process(self, sid):
        print(f"Reserving process for `{sid}`")
        if len(self._idle_processes) == 0:
            # create a new process ready to process requests
            print("Starting new process...")
            if self._use_pipe:
                parent_conn, child_conn = mp.Pipe()
                worker = ComputeWorker(
                    model=self._model,
                    connection=child_conn,
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                )
                self._idle_processes.append((worker, parent_conn))
            else:
                request_queue = mp.Queue()
                response_queue = mp.Queue()
                worker = ComputeWorker(
                    model=self._model,
                    request_queue=request_queue,
                    response_queue=response_queue,
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                )
                self._idle_processes.append((worker, request_queue, response_queue))
            worker.start()
        else:
            print("Reusing existing process...")

        # select the first available process and assign it to process requests for model sid
        selected_process_components = self._idle_processes[0]
        self._idle_processes = self._idle_processes[1:]
        self._sid_to_active_process_mapping[sid] = selected_process_components

    def release_process(self, sid):
        print(f"releasing resources for `{sid}`")
        print(self._sid_to_active_process_mapping)
        print("====")
        self._idle_processes.append(
            self._sid_to_active_process_mapping.pop(sid)
        )
        print(self._sid_to_active_process_mapping)
        print("====")

    def shutdown(self):
        assert len(self._sid_to_active_process_mapping) == 0
        for proc_data in self._idle_processes:
            request = (TaskType.SHUTDOWN, None)
            if self._use_pipe:
                process, connection = proc_data
                connection.send(request)
            else:
                process, request_queue, _ = proc_data
                request_queue.put(request)
            process.join()
        self._idle_processes = []

    def kill_all(self):
        for worker, *_ in self._idle_processes:
            if worker.is_alive():
                worker.terminate()
        for worker, *_ in self._sid_to_active_process_mapping.values():
            if worker.is_alive():
                worker.terminate()
        self._idle_processes.clear()
        self._sid_to_active_process_mapping.clear()

    def marshall_request(self, sid, request):
        if self._use_pipe:
            _, connection = self._sid_to_active_process_mapping[sid]
            connection.send(request)
            return connection.recv()  # blocking
        else:
            _, request_queue, response_queue = self._sid_to_active_process_mapping[sid]
            request_queue.put(request)
            return response_queue.get()

    def get_active_sids(self):
        return list(self._sid_to_active_process_mapping.keys())


class GrpcServerModelManager(ServerModelManager):

    def __init__(
        self,
        model: nn.Sequential,
        strategy: str
    ) -> None:
        super().__init__()
        assert strategy in {"mp_queue", "mp_pipe", "multithreading"}
        self.base_model = model
        if strategy.startswith("mp_"):
            self._multiprocessing = True
            self._model_process_pool = ModelProcessPool(model, communication_type=strategy[3:])
        else:
            self._multiprocessing = False
            self.spanned_server_models = {}

        # forking data is a dictionary that for every layer contains a list of tuples
        # every tuple is a pair containing a set of clients that offloaded that layer
        # and the number of data points that were processed by the layer when the fork
        # happened
        self.forking_data = {layer_idx: [] for layer_idx in range(get_num_layers(self.base_model))}

    def marshall_train_request(self, model_id,  embeddings, labels, num_offloaded_layers):
        sizes = [b.shape[0] for b in labels]
        embeddings = torch.cat([torch.from_numpy(b) for b in embeddings], dim=0)
        labels = torch.cat([torch.from_numpy(b) for b in labels], dim=0)

        # TODO: add permutation logic -- MIXER
        if self._multiprocessing:
            gradient, loss = self._model_process_pool.marshall_request(model_id, (
                TaskType.TRAIN,
                (embeddings, labels, num_offloaded_layers)
            ))
        else:
            gradient, loss = self.spanned_server_models[model_id].train_and_compute_gradient(
                embeddings, labels, num_offloaded_layers
            )

        # unshuffle

        return [(g.numpy(), loss) for g in gradient.split(sizes, dim=0)]

    def marshall_evaluation_request(self, model_id, embeddings, num_offloaded_layers):
        raise Exception("")
        return self._model_process_pool.marshall_request(model_id, (
            TaskType.EVALUATE,
            (embeddings, num_offloaded_layers)
        ))

    def collect_server_fit_results(
        self
    ) -> List[
        Tuple[collections.OrderedDict, Dict[str, float], str]
    ]:
        out = []
        if self._multiprocessing:
            for sid in self._model_process_pool.get_active_sids():
                res = self._model_process_pool.marshall_request(sid, (TaskType.GET_FIT_RESULT, None))
                res.forking_data = self.forking_data
                out.append(res)
                self._model_process_pool.release_process(sid)
        else:
            for sid, model in self.spanned_server_models.items():
                res = model.get_fit_result()
                res.forking_data = self.forking_data
                out.append(res)

        return out

    def end_round(self):
        if not self._multiprocessing:
            self.spanned_server_models = {}
        # self._model_process_pool.shutdown_all()
        pass

    def shutdown(self):
        if self._multiprocessing:
            self._model_process_pool.shutdown()

    def clean_resources(self):
        print("Shutting down...")
        if self._multiprocessing:
            self._model_process_pool.kill_all()

    def initialize_server_models(
        self,
        configs: Union[List[ServerModelFitIns], List[ServerModelEvaluateIns]],
        is_training: bool,
    ):
        self.forking_data = {layer_idx: [] for layer_idx in range(get_num_layers(self.base_model))}
        if len(configs) == 0:
            # no need for any server model
            return

        for config in configs:
            if self._multiprocessing:
                self._model_process_pool.reserve_process(config.sid)
                if is_training:
                    res = self._model_process_pool.marshall_request(
                        config.sid,
                        (TaskType.CONFIGURE_FIT, config)
                    )
                else:
                    res = self._model_process_pool.marshall_request(
                        config.sid,
                        (TaskType.CONFIGURE_EVALUATE, config)
                    )
                assert res is None
            else:
                model = TorchServerModel(self.base_model)
                if is_training:
                    model.configure_fit(config)
                else:
                    model.configure_evaluate(config)
                self.spanned_server_models[config.sid] = model

    def get_global_model_params(self):
        return copy.deepcopy(self.base_model.state_dict())

    def extract_layer_weights(self, layer_from, layer_to):
        # returns a copy of the model weights
        assert len(self.spanned_server_models) == 1 and "" in self.spanned_server_models
        model = self.spanned_server_models[""]
        shift = model.first_server_layer
        return extract_submodel_weights(
            model.get_model(), layer_from-shift, layer_to-shift
        )

    def create_fork(self, forking_clients_and_layers):
        # forking_clients_and_layers contains for every client idx a set of layers that
        # were sent to that client. For instance,
        # {1: [2,3], 2: [2]} means that client 1 was sent layer 2 and 3
        # client 2 was only sent layer 2

        # create an "inverse" dictionary, that for every layer, it states which
        # clients were offloaded this layer
        inverse_dictionary = collections.defaultdict(list)
        if not bool(forking_clients_and_layers):
            return

        # for now we assume that there is a single model on the server, but the logic
        # can in theory be generalized
        assert len(self.spanned_server_models) == 1 and "" in self.spanned_server_models

        for client, layers in forking_clients_and_layers.items():
            for layer in layers:
                inverse_dictionary[layer].append(client)

        for layer_idx in inverse_dictionary.keys():

            self.forking_data[layer_idx].append((
                inverse_dictionary[layer_idx],
                self.spanned_server_models[""].get_layer_processed_data_points(layer_idx)
            ))
