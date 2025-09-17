import copy
from typing import Union, List

import torch
import torch.nn as nn

from slbd.server.server_model.manager.multiprocessing.model_process_pool import (
    ModelProcessPoolWPipe,
    ModelProcessPoolWQueue,
    ModelProcessPool,
    ModelProcessPoolWPipeShareMemory
)
from slbd.torch.model_utils import extract_submodel
from slbd.server.server_model.manager.multiprocessing.model_worker import TaskType
from slbd.server.server_model.manager import BaseServerModelManager
from slbd.common.typing import ServerModelEvaluateIns, ServerModelFitIns, ServerModelFitRes


class MpServerModelManager(BaseServerModelManager):

    def __init__(
        self,
        model: nn.Sequential,
        communication_channel: str,
        reuse_procesess: bool,
        shuffle_data: bool,
        share_memory: bool,
        exp_config,
    ) -> None:
        super().__init__(model, shuffle_data)
        torch.multiprocessing.set_start_method("spawn")
        self.reuse_processes = reuse_procesess

        assert communication_channel in {"queue", "pipe"}

        if share_memory:
            assert communication_channel == "pipe"

        if communication_channel == "queue":
            self._worker_pool: ModelProcessPool = ModelProcessPoolWQueue(model)
        elif communication_channel == "pipe":
            if share_memory:
                m = extract_submodel(model, 0, exp_config.algorithm.layer_allocator_init_args.num_client_layers)
                embeddings_shape = tuple(m(torch.randn(2, 3, 32, 32)).shape[1:])
                batch_size = exp_config.client.init_kwargs.batch_size * exp_config.server_model_servicer.num_concat_batches
                embeddings_shape = (batch_size,) + embeddings_shape
                print("embeddings shape", embeddings_shape)
                shared_tensor_metadata = [
                    ("embeddings", embeddings_shape, torch.float32),
                    ("gradient", embeddings_shape, torch.float32),
                    ("labels", (batch_size,), torch.long),
                ]
                print("Tensor data", shared_tensor_metadata)
                self._worker_pool: ModelProcessPool = ModelProcessPoolWPipeShareMemory(
                    model, shared_tensor_metadata
                )
            else:
                self._worker_pool: ModelProcessPool = ModelProcessPoolWPipe(model)

    def marshall_train_request(self, model_id,  embeddings, labels, num_offloaded_layers):
        embeddings, labels, control = self._concat_and_shuffle_data(embeddings, labels)
        gradient, loss = self._worker_pool.marshall_request(model_id, (
            TaskType.TRAIN,
            (embeddings, labels, num_offloaded_layers)
        ))
        return self._split_and_unshuffle_data(loss, gradient, control)

    def marshall_evaluation_request(self, model_id, embeddings, num_offloaded_layers):
        raise Exception("")
        return self._worker_pool.marshall_request(model_id, (
            TaskType.EVALUATE,
            (embeddings, num_offloaded_layers)
        ))

    def collect_server_fit_results(
        self
    ) -> List[ServerModelFitRes]:
        out = []
        for sid in self._worker_pool.get_active_sids():
            res: ServerModelFitRes = \
                self._worker_pool.marshall_request(sid, (TaskType.GET_FIT_RESULT, None))
            res.forking_data = self.forking_data
            out.append(res)
            self._worker_pool.release_process(sid)

        return out

    def end_round(self):
        if not self.reuse_processes:
            self.shutdown()

    def shutdown(self):
        self._worker_pool.shutdown()

    def clean_resources(self):
        self._worker_pool.kill_all()

    def initialize_server_models(
        self,
        configs: Union[List[ServerModelFitIns], List[ServerModelEvaluateIns]],
        is_training: bool,
    ):
        self.forking_data = self.initialize_forking_data()
        if len(configs) == 0:
            # no need for any server model
            return

        for config in configs:
            self._worker_pool.reserve_process(config.sid)
            if is_training:
                res = self._worker_pool.marshall_request(
                    config.sid,
                    (TaskType.CONFIGURE_FIT, config)
                )
            else:
                res = self._worker_pool.marshall_request(
                    config.sid,
                    (TaskType.CONFIGURE_EVALUATE, config)
                )
            assert res is None

    def get_global_model_params(self):
        return copy.deepcopy(self.base_model.state_dict())

    def extract_layer_weights(self, layer_from, layer_to):
        # returns a copy of the model weights
        raise Exception("TO-DO")
        assert len(self.spanned_server_models) == 1 and "" in self.spanned_server_models
        model = self.spanned_server_models[""]
        shift = model.first_server_layer
        return extract_submodel_weights(
            model.get_model(), layer_from-shift, layer_to-shift
        )
