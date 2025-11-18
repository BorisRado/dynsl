import copy
import asyncio
from typing import Union, List

import torch

from slbd.common.parameter import torch_list_to_bytes
from slbd.server.server_model.torch_server_model import TorchServerModel
from slbd.server.server_model.manager.multiprocessing.model_worker import TaskType
from slbd.server.server_model.manager import BaseServerModelManager
from slbd.common.typing import ServerModelEvaluateIns, ServerModelFitIns, ServerModelFitRes
from slbd.torch.model_utils import extract_submodel_weights
from slbd.torch.sequential_model import FlexibleSequential


class MultithreadingServerModelManager(BaseServerModelManager):

    def __init__(
        self,
        model: FlexibleSequential,
        exp_config,
    ) -> None:
        super().__init__(model, )
        _ = (exp_config,)
        self.spanned_server_models = {}
        self._model_locks = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # caching -- if parameters were extracted very recently, instead of extracting them again,
        # and converting them to bytes from scratch, we re-use recently extracted ones -- Faster

    async def marshall_train_request(self, model_id,  embeddings, labels, num_offloaded_layers):
        embeddings, labels, control = self._concat_and_shuffle_data(embeddings, labels)
        if isinstance(embeddings, tuple):
            embeddings = tuple(e.to(self.device, non_blocking=True) for e in embeddings)
        else:
            embeddings = embeddings.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        async with self._model_locks[model_id]:
            gradient, loss = self.spanned_server_models[model_id].train_and_compute_gradient(
                embeddings, labels, num_offloaded_layers
            )
        return self._split_and_unshuffle_data(loss, gradient, control)

    def marshall_evaluation_request(self, model_id, embeddings, num_offloaded_layers):
        raise Exception("")
        return self._model_process_pool.marshall_request(model_id, (
            TaskType.EVALUATE,
            (embeddings, num_offloaded_layers)
        ))

    def collect_server_fit_results(
        self
    ) -> List[ServerModelFitRes]:
        out = []
        for sid, model in self.spanned_server_models.items():
            res: ServerModelFitRes = model.get_fit_result()
            res.sid = sid
            res.forking_data = {}
            out.append(res)

        return out

    def end_round(self):
        self.spanned_server_models = {}

    def shutdown(self):
        self.spanned_server_models = {}

    def clean_resources(self):
        self.spanned_server_models = {}

    def initialize_server_models(
        self,
        configs: Union[List[ServerModelFitIns], List[ServerModelEvaluateIns]],
        is_training: bool,
    ):
        if len(configs) == 0:
            # no need for any server model
            return

        for config in configs:
            sm = TorchServerModel(self.base_model, device=self.device)
            if is_training:
                sm.configure_fit(config)
            else:
                sm.configure_evaluate(config)
            self.spanned_server_models[config.sid] = sm
            self._model_locks[config.sid] = asyncio.Lock()

    def get_global_model_params(self):
        return copy.deepcopy(self.base_model.state_dict())

    async def extract_layer_weights(self, layer_from, layer_to):
        # returns a copy of the model weights
        assert len(self.spanned_server_models) == 1 and "" in self.spanned_server_models
        model = self.spanned_server_models[""]
        shift = model.first_server_layer
        async with self._model_locks[""]:
            params = extract_submodel_weights(
                model.get_model(), layer_from-shift, layer_to-shift
            )
        params = torch_list_to_bytes(params)
        return params

    def get_num_server_models(self):
        assert len(self.spanned_server_models) == len(self._model_locks)
        return len(self.spanned_server_models)
