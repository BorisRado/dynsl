import copy
import collections
from typing import Union, List

import torch.nn as nn

from slbd.server.server_model.torch_server_model import TorchServerModel
from slbd.server.server_model.manager.multiprocessing.model_worker import TaskType
from slbd.server.server_model.manager import BaseServerModelManager
from slbd.common.typing import ServerModelEvaluateIns, ServerModelFitIns, ServerModelFitRes


class MultithreadingServerModelManager(BaseServerModelManager):

    def __init__(
        self,
        model: nn.Sequential,
        shuffle_data: bool,
    ) -> None:
        super().__init__(model, shuffle_data)
        self.spanned_server_models = {}

    def marshall_train_request(self, model_id,  embeddings, labels, num_offloaded_layers):
        embeddings, labels, control = self._concat_and_shuffle_data(embeddings, labels)
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
            res.forking_data = self.forking_data
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
        self.forking_data = self.initialize_forking_data()
        if len(configs) == 0:
            # no need for any server model
            return

        for config in configs:
            sm = TorchServerModel(self.base_model)
            if is_training:
                sm.configure_fit(config)
            else:
                sm.configure_evaluate(config)
            self.spanned_server_models[config.sid] = sm

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
