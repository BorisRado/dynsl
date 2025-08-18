import copy
import collections
from typing import List, Union, Tuple, Dict

import torch.nn as nn

from slbd.server.server_model.server_model import ServerModel
from slbd.server.server_model.manager import ServerModelManager
from slbd.server.server_model.torch_server_model import TorchServerModel
from slbd.common.typing import ServerModelEvaluateIns, ServerModelFitIns
from slbd.torch.model_utils import extract_submodel_weights


class GrpcServerModelManager(ServerModelManager):

    def __init__(
        self,
        model: nn.Sequential,
    ) -> None:
        super().__init__()
        self.base_model = model
        self.client_locks = {}
        self.spanned_server_models = {}

    def get_server_model(self, sid) -> ServerModel:
        return self.spanned_server_models[sid]

    def collect_server_fit_results(
        self
    ) -> List[
        Tuple[collections.OrderedDict, Dict[str, float], str]
    ]:
        out = []
        for sid, sm in self.spanned_server_models.items():
            res = sm.get_fit_result()
            res.sid = sid
            out.append(res)
        return out

    def end_round(self):
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
            sm = TorchServerModel(self.base_model)

            if is_training:
                sm.configure_fit(config)
            else:
                sm.configure_evaluate(config)

            self.spanned_server_models[config.sid] = sm

    def get_server_model_ids(self):
        return list(self.spanned_server_models.keys())

    def get_global_model_params(self):
        return copy.deepcopy(self.base_model.state_dict())

    def extract_layer_weights(self, layer_from, layer_to):
        # returns a copy of the model weights
        assert len(self.spanned_server_models) == 1 and "" in self.spanned_server_models
        return extract_submodel_weights(self.spanned_server_models[""].get_model(), layer_from, layer_to)
