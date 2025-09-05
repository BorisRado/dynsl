import copy
import collections
from typing import List, Union, Tuple, Dict

import torch.nn as nn

from slbd.server.server_model.server_model import ServerModel
from slbd.server.server_model.manager import ServerModelManager
from slbd.server.server_model.torch_server_model import TorchServerModel
from slbd.common.typing import ServerModelEvaluateIns, ServerModelFitIns
from slbd.torch.model_utils import extract_submodel_weights, get_num_layers


class GrpcServerModelManager(ServerModelManager):

    def __init__(
        self,
        model: nn.Sequential,
    ) -> None:
        super().__init__()
        self.base_model = model
        self.client_locks = {}
        self.spanned_server_models = {}

        # forking data is a dictionary that for every layer contains a list of tuples
        # every tuple is a pair containing a set of clients that offloaded that layer
        # and the number of data points that were processed by the layer when the fork
        # happened
        self.forking_data = {layer_idx: [] for layer_idx in range(get_num_layers(self.base_model))}

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
            res.forking_data = self.forking_data
            out.append(res)
        return out

    def end_round(self):
        self.spanned_server_models = {}

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
