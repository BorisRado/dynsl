import copy
import collections

import torch

from slbd.server.server_model.manager import ServerModelManager
from slbd.torch.model_utils import extract_submodel_weights, get_num_layers


class BaseServerModelManager(ServerModelManager):

    def __init__(
        self,
        model,
        shuffle_data: bool,
    ) -> None:
        super().__init__()
        self.base_model = model
        self._shuffle_data = shuffle_data
        self.forking_data = self.initialize_forking_data()

    def initialize_forking_data(self):
        # forking data is a dictionary that for every layer contains a list of tuples
        # every tuple is a pair containing a set of clients that offloaded that layer
        # and the number of data points that were processed by the layer when the fork
        # happened
        return {layer_idx: [] for layer_idx in range(get_num_layers(self.base_model))}

    def _concat_and_shuffle_data(self, embeddings, labels):
        sizes = [b.shape[0] for b in labels]
        if isinstance(embeddings[0], torch.Tensor):
            embeddings = torch.cat(embeddings, dim=0)
        else:
            embeddings = tuple(
                torch.cat([b[i] for b in embeddings])
                for i in range(len(embeddings[0]))
            )

        labels = torch.cat(labels, dim=0)
        if self._shuffle_data:
            raise Exception("Not implemented yet")
        return embeddings, labels, (sizes,)

    def _split_and_unshuffle_data(self, loss, gradient, control):
        assert len(control) == 1
        sizes = control[0]
        return [(g, loss) for g in gradient.split(sizes, dim=0)]

    def get_global_model_params(self):
        return copy.deepcopy(self.base_model.state_dict())

    def extract_layer_weights(self, layer_from, layer_to):
        # returns a copy of the model weights
        raise Exception("")
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
