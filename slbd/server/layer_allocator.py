import copy
from logging import INFO
from typing import List

from flwr.common import GetPropertiesIns, log
from flwr.server.client_proxy import ClientProxy

from slbd.common.constants import MIN_NUMBER_OF_CLIENT_LAYERS


class LayerAllocator:

    def __init__(self, max_client_layers: int):
        self._per_client_min_layers = {}
        self.max_client_layers = max_client_layers

    def get_client_min_layers(self, client_proxy):
        if client_proxy.cid not in self._per_client_min_layers:
            # call get properties on client
            noins = GetPropertiesIns({})
            res = client_proxy.get_properties(noins, timeout=None, group_id=None)
            min_client_layers = res.properties[MIN_NUMBER_OF_CLIENT_LAYERS]
            self._per_client_min_layers[client_proxy.cid] = min_client_layers
            log(INFO, f"{client_proxy.cid} requires at least {min_client_layers}")
        return self._per_client_min_layers[client_proxy.cid]

    def get_client_suggested_init_layers(self, client_proxy):
        # if client is unknown, suggest the minimum number of layers the client must train
        # otherwise, use prior knowledge to define the number of layers to be set at the
        # beginning of training to every client

        # for now, we do not keep track previous statistics of clients, so we always suggest the
        # minimum number of layers
        return self.get_client_min_layers(client_proxy)

    def init_epoch(
        self,
        epoch_clients: List[ClientProxy],
        is_training: bool,
    ):
        self.epoch_clients = [client_proxy.cid for client_proxy in epoch_clients]
        self.is_training = is_training

        # _client_layer_diff stores the temporary difference in the number of layers
        # of clients
        self._client_layer_diff = {cid: 0 for cid in self.epoch_clients}

        min_client_layers = min([
            self.get_client_suggested_init_layers(client_proxy) for client_proxy in epoch_clients
        ])

        # _offsets_from_base stores how many layers should we skip on the server model
        self._offsets_from_base = {
            proxy.cid: self.get_client_suggested_init_layers(proxy) - min_client_layers for proxy in epoch_clients
        }

        # _per_client_processed_batches - not really useful. Just for debugging
        self._per_client_processed_batches = {proxy.cid: 0 for proxy in epoch_clients}

        # _client_allocated_layers - stores the number of layers every client has
        # at the beginning of every epoch, the number of layers is equal to the suggested value
        # be the allocator
        self._client_allocated_layers = {
            proxy.cid: self.get_client_suggested_init_layers(proxy) for proxy in epoch_clients
        }

    def epoch_end(self):
        self.epoch_clients = None
        self.min_client_layers = None
        self.max_layers = None
        self._client_allocated_layers = None

    def trigger_reallocation(self) -> None:
        for cid in self.epoch_clients:
            if self._per_client_processed_batches[cid] == 5:
                self._client_layer_diff[cid] = 3
            if self._per_client_processed_batches[cid] == 10:
                self._client_layer_diff[cid] = -3
            # send three additional layers to client... just testing for now

    def get_client_local_layers(self, cid):
        return self._client_allocated_layers[cid]

    def get_layers_diff(self, cid):
        previous = self._client_allocated_layers[cid]
        if self._client_layer_diff[cid] != 0:
            diff = self._client_layer_diff[cid]
            self._client_allocated_layers[cid] += diff
            self._offsets_from_base[cid] += diff
            self._client_layer_diff[cid] = 0
            return previous, diff
        else:
            return None

    def update_metrics(self, cid, metrics):
        _ = (metrics, )
        self._per_client_processed_batches[cid] += 1

    def get_client_offset_from_base(self, cid):
        return self._offsets_from_base[cid]
