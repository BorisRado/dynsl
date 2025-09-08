from logging import INFO
from typing import List, Dict, Optional, Tuple

from flwr.common import GetPropertiesIns, log
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager

from slbd.common.typing import TimingMetrics
from slbd.common.constants import MIN_NUMBER_OF_CLIENT_LAYERS


class DynSLClientLayerAllocator:

    def __init__(self, max_client_layers: int, num_model_layers: int):
        """
        Initialize the LayerAllocator.

        Args:
            max_client_layers: Maximum number of layers a client can handle
        """
        self._per_client_min_layers: Dict[str, int] = {}
        self._max_client_layers = max_client_layers
        self._num_model_layers = num_model_layers

        # Configuration for reallocation logic
        self._reallocation_config = {
            "first_trigger_batch": 5,
            # "second_trigger_batch": 10,
            "layers_to_add": 2,
            # "layers_to_remove": 3
        }

    def get_client_min_layers(self, client_proxy: ClientProxy) -> int:
        if client_proxy.cid not in self._per_client_min_layers:
            # Call get_properties on client to fetch requirements
            get_properties_instructions = GetPropertiesIns({})
            result = client_proxy.get_properties(
                get_properties_instructions, timeout=None, group_id=None
            )
            min_client_layers = 2 # int(result.properties[MIN_NUMBER_OF_CLIENT_LAYERS])
            self._per_client_min_layers[client_proxy.cid] = min_client_layers
            log(INFO, f"Client {client_proxy.cid} requires at least {min_client_layers} layers")
        return self._per_client_min_layers[client_proxy.cid]

    def get_initial_layer_count(self, client_proxy: ClientProxy) -> int:
        # TODO: Use historical performance data for smarter allocation
        return self.get_client_min_layers(client_proxy)

    def initialize_epoch(
        self,
        epoch_clients: List[ClientProxy],
        client_manager: ClientManager,
        is_training: bool,
    ):
        _ = (client_manager, )
        self.tmp_update_client = epoch_clients[0].cid
        self._epoch_clients = [client_proxy.cid for client_proxy in epoch_clients]
        self._is_training = is_training

        # _client_layer_diff stores the temporary difference in the number of layers
        # of clients
        self._client_layer_diff = {cid: 0 for cid in self._epoch_clients}

        # _per_client_processed_batches - not really useful. Just for debugging
        self._per_client_processed_batches = {proxy.cid: 0 for proxy in epoch_clients}

        # _client_allocated_layers - stores the number of layers every client has
        # at the beginning of every epoch, the number of layers is equal to the suggested value
        # be the allocator
        self._client_allocated_layers = {
            proxy.cid: self.get_initial_layer_count(proxy) for proxy in epoch_clients
        }
        return [self._client_allocated_layers[proxy.cid]  for proxy in epoch_clients]

    def finalize_epoch(self):
        self._epoch_clients = None
        self.min_client_layers = None
        self.max_layers = None
        self._client_allocated_layers = None

    def trigger_reallocation(self, client_ids) -> None:
        """
        Trigger layer reallocation based on performance metrics.

        Note: This is a placeholder implementation. In production,
        this should use sophisticated algorithms based on timing metrics,
        client capabilities, and server load.
        """
        for client_id in client_ids:
            if client_id != self.tmp_update_client:
                continue
            batches_processed = self._per_client_processed_batches[client_id]
            if batches_processed == self._reallocation_config["first_trigger_batch"]:
                self._client_layer_diff[client_id] = self._reallocation_config["layers_to_add"]
            # elif batches_processed == self._reallocation_config["second_trigger_batch"]:
            #     self._client_layer_diff[client_id] = -self._reallocation_config["layers_to_remove"]

    def get_client_offloaded_layer_count(self, client_id):
        return self._num_model_layers - self._client_allocated_layers[client_id]

    def get_post_reallocation_layer_difference(self, client_id: str) -> Optional[Tuple[int, int]]:
        previous = self._client_allocated_layers[client_id]
        if self._client_layer_diff[client_id] != 0:
            diff = self._client_layer_diff[client_id]
            self._client_allocated_layers[client_id] += diff
            self._client_layer_diff[client_id] = 0
            return previous, diff
        else:
            return None

    def update_client_metrics(self, client_id: str, metrics: TimingMetrics):
        _ = (metrics, )
        self._per_client_processed_batches[client_id] += 1

    def update_server_metrics(self, num_offloaded_layers: int, elapsed_time: float):
        # print(num_offloaded_layers, elapsed_time)
        pass
