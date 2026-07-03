from typing import List, Optional, Tuple

from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager

from slbd.server.client_layer_allocator import ClientLayerAllocator
from slbd.common.typing import SocketTimingMetrics


class FedAvgClientLayerAllocator(ClientLayerAllocator):

    def __init__(
        self,
        num_model_layers,
        meaningful_split_points,
    ):
        _ = (meaningful_split_points, )
        self._num_model_layers = num_model_layers

    def initialize_epoch(
        self,
        epoch_clients: List[ClientProxy],
        client_manager: ClientManager,
        is_training: bool,
    ) -> None:
        _ = (client_manager, is_training, )
        return [self._num_model_layers] * len(epoch_clients)

    def finalize_epoch(self):
        return

    def raise_exception(self):
        raise Exception("Should not be called because no batch is ever to be sent to the server")

    def get_post_reallocation_layer_difference(self, client_id: str) -> Optional[Tuple[int, int]]:
        self.raise_exception()

    def get_client_offloaded_layer_count(self, client_id):
        self.raise_exception()

    def update_client_metrics(self, client_id: str, metrics: SocketTimingMetrics):
        self.raise_exception()

    def update_server_metrics(self, num_offloaded_layers: int, elapsed_time: float):
        self.raise_exception()

    def collect_epoch_metrics(self):
        return {}
