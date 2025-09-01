from typing import List, Optional, Tuple

from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager

from slbd.server.client_layer_allocator import ClientLayerAllocator
from slbd.common.typing import TimingMetrics


class SplitFedClientLayerAllocator(ClientLayerAllocator):

    def __init__(
        self,
        num_client_layers,
    ):
        self._num_client_layers = num_client_layers

    def initialize_epoch(
        self,
        epoch_clients: List[ClientProxy],
        client_manager: ClientManager,
        is_training: bool,
    ) -> None:
        _ = (client_manager, is_training, )
        return [self._num_client_layers] * len(epoch_clients)

    def finalize_epoch(self):
        return

    def trigger_reallocation(self, client_ids) -> None:
        return

    def get_post_reallocation_layer_difference(self, client_id: str) -> Optional[Tuple[int, int]]:
        return

    def get_client_offloaded_layer_count(self, client_id):
        return

    def update_client_metrics(self, client_id: str, metrics: TimingMetrics):
        return

    def update_server_metrics(self, num_offloaded_layers: int, elapsed_time: float):
        return
