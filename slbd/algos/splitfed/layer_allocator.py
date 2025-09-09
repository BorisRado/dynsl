from typing import List, Optional, Tuple

import wandb
import numpy as np
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager

from slbd.server.client_layer_allocator import ClientLayerAllocator
from slbd.common.typing import TimingMetrics


class SplitFedClientLayerAllocator(ClientLayerAllocator):

    def __init__(
        self,
        num_model_layers,
        num_client_layers,
    ):
        self._num_client_layers = num_client_layers
        self._num_offloaded_layers = num_model_layers - num_client_layers
        self._metrics = []

    def initialize_epoch(
        self,
        epoch_clients: List[ClientProxy],
        client_manager: ClientManager,
        is_training: bool,
    ) -> None:
        _ = (client_manager, is_training, )
        return [self._num_client_layers] * len(epoch_clients)

    def finalize_epoch(self):
        if wandb.run is not None:
            wandb.log({"gpu_compute_time": np.mean(self._metrics)})
        print("Average GPU time:", np.mean(self._metrics))
        return

    def trigger_reallocation(self, client_ids) -> None:
        return

    def get_post_reallocation_layer_difference(self, client_id: str) -> Optional[Tuple[int, int]]:
        return

    def get_client_offloaded_layer_count(self, client_id):
        return self._num_offloaded_layers

    def update_client_metrics(self, client_id: str, metrics: TimingMetrics):
        return

    def update_server_metrics(self, num_offloaded_layers: int, elapsed_time: float):
        self._metrics.append(elapsed_time)
        return
