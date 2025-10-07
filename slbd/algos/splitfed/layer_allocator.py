from typing import List, Optional, Tuple

import wandb
import numpy as np
from flwr.common import GetPropertiesIns
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager

from slbd.server.client_layer_allocator import ClientLayerAllocator
from slbd.common.typing import TimingMetrics
from slbd.common.constants import CLIENT_DEVICE_TYPE



class SplitFedClientLayerAllocator(ClientLayerAllocator):

    def __init__(
        self,
        num_model_layers,
        num_client_layers,
    ):
        self._num_client_layers = num_client_layers
        self._num_offloaded_layers = num_model_layers - num_client_layers

        self._server_training_time = []
        self._wait_time_in_ready_state = []
        self._batch_preparation_times = []
        self._cid_to_device_type_mapping = {}

    def _get_properties(self, client_proxies):
        get_properties_instructions = GetPropertiesIns({})
        for client_proxy in client_proxies:
            result = client_proxy.get_properties(
                get_properties_instructions, timeout=None, group_id=None
            )
            device_type = result.properties[CLIENT_DEVICE_TYPE]
            self._cid_to_device_type_mapping[client_proxy.cid] = device_type
            print(f"{client_proxy.cid} is device type {device_type}")

    def initialize_epoch(
        self,
        epoch_clients: List[ClientProxy],
        client_manager: ClientManager,
        is_training: bool,
    ) -> None:
        _ = (client_manager, is_training, )
        if len(self._cid_to_device_type_mapping) == 0:
            self._get_properties(epoch_clients)

        self._server_training_time = []
        self._wait_time_in_ready_state = []
        self._batch_preparation_times = []
        return [self._num_client_layers] * len(epoch_clients)

    def finalize_epoch(self):
        # if wandb.run is not None:
        #     wandb.log({"gpu_compute_time": np.mean(self._metrics)})
        # print("Average GPU time:", np.mean(self._metrics))
        return

    def get_post_reallocation_layer_difference(self, client_id: str) -> Optional[Tuple[int, int]]:
        return

    def get_client_offloaded_layer_count(self, client_id):
        return self._num_offloaded_layers

    def update_client_metrics(self, client_id: str, metrics: TimingMetrics):
        return

    def update_server_metrics(
        self,
        num_offloaded_layers: int,
        elapsed_time: float,
        wait_time_in_ready_state: float,
        batch_preparation_time: float
    ):
        self._server_training_time.append(elapsed_time)
        self._wait_time_in_ready_state.append(wait_time_in_ready_state)
        self._batch_preparation_times.append(batch_preparation_time)

    def collect_epoch_metrics(self):
        return {
            "training_times": self._server_training_time,
            "wait_in_ready_state": self._wait_time_in_ready_state,
            "batch_preparation": self._batch_preparation_times,
        }
