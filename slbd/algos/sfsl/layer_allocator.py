from typing import List, Optional, Tuple

from flwr.common import GetPropertiesIns
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager

from slbd.server.client_layer_allocator import ClientLayerAllocator
from slbd.common.typing import SocketTimingMetrics
from slbd.common.constants import CLIENT_DEVICE_TYPE, CLIENT_DATASET_SIZE


class SFSLClientLayerAllocator(ClientLayerAllocator):

    def __init__(
        self,
        num_model_layers,
        num_client_layers,
        meaningful_split_points,
    ):
        _ = (meaningful_split_points, )
        self.num_model_layers = num_model_layers
        self.num_client_layers = num_client_layers
        self._cid_to_device_type_mapping = {}
        self._num_offloaded_layers = {}
        self.tot_server_computation_time = 0.

    def _get_properties(self, client_proxies):
        get_properties_instructions = GetPropertiesIns({})
        dataset_distributions = []
        for client_proxy in client_proxies:
            result = client_proxy.get_properties(
                get_properties_instructions, timeout=None, group_id=None
            )
            device_type = result.properties[CLIENT_DEVICE_TYPE]
            dataset_distributions.append(
                result.properties[CLIENT_DATASET_SIZE]
            )
            self._cid_to_device_type_mapping[client_proxy.cid] = device_type
            print(f"{client_proxy.cid} is device type {device_type}")
        print(f"=== Client distributions{sorted(dataset_distributions)}")

    def initialize_epoch(
        self,
        epoch_clients: List[ClientProxy],
        client_manager: ClientManager,
        is_training: bool,
    ) -> None:
        self._get_properties(epoch_clients)
        self.tot_server_computation_time = 0.
        self._num_offloaded_layers = {
            proxy.cid: self.num_model_layers - self.num_client_layers
            for proxy in epoch_clients
        }
        return [self.num_client_layers for _ in epoch_clients]

    def finalize_epoch(self):
        # if wandb.run is not None:
        #     wandb.log({"gpu_compute_time": np.mean(self._metrics)})
        # print("Average GPU time:", np.mean(self._metrics))
        return

    def get_post_reallocation_layer_difference(self, client_id: str) -> Optional[Tuple[int, int]]:
        return

    def get_client_offloaded_layer_count(self, client_id):
        return self._num_offloaded_layers[client_id]

    def update_client_metrics(self, client_id: str, metrics: SocketTimingMetrics):
        return

    def update_server_metrics(
        self,
        num_offloaded_layers: int,
        elapsed_time: float,
    ):
        self.tot_server_computation_time += elapsed_time

    def collect_epoch_metrics(self):
        return {
            "tot_server_training_time": self.tot_server_computation_time,
        }
