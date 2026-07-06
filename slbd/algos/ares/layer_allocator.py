import time
from logging import INFO
from typing import List, Optional, Tuple
from threading import Lock
from concurrent.futures import ThreadPoolExecutor

from flwr.common import GetPropertiesIns, log
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager

from slbd.algos.ares.utils import ares_compute_optimal_split_layer
from slbd.server.client_layer_allocator import ClientLayerAllocator
from slbd.common.typing import SocketTimingMetrics


class ARESClientLayerAllocator(ClientLayerAllocator):
    """
    Manages dynamic allocation of model layers between clients and server.

    This class determines how many layers each client should train locally
    versus on the server, based on client capabilities and performance metrics.
    """

    def __init__(
        self,
        client_profiling_frequency: int,
        server_fb_times: List[float],
        activation_sizes: List[int],
        client_throughputs_MBps: float,
        meaningful_split_points: List[int]
    ):
        self._client_profiling_frequency = client_profiling_frequency
        self._server_fb_times = server_fb_times
        self._activation_sizes = activation_sizes
        self._client_throughputs_MBps = client_throughputs_MBps
        self._epoch = 0
        self._client_allocated_layers = {}
        self.tot_server_computation_time = None
        self._num_model_layers = len(server_fb_times)
        self._computing_properties_time = 0.
        self._dict_lock = Lock()

    def _get_client_properties(self, client_proxy):
        get_properties_instructions = GetPropertiesIns({
            "iterations": 20,
            "dry_runs": 4,
        })

        result = client_proxy.get_properties(
            get_properties_instructions, timeout=None, group_id=None
        ).properties
        # the client dataset size appears in equation 3, but in truth the value does not
        # affect in any way the result -- it is just a multiplicative constant

        # extract the time from the result, which is a dictionary
        min_layers = result.pop("__min_number_of_layers__")
        result = {k: v for k, v in result.items() if k.startswith("f+b")}
        client_fb_times = [result[f"f+b{idx}"] for idx in range(len(result))]

        tmp = ares_compute_optimal_split_layer(
            client_fb_times,
            self._server_fb_times,
            self._activation_sizes,
            self._client_throughputs_MBps,
            min_client_layers=min_layers
        )[0]
        # enforce at least one layer on the client -- don't send raw data
        with self._dict_lock:
            self._client_allocated_layers[client_proxy.cid] = tmp

    def initialize_epoch(
        self,
        epoch_clients: List[ClientProxy],
        client_manager: ClientManager,
        is_training: bool,
    ) -> None:
        self._computing_properties_time = -1
        self.tot_server_computation_time = 0.

        if is_training and self._epoch % self._client_profiling_frequency == 0:
            start_time = time.time()
            log(INFO, "Triggering profiling of all clients and re-running optimization algorithm")
            # profile clients
            with ThreadPoolExecutor() as executor:
                executor.map(self._get_client_properties, list(client_manager.all().values()))

            log(INFO, "Re-allocation completed!")
            log(INFO, "New client layer allocations: %s", str(self._client_allocated_layers))
            reallocation_time = time.time() - start_time
            print("Rallocation time", reallocation_time)
            self._computing_properties_time = reallocation_time

        if is_training:
            self._epoch += 1

        return [self._client_allocated_layers[proxy.cid] for proxy in epoch_clients]

    def finalize_epoch(self):
        return

    def get_post_reallocation_layer_difference(self, client_id: str) -> Optional[Tuple[int, int]]:
        _ = (client_id, )
        return

    def get_client_offloaded_layer_count(self, client_id):
        return self._num_model_layers - self._client_allocated_layers[client_id]

    def update_client_metrics(self, client_id: str, metrics: SocketTimingMetrics):
        _ = (client_id, metrics)

    def update_server_metrics(
        self,
        num_offloaded_layers: int,
        elapsed_time: float,
    ):
        self.tot_server_computation_time += elapsed_time

    def collect_epoch_metrics(self):
        return {
            "train/allocated_layers": list(self._client_allocated_layers.items()),
            "train/model_profiling_time": self._computing_properties_time,
            "train/tot_server_training_time": self.tot_server_computation_time,
        }
