import time
from logging import INFO
from typing import List, Dict

from slbd.algos.dynsl.layer_allocator import DynSLClientLayerAllocator


class DynSLClientLayerAllocatorV2(DynSLClientLayerAllocator):

    def __init__(
        self,
        max_client_layers: int,
        num_model_layers: int,
        reallocation_frequency: float,
        num_concat_batches: int,
        meaningful_split_points: List[int],
        cache_client_configuration: bool,
        modeled_layer_complexities: List[float]
    ):
        """
        Initialize the LayerAllocator.

        Args:
            max_client_layers: Maximum number of layers a client can handle
            num_model_layers: Total number of model layers
            alpha: smoothing factor for layer adjustment
        """
        self._per_client_min_layers: Dict[str, int] = {}
        self._num_model_layers: int = num_model_layers

        # per-client dataset and pending batches
        self._dataset_sizes: Dict[str, int] = {}
        self._pending_batches: Dict[str, int] = {}

        # rolling averages
        self._client_training_times: Dict[str, float] = {}
        self._client_round_times: Dict[str, float] = {}
        self._server_training_times: Dict[int, float] = {}

        # we will call the reallocation function with the following frequency
        self.reallocation_frequency = reallocation_frequency
        self._reallocation_stop_event = None
        self._num_concat_batches = num_concat_batches
        self.meaningful_split_points = [msp for msp in meaningful_split_points if msp <= max_client_layers]
        self._cid_to_device_type_mapping = {}

        self.cache_client_configuration = cache_client_configuration
        self._cached_num_layers = {}
        print("Caching?", self.cache_client_configuration)
        self._max_client_layers: int = self.get_previous_split_point(max_client_layers)

        # self._baseline_computational_model_per_split_point is a dictionary that, for every split
        # point (key), contains the computational requirements of training the model up to that
        # point.
        print(modeled_layer_complexities)
        modeled_layer_complexities = [modeled_layer_complexities[idx] for idx in range(len(modeled_layer_complexities))]
        print(modeled_layer_complexities)
        self._baseline_computational_model_per_split_point = {
            start_idx: sum(modeled_layer_complexities[:start_idx])
            for start_idx in self.meaningful_split_points
        }
        print("Meaningful split points: ", self.meaningful_split_points)
        print("Computational complexities", self._baseline_computational_model_per_split_point)

    def trigger_reallocation(self) -> None:
        """
        Dynamically adjust the number of layers assigned to each client during the epoch
        based on predicted finish time and server load.
        """
        start_time = time.time()
        remaining_server_time = 0.
        print("Triggering reallocation...")
        if len(self._client_train_batches_count) == 0 or max(self._pending_batches.values()) < 10:
            print("No need to reconfigure...")
            return

        print("=" * 20)
        print("CURRENT_CONFIGURATION:")
        print("Pending batches:", self._pending_batches)
        print("Current configuration", self._client_allocated_layers)
        print("Client training times", self._client_training_times)
        print("Client round times", self._client_round_times)
        print("Server training times", self._server_training_times)
        print("=" * 20)

        # Step 1.: Compute the remaining training time on the server if the situation does not change
        for cid in self._epoch_clients:
            remaining_server_time += \
                self._pending_batches[cid] * \
                    (self._server_training_times[self.get_client_offloaded_layer_count(cid)] / self._num_concat_batches)
        print("Server remaining time", remaining_server_time)

        # Step 2.: Change client confituration
        for cid in self._epoch_clients:
            # linear interpolation of client training times with respect to the number of layers
            client_pending_batches = self._pending_batches[cid]
            client_training_time = self._client_training_times[cid]
            baseline_layer_time = self._baseline_computational_model_per_split_point[self._client_allocated_layers[cid]]

            tmp = {
                split_point:
                    client_pending_batches * (model_time / baseline_layer_time) * client_training_time
                for split_point, model_time in self._baseline_computational_model_per_split_point.items()
            }
            print("Client predicted remaining times:", tmp)

            # now we have a rough (like life) estimate of how much time training is going to take
            # on the client with respect to every split point
            # we want to push as much computation to the client as possible while still preventing
            # the client from becoming a bottleneck. Thus, we pick the largest split point s,
            # such that the the remaining time is still smaller than that of the server
            # +5 to make sure the server is sure that this is a good choice
            maximum_split_point = max((k for k, v in tmp.items() if v + 5 < remaining_server_time), default=-1)
            print([k for k, v in tmp.items() if v + 5 < remaining_server_time])
            print("MSP", maximum_split_point)
            if maximum_split_point > 0 and maximum_split_point != self._client_allocated_layers[cid]:
                diff = maximum_split_point - self._client_allocated_layers[cid]
                self._client_layer_diff[cid] = diff
                print(f"Detected change if offloading strategy for {cid}: {diff}")

        print(f"REALLOCATION COMPLETE {time.time() - start_time:.3f}")
