import time
from typing import List, Dict

from slbd.algos.dynsl.layer_allocator import DynSLClientLayerAllocator


class DynSLClientLayerAllocatorV2(DynSLClientLayerAllocator):

    def __init__(
        self,
        max_client_layers: int,
        num_model_layers: int,
        reallocation_frequency: float,
        gamma: float,
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
        self._gamma = gamma
        self._per_client_min_layers: Dict[str, int] = {}
        self._per_client_max_layers: Dict[str, int] = {}
        self._num_model_layers: int = num_model_layers

        # per-client dataset and pending batches
        self._dataset_sizes: Dict[str, int] = {}
        self._pending_batches: Dict[str, int] = {}

        self._allocated_layer_history = []

        # rolling averages
        self._client_training_times: Dict[str, float] = {}
        self._client_round_times: Dict[str, float] = {}

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
        self._server_training_times = {
            start_idx: sum(modeled_layer_complexities[start_idx:])
            for start_idx in self.meaningful_split_points
        }
        print("Meaningful split points: ", self.meaningful_split_points)
        print("Computational complexities", self._baseline_computational_model_per_split_point)
        print("Static server training times:", self._server_training_times)

    def trigger_reallocation(self) -> None:
        """
        Dynamically adjust the number of layers assigned to each client during the epoch
        based on predicted finish time and server load.
        """
        start_time = time.time()
        remaining_server_time = 0.
        print("Triggering reallocation...", len(self._client_training_times), "clients missing")
        if len(self._client_train_batches_count) == 0 or max(self._pending_batches.values()) < 10:
            print("No need to reconfigure...")
            return
        self._allocated_layer_history.append(self._client_allocated_layers.copy())

        print("=" * 30)
        print("=" * 30)
        print("CURRENT_CONFIGURATION:")
        print("Pending batches:", self._pending_batches)
        # client allocated layers states how many layers are currently on the client
        print("Current configuration", self._client_allocated_layers)
        print("Client training times", self._client_training_times)
        # print("Client round times", self._client_round_times)
        print("Server training times", self._server_training_times)
        print("#" * 20)

        # Step 1.: Compute the remaining training time on the server if the situation does not change
        client_remaining_time = {}
        for cid in self._epoch_clients:
            ncl = self._client_allocated_layers[cid]
            pending_batches = self._pending_batches[cid]

            remaining_server_time += \
                pending_batches * (self._server_training_times[ncl] / self._num_concat_batches)

            if cid not in self._client_training_times or self._client_train_batches_count[cid] < 5:
                continue
            client_remaining_time[cid] = pending_batches * self._client_training_times[cid]

        print("Server remaining time", remaining_server_time)
        print("Remaining client times:", client_remaining_time)

        # Step 2.: Change client confituration
        for cid, remaining_time in sorted(client_remaining_time.items(), key=lambda item: item[1]):
            print(f"// Iterating {cid} [server remaining time: {remaining_server_time}]")
            if remaining_time < remaining_server_time:
                new_split_point = self._try_increase_num_layers(cid, server_remaining_time=remaining_server_time)
            else:
                new_split_point = self._try_decrease_num_layers(cid, server_remaining_time=remaining_server_time)

            ncl = self._client_allocated_layers[cid]
            if new_split_point != ncl:
                diff = new_split_point - ncl
                self._client_layer_diff[cid] = diff
                server_time_diff = self._get_server_time_diff(self._pending_batches[cid], ncl, new_split_point)
                print(f"Detected change if offloading strategy for {cid}: {diff} [server change: {server_time_diff}]")
                remaining_server_time += server_time_diff

        print(f"REALLOCATION COMPLETE {time.time() - start_time:.3f}")

    def _try_increase_num_layers(self, cid, server_remaining_time):
        ncl = self._client_allocated_layers[cid]
        client_tt = self._client_training_times[cid]
        baseline_model_time = self._baseline_computational_model_per_split_point[ncl]
        pending_batches = self._pending_batches[cid]

        new_split_point = ncl
        for candidate in self.meaningful_split_points:
            if candidate <= ncl or candidate > self._per_client_max_layers[cid]:
                continue
            # now layer is >= than current number of layers
            model_pred_compl = self._baseline_computational_model_per_split_point[candidate]
            predicted_end_time = pending_batches * (model_pred_compl / baseline_model_time) * client_tt
            server_diff = self._get_server_time_diff(pending_batches, ncl, candidate)
            print(f"++ {cid} [{candidate}]: {predicted_end_time} {server_remaining_time} {server_diff}")
            assert server_diff < 0, str(server_diff)
            if server_diff > -2.0:
                print("Benefit too small -- Ignore!!")
                continue

            if predicted_end_time + self._gamma < server_remaining_time + server_diff:
                new_split_point = candidate
        return new_split_point

    def _try_decrease_num_layers(self, cid, server_remaining_time):
        ncl = self._client_allocated_layers[cid]
        client_tt = self._client_training_times[cid]
        baseline_model_time = self._baseline_computational_model_per_split_point[ncl]
        pending_batches = self._pending_batches[cid]

        new_split_point = max(self.meaningful_split_points[0], self._per_client_min_layers[cid])
        for candidate in self.meaningful_split_points:
            if candidate >= ncl or candidate < self._per_client_min_layers[cid]:
                continue

            # now layer is <= than current number of layers
            model_time = self._baseline_computational_model_per_split_point[candidate]
            predicted_end_time = pending_batches * (model_time / baseline_model_time) * client_tt
            server_diff = self._get_server_time_diff(pending_batches, ncl, candidate)
            assert server_diff > 0, str(server_diff) # time on server increases
            print(f"-- {cid} [{candidate}]: {predicted_end_time} {server_remaining_time} {server_diff}")
            if predicted_end_time < server_remaining_time + server_diff + self._gamma:
                new_split_point = candidate
        return new_split_point

    def _get_server_time_diff(self, num_batches, current_layers, candidate_layers):
        server_time_diff = \
            self._server_training_times[candidate_layers] - self._server_training_times[current_layers]
        server_time_diff *= (num_batches / self._num_concat_batches)
        return server_time_diff

    def update_server_metrics(self, *args, **kwargs):
        return
