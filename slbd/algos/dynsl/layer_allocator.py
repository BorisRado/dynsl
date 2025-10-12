import time
import threading
from logging import INFO
from typing import List, Dict, Optional, Tuple

from flwr.common import GetPropertiesIns, log
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager

from slbd.common.typing import TimingMetrics
from slbd.common.constants import (
    MIN_NUMBER_OF_CLIENT_LAYERS,
    CLIENT_DATASET_SIZE,
    CLIENT_DEVICE_TYPE,
    MAX_NUMBER_OF_CLIENT_LAYERS,
)


class DynSLClientLayerAllocator:

    def __init__(
        self,
        max_client_layers: int,
        num_model_layers: int,
        reallocation_frequency: float,
        num_concat_batches: int,
        meaningful_split_points: List[int],
        cache_client_configuration: bool
    ):
        """
        Initialize the LayerAllocator.

        Args:
            max_client_layers: Maximum number of layers a client can handle
            num_model_layers: Total number of model layers
            alpha: smoothing factor for layer adjustment
        """
        self._per_client_min_layers: Dict[str, int] = {}
        self._per_client_max_layers: Dict[str, int] = {}
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
        self._cached_num_layers = {}
        self.cache_client_configuration = cache_client_configuration
        self._cid_to_device_type_mapping = {}
        print("Caching?", self.cache_client_configuration)

    def get_client_min_layers(self, client_proxy: ClientProxy) -> int:
        if client_proxy.cid not in self._per_client_min_layers:
            # Call get_properties on client to fetch their requirements, specifically
            # the minimum number of layers they must train
            get_properties_instructions = GetPropertiesIns({})
            result = client_proxy.get_properties(
                get_properties_instructions, timeout=None, group_id=None
            )
            min_client_layers = int(result.properties[MIN_NUMBER_OF_CLIENT_LAYERS])

            # align min client layers to the model configuration. Find the first value in
            # model.meaningful_split_points that is larger or equal to min layers
            aligned_min_client_layers = \
                next((point for point in self.meaningful_split_points if point >= min_client_layers), None)
            print(f"Corrected min client layer {min_client_layers} -> {aligned_min_client_layers}")
            assert aligned_min_client_layers is not None

            max_client_layers = int(result.properties[MAX_NUMBER_OF_CLIENT_LAYERS])
            # align max client layers to the model configuration. Find the first value in
            # model.meaningful_split_points that is smaller or equal to the max layers
            aligned_max_client_layers = next((point for point in reversed(self.meaningful_split_points) if point <= max_client_layers), None)
            print(f"Corrected MAX client layer {max_client_layers} -> {aligned_max_client_layers}")

            self._per_client_min_layers[client_proxy.cid] = aligned_min_client_layers
            self._per_client_max_layers[client_proxy.cid] = aligned_max_client_layers
            self._dataset_sizes[client_proxy.cid] = int(result.properties[CLIENT_DATASET_SIZE])
            self._cid_to_device_type_mapping[client_proxy.cid] = result.properties[CLIENT_DEVICE_TYPE]
            node_name = result.properties["node_name"]
            log(INFO, f"Client {client_proxy.cid} requires at least {min_client_layers} layers [{result.properties[CLIENT_DEVICE_TYPE]} -- {node_name}]")
        return self._per_client_min_layers[client_proxy.cid]

    def get_initial_layer_count(self, client_proxy: ClientProxy) -> int:
        # always start with the minimum number of layers
        if self.cache_client_configuration and client_proxy.cid in self._cached_num_layers:
            return self._cached_num_layers[client_proxy.cid]
        return self.get_client_min_layers(client_proxy)

    def initialize_epoch(
        self,
        epoch_clients: List[ClientProxy],
        client_manager: ClientManager,
        is_training: bool,
    ):
        self._server_training_time_list = []
        _ = (client_manager, )
        self._epoch_clients = [client_proxy.cid for client_proxy in epoch_clients]
        self._is_training = is_training

        # _client_layer_diff stores the temporary difference in the number of layers of clients
        self._client_layer_diff = {cid: 0 for cid in self._epoch_clients}

        # reinitialize the client times, while for the server we keep historical data
        # dictionary containing the EMA of the training time of every client
        # NOTE: the entry should be restarted every time we chenge # of layers on client
        self._client_training_times = {}
        self._client_round_times = {}

        # we store an integer stating based on how many batches the metrics per client are computed
        self._client_train_batches_count = {proxy.cid: 0 for proxy in epoch_clients}

        # _client_allocated_layers - stores the number of layers every client has
        # at the beginning of every epoch, the number of layers is equal to the suggested value
        # be the allocator
        self._client_allocated_layers = {
            proxy.cid: self.get_initial_layer_count(proxy) for proxy in epoch_clients
        }

        # we have a dictionary saying how many batches the client still has to process
        self._pending_batches = {cid: self._dataset_sizes[cid] for cid in self._epoch_clients}

        self._reallocation_stop_event = threading.Event()

        # Start the periodic reallocation task if frequency is positive
        def periodic_reallocation():
            while not self._reallocation_stop_event.wait(self.reallocation_frequency):
                self.trigger_reallocation()

        self._reallocation_thread = threading.Thread(
            target=periodic_reallocation,
            daemon=True
        )
        self._reallocation_thread.start()
        initial_allocation = [self._client_allocated_layers[proxy.cid]  for proxy in epoch_clients]
        print("Initial allocation", initial_allocation)
        return initial_allocation

    def finalize_epoch(self):
        if self.cache_client_configuration:
            self._cached_num_layers = self._client_allocated_layers.copy()
        self._epoch_clients = None
        self._client_allocated_layers = None
        self._reallocation_stop_event.set()
        self._reallocation_thread.join(timeout=5.0)

    def get_next_split_point(self, current_idx):
        for split_point in self.meaningful_split_points:
            if split_point > current_idx:
                return split_point
        return current_idx

    def get_previous_split_point(self, current_idx, client_id=None):
        previous_split_point = self.meaningful_split_points[0]
        for split_point in reversed(self.meaningful_split_points):
            if split_point < current_idx:
                previous_split_point = split_point
                break
        return previous_split_point if client_id is None \
            else max(previous_split_point, self._per_client_min_layers[client_id])

    def trigger_reallocation(self) -> None:
        """
        Dynamically adjust the number of layers assigned to each client during the epoch
        based on predicted finish time and server load.
        """
        start_time = time.time()
        remaining_server_time = 0.
        client_remaining_times = {}
        print("Triggering reallocation...")
        if len(self._client_train_batches_count) == 0 :
            print("Not enough data...")
            return
        print("Pending batches:", self._pending_batches)
        print("Current configuration", self._client_allocated_layers)
        print("Client training times", self._client_training_times)
        print("Client round times", self._client_round_times)
        print("Server training times", self._server_training_times)
        print("Number of processed batches since last reallocation", self._client_train_batches_count)

        for client in self._epoch_clients:
            # remaining time on the server is equal to the sum of the number of batches every
            # client still has to process times the time it takes to process a batch.
            # If we are concatenating batches, we need to divide the product by the
            # number of concatenating batches
            if self._pending_batches[client] < 10:
                print(f"ignoring {client} as it has too little batches left")
                continue
            if self.get_client_offloaded_layer_count(client) not in self._server_training_times:
                # we don't know the computation time on the server -- thus, we cannot
                # know how to optimally redistribute the computation
                return
            remaining_server_time += \
                self._pending_batches[client] * \
                    (self._server_training_times[self.get_client_offloaded_layer_count(client)] / self._num_concat_batches)

            # the client remaining time is equal to the number of batches the client still
            # has to process times the round time
            # TODO: we will have to differentiate between round time and training time at some point
            if self._client_train_batches_count[client] < 5:
                # print(f"Ignore client {client} as it does not have enough data...")
                continue
            client_remaining_times[client] = self._pending_batches[client] * self._client_round_times[client]

        print("Client remaining times", client_remaining_times)
        print("Server remaining time", remaining_server_time)

        # Step 3: compute layer adjustments
        if len(client_remaining_times) == 0 or max(self._pending_batches.values()) < 10:
            return

        # max_remaining_client_time = max(client_remaining_times.values())
        max_client_local_training_time = max(self._client_training_times.values())
        self._gamma = 10
        for cid, rct in client_remaining_times.items():
            change_offloading = False
            if remaining_server_time - rct > self._gamma:
                # if the client is much faster than the server, send it more layers

                next_configuration = self.get_next_split_point(self._client_allocated_layers[cid])
                if self._cid_to_device_type_mapping[cid] == "JetsonOrinNano" and next_configuration > 11:
                    print("Avoiding to send too many layer to JON to avoid memory pressure")
                    continue
                if self._client_training_times[cid] < max_client_local_training_time / 10:
                    # print("Sending x2 skip")
                    next_configuration = self.get_next_split_point(next_configuration)

                self._client_layer_diff[cid] = next_configuration - self._client_allocated_layers[cid]
                change_offloading = True
            elif rct > remaining_server_time + (self._gamma * 2):
                # cliet is too slow
                previous_configuration = self.get_previous_split_point(self._client_allocated_layers[cid], cid)
                self._client_layer_diff[cid] = previous_configuration - self._client_allocated_layers[cid]
                if self._client_layer_diff[cid] != 0:
                    change_offloading = True
                    print("Reducing number of layers on client")
                # else:
                #     print(f"WARNING === CLIENT {cid} IS A BOTTLENECK NOW == can't do anything aobut it!")
                #     pass

            if change_offloading:
                self._client_train_batches_count[cid] = 0
                try:
                    del self._client_round_times[cid] # restart counting...
                    del self._client_training_times[cid] # restart counting...
                except:
                    # probably client has finished already
                    pass

        print(f"REALLOCATION COMPLETE {time.time() - start_time:.3f}")
        # print("=" * 20)
        # print("=" * 20)
        # print("=" * 20)

    def get_client_offloaded_layer_count(self, client_id):
        # returns the number of layers that the server needs to perform
        # i.e., """model[-this_number:](input)"""
        return self._num_model_layers - self._client_allocated_layers[client_id]

    def get_post_reallocation_layer_difference(self, client_id: str) -> Optional[Tuple[int, int]]:
        if self._client_layer_diff[client_id] != 0:
            previous = self._client_allocated_layers[client_id]
            diff = self._client_layer_diff[client_id]
            self._client_allocated_layers[client_id] += diff
            self._client_layer_diff[client_id] = 0
            return previous, diff

    def update_client_metrics(self, client_id: str, metrics: Optional[TimingMetrics]):
        if metrics is None:
            return

        client_train_time = metrics.client_local_computation_time
        client_round_time = metrics.client_round_time
        if self._client_train_batches_count[client_id] == 0:
            self._client_training_times[client_id] = client_train_time
            self._client_round_times[client_id] = client_round_time
        else:
            # rolling average update
            self._client_training_times[client_id] = \
                self._rolling_average(self._client_training_times[client_id], client_train_time)
            self._client_round_times[client_id] = \
                self._rolling_average(self._client_round_times[client_id], client_round_time)
        self._client_train_batches_count[client_id] += 1
        self._pending_batches[client_id] -= 1

    def _rolling_average(self, average, new_value):
        if average is None:
            return new_value
        return 0.8 * average + 0.2 * new_value

    def update_server_metrics(
        self,
        num_offloaded_layers: int,
        elapsed_time: float,
        wait_time_in_ready_state: float,
        batch_preparation_time: float
    ):
        self._server_training_time_list.append(elapsed_time)
        if num_offloaded_layers in self._server_training_times:
            self._server_training_times[num_offloaded_layers] = \
                self._rolling_average(self._server_training_times[num_offloaded_layers], elapsed_time)
        else:
            self._server_training_times[num_offloaded_layers] = elapsed_time

    def collect_epoch_metrics(self):
        return {
            "training_times": self._server_training_time_list,
        }

    def remove_client(self, cid):
        print(f"Removing {cid} as it has finished")
        self._epoch_clients.remove(cid)
        try:
            del self._client_layer_diff[cid]
            del self._client_round_times[cid]
            del self._client_training_times[cid]
            del self._client_train_batches_count[cid]
            del self._pending_batches[cid]
        except:
            pass


class DynSLClientLayerAllocatorTest:

    def __init__(
        self,
        max_client_layers: int,
        num_model_layers: int,
        reallocation_frequency: float,
        num_concat_batches: int,
        meaningful_split_points: List[int]
    ):
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
            "second_trigger_batch": 10,
            "layers_to_add": 8,
            "layers_to_remove": 3
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
            elif batches_processed == self._reallocation_config["second_trigger_batch"]:
                self._client_layer_diff[client_id] = -self._reallocation_config["layers_to_remove"]
                print("HEREEEEE", self._client_layer_diff[client_id])

    def get_client_offloaded_layer_count(self, client_id):
        return self._num_model_layers - self._client_allocated_layers[client_id]

    def get_post_reallocation_layer_difference(self, client_id: str) -> Optional[Tuple[int, int]]:
        self.trigger_reallocation(self._epoch_clients)
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
        # print(metrics)
        self._per_client_processed_batches[client_id] += 1

    def update_server_metrics(
        self,
        num_offloaded_layers: int,
        elapsed_time: float,
        wait_time_in_ready_state: float,
        batch_preparation_time: float
    ):
        # print(num_offloaded_layers, elapsed_time, wait_time_in_ready_state)
        # print(num_offloaded_layers, elapsed_time)
        pass
