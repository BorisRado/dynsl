import time
import threading
import itertools
from datetime import datetime
from logging import INFO
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

from flwr.common import GetPropertiesIns, log
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager

from slbd.server.client_layer_allocator import ClientLayerAllocator
from slbd.common.typing import TimingMetrics
from slbd.common.constants import (
    MIN_NUMBER_OF_CLIENT_LAYERS,
    CLIENT_DATASET_SIZE,
    CLIENT_DEVICE_TYPE,
    MAX_NUMBER_OF_CLIENT_LAYERS,
)

from slbd.algos.dynsl.optimization_algorithms import genetic_reallocation, bottleneck_descent, binary_search


class DynSLClientLayerAllocatorV2(ClientLayerAllocator):

    def __init__(
        self,
        max_client_layers: int,
        num_model_layers: int,
        reallocation_frequency: float,
        gamma: float,
        num_concat_batches: int,
        layer_allocation_optimizer: str,
        meaningful_split_points: List[int],
        cache_client_data: bool,
        modeled_layer_complexities: List[float],
        static_server_computational_times: List[float],
        per_layer_emb_sizes: List[int],
        per_layer_parameters: Dict[int, int]
    ):
        """Initialize the allocator

        Parameters
        ----------
        max_client_layers : int
            Max number of layers that any client can train, as per MO desire
        num_model_layers : int
            Number of layers that the model consists of
        reallocation_frequency : float
            How often (seconds) the reallocation strategy should run
        gamma : float
            Minimum benefit to update the split vector
        num_concat_batches : int
            How many batches are being concatenated before starting the forward pass
        layer_allocation_optimizer : str
            Strategy to be used for computing the new split vector
        meaningful_split_points : List[int]
            List of indices where it makes sense to split the model
        cache_client_data : bool
            Whether to use info from previous epochs to determine the new configuration
        modeled_layer_complexities : List[float]
            A "model" stating how complex every layer is (e.g., determined via FLOPS or by running
            the model on the server)
        per_layer_emb_sizes : List[int]
            Size (# parameters) per every split point. Here we are talking about embeddings/gradients
        per_layer_parameters : Dict[int]
            Size of model layers. For each key (layer idx) we get the number of FP values
        """
        # safety checks
        assert layer_allocation_optimizer in {"genetic", "bottleneck_descent", "binary_search"}
        assert reallocation_frequency > 0.

        # store basic attributes
        self._gamma = gamma
        self._num_model_layers: int = num_model_layers
        self.reallocation_frequency = reallocation_frequency
        self._num_concat_batches = num_concat_batches
        self.cache_client_data = cache_client_data
        self.layer_allocation_optimizer = layer_allocation_optimizer
        self.meaningful_split_points = [msp for msp in meaningful_split_points if msp <= max_client_layers]
        self._max_client_layers = next((point for point in reversed(self.meaningful_split_points) if point <= max_client_layers))
        self._is_computing_configuration_from_cache = False

        # client desires -- client sets the maximum and minimum number of layers it is willing to train
        self._per_client_min_layers: Dict[str, int] = {}
        self._per_client_max_layers: Dict[str, int] = {}

        # per-client dataset and pending batches
        self._dataset_sizes: Dict[str, int] = {}
        self._pending_batches: Dict[str, int] = {}

        # debugging / tracking
        self._allocated_layer_history = []
        self._cid_to_device_type_mapping = {}
        self.tot_server_computation_time = None
        self._cid_to_node_name = {}

        # rolling averages -- note, they get re-initialized every time the client changes its number of layers
        # they are persisted across epochs if the allocator is configured with caching
        self._client_training_times: Dict[str, float] = {}
        self._client_throughput: Dict[str, float] = {}

        # event for starting stopping the thread that handles layer re-allocation
        self._reallocation_stop_event = None

        # self._baseline_computational_model_per_split_point is a dictionary that, for every split
        # point (key), contains the computational requirements of training the model up to that point
        modeled_layer_complexities = [modeled_layer_complexities[idx] for idx in range(len(modeled_layer_complexities))]
        # print(modeled_layer_complexities)
        comput_compl_per_sp = {
            start_idx: sum(modeled_layer_complexities[:start_idx])
            for start_idx in self.meaningful_split_points
        }
        self.complexity_factors = {
            (a, b): comput_compl_per_sp[a] / comput_compl_per_sp[b]
            for a, b in itertools.product(meaningful_split_points, meaningful_split_points)
        }

        # we assume that the training time on the server do not vary
        static_server_computational_times = \
            [static_server_computational_times[idx] for idx in range(len(static_server_computational_times))]
        self._server_training_times = {
            start_idx: sum(static_server_computational_times[start_idx:])
            for start_idx in self.meaningful_split_points
        }

        # compute the time the server will spend communicating every batch of data given the current split point
        # ==>> in the formula:
        # * 2 ... forward+backward // the server will have to send and recive the data over a single link
        # * 32 ... 32 bit representation of FP values
        self.per_batch_server_communication_time = {
            k: 2 * 32 * per_layer_emb_sizes[k] / 1e9 # server has a 1Gbps connection link, which is shared across all clienst
            for k in self.meaningful_split_points
        }
        # print(per_layer_communication_volume)
        # * 32 as we use 32 bit FP representation,  / 1e6 to convert bits in megabits
        self._per_batch_embedding_sizes_Mbits = {
            k: per_layer_emb_sizes[k] * 32 / 1e6
            for k in self.meaningful_split_points
        }
        # print("per batch", self._per_batch_embedding_sizes_Mbits)
        tmp_split_points = [0,] + list(meaningful_split_points)
        per_layer_parameters = [per_layer_parameters[idx] for idx in range(len(per_layer_parameters))]
        self._per_layer_model_parameters_Mbits = {
            (a, b): sum(per_layer_parameters[a:b]) * 32 / 1e6
            for a, b in itertools.product(tmp_split_points, tmp_split_points)
            if a < b
        }
        print("Meaningful split points: ", self.meaningful_split_points)
        # print("Static server training times:", self._server_training_times)
        # print(f"Complexity factors: {self.complexity_factors}")
        print("Communication:", self.per_batch_server_communication_time)

    def _get_client_properties(self, client_proxy: ClientProxy) -> int:
        if client_proxy.cid in self._per_client_min_layers:
            log(INFO, f"Already have data for client {client_proxy.cid}")
            assert client_proxy.cid in self._dataset_sizes
            assert client_proxy.cid in self._cid_to_device_type_mapping
            return

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
        assert aligned_min_client_layers is not None

        max_client_layers = int(result.properties[MAX_NUMBER_OF_CLIENT_LAYERS])
        # align max client layers to the model configuration. Find the first value in
        # model.meaningful_split_points that is smaller or equal to the max layers
        aligned_max_client_layers = next((point for point in reversed(self.meaningful_split_points) if point <= max_client_layers), None)

        self._per_client_min_layers[client_proxy.cid] = aligned_min_client_layers
        self._per_client_max_layers[client_proxy.cid] = aligned_max_client_layers
        self._dataset_sizes[client_proxy.cid] = int(result.properties[CLIENT_DATASET_SIZE])
        self._cid_to_device_type_mapping[client_proxy.cid] = result.properties[CLIENT_DEVICE_TYPE]
        node_name = result.properties["node_name"]
        self._cid_to_node_name[client_proxy.cid] = node_name
        log(INFO, f"Client {client_proxy.cid} requires at least {min_client_layers} layers [{result.properties[CLIENT_DEVICE_TYPE]} -- {node_name}]")

    def _get_initial_number_of_layers_with_caching(self):
        if len(self._client_training_times) == 0:
            print("Warning: No client data is available. If it is the first epoch, it's fine. If not, not good...")
            print("Warning: Returning minimum number of layers")
            return self._per_client_min_layers
        self._pending_batches = {cid: self._dataset_sizes[cid] for cid in self._epoch_clients} # batches the client still has to process
        self._current_number_of_layers_on_devices = {cid: 0 for cid in self._epoch_clients}
        self._is_computing_configuration_from_cache = True
        if self.layer_allocation_optimizer == "genetic":
            initial_configuration = genetic_reallocation(
                clients=self._epoch_clients,
                meaningful_split_points=self.meaningful_split_points,
                allocated_layers_now=self._client_allocated_layers,
                objective_fn=self._compute_objective
            )
        else:
            assert self.layer_allocation_optimizer in {"bottleneck_descent", "binary_search"}
            initial_configuration = bottleneck_descent(
                clients=self._epoch_clients,
                meaningful_split_points=self.meaningful_split_points,
                allocated_layers_now=self._client_allocated_layers,
                objective_fn=self._compute_objective
            )
        # clear data from the previous epoch
        self._is_computing_configuration_from_cache = False
        self._client_allocated_layers.clear()
        self._client_training_times.clear()
        self._client_throughput.clear()
        print("Using caching -- Got configuration:", initial_configuration)
        return initial_configuration

    def _init_reallocation_thread(self):
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

    def _init_all_client_properties(self, clients):
        # get the properties of every client that is involved in the current epoch
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self._get_client_properties, client): client for client in clients}
        for future in futures:
            try:
                future.result()
            except Exception as e:
                client = futures[future]
                print(f"Error while fetching properties for client {client.cid}: {e}")

    def initialize_epoch(
        self,
        epoch_clients: List[ClientProxy],
        client_manager: ClientManager,
        is_training: bool,
    ):
        self._is_training = is_training
        self._epoch_clients = [client_proxy.cid for client_proxy in epoch_clients]

        self._init_all_client_properties(epoch_clients)
        if self.cache_client_data:
            initial_config_dict = self._get_initial_number_of_layers_with_caching()
        else:
            # if not caching, we set the clients to initially use the minimum number of layers
            initial_config_dict = self._per_client_min_layers

        # init dictionaries storing client configuration
        self._client_layer_diff = {cid: 0 for cid in self._epoch_clients} # temp difference in the number of layers of clients
        self._client_train_batches_count = {cid: 0 for cid in self._epoch_clients} # number of batches the client has processed since it last received a reallocation
        self._client_allocated_layers = {cid: initial_config_dict[cid] for cid in self._epoch_clients} # layers every client has
        self._pending_batches = {cid: self._dataset_sizes[cid] for cid in self._epoch_clients} # batches the client still has to process

        # bookkeeping / debugging
        self._allocated_layer_history = []
        self.tot_server_computation_time = 0.

        self._init_reallocation_thread()
        return [self._client_allocated_layers[proxy.cid]  for proxy in epoch_clients]

    def trigger_reallocation(self) -> None:
        start_time = time.time()
        # Uncomment following lines to log the tracked metrics
        # print(self._client_train_batches_count)
        # print({self._cid_to_node_name[k]: v for k, v in self._client_training_times.items() if k in self._client_layer_diff})
        print(f"Running reallocation {datetime.now().strftime('%H:%M:%S')}")
        print({self._cid_to_node_name[k]: v for k, v in self._client_allocated_layers.items() if k in self._client_layer_diff})
        # print({self._cid_to_node_name[k]: v for k, v in self._pending_batches.items() if k in self._client_layer_diff})
        # print({self._cid_to_node_name[k]: v for k, v in self._client_throughput.items() if k in self._client_layer_diff})
        self._allocated_layer_history.append(self._client_allocated_layers.copy())
        if any(v < 4 for v in self._client_train_batches_count.values()):
            print("Not enough data to try reallocating...")
            return
        if len(self._client_layer_diff) == 0:
            print("All have completed training. Exiting...")
            return

        if self.layer_allocation_optimizer == "genetic":
            print("Using genetic algorithm")
            best_config = genetic_reallocation(
                clients=self._epoch_clients,
                meaningful_split_points=self.meaningful_split_points,
                allocated_layers_now=self._client_allocated_layers,
                objective_fn=self._compute_objective
            )
        elif self.layer_allocation_optimizer == "bottleneck_descent":
            print("Using bottleneck descent")
            best_config = bottleneck_descent(
                clients=self._epoch_clients,
                meaningful_split_points=self.meaningful_split_points,
                allocated_layers_now=self._client_allocated_layers,
                objective_fn=self._compute_objective
            )
        elif self.layer_allocation_optimizer == "binary_search":
            print("Using binary_search")
            best_config = binary_search(
                clients=self._epoch_clients,
                meaningful_split_points=self.meaningful_split_points,
                allocated_layers_now=self._client_allocated_layers,
                client_fn=self.compute_client_time,
                server_fn=self.compute_server_time,
                gamma=self._gamma,
            )

        print("current", self._client_allocated_layers)
        print("predicted", best_config)
        J_current = self._compute_objective(self._client_allocated_layers)
        J_new = self._compute_objective(best_config)
        delta = J_current - J_new
        print(f"Current cost: {J_current:.3f}")
        print(f"Found cost: {J_new:.3f}")

        if delta > self._gamma:
            print(f"[GA] Adopt new configuration (ΔJ={delta:.3f} > γ={self._gamma})")
            for cid, new_layers in best_config.items():
                self._client_layer_diff[cid] = new_layers - self._client_allocated_layers[cid]
                if self._client_layer_diff[cid] != 0:
                    self._client_train_batches_count[cid] = 0
                    del self._client_training_times[cid]
        else:
            print(f"[GA] Keep current configuration (ΔJ={delta:.3f} ≤ γ={self._gamma})")
        print(f"Total reallocation time: {time.time() - start_time:4f}")

    def _compute_objective(self, layer_config):
        # computes the predicted epoch time for the current configuration given in layer_config
        # layer_config is a dictionary that for every client states how many layers it trains in the
        # evaluated configuration

        pred_client_times = {} # compute the predicted time to finish training for every client
        server_time = 0. # Compute the time the server needs to train with the current configuration
        server_communication_time = 0. # time the server is communicating

        for cid, L in layer_config.items():
            B = self._pending_batches[cid] # number of batches the client still has to process
            server_time += B * self._server_training_times[L] / self._num_concat_batches # time the server will be processing client batches
            pred_client_times[cid] = self.compute_client_time(cid, L)
            server_communication_time += B * self.per_batch_server_communication_time[L]

        # slightly increase the server_communication_time to account for the overhead of serializing and deserializing
        server_time = max(server_time, server_communication_time * 1.2)

        # add a regularization term that penalizes the change of the current configuration
        reg_term = sum([
            layer_config[cid] != self._client_allocated_layers[cid] for cid in layer_config
        ])
        # get the time of the slowest client, as in SL, we wait for the slowest client to finish training
        # before we end the current epoch
        max_client_time = max(pred_client_times.values())

        # the final objective is the maximum of the server time and the slowest client time + the regularization
        total_time = max(server_time, max_client_time) + reg_term
        return total_time

    def compute_server_time(self, candidate_split_vector):
        server_time = 0. # Compute the time the server needs to train with the current configuration
        server_communication_time = 0. # time the server is communicating

        for cid, L in candidate_split_vector.items():
            B = self._pending_batches[cid] # number of batches the client still has to process
            server_time += B * self._server_training_times[L] / self._num_concat_batches # time the server will be processing client batches
            server_communication_time += B * self.per_batch_server_communication_time[L]

        # slightly increase the server_communication_time to account for the overhead of serializing and deserializing
        server_time = max(server_time, server_communication_time * 1.2)

        return server_time

    def compute_client_time(self, cid, L):
        # computes the remaining epoch time for the client if the client were to train L layers

        # we have the training time of the client for the configuration `self._client_allocated_layers`
        # that is, when the client has `self._client_allocated_layers[cid]` layers, its local training time
        # is `self._client_training_times[cid]`. To compute the predicted training time for a different number
        # of layers, we perform a simple linear interpolation.

        factor = self.complexity_factors[(L, self._client_allocated_layers[cid])]
        if self.meaningful_split_points.index(L) - self.meaningful_split_points.index(self._client_allocated_layers[cid]) > 2:
            # if considing increasing the number of layers a lot, penalize the predicted increase, so
            # as to make a more conservative choice
            factor *= 1.5
        # client computation
        per_batch_time = self._client_training_times[cid] * factor # client training
        # server computation
        per_batch_time += self._server_training_times[L] # server training
        # communication -- * 2 because we need to both send and receive the data
        per_batch_time += 2 * self._per_batch_embedding_sizes_Mbits[L] / self._client_throughput[cid]

        current_num_layers_on_device = \
            0 if self._is_computing_configuration_from_cache else self._client_allocated_layers[cid]
        layer_movement_time = 0.
        # if current_num_layers_on_device == L -- there is no additional layer movement ==>> layer_movement_time=0
        # if current_num_layers_on_device > L -- layers are moved from device to server. This has to happend either
        #           at the end of the epoch, so we say that the layer_movement_time is still 0.
        if current_num_layers_on_device < L:
            num_moved_layer_Mbits = self._per_layer_model_parameters_Mbits[(current_num_layers_on_device, L)]
            layer_movement_time = 2 * num_moved_layer_Mbits / self._client_throughput[cid]
        return self._pending_batches[cid] * per_batch_time + layer_movement_time

    def update_server_metrics(
        self,
        num_offloaded_layers: int,
        elapsed_time: float,
    ):
        self.tot_server_computation_time += elapsed_time

    def get_post_reallocation_layer_difference(self, client_id: str) -> Optional[Tuple[int, int]]:
        previous = self._client_allocated_layers[client_id]
        if self._client_layer_diff[client_id] != 0:
            diff = self._client_layer_diff[client_id]
            self._client_allocated_layers[client_id] += diff
            self._client_layer_diff[client_id] = 0
            return previous, diff
        return None

    def remove_client(self, cid):
        print(f"Removing {cid} as it has finished", self._pending_batches[cid])
        self._epoch_clients.remove(cid)
        del self._client_layer_diff[cid]
        # we remove the cid from self._client_layer_diff
        # ==>> if a cid is not in self._client_layer_diff, it means the client has finished training

    def update_client_metrics(self, client_id: str, metrics: Optional[TimingMetrics]):
        if metrics is None or metrics.client_local_computation_time < 0:
            self._pending_batches[client_id] -= 1
            return

        client_train_time = metrics.client_local_computation_time
        client_Mbps = metrics.throughput_Mbps
        # we already handled the case where we could not handle the throughput
        assert client_train_time > 0. and client_Mbps > 0
        self._client_training_times.setdefault(client_id, client_train_time)
        self._client_throughput.setdefault(client_id, client_Mbps)

        self._client_training_times[client_id] = \
            self._rolling_average(self._client_training_times[client_id], client_train_time)
        self._client_throughput[client_id] = \
            self._rolling_average(self._client_throughput[client_id], client_Mbps)

        self._client_train_batches_count[client_id] += 1
        self._pending_batches[client_id] -= 1

    def finalize_epoch(self):
        self._epoch_clients = []
        self._reallocation_stop_event.set()
        self._reallocation_thread.join(timeout=5.0)
        if not self.cache_client_data:
            self._client_allocated_layers.clear()
            self._client_training_times.clear()
            self._client_throughput.clear()

    def collect_epoch_metrics(self):
        return {
            "tot_server_training_time": self.tot_server_computation_time,
            "layer_history": self._allocated_layer_history,
            "client_throughputs": self._client_throughput,
        }

    def get_client_offloaded_layer_count(self, client_id):
        # returns the number of layers that the server needs to perform
        # i.e., """model[-this_number:](input)"""
        return self._num_model_layers - self._client_allocated_layers[client_id]

    def _rolling_average(self, average, new_value):
        return 0.8 * average + 0.2 * new_value

    def get_max_client_layers(self, cid):
        print("Getting max number of layers for", cid, self._per_client_max_layers[cid], self._max_client_layers)
        return min(self._per_client_max_layers[cid], self._max_client_layers)

    def get_min_client_layers(self, cid):
        return self._per_client_min_layers[cid]
