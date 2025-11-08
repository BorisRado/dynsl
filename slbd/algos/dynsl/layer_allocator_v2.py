import time
import random
import threading
import itertools
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
        cache_client_configuration: bool,
        modeled_layer_complexities: List[float],
        per_layer_communication_volume: List[int],
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
        cache_client_configuration : bool
            Whether to use info from previous epochs to determine the new configuration
        modeled_layer_complexities : List[float]
            A "model" stating how complex every layer is (e.g., determined via FLOPS or by running
            the model on the server)
        per_layer_communication_volume : List[int]
            Size (# parameters) per every split point
        """
        self._gamma = gamma
        self._num_model_layers: int = num_model_layers
        self.layer_allocation_optimizer = layer_allocation_optimizer
        self.reallocation_frequency = reallocation_frequency
        self._num_concat_batches = num_concat_batches
        self.cache_client_configuration = cache_client_configuration
        self.meaningful_split_points = [
            msp for msp in meaningful_split_points if msp <= max_client_layers
        ]
        self._max_client_layers: int = self.get_previous_split_point(max_client_layers)

        # safety checks
        assert self.layer_allocation_optimizer in {"genetic"}
        assert self.reallocation_frequency > 0.

        # client desides
        self._per_client_min_layers: Dict[str, int] = {}
        self._per_client_max_layers: Dict[str, int] = {}

        # per-client dataset and pending batches
        self._dataset_sizes: Dict[str, int] = {}
        self._pending_batches: Dict[str, int] = {}

        # debugging / tracking
        self._allocated_layer_history = []
        self._cid_to_device_type_mapping = {}
        self.tot_server_computation_time = None

        # rolling averages
        self._client_training_times: Dict[str, float] = {}
        self._client_throughput: Dict[str, float] = {}

        # we will call the reallocation function with the following frequency
        self._reallocation_stop_event = None

        self._cached_num_layers = {}
        print("Caching?", self.cache_client_configuration)

        # self._baseline_computational_model_per_split_point is a dictionary that, for every split
        # point (key), contains the computational requirements of training the model up to that
        # point.
        print(modeled_layer_complexities)
        modeled_layer_complexities = [modeled_layer_complexities[idx] for idx in range(len(modeled_layer_complexities))]
        print(modeled_layer_complexities)
        comput_compl_per_sp = {
            start_idx: sum(modeled_layer_complexities[:start_idx])
            for start_idx in self.meaningful_split_points
        }
        self.complexity_factors = {
            (a, b): comput_compl_per_sp[a] / comput_compl_per_sp[b]
            for a, b in itertools.product(meaningful_split_points, meaningful_split_points)
        }
        print(f"Complexity factors: {self.complexity_factors}")

        self._server_training_times = {
            start_idx: sum(modeled_layer_complexities[start_idx:])
            for start_idx in self.meaningful_split_points
        }
        print("Meaningful split points: ", self.meaningful_split_points)
        print("Static server training times:", self._server_training_times)
        self._cid_to_node_name = {}

        # per_layer_communication_volume contains number of parameters per layer
        # =====>>>>>> in the formula:
        # * 2 ... forward+backward
        # * 4 ... 32 bit representation
        # * 8 ... we transform bytes to bits
        throughput_Gbps = 1e9
        self.per_batch_server_communication_time = {
            k: 2 * 4 * per_layer_communication_volume[k] * 8 / throughput_Gbps
            for k in self.meaningful_split_points
        }
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
        self._cid_to_node_name[client_proxy.cid] = node_name
        log(INFO, f"Client {client_proxy.cid} requires at least {min_client_layers} layers [{result.properties[CLIENT_DEVICE_TYPE]} -- {node_name}]")

    def _get_initial_number_of_layers_with_caching(self, epoch_clients):
        raise Exception("Not implemented yet")

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
        if self.cache_client_configuration:
            initial_config_dict = self._get_initial_number_of_layers_with_caching(epoch_clients)
        else:
            initial_config_dict = self._per_client_min_layers

        # init dictionaries storing client configuration
        self._client_layer_diff = {cid: 0 for cid in self._epoch_clients} # temporary difference in the number of layers of clients
        self._client_train_batches_count = {cid: 0 for cid in self._epoch_clients} # batches the metrics per client are computed
        self._client_allocated_layers = {cid: initial_config_dict[cid] for cid in self._epoch_clients} # layers every client has
        self._pending_batches = {cid: self._dataset_sizes[cid] for cid in self._epoch_clients} # batches the client still has to process

        # bookkeeping / debugging
        self._allocated_layer_history = []
        self.tot_server_computation_time = 0.

        # reinitialize the client times
        self._client_training_times = {}
        self._client_throughput = {}

        self._init_reallocation_thread()
        return [self._client_allocated_layers[proxy.cid]  for proxy in epoch_clients]

    def trigger_reallocation(self) -> None:
        start_time = time.time()
        # print(self._client_train_batches_count)
        print({self._cid_to_node_name[k]: v for k, v in self._client_training_times.items() if k in self._client_layer_diff})
        print({self._cid_to_node_name[k]: v for k, v in self._client_allocated_layers.items() if k in self._client_layer_diff})
        print({self._cid_to_node_name[k]: v for k, v in self._pending_batches.items() if k in self._client_layer_diff})
        self._allocated_layer_history.append(self._client_allocated_layers.copy())
        if any(v < 6 for v in self._client_train_batches_count.values()):
            print("Not enough data to try reallocating...")
            return
        if len(self._client_layer_diff) == 0:
            print("All have completed training. Exiting...")
            return
        if self.layer_allocation_optimizer == "genetic":
            best_config = self._genetic_reallocation()

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

    def _genetic_reallocation(self, num_generations=1000, pop_size=50, mutation_rate=0.2):
        clients = list(self._epoch_clients)
        split_points = self.meaningful_split_points
        baseline = [self._client_allocated_layers[cid] for cid in clients]

        cache = {}
        def fitness(L):
            L = tuple(L)
            if L in cache:
                return cache[L]
            layer_config = {cid: L[i] for i, cid in enumerate(clients)}
            total_time = self._compute_objective(layer_config)
            cache[L] = -total_time
            return -total_time

        def local_mutation(L):
            """Mutate one or more clients by ±1 split point."""
            new_L = L.copy()
            for i in range(len(L)):
                if random.random() < mutation_rate:
                    pos = split_points.index(L[i])
                    step = random.choice([-1, +1])
                    new_pos = max(
                        0,
                        min(
                            len(split_points) - 1,
                            pos + step
                        )
                    )
                    new_L[i] = split_points[new_pos]
            return new_L

        # --- Initialize population near baseline ---
        population = [baseline.copy()]
        while len(population) < pop_size:
            child = local_mutation(baseline)
            population.append(child)

        # --- Evolution loop ---
        best_solution = baseline.copy()
        best_fitness = fitness(best_solution)
        iter_wo_improvement = 0

        for gen in range(num_generations):
            scores = [fitness(ind) for ind in population]
            ranked = sorted(zip(scores, population), reverse=True)
            best_solution = ranked[0][1]
            tmp_best_fitness = ranked[0][0]
            # print(tmp_best_fitness, best_fitness)
            if tmp_best_fitness > best_fitness:
                iter_wo_improvement = 0
            iter_wo_improvement += 1
            best_fitness = ranked[0][0]
            if iter_wo_improvement > 20:
                print("Breaking", gen)
                break

            # print(f"[GA] Gen {gen}: best total_time = {-best_fitness:.3f}")
            next_pop = [ind for _, ind in ranked[:4]]
            # Generate mutants
            while len(next_pop) < pop_size:
                parent = random.choice(ranked[:10])[1]
                child = local_mutation(parent)
                next_pop.append(child)

            population = next_pop

        return {cid: best_solution[i] for i, cid in enumerate(clients)}

    def _compute_objective(self, layer_config):
        pred_client_times = {}
        server_time = 0.
        communication_time = 0.
        for cid, L in layer_config.items():
            B = self._pending_batches[cid]
            server_time += B * self._server_training_times[L] / self._num_concat_batches
            pred_client_times[cid] = self.compute_client_time(cid, L)
            communication_time += B * self.per_batch_server_communication_time[L]

        # additionally penalize every layer change
        server_time = max(server_time, communication_time * 1.2)

        reg_term = sum([
            layer_config[cid] != self._client_allocated_layers[cid] for cid in layer_config
        ])
        max_client_time = max(pred_client_times.values())
        total_time = max(server_time, max_client_time) + reg_term
        return total_time

    def compute_client_time(self, cid, L):
        factor = self.complexity_factors[(L, self._client_allocated_layers[cid])]
        if L in self.meaningful_split_points[3:]:
            factor *= 2.0
        if self.meaningful_split_points.index(L) - self.meaningful_split_points.index(self._client_allocated_layers[cid]) > 2:
            factor *= 1.5
        per_batch_time = self._client_training_times[cid] * factor # client training
        per_batch_time += self._server_training_times[L] # server training
        per_batch_time += (self.per_batch_server_communication_time[L] * 1.2) # * 1.2 to account for server overhead for serialization/deserialization
        reg_term = 0. if L == self._client_allocated_layers[cid] else 2.0
        return self._pending_batches[cid] * per_batch_time + reg_term

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
        try:
            del self._client_layer_diff[cid]
        except:
            pass

    def update_client_metrics(self, client_id: str, metrics: Optional[TimingMetrics], num_received_bytes: int):
        if metrics is None or metrics.client_local_computation_time < 0:
            self._pending_batches[client_id] -= 1
            return

        client_train_time = metrics.client_local_computation_time
        metrics.compute_throughput(num_received_bytes)
        client_Mbps = metrics.throughput
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
        self._epoch_clients = None
        self._client_allocated_layers = None
        self._reallocation_stop_event.set()
        self._reallocation_thread.join(timeout=5.0)

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

    def _rolling_average(self, average, new_value):
        return 0.8 * average + 0.2 * new_value

    def get_max_client_layers(self, cid):
        return min(self._per_client_max_layers[cid], self._max_client_layers)
