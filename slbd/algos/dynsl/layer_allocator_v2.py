import time
import threading
import itertools
from logging import INFO
from datetime import datetime
from functools import partial
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from flwr.common import GetPropertiesIns, log
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager

from slbd.server.client_layer_allocator import ClientLayerAllocator
from slbd.common.typing import SocketTimingMetrics
from slbd.common.constants import (
    MIN_NUMBER_OF_CLIENT_LAYERS,
    CLIENT_DATASET_SIZE,
    CLIENT_DEVICE_TYPE,
    MAX_NUMBER_OF_CLIENT_LAYERS,
)
from slbd.algos.dynsl.optimization_algorithms import exact_solution_v2
from slbd.algos.dynsl.utils import compute_server_time



class DynSLClientLayerAllocatorV2(ClientLayerAllocator):

    def __init__(
        self,
        num_model_layers: int,
        reallocation_frequency: float,
        gamma: float,
        num_concat_batches: int,
        meaningful_split_points: List[int],
        modeled_layer_complexities: List[float],
        static_server_computational_times: List[float],
        per_layer_emb_sizes: List[int],
        per_layer_parameters: Dict[int, int]
    ):
        """Initialize the allocator

        Parameters
        ----------
        num_model_layers : int
            Number of layers that the model consists of
        reallocation_frequency : float
            How often (seconds) the reallocation strategy should run
        gamma : float
            Minimum benefit to update the split vector
        num_concat_batches : int
            How many batches are being concatenated before starting the forward pass
        meaningful_split_points : List[int]
            List of indices where it makes sense to split the model
        modeled_layer_complexities : List[float]
            A "model" stating how complex every layer is (e.g., determined via FLOPS or by running
            the model on the server)
        per_layer_emb_sizes : List[int]
            Size (# parameters) per every split point. Here we are talking about embeddings/gradients
        per_layer_parameters : Dict[int]
            Size of model layers. For each key (layer idx) we get the number of FP values
        """
        # safety checks
        assert reallocation_frequency > 0.
        print("Using updated version")

        # store basic attributes
        self._gamma = gamma
        self._num_model_layers: int = num_model_layers
        self._reallocation_frequency = reallocation_frequency
        self._meaningful_split_points = meaningful_split_points

        # client desires -- client sets the maximum and minimum number of layers it is willing to train
        self._per_client_min_layers: Dict[str, int] = {}
        self._per_client_max_layers: Dict[str, int] = {}

        # per-client dataset and pending batches
        self._dataset_sizes: Dict[str, int] = {}
        self._pending_batches: Dict[str, int] = {}

        # debugging / tracking
        self._allocated_layer_history = []
        self._cid_to_device_type_mapping = {}
        self._num_concat_batches = num_concat_batches
        self._tot_server_computation_time = None
        self._cid_to_node_name = {}

        # rolling averages -- note, they get re-initialized every time the client changes its number of layers
        # they are persisted across epochs if the allocator is configured with caching
        self._client_training_times: Dict[str, List[float]] = {}
        self._client_backward_times: Dict[str, float] = {}
        self._client_throughput: Dict[str, float] = {}

        # event for starting stopping the thread that handles layer re-allocation
        self._reallocation_stop_event = None

        # self._baseline_computational_model_per_split_point is a dictionary that, for every split
        # point (key), contains the computational requirements of training the model up to that point
        modeled_layer_complexities = [modeled_layer_complexities[idx] for idx in range(len(modeled_layer_complexities))]
        # print(modeled_layer_complexities)
        self.comput_compl_per_sp = [sum(modeled_layer_complexities[:start_idx]) for start_idx in range(1, num_model_layers+1)]

        # we assume that the training time on the server do not vary
        static_server_computational_times = \
            [static_server_computational_times[idx] for idx in range(len(static_server_computational_times))]
        self._server_training_times = {
            start_idx: sum(static_server_computational_times[start_idx:])
            for start_idx in self._meaningful_split_points
        }

        # compute the time the server will spend communicating every batch of data given the current split point
        # ==>> in the formula:
        # * 2 ... forward+backward // the server will have to send and recive the data over a single link
        # * 32 ... 32 bit representation of FP values
        self.per_batch_server_communication_time = {
            k: 2 * 32 * per_layer_emb_sizes[k-1] / 1e9 if k != num_model_layers else 0.
            for k in self._meaningful_split_points
        }
        # print(per_layer_communication_volume)
        # * 32 as we use 32 bit FP representation,  / 1e6 to convert bits in megabits
        self._per_batch_embedding_sizes_Mbits = {
            k: per_layer_emb_sizes[k-1] * 32 / 1e6 if k != num_model_layers else 0
            for k in self._meaningful_split_points
        }
        # print("per batch", self._per_batch_embedding_sizes_Mbits)
        tmp_split_points = [0,] + list(meaningful_split_points)
        per_layer_parameters = [per_layer_parameters[idx] for idx in range(len(per_layer_parameters))]
        self._per_layer_model_parameters_Mbits = {
            (a, b): sum(per_layer_parameters[a:b]) * 32 / 1e6
            for a, b in itertools.product(tmp_split_points, tmp_split_points)
            if a < b
        }

    def compute_server_time(self, candidate_split_vector):
        return compute_server_time(
            candidate_split_vector=candidate_split_vector,
            pending_batches=self._pending_batches,
            server_training_times=self._server_training_times,
            per_batch_server_communication_time=self.per_batch_server_communication_time,
            num_concat_batches=self._num_concat_batches,
        )

    def _get_client_properties(self, client_proxy: ClientProxy) -> int:
        if client_proxy.cid in self._per_client_min_layers:
            log(INFO, f"Already have data for client {client_proxy.cid}")
            assert client_proxy.cid in self._dataset_sizes
            assert client_proxy.cid in self._cid_to_device_type_mapping
            return

        # Call get_properties on client to fetch their requirements (e.g., min/max num. of layers to train)
        result = client_proxy.get_properties(
            GetPropertiesIns({}), timeout=None, group_id=None
        )
        min_client_layers = int(result.properties[MIN_NUMBER_OF_CLIENT_LAYERS])

        # align min client layers to the model configuration. Find the first value in
        # model.meaningful_split_points that is larger or equal to min layers
        self._per_client_min_layers[client_proxy.cid] = \
            next((point for point in self._meaningful_split_points if point >= min_client_layers), None)
        assert self._per_client_min_layers[client_proxy.cid] is not None

        max_client_layers = int(result.properties[MAX_NUMBER_OF_CLIENT_LAYERS])
        self._per_client_max_layers[client_proxy.cid] = \
            next((point for point in reversed(self._meaningful_split_points) if point <= max_client_layers), None)

        self._dataset_sizes[client_proxy.cid] = int(result.properties[CLIENT_DATASET_SIZE])
        self._cid_to_device_type_mapping[client_proxy.cid] = result.properties[CLIENT_DEVICE_TYPE]
        node_name = result.properties["node_name"]
        self._cid_to_node_name[client_proxy.cid] = node_name
        log(INFO, f"Client {client_proxy.cid} requires at least {min_client_layers} layers [{result.properties[CLIENT_DEVICE_TYPE]} -- {node_name}]")

    def _init_reallocation_thread(self):
        self._reallocation_stop_event = threading.Event()

        # Start the periodic reallocation task if frequency is positive
        def periodic_reallocation():
            while not self._reallocation_stop_event.wait(self._reallocation_frequency):
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
        initial_config_dict = self._per_client_min_layers

        # init dictionaries storing client configuration
        self._client_layer_diff = {cid: 0 for cid in self._epoch_clients} # temp difference in the number of layers of clients
        self._client_train_batches_count = {cid: 0 for cid in self._epoch_clients} # number of batches the client has processed since it last received a reallocation
        self._client_allocated_layers = {cid: initial_config_dict[cid] for cid in self._epoch_clients} # layers every client has
        self._pending_batches = {cid: self._dataset_sizes[cid] for cid in self._epoch_clients} # batches the client still has to process
        self._client_was_decreased_last_iteration = {cid: False for cid in self._epoch_clients} # track if client was decreased in previous reallocation

        # bookkeeping / debugging
        self._allocated_layer_history = []
        self._tot_server_computation_time = 0.
        self.is_first_iteration = True

        self._init_reallocation_thread()
        return [self._client_allocated_layers[proxy.cid]  for proxy in epoch_clients]

    def trigger_reallocation(self) -> None:
        start_time = time.time()
        log(INFO, f"Running reallocation {datetime.now().strftime('%H:%M:%S')}")
        log(INFO, "Allocated layers", {self._cid_to_node_name[k]: v for k, v in self._client_allocated_layers.items()})
        print("--- layers ---", {self._cid_to_node_name[k]: v for k, v in self._client_allocated_layers.items()}, "---")
        # Uncomment following lines to log the tracked metrics

        # log(INFO, "COUNTS=",{self._cid_to_node_name[k]: v for k, v in self._client_train_batches_count.items() if self._client_allocated_layers[k] != self._num_model_layers})
        self._allocated_layer_history.append({self._cid_to_node_name[k]: v for k, v in self._client_allocated_layers.items()})
        if any(v < 2 for k, v in self._client_train_batches_count.items() if self._client_allocated_layers[k] != self._num_model_layers):
            log(INFO, "Not enough data to try reallocating...\n\n")
            return
        if len(self._client_layer_diff) == 0:
            log(INFO, "All have completed training. Exiting...")
            return

        # print("_client_training_times=",{self._cid_to_node_name[k]: v for k, v in self._client_training_times.items()})
        # print("_pending_batches=",{self._cid_to_node_name[k]: v for k, v in self._pending_batches.items()})
        # print("_client_throughput=",{self._cid_to_node_name[k]: v for k, v in self._client_throughput.items()})
        # print("_per_client_max_layers=",{self._cid_to_node_name[k]: v for k, v in self._per_client_max_layers.items()})
        # print("_client_backward_times=", {self._cid_to_node_name[k]: v for k, v in self._client_backward_times.items()})
        # print("_server_training_times=", self._server_training_times)
        # print("_per_batch_embedding_sizes_Mbits=", self._per_batch_embedding_sizes_Mbits)
        # print("per_batch_server_communication_time=", self.per_batch_server_communication_time)
        # print("comput_compl_per_sp=", self.comput_compl_per_sp)
        # print("_per_layer_model_parameters_Mbits=", self._per_layer_model_parameters_Mbits)
        # print("meaningful_split_points=", self._meaningful_split_points)
        tmp_max_layers = {}
        for k, v in self._per_client_max_layers.items():
            if self._client_was_decreased_last_iteration.get(k, False):
                # If client was decreased last iteration, max is current layers
                tmp_max_layers[k] = self._client_allocated_layers[k]
            # elif self._client_allocated_layers[k] == self._meaningful_split_points[0]:
            #     tmp_max_layers[k] = min(v, self._meaningful_split_points[2+self._meaningful_split_points.index(self._client_allocated_layers[k])])
            else:
                # Default case
                tmp_max_layers[k] = v
        best_config = exact_solution_v2(
            clients=self._epoch_clients,
            per_client_max_layers=tmp_max_layers,
            meaningful_split_points=self._meaningful_split_points,
            allocated_layers_now=self._client_allocated_layers,
            client_fn=self.compute_client_time,
            server_fn=self.compute_server_time,
            gamma=self._gamma,
        )

        J_current = self._compute_objective(self._client_allocated_layers)
        J_new = self._compute_objective(best_config)
        delta = J_current - J_new
        log(INFO, f"Current cost: {J_current:.3f}")
        log(INFO, f"Found cost: {J_new:.3f}")

        # if delta > self._gamma or (self.is_first_iteration and sum(self._client_allocated_layers[k] < best_config[k] for k in self._epoch_clients) > 5):
        if delta > self._gamma or (self.is_first_iteration and best_config != self._client_allocated_layers):
            log(INFO, f"[BS] Adopt new configuration (ΔJ={delta:.3f} > γ={self._gamma})")
            for cid, new_layers in best_config.items():
                current_client_layers = self._client_allocated_layers[cid]
                self._client_layer_diff[cid] = new_layers - current_client_layers
                # Track if client was decreased
                self._client_was_decreased_last_iteration[cid] = new_layers < current_client_layers
            self.is_first_iteration = False
        else:
            log(INFO, f"[BS] Keep current configuration (ΔJ={delta:.3f} ≤ γ={self._gamma})")
            # No changes, so no decreases
            for cid in self._epoch_clients:
                self._client_was_decreased_last_iteration[cid] = False
        log(INFO, f"Total reallocation time: {time.time() - start_time:4f}")

    def _compute_objective(self, layer_config: Dict[str, int]):
        return max(
            self.compute_server_time(layer_config),
            max(self.compute_client_time(cid, sp) for cid, sp in layer_config.items())
        )

    def compute_client_time(self, cid: str, L: int):
        """ Computes the remaining epoch time for the client `cid` if the client were to train `L` layers

        we have the training time of the client for the configuration `self._client_allocated_layers`
        that is, when the client has `self._client_allocated_layers[cid]` layers, its local training time
        is `self._client_training_times[cid]`. To compute the predicted training time for a different number
        of layers, we perform a simple linear interpolation.
        """
        per_batch_time = self._estimate_client_training_time(cid, L) # client computation
        per_batch_time += self._server_training_times[L] # server training
        per_batch_time += 2 * self._per_batch_embedding_sizes_Mbits[L] / self._client_throughput[cid] # communication

        current_num_layers_on_device = self._client_allocated_layers[cid]
        layer_movement_time = 0.
        # if current_num_layers_on_device == L -- there is no additional layer movement ==>> layer_movement_time=0
        # if current_num_layers_on_device > L -- layers are moved from device to server. This has to happend either
        #           at the end of the epoch, so we say that the layer_movement_time is still 0.
        # thus, we only consider layer movement time to the client
        if current_num_layers_on_device < L:
            num_moved_layer_Mbits = self._per_layer_model_parameters_Mbits[(current_num_layers_on_device, L)]
            layer_movement_time = 2 * num_moved_layer_Mbits / self._client_throughput[cid]
        return self._pending_batches[cid] * per_batch_time + layer_movement_time

    def _estimate_client_training_time(self, cid, layer):
        """ Predicts how much time training would take if client `cid` were to train `layer` layers
        """
        if self._client_allocated_layers[cid] == self._num_model_layers:
            return 0.

        num_client_layers = self._client_allocated_layers[cid]

        if layer == num_client_layers:
            # no need to do any estimation -- the client currently trains the number of layers
            return self._client_training_times[cid][-1] + self._client_backward_times[cid]
        if layer < num_client_layers:
            # we know exactly what is the forward time of the client, as the client reports what is the training time
            # for every layer until its last one
            # -1 because because lists are 0 indexed
            predicted_forward_time = self._client_training_times[cid][layer-1]

        else:
            # need to predict the complexity for a **larger** number of layers than the client currently has
            recorded_layers_forward_time = [0,] + self._client_training_times[cid]
            base_complexity = [0,] + self.comput_compl_per_sp[:num_client_layers]
            assert len(base_complexity) == len(recorded_layers_forward_time)
            # now we know the forward time for every layer until the current number of layers of the model
            # we do a linear interpolation, by taking the `self.comput_compl_per_sp` as the complexity model

            # fit linear model for simplicity
            coefficients = np.polyfit(base_complexity, recorded_layers_forward_time, 1)
            new_num_layers_complexity = self.comput_compl_per_sp[layer-1]
            predicted_forward_time = np.polyval(coefficients, new_num_layers_complexity) * 1.5
            if layer == self._num_model_layers:
                predicted_forward_time *= 1.25

        # now we know an estimate of the forward time -- we still need to account for the backward pass
        current_fwd_time = self._client_training_times[cid][-1]
        tot_current_client_time = current_fwd_time + self._client_backward_times[cid]
        factor = tot_current_client_time / current_fwd_time
        return predicted_forward_time * factor

    def update_server_metrics(
        self,
        num_offloaded_layers: int,
        elapsed_time: float,
    ):
        self._tot_server_computation_time += elapsed_time

    def get_post_reallocation_layer_difference(self, client_id: str) -> Optional[Tuple[int, int]]:
        previous = self._client_allocated_layers[client_id]
        if self._client_layer_diff[client_id] != 0:
            diff = self._client_layer_diff[client_id]
            self._client_allocated_layers[client_id] += diff
            self._client_layer_diff[client_id] = 0
            self._client_train_batches_count[client_id] = 0
            del self._client_training_times[client_id]
            if self._client_allocated_layers[client_id] == self._num_model_layers:
                self._pending_batches[client_id] = 0
            return previous, diff
        return None

    def remove_client(self, cid):
        print(f"Removing {cid} as it has finished", self._pending_batches[cid])
        self._epoch_clients.remove(cid)
        del self._client_layer_diff[cid]
        if cid in self._client_train_batches_count:
            del self._client_train_batches_count[cid]
        # we remove the cid from self._client_layer_diff
        # ==>> if a cid is not in self._client_layer_diff, it means the client has finished training

    def update_client_metrics(self, client_id: str, metrics: Optional[SocketTimingMetrics]):
        if metrics.tot_backward_time < 0:
            self._pending_batches[client_id] -= 1
            return

        client_Mbps = metrics.throughput_Mbps
        # we already handled the case where we could not handle the throughput
        assert client_Mbps > 0
        assert len(metrics.per_layer_forward_times) == self._client_allocated_layers[client_id]
        self._client_throughput.setdefault(client_id, client_Mbps)
        self._client_backward_times.setdefault(client_id, metrics.tot_backward_time)

        self._client_throughput[client_id] = \
            self._rolling_average(self._client_throughput[client_id], client_Mbps)
        self._client_backward_times[client_id] = \
            self._rolling_average(self._client_backward_times[client_id], metrics.tot_backward_time)

        assert len(metrics.per_layer_forward_times) == self._client_allocated_layers[client_id]
        if client_id not in self._client_training_times:
            # reallocation happened -- start measuring from scratch, i.e., we override any pre-existing value
            self._client_training_times[client_id] = metrics.per_layer_forward_times
        else:
            # no reallocation happened -- we know metrics.per_layer_forward_times has the same length as
            # self._client_training_times[client_id] ==> we update each value independently with
            # rolling average
            for idx, value in enumerate(metrics.per_layer_forward_times):
                self._client_training_times[client_id][idx] = self._rolling_average(self._client_training_times[client_id][idx], value)

        self._client_train_batches_count[client_id] += 1
        self._pending_batches[client_id] -= 1

    def finalize_epoch(self):
        self._epoch_clients = []
        self._reallocation_stop_event.set()
        self._reallocation_thread.join(timeout=5.0)
        self._client_allocated_layers.clear()
        self._client_training_times.clear()
        self._client_throughput.clear()
        self._client_backward_times.clear()

    def collect_epoch_metrics(self):
        return {
            "train/tot_server_training_time": self._tot_server_computation_time,
            "train/layer_history": self._allocated_layer_history,
            "train/client_throughputs": list(self._client_throughput.items()),
        }

    def get_client_offloaded_layer_count(self, client_id):
        # returns the number of layers that the server needs to perform
        # i.e., """model[-this_number:](input)"""
        return self._num_model_layers - self._client_allocated_layers[client_id]

    def _rolling_average(self, average, new_value):
        return 0.7 * average + 0.3 * new_value

    def get_max_client_layers(self, cid):
        return self._per_client_max_layers[cid]

    def get_min_client_layers(self, cid):
        return self._per_client_min_layers[cid]
