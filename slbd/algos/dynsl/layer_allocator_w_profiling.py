import itertools
from logging import INFO
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

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

from slbd.algos.dynsl.optimization_algorithms import binary_search
from slbd.server.client_layer_allocator import ClientLayerAllocator


class DynSLClientLayerAllocatorWProfiling(ClientLayerAllocator):

    def __init__(
        self,
        max_client_layers: int,
        num_model_layers: int,
        num_concat_batches: int,
        layer_allocation_optimizer: str,
        meaningful_split_points: List[int],
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
        num_concat_batches : int
            How many batches are being concatenated before starting the forward pass
        layer_allocation_optimizer : str
            Strategy to be used for computing the new split vector
        meaningful_split_points : List[int]
            List of indices where it makes sense to split the model
        per_layer_emb_sizes : List[int]
            Size (# parameters) per every split point. Here we are talking about embeddings/gradients
        per_layer_parameters : Dict[int]
            Size of model layers. For each key (layer idx) we get the number of FP values
        """
        # safety checks
        assert layer_allocation_optimizer == "binary_search"

        # store basic attributes
        self._num_model_layers: int = num_model_layers
        self._num_concat_batches = num_concat_batches
        self.meaningful_split_points = [msp for msp in meaningful_split_points if msp <= max_client_layers]
        self._max_client_layers = next((point for point in reversed(self.meaningful_split_points) if point <= max_client_layers))

        # client desires -- client sets the maximum and minimum number of layers it is willing to train
        self._per_client_min_layers: Dict[str, int] = {}
        self._per_client_max_layers: Dict[str, int] = {}

        # per-client dataset and pending batches
        self._dataset_sizes: Dict[str, int] = {}

        # debugging / tracking
        self._cid_to_device_type_mapping = {}
        self.tot_server_computation_time = None
        self._cid_to_node_name = {}

        # metrics for layer allocation
        static_server_computational_times = \
            [static_server_computational_times[idx] for idx in range(len(static_server_computational_times))]
        self._server_training_times = {
            start_idx: sum(static_server_computational_times[start_idx:])
            for start_idx in self.meaningful_split_points
        }
        self._client_profiled_times = {}
        self.per_batch_server_communication_time = {
            k: 2 * 32 * per_layer_emb_sizes[k] / 1e9 # server has a 1Gbps connection link, which is shared across all clienst
            for k in self.meaningful_split_points
        }
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

    def compute_server_time(self, candidate_split_vector):
        server_time = 0. # Compute the time the server needs to train with the current configuration
        server_communication_time = 0. # time the server is communicating

        for cid, L in candidate_split_vector.items():
            B = self._dataset_sizes[cid] # number of batches the client still has to process
            server_time += B * self._server_training_times[L] / self._num_concat_batches # time the server will be processing client batches
            server_communication_time += B * self.per_batch_server_communication_time[L]

        # slightly increase the server_communication_time to account for the overhead of serializing and deserializing
        server_time = max(server_time, server_communication_time * 1.2)

        return server_time

    def _get_client_properties(self, client_proxy: ClientProxy) -> int:
        if client_proxy.cid in self._per_client_min_layers:
            log(INFO, f"Already have data for client {client_proxy.cid}")
            assert client_proxy.cid in self._dataset_sizes
            assert client_proxy.cid in self._cid_to_device_type_mapping
            assert client_proxy.cid in self._client_profiled_times
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

        # extract layer complexities
        tmp_layer_complexities = {int(k[4:]): v for k, v in result.properties.items() if k.startswith("__lc")}
        self._client_profiled_times[client_proxy.cid] = [tmp_layer_complexities[idx] for idx in range(len(tmp_layer_complexities))]
        print("Profiled client times", self._client_profiled_times[client_proxy.cid])
        node_name = result.properties["node_name"]
        self._cid_to_node_name[client_proxy.cid] = node_name
        log(INFO, f"Client {client_proxy.cid} requires at least {min_client_layers} layers [{result.properties[CLIENT_DEVICE_TYPE]} -- {node_name}]")

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
        self.tot_server_computation_time = 0.

        min_layers = self.meaningful_split_points[0]
        self._client_allocated_layers = {cid: min_layers for cid in self._epoch_clients}
        self._client_allocated_layers = binary_search(
            clients=self._epoch_clients,
            meaningful_split_points=self.meaningful_split_points,
            allocated_layers_now=self._client_allocated_layers, # for simplicity
            client_fn=self.compute_client_time,
            server_fn=self.compute_server_time,
            gamma=0.2
        )
        print("Allocated layers", {self._cid_to_node_name[k]: v for k, v in self._client_allocated_layers.items()})
        return [self._client_allocated_layers[cid] for cid in self._epoch_clients]

    def compute_client_time(self, cid, L):

        # client computation
        per_batch_time = self._client_profiled_times[cid][L]

        # server computation
        per_batch_time += self._server_training_times[L] # server training
        # communication -- * 2 because we need to both send and receive the data
        per_batch_time += 2 * self._per_batch_embedding_sizes_Mbits[L] / 1e3

        current_num_layers_on_device = self._client_allocated_layers[cid]

        layer_movement_time = 0.
        # if current_num_layers_on_device == L -- there is no additional layer movement ==>> layer_movement_time=0
        # if current_num_layers_on_device > L -- layers are moved from device to server. This has to happend either
        #           at the end of the epoch, so we say that the layer_movement_time is still 0.
        if current_num_layers_on_device < L:
            num_moved_layer_Mbits = self._per_layer_model_parameters_Mbits[(current_num_layers_on_device, L)]
            layer_movement_time = 2 * num_moved_layer_Mbits / 1_000
        return self._dataset_sizes[cid] * per_batch_time + layer_movement_time

    def update_server_metrics(
        self,
        num_offloaded_layers: int,
        elapsed_time: float,
    ):
        self.tot_server_computation_time += elapsed_time

    def get_post_reallocation_layer_difference(self, client_id: str) -> Optional[Tuple[int, int]]:
        return None

    def remove_client(self, cid):
        return

    def update_client_metrics(self, client_id: str, metrics: Optional[TimingMetrics]):
        return

    def finalize_epoch(self):
        return

    def get_client_offloaded_layer_count(self, client_id):
        # returns the number of layers that the server needs to perform
        # i.e., """model[-this_number:](input)"""
        return self._num_model_layers - self._client_allocated_layers[client_id]

    def collect_epoch_metrics(self):
        return {
            "tot_server_training_time": self.tot_server_computation_time,
            "allocated_layers": self._client_allocated_layers,
        }
