from typing import List

from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager


class OracleSimulationClientLayerAllocator:

    def __init__(
        self,
        num_model_layers,
        meaningful_split_points,
        client_configuration,
    ):
        self.num_model_layers = num_model_layers
        self._num_offloaded_layers = {}
        # flatten client_configuration
        self.client_configuration = {}
        for config in client_configuration:
            client_ids = tuple(map(int, config["ids"].split("-")))
            for cid in range(*client_ids):
                assert cid not in self.client_configuration
                self.client_configuration[cid] = config["num_layers"]
        assert set(self.client_configuration.keys()) == set(range(len(self.client_configuration)))
        print(self.client_configuration)

    def initialize_epoch(
        self,
        epoch_clients: List[ClientProxy],
        client_manager: ClientManager,
        is_training: bool,
    ) -> None:
        self._num_offloaded_layers = {
            proxy.cid: self.num_model_layers - self.client_configuration[proxy.cid]
            for proxy in epoch_clients
        }
        return [self.client_configuration[proxy.cid] for proxy in epoch_clients]

    def finalize_epoch(self):
        return

    def get_client_offloaded_layer_count(self, client_id):
        return self._num_offloaded_layers[client_id]

    def collect_epoch_metrics(self):
        return {}

    def update_client_metrics(self, *args, **kwargs):
        return

    def update_server_metrics(self, *args, **kwargs):
        return

    def get_post_reallocation_layer_difference(self, *args, **kwargs):
        return
