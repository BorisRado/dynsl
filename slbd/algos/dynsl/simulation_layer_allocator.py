from typing import List, Optional, Tuple

import numpy as np
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager

from slbd.common import LayerChangeInfo
from slbd.server.client_layer_allocator import ClientLayerAllocator
from slbd.common.constants import CLIENT_DATASET_SIZE


class DynSLSimulationClientLayerAllocator:

    def __init__(
        self,
        num_model_layers,
        meaningful_split_points,
        client_configuration,
    ):
        self.num_model_layers = num_model_layers
        self.meaningful_split_points = meaningful_split_points

        # flatten client_configuration
        self.client_configuration = {}
        for config in client_configuration:
            client_ids = tuple(map(int, config["ids"].split("-")))
            for cid in range(*client_ids):
                client_config = []
                assert cid not in self.client_configuration
                for evol_step in config["evolution"]:
                    client_config.append((evol_step["start"], evol_step["num_layers"]))
                client_config.append((1.0, -1))
                self.client_configuration[cid] = client_config
        assert set(self.client_configuration.keys()) == set(range(len(self.client_configuration)))
        self.epoch_progression = {}

    def initialize_epoch(
        self,
        epoch_clients: List[ClientProxy],
        client_manager: ClientManager,
        is_training: bool,
    ) -> None:
        self.num_batches_per_client = {client.cid: 0 for client in epoch_clients}
        for proxy in epoch_clients:
            props = proxy.get_properties(None)
            cds = props.properties[CLIENT_DATASET_SIZE]
            progress = self.client_configuration[proxy.cid]
            progress_lst = np.full((cds+1,), progress[0][1])
            for idx in range(1, len(progress)-1):
                start = progress[idx][0]
                num_layers = progress[idx][1]
                end = progress[idx + 1][0]
                for idx in range(int(start * cds), int(end * cds)):
                    progress_lst[idx] = num_layers
            progress_lst[-1] = progress_lst[-2]
            self.epoch_progression[proxy.cid] = progress_lst
        self._num_offloaded_layers = {
            proxy.cid: self.num_model_layers - self.client_configuration[proxy.cid][0][1]
            for proxy in epoch_clients
        }
        return [self.epoch_progression[proxy.cid][0] for proxy in epoch_clients]

    def finalize_epoch(self):
        return

    def get_client_offloaded_layer_count(self, client_id):
        return self._num_offloaded_layers[client_id]

    def collect_epoch_metrics(self):
        return {}

    def get_max_client_layers(self, cid):
        return self.meaningful_split_points[-1]

    def get_min_client_layers(self, cid):
        return self.meaningful_split_points[0]

    def update_client_metrics(self, *args, **kwargs):
        return

    def update_server_metrics(self, *args, **kwargs):
        return

    def get_post_reallocation_layer_difference(self, cid) -> Optional[Tuple[int, int]]:
        self.num_batches_per_client[cid] += 1
        idx = self.num_batches_per_client[cid]
        if self.epoch_progression[cid][idx] == self.epoch_progression[cid][idx -1]:
            return None
        diff = self.epoch_progression[cid][idx] - self.epoch_progression[cid][idx-1]
        self._num_offloaded_layers[cid] -= diff
        return self.epoch_progression[cid][idx-1], diff
