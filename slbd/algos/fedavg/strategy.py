from collections import OrderedDict
from typing import List, Tuple

from flwr.common import FitIns, EvaluateIns
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from slbd.server.client_layer_allocator import ClientLayerAllocator

from slbd.algos.base_strategy import BaseStrategy


class FedAvgStrategy(BaseStrategy):

    def configure_fit(
        self,
        server_round: int,
        model_state_dict: OrderedDict,
        client_manager: ClientManager,
        client_layer_allocator: ClientLayerAllocator
    ) -> List[Tuple[ClientProxy, FitIns]]:
        config = {}
        if self.config_client_fit_fn:
            config = self.config_client_fit_fn(server_round)

        clients = self.sample_epoch_clients(client_manager, True)
        num_initial_client_layers = client_layer_allocator \
            .initialize_epoch(clients, client_manager, True)

        all_clients_fit_ins = self._setup_epoch_clients(
            server_round, clients, num_initial_client_layers, model_state_dict, True, config
        )
        return all_clients_fit_ins, []

    def configure_evaluate(
        self,
        server_round: int,
        model_state_dict: OrderedDict,
        client_manager: ClientManager,
        client_layer_allocator: ClientLayerAllocator,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        _ = (server_round, )
        config = {}
        clients = self.sample_epoch_clients(client_manager, False)

        num_initial_client_layers = client_layer_allocator.initialize_epoch(clients, client_manager, False)
        all_clients_eval_ins = self._setup_epoch_clients(
            server_round, clients, num_initial_client_layers, model_state_dict, True, config
        )

        return all_clients_eval_ins, []
