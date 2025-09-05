from collections import OrderedDict
from typing import List, Tuple

from flwr.common import FitIns, EvaluateIns
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from slbd.algos.base_strategy import BaseStrategy
from slbd.server.client_layer_allocator import ClientLayerAllocator


class ARESStrategy(BaseStrategy):

    def _ares_configure_epoch(
        self,
        server_round: int,
        client_manager,
        client_layer_allocator,
        model_state_dict,
        config,
        is_training,
    ):
        clients = self.sample_epoch_clients(client_manager, is_training)
        num_initial_client_layers = client_layer_allocator \
            .initialize_epoch(
                clients, client_manager, is_training,
                optimizer_config=self.optimizer_config_generator.get_optimizer_config()
            )

        all_clients_fit_ins = self._setup_epoch_clients(
            server_round, clients, num_initial_client_layers, model_state_dict, is_training, config
        )
        all_server_models_fit_ins = self._setup_epoch_server_models(
            [(
                proxy.cid,
                num_initial_client_layers[idx],
                proxy.cid
            ) for idx, proxy in enumerate(clients)],
            model_state_dict, is_training, config
        )
        return all_clients_fit_ins, all_server_models_fit_ins

    def configure_fit(
        self,
        server_round: int,
        model_state_dict: OrderedDict,
        client_manager: ClientManager,
        client_layer_allocator: ClientLayerAllocator
    ) -> List[Tuple[ClientProxy, FitIns]]:
        # In ARES every client gets its own private classification head
        config = self.optimizer_config_generator.get_optimizer_config()
        if self.config_client_fit_fn:
            config |= self.config_client_fit_fn(server_round)
        return self._ares_configure_epoch(
            server_round=server_round,
            client_manager=client_manager,
            client_layer_allocator=client_layer_allocator,
            model_state_dict=model_state_dict,
            config=config,
            is_training=True
        )

    def configure_evaluate(
        self,
        server_round: int,
        model_state_dict: OrderedDict,
        client_manager: ClientManager,
        client_layer_allocator: ClientLayerAllocator,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        # In ARES every client gets its own private classification head
        _ = (server_round, )
        config = {}
        return self._ares_configure_epoch(
            server_round=server_round,
            client_manager=client_manager,
            client_layer_allocator=client_layer_allocator,
            model_state_dict=model_state_dict,
            config=config,
            is_training=False
        )

    def route_client_request_to_server_model(self, client_id):
        return client_id
