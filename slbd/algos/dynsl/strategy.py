from collections import OrderedDict, defaultdict
from logging import ERROR
from typing import List, Tuple, Union, Dict

import wandb
from flwr.common import (
    FitRes,
    FitIns,
    Scalar,
    parameters_to_ndarrays,
    log,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager

import torch

from slbd.common import ServerModelFitRes
from slbd.common.constants import CLIENT_LAST_LAYER_IDX_KEY, CLIENT_NUM_RECEIVED_LAYERS
from slbd.torch.model_utils import get_layer_idx
from slbd.algos.base_strategy import BaseStrategy
from slbd.server.client_layer_allocator import ClientLayerAllocator


class DynSLStrategy(BaseStrategy):

    def __init__(self, dynamic_layers_strategy: str, **kwargs):
        """Initialize

        Parameters
        ----------
        dynamic_layers_strategy : str
            What kind of strategy we use for dynamically changing the number of layers on the client
        """
        super().__init__(**kwargs)
        # cnt -- counter: send entire model to clients, just change the counter of how many layers client is training
        # afl -- use asynchronous FL-like aggregation. Send weight to client when increasing
        #        and client sends weights when decreasing
        assert dynamic_layers_strategy in {"cnt", "afl"}
        self._dynamic_layers_strategy = dynamic_layers_strategy

    def configure_fit(
        self,
        server_round: int,
        model_state_dict: OrderedDict,
        client_manager: ClientManager,
        client_layer_allocator: ClientLayerAllocator
    ) -> List[Tuple[ClientProxy, FitIns]]:
        client_manager.wait_for(self.min_available_clients)
        config = self.optimizer_config_generator.get_optimizer_config()
        if self.config_client_fit_fn:
            config |= self.config_client_fit_fn(server_round)

        clients = self.sample_epoch_clients(client_manager, True)
        if len(clients) == 0:
            return [], []
        num_initial_client_layers = client_layer_allocator \
            .initialize_epoch(clients, client_manager, True)

        if self._dynamic_layers_strategy == "afl":
            # when forking, we sent to the client only the initial weights it will train
            all_clients_fit_ins = self._setup_epoch_clients(
                server_round, clients, num_initial_client_layers, model_state_dict, True, config
            )
        elif self._dynamic_layers_strategy == "cnt":
            # we do not fork server models. For simplicity, we send all the possible
            # weights at the beginning of the epoch
            max_layers = [client_layer_allocator.get_max_client_layers(proxy.cid) for  proxy in clients]
            all_clients_fit_ins = self._setup_epoch_clients(
                server_round, clients, max_layers, model_state_dict, True, config
            )
            for nitl, (_, fit_ins) in zip(num_initial_client_layers, all_clients_fit_ins):
                fit_ins.config[CLIENT_NUM_RECEIVED_LAYERS] = fit_ins.config[CLIENT_LAST_LAYER_IDX_KEY]
                fit_ins.config[CLIENT_LAST_LAYER_IDX_KEY] = nitl

        if self.num_server_models == 1:
            min_layers = min(client_layer_allocator.get_min_client_layers(proxy.cid) for proxy in clients)
            all_server_models_fit_ins = self._setup_epoch_server_models(
                [("", min_layers, [proxy.cid for proxy in clients])],
                model_state_dict, True, config
            )
        elif self.num_server_models == -1:
            all_server_models_fit_ins = self._setup_epoch_server_models([
                (
                    proxy.cid,
                    client_layer_allocator.get_min_client_layers(proxy.cid),
                    [proxy.cid]
                ) for proxy in clients],
                model_state_dict, True, config
            )

        return all_clients_fit_ins, all_server_models_fit_ins

    def aggregate_fit(
        self,
        server_round: int,
        client_results: List[Tuple[ClientProxy, FitRes]],
        server_results: List[ServerModelFitRes],
        layer_allocator: ClientLayerAllocator,
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[OrderedDict, Dict[str, Scalar]]:
        _ = (server_round, )

        if self._dynamic_layers_strategy == "cnt":
            return super().aggregate_fit(
                server_round, client_results, server_results, layer_allocator, failures
            )

        # assert False, "Need to implement AFL aggregation strategy"
        return super().aggregate_fit(
            server_round, client_results, server_results, layer_allocator, failures
        )
