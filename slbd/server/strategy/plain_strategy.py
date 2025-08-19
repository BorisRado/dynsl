import uuid
import copy
from collections import OrderedDict, defaultdict
from logging import ERROR
from typing import Optional, List, Tuple, Union, Dict, Callable

from flwr.common import (
    FitIns,
    EvaluateIns,
    FitRes,
    Scalar,
    EvaluateRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    log,
)
from flwr.server.client_manager import ClientManager
from flwr.server.strategy.aggregate import weighted_loss_avg
from flwr.server.client_proxy import ClientProxy

import torch

from slbd.common.constants import CLIENT_LAST_LAYER_IDX_KEY
from slbd.common import (
    ServerModelEvaluateIns,
    ServerModelFitIns,
    ServerModelFitRes,
)
from slbd.server.strategy import Strategy
from slbd.torch.model_utils import get_model_layer_keys, get_num_layers
from slbd.server.server_model.utils import ClientRequestGroup
from slbd.server.layer_allocator import LayerAllocator


class PlainSlStrategy(Strategy):

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        *,
        model: torch.nn.Sequential,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        config_client_fit_fn: Optional[Callable[[str], Dict[str, Scalar]]] = None,
        evaluate_metrics_aggregation_fn = None,
        fit_metrics_aggregation_fn = None,
        min_fit_clients: int = 1,
        min_evaluate_clients: int = 1,
        min_available_clients: int = 1,
        common_server_model: bool = True,
        process_clients_as_batch: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.config_client_fit_fn = config_client_fit_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self._cid_to_sid_mapping = {}
        self.common_server_model = common_server_model
        self.process_clients_as_batch = process_clients_as_batch
        self._round_active_clients = []
        self.requests_state: Dict[Tuple[str, str], Dict[str, ClientRequestGroup]] = {}

    def evaluate(
        self,
        server_round: int,
        model_state_dict: OrderedDict,
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        _ = (server_round, model_state_dict, )
        return None

    def _setup_epoch(
        self,
        clients,
        model_state_dict,
        client_layer_allocator,
        is_training,
        base_config
    ):
        config = copy.deepcopy(base_config)
        client_layer_allocator.initialize_epoch(clients, is_training=is_training)
        self._round_active_clients = [client.cid for client in clients]

        client_ins_cls = FitIns if is_training else EvaluateIns
        server_ins_cls = ServerModelFitIns if is_training else ServerModelEvaluateIns
        # divide the parameters into client and server part
        all_clients_fit_ins = []
        client_split_layers = []
        for client in clients:
            client_split_layer = client_layer_allocator.get_num_init_client_layers(client)
            client_split_layers.append(client_split_layer)
            client_parameters = ndarrays_to_parameters([
                model_state_dict[key].numpy()
                for layer_idx in range(client_split_layer)
                for key in get_model_layer_keys(self.model)[layer_idx]
            ])
            config[CLIENT_LAST_LAYER_IDX_KEY] = client_split_layer
            client_fit_ins = client_ins_cls(client_parameters, config)
            all_clients_fit_ins.append((client, client_fit_ins))

        # for now, init a single model with enough layers to be able to process any client request
        min_client_layers = min(client_split_layers)
        server_model_state_dict = OrderedDict(
            (key, model_state_dict[key])
            for layer_idx in range(min_client_layers, get_num_layers(self.model))
            for key in get_model_layer_keys(self.model)[layer_idx]
        )
        server_fit_ins = server_ins_cls(server_model_state_dict, min_client_layers, config, sid="")

        # Return client/config pairs
        all_server_models_fit_ins = [server_fit_ins,]
        self._cid_to_sid_mapping = {client.cid: "" for client in clients}
        return all_clients_fit_ins, all_server_models_fit_ins


    def configure_fit(
        self,
        server_round: int,
        model_state_dict: OrderedDict,
        client_manager: ClientManager,
        client_layer_allocator: LayerAllocator
    ) -> List[Tuple[ClientProxy, FitIns]]:
        config = {}
        if self.config_client_fit_fn:
            config = self.config_client_fit_fn(server_round)

        # uninformed and unbiased sampling of clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        all_clients_fit_ins, all_server_models_fit_ins = self._setup_epoch(
            clients, model_state_dict, client_layer_allocator, True, config
        )
        return all_clients_fit_ins, all_server_models_fit_ins

    def configure_evaluate(
        self,
        server_round: int,
        model_state_dict: OrderedDict,
        client_manager: ClientManager,
        client_layer_allocator: LayerAllocator,
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        _ = (server_round, )
        config = {}
        # if self.config_client_fit_fn:
        #     config = self.config_client_fit_fn(server_round)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        all_clients_eval_ins, all_server_models_eval_ins = self._setup_epoch(
            clients, model_state_dict, client_layer_allocator, False, config
        )

        return all_clients_eval_ins, all_server_models_eval_ins

    def _aggregate_layer_parameters(self, trained_values: List[Tuple[torch.tensor, int]]):
        assert len(trained_values) > 0
        assert sum(weight for _, weight in trained_values) == 1000
        total_weights = sum(weight for _, weight in trained_values)
        return sum(tensor * (weight / total_weights) for tensor, weight in trained_values)

    def aggregate_fit(
        self,
        server_round: int,
        client_results: List[Tuple[ClientProxy, FitRes]],
        server_results: List[ServerModelFitRes],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[OrderedDict, Dict[str, Scalar]]:
        _ = (server_round, failures,)
        self.requests_state = {}
        self._round_active_clients = []
        for failure in failures:
            log(ERROR, str(failure))

        # client parameters to state dict
        per_parameter_trained_values = defaultdict(list)
        for _, fit_res in client_results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            trained_per_layer = {
                int(k[len("_proc"):]): v for k, v in fit_res.metrics.items()
                if k.startswith("_proc")
            }

            client_trained_statedict = {
                key: (torch.from_numpy(array), trained_per_layer[int(key.split(".")[0])])
                for key, array in zip(
                    self.model.state_dict().keys(),
                    ndarrays
                )
            }
            for key, tensor in client_trained_statedict.items():
                per_parameter_trained_values[key].append(tensor)

        for res in server_results:
            trained_per_layer = res.per_layer_training_examples
            for key, tensor in res.model_state_dict.items():
                per_parameter_trained_values[key].append(
                    (tensor.detach().cpu(), trained_per_layer[int(key.split(".")[0])])
                )
        out_state_dict = OrderedDict([
            (key, self._aggregate_layer_parameters(per_parameter_trained_values[key]))
            for key in self.model.state_dict().keys()
        ])

        aggregated_metrics = {}
        if self.fit_metrics_aggregation_fn is not None:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in client_results]
            aggregated_metrics = self.fit_metrics_aggregation_fn(fit_metrics)
        return out_state_dict, aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]
    ):
        self.requests_state = {}
        _ = (server_round, failures, )
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        aggregated_metrics = {}
        if self.evaluate_metrics_aggregation_fn is not None:
            evaluation_metrics = [(res.num_examples, res.metrics) for _, res in results]
            aggregated_metrics = self.evaluate_metrics_aggregation_fn(evaluation_metrics)

        self._round_active_clients = []
        return loss_aggregated, aggregated_metrics

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def route_client_request(
        self,
        client_id: str,
    ) -> Tuple[ClientRequestGroup, Optional[Callable]]:
        server_id = self._cid_to_sid_mapping[client_id]
        if server_id not in self.requests_state:
            self.requests_state[server_id] = {}
        for reqg in self.requests_state[server_id].values():
            if not reqg.is_ready():
                return reqg, lambda: None

        # all requests are ready. Create a new one
        request_group = ClientRequestGroup(server_id)
        request_id = str(uuid.uuid4())[:8]
        self.requests_state[server_id][request_id] = request_group
        callback_fn = lambda: self.requests_state[server_id].pop(request_id)
        return request_group, callback_fn

    def mark_ready_requests(self):
        for reqg_dict in self.requests_state.values():
            for reqg in reqg_dict.values():
                if (
                    not self.process_clients_as_batch or # process batch individually
                    reqg.get_num_batches() == len(self._round_active_clients)
                ):
                    # each client is processed individually
                    reqg.mark_as_ready()

    def mark_client_as_done(self, client_id):
        self._round_active_clients.remove(client_id)
        self.mark_ready_requests()

    def get_model_architecture(self):
        return copy.deepcopy(self.model)
