import copy
from collections import OrderedDict, defaultdict
from logging import ERROR
from typing import Optional, List, Tuple, Union, Dict, Callable

import wandb
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
from slbd.server.client_layer_allocator import ClientLayerAllocator
from slbd.torch.utils import OptimizerGenerator


def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]):
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {key: -1.0 for key in metrics[0][1].keys() if not key.startswith("_")}

    weighted_metrics = {}
    for key in metrics[0][1].keys():
        if key.startswith("_"):
            continue
        weighted_sum = sum(num_examples * metric[key] for num_examples, metric in metrics)
        weighted_metrics[key] = weighted_sum / total_examples
    if wandb.run is not None:
        wandb.log(weighted_metrics)
    return weighted_metrics


class BaseStrategy(Strategy):

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        *,
        model: torch.nn.Sequential,
        optimizer_config,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        config_client_fit_fn: Optional[Callable[[str], Dict[str, Scalar]]] = None,
        evaluate_metrics_aggregation_fn = weighted_average,
        fit_metrics_aggregation_fn = weighted_average,
        min_fit_clients: int = 1,
        min_evaluate_clients: int = 1,
        min_available_clients: int = 1,
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
        self.optimizer_config_generator = OptimizerGenerator(
            optimizer_cfg=optimizer_config["optimizer"],
            scheduler_cfg=optimizer_config["scheduler"],
        )

    def evaluate(
        self,
        server_round: int,
        model_state_dict: OrderedDict,
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        _ = (server_round, model_state_dict, )
        return None

    def _setup_epoch_clients(
        self,
        epoch_idx: int,
        clients: List[ClientProxy],
        num_client_layers: List[int],
        model_state_dict,
        is_training,
        base_config,
    ):
        """Generates the FitIns/EvaluateIns for the given clients, assigning to each the number
        of layers indicated in num_client_layers"""
        assert len(clients) == len(num_client_layers)

        config = copy.deepcopy(base_config)
        config["epoch_idx"] = epoch_idx

        client_ins_cls = FitIns if is_training else EvaluateIns
        # divide the parameters into client and server part
        all_clients_fit_ins = []
        for client_split_layer, client in zip(num_client_layers, clients):
            client_config = copy.deepcopy(config)
            client_parameters = ndarrays_to_parameters([
                model_state_dict[key].numpy()
                for layer_idx in range(client_split_layer)
                for key in get_model_layer_keys(self.model)[layer_idx]
            ])
            client_config[CLIENT_LAST_LAYER_IDX_KEY] = client_split_layer
            client_fit_ins = client_ins_cls(client_parameters, client_config)
            all_clients_fit_ins.append((client, client_fit_ins))
        return all_clients_fit_ins

    def _setup_epoch_server_models(
        self,
        sids_and_cids_list: List[Tuple[str, int, List[str]]],
        model_state_dict,
        is_training,
        config,
    ):
        # for now, init a single model with enough layers to be able to process any client request
        server_ins_cls = ServerModelFitIns if is_training else ServerModelEvaluateIns
        all_server_models_fit_ins = []
        self._cid_to_sid_mapping = {}
        for sid, first_server_layer, cids_list in sids_and_cids_list:

            server_model_state_dict = OrderedDict(
                (key, model_state_dict[key])
                for layer_idx in range(first_server_layer, get_num_layers(self.model))
                for key in get_model_layer_keys(self.model)[layer_idx]
            )
            server_fit_ins = server_ins_cls(
                server_model_state_dict, first_server_layer, config, sid=sid
            )
            all_server_models_fit_ins.append(server_fit_ins)
            for cid in cids_list:
                self._cid_to_sid_mapping[cid] = sid

        return all_server_models_fit_ins

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

        all_clients_fit_ins = self._setup_epoch_clients(
            server_round, clients, num_initial_client_layers, model_state_dict, True, config
        )
        all_server_models_fit_ins = self._setup_epoch_server_models(
            [("", min(num_initial_client_layers), [proxy.cid for proxy in clients])],
            model_state_dict, True, config
        )
        return all_clients_fit_ins, all_server_models_fit_ins

    def sample_epoch_clients(self, client_manager, is_training):
        if is_training:
            sample_size, min_num_clients = self.num_fit_clients(
                client_manager.num_available()
            )
            clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=min_num_clients
            )
        else:
            sample_size, min_num_clients = self.num_evaluation_clients(
                client_manager.num_available()
            )
            clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=min_num_clients
            )
        return clients

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
        if len(clients) == 0:
            return [], []

        num_initial_client_layers = client_layer_allocator.initialize_epoch(clients, client_manager, False)
        all_clients_eval_ins = self._setup_epoch_clients(
            server_round, clients, num_initial_client_layers, model_state_dict, False, config
        )
        all_server_models_eval_ins = self._setup_epoch_server_models(
            [("", min(num_initial_client_layers), [proxy.cid for proxy in clients])],
            model_state_dict, False, config
        )

        return all_clients_eval_ins, all_server_models_eval_ins

    def _aggregate_layer_parameters(self, trained_values: List[Tuple[torch.tensor, int]]):
        assert len(trained_values) > 0
        total_weights = sum(weight for _, weight in trained_values)
        return sum(tensor * (weight / total_weights) for tensor, weight in trained_values)

    def aggregate_fit(
        self,
        server_round: int,
        client_results: List[Tuple[ClientProxy, FitRes]],
        server_results: List[ServerModelFitRes],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[OrderedDict, Dict[str, Scalar]]:
        _ = (server_round, )
        self.optimizer_config_generator.end_epoch_callback()
        for failure in failures:
            log(ERROR, str(failure))

        # client parameters to state dict
        per_parameter_trained_values = defaultdict(list)
        for _, fit_res in client_results:
            client_finetuned_ndarrays = parameters_to_ndarrays(fit_res.parameters)
            processed_data_points_per_layer = {
                int(k[len("_proc"):]): v for k, v in fit_res.metrics.items()
                if k.startswith("_proc")
            }

            # convert parameter list to dictionary with keys as the state_dict()
            client_trained_statedict = {
                key: (torch.from_numpy(array), processed_data_points_per_layer[int(key.split(".")[0])])
                for key, array in zip(
                    self.model.state_dict().keys(),
                    client_finetuned_ndarrays
                )
            }

            # add the obtained values to the dictionary of all finetuned values we are building
            for key, tensor in client_trained_statedict.items():
                per_parameter_trained_values[key].append(tensor)

        for res in server_results:
            # method that employ plain aggregation must not have forking
            assert all(len(v) == 0 for v in res.forking_data.values())
            processed_data_points_per_layer = res.per_layer_training_examples
            for key, tensor in res.model_state_dict.items():
                per_parameter_trained_values[key].append(
                    (tensor.detach().cpu(), processed_data_points_per_layer[int(key.split(".")[0])])
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

        return loss_aggregated, aggregated_metrics

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def get_model_architecture(self):
        return copy.deepcopy(self.model)

    def route_client_request_to_server_model(self, client_id):
        _ = (client_id, )
        return ""
