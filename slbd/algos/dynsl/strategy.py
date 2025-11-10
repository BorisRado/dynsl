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

    def __init__(self, fork_server_model: bool, **kwargs):
        """Initialize

        Parameters
        ----------
        fork_server_model : bool
            If true, the latest weights of the server model will be sent to the client whenever increasing the number of layers
        """
        super().__init__(**kwargs)
        self._fork_server_model = fork_server_model

    def _aggregate_layer_parameters_with_forking(self, all_trained_values, forking_data, key):
        trained_values = all_trained_values[key]
        if "" not in trained_values or ("" in trained_values and len(trained_values) == 1):
            # the layer was not trained on the server
            # simply average the client layers with corresponding weights
            tmp = self._aggregate_layer_parameters(trained_values.values())
            del all_trained_values[key]
            return tmp

        # `s_prime` states on how many data points the next model (i.e., the model
        # one layer lower in the tree) has been trained on
        iter_layer, s_prime = trained_values.pop("")

        for forking_clients, forking_num_dp in reversed(forking_data):
            # how many new data points clients that have forked in this iterations contribute
            increase = sum([trained_values[cid][1] for cid in forking_clients])
            iter_layer = self._aggregate_layer_parameters([
                trained_values.pop(cid) for cid in forking_clients
            ] + [(iter_layer, s_prime - forking_num_dp)])
            s_prime += increase

        return iter_layer

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

        if self._fork_server_model:
            # when forking, we sent to the client only the initial weights it will train
            all_clients_fit_ins = self._setup_epoch_clients(
                server_round, clients, num_initial_client_layers, model_state_dict, True, config
            )
        else:
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

        if not self._fork_server_model:
            # if there is no forking, we can aggregate the usual way
            for s_res in server_results:
                assert all(len(v) == 0 for v in s_res.forking_data.values())
            return super().aggregate_fit(
                server_round, client_results, server_results, layer_allocator, failures
            )

        assert False, "Needs to be fixed"
        self.optimizer_config_generator.end_epoch_callback()
        for failure in failures:
            log(ERROR, str(failure))

        # finetuned parameters to state dict
        # for every model parameter (key), we have a list with tuples, containing the finetuned
        # value along with the number of data points. The parameters include both the client- and
        # the server-side trained models
        per_parameter_trained_values = {k: defaultdict(tuple) for k in self.model.state_dict()}

        for client_proxy, fit_res in client_results:

            client_finetuned_ndarrays = parameters_to_ndarrays(fit_res.parameters)
            processed_data_points_per_layer = {
                int(k[len("_proc"):]): v for k, v in fit_res.metrics.items()
                if k.startswith("_proc")
            }

            for key, array in zip(self.model.state_dict().keys(), client_finetuned_ndarrays):
                client_dp = processed_data_points_per_layer[get_layer_idx(key)]
                if client_dp == 0:
                    break
                per_parameter_trained_values[key][client_proxy] = (torch.from_numpy(array), client_dp)

        for server_res in server_results:

            forking_data = server_res.forking_data
            processed_data_points_per_layer = server_res.per_layer_training_examples

            for key, tensor in server_res.model_state_dict.items():
                per_parameter_trained_values[key][""] = \
                    (tensor.detach().cpu(), processed_data_points_per_layer[get_layer_idx(key)])

            out_state_dict = OrderedDict([
                (
                    key,
                    self._aggregate_layer_parameters_with_forking(
                        per_parameter_trained_values,
                        forking_data[get_layer_idx(key)],
                        key
                    )
                )
                for key in self.model.state_dict().keys()
            ])
        assert all(len(v) == 0 for v in per_parameter_trained_values.values())

        aggregated_metrics = {}
        if self.fit_metrics_aggregation_fn is not None:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in client_results]
            cids = [proxy.cid for proxy, _ in client_results]
            aggregated_metrics |= self.fit_metrics_aggregation_fn(fit_metrics, cids)

        if wandb.run is not None:
            wandb.log(
                aggregated_metrics |
                layer_allocator.collect_epoch_metrics() |
                {"fit_round_time": self._fit_epoch_duration}
            )
        return out_state_dict, aggregated_metrics
