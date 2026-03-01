from collections import OrderedDict, defaultdict
from typing import List, Tuple, Union, Dict

import wandb
from flwr.common import (
    FitRes,
    FitIns,
    Scalar,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager

import torch

from slbd.common import ServerModelFitRes
from slbd.torch.model_utils import get_layer_idx
from slbd.algos.base_strategy import BaseStrategy
from slbd.server.client_layer_allocator import ClientLayerAllocator


class DynSLStrategy(BaseStrategy):

    def __init__(self, end_epoch_weight_integration_strategy: str, server_optim_state_handling, **kwargs):
        """Initialize

        Parameters
        ----------
        end_epoch_weight_integration_strategy : str
            Whether we integrate weights immediately or not
        """
        super().__init__(**kwargs)
        if self.num_server_models == -1:
            assert end_epoch_weight_integration_strategy == "hierarchical"

        assert end_epoch_weight_integration_strategy in {"hierarchical", "immediate"}
        if end_epoch_weight_integration_strategy == "hierarchical":
            assert server_optim_state_handling is None
        else:
            assert server_optim_state_handling in {"reset_optim_state", "keep_optim_state"}
        self._server_optim_state_handling = server_optim_state_handling
        self._end_epoch_weight_integration_strategy = end_epoch_weight_integration_strategy

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
            if isinstance(self.config_client_fit_fn, dict):
                config |= self.config_client_fit_fn
            else:
                config |= self.config_client_fit_fn(server_round)

        clients = self.sample_epoch_clients(client_manager, True)
        if len(clients) == 0:
            return [], []
        num_initial_client_layers = client_layer_allocator \
            .initialize_epoch(clients, client_manager, True)

        all_clients_fit_ins = self._setup_epoch_clients(
            server_round, clients, num_initial_client_layers, model_state_dict, True, config
        )

        if self.num_server_models == 1:
            min_layers = self.model.meaningful_split_points[0]
            server_config = config.copy()
            server_config["weight_integration_strategy"] = self._server_optim_state_handling
            print(f"Assigning min layers: {min_layers}")
            all_server_models_fit_ins = self._setup_epoch_server_models(
                [("", min_layers, [proxy.cid for proxy in clients])],
                model_state_dict, True, server_config
            )
        if self.num_server_models == 0:
            raise Exception("Cannot use model per split point --- would it make sense to integrate?")
        elif self.num_server_models == -1:
            assert self._end_epoch_weight_integration_strategy == "hierarchical"
            all_server_models_fit_ins = self._setup_epoch_server_models([
                (
                    proxy.cid,
                    client_layer_allocator.get_min_client_layers(proxy.cid),
                    [proxy.cid]
                ) for proxy in clients],
                model_state_dict, True, config
            )

        return all_clients_fit_ins, all_server_models_fit_ins

    def _aggregate_hierarchical(self, key, afl_data, all_client_statedicts, server_state_dict, server_processed_dp_per_layer):
        layer_idx = get_layer_idx(key)
        clients_num_dp_when_forking = {
            cid: afl_data[cid][layer_idx]
            for cid in all_client_statedicts
            if cid in afl_data and layer_idx in afl_data[cid]
        }
        if key not in server_state_dict:
            # layer was not trained on the server, use averaging of client weights
            tot_num_dp = sum([v[key][1] for v in all_client_statedicts.values()])
            agg_weight = self._aggregate_layer_parameters([v[key] for v in all_client_statedicts.values()])
            return agg_weight, tot_num_dp

        iter_weight = server_state_dict[key]
        previous_fork = server_processed_dp_per_layer[layer_idx]

        for cid, num_cfdp in sorted(clients_num_dp_when_forking.items(), key=lambda x: x[1], reverse=True):
            client_w, client_dp = all_client_statedicts[cid][key]
            num_processed_on_server = previous_fork - num_cfdp
            num_processed_on_client = client_dp

            w1, w2 = num_processed_on_server, num_processed_on_client
            w1, w2 = w1 / (w1 + w2), w2 / (w1 + w2)
            assert client_w.shape == iter_weight.shape
            client_w = client_w.to(iter_weight.device)
            iter_weight = iter_weight.mul_(w1).add_(w2 * client_w)
            previous_fork = previous_fork + num_processed_on_client
        return iter_weight.cpu(), previous_fork

    def aggregate_fit(
        self,
        server_round: int,
        client_results: List[Tuple[ClientProxy, FitRes]],
        server_results: List[ServerModelFitRes],
        layer_allocator: ClientLayerAllocator,
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[OrderedDict, Dict[str, Scalar]]:
        _ = (server_round, )
        if all([server_res.afl_data == {} for server_res in server_results]):
            print("Aggregating with plain aggragation -- ok if w_profiling or immediate aggregation")
            return super().aggregate_fit(
                server_round, client_results, server_results, layer_allocator, failures
            )

        self.optimizer_config_generator.end_epoch_callback()

        # Verify all server_res have afl_data attribute with same value
        afl_data_values = [server_res.afl_data for server_res in server_results]
        assert len(afl_data_values) == len(server_results), "Not all server_results have afl_data attribute"
        assert all(afl_data == afl_data_values[0] for afl_data in afl_data_values), "Not all afl_data values are the same"

        afl_data = {k: v for k, v in afl_data_values[0].items() if len(v) > 0}
        per_parameter_trained_values = defaultdict(list)
        for server_res in server_results:
            assert hasattr(server_res, "afl_data")
            client_res = [cr for cr in client_results if self._cid_to_sid_mapping[cr[0].cid] == server_res.sid]

            all_client_statedicts = {}
            for proxy, cr in client_res:
                client_finetuned_ndarrays = parameters_to_ndarrays(cr.parameters)
                processed_data_points_per_layer = {
                    int(k[len("_proc"):]): int(v) for k, v in cr.metrics.items()
                    if k.startswith("_proc")
                }
                all_client_statedicts[proxy.cid] = {
                    key: (torch.from_numpy(array), processed_data_points_per_layer[get_layer_idx(key)])
                    for key, array in zip(
                        self.model.state_dict().keys(),
                        client_finetuned_ndarrays
                    )
                }
            server_processed_data_points_per_layer = server_res.per_layer_training_examples
            server_state_dict = server_res.model_state_dict
            for key in self.model.state_dict():
                per_parameter_trained_values[key].append(
                    self._aggregate_hierarchical(key, afl_data, all_client_statedicts, server_state_dict, server_processed_data_points_per_layer)
                )

        # ultimately, all the layers must have been trained on the same number of data points ==>
        assert len(set([sum([int(a[1]) for a in v]) for v in per_parameter_trained_values.values()])) == 1
        out_state_dict = OrderedDict([
            (key, self._aggregate_layer_parameters(per_parameter_trained_values[key]))
            for key in self.model.state_dict().keys()
        ])
        # Remove entries with None values
        out_state_dict = OrderedDict([(k, v) for k, v in out_state_dict.items() if v is not None])

        aggregated_metrics = {}
        if self.fit_metrics_aggregation_fn is not None:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in client_results]
            cids = [proxy.cid for proxy, _ in client_results]
            aggregated_metrics |= self.fit_metrics_aggregation_fn(fit_metrics, cids)

        if wandb.run is not None:
            wandb.log(
                aggregated_metrics |
                layer_allocator.collect_epoch_metrics() |
                {
                    "fit_round_time": self._fit_epoch_duration
                    if hasattr(self, "_fit_epoch_duration") else -1.
                }
            )
        return out_state_dict, aggregated_metrics
