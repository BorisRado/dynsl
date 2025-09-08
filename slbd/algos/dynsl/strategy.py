from collections import OrderedDict
from logging import ERROR
from typing import List, Tuple, Union, Dict

from flwr.common import (
    FitRes,
    Scalar,
    parameters_to_ndarrays,
    log,
)
from flwr.server.client_proxy import ClientProxy

import torch

from slbd.common import ServerModelFitRes
from slbd.algos.base_strategy import BaseStrategy


class DynSLStrategy(BaseStrategy):

    def _aggregate_layer_parameters_with_forking(self, trained_values, forking_data, key):
        if "" not in trained_values:
            # the layer was not trained on the server
            # simply average the client layers with corresponding weights
            return self._aggregate_layer_parameters(trained_values.values())
        elif "" in trained_values and len(trained_values) == 1:
            # the layer was only trained on the server -- we just return the server value
            return trained_values[""][0]

        iter_layer, s_prime = trained_values[""]

        for forking_clients, forking_num_dp in reversed(forking_data):
            iter_layer = self._aggregate_layer_parameters([
                trained_values[cid] for cid in forking_clients
            ] + [(iter_layer, s_prime - forking_num_dp)])
            s_prime += sum([trained_values[cid][1] for cid in forking_clients])

            # assert iter_layer.allclose(trained_values[forking_clients[0]][0])

        return iter_layer

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
        # for every parameter, we have a dictionary with client-id [key] and a tuple
        # containing the finetuned value along with the number of trained data points [value]
        # the dictionary also contains the values for the SINGLE server model under the key `""`
        per_parameter_trained_values = {k: {} for k in self.model.state_dict()}

        for client_proxy, fit_res in client_results:
            cid = client_proxy.cid

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
            for key, tensor_dp_tuple in client_trained_statedict.items():
                per_parameter_trained_values[key][cid] = tensor_dp_tuple

        assert len(server_results) == 1
        server_res = server_results[0]
        forking_data = server_res.forking_data

        processed_data_points_per_layer = server_res.per_layer_training_examples

        for key, tensor in server_res.model_state_dict.items():
            per_parameter_trained_values[key][""] = \
                (tensor.detach().cpu(), processed_data_points_per_layer[int(key.split(".")[0])])

        out_state_dict = OrderedDict([
            (
                key,
                self._aggregate_layer_parameters_with_forking(
                    per_parameter_trained_values[key],
                    forking_data[int(key.split(".")[0])],
                    key
                )
            )
            for key in self.model.state_dict().keys()
        ])

        aggregated_metrics = {}
        if self.fit_metrics_aggregation_fn is not None:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in client_results]
            aggregated_metrics = self.fit_metrics_aggregation_fn(fit_metrics)
        return out_state_dict, aggregated_metrics
