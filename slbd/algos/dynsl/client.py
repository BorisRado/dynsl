import os
from typing import Dict

import torch
from flwr.common import Config, Scalar

from slbd.algos.base_numpy_client import BaseNumpyClientAdapter
from slbd.common.constants import (
    MIN_NUMBER_OF_CLIENT_LAYERS,
    CLIENT_DATASET_SIZE,
    CLIENT_DEVICE_TYPE,
    MAX_NUMBER_OF_CLIENT_LAYERS
)

from exp.src.common.k8s import get_node_name


class DynSLClientAdapter(BaseNumpyClientAdapter):

    def __init__(self, end_epoch_weight_integration_strategy, **kwargs):
        super().__init__(**kwargs)
        assert end_epoch_weight_integration_strategy in {"immediate", "hierarchical"}
        self.end_epoch_weight_integration_strategy = end_epoch_weight_integration_strategy

    def fit(self, parameters, config):
        trained_parameters, dataset_size, combined_metrics = super().fit(parameters, config)
        if len(trained_parameters) == 0:
            assert combined_metrics == {}
            assert dataset_size == 0
            return trained_parameters, dataset_size, combined_metrics

        # immediately send the initial part of the model to the server for integration
        if self.end_epoch_weight_integration_strategy == "immediate":
            print("Immediately sending upper model parameters to the server")
            n_late_params = 0 # "late" parameters will be aggregated at the end of the epoch as usual
            for layer, _ in zip(self.whole_model, range(self.whole_model.meaningful_split_points[0])):
                n_late_params += len(list(layer.parameters()))

            trained_parameters, immediate_integration_parameters = \
                trained_parameters[:n_late_params], trained_parameters[n_late_params:]
            nl_end = len([k for k in combined_metrics if k.startswith("_proc")])
            extract_keys = [f"_proc{idx}" for idx in range(self.whole_model.meaningful_split_points[0], nl_end)]
            send_dps_dict = {k: combined_metrics.pop(k) for k in extract_keys}
            print("DP dict", send_dps_dict)
            print("num sending parameters", len(immediate_integration_parameters))
            print("Num keep layers", n_late_params)
            if len(send_dps_dict) > 0:
                self.server_model_proxy.integrate_weights(immediate_integration_parameters, send_dps_dict)
            else:
                print("Client was training the minimum number of layers -- skip sending any request")

        return trained_parameters, dataset_size, combined_metrics

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        _ = (config, )
        # no limits if client has GPU, otherwise ... how to determine max layers the client can afford? TODO
        print("Client GPU?", torch.cuda.is_available())
        device_type = os.environ.get("COLEXT_DEVICE_TYPE", "")
        try:
            node_name = get_node_name()
        except:
            node_name = str(torch.cuda.is_available())
        return {
            MIN_NUMBER_OF_CLIENT_LAYERS: 2,
            CLIENT_DATASET_SIZE: self.torch_client.get_trainloader_size(),
            CLIENT_DEVICE_TYPE: device_type,
            MAX_NUMBER_OF_CLIENT_LAYERS: 100,
            "node_name": node_name
        }
