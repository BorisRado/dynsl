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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, parameters, config):
        trained_parameters, dataset_size, combined_metrics = super().fit(parameters, config)
        if len(trained_parameters) == 0:
            assert combined_metrics == {}
            assert dataset_size == 0
            return trained_parameters, dataset_size, combined_metrics

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

        max_client_layers = 100
        if device_type == "JetsonNano":
            # because of memory limits
            max_client_layers = self.find_layer_index(5_000_000) # train AT MOST 5M parameters
        print(f"Maximum number of layer {max_client_layers}")

        return {
            MIN_NUMBER_OF_CLIENT_LAYERS: 2,
            CLIENT_DATASET_SIZE: self.torch_client.get_trainloader_size(),
            CLIENT_DEVICE_TYPE: device_type,
            MAX_NUMBER_OF_CLIENT_LAYERS: max_client_layers,
            "node_name": node_name
        }

    def count_parameters(self, layer):
        return sum(p.numel() for p in layer.parameters())

    def find_layer_index(self, threshold=3_000_000):
        param_sum = 0
        for i, layer in enumerate(self.whole_model):
            param_sum += self.count_parameters(layer)
            if param_sum >= threshold:
                return i
        return len(self.whole_model)
