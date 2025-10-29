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
