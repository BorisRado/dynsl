import os
from typing import Dict

import torch
from flwr.common import Config, Scalar

from slbd.common.profiling import profile_cumulative_per_layer_time
from slbd.algos.base_numpy_client import BaseNumpyClientAdapter
from slbd.common.constants import (
    MIN_NUMBER_OF_CLIENT_LAYERS,
    CLIENT_DATASET_SIZE,
    CLIENT_DEVICE_TYPE,
    MAX_NUMBER_OF_CLIENT_LAYERS
)

from exp.src.common.k8s import get_node_name


class DynSLClientAdapterWProfiling(BaseNumpyClientAdapter):

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        _ = (config, )
        # no limits if client has GPU, otherwise ... how to determine max layers the client can afford? TODO
        print("Client GPU?", torch.cuda.is_available())
        device_type = os.environ.get("COLEXT_DEVICE_TYPE", "")
        try:
            node_name = get_node_name()
        except:
            node_name = str(torch.cuda.is_available())

        per_layer_complexity = profile_cumulative_per_layer_time(
            model=self.whole_model,
            optimizer="adam",
            batch_size=self.torch_client.batch_size,
            iterations=300 if torch.cuda.is_available() else 100,
            dry_runs=50 if torch.cuda.is_available() else 10,
        )
        # convert list to a dictionary as flower does not allow sending lists
        per_layer_complexity = {f"__lc{idx}": per_layer_complexity[idx] for idx in range(len(per_layer_complexity))}
        return {
            MIN_NUMBER_OF_CLIENT_LAYERS: 2,
            CLIENT_DATASET_SIZE: self.torch_client.get_trainloader_size(),
            CLIENT_DEVICE_TYPE: device_type,
            MAX_NUMBER_OF_CLIENT_LAYERS: 100,
            "node_name": node_name,
            **per_layer_complexity,
        }
