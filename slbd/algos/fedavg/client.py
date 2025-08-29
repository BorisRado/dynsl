from typing import Dict

from flwr.common import Config, Scalar

from slbd.algos.base_numpy_client import BaseNumpyClientAdapter


class FedAvgNumpyClientAdapter(BaseNumpyClientAdapter):

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        _ = (config)
        return {}
