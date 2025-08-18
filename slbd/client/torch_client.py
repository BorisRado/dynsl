
from typing import Tuple, Dict

from flwr.common import Config, Scalar

from slbd.client.model_trainer import ModelTrainer


class TorchClient:

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        _ = (config,)
        return {}

    def fit(self, config: Config, model_trainer: ModelTrainer) -> Tuple[int, Dict[str, Scalar]]:
        _ = (config, model_trainer, )
        return 0, {}

    def evaluate(self, config: Config) -> Tuple[float, int, Dict[str, Scalar]]:
        _ = (config,)
        pass
