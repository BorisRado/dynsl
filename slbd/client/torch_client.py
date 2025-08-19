from abc import ABC, abstractmethod
from typing import Dict

from flwr.common import Config, Scalar

from slbd.client.model_trainer import ModelTrainer


class AbstractTorchClient(ABC):

    @abstractmethod
    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        _ = (config,)
        return {}

    @abstractmethod
    def fit(self, config: Config, model_trainer: ModelTrainer) -> None:
        _ = (config, model_trainer, )

    @abstractmethod
    def evaluate(self, config: Config, model_trainer: ModelTrainer) -> None:
        _ = (config, model_trainer)
