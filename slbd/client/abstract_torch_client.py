from typing import Dict
from abc import ABC, abstractmethod

from flwr.common import Config, Scalar

from slbd.client.model_trainer import ModelTrainer


class AbstractTorchClient(ABC):

    @abstractmethod
    def get_properties(self) -> Dict[str, Scalar]:
        """Called once the client is sampled for the very first time. The client is expected
        to return its basic properties, such as the minimum number of layers it requires
        to be trained locally. If no value is provided, the server uses a default value

        Returns
        -------
        Dict[str, Scalar]
           Dictionary with the client configuration. Keys are specified in
           `slbd.common.constants.py`. Eventually, client configuration should be
           read from a yaml file and not be overridden by the model owner
        """
        return {}

    @abstractmethod
    def fit(self, config: Config, model_trainer: ModelTrainer) -> None:
        """Method triggered when a client is sampled for training. The client is expected
        to train its model on its local dataset

        Parameters
        ----------
        config : Config
            Configuration sent by the server. It may include parameters regarding data loading
            (e.g., which image transformation to use), the number of local training epochs before
            the client sends the updated weights to the server, etc. Note, that many hyperparameters
            such as optimizer configuration are handled by the framework and not hence are not
            visible by the training logic
        model_trainer : ModelTrainer
            Component that handles the forward and backward pass as well as the optimizer step.
            Inside the client training logic, you only need to provide the input data
            (e.g., images) and the output data (e.g, target labels). All the rest is handled
            by the framework.
        """
        _ = (config, model_trainer, )

    @abstractmethod
    def evaluate(self, config: Config, model_trainer: ModelTrainer) -> None:
        """Method triggered when a client is sampled for evaluation. The client is expected
        to evaluate the global model on its local dataset

        Parameters
        ----------
        config : Config
            Configuration sent by the server
        model_trainer : ModelTrainer
            Component that handles the forward pass and metric bookkeeping. Inside the client
            training logic, you only need to provide the input data (e.g., images) and the
            output data (e.g, target labels). All the rest is handled by the framework.
        """
        _ = (config, model_trainer)
