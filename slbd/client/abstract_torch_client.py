from abc import ABC, abstractmethod

from flwr.common import Config

from slbd.client.model_trainer import ModelTrainer


class AbstractTorchClient(ABC):

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
