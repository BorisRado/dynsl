from typing import Dict, Tuple

import torch

from flwr.common import (
    Config,
    Context,
    NDArrays,
    Scalar,
)
from flwr.client import NumPyClient as FlwrNumpyClient
from slbd.client.abstract_torch_client import AbstractTorchClient
from slbd.client.model_trainer import ModelTrainer
from slbd.common.constants import (
    CLIENT_LAST_LAYER_IDX_KEY,
    MIN_NUMBER_OF_CLIENT_LAYERS,
    CLIENT_DATASET_SIZE
)
from slbd.server.server_model.proxy.grpc_server_model_proxy import GrpcServerModelProxy


class BaseNumpyClientAdapter(FlwrNumpyClient):
    """Adapter class for Flower clients using NumPy in split learning context."""

    context: Context
    def __init__(
        self,
        torch_client: AbstractTorchClient,
        whole_model: torch.nn.Sequential,
        server_model_proxy: GrpcServerModelProxy,
    ):
        self.torch_client = torch_client
        self.whole_model = whole_model
        self.server_model_proxy = server_model_proxy

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        _ = (config, )
        return {
            # MIN_NUMBER_OF_CLIENT_LAYERS: self.config["min_layers"],
            CLIENT_DATASET_SIZE: self.torch_client.get_trainloader_size()
        }

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        # initialize local model
        last_client_layer_idx = config.pop(CLIENT_LAST_LAYER_IDX_KEY)
        optimizer_config = {k: v for k, v in config.items() if k.startswith("optim.")}

        # initialize the model trainer
        model_trainer = ModelTrainer(
            whole_model=self.whole_model,
            last_layer_idx=last_client_layer_idx,
            init_parameters=parameters,
            server_model_proxy=self.server_model_proxy,
            optimizer_config=optimizer_config
        )

        # call the model-owner training logic
        self.torch_client.fit(config, model_trainer)

        # extract trained parameters
        trained_parameters, processed_data_points_dict, metrics = model_trainer.get_fit_results()
        return trained_parameters, model_trainer._dataset_size, processed_data_points_dict | metrics

    def evaluate(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        last_client_layer_idx = config.pop(CLIENT_LAST_LAYER_IDX_KEY)

        # initialize the model trainer
        model_trainer = ModelTrainer(
            whole_model=self.whole_model,
            last_layer_idx=last_client_layer_idx,
            init_parameters=parameters,
            server_model_proxy=self.server_model_proxy,
            is_training=False
        )

        # call the model-owner training logic
        self.torch_client.evaluate(config, model_trainer)

        # extract trained parameters
        eval_metrics = model_trainer.get_evaluation_metrics()
        return eval_metrics["loss"], model_trainer._dataset_size, eval_metrics

    def get_parameters(self, config):
        raise Exception("Should not be called!!!")
