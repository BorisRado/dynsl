import os
import time
import traceback
from typing import Dict, Tuple, Optional

from hydra.utils import instantiate

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
    CLIENT_DATASET_SIZE,
    CLIENT_DEVICE_TYPE,
    CLIENT_NUM_RECEIVED_LAYERS,
)
from slbd.server.server_model.proxy.grpc_server_model_proxy import GrpcServerModelProxy
from slbd.torch.sequential_model import FlexibleSequential


class BaseNumpyClientAdapter(FlwrNumpyClient):
    """Adapter class for Flower clients using NumPy in split learning context."""

    context: Context
    def __init__(
        self,
        torch_client: AbstractTorchClient,
        whole_model: FlexibleSequential,
        server_model_proxy: GrpcServerModelProxy,
        data_manipulation_config,
        dynamic_layers_strategy: Optional[str]=None,
        exchange_optimizer_states: Optional[bool]=False,
    ):
        self.torch_client = torch_client
        self.whole_model = whole_model
        self.server_model_proxy = server_model_proxy
        self.sending_transform_fn = instantiate(data_manipulation_config.sending_fn, _partial_=True)
        self.receiving_transform_fn = instantiate(data_manipulation_config.receiving_fn, _partial_=True)
        self.dynamic_layers_strategy = dynamic_layers_strategy
        self.exchange_optimizer_states = exchange_optimizer_states
        print("Initialized client adapter")
        print(f"\t\tDynamic layers strategy: {self.dynamic_layers_strategy}")
        print(f"\t\tExchange optimizer states: {self.exchange_optimizer_states}")

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        _ = (config, )
        return {
            MIN_NUMBER_OF_CLIENT_LAYERS: 2,
            CLIENT_DATASET_SIZE: self.torch_client.get_trainloader_size(),
            CLIENT_DEVICE_TYPE: os.environ.get("COLEXT_DEVICE_TYPE", ""),
        }

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        # initialize local model
        num_received_layers = config.pop(CLIENT_NUM_RECEIVED_LAYERS)
        optimizer_config = {k: v for k, v in config.items() if k.startswith("optim.")}

        # initialize the model trainer
        model_trainer = ModelTrainer(
            whole_model=self.whole_model,
            num_received_layers=num_received_layers,
            init_parameters=parameters,
            server_model_proxy=self.server_model_proxy,
            optimizer_config=optimizer_config,
            sending_transform_fn=self.sending_transform_fn,
            receiving_transform_fn=self.receiving_transform_fn,
            is_training=True,
            dynamic_layers_strategy=self.dynamic_layers_strategy,
            exchange_optimizer_states=self.exchange_optimizer_states,
        )

        # call the model-owner training logic, and measure the time it takes
        start_time = time.time()
        trained_parameters, dataset_size, combined_metrics = [], 0, {}
        try:
            self.torch_client.fit(config, model_trainer)
            trained_parameters, processed_data_points_dict, metrics = model_trainer.get_fit_results()
            runtime = time.time() - start_time
            combined_metrics = processed_data_points_dict | metrics | {"local_train_time": runtime}
            dataset_size = model_trainer._dataset_size
        except Exception as e:
            print("Exception!!")
            traceback.print_exc()
            self.server_model_proxy.close_stream(error=True)

        # extract trained parameters
        del model_trainer
        return trained_parameters, dataset_size, combined_metrics

    def evaluate(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        last_client_layer_idx = config.pop(CLIENT_LAST_LAYER_IDX_KEY)

        # initialize the model trainer
        model_trainer = ModelTrainer(
            whole_model=self.whole_model,
            num_training_layers=last_client_layer_idx,
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
