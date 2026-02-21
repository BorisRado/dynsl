import os
import time
import traceback
from typing import Dict, Tuple, Optional

import torch
from hydra.utils import instantiate

from flwr.common import (
    Config,
    Context,
    NDArrays,
    Scalar,
)
from slbd.client.abstract_torch_client import AbstractTorchClient
from slbd.client.model_trainer import ModelTrainer
from slbd.torch.model_utils import param_list_to_state_dict
from slbd.common.constants import (
    CLIENT_DEVICE_TYPE,
    CLIENT_NUM_RECEIVED_LAYERS,
)
from slbd.server.server_model.proxy.grpc_server_model_proxy import GrpcServerModelProxy
from slbd.torch.sequential_model import FlexibleSequential

from slbd.algos.base_numpy_client import BaseNumpyClientAdapter
from slbd.torch.utils import create_classification_head


class SFSLClientAdapter(BaseNumpyClientAdapter):
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
        clf_head_dimensions=[256,],
    ):
        self.torch_client = torch_client
        self.whole_model = whole_model
        self.server_model_proxy = server_model_proxy
        self.sending_transform_fn = instantiate(data_manipulation_config.sending_fn, _partial_=True)
        self.receiving_transform_fn = instantiate(data_manipulation_config.receiving_fn, _partial_=True)
        self.dynamic_layers_strategy = dynamic_layers_strategy
        self.exchange_optimizer_states = exchange_optimizer_states
        self.clf_head_dimensions = clf_head_dimensions

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        return {
            CLIENT_DEVICE_TYPE: os.environ.get("COLEXT_DEVICE_TYPE", ""),
        }

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        # initialize local model
        num_received_layers = config.pop(CLIENT_NUM_RECEIVED_LAYERS)
        optimizer_config = {k: v for k, v in config.items() if k.startswith("optim.")}
        num_clf_head_params = (len(self.clf_head_dimensions) + 1) * 2
        encoder_parameters, clf_head_parameters = parameters[:-num_clf_head_params], parameters[-num_clf_head_params:]

        # initialize the model trainer
        model_trainer = ModelTrainer(
            whole_model=self.whole_model,
            num_received_layers=num_received_layers,
            init_parameters=encoder_parameters,
            server_model_proxy=self.server_model_proxy,
            optimizer_config=optimizer_config,
            sending_transform_fn=self.sending_transform_fn,
            receiving_transform_fn=self.receiving_transform_fn,
            is_training=True,
            dynamic_layers_strategy=self.dynamic_layers_strategy,
            exchange_optimizer_states=self.exchange_optimizer_states,
            separate_client_clf_head=None,
        )
        assert parameters[-1].ndim == 1
        clf_head = create_classification_head(
            encoder_model=model_trainer._model,
            num_classes=parameters[-1].shape[0],
            fc_dims=self.clf_head_dimensions
        )
        # load state dict onto the classification head
        state_dict = param_list_to_state_dict(clf_head_parameters, clf_head)
        clf_head.load_state_dict(state_dict, strict=True)
        if torch.cuda.is_available():
            clf_head.cuda()

        model_trainer._separate_client_clf_head = clf_head

        # call the model-owner training logic, and measure the time it takes
        start_time = time.time()
        try:
            self.torch_client.fit(config, model_trainer)
        except Exception as e:
            print("Exception!!")
            traceback.print_exc()
            self.server_model_proxy.close_stream(error=True)
        runtime = time.time() - start_time

        # extract trained parameters also for classification head
        trained_parameters, processed_data_points_dict, metrics = model_trainer.get_fit_results()
        for _, param in clf_head.state_dict().items():
            trained_parameters.append(param.detach().cpu().numpy())
        combined_metrics = processed_data_points_dict | metrics | {"local_train_time": runtime}
        combined_metrics["train/client_loss"] = combined_metrics.pop("train/loss")
        dataset_size = model_trainer._dataset_size
        del model_trainer
        return trained_parameters, dataset_size, combined_metrics
