from logging import DEBUG
from collections import defaultdict, OrderedDict
from typing import List, Optional

import torch
from flwr.common import log

from slbd.torch.model_utils import (
    extract_submodel,
    param_list_to_state_dict,
    attach_layers,
    extract_submodel_weights,
    get_num_layers
)

import torch
import torch.nn as nn
from flwr.common import log

from slbd.common.utils import measure_time
from slbd.torch.sequential_model import FlexibleSequential
from slbd.torch.utils import init_optimizer
from slbd.torch.model_utils import (
    extract_submodel,
    param_list_to_state_dict,
    attach_layers,
    get_num_layers,
    extract_submodel_state_dict,
)


class ModelTrainer:
    """
    Handles client-side model training and communication with server.

    This class manages the client-side portion of the split model,
    coordinates forward/backward passes with the server, and handles
    dynamic layer allocation changes.
    """

    def __init__(
        self,
        whole_model: FlexibleSequential,
        last_layer_idx: int,
        init_parameters,
        server_model_proxy,
        optimizer_config=None,
        is_training: bool = True,
    ):
        """
        Initialize the ModelTrainer.

        Args:
            whole_model: The complete model architecture
            last_layer_idx: Index of the last layer to train on client
            init_parameters: Initial parameters for client layers
            server_model_proxy: Proxy for server communication
            is_training: Whether in training or evaluation mode
            learning_rate: Learning rate for optimizer
            momentum: Momentum for SGD optimizer
        """
        self._use_cuda = torch.cuda.is_available()
        self._whole_model = whole_model
        self._last_layer_idx = last_layer_idx
        self._model: Optional[FlexibleSequential] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._device = torch.device("cuda" if self._use_cuda else "cpu")
        self._server_model_proxy = server_model_proxy
        self._is_training = is_training
        self._criterion = torch.nn.CrossEntropyLoss()

        # Collected metrics
        self._metrics = {"loss": 0., "accuracy": 0.}
        self._dataset_size = 0
        self._num_processed_data_points = 0
        self._per_parameter_data_points = defaultdict(int)
        self._forward_time: float = -1.
        self._backward_time: float = -1.
        self._removed_parameters: List[OrderedDict] = []
        self._tot_local_computation_time = 0.

        self._init_model(last_layer_idx, init_parameters, optimizer_config)

        self._tmp_idx = 0  # TODO: Remove this debugging variable

    def _init_model(self, last_layer_idx: int, init_parameters, optimizer_config) -> None:
        """
        Initialize the client-side model.

        Args:
            last_layer_idx: Index of the last layer to include in client model
            init_parameters: Initial parameters for the model
        """
        self._model = extract_submodel(self._whole_model, 0, last_layer_idx)
        state_dict = param_list_to_state_dict(init_parameters, self._model)
        self._model.load_state_dict(state_dict)
        self._model.to(self._device)

        if self._is_training:
            self._model.train()
            self._optimizer = init_optimizer(self._model.parameters(), **optimizer_config)
        else:
            assert optimizer_config is None
            self._optimizer = None
            self._model.eval()

    def _update_model(self, control):
        if control is None or control.layers_diff == 0:
            # if not None, it means that we need to change the architecture of the client model
            # i.e., either add or remove some layers
            return

        if control.layers_diff > 0:
            log(DEBUG, "Receiving %d new layers" % control.layers_diff)
            self._model = attach_layers(
                self._whole_model,
                self._model,
                control.layers_diff,
                control.additional_layers_parameters,
                self._optimizer,
            )
            print("Increased model size", self._model)
        else:
            # need to better think how to properly handle this. For now, we store the
            # weights locally and upload to the server at the end of the training round
            # along with the other layers
            log(DEBUG, "Removing %d layers from client model" % -control.layers_diff)
            new_num_layers = get_num_layers(self._model) + control.layers_diff

            params = extract_submodel_state_dict(self._model, new_num_layers, get_num_layers(self._model))

            self._removed_parameters.append(params)
            self._model = extract_submodel(self._model, 0, new_num_layers)
            print("Reduced model size", self._model)

    def __call__(self, input_vars, labels):
        self._tmp_idx += 1

        input_vars = input_vars.to(self._device)
        self._optimizer.zero_grad()

        # forward on client model
        with measure_time(self._use_cuda) as timer:
            client_output = self._model(input_vars)
        self._forward_time = timer.elapsed # perhaps could have a rolling average
        self._tot_local_computation_time += self._forward_time

        if self._model._is_complete:
            # if model is in its entirety on the client, compute the loss on the client
            with measure_time(self._use_cuda) as timer:
                loss = self._criterion(client_output, labels.to(self._device))
                loss.backward()
                self._optimizer.step()
            self._backward_time = timer.elapsed
            loss = loss.item()
            control = None
        else:
            # request over network to the server -- SL step
            gradient, control, loss = self._server_model_proxy(
                embeddings=client_output,
                labels=labels,
                local_computation_time=-1 if self._backward_time < 0 else \
                    self._backward_time + self._forward_time
            )
            with measure_time(self._use_cuda) as timer:
                client_output.backward(gradient.to(client_output.device))
                self._optimizer.step()
            self._backward_time = timer.elapsed
        self._tot_local_computation_time += self._backward_time

        self._metrics["loss"] += loss * labels.shape[0]
        for layer_idx in range(get_num_layers(self._model)):
            self._per_parameter_data_points[layer_idx] += input_vars.shape[0]
        self._update_model(control)

        self._dataset_size += input_vars.shape[0]
        self._num_processed_data_points += labels.shape[0]

    def evaluate(self, input_vars, output_vars):
        input_vars = input_vars.to(self._device)

        with torch.no_grad():
            with measure_time(self._use_cuda) as timer:
                client_output = self._model(input_vars)
            self._forward_time = timer.elapsed

        self._dataset_size += input_vars.shape[0]
        control = None
        if self._model._is_complete:
            logits = client_output
        else:
            logits, control, _ = self._server_model_proxy(
                embeddings=client_output,
                labels=None,
                training=False,
                local_computation_time=self._forward_time,
            )
        logits = logits.to(self._device)
        self._num_processed_data_points += output_vars.shape[0]
        self._metrics["loss"] += self._criterion(logits, output_vars) * output_vars.shape[0]
        self._metrics["accuracy"] += (logits.argmax(dim=1) == output_vars).sum()
        self._update_model(control)

    def get_evaluation_metrics(self):
        return {
            k: v.item() / self._num_processed_data_points for k, v in self._metrics.items()
        }

    def get_training_metrics(self):
        print("Total local computation time", self._tot_local_computation_time)
        print({"training_loss": self._metrics["loss"] / self._num_processed_data_points})
        return {
            "training_loss": self._metrics["loss"] / self._num_processed_data_points,
            "local_computation_time": self._tot_local_computation_time
        }

    def get_fit_results(self):
        # TODO Create a dictionary with the latest values for each key
        # probably not ideal, as previous versions are discarded
        combined_state_dict = {}
        for params_dict in self._removed_parameters:
            for key, value in params_dict.items():
                combined_state_dict[key] = value

        # First add the current model state dict
        combined_state_dict.update(dict(self._model.state_dict()))

        # Convert the combined state dict to list of numpy arrays
        params = []
        for k in self._whole_model.state_dict():
            if k in combined_state_dict:
                params.append(combined_state_dict[k].detach().cpu().numpy())
                del combined_state_dict[k]
            else:
                assert len(combined_state_dict) == 0
                break
        return (
            params,
            {f"_proc{k}": v for k, v in self._per_parameter_data_points.items()},
            self.get_training_metrics()
        )

    def stupid_forward(self, x, y):
        # for testing the server... just send the data to the server
        # without any local computation
        _ = self._server_model_proxy(
            embeddings=x,
            labels=y,
            local_computation_time=-1
        )
        # dummy values
        self._dataset_size += 1
        self._num_processed_data_points += 1
        for layer_idx in range(get_num_layers(self._model)):
            self._per_parameter_data_points[layer_idx] += 1
