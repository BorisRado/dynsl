import time
import psutil
from logging import DEBUG
from typing import Optional
from collections import defaultdict

import torch
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
        *,
        whole_model: FlexibleSequential,
        last_layer_idx: int,
        num_received_layers: int,
        init_parameters,
        server_model_proxy,
        optimizer_config,
        is_training,
        sending_transform_fn,
        receiving_transform_fn,
        server_forking=False,
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
        self._num_layers_whole_model = get_num_layers(self._whole_model)
        self._last_layer_idx = last_layer_idx
        self._num_received_layers = num_received_layers
        self._model: Optional[FlexibleSequential] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._device = torch.device("cuda" if self._use_cuda else "cpu")
        self._server_model_proxy = server_model_proxy
        self._is_training = is_training
        self._server_forking = server_forking
        self._criterion = torch.nn.CrossEntropyLoss()

        # Collected metrics
        self._metrics = {"loss": 0., "accuracy": 0.}
        self._dataset_size = 0
        self._per_parameter_data_points = defaultdict(int)
        self._num_dp_when_removed = defaultdict(int)
        self._forward_time: float = -1.
        self._backward_time: float = -1.
        self._last_server_msg_time: float = -1.
        self._removed_parameters = {}
        self._tot_local_computation_time = 0.

        self.sending_transform_fn = sending_transform_fn
        self.receiving_transform_fn = receiving_transform_fn

        print("Received number of layers: ", num_received_layers)
        print("Num train layers", last_layer_idx)
        self._init_model(num_received_layers, init_parameters, optimizer_config)

        self._iter_since_last_model_change = 0
        self._num_model_layers = get_num_layers(self._model)

    def _init_model(self, num_received_layers: int, init_parameters, optimizer_config) -> None:
        """
        Initialize the client-side model.

        Args:
            last_layer_idx: Index of the last layer to include in client model
            init_parameters: Initial parameters for the model
        """
        self._model = extract_submodel(self._whole_model, 0, num_received_layers)
        state_dict = param_list_to_state_dict(init_parameters, self._model)
        self._model.load_state_dict(state_dict, strict=True)
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
            self._iter_since_last_model_change = 0
            additional_layers_parameters = control.additional_layers_parameters
            if self._server_forking:
                assert additional_layers_parameters is not None
                self._model = attach_layers(
                    self._whole_model,
                    self._model,
                    control.layers_diff,
                    control.additional_layers_parameters,
                    self._optimizer,
                )
            else:
                assert additional_layers_parameters is None
                self._last_layer_idx = self._last_layer_idx + control.layers_diff
                print("Server is NOT forking, just increasing counter", self._last_layer_idx)
                assert self._last_layer_idx <= get_num_layers(self._model)

            print("Increased model size", self._model)
        else:
            # need to better think how to properly handle this. For now, we store the
            # weights locally and upload to the server at the end of the training round
            # along with the other layers
            log(DEBUG, "Removing %d layers from client model" % -control.layers_diff)
            self._last_layer_idx = self._last_layer_idx + control.layers_diff
            if not self._server_forking:
                print("Server is NOT forking, just decreasing counter", self._last_layer_idx)
                return
            new_num_layers = self._num_model_layers + control.layers_diff

            assert False
            params = extract_submodel_state_dict(self._model, new_num_layers, self._num_model_layers)

            for removed_key, removed_param in params.state_dict():
                if removed_key in self._removed_parameters:
                    assert removed_key in self._num_dp_when_removed
                    value = self._weighted_average(
                        old_value=self._removed_parameters[removed_key],
                        new_value=removed_param,
                        old_weight=self._num_dp_when_removed[removed_key],
                        new_weight=self._per_parameter_data_points[removed_key]-self._num_dp_when_removed[removed_key]
                    )

                    self._removed_parameters[removed_key] = value
                    self._num_dp_when_removed[removed_key] = self._per_parameter_data_points[removed_key]

                else:
                    self._removed_parameters[removed_key] = removed_param
                    self._num_dp_when_removed[removed_key] = self._per_parameter_data_points[removed_key]

            self._model = extract_submodel(self._model, 0, new_num_layers)
            print("Reduced model size", self._model)

    def _weighted_average(self, old_value, new_value, old_weight, new_weight):
        value = (old_value * old_weight + new_value * new_weight) / (old_weight + new_weight)
        return value.detach().cpu()

    def __call__(self, input_vars, labels):
        self._iter_since_last_model_change = self._iter_since_last_model_change + 1

        # forward on client model
        with measure_time(self._use_cuda) as timer:

            is_tuple = isinstance(input_vars, tuple)
            if is_tuple:
                input_vars = tuple(t.to(self._device) for t in input_vars)
            else:
                input_vars = input_vars.to(self._device)
            self._optimizer.zero_grad()
            client_output = self._forward_sequence_of_layers(input_vars)

        self._forward_time = timer.elapsed # perhaps could have a rolling average
        self._tot_local_computation_time += self._forward_time

        if self._last_layer_idx == self._num_layers_whole_model:
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
            send_client_output = client_output
            if self.sending_transform_fn is not None:
                send_client_output = self.sending_transform_fn(client_output)
            client_time = self._backward_time + self._forward_time
            if psutil.virtual_memory().percent > 90:
                client_time *= 3.0
            if psutil.cpu_percent(interval=None) > 90:
                client_time *= 1.5
            gradient, control, loss = self._server_model_proxy(
                embeddings=send_client_output,
                labels=labels,
                local_computation_time=-1 if self._iter_since_last_model_change < 4 else client_time,
                tot_local_time=time.perf_counter() - self._last_server_msg_time,
            )
            self._last_server_msg_time = time.perf_counter()
            if self.receiving_transform_fn is not None:
                gradient = self.receiving_transform_fn(gradient, client_output=client_output)
            with measure_time(self._use_cuda) as timer:
                if is_tuple:
                    # TODO: for now let's assume that the first tensor is the one that requires grad
                    client_output[0].backward(gradient.to(self._device))
                else:
                    client_output.backward(gradient.to(self._device))
                self._optimizer.step()
            self._backward_time = timer.elapsed
        self._tot_local_computation_time += self._backward_time

        self._update_model(control)

        _this_batch_size = labels.shape[0]
        self._metrics["loss"] += loss * _this_batch_size
        self._dataset_size += _this_batch_size

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
        self._metrics["loss"] += self._criterion(logits, output_vars) * output_vars.shape[0]
        self._metrics["accuracy"] += (logits.argmax(dim=1) == output_vars).sum()
        self._update_model(control)

    def get_evaluation_metrics(self):
        return {
            k: v.item() / self._dataset_size for k, v in self._metrics.items()
        }

    def get_training_metrics(self):
        print("Total local computation time", self._tot_local_computation_time)
        print({"training_loss": self._metrics["loss"] / self._dataset_size})
        return {
            "training_loss": self._metrics["loss"] / self._dataset_size,
            "local_computation_time": self._tot_local_computation_time
        }

    def get_fit_results(self):

        # Convert the combined state dict to list of numpy arrays
        params = []

        for k, param in self._model.state_dict().items():
            layer_idx = int(k.split(".")[0])
            if self._per_parameter_data_points[layer_idx] == 0:
                # we did not train any further model parameters
                break
            elif k not in self._removed_parameters:
                # we have never removed the layer from local training
                params.append(param.detach().cpu().numpy())
            elif self._num_dp_when_removed[k] == self._per_parameter_data_points[k]:
                assert False
                params.append(self._removed_parameters[k])
            else:
                # we have removed it, and trained again. Thus, we need a weighted average
                assert False
                value = self._weighted_average(
                    old_value=self._removed_parameters[k],
                    new_value=param,
                    old_weight=self._num_dp_when_removed[k],
                    new_weight=self._per_parameter_data_points[k]-self._num_dp_when_removed[k]
                )
                params.append(value.detach().cpu().numpy())

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
        for layer_idx in range(get_num_layers(self._model)):
            self._per_parameter_data_points[layer_idx] += 1

    def _forward_sequence_of_layers(self, data):
        layers = list(self._model)[:self._last_layer_idx]

        x = data
        if isinstance(x, torch.Tensor):
            bs = x.shape[0]
            for idx, layer in enumerate(layers):
                x = layer(x)
                self._per_parameter_data_points[idx] += bs
        else:
            bs = x[0].shape[0]
            for idx, layer in enumerate(layers):
                x = layer(*x)
                self._per_parameter_data_points[idx] += bs
        return x
