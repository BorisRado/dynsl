import time
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
    remove_layers_from_model,
    param_list_to_state_dict,
    attach_layers,
    get_num_layers,
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
        num_training_layers: int,
        num_received_layers: int,
        init_parameters,
        server_model_proxy,
        optimizer_config,
        is_training,
        sending_transform_fn,
        receiving_transform_fn,
        dynamic_layers_strategy,
    ):
        """
        Initialize the ModelTrainer.

        Args:
            whole_model: The complete model architecture
            num_training_layers: Index of the last layer to train on client
            init_parameters: Initial parameters for client layers
            server_model_proxy: Proxy for server communication
            is_training: Whether in training or evaluation mode
            learning_rate: Learning rate for optimizer
            momentum: Momentum for SGD optimizer
        """
        # sanity checks
        assert dynamic_layers_strategy in {None, "cnt", "afl"}
        assert num_training_layers <= num_received_layers

        self._use_cuda = torch.cuda.is_available()
        self._whole_model = whole_model
        self._num_layers_entire_model = get_num_layers(self._whole_model)
        self._num_training_layers = num_training_layers
        self._num_received_layers = num_received_layers
        self._model: Optional[FlexibleSequential] = None
        self._optimizer: Optional[torch.optim.Optimizer] = None
        self._device = torch.device("cuda" if self._use_cuda else "cpu")
        self._server_model_proxy = server_model_proxy
        self._is_training = is_training
        self._dynamic_layers_strategy = dynamic_layers_strategy
        self._criterion = torch.nn.CrossEntropyLoss()
        self._removed_weights = []

        # Collected metrics
        self._metrics = {"loss": 0., "accuracy": 0.}
        self._dataset_size = 0
        self._per_layer_data_points = defaultdict(int)
        self._forward_time: float = -1.
        self._backward_time: float = -1.
        self._last_server_msg_time: float = -1.
        self._tot_local_computation_time = 0.

        self.sending_transform_fn = sending_transform_fn
        self.receiving_transform_fn = receiving_transform_fn

        print(f"Received number of layers: {num_received_layers} ({num_training_layers})")
        self._init_model(num_received_layers, init_parameters, optimizer_config)

        self._iter_since_last_model_change = 0

    def _init_model(self, num_received_layers: int, init_parameters, optimizer_config) -> None:
        """
        Initialize the client-side model.

        Args:
            num_training_layers: Index of the last layer to include in client model
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

        assert self._dynamic_layers_strategy is not None

        prev_num_of_layers = self._num_training_layers
        self._num_training_layers += control.layers_diff

        if control.layers_diff > 0:
            log(DEBUG, "Receiving %d new layers" % control.layers_diff)
            self._iter_since_last_model_change = 0
            if len(control.additional_layers_parameters) > 0:
                assert self._dynamic_layers_strategy == "afl"
                self._model = attach_layers(
                    self._whole_model,
                    self._model,
                    control.layers_diff,
                    control.additional_layers_parameters,
                    self._optimizer,
                )
                # reset counter
                for k in range(prev_num_of_layers, self._num_training_layers):
                    self._per_layer_data_points[k] = 0
            else:
                assert self._dynamic_layers_strategy == "cnt"
                print("Server is NOT forking, just increasing counter", self._num_training_layers)
                assert self._num_training_layers <= get_num_layers(self._model)

        else:

            log(DEBUG, "Removing %d layers from client model" % -control.layers_diff)
            if self._dynamic_layers_strategy == "cnt":
                log(DEBUG, "Using CNT strategy, just decreasing counter %d", self._num_training_layers)
                return

            elif self._dynamic_layers_strategy == "afl":
                self._model, self._removed_weights = \
                    remove_layers_from_model(self._model, self._optimizer, self._num_training_layers)
                # attach a vector stating how many data points have been processed by these layers
                self._removed_weights += [torch.tensor([
                    self._per_layer_data_points[k]
                    for k in range(self._num_training_layers, prev_num_of_layers)
                ]),]

                log(DEBUG, "Reduced model size to %d", get_num_layers(self._model))
                for k in range(self._num_training_layers, prev_num_of_layers):
                    self._per_layer_data_points[k] = 0

        if self._dynamic_layers_strategy == "afl":
            assert self._num_training_layers == get_num_layers(self._model)

    def __call__(self, input_vars, labels):
        payload, client_output = self.forward_pass(input_vars, labels)
        if payload is None:
            return
        gradient, control, loss = self._server_model_proxy(**payload)
        self.backward_pass(gradient, control, loss, client_output)

    def forward_pass(self, input_vars, labels):
        self._iter_since_last_model_change += + 1

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

        if self._num_training_layers == self._num_layers_entire_model:
            # if model is in its entirety on the client, compute the loss on the client
            with measure_time(self._use_cuda) as timer:
                loss = self._criterion(client_output, labels.to(self._device))
                loss.backward()
                self._optimizer.step()
            self._backward_time = timer.elapsed
            loss = loss.item()
            self._update_metrics(loss, labels.shape[0])
            return None, None

        send_client_output = client_output
        if self.sending_transform_fn is not None:
            send_client_output = self.sending_transform_fn(client_output)
        client_time = self._backward_time + self._forward_time
        return {
            "embeddings": send_client_output,
            "labels": labels,
            "local_computation_time": -1 if self._iter_since_last_model_change < 4 else client_time,
            "tot_local_time": time.perf_counter() - self._last_server_msg_time,
            "__removed_parameters": self._removed_weights,
        }, client_output

    def backward_pass(self, gradient, control, loss, client_output):
        self._removed_weights = []
        self._last_server_msg_time = time.perf_counter()
        if self.receiving_transform_fn is not None:
            gradient = self.receiving_transform_fn(gradient, client_output=client_output)
        with measure_time(self._use_cuda) as timer:
            if not isinstance(client_output, torch.Tensor):
                client_output[0].backward(gradient.to(self._device))
            else:
                client_output.backward(gradient.to(self._device))
            self._optimizer.step()
        self._backward_time = timer.elapsed
        self._tot_local_computation_time += self._backward_time
        self._update_model(control)
        self._update_metrics(loss, gradient.shape[0])

    def _update_metrics(self, loss, batch_size):
        self._metrics["loss"] += loss * batch_size
        self._dataset_size += batch_size

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
        # print("Total local computation time", self._tot_local_computation_time)
        # print({"training_loss": self._metrics["loss"] / self._dataset_size})
        return {
            "training_loss": self._metrics["loss"] / self._dataset_size,
            "local_computation_time": self._tot_local_computation_time
        }

    def get_fit_results(self):
        params = []
        for k, param in self._model.state_dict().items():
            layer_idx = int(k.split(".")[0])
            if self._per_layer_data_points[layer_idx] == 0:
                # we did not train any further model parameters
                break
            params.append(param.detach().cpu().numpy())
            print(k, params[-1].flatten().tolist()[:2])

        return (
            params,
            {f"_proc{k}": v for k, v in self._per_layer_data_points.items()},
            self.get_training_metrics()
        )

    def _forward_sequence_of_layers(self, data):
        layers = list(self._model)[:self._num_training_layers]

        x = data
        if isinstance(x, torch.Tensor):
            bs = x.shape[0]
            for idx, layer in enumerate(layers):
                x = layer(x)
                self._per_layer_data_points[idx] += bs
        else:
            bs = x[0].shape[0]
            for idx, layer in enumerate(layers):
                x = layer(*x)
                self._per_layer_data_points[idx] += bs
        return x
