import time
from logging import DEBUG
from collections import defaultdict
from contextlib import contextmanager

import torch
import torch.nn as nn
from flwr.common import log

from slbd.torch.model_utils import (
    extract_submodel,
    param_list_to_state_dict,
    attach_layers,
    extract_submodel_weights,
    get_num_layers
)


@contextmanager
def measure_time(time_list, use_cuda):
    if use_cuda:
        torch.cuda.synchronize()
    start = time.time()
    yield
    if use_cuda:
        torch.cuda.synchronize()
    end = time.time()
    time_list.append(end - start)


class ModelTrainer:
    def __init__(
        self,
        whole_model: nn.Sequential,
        last_layer_idx: int,
        init_parameters,
        server_model_proxy,
        is_training=True
    ):
        self._use_cuda = torch.cuda.is_available()
        self._whole_model = whole_model
        self._last_layer_idx = last_layer_idx
        self._model = None
        self._optimizer = None
        self._last_layer_idx = last_layer_idx
        self._device = torch.device("cuda" if self._use_cuda else "cpu")
        self._server_model_proxy = server_model_proxy
        self._is_training = is_training
        self._criterion = torch.nn.CrossEntropyLoss()

        # collected metrics
        self._metrics = {"loss": 0., "accuracy": 0.}
        self._dataset_size = 0
        self._num_batches = 0
        self._per_parameter_data_points = defaultdict(int)
        self.forward_times = []
        self.backward_times = []
        self._init_model(last_layer_idx, init_parameters)
        self._removed_parameters = []

        self._tmp_idx = 0 # for debugging/testing

    def _init_model(self, last_layer_idx, init_parameters):
        self._model = extract_submodel(self._whole_model, 0, last_layer_idx)
        state_dict = param_list_to_state_dict(init_parameters, self._model)
        self._model.load_state_dict(state_dict)
        self._model.to(self._device)
        if self._is_training:
            self._model.train()
            self._optimizer = torch.optim.SGD(self._model.parameters(), lr=0.01, momentum=0.9)
        else:
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
        else:
            # need to better think how to properly handle this. For now, we store the
            # weights locally and upload to the server at the end of the training round
            # along with the other layers
            log(DEBUG, "Removing %d layers from client model" % -control.layers_diff)
            params = extract_submodel_weights(self._model, -control.layers_diff, -1)
            # append at the beginning
            self._removed_parameters = params + self._removed_parameters
            self._model = extract_submodel(self._model, 0, -control.layers_diff)

    def __call__(self, input_vars, labels):
        self._tmp_idx += 1

        input_vars = input_vars.to(self._device)
        self._optimizer.zero_grad()

        # forward on client model
        with measure_time(self.forward_times, self._use_cuda):
            # if 5 < self._tmp_idx < 12:
            #     time.sleep(0.8)
            client_output = self._model(input_vars)

        if self._model._is_complete:
            # if model is in its entirety on the client, compute the loss on the client
            with measure_time(self.backward_times, self._use_cuda):
                loss = self._criterion(client_output, labels.to(self._device))
                loss.backward()
                self._optimizer.step()
            control = None
        else:
            # request over network to the server -- SL step
            gradient, control = self._server_model_proxy(
                embeddings=client_output,
                labels=labels,
                local_computation_time=-1 if len(self.backward_times) == 0 else \
                    self.backward_times[-1] + self.forward_times[-1]
            )
            with measure_time(self.backward_times, self._use_cuda):
                client_output.backward(gradient.to(client_output.device))
                self._optimizer.step()

        for layer_idx in range(get_num_layers(self._model)):
            self._per_parameter_data_points[layer_idx] += input_vars.shape[0]
        self._update_model(control)

        self._dataset_size += input_vars.shape[0]
        self._num_batches += 1

    def evaluate(self, input_vars, output_vars):
        input_vars = input_vars.to(self._device)

        with torch.no_grad():
            with measure_time(self.forward_times, self._use_cuda):
                client_output = self._model(input_vars)

        self._dataset_size += input_vars.shape[0]
        control = None
        if self._model._is_complete:
            logits = client_output
        else:
            logits, control = self._server_model_proxy(
                embeddings=client_output,
                labels=None,
                training=False,
                local_computation_time=self.forward_times[-1],
            )
        logits = logits.to(self._device)
        self._num_batches += 1
        self._metrics["loss"] += self._criterion(logits, output_vars)
        self._metrics["accuracy"] += (logits.argmax(dim=1) == output_vars).sum() / logits.shape[0]
        self._update_model(control)

    def get_evaluation_metrics(self):
        return {
            k: v.item() / self._num_batches for k, v in self._metrics.items()
        }

    def get_fit_results(self):
        params = [
            v.cpu().numpy() for v in self._model.state_dict().values()
        ] + self._removed_parameters
        return params, {f"_proc{k}": v for k, v in self._per_parameter_data_points.items()}
