from typing import List, Dict, Optional
from collections import defaultdict

import numpy as np
import torch

from slbd.torch.model_utils import shift_state_dict
from slbd.torch.utils import init_optimizer
from slbd.common.typing import ServerModelFitIns, ServerModelEvaluateIns, ServerModelFitRes
from slbd.torch.model_utils import extract_submodel, get_num_layers
from slbd.torch.sequential_model import FlexibleSequential


class TorchServerModel:
    """Class is fully controlled by the framework and cannot be overridden by the model owner."""
    def __init__(
        self,
        model: FlexibleSequential,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        self._whole_model = model
        self._model = None

        self.device = device
        self.first_server_layer = -1
        self._num_model_layers = get_num_layers(model)
        self._optimizer = None
        self._criterion = torch.nn.CrossEntropyLoss()
        self._per_layer_data_points = defaultdict(int)
        self._metrics_dict = {}

    def get_train_results(self):
        state_dict = shift_state_dict(
            self._model.state_dict(),
            self.first_server_layer
        )
        return ServerModelFitRes(
            model_state_dict=state_dict,
            per_layer_training_examples={k: int(v) for k, v in self._per_layer_data_points.items()},
            metrics=self._metrics_dict,
        )

    def _get_model_from_ins(self, instructions):
        self.first_server_layer = instructions.first_server_model_layer
        model = extract_submodel(self._whole_model, instructions.first_server_model_layer, -1)
        state_dict = shift_state_dict(instructions.model_state_dict, -self.first_server_layer)
        model.load_state_dict(state_dict)
        return model

    def get_fit_result(self) -> ServerModelFitRes:
        return self.get_train_results()

    def configure_fit(
        self,
        instructions: ServerModelFitIns
    ) -> None:
        self._model = self._get_model_from_ins(instructions)
        self._model.train()
        self._model.to(self.device)
        optim_config = {k: v for k, v in instructions.config.items() if k.startswith("optim.")}
        self._optimizer = init_optimizer(self._model.parameters(), **optim_config)

    def configure_evaluate(
        self,
        ins: ServerModelEvaluateIns
    ) -> None:
        self._model = self._get_model_from_ins(ins)

        self._model.eval()
        self._model.to(self.device)

    def train_and_compute_gradient(
        self,
        embeddings,
        labels,
        num_offloaded_layers,
    ):
        is_tuple = isinstance(embeddings, tuple)
        if is_tuple:
            embeddings[0].requires_grad_(True)
            embeddings[0].grad = None # required because otherwise the gradient may be accumulated
        else:
            embeddings.requires_grad_(True)
            embeddings.grad = None # required because otherwise the gradient may be accumulated

        self._optimizer.zero_grad(set_to_none=True)
        logits = self._forward_sequence_of_layers(embeddings, num_offloaded_layers)
        loss = self._criterion(logits, labels)
        loss.backward()
        self._optimizer.step()

        if is_tuple:
            gradient = embeddings[0].grad
        else:
            gradient = embeddings.grad

        # all clients get assigned the same loss
        return gradient.detach().cpu(), loss.item()

    def _forward_sequence_of_layers(self, data, num_offloaded_layers):
        layers = list(self._model)[-num_offloaded_layers:]

        x = data
        if isinstance(x, torch.Tensor):
            bs = x.shape[0]
            for idx, layer in enumerate(layers, start=self._num_model_layers - num_offloaded_layers):
                x = layer(x)
                self._per_layer_data_points[idx] += bs
        else:
            bs = x[0].shape[0]
            for idx, layer in enumerate(layers, start=self._num_model_layers - num_offloaded_layers):
                x = layer(*x)
                self._per_layer_data_points[idx] += bs
        return x

    def eval_get_logits(self, batches, num_offloaded_layers):
        sizes = [b.shape[0] for b in batches["embeddings"]]
        embeddings = torch.cat([torch.from_numpy(b) for b in batches["embeddings"]], dim=0)

        with torch.no_grad():
            logits = self._forward_sequence_of_layers(embeddings.to(self.device), num_offloaded_layers)

        logits = [(l.detach().cpu().numpy(), -1) for l in logits.split(sizes, dim=0)]
        return logits

    def get_model(self):
        return self._model

    def get_layer_processed_data_points(self, layer_idx):
        return self._per_layer_data_points[layer_idx]

    def integrate_client_parameters(
        self,
        parameters: List[torch.Tensor],
        num_dp_per_layer: Dict[int, int],
        num_dp_when_forked: Dict[int, int],
        optimizer_states: Optional[List[torch.Tensor]] = None,
    ):
        idx = 0
        layers = sorted(num_dp_per_layer.keys())
        named_params = dict(self._model.named_parameters())

        # Collect parameters that will be updated in layer order (consistent with extraction)
        updated_params = []

        # Get parameters in the same order as extraction (layer by layer)
        for layer_idx in layers:
            actual_layer_idx = layer_idx - self.first_server_layer
            if actual_layer_idx >= 0 and actual_layer_idx < len(list(self._model.children())):
                layer = list(self._model.children())[actual_layer_idx]
                updated_params.extend(list(layer.parameters()))

        # Update parameters using state_dict order (as before)
        idx = 0
        for param_key in self._model.state_dict():
            if int(param_key.split(".")[0]) + self.first_server_layer in layers:
                tensor_value = torch.from_numpy(parameters[idx])
                idx += 1
                if param_key in named_params:
                    parameter = named_params[param_key]
                    if len(self._optimizer.state) > 0:
                        del self._optimizer.state[parameter]
                    parameter.data.copy_(tensor_value)
                else:
                    self._model.state_dict()[param_key].copy_(tensor_value)

        # Restore optimizer states if provided
        if optimizer_states is not None:
            self._restore_optimizer_states_for_params(updated_params, optimizer_states)

    def reset_per_layer_counters(self, layer_from, layer_to):
        for idx in range(layer_from, layer_to):
            self._per_layer_data_points[idx] = 0

    def set_per_layer_counter(self, idx, value):
        self._per_layer_data_points[idx] = value

    def _restore_optimizer_states_for_params(self, parameters, optimizer_states_list):
        """
        Restore optimizer states for the given parameters from a flattened list of states.

        Args:
            parameters: The parameters to restore states for (should be in consistent order)
            optimizer_states_list: Flattened list of optimizer state tensors from client
        """
        if not optimizer_states_list:
            return

        state_idx = 0
        # Ensure consistent ordering of parameters
        param_list = list(parameters) if not isinstance(parameters, list) else parameters

        for param in param_list:
            if state_idx >= len(optimizer_states_list):
                break

            # Initialize optimizer state for this parameter if it doesn't exist
            if param not in self._optimizer.state:
                self._optimizer.state[param] = {}

            state = self._optimizer.state[param]

            # Restore states in the same order they were extracted
            if isinstance(self._optimizer, torch.optim.SGD):
                # Only momentum_buffer for SGD
                if state_idx < len(optimizer_states_list):
                    momentum_tensor = torch.from_numpy(optimizer_states_list[state_idx])
                    if momentum_tensor.shape != param.shape:
                        # Skip this mismatched state
                        state_idx += 1
                        continue
                    # Only set momentum_buffer if it's not all zeros (wasn't a placeholder)
                    if not np.allclose(momentum_tensor.numpy(), 0):
                        state['momentum_buffer'] = momentum_tensor.to(param.device)
                        assert momentum_tensor.shape == param.shape
                    state_idx += 1
            elif isinstance(self._optimizer, torch.optim.Adam):
                # Adam: step (scalar), exp_avg, exp_avg_sq in that order
                if state_idx < len(optimizer_states_list):
                    step_val = int(optimizer_states_list[state_idx][0])
                    if step_val > 0:  # Only set if not a placeholder
                        state['step'] = step_val
                    state_idx += 1
                if state_idx < len(optimizer_states_list):
                    exp_avg_tensor = torch.from_numpy(optimizer_states_list[state_idx])
                    if not np.allclose(exp_avg_tensor.numpy(), 0):  # Only set if not a placeholder
                        state['exp_avg'] = exp_avg_tensor.to(param.device)
                    state_idx += 1
                if state_idx < len(optimizer_states_list):
                    exp_avg_sq_tensor = torch.from_numpy(optimizer_states_list[state_idx])
                    if not np.allclose(exp_avg_sq_tensor.numpy(), 0):  # Only set if not a placeholder
                        state['exp_avg_sq'] = exp_avg_sq_tensor.to(param.device)
                    state_idx += 1
