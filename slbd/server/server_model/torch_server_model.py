from typing import List, Dict, Optional
from collections import defaultdict

import torch

from slbd.torch.model_utils import shift_state_dict
from slbd.torch.utils import init_optimizer
from slbd.common.typing import ServerModelFitIns, ServerModelEvaluateIns, ServerModelFitRes
from slbd.torch.model_utils import extract_submodel, get_num_layers, override_optimizer_state, iterate_params_and_state_dict
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
        self._per_layer_data_points = defaultdict(int) # layer idxs are absolute!!!
        self._metrics_dict = {"train/server_loss": 0.}
        self._weight_integration_strategy = None

    def get_train_results(self):
        if len(self._per_layer_data_points) == 0:
            return None
        state_dict = shift_state_dict(
            self._model.state_dict(),
            self.first_server_layer
        )
        per_layer_training_examples = {k: int(v) for k, v in self._per_layer_data_points.items()}
        if len(per_layer_training_examples) != len(self._model):
            delete_list = []
            for k in state_dict:
                if int(k.split(".")[0]) not in per_layer_training_examples:
                    delete_list.append(k)
            for k in delete_list:
                del state_dict[k]
        self._metrics_dict["train/server_loss"] /= max(per_layer_training_examples.values())
        return ServerModelFitRes(
            model_state_dict=state_dict,
            metrics=self._metrics_dict,
            per_layer_training_examples=per_layer_training_examples,
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
        self._weight_integration_strategy = instructions.config.get("weight_integration_strategy", None)
        print("===", self._weight_integration_strategy)
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
        self._metrics_dict["train/server_loss"] += loss.item() * logits.shape[0]
        loss.backward()
        self._optimizer.step()

        if is_tuple:
            gradient = embeddings[0].grad
        else:
            gradient = embeddings.grad

        # all clients get assigned the same loss
        return gradient.detach(), loss.item()

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

    def integrate_client_parameters(
        self,
        parameters: List[torch.Tensor],
        received_layers: List[int],
        client_weights: Dict[int, int],
        server_model_weights: Dict[int, int],
        optimizer_states: Optional[List[torch.Tensor]] = None,
    ):
        if optimizer_states:
            assert self._weight_integration_strategy == "override_optim_states"
        else:
            assert self._weight_integration_strategy in {"reset_optim_state", "keep_optim_state", "ignore"}
        if self._weight_integration_strategy == "ignore":
            return
        idx = 0
        integrated_parameters = []
        for layer_idx in received_layers:
            server_layer_idx = layer_idx - self.first_server_layer
            s_weight = server_model_weights[layer_idx]
            c_weight = client_weights[layer_idx]
            tot_dp = s_weight + c_weight
            s_weight, c_weight = s_weight / tot_dp, c_weight / tot_dp

            for _, server_param in iterate_params_and_state_dict(self._model[server_layer_idx]):
                client_param = parameters[idx]
                assert client_param.shape == server_param.shape
                if server_param.data.dtype == torch.float32:
                    server_param.data \
                        .mul_(s_weight) \
                        .add_(
                            client_param.to(server_param.device) * c_weight
                        )
                elif server_param.data.dtype == torch.long:
                    server_param.data.copy_(
                        (server_param.data.float() * s_weight +
                         client_param.to(server_param.device).float() * c_weight)
                        .round().long()
                    )
                else:
                    raise Exception(f"Unknown type {server_param.dtype}")
                integrated_parameters.append(server_param)

                if isinstance(server_param, torch.nn.Parameter):
                    # since we "override" the weight, we reset the optimizer state for the considered parameter
                    if self._weight_integration_strategy == "reset_optim_state":
                        self._optimizer.state[server_param] = {}
                    elif self._weight_integration_strategy == "keep_optim_state":
                        pass
                    else:
                        assert self._weight_integration_strategy == "override_optim_states"

                idx += 1
            if idx == len(parameters):
                break

        # Restore optimizer states if provided -- should only be used when clients are paired
        # with a dedicated classificaiton head
        if optimizer_states:
            assert self._weight_integration_strategy == "override_optim_states"
            override_optimizer_state(self._optimizer, optimizer_states, integrated_parameters)

    def get_current_layer_data_points(self, layer_idx):
        return self._per_layer_data_points[layer_idx]

    def increase_per_layer_counter(self, idx, increase):
        self._per_layer_data_points[idx] += increase
