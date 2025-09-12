from collections import defaultdict

import torch

from slbd.torch.model_utils import shift_state_dict
from slbd.torch.utils import init_optimizer
from slbd.common.typing import ServerModelFitIns, ServerModelEvaluateIns, ServerModelFitRes
from slbd.torch.model_utils import extract_submodel, get_num_layers


class TorchServerModel:
    """Class is fully controlled by the framework and cannot be overridden by the model owner."""
    def __init__(
        self,
        model: torch.nn.Sequential,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        self._whole_model = model
        self._model = None

        self.device = device
        self.first_server_layer = -1
        self._num_model_layers = get_num_layers(model)
        self._optimizer = None
        self._criterion = torch.nn.CrossEntropyLoss()
        self._per_parameter_data_points = defaultdict(int)
        self._metrics_dict = {}
        self._mix_before_model = False

    def get_train_results(self):
        state_dict = shift_state_dict(
            self._model.state_dict(),
            self.first_server_layer
        )
        return ServerModelFitRes(
            model_state_dict=state_dict,
            per_layer_training_examples=self._per_parameter_data_points,
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
        embeddings.requires_grad_(True)

        self._optimizer.zero_grad(set_to_none=True)
        logits = self._forward_sequence_of_layers(embeddings, num_offloaded_layers)
        loss = self._criterion(logits, labels)
        loss.backward()
        self._optimizer.step()

        gradient = embeddings.grad

        # all clients get assigned the same loss
        return gradient.detach().cpu(), loss.item()

    def _forward_sequence_of_layers(self, data, num_offloaded_layers):
        layers = self._model[-num_offloaded_layers:].children()

        x = data
        for idx, layer in enumerate(layers, start=self._num_model_layers - num_offloaded_layers):
            x = layer(x)
            self._per_parameter_data_points[idx] += x.shape[0]
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
        return self._per_parameter_data_points[layer_idx]
