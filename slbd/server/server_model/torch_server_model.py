from collections import OrderedDict, defaultdict
from typing import Tuple

import torch

from slbd.torch.model_utils import shift_state_dict
from slbd.common.typing import ServerModelFitIns, ServerModelEvaluateIns, ServerModelFitRes
from slbd.torch.model_utils import extract_submodel


class TorchServerModel:
    """Class is fully controlled by the framework and cannot be overridden by the model owner."""
    def __init__(
        self,
        model: torch.nn.Sequential,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ):
        self._model = model
        self.device = device
        self.first_server_layer = -1
        self._optimizer = None
        self._criterion = torch.nn.CrossEntropyLoss()
        self._per_parameter_data_points = defaultdict(int)
        self._metrics_dict = {}

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

    def _get_model_from_ins(self, ins):
        self.first_server_layer = ins.first_server_model_layer
        model = extract_submodel(self._model, ins.first_server_model_layer, -1)
        state_dict = shift_state_dict(ins.model_state_dict, -self.first_server_layer)
        model.load_state_dict(state_dict)
        return model

    def get_fit_result(self) -> Tuple[OrderedDict]:
        return self.get_train_results()

    def configure_fit(
        self,
        ins: ServerModelFitIns
    ) -> None:
        self._model = self._get_model_from_ins(ins)
        self._model.train()
        self._model.to(self.device)
        self._optimizer = torch.optim.SGD(self._model.parameters(), momentum=0.9, lr=0.01)

    def configure_evaluate(
        self,
        ins: ServerModelEvaluateIns
    ) -> None:
        self._model = self._get_model_from_ins(ins)

        self._model.eval()
        self._model.to(self.device)

    def train_and_compute_gradient(
        self,
        batches,
        offset,
    ):
        sizes = [b.shape[0] for b in batches["labels"]]
        embeddings = \
            torch.cat([torch.from_numpy(b) for b in batches["embeddings"]], dim=0).to(self.device)
        labels = torch.cat([torch.from_numpy(b) for b in batches["labels"]], dim=0).to(self.device)

        embeddings.requires_grad_(True)

        self._optimizer.zero_grad(set_to_none=True)
        logits = self._forward_sequence_of_layers(embeddings, offset)
        loss = self._criterion(logits, labels)
        loss.backward()
        self._optimizer.step()

        gradients = [g.detach().cpu().numpy() for g in embeddings.grad.split(sizes, dim=0)]
        return gradients

    def _forward_sequence_of_layers(self, data, start_idx, end_idx=None):
        if end_idx is None:
            remaining_layers = list(self._model.children())[start_idx:]
        else:
            remaining_layers = list(self._model.children())[start_idx:end_idx]

        x = data
        for idx, layer in enumerate(remaining_layers, start=start_idx+self.first_server_layer):
            x = layer(x)
            self._per_parameter_data_points[idx] += x.shape[0]
        return x

    def eval_get_logits(self, batches, offset):
        sizes = [b.shape[0] for b in batches["embeddings"]]
        embeddings = torch.cat([torch.from_numpy(b) for b in batches["embeddings"]], dim=0)

        with torch.no_grad():
            logits = self._forward_sequence_of_layers(embeddings.to(self.device), offset)

        logits = [l.detach().cpu().numpy() for l in logits.split(sizes, dim=0)]
        return logits

    def get_model(self):
        return self._model
