import copy
import asyncio
from typing import Union, List, Optional, Tuple

import numpy as np
import torch

from slbd.server.server_model.torch_server_model import TorchServerModel
from slbd.server.server_model.manager.multiprocessing.model_worker import TaskType
from slbd.server.server_model.manager import BaseServerModelManager
from slbd.common.typing import ServerModelEvaluateIns, ServerModelFitIns, ServerModelFitRes
from slbd.torch.model_utils import extract_submodel_weights
from slbd.torch.sequential_model import FlexibleSequential


class MultithreadingServerModelManager(BaseServerModelManager):

    def __init__(
        self,
        model: FlexibleSequential,
        exp_config,
    ) -> None:
        super().__init__(model, )
        _ = (exp_config,)
        self.spanned_server_models = {}
        self._model_locks = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # data for enabling AFL integration of model weights
        # specifically, for every client we store the number of DP processed on the server
        # when a given layer forked
        self._afl_coord_data = {}

    async def marshall_train_request(self, model_id,  embeddings, labels, num_offloaded_layers):
        embeddings, labels, control = self._concat_data(embeddings, labels)
        if isinstance(embeddings, tuple):
            embeddings = tuple(e.to(self.device, non_blocking=True) for e in embeddings)
        else:
            embeddings = embeddings.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        async with self._model_locks[model_id]:
            gradient, loss = self.spanned_server_models[model_id].train_and_compute_gradient(
                embeddings, labels, num_offloaded_layers
            )
        return self._split_data(loss, gradient, control)

    def marshall_evaluation_request(self, model_id, embeddings, num_offloaded_layers):
        raise Exception("")
        return self._model_process_pool.marshall_request(model_id, (
            TaskType.EVALUATE,
            (embeddings, num_offloaded_layers)
        ))

    async def marshall_weight_integration_request(
        self,
        model_id: str,
        client_parameters: List[np.ndarray],
        cid: str,
        client_optimizer_states: Optional[List[np.ndarray]] = None,
    ):
        *client_parameters, num_dp_per_layer = client_parameters
        num_layers = len(num_dp_per_layer)
        # received layers have an absolute value
        if cid in self._afl_coord_data:
            received_layers = sorted(self._afl_coord_data[cid])[-num_layers:]
            num_server_dp_when_forked = {k: self._afl_coord_data[cid].pop(k) for k in received_layers}
        else:
            print(f"Warning -- {cid} not tracked. Assuming layer was sent at the beginning of the epoch.")
            initial_received_layer = self.base_model.meaningful_split_points[0]
            received_layers = range(initial_received_layer, initial_received_layer+num_layers)
            num_server_dp_when_forked = {k: 0 for k in received_layers}
        num_client_dp_per_layer = {k: v for k, v in zip(received_layers, num_dp_per_layer)}
        async with self._model_locks[model_id]:
            integration_model = self.spanned_server_models[model_id]
            num_current_server_dp = {}
            for layer_idx in num_server_dp_when_forked:
                num_current_server_dp[layer_idx] = \
                    integration_model.get_current_layer_data_points(layer_idx)
                integration_model.increase_per_layer_counter(
                    idx=layer_idx,
                    increase=num_client_dp_per_layer[layer_idx]
                )

            integration_model.integrate_client_parameters(
                parameters=client_parameters,
                received_layers=received_layers,
                client_weights=num_client_dp_per_layer,
                server_model_weights={idx: num_current_server_dp[idx] - num_server_dp_when_forked[idx] for idx in received_layers},
                optimizer_states=client_optimizer_states,
            )

    def collect_server_fit_results(
        self
    ) -> List[ServerModelFitRes]:
        out = []
        for sid, model in self.spanned_server_models.items():
            res: ServerModelFitRes = model.get_fit_result()
            res.sid = sid
            res.afl_data = self._afl_coord_data.copy() # simply set that all have the same AFL data
            out.append(res)
        self._afl_coord_data.clear()

        return out

    def end_round(self):
        self.spanned_server_models = {}

    def shutdown(self):
        self.spanned_server_models = {}

    def clean_resources(self):
        self.spanned_server_models = {}

    def initialize_server_models(
        self,
        configs: Union[List[ServerModelFitIns], List[ServerModelEvaluateIns]],
        is_training: bool,
    ):
        if len(configs) == 0:
            # no need for any server model
            return

        for config in configs:
            sm = TorchServerModel(self.base_model, device=self.device)
            if is_training:
                sm.configure_fit(config)
            else:
                sm.configure_evaluate(config)
            self.spanned_server_models[config.sid] = sm
            self._model_locks[config.sid] = asyncio.Lock()

    def get_global_model_params(self):
        return copy.deepcopy(self.base_model.state_dict())

    async def extract_layer_weights_and_optimizer_states(
        self,
        layer_from: int,
        layer_to: int,
        server_id: str,
        cid: str,
        extract_optimizer_states: bool = False
    ) -> Tuple[List[np.ndarray], Optional[List[np.ndarray]]]:
        """
        Extract layer weights and optionally optimizer states.

        Returns:
            Tuple of (parameters, optimizer_states)
        """
        model = self.spanned_server_models[server_id]
        shift = model.first_server_layer
        if cid not in self._afl_coord_data:
            self._afl_coord_data[cid] = {}

        async with self._model_locks[server_id]:
            for idx in range(layer_from, layer_to):
                self._afl_coord_data[cid][idx] = model._per_layer_data_points[idx]

            # Extract parameters
            params, optimizer_states = extract_submodel_weights(
                model.get_model(), layer_from-shift, layer_to-shift, model._optimizer, extract_optimizer_states
            )

        return params, optimizer_states

    def get_num_server_models(self):
        assert len(self.spanned_server_models) == len(self._model_locks)
        return len(self.spanned_server_models)
