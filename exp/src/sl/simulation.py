import copy
import random
from typing import List, Tuple

import torch
from flwr.server.client_manager import ClientManager
from flwr.common import (
    FitIns,
    GetPropertiesRes,
    parameters_to_ndarrays,
    ndarrays_to_parameters,
    FitRes,
    Status,
    Code
)

from slbd.server.server_model.torch_server_model import TorchServerModel
from slbd.client.model_trainer import ModelTrainer
from slbd.common.typing import ServerModelFitIns

from slbd.common.constants import (
    CLIENT_LAST_LAYER_IDX_KEY,
    CLIENT_NUM_RECEIVED_LAYERS,
    CLIENT_DATASET_SIZE,
    CLIENT_DEVICE_TYPE,
)


class SimulationClientProxy:
    def __init__(self, cid, client):
        self.cid = cid
        self.client = client

    def get_properties(self, config, **kwargs):
        return GetPropertiesRes(status=None, properties={
            CLIENT_DATASET_SIZE: self.client.get_trainloader_size(),
            CLIENT_DEVICE_TYPE: self.cid,
        })

    def fit(self, config, model_trainer):
        self.client.fit(config, model_trainer)


class SimulationClientManager(ClientManager):
    def __init__(self, clients):
        self.clients = {c.cid: c for c in clients}

    def num_available(self):
        return len(self.clients)

    def register(self, client):
        return True

    def unregister(self, client):
        return

    def all(self):
        return self.clients

    def wait_for(self, num_clients, timeout=None):
        return True

    def sample(self, num_clients, min_num_clients):
        _ = (min_num_clients, )
        available_cids = list(self.clients)
        sampled_cids = random.sample(available_cids, num_clients)

        return [self.clients[cid] for cid in sampled_cids]


class SimulationServerModelManager:
    def __init__(self, model, client_layer_allocator):
        self.model = model
        self.client_layer_allocator=client_layer_allocator
        self._training_models = {}

    def initialize_server_models(self, server_model_configs: List[ServerModelFitIns], is_training: bool):
        _ = (is_training,)
        for config in server_model_configs:
            server_model = TorchServerModel(copy.deepcopy(self.model))
            server_model.configure_fit(config)
            self._training_models[config.sid] = server_model

    def collect_server_fit_results(self):
        out = []
        for sid, model in self._training_models.items():
            res = model.get_fit_result()
            res.sid = sid
            res.forking_data = {}
            out.append(res)
        return out

    def get_num_server_models(self):
        return len(self._training_models)

    def get_server_model_proxy(self, sid):
        if sid in self._training_models:
            return SimulationServerModelProxy(sid, self._training_models[sid], self.client_layer_allocator)
        assert len(self._training_models) == 0, "Should not happen"
        return None


class SimulationServerModelProxy:
    def __init__(self, sid, server_model: TorchServerModel, client_layer_allocator):
        self.sid = sid
        self._server_model = server_model
        self.client_layer_allocator = client_layer_allocator
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, embeddings, labels, local_computation_time, tot_local_time):
        _ = (local_computation_time, tot_local_time)
        gradient, loss = self._server_model.train_and_compute_gradient(
            embeddings=embeddings.detach().clone().to(self.device),
            labels=labels.to(self.device),
            num_offloaded_layers=self.client_layer_allocator.get_client_offloaded_layer_count(self.sid)
        )
        control = None
        if hasattr(self.client_layer_allocator, "simulation_get_post_computation_reallocation"):
            control = self.client_layer_allocator \
                .simulation_get_post_computation_reallocation(self.sid)
        return gradient, control, loss


def _fit_client_private_head(instructions, client_proxy, whole_model, server_model_manager):
    config = instructions.config
    last_client_layer_idx = config.pop(CLIENT_LAST_LAYER_IDX_KEY)
    num_received_layers = config.pop(CLIENT_NUM_RECEIVED_LAYERS, last_client_layer_idx)
    optimizer_config = {k: v for k, v in config.items() if k.startswith("optim.")}

    model_trainer = ModelTrainer(
        whole_model=whole_model,
        last_layer_idx=last_client_layer_idx,
        num_received_layers=num_received_layers,
        init_parameters=parameters_to_ndarrays(instructions.parameters),
        server_model_proxy=server_model_manager.get_server_model_proxy(client_proxy.cid),
        optimizer_config=optimizer_config,
        is_training=True,
        sending_transform_fn=None,
        receiving_transform_fn=None,
        server_forking=False,
    )
    client_proxy.fit(config, model_trainer)
    trained_parameters, processed_data_points_dict, metrics = model_trainer.get_fit_results()
    combined_metrics = processed_data_points_dict | metrics
    dataset_size = model_trainer._dataset_size
    return client_proxy, FitRes(
        status=Status(code=Code.OK, message="Success"),
        parameters=ndarrays_to_parameters(trained_parameters),
        num_examples=dataset_size,
        metrics=combined_metrics,
    )


def fit_clients_simulation(
    client_instructions: List[Tuple[SimulationClientProxy, FitIns]],
    strategy,
    server_model_manager: SimulationServerModelManager,
    whole_model,
):

    assert server_model_manager.get_num_server_models() in {0, len(client_instructions)}, \
        "For now we assume that every client has its own classification head on the server - easier"

    results = [_fit_client_private_head(ins, proxy, whole_model, server_model_manager)
               for proxy, ins in client_instructions]
    return results
