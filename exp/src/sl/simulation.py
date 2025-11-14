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
from exp.src.data.utils import get_dataloader
from exp.src.common.stochasticity import StatefulRng

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
        self.client_layer_allocator = client_layer_allocator
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

    def get_server_model_proxy(self, sid, client_cid):
        if sid in self._training_models:
            return SimulationServerModelProxy(client_cid, self._training_models[sid], self.client_layer_allocator)
        assert len(self._training_models) == 0, "Should not happen"
        return None


class SimulationServerModelProxy:
    def __init__(self, cid, server_model: TorchServerModel, client_layer_allocator):
        self.cid = cid
        self._server_model = server_model
        self.client_layer_allocator = client_layer_allocator
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, embeddings, labels, local_computation_time, tot_local_time):
        _ = (local_computation_time, tot_local_time)
        gradient, loss = self._server_model.train_and_compute_gradient(
            embeddings=embeddings.detach().clone().to(self.device),
            labels=labels.to(self.device),
            num_offloaded_layers=self.client_layer_allocator.get_client_offloaded_layer_count(self.cid)
        )
        control = None
        if hasattr(self.client_layer_allocator, "simulation_get_post_computation_reallocation"):
            control = self.client_layer_allocator \
                .simulation_get_post_computation_reallocation(self.cid)
        return gradient, control, loss


class SLExecutor:
    """Class used in simulations for:
    1. Routing client requests to the appropriate server model. Which model should
    process a client request can be obtained via `strategy.route_client_request_to_server_model`
    2. Synchronize client requests. That is, somehow the class should receive batches from
    every client, and then process them on the server in a random order. This is in order
    to ensure that all clients process their dataset with the same speed (i.e., it should
    happend that one client finishes processing its dataset when some other client is only at
    the beginning). You can assume that every client has the same dataset size
    """
    def __init__(self, client_instructions, strategy, server_model_manager, whole_model):
        self.client_instructions = client_instructions
        self.strategy = strategy
        self.server_model_manager = server_model_manager
        self.whole_model = whole_model

        # Prepare per-client model trainers and dataloaders
        self._clients = []
        for proxy, ins in client_instructions:
            config = ins.config
            seed = proxy.client.seed + config.get("epoch_idx")
            last_client_layer_idx = config.get(CLIENT_LAST_LAYER_IDX_KEY)
            num_received_layers = config.get(CLIENT_NUM_RECEIVED_LAYERS, last_client_layer_idx)
            optimizer_config = {k: v for k, v in config.items() if k.startswith("optim.")}

            try:
                sid = self.strategy.route_client_request_to_server_model(proxy.cid)
                server_model_proxy = self.server_model_manager.get_server_model_proxy(sid, proxy.cid)
            except:
                server_model_proxy = None
            stateful_rng = StatefulRng(seed)

            model_trainer = ModelTrainer(
                whole_model=self.whole_model,
                last_layer_idx=last_client_layer_idx,
                num_received_layers=num_received_layers,
                init_parameters=parameters_to_ndarrays(ins.parameters),
                server_model_proxy=server_model_proxy,
                optimizer_config=optimizer_config,
                is_training=True,
                sending_transform_fn=None,
                receiving_transform_fn=None,
                server_forking=False,
            )

            # build dataloader for this client so we can step through batches in lockstep
            client_obj = proxy.client
            batch_size = client_obj.batch_size
            dataloader = get_dataloader(client_obj.trainset, batch_size=batch_size)
            with stateful_rng:
                iterator = iter(dataloader)

            self._clients.append({
                "proxy": proxy,
                "ins": ins,
                "trainer": model_trainer,
                "dataloader": dataloader,
                "iterator": iterator,
                "dataset_type": getattr(client_obj.trainset, "_dataset_type", "cv"),
                "rng": stateful_rng
            })

    def run(self):
        # Process batches in rounds; in each round shuffle clients and process one batch per client
        clients = self._clients
        # compute number of batches per client (dataloader drop_last=True so lengths equal-ish)
        remaining = [len(c["dataloader"]) for c in clients]

        results = []

        while any(r > 0 for r in remaining):
            order = [i for i, r in enumerate(remaining) if r > 0]
            random.shuffle(order)
            for idx in order:
                entry = clients[idx]
                try:
                    with entry["rng"]:
                        batch = next(entry["iterator"])
                except StopIteration:
                    remaining[idx] = 0
                    continue

                # call trainer with batch in the same way MOTorchClient does
                if entry["dataset_type"] == "cv":
                    entry["trainer"](batch["img"], batch["label"])
                else:
                    entry["trainer"](input_vars=(batch["input_ids"], batch["attention_mask"]), labels=batch["labels"])

                remaining[idx] -= 1

        # collect results
        for entry in clients:
            trained_parameters, processed_data_points_dict, metrics = entry["trainer"].get_fit_results()
            combined_metrics = processed_data_points_dict | metrics
            dataset_size = entry["trainer"]._dataset_size
            results.append(
                (
                    entry["proxy"],
                    FitRes(
                        status=Status(code=Code.OK, message="Success"),
                        parameters=ndarrays_to_parameters(trained_parameters),
                        num_examples=dataset_size,
                        metrics=combined_metrics,
                    )
                )
            )

        return results



def fit_clients_simulation(
    client_instructions: List[Tuple[SimulationClientProxy, FitIns]],
    strategy,
    server_model_manager: SimulationServerModelManager,
    whole_model,
):

    assert server_model_manager.get_num_server_models() in {0, 1, len(client_instructions)}

    # Shared single server model: process client batches in randomized interleaved order
    executor = SLExecutor(client_instructions, strategy, server_model_manager, whole_model)
    results = executor.run()
    return results
