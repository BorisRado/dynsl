import random
import asyncio
from typing import List, Tuple

import numpy as np
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

from slbd.client.model_trainer import ModelTrainer
from slbd.common import BatchDataForward, BatchDataBackward, LayerChangeInfo
from slbd.server.grpc.queue_server_model_servicer import BatchingWorkerManager


from exp.src.data.utils import get_dataloader
from exp.src.common.stochasticity import StatefulRng

from slbd.common.constants import (
    CLIENT_LAST_LAYER_IDX_KEY,
    CLIENT_NUM_RECEIVED_LAYERS,
    CLIENT_DATASET_SIZE,
    CLIENT_DEVICE_TYPE,
)


def torch_to_numpy(values_dict):
    for k, v in values_dict.items():
        if isinstance(v, torch.Tensor):
            values_dict[k] = v.detach().cpu().numpy()
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
            try:
                values_dict[k] = [a.detach().cpu().numpy() for a in v]
            except Exception as e:
                for a in v:
                    print(type(a))
                raise e
    return values_dict


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
    def __init__(
        self,
        strategy,
        server_model_manager,
        whole_model,
        client_layer_allocator,
        num_concat_batches: int,
        request_group_timeout: float,
        dynamic_layers_strategy=None,
    ):
        self.strategy = strategy
        self.server_model_manager = server_model_manager
        self.whole_model = whole_model
        self.client_layer_allocator = client_layer_allocator
        self.dynamic_layers_strategy = dynamic_layers_strategy
        self.event_loop = asyncio.new_event_loop()
        self._worker_manager = BatchingWorkerManager(
            server_model_manager=server_model_manager,
            client_layer_allocator=client_layer_allocator,
            num_concat_batches=num_concat_batches,
            request_group_timeout=request_group_timeout,
            sending_transform_fn=None,
            receiving_transform_fn=None,
            dynamic_layers_strategy=dynamic_layers_strategy,
            is_cuda_available=torch.cuda.is_available(),
            event_loop=self.event_loop,
        )

    def init_epoch(self, client_instructions):
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
                num_training_layers=last_client_layer_idx,
                num_received_layers=num_received_layers,
                init_parameters=parameters_to_ndarrays(ins.parameters),
                server_model_proxy=server_model_proxy,
                optimizer_config=optimizer_config,
                is_training=True,
                sending_transform_fn=None,
                receiving_transform_fn=None,
                dynamic_layers_strategy=self.dynamic_layers_strategy,
            )

            # build dataloader for this client so we can step through batches in lockstep
            client_obj = proxy.client
            batch_size = client_obj.batch_size
            dataloader = get_dataloader(client_obj.trainset, batch_size=batch_size)
            with stateful_rng:
                iterator = iter(dataloader)

            self._clients.append({
                "proxy": proxy,
                "cid": proxy.cid,
                "ins": ins,
                "trainer": model_trainer,
                "dataloader": dataloader,
                "iterator": iterator,
                "dataset_type": getattr(client_obj.trainset, "_dataset_type", "cv"),
                "rng": stateful_rng
            })

    def finalize_epoch(self):
        self._clients = None

    def shutdown(self):
        self._worker_manager.shutdown_all_workers()
        self.event_loop.close()

    async def run(self):
        # Process batches in rounds; in each round shuffle clients and process one batch per client
        clients = self._clients
        # compute number of batches per client (dataloader drop_last=True so lengths equal-ish)
        remaining = [len(c["dataloader"]) for c in clients]

        results = []

        while any(r > 0 for r in remaining):
            order = [i for i, r in enumerate(remaining) if r > 0]
            random.shuffle(order)
            futures = []
            client_outputs = []
            for idx in order:
                entry = clients[idx]
                try:
                    with entry["rng"]:
                        batch = next(entry["iterator"])
                        if entry["dataset_type"] == "cv":
                            payload, client_output = \
                                entry["trainer"].forward_pass(batch["img"], batch["label"])
                        else:
                            payload, client_output = \
                                entry["trainer"].forward_pass(
                                    input_vars=(batch["input_ids"], batch["attention_mask"]),
                                    labels=batch["labels"]
                                )
                except StopIteration:
                    remaining[idx] = 0
                    continue

                payload = torch_to_numpy(payload)
                del payload["local_computation_time"]
                del payload["tot_local_time"]

                data = BatchDataForward(data=payload, is_training=True, client_timing_metrics=None)
                future = self._worker_manager \
                    .enqueue_and_get_future(client_id=entry["cid"], data=data)
                futures.append(future)
                client_outputs.append(client_output)

            for idx, future, client_output in zip(order, futures, client_outputs):
                entry = clients[idx]
                trainer = entry["trainer"]
                backward_data: BatchDataBackward = await future
                gradient = backward_data.data
                num_layers_diff = backward_data.layers_diff
                assert isinstance(gradient, np.ndarray)
                gradient = torch.from_numpy(gradient)
                layer_change_info = LayerChangeInfo(
                    layers_diff=num_layers_diff,
                    additional_layers_parameters=backward_data.additional_layers_parameters
                )
                trainer.backward_pass(
                    gradient=gradient,
                    control=layer_change_info,
                    loss=backward_data.training_loss,
                    client_output=client_output
                )
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
        self,
        client_instructions: List[Tuple[SimulationClientProxy, FitIns]],
    ):

        self.init_epoch(client_instructions)
        results = self.event_loop.run_until_complete(self.run())
        return results
