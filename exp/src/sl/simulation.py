import random
import asyncio
import threading
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

from slbd.client.model_trainer import ModelTrainer
from exp.src.data.utils import get_dataloader
from exp.src.common.stochasticity import StatefulRng

from slbd.server.grpc.queue_server_model_servicer import BatchingWorkerManager
from slbd.server.server_model.manager.multithreading_server_model_manager import MultithreadingServerModelManager

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


class SimulationServerModelProxy:
    def __init__(self, cid, servicer):
        # Proxy that delegates incoming client requests to the servicer
        self.cid = cid
        self._servicer = servicer

    def __call__(self, embeddings, labels, local_computation_time, tot_local_time, client_model_weights):
        # Delegate to the servicer which simulates the gRPC servicer logic.
        return self._servicer.handle_request(
            client_id=self.cid,
            embeddings=embeddings,
            labels=labels,
            local_computation_time=local_computation_time,
            tot_local_time=tot_local_time,
            client_model_weights=client_model_weights,
        )


class SimulationServicer:
    """Simulates ServerModelServicer behavior for use in simulations.

    Responsibilities:
    - route client requests to the correct server model using the strategy
    - obtain the client's current offloaded layer count from the allocator
    - call the underlying TorchServerModel and return gradient/control/loss
    """

    def __init__(
        self,
        server_model_manager: MultithreadingServerModelManager,
        client_layer_allocator,
        strategy,
        num_concat_batches: int,
        request_group_timeout: float,
        send_server_parameters_when_forking: bool = False,
        sending_transform_fn=None,
        receiving_transform_fn=None,
    ):
        self.server_model_manager = server_model_manager
        self.client_layer_allocator = client_layer_allocator
        self.strategy = strategy

        # Create the request queues and batching worker to reuse the same
        # batching/computation logic used by the gRPC servicer. The worker
        # will run in a dedicated asyncio event loop on a background thread.
        self._request_queues = {}
        self._workers = {}

        # background event loop for the batching worker(s)
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._start_loop, daemon=True)
        self._thread.start()

        # instantiate batching worker manager (it will use queues created in the loop)
        self._batching_worker = BatchingWorkerManager(
            server_model_manager=self.server_model_manager,
            client_layer_allocator=self.client_layer_allocator,
            request_queues=self._request_queues,
            num_concat_batches=num_concat_batches,
            request_group_timeout=request_group_timeout,
            sending_transform_fn=sending_transform_fn,
            receiving_transform_fn=receiving_transform_fn,
            send_server_parameters_when_forking=send_server_parameters_when_forking,
            is_cuda_available=torch.cuda.is_available(),
            event_loop=self._loop,
        )

    def _start_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    async def _create_queue_and_start_worker(self, key):
        # create the asyncio.Queue in this loop and start the worker task
        q = asyncio.Queue()
        self._request_queues[key] = q
        self._workers[key] = self._loop.create_task(self._batching_worker.run(*key))
        return True

    async def _create_and_put(self, key, kwargs, client_id):
        # create a future in the loop, push the tuple into the queue and wait
        fut = asyncio.get_running_loop().create_future()
        await self._request_queues[key].put((kwargs, fut, client_id))
        return await fut

    def handle_request(self, client_id, embeddings, labels, local_computation_time, tot_local_time, client_model_weights):
        # Determine which server model should handle this client
        sid = self.strategy.route_client_request_to_server_model(client_id)

        num_offloaded_layers = self.client_layer_allocator.get_client_offloaded_layer_count(client_id)

        # Convert tensors to cpu numpy arrays (batching worker expects numpy)
        emb_np = embeddings.detach().cpu().numpy()
        lbl_np = labels.detach().cpu().numpy() if labels is not None else None

        payload = {"embeddings": emb_np}
        if lbl_np is not None:
            payload["labels"] = lbl_np

        key = (num_offloaded_layers, sid)

        # ensure the queue and worker are created in the background loop
        if key not in self._workers:
            asyncio.run_coroutine_threadsafe(self._create_queue_and_start_worker(key), self._loop).result()

        # enqueue the request and wait for the result (future created in the loop)
        concurrent = asyncio.run_coroutine_threadsafe(self._create_and_put(key, payload, client_id), self._loop)
        res = concurrent.result()
        # decode gradient bytes back to numpy and then to torch
        grad_dict = res.data
        # assume single tensor gradient or list; try to convert accordingly
        if isinstance(grad_dict, dict) and len(grad_dict) == 1:
            # pick first entry
            g = next(iter(grad_dict.values()))
            gradient = torch.from_numpy(g)
        else:
            # fallback: pack as tuple/list of tensors
            if isinstance(grad_dict, dict):
                gradient = {k: torch.from_numpy(v) for k, v in grad_dict.items()}
            else:
                gradient = torch.from_numpy(grad_dict)

        training_loss = res.training_loss

        control = None
        if hasattr(self.client_layer_allocator, "simulation_get_post_computation_reallocation"):
            control = self.client_layer_allocator.simulation_get_post_computation_reallocation(client_id)

        return gradient, control, training_loss


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
    def __init__(self, client_instructions, whole_model, servicer):

        self._clients = []
        for proxy, ins in client_instructions:
            config = ins.config
            seed = proxy.client.seed + config.get("epoch_idx")
            last_client_layer_idx = config.get(CLIENT_LAST_LAYER_IDX_KEY)
            num_received_layers = config.get(CLIENT_NUM_RECEIVED_LAYERS, last_client_layer_idx)
            optimizer_config = {k: v for k, v in config.items() if k.startswith("optim.")}

            stateful_rng = StatefulRng(seed)
            model_trainer = ModelTrainer(
                whole_model=whole_model,
                num_training_layers=last_client_layer_idx,
                num_received_layers=num_received_layers,
                init_parameters=parameters_to_ndarrays(ins.parameters),
                server_model_proxy=SimulationServerModelProxy(proxy.cid, servicer),
                optimizer_config=optimizer_config,
                is_training=True,
                sending_transform_fn=None,
                receiving_transform_fn=None,
                decrease_layers_strategy=None, # TODO
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
        # Run one dedicated thread per client so they can enqueue requests in
        # parallel. This avoids the timeout problem when the batching worker is
        # configured with num_concat_batches > 1 and expects multiple
        # concurrent requests.
        clients = self._clients

        thread_exceptions = []

        def _client_worker(entry):
            try:
                dataloader = entry["dataloader"]
                rng = entry["rng"]
                trainer = entry["trainer"]
                dataset_type = entry["dataset_type"]

                # iterate through the dataloader for this client and call the
                # trainer for every batch. Using the client's StatefulRng
                # context keeps behaviour deterministic per-client.
                with rng:
                    for batch in dataloader:
                        if dataset_type == "cv":
                            trainer(batch["img"], batch["label"])
                        else:
                            trainer(input_vars=(batch["input_ids"], batch["attention_mask"]), labels=batch["labels"])
            except Exception as e:
                # store the first exception to re-raise in the main thread
                thread_exceptions.append(e)

        threads = []
        for entry in clients:
            t = threading.Thread(target=_client_worker, args=(entry,), daemon=False)
            t.start()
            threads.append(t)

        # wait for all clients to finish
        for t in threads:
            t.join()

        # if any thread raised, propagate the first exception
        if thread_exceptions:
            raise thread_exceptions[0]

        # collect results (same as before)
        results = []
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
    whole_model,
    servicer,
):

    # Shared single server model: process client batches in randomized interleaved order
    executor = SLExecutor(client_instructions, whole_model, servicer)
    results = executor.run()
    return results
