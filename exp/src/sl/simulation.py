import copy
import random
import asyncio
from typing import List, Tuple, Optional

import uvloop
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
from slbd.common import BatchDataForward, BatchDataBackward, LayerChangeInfo, SocketTimingMetrics
from slbd.server.batching_worker_manager import BatchingWorkerManager
from slbd.torch.model_utils import param_list_to_state_dict

from exp.src.data.utils import get_dataloader
from exp.src.common.stochasticity import StatefulRng

from slbd.common.constants import (
    CLIENT_LAST_LAYER_IDX_KEY,
    CLIENT_NUM_RECEIVED_LAYERS,
    CLIENT_DATASET_SIZE,
    CLIENT_DEVICE_TYPE,
    SL_BACKWARD_TIME,
    SL_PER_LAYER_FORWARD_TIMES,
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
        training_speed_imbalance_value: Optional[float],
        end_epoch_weight_integration_strategy,
        client_optim_state_resetting_frequency=-1,
        client_clf_head = None,
        client_exchange_optimizer_states=None,
        server_exchange_optimizer_states=None,
    ):
        self.strategy = strategy
        self.server_model_manager = server_model_manager
        self.whole_model = whole_model
        self.client_layer_allocator = client_layer_allocator
        self.client_clf_head = client_clf_head
        self.client_exchange_optimizer_states = client_exchange_optimizer_states
        self.client_optim_state_resetting_frequency = client_optim_state_resetting_frequency
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        self.event_loop = asyncio.new_event_loop()
        self._no_metrics = SocketTimingMetrics(
            uplink_throughput_Mbps=1e8,
            downlink_throughput_Mbps=1e8,
            per_layer_forward_times=[],
            tot_backward_time=-1
        )
        self.training_speed_imbalance_value = training_speed_imbalance_value
        self._end_epoch_weight_integration_strategy = end_epoch_weight_integration_strategy
        self._worker_manager = BatchingWorkerManager(
            server_model_manager=server_model_manager,
            client_layer_allocator=client_layer_allocator,
            num_concat_batches=num_concat_batches,
            request_group_timeout=request_group_timeout,
            sending_transform_fn=None,
            receiving_transform_fn=None,
            exchange_optimizer_states=server_exchange_optimizer_states,
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

            init_parameters = parameters_to_ndarrays(ins.parameters)
            if self.client_clf_head is not None:
                if isinstance(self.client_clf_head, dict):
                    nclf_params = len(list(self.client_clf_head[num_received_layers].parameters()))
                else:
                    nclf_params = len(list(self.client_clf_head.parameters()))
                init_parameters, clf_parameters = init_parameters[:-nclf_params], init_parameters[-nclf_params:]

            model_trainer = ModelTrainer(
                whole_model=self.whole_model,
                num_received_layers=num_received_layers,
                init_parameters=init_parameters,
                server_model_proxy=server_model_proxy,
                optimizer_config=optimizer_config,
                is_training=True,
                sending_transform_fn=None,
                receiving_transform_fn=None,
                exchange_optimizer_states=self.client_exchange_optimizer_states,
            )
            if self.client_clf_head is not None:
                if isinstance(self.client_clf_head, dict):
                    tmp_head = self.client_clf_head[num_received_layers]
                else:
                    tmp_head = self.client_clf_head
                assert clf_parameters[-1].ndim == 1
                client_clf_head = copy.deepcopy(tmp_head)
                state_dict = param_list_to_state_dict(clf_parameters, client_clf_head)
                client_clf_head.load_state_dict(state_dict, strict=True)
                if torch.cuda.is_available():
                    client_clf_head.cuda()
                model_trainer._separate_client_clf_head = client_clf_head

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

    def collect_results(self, clients):
        # collect results
        results = []
        for entry in clients:
            trained_parameters, processed_data_points_dict, metrics = entry["trainer"].get_fit_results()
            combined_metrics = processed_data_points_dict | metrics
            if entry["trainer"]._separate_client_clf_head is not None:
                for _, param in entry["trainer"]._separate_client_clf_head.state_dict().items():
                    trained_parameters.append(param.detach().cpu().numpy())
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

    async def run(self):
        # Process batches in rounds; in each round shuffle clients and process one batch per client
        clients = self._clients
        # compute number of batches per client (dataloader drop_last=True so lengths equal-ish)
        remaining = [len(c["dataloader"]) for c in clients]

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

                if payload is None:
                    continue

                del payload[SL_BACKWARD_TIME]
                del payload[SL_PER_LAYER_FORWARD_TIMES]

                data = BatchDataForward(data=payload, is_training=True, client_timing_metrics=self._no_metrics)
                future = self._worker_manager \
                    .enqueue_and_get_future(client_id=entry["cid"], data=data)
                futures.append(future)
                client_outputs.append(client_output)

            for idx, future, client_output in zip(order, futures, client_outputs):
                entry = clients[idx]
                trainer = entry["trainer"]
                backward_data: BatchDataBackward = await future
                gradient = backward_data.data

                layer_change_info = None
                if backward_data.layers_diff != 0:
                    layer_change_info = LayerChangeInfo(
                        layers_diff=backward_data.layers_diff,
                        additional_layers_parameters=backward_data.additional_layers_parameters,
                        additional_optimizer_states=backward_data.additional_optimizer_states
                    )
                trainer.backward_pass(
                    gradient=gradient,
                    control=layer_change_info,
                    loss=backward_data.training_loss,
                    client_output=client_output
                )
                remaining[idx] -= 1

        return self.collect_results(clients)

    async def _iterate_client_fw_bck_pass(self, clients, sampled_client_idx, remaining):
        entry = clients[sampled_client_idx]
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
            remaining[sampled_client_idx] = 0
            return

        if payload is None:
            return

        del payload[SL_BACKWARD_TIME]
        del payload[SL_PER_LAYER_FORWARD_TIMES]

        data = BatchDataForward(data=payload, is_training=True, client_timing_metrics=self._no_metrics)
        backward_data: BatchDataBackward = await self._worker_manager \
            .enqueue_and_get_future(client_id=entry["cid"], data=data)
        gradient = backward_data.data
        layer_change_info = None
        if backward_data.layers_diff != 0:
            layer_change_info = LayerChangeInfo(
                layers_diff=backward_data.layers_diff,
                additional_layers_parameters=backward_data.additional_layers_parameters,
                additional_optimizer_states=backward_data.additional_optimizer_states
            )
        entry["trainer"].backward_pass(
            gradient=gradient,
            control=layer_change_info,
            loss=backward_data.training_loss,
            client_output=client_output
        )
        remaining[sampled_client_idx] -= 1

    async def run_wo_concatenation(self):
        # has lower memory usage

        # Process batches in rounds; in each round shuffle clients and process one batch per client
        clients = self._clients
        # compute number of batches per client (dataloader drop_last=True so lengths equal-ish)
        remaining = [len(c["dataloader"]) for c in clients]

        iteration = 0
        while any(r > 0 for r in remaining):
            order = [i for i, r in enumerate(remaining) if r > 0]
            random.shuffle(order)
            for idx in order:
                await self._iterate_client_fw_bck_pass(clients, idx, remaining)

            iteration += 1
            if self.client_optim_state_resetting_frequency > 0 and iteration % self.client_optim_state_resetting_frequency == 0:
                # reset all the client optimizers
                print("Resetting optimizer states")
                for client in clients:
                    client["trainer"]._optimizer.state.clear()
        return self.collect_results(clients)

    async def run_stochastic_order(self):
        # Stochastic order: sample a client each step according to a per-client "speed".
        clients = self._clients
        n_clients = len(clients)
        remaining = [len(c["dataloader"]) for c in clients]

        num_clients = len(clients)
        sigma = float(self.training_speed_imbalance_value)
        mu = -0.5 * sigma * sigma
        speeds = np.random.lognormal(mean=mu, sigma=sigma, size=num_clients)
        speeds = speeds / float(speeds.mean())  # normalize so mean speed == 1.0

        iter_since_sampling = [3] * len(clients)
        results = []
        while any(r > 0 for r in remaining):
            active = [i for i, r in enumerate(remaining) if r > 0]
            weights = [
                speeds[i] * 3 if iter_since_sampling[i] > 2 * n_clients
                else 0. if (iter_since_sampling[i] < 2 and len(active) > 4)
                else speeds[i]
                for i in active
            ]
            assert sum(weights) > 0
            sampled_client_idx = random.choices(active, weights=weights, k=1)[0]
            for idx in range(n_clients):
                iter_since_sampling[idx] += 1
            iter_since_sampling[sampled_client_idx] = 0
            await self._iterate_client_fw_bck_pass(clients, sampled_client_idx, remaining)
            if remaining[sampled_client_idx] == 0 and self._end_epoch_weight_integration_strategy == "immediate":
                results.append(
                    await self._handle_early_finishing_client(clients[sampled_client_idx])
                )
        if self._end_epoch_weight_integration_strategy == "immediate":
            print("returning previously collected results")
            return results
        return self.collect_results(clients)

    async def _handle_early_finishing_client(self, entry):
        # immediately extract the weights and send them to the server
        trained_parameters, processed_data_points_dict, metrics = entry["trainer"].get_fit_results()
        combined_metrics = processed_data_points_dict | metrics
        print("Client has finished training!", len(entry["trainer"]._model))
        assert entry["trainer"]._separate_client_clf_head is None

        dataset_size = entry["trainer"]._dataset_size

        n_late_params = 0 # "late" parameters will be aggregated at the end of the epoch as usual
        for layer, _ in zip(self.whole_model, range(self.whole_model.meaningful_split_points[0])):
            n_late_params += len(list(layer.parameters()))

        trained_parameters, immediate_integration_parameters = \
            trained_parameters[:n_late_params], trained_parameters[n_late_params:]
        nl_end = len([k for k in combined_metrics if k.startswith("_proc")])
        extract_keys = [f"_proc{idx}" for idx in range(self.whole_model.meaningful_split_points[0], nl_end)]
        send_dps_dict = {k: combined_metrics.pop(k) for k in extract_keys}

        layer_idxs = sorted(int(k[len("_proc"):]) for k in send_dps_dict)
        proc_dp = np.array([send_dps_dict[f"_proc{idx}"] for idx in layer_idxs])
        immediate_integration_parameters.append(proc_dp)
        await self._worker_manager.immediately_integrate_client_parameters(
            cid=entry["cid"],
            client_parameters=[torch.from_numpy(t) for t in immediate_integration_parameters],
        )
        return (
            entry["proxy"],
            FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=ndarrays_to_parameters(trained_parameters),
                num_examples=dataset_size,
                metrics=combined_metrics,
            )
        )


    async def run_no_backward_wo_concatenation(self):
        # has lower memory usage

        # Process batches in rounds; in each round shuffle clients and process one batch per client
        clients = self._clients
        # compute number of batches per client (dataloader drop_last=True so lengths equal-ish)
        remaining = [len(c["dataloader"]) for c in clients]

        while any(r > 0 for r in remaining):
            order = [i for i, r in enumerate(remaining) if r > 0]
            random.shuffle(order)
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

                if payload is None:
                    continue

                del payload[SL_BACKWARD_TIME]
                del payload[SL_PER_LAYER_FORWARD_TIMES]

                data = BatchDataForward(data=payload, is_training=True, client_timing_metrics=self._no_metrics)
                await self._worker_manager \
                    .enqueue_and_get_future(client_id=entry["cid"], data=data)
                remaining[idx] -= 1

        return self.collect_results(clients)

    def fit_clients_simulation(
        self,
        has_batch_concatenation,
        client_instructions: List[Tuple[SimulationClientProxy, FitIns]],
    ):

        self.init_epoch(client_instructions)
        if self.training_speed_imbalance_value is not None:
            results = self.event_loop.run_until_complete(self.run_stochastic_order())
        elif self.client_clf_head is not None and not has_batch_concatenation:
            print("Running SFSL")
            results = self.event_loop.run_until_complete(self.run_no_backward_wo_concatenation())
        elif self.client_clf_head is not None and has_batch_concatenation:
            raise Exception("Still need to implement this")
        elif has_batch_concatenation:
            results = self.event_loop.run_until_complete(self.run())
        else:
            results = self.event_loop.run_until_complete(self.run_wo_concatenation())
        torch.cuda.empty_cache()
        return results
