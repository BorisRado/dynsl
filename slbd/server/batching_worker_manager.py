import asyncio
from collections import defaultdict
from logging import INFO, WARNING
from typing import Callable, Dict, Tuple, Optional

import torch
import numpy as np
from flwr.common import log

from slbd.common import BatchDataBackward, BatchDataForward
from slbd.common.utils import measure_time
from slbd.server.client_layer_allocator import ClientLayerAllocator
from slbd.server.server_model.manager import ServerModelManager
from slbd.common.constants import SL_OPTIM_STATES, SL_WEIGHTS


class BatchingWorkerManager:
    """Encapsulates batching, computation and result distribution logic.

    This separates the batching and model-training related logic from the
    gRPC servicer so the servicer stays focused on stream handling and
    request/future plumbing.
    """

    def __init__(
        self,
        server_model_manager: ServerModelManager,
        client_layer_allocator: ClientLayerAllocator,
        num_concat_batches: int,
        request_group_timeout: float,
        sending_transform_fn: Optional[Callable],
        receiving_transform_fn: Optional[Callable],
        dynamic_layers_strategy: str,
        exchange_optimizer_states: Optional[bool],
        is_cuda_available: bool,
        event_loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        self.server_model_manager = server_model_manager
        self.client_layer_allocator = client_layer_allocator
        self._request_queues: Dict[Tuple[int, str], asyncio.Queue] = defaultdict(asyncio.Queue)
        self._num_concat_batches = num_concat_batches
        self.request_group_timeout = request_group_timeout
        self.sending_transform_fn = sending_transform_fn
        self.receiving_transform_fn = receiving_transform_fn
        self._dynamic_layers_strategy = dynamic_layers_strategy
        self._exchange_optimizer_states = exchange_optimizer_states
        self._is_cuda_available = is_cuda_available
        self._workers = {}
        # Use provided event loop if available (helps running in background
        # threads or simulation contexts). Fall back to the current running
        # loop in the caller's thread.
        self._loop = event_loop if event_loop is not None else asyncio.get_running_loop()

    def enqueue_and_get_future(
        self,
        client_id: str,
        data: BatchDataForward,
    ) -> BatchDataBackward:

        future = self._loop.create_future()
        server_id = self.server_model_manager.strategy \
            .route_client_request_to_server_model(client_id)

        self.client_layer_allocator.update_client_metrics(
            client_id=client_id,
            metrics=data.client_timing_metrics,
        )
        num_offloaded_layers = self.client_layer_allocator \
            .get_client_offloaded_layer_count(client_id)
        key = (num_offloaded_layers, server_id)

        if key not in self._workers or self._workers[key].done():
            # start the consumer process using the batching worker
            self._workers[key] = self._loop.create_task(self.run(*key))
        self._request_queues[key].put_nowait((data.data, future, client_id))

        return future

    async def run(self, num_offloaded_layers: int, server_id: str):
        q = self._request_queues[(num_offloaded_layers, server_id)]
        computation_tasks = []

        while True:
            kwargs_list, futures, client_ids = [], [], []

            # collect up to num_concat_batches (with timeout)
            try:
                start_time = self._loop.time()
                for _ in range(self._num_concat_batches):
                    remaining_time = self.request_group_timeout - (self._loop.time() - start_time)
                    if remaining_time <= 0:
                        break
                    received_data = await asyncio.wait_for(q.get(), remaining_time)
                    if received_data is None:
                        assert len(futures) == 0
                        return

                    kwargs, future, client_id = received_data
                    start_time = self._loop.time()
                    kwargs_list.append(kwargs)
                    futures.append(future)
                    client_ids.append(client_id)

            except asyncio.TimeoutError:
                if len(kwargs_list) == 0:
                    continue
                log(WARNING, f"Did not attain sufficient number of batches. Processing {len(kwargs_list)} {num_offloaded_layers} {server_id} {client_ids}...")

            # Start computation as a separate task and continue immediately
            computation_task = self._loop.create_task(
                self._run_computation_and_distribute_results(
                    kwargs_list, futures, client_ids, num_offloaded_layers, server_id
                )
            )
            computation_tasks.append(computation_task)

            # Clean up completed tasks to prevent memory leaks
            computation_tasks = [task for task in computation_tasks if not task.done()]

    def shutdown_all_workers(self):
        for key in self._request_queues:
            self._request_queues[key].put_nowait(None)
            self._loop.run_until_complete(self._workers[key])

    async def _run_computation_and_distribute_results(
        self, kwargs_list, futures, client_ids, num_offloaded_layers, server_id
    ):
        """Run computation and distribute results back to futures"""
        try:
            result = await self._run_computation(
                kwargs_list, client_ids, num_offloaded_layers, server_id
            )
            # distribute results back to futures
            for fut, res in zip(futures, result):
                assert not fut.done()
                fut.set_result(res)

        except Exception as e:
            # Set exception on all futures
            for fut in futures:
                assert not fut.done()
                fut.set_exception(e)

    async def _run_computation(self, kwargs_list, client_ids, num_offloaded_layers, server_id):

        for cid, client_kwargs in zip(client_ids, kwargs_list):
            client_parameters = client_kwargs.pop(SL_WEIGHTS, None)
            client_optimizer_states = client_kwargs.pop(SL_OPTIM_STATES, None)
            if client_parameters:
                log(INFO, "Server has received client parameters. Integrating them...")
                await self.server_model_manager \
                    .marshall_weight_integration_request(
                        server_id, client_parameters, cid, client_optimizer_states
                    )

        kwargs = {}
        for key in kwargs_list[0]:
            if isinstance(kwargs_list[0][key], np.ndarray):
                kwargs[key] = [torch.from_numpy(d[key]) for d in kwargs_list]
            else:
                tpls = [d[key] for d in kwargs_list]
                n_elements = len(tpls[0])
                kwargs[key] = tuple(
                    [torch.from_numpy(tpl[i]) for tpl in tpls]
                    for i in range(n_elements)
                )

        if self.receiving_transform_fn:
            kwargs["embeddings"] = [self.receiving_transform_fn(d) for d in kwargs["embeddings"]]

        with measure_time(self._is_cuda_available) as timer:
            grad_results = await self.server_model_manager.marshall_train_request(
                model_id=server_id,
                num_offloaded_layers=num_offloaded_layers,
                **kwargs
            )

        self.client_layer_allocator.update_server_metrics(num_offloaded_layers, timer.elapsed)

        results = []
        for (gradient, training_loss), client_id, received_data in zip(grad_results, client_ids, kwargs_list):
            if self.sending_transform_fn is not None:
                gradient = self.sending_transform_fn(gradient, received_data=received_data)

            diff_data = self.client_layer_allocator.get_post_reallocation_layer_difference(client_id)

            num_layers_diff, params, optimizer_states = 0, None, None
            if diff_data is not None:
                previous, num_layers_diff = diff_data
                log(INFO, "\tChange in offload strategy for %s. Difference in number of layers: %d",
                    client_id, num_layers_diff)
                assert self._dynamic_layers_strategy == "afl"

                if num_layers_diff > 0:
                    params, optimizer_states = await self.server_model_manager \
                        .extract_layer_weights_and_optimizer_states(
                            previous, previous+num_layers_diff, server_id, client_id,
                            extract_optimizer_states=self._exchange_optimizer_states
                        )

            results.append(BatchDataBackward(
                data=gradient,
                layers_diff=num_layers_diff,
                additional_layers_parameters=params,
                additional_optimizer_states=optimizer_states,
                training_loss=training_loss,
            ))

        return results
