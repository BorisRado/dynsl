import time
import asyncio
from logging import INFO, WARNING
from collections import defaultdict
from typing import AsyncIterable, Dict, Tuple, Optional, Callable

import numpy as np
import torch
from flwr.common import log
from hydra.utils import instantiate

from slbd.proto import server_model_pb2_grpc
from slbd.proto import server_model_pb2
from slbd.server.server_model.manager import ServerModelManager
from slbd.common.utils import measure_time
from slbd.common import BatchDataForward, BatchDataBackward, ControlCode, TimingMetrics
from slbd.common.parameter import ndarray_dict_to_bytes, bytes_to_ndarray_dict, ndarray_list_to_bytes
from slbd.server.client_layer_allocator import ClientLayerAllocator
from slbd.common.serde import (
    control_code_from_proto,
    control_code_to_proto,
    from_grpc_format,
    to_grpc_format,
    bytes_list_to_grpc_format
)


# pylint: disable=no-member
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
            client_parameters = client_kwargs.pop("client_model_weights", None)
            client_optimizer_states = client_kwargs.pop("removed_optimizer_states", None)
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
        for result_item, client_id, received_data in zip(grad_results, client_ids, kwargs_list):
            gradient, training_loss = result_item
            if self.sending_transform_fn is not None:
                gradient = self.sending_transform_fn(gradient, received_data=received_data)
            if isinstance(gradient, torch.Tensor):
                gradient = gradient.numpy()
            else:
                gradient = [g.numpy() for g in gradient]

            diff_data = self.client_layer_allocator.get_post_reallocation_layer_difference(client_id)

            num_layers_diff, params, optimizer_states = 0, [], []
            if diff_data is not None:
                previous, num_layers_diff = diff_data
                log(INFO, "\tChange in offload strategy for %s. Difference in number of layers: %d",
                    client_id, num_layers_diff)
                if num_layers_diff > 0 and self._dynamic_layers_strategy == "afl":
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

class ServerModelServicer(server_model_pb2_grpc.ServerModelServicer):

    def __init__(
        self,
        server_model_manager: ServerModelManager,
        client_layer_allocator: ClientLayerAllocator,
        num_concat_batches: int,
        dynamic_layers_strategy: str,
        request_group_timeout: float,
        sending_transform_fn_config: Optional[Callable],
        receiving_transform_fn_config: Optional[Callable],
        exchange_optimizer_states: bool
    ) -> None:
        super().__init__()
        self.server_model_manager = server_model_manager
        self.request_group_timeout = request_group_timeout
        assert dynamic_layers_strategy in {None, "cnt", "afl"}
        self._dynamic_layers_strategy = dynamic_layers_strategy
        self._is_cuda_available = torch.cuda.is_available()
        self.client_layer_allocator: ClientLayerAllocator = client_layer_allocator

        self._num_concat_batches = num_concat_batches
        self._loop = asyncio.get_running_loop()
        self.sending_transform_fn = instantiate(sending_transform_fn_config, _partial_=True)
        self.receiving_transform_fn = instantiate(receiving_transform_fn_config, _partial_=True)

        # Create batching worker instance (separates model/training logic)
        self._batching_worker = BatchingWorkerManager(
            server_model_manager=server_model_manager,
            client_layer_allocator=client_layer_allocator,
            num_concat_batches=num_concat_batches,
            request_group_timeout=request_group_timeout,
            sending_transform_fn=self.sending_transform_fn,
            receiving_transform_fn=self.receiving_transform_fn,
            dynamic_layers_strategy=dynamic_layers_strategy,
            is_cuda_available=self._is_cuda_available,
            exchange_optimizer_states=exchange_optimizer_states,
        )

    def _to_grpc(self, data: BatchDataBackward) -> server_model_pb2.BatchDataBackward:
        layer_params = None
        if data.additional_layers_parameters is not None:
            layer_params = bytes_list_to_grpc_format(
                ndarray_list_to_bytes(data.additional_layers_parameters)
            )

        send_data = {"grad": data.data}
        if data.additional_optimizer_states:
            assert len(data.additional_optimizer_states) > 0
            send_data["optim_states"] = ndarray_list_to_bytes(data.additional_optimizer_states)

        return server_model_pb2.BatchDataBackward(
            data=to_grpc_format(ndarray_dict_to_bytes(send_data)),
            control_code=control_code_to_proto(ControlCode.OK),
            layers_diff=data.layers_diff,
            additional_layers_parameters=layer_params,
            training_loss=data.training_loss,
        )

    async def StreamRequest(
        self,
        request_iterator: AsyncIterable[server_model_pb2.BatchDataForward],
        context,
    ) -> AsyncIterable[server_model_pb2.BatchDataBackward]:
        # get first request, which just states the cid
        _ = (context, )
        req = await request_iterator.__anext__()
        assert control_code_from_proto(req.control_code) == ControlCode.INIT_STREAM
        client_id = req.cid
        data_sending_time, data_received_time = -1, -1
        stream_start = time.perf_counter()

        async for req in request_iterator:
            if control_code_from_proto(req.control_code) in {ControlCode.DO_CLOSE_STREAM, ControlCode.ERROR_PROCESSING_STREAM}:
                break
            data_received_time = time.perf_counter()
            batch_data = bytes_to_ndarray_dict(from_grpc_format(req.data))
            # NOTE: if local_computation_time is -1, the throughput will be set to -1
            # thus, the response nbytes will only be considered for cases where the sent data
            # is the same as the returned data, at least in terms of size
            # we skip the iterations where sent data != received data as in this case the
            # communication is not symmetric. +512 to account for the grpc framing overhead
            # -- this is just an approximation!!!
            response_nbytes = batch_data["embeddings"].nbytes + batch_data["labels"].nbytes + 512
            timing_metrics = TimingMetrics(
                client_local_computation_time=req.local_computation_time,
                client_round_time=data_received_time - data_sending_time,
                client_tot_local_time=req.tot_local_time,
                response_length=response_nbytes
            ) if data_sending_time > 0 else None
            data_fwd = BatchDataForward(data=batch_data, is_training=req.training, client_timing_metrics=timing_metrics)
            future = self._batching_worker.enqueue_and_get_future(client_id, data_fwd)
            res = await future

            assert res is not None
            tmp = self._to_grpc(res)
            data_sending_time = time.perf_counter()
            yield tmp

        extra_info = ""
        if hasattr(self.client_layer_allocator, "_cid_to_device_type_mapping") and client_id in self.client_layer_allocator._cid_to_device_type_mapping:
            extra_info = self.client_layer_allocator._cid_to_device_type_mapping[client_id]
        if req.control_code == ControlCode.ERROR_PROCESSING_STREAM:
            extra_info += " ERROR!!!"
        log(INFO, f"[Server Profiling] StreamRequest client_id={client_id} total_time={time.perf_counter()-stream_start:.4f}s -- [device type: {extra_info}]")

        if hasattr(self.client_layer_allocator, "remove_client"):
            self.client_layer_allocator.remove_client(client_id)

        yield server_model_pb2.BatchDataBackward(
            control_code=control_code_to_proto(ControlCode.STREAM_CLOSED_OK)
        )
