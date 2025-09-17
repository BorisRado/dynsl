import os
import json
import time
import asyncio
from logging import INFO
from collections import defaultdict
from typing import AsyncIterable, Dict, Tuple

import torch
from flwr.common import log

from slbd.proto import server_model_pb2_grpc
from slbd.proto import server_model_pb2
from slbd.server.server_model.manager import ServerModelManager
from slbd.common import BatchDataForward, BatchDataBackward, ControlCode, TimingMetrics
from slbd.common.parameter import ndarray_dict_to_bytes, bytes_to_ndarray_dict, torch_list_to_bytes
from slbd.server.client_layer_allocator import ClientLayerAllocator
from slbd.common.serde import (
    control_code_from_proto,
    control_code_to_proto,
    from_grpc_format,
    to_grpc_format,
    bytes_list_to_grpc_format
)


import functools
import traceback
# debugging --- remove when everything works smoothly
def with_exception_logging(func):
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            traceback.print_exc()
            raise

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            traceback.print_exc()
            raise

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


# pylint: disable=no-member
class ServerModelServicer(server_model_pb2_grpc.ServerModelServicer):
    def __init__(
        self,
        server_model_manager: ServerModelManager,
        client_layer_allocator: ClientLayerAllocator,
        num_concat_batches: int,
        request_group_timeout,
    ) -> None:
        super().__init__()
        self.server_model_manager = server_model_manager
        self.request_group_timeout = request_group_timeout
        self._is_cuda_available = torch.cuda.is_available()
        self.client_layer_allocator: ClientLayerAllocator = client_layer_allocator
        self._request_queues: Dict[Tuple[int, str], asyncio.Queue] = defaultdict(asyncio.Queue)
        self._workers = {}
        self._model_locks = {}

        self._num_concat_batches = num_concat_batches
        self._loop = asyncio.get_running_loop()

    async def _compute_response(
        self,
        client_id: str,
        request: server_model_pb2.BatchDataForward,
        timing_metrics: TimingMetrics,
    ) -> BatchDataBackward:
        data = BatchDataForward(
            data = from_grpc_format(request.data),
            control_code=control_code_from_proto(request.control_code),
            is_training=request.training,
            local_computation_time=request.local_computation_time
        )
        future = self._loop.create_future()
        server_id = self.server_model_manager.strategy.route_client_request_to_server_model(client_id)
        if server_id not in self._model_locks:
            self._model_locks[server_id] = asyncio.Lock()

        self.client_layer_allocator.update_client_metrics(client_id, timing_metrics)
        num_offloaded_layers = self.client_layer_allocator.get_client_offloaded_layer_count(client_id)
        key = (num_offloaded_layers, server_id)

        if key not in self._workers or self._workers[key].done():
            # start the consumer process
            self._workers[key] = asyncio.create_task(self.batching_worker(*key))
        await self._request_queues[key].put((data, future))

        return await future

    async def batching_worker(self, num_offloaded_layers: int, server_id: str):
        q = self._request_queues[(num_offloaded_layers, server_id)]
        while True:
            batches, futures = [], []

            # collect up to num_concat_batches (with timeout)
            start_time = asyncio.get_event_loop().time()
            try:
                for _ in range(self._num_concat_batches):
                    remaining_time = self.request_group_timeout - (
                        asyncio.get_event_loop().time() - start_time
                    )
                    if remaining_time <= 0:
                        break
                    data, future = await asyncio.wait_for(q.get(), remaining_time)
                    batches.append(data)
                    futures.append(future)
            except asyncio.TimeoutError:
                if len(batches) == 0:
                    del self._request_queues[(num_offloaded_layers, server_id)]
                    return
                pass  # process what we have

            # run computation (in executor so loop isn’t blocked)
            result = await self._run_computation(batches, num_offloaded_layers, server_id)

            # distribute results back to futures
            for fut, res in zip(futures, result):
                fut.set_result(res)

    async def _run_computation(self, batches, num_offloaded_layers, server_id) -> None:

        kwargs_list = [bytes_to_ndarray_dict(b.data) for b in batches]
        kwargs = {key: [d[key] for d in kwargs_list] for key in kwargs_list[0]}
        with self._model_locks[server_id]:
            grad_results = self.server_model_manager.marshall_train_request(
                model_id=server_id,
                num_offloaded_layers=num_offloaded_layers,
                **kwargs
            )

        results = []
        for result_item in grad_results:
            gradient, training_loss = result_item
            num_layers_diff, params = 0, None

            results.append(BatchDataBackward(
                data=ndarray_dict_to_bytes(gradient),
                control_code=ControlCode.OK,
                layers_diff=num_layers_diff,
                additional_layers_parameters=torch_list_to_bytes(params),
                training_loss=training_loss,
            ))
        return results

    def _to_grpc(self, data: BatchDataBackward) -> server_model_pb2.BatchDataBackward:

        layer_params = bytes_list_to_grpc_format(data.additional_layers_parameters) \
            if data.additional_layers_parameters is not None else None
        return server_model_pb2.BatchDataBackward(
            data=to_grpc_format(data.data),
            control_code=control_code_to_proto(data.control_code),
            layers_diff=data.layers_diff,
            additional_layers_parameters=layer_params,
            training_loss=data.training_loss,
        )

    async def StreamRequest(
        self,
        request_iterator: AsyncIterable[server_model_pb2.BatchDataForward],
        context,
    ) -> AsyncIterable[server_model_pb2.BatchDataBackward]:
        _ = (context, )
        stream_start = time.time()
        req = await request_iterator.__anext__()
        assert control_code_from_proto(req.control_code) == ControlCode.INIT_STREAM
        client_id = req.cid
        data_sending_time, data_received_time, client_round_time = -1, -1, -1
        req_count = 0
        async for req in request_iterator:
            if control_code_from_proto(req.control_code) == ControlCode.DO_CLOSE_STREAM:
                break
            data_received_time = time.time()
            if data_sending_time > 0:
                client_round_time = data_received_time - data_sending_time
            timing_metrics = TimingMetrics(
                client_local_computation_time=req.local_computation_time,
                client_round_time=client_round_time,
            )
            req_count += 1
            res = await self._compute_response(client_id, req, timing_metrics)

            data_sending_time = time.time()
            if res is not None:
                tmp = self._to_grpc(res)
                yield tmp

        stream_end = time.time()
        log(INFO, f"[Server Profiling] StreamRequest client_id={client_id} total_time={stream_end-stream_start:.4f}s requests={req_count}")

        yield server_model_pb2.BatchDataBackward(
            control_code=control_code_to_proto(ControlCode.STREAM_CLOSED_OK)
        )
