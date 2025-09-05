import time
import asyncio
from logging import INFO
from typing import AsyncIterable, Dict

import torch
from flwr.common import log

from slbd.proto import server_model_pb2_grpc
from slbd.proto import server_model_pb2
from slbd.server.server_model.manager import ServerModelManager
from slbd.server.server_model.utils import ClientRequestGroup
from slbd.common import BatchDataForward, BatchDataBackward, ControlCode, TimingMetrics
from slbd.common.utils import measure_time
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


class EventWithReturnValue(asyncio.Event):
    def __init__(self):
        super().__init__()
        self.result = None

    def set_result(self, result: BatchDataBackward):
        self.result = result
        self.set()

    def get_result(self) -> BatchDataBackward:
        return self.result


# pylint: disable=no-member
class ServerModelServicer(server_model_pb2_grpc.ServerModelServicer):
    def __init__(
        self,
        server_model_manager: ServerModelManager,
        client_layer_allocator: ClientLayerAllocator,
        request_group_timeout: float,
        num_concat_batches: int,
    ) -> None:
        super().__init__()
        self.server_model_manager = server_model_manager
        self._lock = asyncio.Lock()
        self._is_cuda_available = torch.cuda.is_available()
        self.request_queues = {}
        self.client_layer_allocator: ClientLayerAllocator = client_layer_allocator
        self._outstanding_batches: Dict[int, ClientRequestGroup] = {}
        self._num_concat_batches = num_concat_batches
        self._requeset_group_timeout = request_group_timeout

    async def _compute_response(
        self,
        client_id: str,
        request: server_model_pb2.BatchDataForward,
        timing_metrics: TimingMetrics
    ) -> BatchDataBackward:
        data = BatchDataForward(
            data = from_grpc_format(request.data),
            control_code=control_code_from_proto(request.control_code),
            is_training=request.training,
            local_computation_time=request.local_computation_time
        )

        # store training metrics
        self.client_layer_allocator.update_client_metrics(client_id, timing_metrics)
        num_offloaded_layers = self.client_layer_allocator \
            .get_client_offloaded_layer_count(client_id)
        event = EventWithReturnValue()

        is_new = False
        async with self._lock:
            if (request_group := self._outstanding_batches.get(num_offloaded_layers)) is None:
                request_group = ClientRequestGroup(
                    num_offloaded_layers=num_offloaded_layers,
                    server_id=self.server_model_manager.strategy.route_client_request_to_server_model(client_id),
                    timeout=self._requeset_group_timeout,
                )
                self._outstanding_batches[num_offloaded_layers] = request_group
                is_new = True
            request_group.add(data, event, client_id)
            if request_group.get_num_batches() == self._num_concat_batches:
                request_group.mark_as_ready()

        if is_new:
            await request_group._is_ready.wait()
            async with self._lock:
                self._trigger_computation(request_group)
                del self._outstanding_batches[num_offloaded_layers]

        await event.wait()

        return event.get_result()

    @with_exception_logging
    def _trigger_computation(self, request_group: ClientRequestGroup) -> None:
        batches, events, client_ids = request_group.get_data()

        kwargs_list = [bytes_to_ndarray_dict(b.data) for b in batches]
        kwargs = {key: [d[key] for d in kwargs_list] for key in kwargs_list[0]}
        server_model = self.server_model_manager.get_server_model(request_group.server_id)
        with measure_time(self._is_cuda_available) as timer:
            if request_group.is_training:
                result = server_model.train_and_compute_gradient(
                    kwargs,
                    num_offloaded_layers=request_group.num_offloaded_layers
                )
            else:
                result = server_model.eval_get_logits(
                    kwargs,
                    num_offloaded_layers=request_group.num_offloaded_layers
                )
        computation_time = timer.elapsed
        self.client_layer_allocator.update_server_metrics(request_group.num_offloaded_layers, computation_time)

        self.client_layer_allocator.trigger_reallocation(client_ids)

        forking_data = {}
        for event, client_id, result_item in zip(events, client_ids, result):
            diff_data = self.client_layer_allocator.get_post_reallocation_layer_difference(client_id)
            num_layers_diff, params = 0, None
            if diff_data is not None:
                previous, num_layers_diff = diff_data
                log(INFO, "\tChange in offload strategy for %s. Difference in number of layers: %d",
                    client_id, num_layers_diff)
                if num_layers_diff > 0:
                    forking_data[client_id] = list(range(previous, previous+num_layers_diff))
                    params = self.server_model_manager \
                        .extract_layer_weights(previous, previous+num_layers_diff)

            event.set_result(BatchDataBackward(
                data=ndarray_dict_to_bytes(result_item),
                control_code=ControlCode.OK,
                layers_diff=num_layers_diff,
                additional_layers_parameters=torch_list_to_bytes(params)
            ))
        self.server_model_manager.create_fork(forking_data)

    def _to_grpc(self, data: BatchDataBackward) -> server_model_pb2.BatchDataBackward:

        layer_params = bytes_list_to_grpc_format(data.additional_layers_parameters) \
            if data.additional_layers_parameters is not None else None
        return server_model_pb2.BatchDataBackward(
            data=to_grpc_format(data.data),
            control_code=control_code_to_proto(data.control_code),
            layers_diff=data.layers_diff,
            additional_layers_parameters=layer_params,
        )

    async def StreamRequest(
        self,
        request_iterator: AsyncIterable[server_model_pb2.BatchDataForward],
        context,
    ) -> AsyncIterable[server_model_pb2.BatchDataBackward]:
        _ = (context, )
        req = await request_iterator.__anext__()
        assert control_code_from_proto(req.control_code) == ControlCode.INIT_STREAM
        client_id = req.cid
        data_sending_time, data_received_time, client_round_time = -1, -1, -1
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
            if client_round_time > 0:
                assert timing_metrics.client_local_computation_time < timing_metrics.client_round_time
            res = await self._compute_response(client_id, req, timing_metrics)

            data_sending_time = time.time()
            if res is not None:
                yield self._to_grpc(res)

        # async with self._lock:
        #     self.server_model_manager.strategy.mark_client_as_done(client_id)
        yield server_model_pb2.BatchDataBackward(
            control_code=control_code_to_proto(ControlCode.STREAM_CLOSED_OK)
        )
