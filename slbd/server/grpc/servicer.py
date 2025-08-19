from typing import AsyncIterable
import asyncio

from slbd.proto import server_model_pb2_grpc
from slbd.proto import server_model_pb2
from slbd.server.server_model.manager import ServerModelManager
from slbd.server.server_model.utils import ClientRequestGroup
from slbd.common import BatchDataForward, BatchDataBackward, ControlCode, TimingMetrics
from slbd.common.parameter import ndarray_dict_to_bytes, bytes_to_ndarray_dict, torch_list_to_bytes
from slbd.server.layer_allocator import LayerAllocator
from slbd.common.serde import (
    control_code_from_proto,
    control_code_to_proto,
    from_grpc_format,
    to_grpc_format,
    bytes_list_to_grpc_format
)


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
        client_layer_allocator: LayerAllocator
    ) -> None:
        super().__init__()
        self.server_model_manager = server_model_manager
        self._lock = asyncio.Lock()
        self.request_queues = {}
        self.client_layer_allocator: LayerAllocator = client_layer_allocator

    async def _compute_response(
        self,
        cid: str,
        request: server_model_pb2.BatchDataForward
    ) -> BatchDataBackward:
        data = BatchDataForward(
            data = from_grpc_format(request.data),
            control_code=control_code_from_proto(request.control_code),
            is_training=request.training,
            local_computation_time=request.local_computation_time
        )
        timing_metrics = TimingMetrics(
            client_local_computation_time=data.local_computation_time,
            client_round_time=-1,
            server_computation_time=-1,
        )

        # store training metrics
        self.client_layer_allocator.update_metrics(cid, timing_metrics)
        event = EventWithReturnValue()
        async with self._lock:
            request_group, callback_fn = self.server_model_manager.strategy.route_client_request(
                cid=cid,
            )
            request_group.add(data, event, cid)
            is_new = request_group.is_new()
            self.server_model_manager.strategy.mark_ready_requests()

        if is_new:
            await request_group._is_ready.wait()
            self._trigger_computation(request_group)
            if callback_fn:
                callback_fn()
        await event.wait()

        return event.get_result()

    def _trigger_computation(self, request_group: ClientRequestGroup):
        batches, events, client_ids = request_group.get_data()
        assert len(batches) == 1
        offset = self.client_layer_allocator.get_client_offset_from_base(client_ids[0])

        kwargs_list = [bytes_to_ndarray_dict(b.data) for b in batches]
        kwargs = {key: [d[key] for d in kwargs_list] for key in kwargs_list[0]}
        server_model = self.server_model_manager.get_server_model(request_group.server_id)
        if request_group.is_training:
            result = server_model.train_and_compute_gradient(kwargs, offset=offset)
        else:
            result = server_model.eval_get_logits(kwargs, offset=offset)
        self.client_layer_allocator.trigger_reallocation()

        for event, client_id, result_item in zip(events, client_ids, result):
            diff_data = self.client_layer_allocator.get_layers_diff(client_id)
            num_layers_diff, params = 0, None
            if diff_data is not None:
                previous, num_layers_diff = diff_data
                params = self.server_model_manager \
                    .extract_layer_weights(previous, previous+num_layers_diff)

            event.set_result(BatchDataBackward(
                data=ndarray_dict_to_bytes(result_item),
                control_code=ControlCode.OK,
                layers_diff=num_layers_diff,
                additional_layers_parameters=torch_list_to_bytes(params)
            ))

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
        cid = req.cid
        async for req in request_iterator:
            if control_code_from_proto(req.control_code) == ControlCode.DO_CLOSE_STREAM:
                break
            res = await self._compute_response(cid, req)

            if res is not None:
                yield self._to_grpc(res)

        async with self._lock:
            self.server_model_manager.strategy.mark_client_as_done(cid)
        yield server_model_pb2.BatchDataBackward(
            control_code=control_code_to_proto(ControlCode.STREAM_CLOSED_OK)
        )
