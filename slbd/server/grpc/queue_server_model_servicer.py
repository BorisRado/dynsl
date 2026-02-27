import time
import asyncio
from logging import INFO
from typing import AsyncIterable, Optional, Callable

from slbd.server.batching_worker_manager import BatchingWorkerManager
import torch
from flwr.common import log
from hydra.utils import instantiate

from slbd.proto import server_model_pb2_grpc
from slbd.proto import server_model_pb2
from slbd.server.server_model.manager import ServerModelManager
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


class ServerModelServicer(server_model_pb2_grpc.ServerModelServicer):

    def __init__(
        self,
        server_model_manager: ServerModelManager,
        client_layer_allocator: ClientLayerAllocator,
        num_concat_batches: int,
        request_group_timeout: float,
        sending_transform_fn_config: Optional[Callable],
        receiving_transform_fn_config: Optional[Callable],
        exchange_optimizer_states: bool
    ) -> None:
        super().__init__()
        self.server_model_manager = server_model_manager
        self.request_group_timeout = request_group_timeout
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
