import time
import asyncio
from logging import INFO, WARNING
from collections import defaultdict
from typing import AsyncIterable, Dict, Tuple

import torch
from flwr.common import log

from slbd.proto import server_model_pb2_grpc
from slbd.proto import server_model_pb2
from slbd.server.server_model.manager import ServerModelManager
from slbd.common.utils import measure_time
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
    # for testing: servicer that just returns the same data the client sent
    def __init__(
        self,
        server_model_manager: ServerModelManager,
        client_layer_allocator: ClientLayerAllocator,
        num_concat_batches: int,
        request_group_timeout,
    ) -> None:
        super().__init__()
        print("PING PONG>>>")
        self.client_layer_allocator: ClientLayerAllocator = client_layer_allocator

    async def StreamRequest(
        self,
        request_iterator: AsyncIterable[server_model_pb2.BatchDataForward],
        context,
    ) -> AsyncIterable[server_model_pb2.BatchDataBackward]:
        _ = (context, )
        stream_start = time.time()
        req = await request_iterator.__anext__()
        client_id = req.cid
        assert control_code_from_proto(req.control_code) == ControlCode.INIT_STREAM

        batch_sent_time = None
        async for req in request_iterator:
            if control_code_from_proto(req.control_code) == ControlCode.DO_CLOSE_STREAM:
                break

            if batch_sent_time is not None:
                print(time.time() - batch_sent_time, req.local_computation_time)
            yield server_model_pb2.BatchDataBackward(
                data=to_grpc_format({"": from_grpc_format(req.data)["embeddings"]}),
                control_code=control_code_to_proto(req.control_code),
                layers_diff=0,
                additional_layers_parameters=None,
                training_loss=0.,
            )
            batch_sent_time = time.time()

        stream_end = time.time()
        log(INFO, f"[Server Profiling] StreamRequest client_id={client_id} total_time={stream_end-stream_start:.4f}s")

        yield server_model_pb2.BatchDataBackward(
            control_code=control_code_to_proto(ControlCode.STREAM_CLOSED_OK)
        )
