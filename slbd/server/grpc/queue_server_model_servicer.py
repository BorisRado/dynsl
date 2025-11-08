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
from slbd.common.parameter import ndarray_dict_to_bytes, bytes_to_ndarray_dict, torch_list_to_bytes
from slbd.server.client_layer_allocator import ClientLayerAllocator
from slbd.common.serde import (
    control_code_from_proto,
    control_code_to_proto,
    from_grpc_format,
    to_grpc_format,
    bytes_list_to_grpc_format
)


# pylint: disable=no-member
class ServerModelServicer(server_model_pb2_grpc.ServerModelServicer):

    def __init__(
        self,
        server_model_manager: ServerModelManager,
        client_layer_allocator: ClientLayerAllocator,
        num_concat_batches: int,
        send_server_parameters_when_forking: bool,
        request_group_timeout: float,
        sending_transform_fn_config: Optional[Callable],
        receiving_transform_fn_config: Optional[Callable],
    ) -> None:
        super().__init__()
        self.server_model_manager = server_model_manager
        self.request_group_timeout = request_group_timeout
        self._send_server_parameters_when_forking = send_server_parameters_when_forking
        self._is_cuda_available = torch.cuda.is_available()
        self.client_layer_allocator: ClientLayerAllocator = client_layer_allocator
        self._request_queues: Dict[Tuple[int, str], asyncio.Queue] = defaultdict(asyncio.Queue)
        self._workers = {}

        self._num_concat_batches = num_concat_batches
        self._loop = asyncio.get_running_loop()
        self.sending_transform_fn = instantiate(sending_transform_fn_config, _partial_=True)
        self.receiving_transform_fn = instantiate(receiving_transform_fn_config, _partial_=True)

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

        self.client_layer_allocator.update_client_metrics(
            client_id=client_id,
            metrics=timing_metrics,
            num_received_bytes=len(data.data["embeddings"])
        )
        num_offloaded_layers = self.client_layer_allocator.get_client_offloaded_layer_count(client_id)
        key = (num_offloaded_layers, server_id)

        if key not in self._workers or self._workers[key].done():
            # start the consumer process
            self._workers[key] = asyncio.create_task(self.batching_worker(*key))
        await self._request_queues[key].put((data, future, client_id))

        return await future

    async def batching_worker(self, num_offloaded_layers: int, server_id: str):
        q = self._request_queues[(num_offloaded_layers, server_id)]
        computation_tasks = []

        while True:
            batches, futures, client_ids = [], [], []

            # collect up to num_concat_batches (with timeout)
            try:
                start_time = asyncio.get_event_loop().time()
                for _ in range(self._num_concat_batches):
                    remaining_time = self.request_group_timeout - (
                        asyncio.get_event_loop().time() - start_time
                    )
                    if remaining_time <= 0:
                        break
                    data, future, client_id = await asyncio.wait_for(q.get(), remaining_time)
                    batches.append(data)
                    futures.append(future)
                    client_ids.append(client_id)

            except asyncio.TimeoutError:
                if len(batches) == 0:
                    continue
                log(WARNING, f"Did not attain sufficient number of batches. Processing {len(batches)}...")

            # Start computation as a separate task and continue immediately
            computation_task = asyncio.create_task(
                self._run_computation_and_distribute_results(
                    batches, futures, client_ids, num_offloaded_layers, server_id
                )
            )
            computation_tasks.append(computation_task)

            # Clean up completed tasks to prevent memory leaks
            computation_tasks = [task for task in computation_tasks if not task.done()]

    async def _run_computation_and_distribute_results(
        self, batches, futures, client_ids, num_offloaded_layers, server_id
    ):
        """Run computation and distribute results back to futures"""
        try:
            result = await self._run_computation(
                batches, client_ids, num_offloaded_layers, server_id
            )
            # distribute results back to futures
            for fut, res in zip(futures, result):
                if not fut.done():  # Check if future wasn't cancelled
                    fut.set_result(res)
        except Exception as e:
            # Set exception on all futures
            for fut in futures:
                if not fut.done():
                    fut.set_exception(e)

    async def _run_computation(self, batches, client_ids, num_offloaded_layers, server_id):

        kwargs_list = [bytes_to_ndarray_dict(b.data) for b in batches]
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
        forking_data = {}
        for result_item, client_id, received_data in zip(grad_results, client_ids, kwargs_list):
            gradient, training_loss = result_item
            if self.sending_transform_fn is not None:
                gradient = self.sending_transform_fn(gradient, received_data=received_data)
            if isinstance(gradient, torch.Tensor):
                gradient = gradient.numpy()
            else:
                gradient = [g.numpy() for g in gradient]

            diff_data = self.client_layer_allocator.get_post_reallocation_layer_difference(client_id)

            num_layers_diff, params = 0, None
            if diff_data is not None:
                previous, num_layers_diff = diff_data
                log(INFO, "\tChange in offload strategy for %s. Difference in number of layers: %d",
                    client_id, num_layers_diff)
                if num_layers_diff > 0 and self._send_server_parameters_when_forking:
                    forking_data[client_id] = list(range(previous, previous+num_layers_diff))
                    params = await self.server_model_manager \
                        .extract_layer_weights(previous, previous+num_layers_diff)

            results.append(BatchDataBackward(
                data=ndarray_dict_to_bytes(gradient),
                control_code=ControlCode.OK,
                layers_diff=num_layers_diff,
                additional_layers_parameters=params,
                training_loss=training_loss,
            ))

        self.server_model_manager.create_fork(forking_data)
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
            timing_metrics = TimingMetrics(
                client_local_computation_time=req.local_computation_time,
                client_round_time=data_received_time - data_sending_time,
                client_tot_local_time=req.tot_local_time,
            ) if data_sending_time > 0 else None
            res = await self._compute_response(client_id, req, timing_metrics)

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
