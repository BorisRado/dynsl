"""Socket-based server for handling client requests."""

import time
import asyncio
import socket
from logging import INFO
from typing import Optional, Callable

import torch
from flwr.common import log
from hydra.utils import instantiate

from slbd.common import BatchDataForward, BatchDataBackward, SocketTimingMetrics
from slbd.common.constants import (
    SL_DOWNLINK_THROUGHPUT, SL_OPTIM_STATES, SL_WEIGHTS, SL_GRAD,
    SL_PER_LAYER_FORWARD_TIMES, SL_BACKWARD_TIME
)
from slbd.common.parameter import ndarray_dict_to_bytes, bytes_to_ndarray_dict
from slbd.common.socket_utils import send_message_async, recv_message_async
from slbd.server.batching_worker_manager import BatchingWorkerManager
from slbd.server.server_model.manager import ServerModelManager
from slbd.server.client_layer_allocator import ClientLayerAllocator


class SocketServerHandler:
    """Handles socket-based client connections and requests."""

    def __init__(
        self,
        server_model_manager: ServerModelManager,
        client_layer_allocator: ClientLayerAllocator,
        num_concat_batches: int,
        dynamic_layers_strategy: str,
        request_group_timeout: float,
        sending_transform_fn_config: Optional[Callable],
        receiving_transform_fn_config: Optional[Callable],
        exchange_optimizer_states: bool,
        event_loop: asyncio.AbstractEventLoop,
    ) -> None:
        self.server_model_manager = server_model_manager
        self.client_layer_allocator = client_layer_allocator
        self.request_group_timeout = request_group_timeout
        assert dynamic_layers_strategy in {None, "cnt", "afl"}
        self._dynamic_layers_strategy = dynamic_layers_strategy
        self._loop = event_loop
        self._is_cuda_available = torch.cuda.is_available()

        self.sending_transform_fn = instantiate(sending_transform_fn_config, _partial_=True)
        self.receiving_transform_fn = instantiate(receiving_transform_fn_config, _partial_=True)

        # Create batching worker instance
        self._batching_worker = BatchingWorkerManager(
            server_model_manager=server_model_manager,
            client_layer_allocator=client_layer_allocator,
            num_concat_batches=num_concat_batches,
            request_group_timeout=request_group_timeout,
            sending_transform_fn=self.sending_transform_fn,
            receiving_transform_fn=self.receiving_transform_fn,
            dynamic_layers_strategy=dynamic_layers_strategy,
            exchange_optimizer_states=exchange_optimizer_states,
            is_cuda_available=self._is_cuda_available,
            event_loop=event_loop,
        )

    async def handle_client(self, client_socket: socket.socket, client_address):
        """Handle a single client connection.

        Args:
            client_socket: Socket connected to the client
            client_address: Client address tuple
        """
        client_id = None
        stream_start = time.perf_counter()

        try:
            # Receive initialization message
            init_msg = (await recv_message_async(client_socket, self._loop))[0]
            assert init_msg["type"] == "init"
            client_id = init_msg["cid"]

            # Send acknowledgment
            await send_message_async(client_socket, {"type": "init_ack"}, self._loop)

            log(INFO, f"Socket client connected: {client_id} from {client_address}")

            # Handle client requests in a loop
            while True:
                message, uplink_throughput = await recv_message_async(client_socket, self._loop)
                if message is None:
                    break

                elif message["type"] == "close":
                    # Client is closing the connection
                    await send_message_async(client_socket, {"type": "close_ack"}, self._loop)
                    break

                elif message["type"] == "fwd":

                    # Create timing metrics
                    timing_metrics = SocketTimingMetrics(
                        uplink_throughput_Mbps=uplink_throughput,
                        downlink_throughput_Mbps=message[SL_DOWNLINK_THROUGHPUT],
                        per_layer_forward_times=message[SL_PER_LAYER_FORWARD_TIMES],
                        tot_backward_time=message[SL_BACKWARD_TIME],
                    )

                    # Create forward data
                    data_fwd = BatchDataForward(
                        data=message["data"],
                        is_training=message["training"],
                        client_timing_metrics=timing_metrics
                    )

                    res = await self._batching_worker.enqueue_and_get_future(client_id, data_fwd)

                    # Send response
                    await send_message_async(client_socket, self._to_socket_response(res), self._loop)

                elif message["type"] == "sfwd":

                    timing_metrics = SocketTimingMetrics(-1, -1, [], -1) # dummy values

                    # Create forward data
                    data_fwd = BatchDataForward(
                        data=message["data"],
                        is_training=message["training"],
                        client_timing_metrics=timing_metrics
                    )

                    _ = await self._batching_worker.enqueue_and_get_future(client_id, data_fwd)
                    # in streaming FSL, the client trains only on the local loss -- we do not send the gradient back

        except Exception as e:
            import traceback
            log(INFO, f"Error handling socket client {client_id}: {e}")
            traceback.print_exc()

        finally:
            extra_info = ""
            if hasattr(self.client_layer_allocator, "_cid_to_device_type_mapping") and \
               client_id in self.client_layer_allocator._cid_to_device_type_mapping:
                extra_info = self.client_layer_allocator._cid_to_device_type_mapping[client_id]

            log(INFO, f"[Server Profiling] SocketClient client_id={client_id} total_time={time.perf_counter()-stream_start:.4f}s -- [device type: {extra_info}]")

            client_socket.close()

    def _to_socket_response(self, data: BatchDataBackward) -> dict:
        """Convert BatchDataBackward to socket response message.

        Args:
            data: Backward data to send to client

        Returns:
            Dictionary ready to be serialized and sent
        """
        backward_data = {
            SL_GRAD: data.data,
        }
        if data.additional_layers_parameters is not None:
            backward_data[SL_WEIGHTS] = data.additional_layers_parameters
            if data.additional_optimizer_states:
                assert len(data.additional_optimizer_states) > 0
                backward_data[SL_OPTIM_STATES] = data.additional_optimizer_states

        return {
            "type": "backward",
            "data": backward_data,
            "layers_diff": data.layers_diff,
            "training_loss": data.training_loss,
        }


async def start_socket_server(
    handler: SocketServerHandler,
    host: str,
    port: int,
) -> None:
    """Start the socket server.

    Args:
        handler: Handler instance for processing client requests
        host: Host address to bind to
        port: Port to listen on
    """
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((host, port))
    server_socket.listen(5)
    server_socket.setblocking(False)

    log(INFO, f"Socket server listening on {host}:{port}")

    loop = asyncio.get_running_loop()

    try:
        while True:
            # Accept connections asynchronously
            client_socket, client_address = await loop.sock_accept(server_socket)
            # Keep client socket non-blocking for async recv/send operations
            client_socket.setblocking(False)
            print(f"Received connection from {client_address}")
            # Handle each client in a separate task
            asyncio.create_task(handler.handle_client(client_socket, client_address))
    finally:
        server_socket.close()
