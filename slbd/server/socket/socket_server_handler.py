"""Socket-based server for handling client requests."""

import time
import asyncio
import socket
import pickle
import threading
from logging import INFO
from typing import Optional, Callable

import torch
from flwr.common import log
from hydra.utils import instantiate

from slbd.common import BatchDataForward, BatchDataBackward, TimingMetrics
from slbd.common.parameter import ndarray_dict_to_bytes, bytes_to_ndarray_dict, ndarray_list_to_bytes
from slbd.common.socket_utils import send_message, recv_message
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
        data_sending_time, data_received_time = -1, -1
        stream_start = time.perf_counter()

        try:
            # Receive initialization message
            init_msg = recv_message(client_socket)
            assert init_msg["type"] == "init"
            client_id = init_msg["cid"]

            # Send acknowledgment
            send_message(client_socket, pickle.dumps({"type": "init_ack"}))

            log(INFO, f"Socket client connected: {client_id} from {client_address}")

            # Handle client requests in a loop
            while True:
                message = recv_message(client_socket)
                data_received_time = time.perf_counter()
                if message is None:
                    break

                if message["type"] == "close":
                    # Client is closing the connection
                    send_message(client_socket, pickle.dumps({"type": "close_ack"}))
                    break

                if message["type"] == "forward":

                    # Parse incoming data
                    batch_data = bytes_to_ndarray_dict(message["data"])
                    response_nbytes = batch_data["embeddings"].nbytes + batch_data["labels"].nbytes

                    # Create timing metrics
                    timing_metrics = TimingMetrics(
                        client_local_computation_time=message["local_computation_time"],
                        client_round_time=data_received_time - data_sending_time,
                        client_tot_local_time=message["tot_local_time"],
                        response_length=response_nbytes
                    ) if data_sending_time > 0 else None

                    # Create forward data
                    data_fwd = BatchDataForward(
                        data=batch_data,
                        is_training=message["training"],
                        client_timing_metrics=timing_metrics
                    )

                    # Enqueue and wait for result
                    future = self._batching_worker.enqueue_and_get_future(client_id, data_fwd)
                    res = await future

                    # Prepare response
                    response = self._to_socket_response(res)
                    serialized_response = pickle.dumps(response)
                    data_sending_time = time.perf_counter()

                    # Send response
                    send_message(client_socket, serialized_response)

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

            if hasattr(self.client_layer_allocator, "remove_client"):
                self.client_layer_allocator.remove_client(client_id)

            client_socket.close()

    def _to_socket_response(self, data: BatchDataBackward) -> dict:
        """Convert BatchDataBackward to socket response message.

        Args:
            data: Backward data to send to client

        Returns:
            Dictionary ready to be serialized and sent
        """
        layer_params = None
        if data.additional_layers_parameters is not None:
            layer_params = ndarray_list_to_bytes(data.additional_layers_parameters)

        send_data = {"grad": data.data}
        if data.additional_optimizer_states:
            assert len(data.additional_optimizer_states) > 0
            send_data["optim_states"] = ndarray_list_to_bytes(data.additional_optimizer_states)

        return {
            "type": "backward",
            "data": ndarray_dict_to_bytes(send_data),
            "layers_diff": data.layers_diff,
            "additional_layers_parameters": layer_params,
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
            # Make client socket blocking for synchronous recv/send operations
            client_socket.setblocking(True)
            print(f"received connection from {client_address}")
            # Handle each client in a separate task
            asyncio.create_task(handler.handle_client(client_socket, client_address))
    finally:
        server_socket.close()
