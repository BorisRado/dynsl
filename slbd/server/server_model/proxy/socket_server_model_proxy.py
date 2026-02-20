import socket
from logging import DEBUG
from typing import Optional

from flwr.common import log

from slbd.common import LayerChangeInfo, bytes_to_ndarray_dict, ndarray_dict_to_bytes
from slbd.common.constants import (
    SL_EMBEDDINGS, SL_LABELS, SL_WEIGHTS, SL_OPTIM_STATES, SL_GRAD,
    SL_DOWNLINK_THROUGHPUT, SL_PER_LAYER_FORWARD_TIMES, SL_BACKWARD_TIME
)
from slbd.common.socket_utils import send_message, recv_message
from slbd.server.server_model.proxy.server_model_proxy import ServerModelProxy


class SocketServerModelProxy(ServerModelProxy):
    """Socket-based proxy for communicating with the server model."""

    def __init__(
        self,
        *,
        server_address: str,
        server_port: int,
        cid: str
    ) -> None:
        super().__init__(cid=cid)
        log(DEBUG, "Initializing socket proxy for CID %s", cid)
        self.server_address = server_address
        self.server_port = server_port
        self.sock: Optional[socket.socket] = None
        self.downlink_throughput = None
        self._initialize_connection()

    def _initialize_connection(self):
        """Establish socket connection to server."""
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.server_address, self.server_port))
        # Send initialization message with client ID
        send_message(self.sock, {"type": "init", "cid": self.cid})
        response = recv_message(self.sock)[0]
        assert response["type"] == "init_ack"

    def __call__(
        self,
        training=True,
        _requires_response_=True,
        **kwargs,
    ):
        """Send data to server and receive gradient.

        Args:
            embeddings: Client model output
            labels: Training labels
            local_computation_time: Time spent on local computation
            client_model_weights: Client model weights if sending to server
            removed_optimizer_states: Optimizer states if sending to server
            training: Whether in training mode

        Returns:
            Tuple of (gradient, layer_change_info, training_loss)
        """
        # Serialize data
        batch_data = {
            SL_EMBEDDINGS: kwargs[SL_EMBEDDINGS],
            SL_LABELS: kwargs[SL_LABELS],
        }
        if kwargs[SL_WEIGHTS]:
            batch_data[SL_WEIGHTS] = kwargs[SL_WEIGHTS]
            if kwargs[SL_OPTIM_STATES]:
                batch_data[SL_OPTIM_STATES] = kwargs[SL_OPTIM_STATES]

        # Prepare message
        message = {
            "type": "fwd" if _requires_response_ else "sfwd", # sfwd for "streaming" fwd
            "data": ndarray_dict_to_bytes(**batch_data),
            "training": training,
            SL_DOWNLINK_THROUGHPUT: self.downlink_throughput,
            SL_PER_LAYER_FORWARD_TIMES: kwargs[SL_PER_LAYER_FORWARD_TIMES],
            SL_BACKWARD_TIME: kwargs[SL_BACKWARD_TIME],
        }

        # Send and receive
        send_message(self.sock, message)
        if not _requires_response_:
            return
        response, self.downlink_throughput = recv_message(self.sock)

        # Parse response
        num_layers_diff = response["layers_diff"]
        backward_data = bytes_to_ndarray_dict(**response["data"])
        gradient = backward_data[SL_GRAD]

        layer_change_info = None
        if num_layers_diff != 0:
            layer_change_info = LayerChangeInfo(
                layers_diff=num_layers_diff,
                additional_layers_parameters=backward_data.get(SL_WEIGHTS, None),
                additional_optimizer_states=backward_data.get(SL_OPTIM_STATES, None),
            )

        return gradient, layer_change_info, response["training_loss"]

    def close_stream(self, error=False):
        """Close the socket connection.

        Args:
            error: Whether closing due to error
        """
        if self.sock is None:
            return

        message = {
            "type": "close",
            "error": error
        }
        send_message(self.sock, message)
        response = recv_message(self.sock)[0]
        assert response["type"] == "close_ack"

        self.sock.close()
        self.sock = None
