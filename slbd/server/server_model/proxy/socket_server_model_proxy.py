import time
import pickle
import socket
from logging import DEBUG
from typing import Optional

from flwr.common import log

from slbd.common import LayerChangeInfo, bytes_to_torch, torch_to_bytes
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
        embeddings,
        labels,
        local_computation_time,
        client_model_weights,
        removed_optimizer_states,
        training=True
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
        batch_data = torch_to_bytes(
            embeddings=embeddings,
            labels=labels,
            client_model_weights=client_model_weights,
            removed_optimizer_states=removed_optimizer_states,
        )

        # Prepare message
        message = {
            "type": "forward",
            "data": batch_data,
            "training": training,
            "local_computation_time": local_computation_time,
            "downlink_throughput": self.downlink_throughput
        }

        # Send and receive
        send_message(self.sock, message)
        response, self.downlink_throughput = recv_message(self.sock)

        # Parse response
        num_layers_diff = response["layers_diff"]
        gradient_data = bytes_to_torch(response["data"])
        gradient = gradient_data["grad"]
        additional_optimizer_states = gradient_data.get("optim_states", None)

        layer_change_info = LayerChangeInfo(
            layers_diff=num_layers_diff,
            additional_layers_parameters=None if num_layers_diff <= 0
                else bytes_to_torch({"": response["additional_layers_parameters"]}),
            additional_optimizer_states=additional_optimizer_states,
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
