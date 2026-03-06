import socket
from logging import DEBUG
from typing import Optional

import numpy as np
import torch
from flwr.common import log

from slbd.common import LayerChangeInfo
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
            SL_EMBEDDINGS: [d.detach().cpu().numpy() for d in kwargs[SL_EMBEDDINGS]] if isinstance(kwargs[SL_EMBEDDINGS], list) else kwargs[SL_EMBEDDINGS].detach().cpu().numpy(),
            SL_LABELS: kwargs[SL_LABELS].numpy(),
        }
        if kwargs[SL_WEIGHTS]:
            batch_data[SL_WEIGHTS] = [t.detach().cpu().numpy() for t in kwargs[SL_WEIGHTS]]
            if kwargs[SL_OPTIM_STATES]:
                batch_data[SL_OPTIM_STATES] = [{k: t.detach().cpu().numpy() for k, t in dct.items()} for dct in kwargs[SL_OPTIM_STATES]]

        # Prepare message
        message = {
            "type": "fwd" if _requires_response_ else "sfwd", # sfwd for "streaming" fwd
            "data": batch_data,
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
        backward_data = response["data"]
        gradient = torch.from_numpy(backward_data[SL_GRAD])

        layer_change_info = None
        if num_layers_diff != 0:
            additional_layers_parameters, additional_optimizer_states = None, None
            if SL_WEIGHTS in backward_data:
                additional_layers_parameters = [torch.from_numpy(t) for t in backward_data[SL_WEIGHTS]]
                if SL_OPTIM_STATES in backward_data:
                    additional_optimizer_states = [{k: torch.from_numpy(v) for k, v in dct.items()} for dct in backward_data[SL_OPTIM_STATES]]
            layer_change_info = LayerChangeInfo(
                layers_diff=num_layers_diff,
                additional_layers_parameters=additional_layers_parameters,
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

    def integrate_weights(self, model_parameters, processed_data_points_per_layer):
        # convert dictionary to list as expected on the server
        layer_idxs = sorted(int(k[len("_proc"):]) for k in processed_data_points_per_layer)
        proc_dp = np.array([processed_data_points_per_layer[f"_proc{idx}"] for idx in layer_idxs])
        model_parameters.append(proc_dp)
        message = {
            "type": "weight_integration",
            SL_WEIGHTS: model_parameters,
        }
        send_message(self.sock, message)
        response = recv_message(self.sock)[0]
        assert response == {"message": "ok"}
