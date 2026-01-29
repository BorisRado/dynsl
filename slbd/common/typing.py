from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Union, List, Optional, Tuple

import numpy as np
import torch

from flwr.common import Config


BYTES_TO_MEGABITS = 8 / 1e6

class ControlCode(Enum):
    OK = 0
    DO_CLOSE_STREAM = 1
    STREAM_CLOSED_OK = 2
    ERROR_PROCESSING_STREAM = 3
    INIT_STREAM = 4


@dataclass
class TimingMetrics:
    client_local_computation_time: float
    client_round_time: float
    client_tot_local_time: float
    response_length: int # expressed in number of bytes
    communication_time: float = field(init=False)
    throughput_Mbps: float = field(init=False)

    def __post_init__(self):
        # client_round_time and client_tot_local_time are in seconds
        self.communication_time = self.client_round_time - self.client_tot_local_time

        # throughput is in Mbps -- divide by 1_000_000 to go from bps to Mbps
        # * 8 to convert Bytes to bits
        # * 2 as communication time includes forward and backward computation
        #       thus, if the communication time was T, on average T/2 was spent sending and T/2 receiving the message
        #       for the current application, it makes no sense to differentiate between upload/download speed
        if self.client_local_computation_time > 0:
            self.throughput_Mbps = 2 * BYTES_TO_MEGABITS * self.response_length / self.communication_time
        else:
            # client_local_computation_time is < 0 if reallocation happened recently
            # we say that we cannot compute the throughput because in the current iteration, the
            # amount of sent and received data differs (for instance, because sending the model
            # weights from the server to the client or vice versa). Thus, we ignore the throughput
            # measurement for this iteration by setting it to -1
            self.throughput_Mbps = -1


@dataclass
class BatchDataForward:
    """
    Generic class being passed between client and server during training/evaluation
    """
    data: Dict[str, Union[np.ndarray, List[np.ndarray]]]
    is_training: bool
    client_timing_metrics: Optional[TimingMetrics] = None


@dataclass
class BatchDataBackward:
    """
    Generic class being passed between client and server during training/evaluation
    """
    data: Dict[str, Union[np.ndarray, List[np.ndarray]]]
    layers_diff: int
    training_loss: float
    additional_layers_parameters: List[np.ndarray] = None
    additional_optimizer_states: List[np.ndarray] = None


@dataclass
class LayerChangeInfo:
    layers_diff: int
    additional_layers_parameters: List[torch.Tensor] = None
    additional_optimizer_states: List[torch.Tensor] = None


@dataclass
class ServerModelFitIns:
    """
    Configuration for the server model prior training:
    - `parameters` contains the currect weights of the server model
    - `config` may contain arbitrary user-defined values (learning rate, optimizer, ...)
    """
    model_state_dict: OrderedDict
    first_server_model_layer: int
    config: Config
    sid: str


@dataclass
class ServerModelEvaluateIns:
    """
    Configuration for the server model of the model prior evaluation
    """
    model_state_dict: OrderedDict
    first_server_model_layer: int
    config: Config
    sid: str


@dataclass
class ServerModelFitRes:
    """
    Summary of the final version of the server model, with the updated model weights (`parameters`)
    and the number of training example of the server model has processed
    """
    model_state_dict: OrderedDict
    per_layer_training_examples: Dict[str, int]
    metrics: Dict[str, float]
    config: Config = None
    sid: Optional[str] = None

    def __post_init__(self):
        # user should not set the sid, as it is automatically set by the framework
        assert self.sid is None
