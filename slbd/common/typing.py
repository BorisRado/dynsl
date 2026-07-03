from enum import Enum
from dataclasses import dataclass
from collections import OrderedDict
from typing import Dict, Union, List, Optional

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
class SocketTimingMetrics:
    uplink_throughput_Mbps: float
    downlink_throughput_Mbps: float
    per_layer_forward_times: List[float]
    tot_backward_time: float

    def __post_init__(self):
        self.throughput_Mbps = -1.
        if self.downlink_throughput_Mbps:
            # self.throughput_Mbps = (self.uplink_throughput_Mbps + self.downlink_throughput_Mbps) / 2
            self.throughput_Mbps = self.downlink_throughput_Mbps

            # FIGURE: Uncomment the following line when running experiments for network throughput estimation
            # self.throughput_Mbps = self.uplink_throughput_Mbps


@dataclass
class BatchDataForward:
    """
    Generic class being passed between client and server during training/evaluation
    """
    data: Dict[str, Union[np.ndarray, List[np.ndarray]]]
    is_training: bool
    client_timing_metrics: Optional[SocketTimingMetrics] = None


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
