from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Union, List, Optional, Tuple

import numpy as np
import torch

from flwr.common import Config


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
    communication_time: float = field(init=False)
    throughput: float = field(init=False)

    def __post_init__(self):
        self.communication_time = self.client_round_time - self.client_tot_local_time

    def compute_throughput(self, num_bytes):
        # communication time includes forward and backward computation, thus we multiply times 2
        # expressed in bits per second
        self.throughput = 2 * (num_bytes * 8 / self.communication_time)


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
    forking_data: Optional[Dict[int, Tuple[List[str], int]]] = None

    def __post_init__(self):
        # user should not set the sid, as it is automatically set by the framework
        assert self.sid is None
