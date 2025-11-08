from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Union, List, Optional, Tuple

import torch

from flwr.common import Config


class ControlCode(Enum):
    OK = 0
    DO_CLOSE_STREAM = 1
    STREAM_CLOSED_OK = 2
    ERROR_PROCESSING_STREAM = 3
    INIT_STREAM = 4


@dataclass
class BatchDataForward:
    """
    Generic class being passed between client and server during training/evaluation
    """
    data: Dict[str, Union[bytes, List[bytes]]]
    control_code: ControlCode
    is_training: bool
    local_computation_time: float


@dataclass
class BatchDataBackward:
    """
    Generic class being passed between client and server during training/evaluation
    """
    data: Dict[str, Union[bytes, List[bytes]]]
    control_code: ControlCode
    layers_diff: int
    training_loss: float
    additional_layers_parameters: List[bytes] = None


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
        self.throughput = 2 * (num_bytes * 8 / self.communication_time) / 1e6


@dataclass
class LayerChangeInfo:
    layers_diff: int
    additional_layers_parameters: List[torch.Tensor] = None


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
