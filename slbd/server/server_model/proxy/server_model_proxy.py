from typing import Dict, Union, List
from abc import ABC, abstractmethod

import numpy as np
try:
    import torch
except ImportError:
    torch = None

from slbd.common import ControlCode, BatchDataBackward, BatchDataForward
from slbd.common.parameter import (
    ndarray_dict_to_bytes,
    bytes_to_ndarray_dict
)

NumpyBatchData = Dict[str, Union[np.ndarray, List[np.ndarray]]]
FORBIDDEN_SERVER_MODEL_METHODS = {
    "get_parameters",
    "configure_fit",
    "configure_evaluate",
    "to_server_model",
}


class ServerModelProxy(ABC):

    def __init__(self, cid: str):
        """Initialize a server model proxt

        Parameters
        ----------
        cid : str
            ID of the client associated with the given proxy
        """
        self.cid = cid
        self.device = None

    @abstractmethod
    def __call__(self, embeddings, labels, _training_: bool=True):
        """Sends a batch of data over the network to the server and waits until receiving the grad

        Parameters
        ----------
        embeddings : _type_
            _description_
        labels : _type_
            _description_
        """

    @abstractmethod
    def close_stream(self):
        """Wait until the server finishes processing the data the client sent to it
        """
