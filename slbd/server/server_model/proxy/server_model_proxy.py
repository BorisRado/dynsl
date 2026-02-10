from abc import ABC, abstractmethod
from typing import Dict, Union, List

import numpy as np


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
    def __call__(self, embeddings, labels, local_computation_time, client_model_weights, removed_optimizer_states, training=True):
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
