from typing import List, Union
from abc import ABC, abstractmethod
from collections import OrderedDict

from slbd.server.server_model.proxy.server_model_proxy import ServerModelProxy
from slbd.common import ServerModelFitRes, ServerModelEvaluateIns, ServerModelFitIns


class ServerModelManager(ABC):

    def __init__(self):
        self.fit_config, self.evaluation_config = None, None
        self.strategy = None

    @abstractmethod
    def get_server_model(self, sid) -> ServerModelProxy:
        """Create a brand new server trainer proxy to which clients can connect"""

    @abstractmethod
    def collect_server_fit_results(
        self,
    ) -> List[ServerModelFitRes]:
        """Get all the necessary data for server side model aggregation"""

    def set_strategy(self, strategy):
        self.strategy = strategy

    @abstractmethod
    def initialize_server_models(
        self,
        configs: List[Union[ServerModelEvaluateIns, ServerModelFitIns]],
        is_training: bool,
    ):
        """Initialize the server models"""

    @abstractmethod
    def get_server_model_ids(self):
        """Get all the server model ids"""

    @abstractmethod
    def get_global_model_params(self) -> OrderedDict:
        """Get the initial weights of the global model"""

    @abstractmethod
    def extract_layer_weights(self, layer_from, layer_to):
        # returns a copy of the model weights
        ""
