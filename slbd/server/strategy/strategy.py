from collections import OrderedDict
from abc import abstractmethod
from typing import Optional, List, Tuple, Dict, Union

from flwr.common import Scalar
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from slbd.common import (
    ServerModelEvaluateIns,
    ServerModelFitIns,
    ServerModelFitRes,
)
from slbd.server.client_layer_allocator import ClientLayerAllocator
from slbd.torch.sequential_model import FlexibleSequential


class Strategy:

    def initialize_parameters(self, client_manager) -> Optional[OrderedDict]:
        return None

    @abstractmethod
    def configure_fit(
        self,
        server_round: int,
        model_state_dict: OrderedDict,
        client_manager: ClientManager,
        client_layer_allocator: ClientLayerAllocator
    ) -> Tuple[List[Tuple[ClientProxy, FitIns]], List[ServerModelFitIns]]:
        """Configure the next round of training.

        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        parameters : OrderedDict
            The current (global) model parameters.
        client_manager : ClientManager
            The client manager which holds all currently connected clients.

        Returns
        -------
        Tuple[List[Tuple[ClientProxy, FitIns]], List[ServerModelFitIns]]
            Tuple consising of:
            - List of clients sampled for training along with the corresponding fit instructions
            - List of server model instructions. The length of this list determines the number of
            server models that will be training in the current training round
        """

    @abstractmethod
    def aggregate_fit(
        self,
        server_round: int,
        client_results: List[Tuple[ClientProxy, FitRes]],
        server_results: List[ServerModelFitRes],
        layer_allocator: ClientLayerAllocator,
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[OrderedDict, Dict[str, Scalar]]:
        """Aggregate training results.

        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        results : List[Tuple[ClientProxy, FitRes]]
            Successful updates from the previously selected and configured
            clients. Each pair of `(ClientProxy, FitRes)` constitutes a
            successful update from one of the previously selected clients. Not
            that not all previously selected clients are necessarily included in
            this list: a client might drop out and not submit a result. For each
            client that did not submit an update, there should be an `Exception`
            in `failures`.
        server_results List[ServerModelFitRes]
            List of training results of server models
        failures : List[Union[Tuple[ClientProxy, FitRes], BaseException]]
            Exceptions that occurred while the server was waiting for client
            updates.

        Returns
        -------
        parameters : Tuple[Optional[OrderedDict], Dict[str, Scalar]]
            The first returned value is an OrderedDict. If the parameters
            associated with a given key are not to be overridden (e.g., because
            all clients in the current round failed and no new version of the initial
            layers are available), the the value associated with that key is to
            be set to `None`. The second value is a dictionary with aggregated
            training metrics
        """

    @abstractmethod
    def configure_evaluate(
        self,
        server_round: int,
        model_state_dict: OrderedDict,
        client_manager: ClientManager,
        client_layer_allocator: ClientLayerAllocator
    ) -> Tuple[List[Tuple[ClientProxy, EvaluateIns]], List[ServerModelEvaluateIns]]:
        """Configure the next round of evaluation.

        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        parameters : OrderedDict
            The current (global) model parameters.
        client_manager : ClientManager
            The client manager which holds all currently connected clients.

        Returns
        -------
        Tuple[List[Tuple[ClientProxy, FitIns]], List[ServerModelFitIns]]
            Tuple consising of:
            - List of clients sampled for evaluation along with the corresponding eval instructions
            - List of server model instructions. The length of this list determines the number of
            server models that will be used for evaluation in the current evaluation round
        """

    @abstractmethod
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results.

        Parameters
        ----------
        server_round : int
            The current round of federated learning.
        results : List[Tuple[ClientProxy, FitRes]]
            Successful updates from the
            previously selected and configured clients. Each pair of
            `(ClientProxy, FitRes` constitutes a successful update from one of the
            previously selected clients. Not that not all previously selected
            clients are necessarily included in this list: a client might drop out
            and not submit a result. For each client that did not submit an update,
            there should be an `Exception` in `failures`.
        failures : List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]
            Exceptions that occurred while the server was waiting for client updates.

        Returns
        -------
        aggregation_result : Tuple[Optional[float], Dict[str, Scalar]]
            The aggregated evaluation result. Aggregation typically uses some variant
            of a weighted average. Note: by design, we make clients return their
            metrics even though the predictions are made by the server to allow
            a more fine-grained overview of the metrics (a server model may
            predict values for multiple clients)
        """

    # pylint: disable=arguments-differ
    def evaluate(
        self,
        server_round: int,
        parameters: OrderedDict,
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate the current client+server parameters on the server

        Parameters
        ----------
        server_round : int
            server_round
        client_parameters : Parameters
            parameters of the server model
        server_parameters : Parameters
            parameters of the server model

        Returns
        -------
        Optional[Tuple[float, Dict[str, Scalar]]]
            metrics
        """

    @abstractmethod
    def get_model_architecture(self) -> FlexibleSequential:
        """Returns a copy of the initial model
        """

    @abstractmethod
    def route_client_request_to_server_model(self, client_id: str) -> str:
        """Determines which one of the available server models is supposed to process the client req

        Parameters
        ----------
        client_id : str
            ID of the client

        Returns
        -------
        str
            ID of the server model
        """
