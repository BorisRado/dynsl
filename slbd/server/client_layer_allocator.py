from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict, Any

from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager

from slbd.common.typing import TimingMetrics


class ClientLayerAllocator(ABC):
    """
    Manages all the aspects of layer placement on the clients, deciding:
    - how many layers to place on every client at the beginning of every epoch
    - if more/less layers are to be trained on the client during an epoch
    """

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """
        Initialize the LayerAllocator.
        """

    @abstractmethod
    def initialize_epoch(
        self,
        epoch_clients: List[ClientProxy],
        client_manager: ClientManager,
        is_training: bool,
        **kwargs
    ) -> List[int]:
        """Gets called at the beginning of every training/evaluation epoch, so the allocator
        can execute all the necessary setup procedures

        Parameters
        ----------
        epoch_clients : List[ClientProxy]
            List of clients that have been sampled for training in the current epoch
        client_manager : ClientManager
            Client manager
        is_training : bool
            Whether it is a training or evaluation epoch

        Returns
        -------
        List[int]
            Number of layers to be sent to the client at the beginning of the epoch

        """

    @abstractmethod
    def finalize_epoch(self):
        """Gets called after the end of the current training/evaluation epoch
        """

    @abstractmethod
    def get_post_reallocation_layer_difference(self, client_id: str) -> Optional[Tuple[int, int]]:
        """If in the last reallocation the allocator identified that the client has to
        process more or less layers, the method returns a tuple containing
        how many layers the client was processing before and the diffence to be applied,
        for instance +2 means that the client should train two more layers

        Parameters
        ----------
        client_id : cid
            ID for the considered client

        Returns
        -------
        Optional[Tuple[int, int]]
            Number of layers the client was training before, difference in the number of layers
        """

    @abstractmethod
    def get_client_offloaded_layer_count(self, client_id: str) -> int:
        """Returns how many layers the server has to process for that given client. For instance,
        if the method returns 7, it means that the server must train the latest 7 layers of the model

        Parameters
        ----------
        client_id : str
            ID of the client

        Returns
        -------
        int
            How many layers the server has to train
        """

    @abstractmethod
    def update_client_metrics(self, client_id: str, metrics: Optional[TimingMetrics], num_received_bytes: int):
        """The layer allocator is given the training time of the client every time the
        client sends embeddings to the server

        Parameters
        ----------
        client_id : str
            CID for the client being considered
        metrics : TimingMetrics
            Client metrics
        """

    @abstractmethod
    def update_server_metrics(
        self,
        num_offloaded_layers: int,
        elapsed_time: float,
    ):
        """The layer allocator is given the time for processing a batch of data on the server

        Parameters
        ----------
        num_processed_layers : int
            Nuber of layers that have been processed on the server
        elapsed_time : float
            Training time
        """

    @abstractmethod
    def collect_epoch_metrics(self) -> Dict[str, Any]:
        """Returns the metrics that need to be saved for this epoch

        Returns
        -------
        Dict[str, Any]
            Metrics
        """
