from typing import Type

from flwr.client.mod.utils import make_ffn
from flwr.client import NumPyClient, ClientFn
from flwr.common import Context, Message

from slbd.client.grpc.message_handler import handle_legacy_message_from_msgtype


class ClientAppException(Exception):
    """Exception raised when an exception is raised while executing a ClientApp."""

    def __init__(self, message: str):
        ex_name = self.__class__.__name__
        self.message = f"\nException {ex_name} occurred. Message: " + message
        super().__init__(self.message)


class ClientApp:
    """Flower ClientApp.

    Examples
    --------
    Assuming a typical `Client` implementation named `FlowerClient`, you can wrap it in
    a `ClientApp` as follows:

    >>> class FlowerClient(NumPyClient):
    >>>     # ...
    >>>
    >>> def client_fn(cid):
    >>>    return FlowerClient().to_client()
    >>>
    >>> app = ClientApp(client_fn)

    If the above code is in a Python module called `client`, it can be started as
    follows:

    >>> flower-client-app client:app --insecure

    In this `client:app` example, `client` refers to the Python module `client.py` in
    which the previous code lives in and `app` refers to the global attribute `app` that
    points to an object of type `ClientApp`.
    """

    def __init__(
        self,
        model,
        client_fn: ClientFn,
        client_adapter_cls: Type[NumPyClient],
        socket_address: str = None,
        socket_port: int = None,
    ) -> None:

        def ffn(
            message: Message,
            context: Context,
        ) -> Message:  # pylint: disable=invalid-name
            out_message = handle_legacy_message_from_msgtype(
                client_fn=client_fn,
                model=model,
                message=message,
                client_adapter_cls=client_adapter_cls,
                socket_address=socket_address,
                socket_port=socket_port,
            )
            return out_message

        # Wrap mods around the wrapped handle function
        self._call = make_ffn(ffn, [])


    def __call__(
        self,
        message: Message,
        context: Context,
    ) -> Message:
        """Execute `ClientApp`."""
        return self._call(message, context)
