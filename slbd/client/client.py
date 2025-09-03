from flwr.client.client import Client as FlwrClient


class Client(FlwrClient):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
