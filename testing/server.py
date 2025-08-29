from slbd.server.app import start_server

from testing.model import Model
from testing.utils import set_seed

set_seed()
model = Model()

start_server(
    model=model,
    config_file="testing/config.yaml",
    server_address="0.0.0.0:8080",
)
