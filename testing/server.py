import torch

from flwr.server import ServerConfig
from slbd.server.app import start_server
from slbd.server.strategy import PlainSlStrategy

from testing.model import Model


import random
import numpy as np
random.seed(10)
np.random.seed(10)
torch.manual_seed(10)

model = Model()

nc = 1
strategy = PlainSlStrategy(
    model=model,
    fraction_fit=1.0,
    fraction_evaluate=1.,
    min_available_clients=nc,
    min_fit_clients=nc,
)

start_server(
    server_address="0.0.0.0:8080",
    strategy=strategy,
    config=ServerConfig(num_rounds=30)
)
