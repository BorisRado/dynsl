import torch

from slbd.server.app import start_server

from testing.model import Model


import random
import numpy as np
random.seed(10)
np.random.seed(10)
torch.manual_seed(10)


model = Model()
start_server(
    model=model,
    config_file="testing/config.yaml",
    server_address="0.0.0.0:8080",
)
