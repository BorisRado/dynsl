from typing import Tuple, Dict

import torch
import torchvision
import torchvision.transforms as transforms
from flwr.common import Config, Scalar

from slbd.client.app import start_client
from slbd.client.torch_client import TorchClient
from slbd.client.model_trainer import ModelTrainer

from testing.model import Model

class Client(TorchClient):

    def __init__(self, cid):
        _ = (cid, )
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.dataset_folder = torch.hub.get_dir() + '/datasets'
        self.batch_size = 64


    def get_properties(self, config):
        _ = (config,)
        return {}

    def fit(self, config, model_trainer):
        _ = (config, )
        # Define transformations for the data

        # Load CIFAR-10 training dataset
        trainset = torchvision.datasets.CIFAR10(
            root=self.dataset_folder, train=True, download=False, transform=self.transform
        )
        # Select a consistent subset of 10,000 images
        # Using a fixed seed ensures the same indices are selected each time
        torch.manual_seed(42)  # Fixed seed for reproducibility
        indices = torch.randperm(len(trainset))[:1000]
        trainset = torch.utils.data.Subset(trainset, indices)
        # Reset the random seed to avoid affecting other random operations
        torch.manual_seed(torch.initial_seed())

        # Create data loader
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )
        for img, label in trainloader:
            model_trainer(img, label)

    def evaluate(self, config, model_trainer):
        _ = (config,)
        testset = torchvision.datasets.CIFAR10(
            root=self.dataset_folder, train=True, download=False, transform=self.transform
        )
        torch.manual_seed(42)  # Fixed seed for reproducibility
        indices = torch.randperm(len(testset))[:1000]
        testset = torch.utils.data.Subset(testset, indices)
        # Reset the random seed to avoid affecting other random operations
        torch.manual_seed(torch.initial_seed())

        testloader = torch.utils.data.DataLoader(
            testset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )
        for img, label in testloader:
            model_trainer.evaluate(img, label)

    def get_properties(self, config):
        _ = (config, )
        return {"min_layers": 3}


if __name__ == "__main__":
    import random
    import numpy as np
    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(10)

    start_client(
        server_address="0.0.0.0:8080",
        client_fn=Client,
        model=Model(),
    )
