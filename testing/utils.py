import random

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


def get_train_dataloader():
    set_seed()
    dataset_folder = torch.hub.get_dir() + "/datasets"
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(
        root=dataset_folder, train=True, download=False, transform=transform
    )
    indices = torch.randperm(len(trainset))[:128]
    trainset = torch.utils.data.Subset(trainset, indices)

    batch_size = 8
    return torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

def set_seed():
    random.seed(10)
    np.random.seed(10)
    torch.manual_seed(10)
