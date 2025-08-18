
import random

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from testing.model import Model

random.seed(10)
np.random.seed(10)
torch.manual_seed(10)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(device)

torch.manual_seed(42)  # Fixed seed for reproducibility
dataset_folder = torch.hub.get_dir() + '/datasets'
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
criterion = torch.nn.CrossEntropyLoss()


for epoch in range(30):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    trainset = torchvision.datasets.CIFAR10(
        root=dataset_folder, train=True, download=False, transform=transform
    )
    indices = torch.randperm(len(trainset))[:1000]
    trainset = torch.utils.data.Subset(trainset, indices)
    # Reset the random seed to avoid affecting other random operations
    torch.manual_seed(torch.initial_seed())

    batch_size = 64
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    model.train()
    for img, label in trainloader:
        img, label = img.to(device), label.to(device)
        logits = model(img)
        loss = criterion(logits, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    testset = torchvision.datasets.CIFAR10(
        root=dataset_folder, train=True, download=False, transform=transform
    )
    torch.manual_seed(42)  # Fixed seed for reproducibility
    indices = torch.randperm(len(testset))[:1000]
    testset = torch.utils.data.Subset(testset, indices)
    # Reset the random seed to avoid affecting other random operations
    torch.manual_seed(torch.initial_seed())

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    model.eval()
    tot_loss = 0.
    acc = 0.
    for img, label in trainloader:
        img, label = img.to(device), label.to(device)
        with torch.no_grad():
            logits = model(img)
        tot_loss += criterion(logits, label).item()
        acc += (logits.argmax(dim=1) == label).sum() / label.shape[0]
    print(epoch, "loss", tot_loss / len(testloader))
