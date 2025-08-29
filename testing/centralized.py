import torch

from testing.model import Model
from testing.utils import get_train_dataloader, set_seed


set_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(device)
criterion = torch.nn.CrossEntropyLoss()


for epoch in range(10):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    trainloader = get_train_dataloader()
    set_seed()
    model.train()
    for img, label in trainloader:
        img, label = img.to(device), label.to(device)
        logits = model(img)
        loss = criterion(logits, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    tot_loss = 0.
    acc = 0.
    for img, label in trainloader:
        img, label = img.to(device), label.to(device)
        with torch.no_grad():
            logits = model(img)
        tot_loss += criterion(logits, label).item()
        acc += (logits.argmax(dim=1) == label).sum() / label.shape[0]
    print(epoch, "loss", tot_loss / len(trainloader))
