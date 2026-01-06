import torch
import torch.nn as nn


def warmup_device(steps: int = 10):
    """
    Warm up the training environment by performing a few lightweight forward and backward passes.
    Automatically uses GPU if available, otherwise CPU.
    """

    # Automatically detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Warm-up] Using device: {device}")

    # Define a minimal dummy model
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    ).to(device)

    # Dummy input and target
    x = torch.randn(32, 128, device=device)  # small batch, small input
    target = torch.randint(0, 10, (32,), device=device)

    # Simple loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer (any small one)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Perform a few dummy forward + backward passes
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

    # Synchronize (important for GPU warm-up)
    if device.type == "cuda":
        torch.cuda.synchronize()

    print("[Warm-up] Completed.")


def test_model(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            inputs, targets = batch["img"].to(device), batch["label"].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total_samples += targets.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples
    print(f"Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy
