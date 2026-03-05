import torch
import copy
import torch.nn as nn


def warmup_device(model: nn.Sequential):
    """
    Warm up the training environment by performing a few lightweight forward and backward passes.
    Automatically uses GPU if available, otherwise CPU.
    """

    # Automatically detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Warm-up] Using device: {device}")

    # Define a minimal dummy model
    is_meta = next(iter(model.parameters())).device.type == "meta"
    if not is_meta:
        state_dict_copy = copy.deepcopy(model.state_dict())
        model.to(device)
    else:
        model = copy.deepcopy(model)
        model = model.to_empty(device=device)


    # Dummy input and target
    bs = 8
    steps = 50 if torch.cuda.is_available() else 10
    x = torch.randn(bs, 3, 32, 32, device=device)  # small batch, small input
    target = torch.randint(0, 10, (bs,), device=device)

    # Simple loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer (any small one)
    optimizer = torch.optim.Adam(model.parameters())

    # Perform a few dummy forward + backward passes
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()

    del optimizer

    # Synchronize (important for GPU warm-up)
    if device.type == "cuda":
        torch.cuda.synchronize()

    if not is_meta:
        model.cpu()
        model.load_state_dict(state_dict_copy)

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


def replace_bn_with_gn(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            channels = child.num_features
            channels_per_group = 2
            for cpg in [16, 12, 8, 6, 4]:
                if channels % cpg == 0:
                    channels_per_group = cpg
                    break
            gn = nn.GroupNorm(
                num_groups=channels // channels_per_group,
                num_channels=channels,
                affine=True
            )
            setattr(module, name, gn)
        else:
            replace_bn_with_gn(child)
