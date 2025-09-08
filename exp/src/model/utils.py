import copy

import torch


def warmup_gpu(model, cfg):
    if not torch.cuda.is_available():
        # Warmup only makes sense if a GPU is available
        return

    model = copy.deepcopy(model)

    # just a random optimizer, not really important which one it is as we only
    # use it for "dummy" updates
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    for _ in range(20):
        x = torch.randn(32, 3, 32, 32)
        y = torch.randint(0, cfg.data.num_classes, (32,))
        logits = model(x)
        loss = criterion(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    del optimizer
    del model
