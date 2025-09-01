import torch
from torch.utils.data import DataLoader


def get_dataloader(dataset, batch_size, num_workers=2, shuffle=True):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )
