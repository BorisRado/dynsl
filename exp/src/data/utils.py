import os

import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding


def get_dataloader(dataset, batch_size, shuffle=True):
    kwargs = {}
    if dataset._dataset_type == "nlp":
        kwargs["collate_fn"] = DataCollatorWithPadding(
            tokenizer=dataset._employed_tokenizer,
            padding="longest"
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=1,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=True,
        **kwargs
    )
