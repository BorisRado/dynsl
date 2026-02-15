import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding


def get_dataloader(dataset, batch_size, shuffle=True):
    kwargs = {}
    num_workers = 1
    if dataset._dataset_type == "nlp":
        kwargs["collate_fn"] = DataCollatorWithPadding(
            tokenizer=dataset._employed_tokenizer,
            padding="longest"
        )
    if dataset._dataset_type == "cv" and dataset._keep_in_memory:
        num_workers = 0
        def cv_collate_fn(batch):
            imgs, labels = zip(*batch)
            imgs = [img.flip(-1) if torch.rand(1).item() < 0.5 else img for img in imgs]
            return {"img": torch.stack(imgs), "label": torch.stack(labels)}

        kwargs["collate_fn"] = cv_collate_fn

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=num_workers > 0,
        **kwargs
    )


class ToRgb(torch.nn.Module):
    def forward(self, img):
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
