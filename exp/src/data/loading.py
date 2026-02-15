from functools import partial

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from datasets import load_dataset
from hydra.utils import instantiate
from omegaconf import OmegaConf
from flwr_datasets import FederatedDataset
from datasets.utils.logging import disable_progress_bar
from transformers import AutoTokenizer

from exp.src.common.stochasticity import set_seed

disable_progress_bar()


def _apply_transforms_fn(batch, transforms):
  batch["img"] = [transforms(img) for img in batch["img"]]
  return batch


def load_cv_testset(dataset_config):
    test_transforms = instantiate(dataset_config.test_transforms)
    dataset = load_dataset(dataset_config.name, split="test")

    apply_test_transforms = partial(_apply_transforms_fn, transforms=test_transforms)

    dataset = dataset.with_transform(apply_test_transforms)
    if "image" in dataset.column_names:
        dataset = dataset.rename_column("image", "img")
    dataset = dataset.select_columns(["img", "label"])
    dataset._dataset_type = "cv"
    dataset._keep_in_memory = False
    return dataset


def _load_datasets_onto_memory(datasets, batch_size=1024, num_workers=0):
    # Use DataLoader for faster batch loading
    tensor_datasets = []
    for ds in datasets:
        imgs = []
        labels = []
        loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
        for batch in loader:
            assert isinstance(batch["img"], torch.Tensor)
            assert isinstance(batch["label"], torch.Tensor)
            imgs.append(batch["img"])
            labels.append(batch["label"])
        imgs_tensor = torch.cat(imgs)
        labels_tensor = torch.cat([torch.tensor(l) if not torch.is_tensor(l) else l for l in labels])
        tensor_datasets.append(TensorDataset(imgs_tensor, labels_tensor))
    return tensor_datasets


def _load_cv_dataset(client_idxs, dataset_config, partitioning_config, seed, keep_in_memory):
    train_transforms = instantiate(dataset_config.train_transforms)
    partitioner = instantiate(partitioning_config)

    fds = FederatedDataset(
        dataset=dataset_config.name,
        partitioners={"train": partitioner},
        seed=seed,
        keep_in_memory=False,
    )

    if keep_in_memory:
        # Remove RandomHorizontalFlip if present in the Compose transforms
        train_transforms.transforms = [
            t for t in train_transforms.transforms
            if t.__class__.__name__ != "RandomHorizontalFlip"
        ]
    apply_train_transforms = partial(_apply_transforms_fn, transforms=train_transforms)
    assert isinstance(client_idxs, list)

    trainsets = [fds.load_partition(idx) for idx in client_idxs]
    trainsets = [ts.select_columns(["img", "label"]) for ts in trainsets]
    trainsets = [ts.with_transform(apply_train_transforms) for ts in trainsets]

    if keep_in_memory:
        trainsets = _load_datasets_onto_memory(trainsets)

    return trainsets


def _load_nlp_dataset(client_idx, dataset_config, partitioning_config, seed):
    partitioner = instantiate(partitioning_config)
    fds = FederatedDataset(
        dataset=dataset_config.name,
        partitioners={"train": partitioner},
        seed=seed,
        keep_in_memory=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(dataset_config.tokenizer_name)
    # tokenize_fn = partial(tokenize_function, tokenizer=tokenizer, max_length=dataset_config.max_len)
    dataset = fds.load_partition(client_idx)
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=dataset_config.max_len
        )
    dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    dataset._employed_tokenizer = tokenizer
    return dataset


def load_client_trainset(client_idx, dataset_config, partitioning_config, seed, keep_in_memory=False):
    if isinstance(client_idx, int):
        client_idx = [client_idx]
        is_single = True
    else:
        is_single = False

    if dataset_config.type == "cv":
        dataset = _load_cv_dataset(client_idx, dataset_config, partitioning_config, seed, keep_in_memory)
        for ds in dataset:
            ds._keep_in_memory = keep_in_memory
    elif dataset_config.type == "nlp":
        assert not keep_in_memory, "Keep in memory only works for CV datasets"
        dataset = _load_nlp_dataset(client_idx, dataset_config, partitioning_config, seed)
    else:
        raise Exception(f"Unknown dataset type {dataset_config.type}")

    for ds in dataset:
        ds._dataset_type = dataset_config.type

    if is_single:
        return dataset[0]

    return dataset


if __name__ == "__main__":
    config_string = """
partitioning:
    _target_: flwr_datasets.partitioner.IidPartitioner
    num_partitions: 20
dataset:
    name: cifar10
    input_shape: [3, 32, 32]
    num_classes: 10
    train_transforms:
        _target_: torchvision.transforms.Compose
        transforms:
            - _target_: torchvision.transforms.RandomCrop
              size: 32
              padding: 4
            - _target_: torchvision.transforms.RandomHorizontalFlip
              p: 0.5
            - _target_: torchvision.transforms.ToTensor
            - _target_: torchvision.transforms.Normalize
              mean: [0.4914, 0.4822, 0.4465]
              std: [0.2023, 0.1994, 0.2010]
"""
    cfg = OmegaConf.create(config_string)
    print(OmegaConf.to_yaml(cfg))
    ds1 = load_client_trainset(0, cfg.dataset, cfg.partitioning, 10)
    ds2 = load_client_trainset(0, cfg.dataset, cfg.partitioning, 10)

    set_seed(123)
    ds3 = load_client_trainset(0, cfg.dataset, cfg.partitioning, 10)
    assert len(ds1) == len(ds2) == len(ds3) == 50_000 // 20

    lbs1 = [d["label"] for d in ds1]
    lbs2 = [d["label"] for d in ds2]
    lbs3 = [d["label"] for d in ds3]

    assert lbs1 == lbs2 == lbs3
