from functools import partial

from hydra.utils import instantiate
from omegaconf import OmegaConf
from flwr_datasets import FederatedDataset
from datasets.utils.logging import disable_progress_bar


disable_progress_bar()


def _apply_transforms_fn(batch, transforms):
  batch["img"] = [transforms(img) for img in batch["img"]]
  return batch


def load_client_trainset(client_idx, dataset_config, partitioning_config, seed):
    train_transforms = instantiate(dataset_config.train_transforms)
    partitioner = instantiate(partitioning_config)

    fds = FederatedDataset(
        dataset=dataset_config.name,
        partitioners={"train": partitioner},
        seed=seed,
        keep_in_memory=False,
    )

    apply_train_transforms = partial(_apply_transforms_fn, transforms=train_transforms)

    trainset = fds.load_partition(client_idx)
    trainset = trainset.select_columns(["img", "label"])
    trainset = trainset.with_transform(apply_train_transforms)

    return trainset


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
    dataset = load_client_trainset(0, cfg.dataset, cfg.partitioning, 10)
    assert len(dataset) == 50_000 // 20
