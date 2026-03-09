import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from omegaconf import DictConfig, OmegaConf


def init_optimizer(parameters, **kwargs):
    prefix_len = len("optim.")
    kwargs = {k[prefix_len:]: v for k, v in kwargs.items()}

    optimizer_name = kwargs.pop("optimizer_name")
    return {
        "adam": optim.Adam,
        "sgd": optim.SGD,
        "adamw": optim.AdamW,
    }[optimizer_name](parameters, **kwargs)


class OptimizerGenerator:
    def __init__(self, optimizer_cfg, scheduler_cfg):
        super().__init__()
        assert {"lr", "optimizer_name"}.issubset(optimizer_cfg.keys())
        assert {"strategy"}.issubset(scheduler_cfg.keys())

        if isinstance(optimizer_cfg, DictConfig):
            optimizer_cfg = OmegaConf.to_container(optimizer_cfg)
        if isinstance(scheduler_cfg, DictConfig):
            scheduler_cfg = OmegaConf.to_container(scheduler_cfg)

        self._epoch = 0

        # extract optimizer config
        self._initial_lr = optimizer_cfg["lr"]
        self._optim_config = optimizer_cfg

        # extract lr decay config
        self._decay_strategy = scheduler_cfg.pop("strategy")
        self._scheduler_config = scheduler_cfg

    def get_optimizer_config(self):
        return {
            f"optim.{k}": v for k, v in self._optim_config.items()
        }

    def end_epoch_callback(self):
        self._epoch += 1

        if self._decay_strategy == "constant":
            return

        elif self._decay_strategy == "step":
            # every `step_size` steps multiply the learning rate by `gamma`
            if self._epoch % self._scheduler_config["step_size"] == 0:
                self._optim_config["lr"] *= self._scheduler_config["gamma"]

        elif self._decay_strategy == "cosine":
            T_max = self._scheduler_config.get("T_max")
            eta_min = self._scheduler_config.get("eta_min")
            eta_max = self._initial_lr
            T_cur = self._epoch
            self._optim_config["lr"] = \
                eta_min + 0.5 * (eta_max - eta_min) * (1 + np.cos((T_cur / T_max) * np.pi))

    def get_current_lr(self):
        return self._optim_config["lr"]


def create_classification_head(encoder_model, num_classes, fc_dims=[256,]):
    shp = list(encoder_model.parameters())[-1].shape
    layers = []
    layers.append(nn.AdaptiveAvgPool2d(2))
    layers.append(nn.Flatten())
    if len(shp) == 1:
        prev_shape = shp[0] * 4
    else:
        print("Warning -- last parameter should be single dimensional...")
        avg = nn.AdaptiveAvgPool2d(2)
        prev_shape = avg(encoder_model(torch.randn(2, 3, 32, 32))).numel() // 2

    for fc_dim in fc_dims:
        layers.append(nn.Linear(prev_shape, fc_dim))
        layers.append(nn.ReLU())

    layers.append(nn.Linear(fc_dims[-1], num_classes))

    clf_head = nn.Sequential(*layers)

    # Re-initialize model weights using PyTorch's default initialization with a generator
    g = torch.Generator().manual_seed(42)
    for m in clf_head.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=np.sqrt(5), generator=g)
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(m.bias, -bound, bound, generator=g)
    return clf_head


if __name__ == "__main__":
    # check that the learning rates returned by the optimizer generator match
    # the ones of plain pytorch -- here we test cosine scheduling

    optimizer_cfg = {
        "optimizer_name": "adam",
        "lr": 0.1
    }
    scheduler_cfg = {
        "strategy": "cosine",
        "eta_min": 1e-10,
        "T_max": 200,
    }

    optimizer = optim.Adam(torch.nn.Linear(1, 1).parameters(), lr=optimizer_cfg["lr"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=scheduler_cfg["T_max"], eta_min=scheduler_cfg["eta_min"]
    )

    lrs = []
    for _ in range(500):
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()

    generator = OptimizerGenerator(optimizer_cfg, scheduler_cfg)
    lrs2 = []
    for _ in range(500):
        lrs2.append(generator.get_current_lr())
        generator.end_epoch_callback()

    for a, b in zip(lrs, lrs2):
        assert np.abs(a - b) < 1e-10

    print("Cosine scheduling works just fine!")
