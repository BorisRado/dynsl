import time
import random

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import torch

from slbd.client.app import start_client
from slbd.client.abstract_torch_client import AbstractTorchClient

from exp.src.common.other import set_torch_flags


class StupidTorchClient(AbstractTorchClient):

    def __init__(
        self,
        cid,
        min_layers,
        batch_size,
        iterations,
        shape,
        dataset_config,
        partitioning_config,
        seed,
        simulated_compute_time=0.0
    ):
        # client for testing - it just sends random data to the server, performs no training
        _ = (dataset_config, partitioning_config, seed, )
        self.cid = int(cid)
        self.min_layers = min_layers
        self.batch_size = batch_size
        self.shape = tuple(int(a) for a in shape.split("-"))
        print(self.shape)
        self.iterations = iterations
        self.num_classes = dataset_config.num_classes
        self.simulated_compute_time = simulated_compute_time

    def fit(self, config, model_trainer):

        _ = (config, )
        overall_shape = (self.batch_size, ) + self.shape
        print(overall_shape)
        x = torch.randn(*overall_shape)
        y = torch.randint(0, self.num_classes, (self.batch_size,))
        for _ in range(self.iterations):
            model_trainer.stupid_forward(x, y)
            if self.simulated_compute_time > 0.005:
                time.sleep(self.simulated_compute_time + random.uniform(-0.005, 0.005))
            elif self.simulated_compute_time > 0.:
                time.sleep(self.simulated_compute_time)

    def evaluate(self, config, model_trainer):
        raise Exception

    def get_trainloader_size(self):
        return self.iterations



@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def run(cfg: DictConfig):
    # remove configuration that does not pertain to the client
    del cfg.server
    del cfg.strategy
    del cfg.optimizer

    print(OmegaConf.to_yaml(cfg, resolve=True))

    client = StupidTorchClient(
        **OmegaConf.to_container(cfg.client.init_kwargs),
        dataset_config=cfg.data,
        partitioning_config=cfg.partitioning,
        seed=cfg.general.seed
    )

    # NOTE!!! if passing `client` the same client will be used
    # if passing client_fn instead, the client will be re-initialized every the client
    # is selected for training or evaluation
    start_client(
        config=cfg,
        client=client,
        model=instantiate(cfg.model),
    )


if __name__ == "__main__":
    set_torch_flags()
    run()
