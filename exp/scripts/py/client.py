from functools import partial

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from slbd.client.app import start_client

from exp.src.sl.mo_client import MOTorchClient


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def run(cfg: DictConfig):
    # remove configuration that does not pertain to the client
    del cfg.server
    del cfg.strategy
    del cfg.optimizer

    print(OmegaConf.to_yaml(cfg, resolve=True))
    client_fn = partial(
        MOTorchClient,
        dataset_config=cfg.data,
        partitioning_config=cfg.partitioning,
        seed=cfg.general.seed
    )

    start_client(
        config=cfg,
        client_fn=client_fn,
        model=instantiate(cfg.model),
    )


if __name__ == "__main__":
    run()
