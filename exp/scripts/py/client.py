import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from slbd.client.app import start_client

from exp.src.sl.mo_client import MOTorchClient
from exp.src.common.other import set_torch_flags


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def run(cfg: DictConfig):
    # remove configuration that does not pertain to the client
    del cfg.server
    del cfg.strategy
    del cfg.optimizer

    print(OmegaConf.to_yaml(cfg, resolve=True))

    client = MOTorchClient(
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
