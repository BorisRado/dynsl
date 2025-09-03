import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig

from slbd.server.app import start_server

from exp.src.common.stochasticity import set_seed
from exp.src.common.other import set_torch_flags


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def run(cfg: DictConfig):
    # remove configuration that does not pertain to the server
    del cfg.client
    del cfg.partitioning

    print(OmegaConf.to_yaml(cfg))

    set_seed(cfg.general.seed)

    model = instantiate(cfg.model)

    start_server(
        model=model,
        config=cfg,
        server_address=f"0.0.0.0:{cfg.server.port}",
    )


if __name__ == "__main__":
    set_torch_flags()
    run()
