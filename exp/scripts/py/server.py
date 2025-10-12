import os

from dotenv import load_dotenv
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig

from slbd.server.app import start_server as start_async_server

from exp.src.common.wandb import init_wandb, finish_wandb, run_exists_already
from exp.src.common.stochasticity import set_seed
from exp.src.common.other import set_torch_flags
from exp.src.model.utils import warmup_device


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def run(cfg: DictConfig):
    if cfg.wandb:
        if run_exists_already(cfg):
            print("Run exists already")
            return

        init_wandb(cfg)
        print("Logging data to", os.getenv("WANDB_PROJECT"))

    # remove configuration that does not pertain to the server
    # del cfg.client # cannot remove it as we need batch size
    del cfg.partitioning

    print(OmegaConf.to_yaml(cfg))

    set_seed(cfg.general.seed)

    model = instantiate(cfg.model)
    warmup_device()

    start_kwargs = {
        "model": model,
        "config": cfg,
        "server_address": f"0.0.0.0:{cfg.server.port}",
    }
    assert cfg.server.async_
    start_async_server(**start_kwargs)
    finish_wandb()



if __name__ == "__main__":
    load_dotenv()
    set_torch_flags()
    run()
