import os
import time
import asyncio

import torch
from dotenv import load_dotenv
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig

from slbd.server.app import start_server as start_async_server

from exp.src.common.wandb import init_wandb, finish_wandb, run_exists_already
from exp.src.common.stochasticity import set_seed
from exp.src.common.other import set_torch_flags
from exp.src.model.utils import test_model
from exp.src.model.utils import warmup_device
from exp.src.data.loading import load_cv_testset
from exp.src.data.utils import get_dataloader


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def run(cfg: DictConfig):

    try:
        # try using uvloop if possible
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except:
        pass

    OmegaConf.resolve(cfg)
    if cfg.wandb:
        if run_exists_already(cfg):
            print("Run exists already")
            return

        init_wandb(cfg)
        print("Logging data to", os.getenv("WANDB_PROJECT"))

    testloader = torch.utils.data.DataLoader(
        load_cv_testset(cfg.data), batch_size=256, shuffle=False, pin_memory=True, num_workers=2
    )

    # remove configuration that does not pertain to the server
    # del cfg.client # cannot remove it as we need batch size
    print(OmegaConf.to_yaml(cfg))
    del cfg.partitioning

    set_seed(cfg.general.seed)

    model = instantiate(cfg.model)
    warmup_device(model)

    start_kwargs = {
        "model": model,
        "config": cfg,
        "server_address": f"0.0.0.0:{cfg.server.port}",
        "model_evaluate_fn": lambda model: test_model(model, testloader)
    }
    assert cfg.server.async_
    start_async_server(**start_kwargs)
    finish_wandb()
    print("Experiment completed!")
    time.sleep(10.)
    os._exit(0)



if __name__ == "__main__":
    load_dotenv()
    set_torch_flags()
    run()
