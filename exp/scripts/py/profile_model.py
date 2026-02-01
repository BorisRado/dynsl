import os
import time

import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from exp.src.common.other import set_torch_flags
from exp.src.model.utils import warmup_device
from slbd.common.profiling import measure_per_layer_complexity
from slbd.torch.utils import OptimizerGenerator, init_optimizer

from exp.src.common.background_process_manager import BackgroundProcessManager


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def run(cfg: DictConfig):
    print("PROFILING MODEL >>>")

    model = instantiate(cfg.model)
    num_bg_processes = cfg.get("num_bg_processes", 0)
    print(f"Setting number of background processes to: {num_bg_processes}")
    if num_bg_processes > 0:
        manager = BackgroundProcessManager(
            num_processes=num_bg_processes,
            bg_device=torch.device("cpu")
        )
        manager.resume_processes(delay=0., duration=1e8)
        time.sleep(1.0)

    warmup_device()

    print("Number of model parameters: ", sum(p.numel() for p in model.parameters()))
    optim_generator = OptimizerGenerator(cfg.optimizer.optimizer, cfg.optimizer.scheduler)
    optimizer = init_optimizer(
        model.parameters(),
        **optim_generator.get_optimizer_config()
    )
    complexities, _ = measure_per_layer_complexity(
        model=model,
        optimizer=optimizer,
        batch_size=cfg.client.init_kwargs.batch_size,
        iterations=200,
        dry_runs=50,
    )
    client_device_type = os.environ.get("COLEXT_DEVICE_TYPE", "SERVER")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"RESULTS [{client_device_type} {device}]: {complexities}")

    time.sleep(10.0)
    if num_bg_processes > 0:
        manager.terminate_processes()


if __name__ == "__main__":
    set_torch_flags()
    run()
