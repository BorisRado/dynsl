import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from exp.src.common.other import set_torch_flags
from exp.src.model.utils import warmup_device
from slbd.common.profiling import measure_per_layer_complexity
from slbd.torch.utils import OptimizerGenerator, init_optimizer


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def run(cfg: DictConfig):

    model = instantiate(cfg.model)
    warmup_device()
    optim_generator = OptimizerGenerator(cfg.optimizer.optimizer, cfg.optimizer.scheduler)
    optimizer = init_optimizer(
        model.parameters(),
        **optim_generator.get_optimizer_config()
    )
    complexities = measure_per_layer_complexity(
        model=model,
        optimizer=optimizer,
        batch_size=cfg.client.init_kwargs.batch_size,
        iterations=100,
        dry_runs=20,
    )
    device_type = os.environ.get("COLEXT_DEVICE_TYPE", "SEERVER")
    print(f"RESULTS [{device_type}]: {complexities}")


if __name__ == "__main__":
    set_torch_flags()
    run()
