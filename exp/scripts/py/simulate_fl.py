import os
from dotenv import load_dotenv

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig

from exp.src.common.wandb import init_wandb, finish_wandb, run_exists_already
from exp.src.common.stochasticity import set_seed
from exp.src.common.other import set_torch_flags
from exp.src.data.loading import load_client_trainset, load_cv_testset
from exp.src.data.utils import get_dataloader
from exp.src.fl.train import train_flower, CustomFedAvg



@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def run(cfg: DictConfig):
    OmegaConf.resolve(cfg)

    # remove what is not needed in simulations
    del cfg.client.server_address
    del cfg.server
    del cfg.client.socket_port

    # sanity checks
    assert cfg.algorithm.name == "fedavg"

    set_seed(cfg.general.seed)
    model = instantiate(cfg.model)

    if cfg.wandb:
        if run_exists_already(cfg):
            print("Run exists already")
            return

        init_wandb(cfg)
        print("Logging data to", os.getenv("WANDB_PROJECT"))

    print(OmegaConf.to_yaml(cfg, resolve=True))

    trainsets = load_client_trainset(
        client_idx=list(range(cfg.partitioning.num_partitions)),
        dataset_config=cfg.data,
        partitioning_config=cfg.partitioning,
        seed=cfg.general.seed,
    )

    testloader = get_dataloader(load_cv_testset(cfg.data), batch_size=256, shuffle=False)

    # instantiate the layer allocator and the strategy
    strategy = CustomFedAvg(
        model=model,
        optimizer_config=OmegaConf.to_container(cfg.optimizer),
        testloader=testloader,
        **OmegaConf.to_container(cfg.strategy, resolve=True),
    )
    print("client resources: ", {"num_cpus": 4, "num_gpus": 0.5})
    train_flower(
        model=model,
        client_resources={"num_cpus": 4, "num_gpus": 0.5},
        trainsets=trainsets,
        strategy=strategy,
        n_rounds=cfg.general.rounds,
        seed=cfg.general.seed,
        client_init_kwargs=OmegaConf.to_container(cfg.client.init_kwargs)
    )

    finish_wandb()
    print("Simulation ended")



if __name__ == "__main__":
    load_dotenv()
    set_torch_flags()
    run()
