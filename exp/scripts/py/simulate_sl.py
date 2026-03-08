import os
import time
import copy
import logging
from logging import INFO
from dotenv import load_dotenv

import torch
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig
from flwr.common import log

from slbd.common.constants import CLIENT_ID_CONFIG_KEY
from slbd.algos.dynsl import DynSLStrategy
from slbd.algos.dynsl.simulation_layer_allocator import DynSLSimulationClientLayerAllocator
from slbd.algos.fedavg import FedAvgStrategy, FedAvgClientLayerAllocator
from slbd.algos.splitfed import SplitFedStrategy, SplitFedClientLayerAllocator
from slbd.algos.dtfl import DTFLStrategy, DTFLSimulationClientLayerAllocator
from slbd.algos.oracle import OracleSimulationClientLayerAllocator, OracleStrategy
from slbd.algos.sfsl import SFSLClientLayerAllocator, SFSLStrategy
from slbd.torch.model_utils import get_num_layers
from slbd.server.server_model.manager.multithreading_server_model_manager import MultithreadingServerModelManager
from slbd.torch.utils import create_classification_head
from slbd.torch.model_utils import extract_submodel

from exp.src.common.wandb import init_wandb, finish_wandb, run_exists_already
from exp.src.common.stochasticity import set_seed
from exp.src.common.other import set_torch_flags
from exp.src.data.loading import load_client_trainset, load_cv_testset
from exp.src.data.utils import get_dataloader
from exp.src.model.utils import test_model

from exp.src.sl.mo_client import MOTorchClient
from exp.src.sl.simulation import (
    SimulationClientManager,
    SimulationClientProxy,
    SLExecutor,
)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def run(cfg: DictConfig):
    OmegaConf.resolve(cfg)

    # remove what is not needed in simulations
    del cfg.client.server_address
    del cfg.server
    del cfg.client.communication_channel
    del cfg.client.socket_port
    del cfg.server_model_manager
    del cfg.server_model_servicer.servicer_name

    # sanity checks
    assert len([v for v in cfg.client.init_kwargs.keys() if v.startswith("bg_")]) == 0, \
        "Makes no sense to use background processes when using simulations"

    set_seed(cfg.general.seed)
    model = instantiate(cfg.model)
    client_clf_head = None
    if cfg.algorithm.name == "sfsl":
        encoder = extract_submodel(model, 0, cfg.algorithm.num_client_layers)
        client_clf_head = create_classification_head(encoder, cfg.data.num_classes, cfg.algorithm.strategy_init_args.clf_head_dimensions)
    elif cfg.algorithm.name.startswith("dtfl"):
        client_clf_head = {}
        for sp in cfg.algorithm.strategy_init_args.all_split_points:
            encoder = extract_submodel(model, 0, sp)
            sp_head = create_classification_head(encoder, cfg.data.num_classes, cfg.algorithm.strategy_init_args.clf_head_dimensions)
            client_clf_head[sp] = sp_head
    elif cfg.algorithm.name.startswith("dynsl") and cfg.algorithm.end_epoch_weight_integration_strategy == "immediate":
        assert cfg.general.get("training_speed_imbalance_value") > 0

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
        keep_in_memory=cfg.data.load_onto_memory,
    )
    testloader = torch.utils.data.DataLoader(
        load_cv_testset(cfg.data),
        batch_size=512,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )

    clients = [
        SimulationClientProxy(cid, MOTorchClient(
            **OmegaConf.to_container(cfg.client.init_kwargs),
            trainset=ts,
            seed=cfg.general.seed,
        )) for cid, ts in enumerate(trainsets)
    ]

    # instantiate the layer allocator and the strategy
    strategy = {
        "dynsl_simulation": DynSLStrategy,
        "dynsl_w_profiling_simulation": DynSLStrategy,
        "splitfed_v1": SplitFedStrategy,
        "splitfed_v2": SplitFedStrategy,
        "dtfl_simulation": DTFLStrategy,
        "sfsl": SFSLStrategy,
        "oracle": OracleStrategy,
        "fedavg": FedAvgStrategy,
    }[cfg.algorithm.name](
        model=model,
        optimizer_config=OmegaConf.to_container(cfg.optimizer),
        **OmegaConf.to_container(cfg.strategy, resolve=True),
        **OmegaConf.to_container(cfg.algorithm.strategy_init_args, resolve=True),
        model_evaluate_fn=lambda model: test_model(model, testloader)
    )

    client_layer_allocator = {
        "dynsl_simulation": DynSLSimulationClientLayerAllocator,
        "dynsl_w_profiling_simulation": OracleSimulationClientLayerAllocator,
        "splitfed_v1": SplitFedClientLayerAllocator,
        "splitfed_v2": SplitFedClientLayerAllocator,
        "dtfl_simulation": DTFLSimulationClientLayerAllocator,
        "sfsl": SFSLClientLayerAllocator,
        "oracle": OracleSimulationClientLayerAllocator,
        "fedavg": FedAvgClientLayerAllocator
    }[cfg.algorithm.name](
        num_model_layers=get_num_layers(model),
        **OmegaConf.to_container(cfg.algorithm.layer_allocator_init_args, resolve=True),
        meaningful_split_points=model.meaningful_split_points,
    )

    client_manager = SimulationClientManager(clients)
    server_model_manager = MultithreadingServerModelManager(
        model=copy.deepcopy(model),
        exp_config=None,
    )
    server_model_manager.set_strategy(strategy)

    training_speed_imbalance_value = cfg.general.get("training_speed_imbalance_value", None)
    if training_speed_imbalance_value is None:
        print("Imbalance factor set to None -- considering uniform order of training")
    sl_executor = SLExecutor(
        server_model_manager=server_model_manager,
        client_layer_allocator=client_layer_allocator,
        strategy=strategy,
        request_group_timeout=cfg.server_model_servicer.request_group_timeout,
        num_concat_batches=cfg.server_model_servicer.num_concat_batches,
        training_speed_imbalance_value=training_speed_imbalance_value,
        client_optim_state_resetting_frequency=cfg.general.get("optim_reset_frequency", -1),
        end_epoch_weight_integration_strategy=cfg.algorithm.get("end_epoch_weight_integration_strategy", None),
        whole_model=copy.deepcopy(model),
        client_clf_head=client_clf_head,
        server_exchange_optimizer_states=cfg.algorithm.get("exchange_optimizer_states", None),
        client_exchange_optimizer_states=cfg.algorithm.get("client_adapter_args", {}).get("exchange_optimizer_states", None),
    )

    model_state_dict = model.state_dict()

    start_time = time.time()

    all_accuracies = []
    for epoch_idx in range(1, cfg.general.rounds+1):
        client_instructions, server_model_configs = strategy.configure_fit(
            server_round=epoch_idx,
            model_state_dict=model_state_dict,
            client_manager=client_manager,
            client_layer_allocator=client_layer_allocator,
        )

        client_instructions = [(proxy, copy.deepcopy(ins)) for proxy, ins in client_instructions]
        for proxy, ins in client_instructions:
            ins.config[CLIENT_ID_CONFIG_KEY] = proxy.cid

        log(
            INFO,
            "fit_round %s: strategy sampled %s clients (out of %s)",
            epoch_idx,
            len(client_instructions),
            client_manager.num_available(),
        )

        server_model_manager.initialize_server_models(server_model_configs, is_training=True)

        # Collect `fit` results from all clients participating in this round
        results = sl_executor.fit_clients_simulation(
            client_instructions=client_instructions,
            has_batch_concatenation=cfg.server_model_servicer.num_concat_batches > 1
        )
        server_fit_res = server_model_manager.collect_server_fit_results()
        model_state_dict, _ = strategy.aggregate_fit(
            server_round=epoch_idx,
            client_results=results,
            server_results=server_fit_res,
            layer_allocator=client_layer_allocator,
            failures=[],
        )
        test_metrics = strategy.evaluate(
            server_round=epoch_idx,
            model_state_dict=model_state_dict
        )[1]
        client_layer_allocator.finalize_epoch()
        all_accuracies.append(test_metrics["test/accuracy"])
        print(epoch_idx, all_accuracies)
        if strategy.has_converged:
            print("Early exit....")
            break
    print(f"Total runtime {time.time() - start_time:.2f}")
    sl_executor.shutdown()
    finish_wandb()



if __name__ == "__main__":
    load_dotenv()
    set_torch_flags()
    logging.getLogger("flwr").setLevel("ERROR")
    run()
