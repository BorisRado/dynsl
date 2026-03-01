import os

import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from slbd.client.app import start_client

from exp.src.sl.mo_client import MOTorchClient
from exp.src.common.other import set_torch_flags
from exp.src.common.k8s import get_all_pods_and_nodes, get_node_name
from exp.src.model.utils import warmup_device
from exp.src.data.loading import load_client_trainset


def handle_bg_process_config_colext(client_init_kwargs, cfg):
    if "COLEXT_SERVER_ADDRESS" not in os.environ:
        print("Not running on CoLExT -- ignoring bg-process override")
        return

    if client_init_kwargs.get("bg_num_procs", 0) > 0:
        if "bg_proc_devices" in cfg:
            nodes = list(get_all_pods_and_nodes().values()) # all nodes selected in this run
            print("ALL NODES", nodes)
            this_node = get_node_name()
            print("THIS NODE", this_node)
            has_been_selected = False
            for device_type, num_w_bg_procs in cfg.bg_proc_devices.items():
                device_nodes = [node for node in nodes if node.startswith(device_type)]
                selected_nodes = sorted(device_nodes)[:num_w_bg_procs]
                if this_node in selected_nodes:
                    has_been_selected = True
                    break
            if not has_been_selected:
                to_delete_keys = [k for k in client_init_kwargs if k.startswith("bg_")]
                for k in to_delete_keys:
                    del client_init_kwargs[k]


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def run(cfg: DictConfig):
    OmegaConf.resolve(cfg)

    # remove configuration that does not pertain to the client
    del cfg.server
    del cfg.strategy
    del cfg.optimizer
    if "COLEXT_SERVER_ADDRESS" in os.environ:
        print("Overriding server address and client ID with CoLExT configuration")
        cfg.client.server_address = os.environ["COLEXT_SERVER_ADDRESS"]
        cfg.client.init_kwargs.cid = int(os.environ["COLEXT_CLIENT_ID"])
        print("Server address:", cfg.client.server_address)
        print("This client CID:", cfg.client.init_kwargs.cid)

    client_init_kwargs = OmegaConf.to_container(cfg.client.init_kwargs)
    handle_bg_process_config_colext(client_init_kwargs, cfg)
    print("Client init kwargs:", client_init_kwargs)
    print(OmegaConf.to_yaml(cfg, resolve=True))

    trainset = load_client_trainset(
        client_idx=client_init_kwargs["cid"],
        dataset_config=cfg.data,
        partitioning_config=cfg.partitioning,
        seed=cfg.general.seed,
    )

    model = instantiate(cfg.model).to("meta")
    warmup_device(model)
    torch.cuda.empty_cache()
    assert next(iter(model.parameters())).device.type == "meta"
    client = MOTorchClient(
        **client_init_kwargs,
        trainset=trainset,
        seed=cfg.general.seed,
    )

    # NOTE!!! if passing `client` the same client will be used
    # if passing client_fn instead, the client will be re-initialized every the client
    # is selected for training or evaluation
    start_client(
        config=cfg,
        client=client,
        model=model,
    )
    if hasattr(client, "bg_proc_mng"):
        client.bg_proc_mng.terminate_processes()


if __name__ == "__main__":
    set_torch_flags()
    run()
