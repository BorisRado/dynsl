import os
import json
import wandb
from hydra.core.config_store import OmegaConf


def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def get_value(config, key):
    key_list = key.split(".")
    v = config
    for k in key_list:
        v = v[k]
    return v


def init_wandb(cfg, further_config_dict={}):
    config = OmegaConf.to_container(cfg)
    flat_config = flatten_dict(config) | further_config_dict
    wandb.init(
        config=flat_config,
    )


def finish_wandb():
    if wandb.run is not None:
        wandb.finish()


def run_exists_already(config):
    def dict_to_consistent_string(d):
        return json.dumps(d, sort_keys=True)

    if not config.wandb:
        return False  # as we are debugging

    tmp_conf = flatten_dict(OmegaConf.to_container(config))
    new_conf = dict_to_consistent_string(tmp_conf)

    # bug in wandb: https://github.com/wandb/wandb/issues/7666
    # another one https://github.com/wandb/wandb/issues/3087
    for filename in os.listdir(".wandb_cache"):
        with open(f".wandb_cache/{filename}", "r") as fp:
            comparison_config = json.load(fp)
        if dict_to_consistent_string(comparison_config) == new_conf:
            return True

    return False
