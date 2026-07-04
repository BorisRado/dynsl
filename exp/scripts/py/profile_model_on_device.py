import yaml

import torch
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from slbd.common.profiling import profile_cumulative_per_layer_time_individual_layers

from exp.src.common.k8s import get_node_name
from exp.src.model.utils import warmup_device


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def run(cfg):
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))

    model = instantiate(cfg.model)
    assert tuple(cfg.data.input_shape) == (3, 32, 32)

    warmup_device(model)

    num_iterations = 100 if torch.cuda.is_available() else 30
    dry_runs = 20 if torch.cuda.is_available() else 10
    out_data = {}
    for batch_size in [4, 8, 16, 32]:
        print("Working with batch size:", batch_size, flush=True)
        times = profile_cumulative_per_layer_time_individual_layers(model, batch_size, num_iterations, dry_runs=dry_runs)
        out_data[batch_size] = times

    out_data = {get_node_name(): out_data}
    print("========== START OF ACTUAL RESULTS\n")
    print(yaml.dump(out_data, default_flow_style=None))
    print("========== END OF ACTUAL RESULTS")



if __name__ == "__main__":
    def dict_representer(dumper, data):
        return dumper.represent_mapping(
            'tag:yaml.org,2002:map', data, flow_style=False
        )
    yaml.add_representer(dict, dict_representer)
    run()
