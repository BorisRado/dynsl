import sys
import yaml
from collections import defaultdict

import numpy as np


def run(filename):
    with open(filename) as f:
        data = yaml.safe_load(f)

    averaged_dict = defaultdict(list)
    for k, v in data.items():
        print(f"Adding {k} with type {k[:-1]}")
        averaged_dict[k[:-1]].append(v)

    result = {}
    for device_type, dicts in averaged_dict.items():
        result[device_type] = {}
        for key in dicts[0].keys():
            values = [np.array(d[key]) for d in dicts]
            result[device_type][key] = (sum(values) / len(values)).tolist()
    output_filename = f"exp/conf/profiled_times/{filename}"
    with open(output_filename, 'w') as f:
        yaml.dump(result, f, default_flow_style=None)
    return result



if __name__ == "__main__":
    """First run `profile_model_layers.sh`
    Then run this script to average the times of the clients (e.g., `python average_layer_times.py resnet18.yaml`)
    Then move the output file to the `exp/conf/profiled_times` folder
    """
    assert len(sys.argv) == 2
    run(sys.argv[1])
