import time
import yaml

import torch
import hydra
from hydra.utils import instantiate

from slbd.torch.model_utils import extract_submodel

from exp.src.common.k8s import get_node_name
from exp.src.model.utils import warmup_device


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def run(cfg):
    model = instantiate(cfg.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert tuple(cfg.data.input_shape) == (3, 32, 32)

    if torch.cuda.is_available():
        warmup_device()

    split_points = model.meaningful_split_points
    num_iterations = 500
    out_data = {}
    for batch_size in [4, 8, 16, 32, 64]:
        out_data[batch_size] = {}
        for split_point in split_points:
            submodel = extract_submodel(model, 0, split_point)
            submodel.to(device)
            optimizer = torch.optim.Adam(submodel.parameters(), lr=0.001)
            start_time = time.perf_counter()
            for _ in range(num_iterations):
                optimizer.zero_grad()
                batch = torch.randn(batch_size, *cfg.data.input_shape)
                batch = batch.to(device)
                logits = submodel(batch)
                loss = logits.norm() # compute a random loss
                loss.backward()
                optimizer.step()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            out_data[batch_size][split_point] = (time.perf_counter() - start_time) / num_iterations

    out_data = {get_node_name(): out_data}
    print(yaml.dump(out_data, indent=4))



if __name__ == "__main__":
    run()