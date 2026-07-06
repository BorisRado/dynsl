import gc
import time
import hydra
import torch
from hydra.utils import instantiate

from slbd.common.profiling import (
    profile_cumulative_per_layer_time_full_model,
    profile_cumulative_per_layer_time_partial_model,
    profile_cumulative_per_layer_time_individual_layers
)


def clear_resources():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(2.)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def run(cfg):
    iterations = 100 if torch.cuda.is_available() else 20
    dry_runs = 20 if torch.cuda.is_available() else 4
    model = instantiate(cfg.model)

    if torch.cuda.is_available():
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters())

    x = torch.randn(32, 3, 32, 32)
    if torch.cuda.is_available():
        x = x.cuda()
    for _ in range(dry_runs):
        start_time = time.perf_counter()
        optimizer.zero_grad()
        model(x).norm().backward()
        optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        print(f"Iter time: {end - start_time:.6f}")

    clear_resources()
    t1 = profile_cumulative_per_layer_time_full_model(model, 32, iterations, dry_runs)
    print("full_model =", t1)

    clear_resources()
    t2 = profile_cumulative_per_layer_time_partial_model(model, 32, iterations, dry_runs)
    print("partial_model =", t2)

    clear_resources()
    t3 = profile_cumulative_per_layer_time_individual_layers(model, 32, iterations, dry_runs)
    print("individual_layers =", t3)


if __name__ == "__main__":
    run()
