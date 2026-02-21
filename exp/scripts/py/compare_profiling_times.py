import time
import hydra
import torch
from hydra.utils import instantiate

from slbd.common.profiling import (
    profile_cumulative_per_layer_time_full_model,
    profile_cumulative_per_layer_time_partial_model,
    profile_cumulative_per_layer_time_individual_layers
)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def run(cfg):
    iterations = 200 if torch.cuda.is_available() else 50
    dry_runs = 50 if torch.cuda.is_available() else 10
    model = instantiate(cfg.model)
    t1 = profile_cumulative_per_layer_time_full_model(model, "adam", 16, iterations, dry_runs)
    print("full_model=", t1)
    time.sleep(2.)
    t2 = profile_cumulative_per_layer_time_partial_model(model, "adam", 16, iterations, dry_runs)
    print("partial_model=", t2)
    time.sleep(2.)
    t3 = profile_cumulative_per_layer_time_individual_layers(model, "adam", 16, iterations, dry_runs)
    print("individual_layers=", t3)

    if torch.cuda.is_available():
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters())

    x = torch.randn(16, 3, 32, 32)
    if torch.cuda.is_available():
        x = x.cuda()
    for _ in range(dry_runs):
        optimizer.zero_grad()
        model(x).norm().backward()
        optimizer.step()
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    for _ in range(iterations):
        optimizer.zero_grad()
        model(x).norm().backward()
        optimizer.step()
    end = time.perf_counter()
    print(f"Time {(end - start_time) / iterations:.6f}")


if __name__ == "__main__":
    run()
