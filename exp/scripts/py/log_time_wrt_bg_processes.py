import os
import time

import torch
import torch.nn.functional as F
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from slbd.torch.model_utils import extract_submodel
from slbd.common.utils import measure_time

from exp.src.common.k8s import get_node_name
from exp.src.model.utils import warmup_device
from exp.src.common.background_process_manager import BackgroundProcessManager


def generate_random_batch(batch_size):
    return torch.randn(batch_size, 3, 32, 32), torch.randint(0, 10, (batch_size,))


def run_training_iteration(model, optimizer, device, batch_size):
    imgs, labels = generate_random_batch(batch_size)
    imgs, labels = imgs.to(device), labels.to(device)
    optimizer.zero_grad()
    logits = model(imgs)
    loss = F.cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def run(cfg):
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))

    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    batch_size = cfg.client.init_kwargs.batch_size

    model = instantiate(cfg.model)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    assert tuple(cfg.data.input_shape) == (3, 32, 32)

    if cuda_available:
        print("Cuda is available... => Warming up divece")
        warmup_device()

    for _ in range(20):
        run_training_iteration(model, optimizer, device, batch_size)

    num_iterations_per_configuration = 100
    out_results = []

    for num_bg_processes in [0, 1, 0, 2, 4, 0]:
        # we start the given number of background processes
        manager = BackgroundProcessManager(num_bg_processes, "cpu")
        manager.resume_processes(0., 1e8)
        time.sleep(0.2) # so that the processes start

        # train on random data
        current_training_times = []
        for _ in range(num_iterations_per_configuration):
            with measure_time(cuda_available) as timer:
                run_training_iteration(model, optimizer, device, batch_size)
            current_training_times.append(timer.elapsed)
        out_results.append((num_bg_processes, current_training_times))
        manager.terminate_processes()
        time.sleep(0.2)

    print("RESULTS", out_results)
    client_device_type = os.environ.get("COLEXT_DEVICE_TYPE", "??")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"RESULTS [{client_device_type} {gpu_name}]: {out_results}")


if __name__ == "__main__":
    run()
