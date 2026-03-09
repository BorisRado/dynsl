import os
import time
import psutil

import torch
import torch.nn.functional as F
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from slbd.common.utils import measure_time

from exp.src.model.utils import warmup_device
from exp.src.common.background_process_manager import BackgroundProcessManager
from exp.src.common.k8s import get_node_name


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


def set_cpu_affinity(core):
    node_name = get_node_name()
    print("node", node_name)
    # if node_name.startswith("jao") or node_name.startswith("jon"):
    #     print("Setting CPU affinity")
    #     os.sched_setaffinity(0, {core})


def get_temperature(core):
    temps = psutil.sensors_temperatures()

    # Most Intel/AMD CPUs expose per-core temps under "coretemp"
    coretemps = temps.get("coretemp")

    if not coretemps:
        raise RuntimeError("coretemp sensor not available")

    # Match "Core X"
    for entry in coretemps:
        if entry.label == f"Core {core}":
            temperature_c = float(entry.current)
            break
    else:
        raise RuntimeError(f"Temperature for Core {core} not found")

    return temperature_c


@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def run(cfg):
    OmegaConf.resolve(cfg)
    print(OmegaConf.to_yaml(cfg))
    CORE = 0
    set_cpu_affinity(CORE)

    try:
        _ = get_temperature(CORE)
        measure_temperature = True
    except Exception as e:
        print(e)
        measure_temperature = False
    print("Measuring temperature?", measure_temperature)

    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    batch_size = cfg.client.init_kwargs.batch_size

    model = instantiate(cfg.model)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    assert tuple(cfg.data.input_shape) == (3, 32, 32)

    if cuda_available:
        print("Cuda is available... => Warming up divece")
        warmup_device(model)

    for _ in range(20):
        run_training_iteration(model, optimizer, device, batch_size)

    num_iterations_per_configuration = 100
    out_results = []
    temperatures = []
    frequencies = []

    tot_idx = 0
    for num_bg_processes in [0, 1, 0, 2, 4, 0]:
        print(f"RUNNING {num_bg_processes} bg processes", flush=True)
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
            if measure_temperature and tot_idx % 20 == 0:
                tmp_temp = get_temperature(CORE)
                temperatures.append(tmp_temp)
                print(f"Temperature {tmp_temp}")
            if tot_idx % 20 == 0:
                frequencies.append(psutil.cpu_freq().current)
            tot_idx += 1
        out_results.append((num_bg_processes, current_training_times))
        manager.terminate_processes()
        if measure_temperature:
            tmp_temp = get_temperature(CORE)
            temperatures.append(tmp_temp)
            print(f"Temperature {tmp_temp}")
        time.sleep(0.2)

    client_device_type = os.environ.get("COLEXT_DEVICE_TYPE", "??")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"RESULTS [{client_device_type} {gpu_name}]: {out_results}")
    print(f"TEMPERATURES [{client_device_type} {gpu_name}]: {temperatures}")
    print(f"FREQUENCIES [{client_device_type} {gpu_name}]: {frequencies}")


if __name__ == "__main__":
    run()
