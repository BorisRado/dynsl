# DynSL: Resource-Adaptive Model Training across Heterogeneous Edge Devices

This repository contains the framework and experimental code accompanying our paper, *DynSL: Resource-Adaptive Model Training across Heterogeneous Edge Devices*.

DynSL is a **dynamic split-learning** framework. A neural network is split between edge clients and a compute server: each client trains the first *N* layers locally, and the server trains the rest. Unlike prior split-learning approaches, DynSL does not commit to a single, upfront split. Instead, a layer allocator on the server continuously monitors client compute, network bandwidth, and server load, periodically solves a lightweight optimization problem, and **reallocates layers between client and server mid-epoch** whenever doing so lowers the estimated epoch time. This lets the system react to runtime dynamics — a background process slowing a client, a link degrading, a growing client population contending for the server — without ever pausing training.

On a real-world testbed of 24 heterogeneous edge devices, DynSL cuts training time by up to ~30% over the strongest adaptive baseline and by more than 50% over an uninformed fixed split, while leaving the computation (and hence convergence) unchanged. See the paper for the full evaluation.

Most experiments run on [CoLExT](https://sands.kaust.edu.sa/project/colext/), a public federated-learning testbed.

## Repository layout

The code is organized into two layers:

- **`slbd/`** — the reusable framework, installable as a package (`pip install -e .`). This is self-contained and never depends on the experiment harness.
- **`exp/`** — the experiment harness: [Hydra](https://hydra.cc/) configs, dataset loading and partitioning, model architectures, logging, and the CoLExT launch scripts.

The framework and the experiments are decoupled on purpose, so the DynSL algorithm can be reused independently of the specific evaluation setup in the paper.

## Setup

```bash
pip install -r requirements.txt   # pinned dependencies (torch 2.4.1, flwr 1.9.0, hydra, ...)
pip install -e .                  # install the slbd package
```

Weights & Biases (`WANDB_*`) and Hugging Face (`HF_TOKEN`) credentials, if used, are read from a `.env` file at the repository root.

## Running experiments

Everything is run from the repository root, e.g. `PYTHONPATH=. python exp/scripts/py/<script>.py`. All entry points read `exp/conf/config.yaml` and accept standard Hydra overrides.

**Simulation.** The fastest way to iterate: virtual clients run as Python objects in a single process.

```bash
PYTHONPATH=. python exp/scripts/py/simulate_sl.py algorithm=dynsl_simulation model=resnet18 data=cifar10
```

**Distributed.** Real client and server processes communicating over the network. Start the server first, then launch one or more clients:

```bash
PYTHONPATH=. python exp/scripts/py/server.py
PYTHONPATH=. python exp/scripts/py/client.py client.init_kwargs.cid=0
```

Common overrides:

```bash
algorithm=dynsl_v2                 # run the DynSL algorithm
algorithm=splitfed_v1              # or a baseline (splitfed_v1, ares, sfsl, dtfl, ...)
optimizer=adam                     # optimizer choice
client.init_kwargs.batch_size=32   # batch size
strategy.num_server_models=1       # one shared server head (min layers); -1 = one head per client
```

**CoLExT testbed.** After requesting testbed access, generate a job configuration and launch it. The generator writes `colext_config.yaml`, which `colext_launch_job` then runs across the device fleet:

```bash
python exp/scripts/py/generate_colext_config.py --algorithm=dynsl_v2 --model-name=resnet18 --num-server-models=1
colext_launch_job
```

`python exp/scripts/py/generate_colext_config.py --help` lists every configurable knob.

For convenience, the full experiment sweeps used in the paper are wrapped in shell scripts. Requirements are installed automatically and the experiment starts immediately on the devices:

```bash
./exp/scripts/sh/colext_separate_heads.sh   # separate-head experiments (Section 6.1)
./exp/scripts/sh/colext_common_head.sh      # shared-head convergence experiments (Section 6.2)
```

## Reproducing the paper's results

`REPRODUCING_RESULTS.md` maps individual tables and figures to the exact scripts and commands that generate them. In short, the mapping is:

| Result | Script |
| --- | --- |
| Table 3, Figures 7–8 | `exp/scripts/new_sh/separate_heads.sh` |
| Table 4 (compute slowdown) | `colext_separate_heads_slowdown_gpu.sh`, `colext_separate_heads_bg_procs.sh` |
| Table 5 (network slowdown) | `exp/scripts/sh/exp_w_reduced_throughput.sh` |
| Tables 7–8 (single-head accuracy) | `exp/scripts/slurm/comparison_single_head.sh` |
| Figure 2 (motivational) | `exp/scripts/sh/motivation_runtime_wrt_num_ops.sh` |
| Figures 3, 5 (profiling) | `exp/scripts/sh/profile_model_on_devices.sh` |
| Figure 6 (bandwidth estimation) | `exp/scripts/sh/test_throughput_single_client.sh` |

## How the algorithms are structured

Every algorithm lives under `slbd/algos/<name>/` and provides two cooperating classes, following [Flower](https://flower.ai/) terminology:

1. **Strategy** — samples clients at the start of each epoch, slices the global model into the per-client and server-side layers, and aggregates the returned weights (layer-wise weighted averaging, FedAvg-style).
2. **LayerAllocator** — decides how many layers each client starts the epoch with, and is polled during the epoch to decide whether to move layers between client and server. For every algorithm *except* DynSL this decision is static; DynSL is the only one that reconfigures the split mid-epoch.

Alongside DynSL (`dynsl_v2`, `dynsl_w_profiling`) the repository implements the baselines used in the evaluation: `splitfed` (v1/v2), `ares`, `sfsl`, `dtfl`, `oracle`, and `fedavg`. The DynSL allocation optimizers live in `slbd/algos/dynsl/optimization_algorithms.py`.

At runtime, the *controller* that orchestrates communication between clients, classification heads, and the allocator (Section 5.1 of the paper) is implemented by the `BatchingWorkerManager` class in `slbd/server/batching_worker_manager.py`.

**Notes on the baselines.** All baselines are reimplemented within this framework so that every method shares the same training loop, batching, and aggregation, and differs only in how it allocates layers. We follow the original papers as faithfully as possible. For ARES, the objective is the weighted sum of training time and energy from Eq. (7) of its paper, governed by a hyperparameter α; since DynSL targets training time, we set α=1 to consider time alone (see `slbd/algos/ares/utils.py`). This keeps the algorithm unchanged while placing ARES on the most favorable footing for the metric under comparison, so that the difference reflects the allocation strategy — server-aware vs. server-blind — rather than a mismatch in objectives.

Every model is expressed as a `slbd.torch.sequential_model.FlexibleSequential` -- an ordered list of layers, where "splitting the model" simply means keeping the first *N* on the client and the rest on the server. Model definitions are in `exp/src/model/architectures/`.
