# Dynamic Split Learning Framework

This repository contains the implementation and experimental code for **DynSL**.

Most experiments are run on [CoLExT](https://sands.kaust.edu.sa/project/colext/), a public FL experimentation testbed.

## Quickstart

To run the experiments, request access to the testbed, then run the following scripts on the server. The requirements will be automatically downloaded, and the experiment will immediately start running in a distributed manner over the heterogeneous devices:

```bash
./exp/scripts/sh/colext_common_head.sh      # for experiments in 6.2
./exp/scripts/sh/colext_separate_heads.sh   # for experiments in 6.3
```

To manually run the experiments, first run the script `python exp/scripts/py/generate_colext_config.py` by overriding the necessary parameters. Use `python exp/scripts/py/generate_colext_config.py --help` for a complete list of parameters that can be configured.

The script generates the `colext_config.yaml` file with all the necessary configuration for running the experiment

Afterwards, use the `colext_launch_job` command, which runs the experiment.

For distributed experiments, the two entry points are:
* `exp/scripts/py/server.py`: runs the server logic
* `exp/scripts/py/client.py`: runs the client logic

Both scripts get the configuration from `exp/conf/config.yaml`. You may configure the experiment by simply overriding the entries using standard [hydra](https://hydra.cc/docs/intro/) syntax, for instance:
```bash
optimizer=adam # use the adam optimizer
client.init_kwargs.batch_size=32 # set the batch size to 32
algorithm=dynsl_v2 # use the dynsl_v2 class for running the DynSL algorithm
algorithm=splitfed_v1 # use the splitfed-v1 algorithm
algorithm.num_server_models=1 # train a single classification head on the server during an epoch
```

You may also run simulations, which create virtual clients as python objects. The entry script is `exp/scripts/py/simulate_sl.py`

---

## Project structure

For each algorithm, we implement the following classes:

1. Strategy (we use the terminology from [flower](https://flower.ai/)): samples clients at the beginning of the epoch, configures the clients by preparing the appropriate parameters, and aggregates the parameters at the end of the epoch.
2. LayerAllocator: is the component that determines the initial number of layers the clients should train. The allocator also gets polled to determine if the client should change the number of layers it trains. For all algorithms apart from DynSL, no changes are ever performed.

The implementation of all the algorithms can be found in `slbd/algos`.

The role of the controller is in code taken by the `BatchingWorkerManager` class (`slbd/server/batching_worker_manager.py`).

---
