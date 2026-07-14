import time
import pickle
import asyncio
import os
import fcntl
import json
import math
from collections import OrderedDict, defaultdict
from typing import Tuple, Dict, List

import torch
import numpy as np
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from flwr.common import GetPropertiesRes

from slbd.common.constants import (
    CLIENT_DEVICE_TYPE,
    MIN_NUMBER_OF_CLIENT_LAYERS,
    CLIENT_DATASET_SIZE,
    CLIENT_DEVICE_TYPE,
    MAX_NUMBER_OF_CLIENT_LAYERS
)
from slbd.common.profiling import measure_per_layer_complexity
from slbd.common import BatchDataForward, ServerModelFitIns, SocketTimingMetrics
from slbd.server.batching_worker_manager import BatchingWorkerManager
from slbd.torch.sequential_model import FlexibleSequential
from slbd.server.server_model.manager.multithreading_server_model_manager import MultithreadingServerModelManager
from slbd.algos.ares import ARESClientLayerAllocator
from slbd.algos.dynsl import DynSLClientLayerAllocatorV2, DynSLClientLayerAllocatorWProfiling
from slbd.algos.fedavg import FedAvgClientLayerAllocator
from slbd.algos.splitfed import SplitFedClientLayerAllocator
from slbd.algos.oracle import OracleClientLayerAllocator
from slbd.torch.sequential_model import FlexibleSequential
from slbd.torch.model_utils import get_model_layer_keys
from slbd.common.profiling import get_flops_per_layer_fvcore


NETWORK_SPEED=50

class BatchingWorkerManagerInterface:
    # just a utility class to add some overhead to the server for serialization and deserialization
    def __init__(self, batching_worker_manager):
        self.batching_worker_manager = batching_worker_manager

    async def __call__(self, client_id, raw_data, current_fwd_times, current_backward_time):
        current_data = pickle.loads(raw_data)
        batch = BatchDataForward(
            data=current_data,
            is_training=True,
            client_timing_metrics=SocketTimingMetrics(
                uplink_throughput_Mbps=NETWORK_SPEED,
                downlink_throughput_Mbps=NETWORK_SPEED,
                per_layer_forward_times=current_fwd_times,
                tot_backward_time=current_backward_time,
            ),
        )
        deserialized_response = await self.batching_worker_manager.enqueue_and_get_future(
            client_id=client_id,
            data=batch,
        )
        _ = pickle.dumps(deserialized_response)
        return deserialized_response.layers_diff


class SimulationClient:
    def __init__(
        self,
        client_id: int,
        device_type: str,
        num_iterations: int,
        embedding_sizes: Dict[int, Tuple[int]],
        num_classes: int,
        base_training_times: Dict[str, Dict[int, float]],
        model_weights_dictionary: Dict[int, List[np.ndarray]],
        meaningful_split_points,
    ):
        self.cid = client_id
        self.device_type = device_type
        self.embedding_sizes = embedding_sizes
        self.num_iterations = num_iterations
        self.num_classes = num_classes
        self.manager = None
        self.base_training_times = base_training_times[self.device_type]
        self.model_weights_dictionary = model_weights_dictionary
        self.meaningful_split_points = meaningful_split_points
        assert len(set([a[0] for a in self.embedding_sizes.values()])) == 1
        # print("Initialized virtual", device_type, flush=True)

    def set_batching_worker_manager(self, manager):
        self.manager = manager

    def set_epoch_initial_num_layers(self, num_layers):
        self.initial_num_layers = num_layers

    def _get_packet_size_Mbps(self, num_layers):
        num_parameters = math.prod(self.embedding_sizes[num_layers])
        return num_parameters * 32 / 1e6

    async def _wait_network(self, network_size):
        sleep_time = network_size / NETWORK_SPEED
        await asyncio.sleep(2 * sleep_time + 0.01) # * 2 because forward/backward

    async def run(self):
        # Simulate sending data to the BatchingWorkerManager
        current_num_layers = self.initial_num_layers
        if current_num_layers == len(self.base_training_times):
            # the model is trained fully locally. No need to involve the server any longer
            return
        current_fwd_times = [self.base_training_times[k] / 2 for k in range(current_num_layers)]
        current_backward_time = self.base_training_times[current_num_layers - 1] / 2
        batch_size = self.embedding_sizes[current_num_layers][0]
        network_size = self._get_packet_size_Mbps(current_num_layers)

        current_data = pickle.dumps({
            "embeddings": torch.randn(*self.embedding_sizes[current_num_layers]),
            "labels": torch.randint(0, self.num_classes, (batch_size,)),
        })
        has_client_weights = False
        for iter_idx in range(self.num_iterations):
            local_computation_time = self.base_training_times[current_num_layers - 1]
            if current_num_layers == len(self.base_training_times):
                # the model is trained fully locally. No need to involve the server any longer
                return
            if iter_idx > 0:
                sleep_time = np.random.uniform(local_computation_time * 0.95, local_computation_time * 1.05)
                await asyncio.sleep(sleep_time)
            await self._wait_network(network_size)

            layers_diff = await self.manager(
                client_id=self.cid,
                raw_data=current_data,
                current_fwd_times=current_fwd_times,
                current_backward_time=current_backward_time
            )
            if has_client_weights:
                current_data = pickle.dumps({
                    "embeddings": torch.randn(*self.embedding_sizes[current_num_layers]),
                    "labels": torch.randint(0, self.num_classes, (batch_size,)),
                })
                has_client_weights = False

            # ignore additional_layers_parameters and additional_optimizer_states, just change
            # the layer, thus the time for client computation
            if layers_diff:
                assert isinstance(layers_diff, int)
                prev_num_layers = current_num_layers
                current_num_layers += layers_diff
                if current_num_layers == len(self.base_training_times):
                    # the model is trained fully locally. No need to involve the server any longer
                    return
                network_size = self._get_packet_size_Mbps(current_num_layers)
                current_fwd_times = [self.base_training_times[k] / 2 for k in range(current_num_layers)]
                current_backward_time = self.base_training_times[current_num_layers - 1] / 2

                local_computation_time = self.base_training_times[current_num_layers - 1]
                new_data = {
                    "embeddings": torch.randn(*self.embedding_sizes[current_num_layers]),
                    "labels": torch.randint(0, self.num_classes, (batch_size,)),
                }
                if layers_diff < 0:
                    # prepare the layer parameters to be sent to the server
                    model_weights = [
                        torch.from_numpy(w)
                        for idx in range(current_num_layers, prev_num_layers)
                        for w in self.model_weights_dictionary[idx]
                    ]
                    # the last tensors states how many data points have been trained on the client
                    model_weights.append(torch.tensor([10] * -layers_diff))
                    new_data["client_model_weights"] = model_weights
                    has_client_weights = True
                current_data = pickle.dumps(new_data)

    def get_properties(self, *args, **kwargs):
        individual_layer_times = self.base_training_times
        return GetPropertiesRes(None, {
            CLIENT_DEVICE_TYPE: self.device_type,
            MIN_NUMBER_OF_CLIENT_LAYERS: self.meaningful_split_points[0],
            CLIENT_DATASET_SIZE: self.num_iterations,
            MAX_NUMBER_OF_CLIENT_LAYERS: 100,
            "node_name": self.device_type + str(self.cid)
        } \
        | {f"__lc{idx}": self.base_training_times[idx] for idx in range(len(self.base_training_times))} \
        | {f"f+b{idx}": v for idx, v in enumerate(individual_layer_times)})


def _get_server_fit_ins(model, config):
    config = {f"optim.{k}": v for k, v in config.items()}
    server_model_state_dict = OrderedDict(
        (key, model.state_dict()[key])
        for layer_idx in range(min(model.meaningful_split_points), len(model))
        for key in get_model_layer_keys(model)[layer_idx]
    )
    server_fit_ins = ServerModelFitIns(
        server_model_state_dict, min(model.meaningful_split_points), config, sid=0
    )
    return [server_fit_ins,]


class _TrivialStrategy:
        def route_client_request_to_server_model(self, *args, **kwargs):
            return 0


async def _train_epoch(model_manager, model, allocator, clients, optimization_config):
    model_manager.initialize_server_models(
        _get_server_fit_ins(model, optimization_config),
        is_training=True
    )
    for sm in model_manager.spanned_server_models.values():
        sm._weight_integration_strategy = "ignore"

    class _ClientManager():
        def all(self):
            return {c.cid: c for c in clients}

    initial_allocation = allocator.initialize_epoch(clients, _ClientManager(), True)
    for client, initial_nl in zip(clients, initial_allocation):
        client.set_epoch_initial_num_layers(initial_nl)
    start_time = time.perf_counter()
    await asyncio.gather(*[client.run() for client in clients])
    end_time = time.perf_counter()
    epoch_time = end_time - start_time
    allocator_metrics = allocator.collect_epoch_metrics()
    allocator.finalize_epoch()
    print(f"Epoch time: {epoch_time:.3f}", flush=True)
    return {
        "epoch_time": epoch_time,
        "server_training_time": allocator_metrics["train/tot_server_training_time"],
    }


def _initialize_clients(cfg, model):
    # if a client is instructed to reduce the number of layer it trains, it needs to send the
    # tensors of the corresponding layers to the server
    # We pre-compute such tensors, so that the client does not need to hold a reference to the model
    # and extract the weight during runtime
    dummy_model_parameters = {}
    for idx in range(len(model)):
        dummy_model_parameters[idx] = [
            tensor.detach().cpu().numpy() for tensor in model[idx].state_dict().values()
        ]

    # compute embedding size per every split point
    input_data = torch.randn(cfg.experiment.batch_size, *cfg.data.input_shape)
    embedding_sizes = {
        layer_idx: tuple(FlexibleSequential(*list(model.children())[:layer_idx])(input_data).shape)
        for layer_idx in range(len(model))
    }

    # initialize the actual clients
    device_types = [device_type for device_type, count in cfg.experiment.device_counts.items() for _ in range(count)]
    clients = []
    for client_id, device_type in enumerate(device_types):
        clients.append(
            SimulationClient(
                client_id=client_id,
                device_type=device_type,
                embedding_sizes=embedding_sizes,
                base_training_times=cfg.profiled_times,
                model_weights_dictionary=dummy_model_parameters,
                **OmegaConf.to_container(cfg.client),
                meaningful_split_points=model.meaningful_split_points
            )
        )
    return clients


def _initialize_layer_allocator(cfg, model):
    layer_allocator_init_args = OmegaConf.to_container(cfg.algorithm.get("layer_allocator_init_args", {}))
    layer_allocator_init_args["num_model_layers"] = len(model)
    layer_allocator_init_args["meaningful_split_points"] = model.meaningful_split_points
    if cfg.algorithm.name.startswith("dynsl"):
        tmp_optimizer = torch.optim.Adam(model.parameters())
        complexity_kwargs = {
            "model": model,
            "optimizer": tmp_optimizer,
        }
        print("Testing with batch size", cfg.experiment.batch_size * cfg.server_model_servicer.num_concat_batches)
        ncb = cfg.server_model_servicer.num_concat_batches
        layer_complexities_cuda, tensor_sizes = measure_per_layer_complexity(
            iterations=200,
            dry_runs=30,
            batch_size=cfg.experiment.batch_size * ncb,
            **complexity_kwargs,
        )
        assert all(v % ncb == 0 for v in tensor_sizes.values())
        tensor_sizes = {k: v // cfg.server_model_servicer.num_concat_batches for k, v in tensor_sizes.items()}
        del tmp_optimizer
        per_layer_parameters = {
            idx: sum(p.numel() for p in layer.state_dict().values())
            for idx, layer in enumerate(model.children())
        }
        if cfg.algorithm.name == "dynsl_v2":
            layer_allocator_init_args["modeled_layer_complexities"] = get_flops_per_layer_fvcore(model)
        else:
            layer_allocator_init_args["connection_bandwidth"] = NETWORK_SPEED
        layer_allocator_init_args["per_layer_emb_sizes"] = tensor_sizes
        layer_allocator_init_args["static_server_computational_times"] = layer_complexities_cuda
        layer_allocator_init_args["per_layer_parameters"] = per_layer_parameters
    elif cfg.algorithm.name == "ares":
        server_fb_times, tensor_sizes = measure_per_layer_complexity(
            iterations=100,
            dry_runs=20,
            disable_cuda=False,
            batch_size=cfg.experiment.batch_size,
            model=model,
            optimizer=torch.optim.Adam(model.parameters()),
        )
        layer_allocator_init_args['server_fb_times'] = [server_fb_times[i] for i in range(len(server_fb_times))]
        layer_allocator_init_args["activation_sizes"] = \
            {k: v // cfg.server_model_servicer.num_concat_batches for k, v in tensor_sizes.items()}
        layer_allocator_init_args['client_throughputs_MBps'] = NETWORK_SPEED
        del layer_allocator_init_args["num_model_layers"]
    allocator = {
        "ares": ARESClientLayerAllocator,
        "dynsl_v2": DynSLClientLayerAllocatorV2,
        "dynsl_w_profiling": DynSLClientLayerAllocatorWProfiling,
        "fedavg": FedAvgClientLayerAllocator,
        "splitfed_v2": SplitFedClientLayerAllocator, # right now we do not support splitfed_v1
        "splitfed_v1": SplitFedClientLayerAllocator, # right now we do not support splitfed_v1
        "oracle": OracleClientLayerAllocator,
    }[cfg.algorithm.name](**layer_allocator_init_args)
    return allocator


async def run_experiment(cfg):

    model = instantiate(cfg.model)
    clients = _initialize_clients(cfg, model)
    model_manager = MultithreadingServerModelManager(model, None)
    model_manager.strategy = _TrivialStrategy()
    allocator = _initialize_layer_allocator(cfg, model)

    # initialize the batching worker manager
    batching_worker_manager = BatchingWorkerManager(
        server_model_manager=model_manager,
        client_layer_allocator=allocator,
        is_cuda_available=torch.cuda.is_available(),
        sending_transform_fn=None,
        receiving_transform_fn=None,
        **OmegaConf.to_container(cfg.server_model_servicer),
    )
    batching_worker_manager_interface = BatchingWorkerManagerInterface(batching_worker_manager)
    for client in clients: client.set_batching_worker_manager(batching_worker_manager_interface)

    # Run all clients in parallel
    epoch_times = defaultdict(list)
    for epoch_idx in range(cfg.experiment.num_rounds):
        print("Running", epoch_idx)
        tmp = await _train_epoch(
            model_manager=model_manager,
            model=model,
            allocator=allocator,
            clients=clients,
            optimization_config=OmegaConf.to_container(cfg.optimizer.optimizer)
        )
        for k, v in tmp.items():
            epoch_times[k].append(v)
    return epoch_times


@hydra.main(config_path="../../conf", config_name="test_policy", version_base=None)
def run(cfg):
    OmegaConf.resolve(cfg)

    # remove irrelevant data for the training times -- we only care about the batch size of the current experiment
    batch_size = cfg.experiment.batch_size
    for device_type in cfg.profiled_times:
        cfg.profiled_times[device_type] = cfg.profiled_times[device_type][batch_size]

    print(OmegaConf.to_yaml(cfg), flush=True)
    epoch_metrics = asyncio.run(run_experiment(cfg))

    filepath = cfg.filepath
    if filepath == "_auto_":
        vals = [f"{k}{cfg.experiment.device_counts[k]}" for k in sorted(cfg.experiment.device_counts.keys())]
        filepath = "_".join(vals)

    times_json_path = filepath + ".json"
    entry = {
        "algorithm": cfg.algorithm.name,
        "num_clients": OmegaConf.to_container(cfg.experiment.device_counts),
        "num_layers": "adaptive" if "splitfed" not in cfg.algorithm.name else cfg.algorithm.layer_allocator_init_args.num_client_layers,
        **epoch_metrics,
    }

    # Multiple copies of this script may target the same output file. Serialize the
    # read-modify-write behind an exclusive file lock so concurrent runs cannot clobber
    # each other's results or leave a half-written (corrupted) JSON file. The lock is
    # held on a dedicated .lock sidecar so it survives the truncating rewrite below.
    lock_path = times_json_path + ".lock"
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            if os.path.exists(times_json_path):
                with open(times_json_path, "r") as f:
                    times_json_contents = json.load(f)
            else:
                times_json_contents = []
            times_json_contents.append(entry)
            print(times_json_contents)
            with open(times_json_path, "w") as fp:
                json.dump(times_json_contents, fp, indent=2)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


if __name__ == "__main__":
    run()
