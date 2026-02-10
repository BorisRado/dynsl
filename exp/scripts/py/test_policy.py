import time
import asyncio
from collections import OrderedDict
from typing import Tuple, Dict, Optional, List

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
from slbd.common import BatchDataForward, BatchDataBackward, TimingMetrics
from slbd.server.batching_worker_manager import BatchingWorkerManager
from slbd.torch.sequential_model import FlexibleSequential
from slbd.server.server_model.manager.multithreading_server_model_manager import MultithreadingServerModelManager
from slbd.algos.ares import ARESClientLayerAllocator
from slbd.algos.dynsl import DynSLClientLayerAllocatorV2
from slbd.algos.fedavg import FedAvgClientLayerAllocator
from slbd.algos.splitfed import SplitFedClientLayerAllocator
from slbd.algos.oracle import OracleClientLayerAllocator
from slbd.torch.sequential_model import FlexibleSequential

from slbd.common import ServerModelFitIns
from slbd.torch.model_utils import get_model_layer_keys, remove_layers_from_model



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
    ):
        self.cid = client_id
        self.device_type = device_type
        self.embedding_sizes = embedding_sizes
        self.num_iterations = num_iterations
        self.num_classes = num_classes
        self.manager = None
        self.base_training_times = base_training_times[self.device_type]
        self.model_weights_dictionary = model_weights_dictionary
        assert len(set([a[0] for a in self.embedding_sizes.values()])) == 1
        print("Initialized virtual", device_type, flush=True)

    def set_batching_worker_manager(self, manager):
        self.manager = manager

    def set_epoch_initial_num_layers(self, num_layers):
        self.initial_num_layers = num_layers

    async def run(self):
        # Simulate sending data to the BatchingWorkerManager
        current_num_layers = self.initial_num_layers or min(self.embedding_sizes.keys())
        batch_size = self.embedding_sizes[current_num_layers][0]

        current_data = {
            "embeddings": np.random.randn(*self.embedding_sizes[current_num_layers]).astype(np.float32),
            "labels": np.random.randint(0, self.num_classes, (batch_size,)),
        }
        for iter_idx in range(self.num_iterations):
            local_computation_time = self.base_training_times[current_num_layers]
            if iter_idx > 0: await asyncio.sleep(local_computation_time)
            batch = BatchDataForward(
                data=current_data,
                is_training=True,
                client_timing_metrics=TimingMetrics(
                    client_local_computation_time=local_computation_time,
                    client_tot_local_time=local_computation_time,
                    client_round_time=local_computation_time+1e-10, # this way the throughput measured on the server will be very high
                    response_length=1e8, # dummy value, not really important -> network is super fast either way
                ),
            )
            _ = current_data.pop("client_model_weights", None)
            data_backward: BatchDataBackward = await self.manager.enqueue_and_get_future(
                client_id=self.cid,
                data=batch,
            )

            # ignore additional_layers_parameters and additional_optimizer_states, just change
            # the layer, thus the time for client computation
            if data_backward.layers_diff:
                assert isinstance(data_backward.layers_diff, int)
                prev_num_layers = current_num_layers
                current_num_layers += data_backward.layers_diff
                local_computation_time = self.base_training_times[current_num_layers]
                print(f"{self.cid} >>> Changed number of layers: {prev_num_layers} -> {current_num_layers}", flush=True)
                current_data["embeddings"] = np.random.randn(*self.embedding_sizes[current_num_layers]).astype(np.float32)
                if data_backward.layers_diff < 0:
                    # prepare the layer parameters to be sent to the server
                    model_weights = [
                        w
                        for idx in range(current_num_layers, prev_num_layers)
                        for w in self.model_weights_dictionary[idx]
                    ]
                    # the last tensors states how many data points have been trained on the client
                    model_weights.append(np.array([10] * len(model_weights)))
                    current_data["client_model_weights"] = model_weights

    def get_properties(self, *args, **kwargs):
        return GetPropertiesRes(None, {
            CLIENT_DEVICE_TYPE: self.device_type,
            MIN_NUMBER_OF_CLIENT_LAYERS: 2,
            CLIENT_DATASET_SIZE: self.num_iterations,
            MAX_NUMBER_OF_CLIENT_LAYERS: 100,
            "node_name": self.device_type + str(self.cid)
        })


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

    initial_allocation = allocator.initialize_epoch(clients, model_manager, True)
    for client, initial_nl in zip(clients, initial_allocation):
        client.set_epoch_initial_num_layers(initial_nl)
    start_time = time.perf_counter()
    await asyncio.gather(*[client.run() for client in clients])
    end_time = time.perf_counter()
    print(f"Epoch time: {end_time - start_time:.3f}", flush=True)


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
        for layer_idx in model.meaningful_split_points
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
            )
        )
    return clients


def _initialize_layer_allocator(cfg, model):
    layer_allocator_init_args = OmegaConf.to_container(cfg.algorithm.layer_allocator_init_args)
    layer_allocator_init_args["num_model_layers"] = len(model)
    layer_allocator_init_args["meaningful_split_points"] = model.meaningful_split_points
    if cfg.algorithm.name == "dynsl_v2":
        tmp_optimizer = torch.optim.Adam(model.parameters())
        complexity_kwargs = {
            "model": model,
            "optimizer": tmp_optimizer,
            "batch_size": cfg.experiment.batch_size * cfg.server_model_servicer.num_concat_batches,
        }
        layer_complexities_cuda, tensor_sizes = measure_per_layer_complexity(
            iterations=400,
            dry_runs=50,
            **complexity_kwargs,
        )
        layer_complexities_cpu, _ = measure_per_layer_complexity(
            iterations=50,
            dry_runs=5,
            disable_cuda=True,
            **complexity_kwargs,
        )
        layer_complexities = {
            k: (layer_complexities_cpu[k] + layer_complexities_cuda[k]) / 2
            for k in layer_complexities_cpu
        }
        del tmp_optimizer
        per_layer_parameters = {
            idx: sum(p.numel() for p in layer.state_dict().values())
            for idx, layer in enumerate(model.children())
        }
        layer_allocator_init_args["modeled_layer_complexities"] = layer_complexities
        layer_allocator_init_args["per_layer_emb_sizes"] = tensor_sizes
        layer_allocator_init_args["static_server_computational_times"] = layer_complexities_cuda
        layer_allocator_init_args["per_layer_parameters"] = per_layer_parameters
    allocator = {
        "ares": ARESClientLayerAllocator,
        "dynsl_v2": DynSLClientLayerAllocatorV2,
        "fedavg": FedAvgClientLayerAllocator,
        "splitfed_v2": SplitFedClientLayerAllocator, # right now we do not support splitfed_v1
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
    for client in clients: client.set_batching_worker_manager(batching_worker_manager)

    # Run all clients in parallel
    for _ in range(cfg.experiment.num_rounds):
        await _train_epoch(
            model_manager=model_manager,
            model=model,
            allocator=allocator,
            clients=clients,
            optimization_config=OmegaConf.to_container(cfg.optimizer.optimizer)
        )


@hydra.main(config_path="../../conf", config_name="test_policy", version_base=None)
def run(cfg):
    OmegaConf.resolve(cfg)

    # remove irrelevant data for the training times -- we only care about the batch size of the current experiment
    batch_size = cfg.experiment.batch_size
    for device_type in cfg.profiled_times:
        cfg.profiled_times[device_type] = cfg.profiled_times[device_type][batch_size]

    print(OmegaConf.to_yaml(cfg), flush=True)
    asyncio.run(run_experiment(cfg))


if __name__ == "__main__":
    run()
