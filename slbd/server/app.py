import sys
import signal
import asyncio
import traceback
import concurrent.futures
from functools import partial
from logging import INFO, ERROR
from typing import Optional, Tuple

import grpc
import grpc.aio
import torch.optim as optim
from hydra.utils import instantiate
from omegaconf import OmegaConf, DictConfig

from flwr.common import GRPC_MAX_MESSAGE_LENGTH, EventType, event
from flwr.common.address import parse_address
from flwr.common import log
from flwr.server.history import History
from flwr.server.app import ServerConfig, run_fl
from flwr.proto.transport_pb2_grpc import add_FlowerServiceServicer_to_server
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.server.superlink.fleet.grpc_bidi.grpc_server import valid_certificates

from slbd.server.client_layer_allocator import ClientLayerAllocator
from slbd.server.grpc.flower_servicer import FlowerServiceServicer
from slbd.server.server_model.manager import ServerModelManager
from slbd.server.grpc.server_model_servicer import ServerModelServicer
from slbd.server.grpc.queue_server_model_servicer import ServerModelServicer as QueueServerModelServicer
from slbd.server.grpc.ping_pong_servicer import ServerModelServicer as PPServerModelServicer
from slbd.server.server import Server
from slbd.common.utils import run_async
from slbd.proto import server_model_pb2_grpc
from slbd.torch.model_utils import get_num_layers
from slbd.torch.utils import init_optimizer, OptimizerGenerator

# import of algorithm components
from slbd.common.profiling import (
    measure_per_layer_time,
    get_flops_per_layer_fvcore,
    measure_per_layer_complexity
)
from slbd.algos.ares import ARESStrategy, ARESClientLayerAllocator
from slbd.algos.dynsl import DynSLStrategy, DynSLClientLayerAllocator, DynSLClientLayerAllocatorV2
from slbd.algos.fedavg import FedAvgStrategy, FedAvgClientLayerAllocator
from slbd.algos.splitfed import SplitFedStrategy, SplitFedClientLayerAllocator
from slbd.algos.oracle import OracleClientLayerAllocator, OracleStrategy
from slbd.torch.sequential_model import FlexibleSequential


ADDRESS_DRIVER_API = "0.0.0.0:9091"
ADDRESS_FLEET_API_GRPC_RERE = "0.0.0.0:9092"
ADDRESS_FLEET_API_GRPC_BIDI = "[::]:8080"  # IPv6 to keep start_server compatible
ADDRESS_FLEET_API_REST = "0.0.0.0:9093"

DATABASE = ":flwr-in-memory-state:"


def init_dynsl_components(config, model):
    strategy = DynSLStrategy(
        model=model,
        optimizer_config=OmegaConf.to_container(config.optimizer),
        **OmegaConf.to_container(config.strategy),
    )
    client_layer_allocator = DynSLClientLayerAllocator(
        num_model_layers=get_num_layers(model),
        meaningful_split_points=model.meaningful_split_points,
        **OmegaConf.to_container(config.algorithm.layer_allocator_init_args, resolve=True),
    )
    return strategy, client_layer_allocator


def init_oracle_components(config, model):
    strategy = OracleStrategy(
        model=model,
        optimizer_config=OmegaConf.to_container(config.optimizer),
        **OmegaConf.to_container(config.strategy),
    )
    client_layer_allocator = OracleClientLayerAllocator(
        num_model_layers=get_num_layers(model),
        **OmegaConf.to_container(config.algorithm.layer_allocator_init_args, resolve=True),
    )
    return strategy, client_layer_allocator


def init_dynsl_v2_components(config, model):
    strategy = DynSLStrategy(
        model=model,
        optimizer_config=OmegaConf.to_container(config.optimizer),
        **OmegaConf.to_container(config.strategy),
    )
    layer_allocator_config = OmegaConf.to_container(config.algorithm.layer_allocator_init_args, resolve=True)
    tmp_optimizer = optim.SGD(model.parameters(), momentum=0.9)
    layer_complexities_cuda, tensor_sizes = measure_per_layer_complexity(
        model=model,
        optimizer=tmp_optimizer,
        batch_size=config.client.init_kwargs.batch_size * config.server_model_servicer.num_concat_batches,
        iterations=400,
        dry_runs=50,
    )
    layer_complexities_cpu, _ = measure_per_layer_complexity(
        model=model,
        optimizer=tmp_optimizer,
        batch_size=config.client.init_kwargs.batch_size * config.server_model_servicer.num_concat_batches,
        iterations=50,
        dry_runs=5,
        disable_cuda=True
    )
    layer_complexities = {
        k: (layer_complexities_cpu[k] + layer_complexities_cuda[k]) / 2
        for k in layer_complexities_cpu
    }
    del tmp_optimizer
    if layer_allocator_config["max_client_layers"] < 0:
        layer_allocator_config["max_client_layers"] = model.meaningful_split_points[-1]
    assert layer_allocator_config["max_client_layers"] < get_num_layers(model)
    client_layer_allocator = DynSLClientLayerAllocatorV2(
        num_model_layers=get_num_layers(model),
        meaningful_split_points=model.meaningful_split_points,
        per_layer_communication_volume=tensor_sizes,
        **layer_allocator_config,
        modeled_layer_complexities=layer_complexities,
    )

    return strategy, client_layer_allocator


def init_ares_components(config, model):

    strategy = ARESStrategy(
        model=model,
        optimizer_config=OmegaConf.to_container(config.optimizer),
        **OmegaConf.to_container(config.strategy),
    )

    # profile the model on the server
    optimizer = init_optimizer(
        model.parameters(),
        **OptimizerGenerator(config.optimizer.optimizer, {"strategy": "constant"}).get_optimizer_config()
    )
    log(INFO, "Profiling training time on the server")
    forward_times, backward_times, activation_sizes = measure_per_layer_time(
        model=model,
        optimizer=optimizer,
        batch_size=config.client.init_kwargs.batch_size,
        iterations=10,
        dry_runs=0,
    )
    log(INFO, "Server profiling completed")
    client_layer_allocator = ARESClientLayerAllocator(
        client_profiling_frequency=4,
        server_fb_times=[fw+bw for fw, bw in zip(forward_times, backward_times)],
        activation_sizes=activation_sizes,
        client_throughputs_MBps=1000, # TODO: it is hardcoded right now
        **OmegaConf.to_container(config.algorithm.layer_allocator_init_args),
        meaningful_split_points=model.meaningful_split_points,
    )

    return strategy, client_layer_allocator


def init_fedavg_components(config, model):
    strategy = FedAvgStrategy(
        model=model,
        optimizer_config=OmegaConf.to_container(config.optimizer),
        **OmegaConf.to_container(config.strategy),
        **OmegaConf.to_container(config.algorithm.strategy_init_args),
    )
    client_layer_allocator = FedAvgClientLayerAllocator(
        num_model_layers=get_num_layers(model),
        **OmegaConf.to_container(config.algorithm.layer_allocator_init_args),
    )
    return strategy, client_layer_allocator


def init_splitfed_components(config, model):

    strategy = SplitFedStrategy(
        model=model,
        optimizer_config=OmegaConf.to_container(config.optimizer),
        **OmegaConf.to_container(config.strategy),
        **OmegaConf.to_container(config.algorithm.strategy_init_args),
    )

    client_layer_allocator = SplitFedClientLayerAllocator(
        num_model_layers=get_num_layers(model),
        **OmegaConf.to_container(config.algorithm.layer_allocator_init_args),
    )

    return strategy, client_layer_allocator


async def start_grpc_server(  # pylint: disable=too-many-arguments
    client_manager: ClientManager,
    server_model_manager: ServerModelManager,
    client_layer_allocator: ClientLayerAllocator,
    server_model_servicer_config,
    data_manipulation_config,
    server_address: str,
    max_concurrent_workers: int = 1000,
    max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    keepalive_time_ms: int = 210000,
    certificates: Optional[Tuple[bytes, bytes, bytes]] = None,
) -> grpc.Server:

    options = [
        ("grpc.max_concurrent_streams", max(100, max_concurrent_workers)),
        ("grpc.max_send_message_length", max_message_length),
        ("grpc.max_receive_message_length", max_message_length),
        ("grpc.keepalive_time_ms", keepalive_time_ms),
        ("grpc.http2.max_pings_without_data", 0),
        ("grpc.keepalive_permit_without_calls", 0),
    ]

    server = grpc.aio.server(
        migration_thread_pool=concurrent.futures.ThreadPoolExecutor(max_concurrent_workers),
        maximum_concurrent_rpcs=max_concurrent_workers,
        options=options,
    )

    if certificates is not None:
        if not valid_certificates(certificates):
            sys.exit(1)

        root_certificate_b, certificate_b, private_key_b = certificates

        server_credentials = grpc.ssl_server_credentials(
            ((private_key_b, certificate_b),),
            root_certificates=root_certificate_b,
            require_client_auth=False,
        )
        server.add_secure_port(server_address, server_credentials)
    else:
        server.add_insecure_port(server_address)

    server_model_servicer_name = server_model_servicer_config.pop("servicer_name")
    if server_model_servicer_name == "queue":
        print("Using queue servicer")
        slbd_servicer = QueueServerModelServicer(
            server_model_manager=server_model_manager,
            client_layer_allocator=client_layer_allocator,
            sending_transform_fn_config=data_manipulation_config.server.sending_fn,
            receiving_transform_fn_config=data_manipulation_config.server.receiving_fn,
            **server_model_servicer_config
        )
    elif server_model_servicer_name == "ping_pong":
        slbd_servicer = PPServerModelServicer(server_model_manager, client_layer_allocator, **server_model_servicer_config)
    else:
        print("Using usual servicer")
        slbd_servicer = ServerModelServicer(server_model_manager, client_layer_allocator, **server_model_servicer_config)
    server_model_pb2_grpc.add_ServerModelServicer_to_server(slbd_servicer, server)

    flwr_servicer = FlowerServiceServicer(client_manager)
    add_FlowerServiceServicer_to_server(flwr_servicer, server)

    await server.start()

    return server


def setup_signal_handlers(worker_manager):
    def handle_exit(sig, frame):
        print(f"\n[Main] Caught signal {sig}, cleaning up workers...")
        worker_manager.clean_resources()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)   # Ctrl+C
    signal.signal(signal.SIGTERM, handle_exit)  # kill <pid>


@run_async
async def start_server(  # pylint: disable=too-many-arguments,too-many-locals
    *,
    model: FlexibleSequential,
    config_file: Optional[str] = None,
    config: Optional[DictConfig] = None,
    server_address: str = ADDRESS_FLEET_API_GRPC_BIDI,
    grpc_max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    certificates: Optional[Tuple[bytes, bytes, bytes]] = None,
) -> History:

    event(EventType.START_SERVER_ENTER)

    # Read and parse the model owner's config file with all experiment configuration
    assert (config_file is None) != (config is None), "Either config or config file must be given"
    if config_file is not None:
        config = OmegaConf.load(config_file)

    # initialize the server
    strategy, client_layer_allocator = {
        "dynsl": init_dynsl_components,
        "dynsl_v2": init_dynsl_v2_components,
        "ares": init_ares_components,
        "fedavg": init_fedavg_components,
        "splitfed_v1": init_splitfed_components,
        "splitfed_v2": init_splitfed_components,
        "oracle": init_oracle_components,
    }[config.algorithm.name](config, model)

    server_model_manager = instantiate(
        config.server_model_manager,
        model=strategy.get_model_architecture(),
        exp_config=OmegaConf.to_container(config.copy()),
        _recursive_=False
    )
    initialized_server = Server(
        client_manager=SimpleClientManager(),
        strategy=strategy,
        server_model_manager=server_model_manager,
        client_layer_allocator=client_layer_allocator
    )

    server_config = ServerConfig(num_rounds=config.general.rounds)

    # Parse IP address
    parsed_address = parse_address(server_address)
    if not parsed_address:
        sys.exit(f"Server IP address ({server_address}) cannot be parsed.")
    host, port, is_v6 = parsed_address
    address = f"[{host}]:{port}" if is_v6 else f"{host}:{port}"

    log(
        INFO,
        "Starting Flower server, config: %s",
        server_config,
    )

    setup_signal_handlers(initialized_server.server_model_manager)

    # Start gRPC server
    grpc_server = await start_grpc_server(
        client_manager=initialized_server.client_manager(),
        server_model_manager=initialized_server.server_model_manager,
        server_model_servicer_config=OmegaConf.to_container(config.server_model_servicer),
        client_layer_allocator=initialized_server.get_client_layer_allocator(),
        data_manipulation_config=config.data_manipulation,
        server_address=address,
        max_message_length=grpc_max_message_length,
        certificates=certificates,
    )
    log(
        INFO,
        "Flower ECE: gRPC server running (%s rounds), SSL is %s",
        server_config.num_rounds,
        "enabled" if certificates is not None else "disabled",
    )

    # Start training
    try:
        run_fl_ = partial(run_fl, server=initialized_server, config=server_config)
        hist = await asyncio.to_thread(run_fl_)

    except Exception as e:  # pylint: disable=broad-except
        log(ERROR, "Exception in start_server: %s", e)
        log(ERROR, traceback.format_exc())
        hist = None

    finally:
        initialized_server.server_model_manager.clean_resources()
        await grpc_server.stop(grace=1)

    event(EventType.START_SERVER_LEAVE)

    return hist
