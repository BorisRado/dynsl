import sys
import yaml
import asyncio
import traceback
import concurrent.futures
from functools import partial
from logging import INFO, WARN, ERROR
from typing import Optional, Tuple

import grpc
import grpc.aio
import torch.nn as nn

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
from slbd.server.server_model.manager import ServerModelManager, GrpcServerModelManager
from slbd.server.strategy import Strategy
from slbd.server.grpc.server_model_servicer import ServerModelServicer
from slbd.server.server import Server
from slbd.common.utils import run_async
from slbd.proto import server_model_pb2_grpc
from slbd.torch.model_utils import get_num_layers

# import of algorithm components
from slbd.common.profiling import measure_per_layer_time
from slbd.algos.ares import ARESStrategy, ARESClientLayerAllocator
from slbd.algos.dynsl import DynSLStrategy, DynSLClientLayerAllocator
from slbd.algos.fedavg import FedAvgStrategy, FedAvgClientLayerAllocator


ADDRESS_DRIVER_API = "0.0.0.0:9091"
ADDRESS_FLEET_API_GRPC_RERE = "0.0.0.0:9092"
ADDRESS_FLEET_API_GRPC_BIDI = "[::]:8080"  # IPv6 to keep start_server compatible
ADDRESS_FLEET_API_REST = "0.0.0.0:9093"

DATABASE = ":flwr-in-memory-state:"


def init_dynsl_server(config, model):
    client_manager = SimpleClientManager()

    strategy = DynSLStrategy(
        model=model,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        config_client_fit_fn=None,
        evaluate_metrics_aggregation_fn=None,
        fit_metrics_aggregation_fn=None,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
    )
    server_model_manager = GrpcServerModelManager(
        model=strategy.get_model_architecture(),
    )

    # profile the model on the server
    client_layer_allocator = DynSLClientLayerAllocator(8, get_num_layers(model))
    server = Server(
        client_manager=client_manager,
        strategy=strategy,
        server_model_manager=server_model_manager,
        client_layer_allocator=client_layer_allocator
    )

    return server


def init_ares_server(config, model):

    client_manager = SimpleClientManager()

    strategy = ARESStrategy(
        model=model,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        config_client_fit_fn=None,
        evaluate_metrics_aggregation_fn=None,
        fit_metrics_aggregation_fn=None,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
    )
    server_model_manager = GrpcServerModelManager(
        model=strategy.get_model_architecture(),
    )

    # profile the model on the server
    import torch
    optimizer = torch.optim.SGD(model.parameters(), momentum=0.9)
    forward_times, backward_times, activation_sizes = measure_per_layer_time(
        model=model,
        optimizer=optimizer,
        input_shape=(64, 3, 32, 32), # TODO: it is hardcoded right now
        iterations=20,
        dry_runs=5,
    )
    client_layer_allocator = ARESClientLayerAllocator(
        client_profiling_frequency=5,
        server_fb_times=[fw+bw for fw, bw in zip(forward_times, backward_times)],
        activation_sizes=activation_sizes,
        client_throughputs_MBps=10, # TODO: it is hardcoded right now
    )
    server = Server(
        client_manager=client_manager,
        strategy=strategy,
        server_model_manager=server_model_manager,
        client_layer_allocator=client_layer_allocator
    )

    return server


def init_fedavg_server(config, model):
    client_manager = SimpleClientManager()

    strategy = FedAvgStrategy(
        model=model,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        config_client_fit_fn=None,
        evaluate_metrics_aggregation_fn=None,
        fit_metrics_aggregation_fn=None,
        min_fit_clients=1,
        min_evaluate_clients=1,
        min_available_clients=1,
    )
    server_model_manager = GrpcServerModelManager(
        model=strategy.get_model_architecture(),
    )

    # profile the model on the server
    client_layer_allocator = FedAvgClientLayerAllocator(num_model_layers=get_num_layers(model))
    server = Server(
        client_manager=client_manager,
        strategy=strategy,
        server_model_manager=server_model_manager,
        client_layer_allocator=client_layer_allocator
    )
    return server


async def start_grpc_server(  # pylint: disable=too-many-arguments
    client_manager: ClientManager,
    server_model_manager: ServerModelManager,
    client_layer_allocator: ClientLayerAllocator,
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

    slbd_servicer = ServerModelServicer(server_model_manager, client_layer_allocator)
    server_model_pb2_grpc.add_ServerModelServicer_to_server(slbd_servicer, server)

    flwr_servicer = FlowerServiceServicer(client_manager)
    add_FlowerServiceServicer_to_server(flwr_servicer, server)

    await server.start()

    return server


@run_async
async def start_server(  # pylint: disable=too-many-arguments,too-many-locals
    *,
    model: nn.Sequential,
    config_file: str,
    server_address: str = ADDRESS_FLEET_API_GRPC_BIDI,
    grpc_max_message_length: int = GRPC_MAX_MESSAGE_LENGTH,
    certificates: Optional[Tuple[bytes, bytes, bytes]] = None,
) -> History:

    event(EventType.START_SERVER_ENTER)

    # Read and parse the model owner's config file with all experiment configuration
    try:
        with open(config_file, "r") as f:
            config_dict = yaml.safe_load(f)
        print(yaml.dump(config_dict, default_flow_style=False, indent=2))
    except FileNotFoundError:
        log(ERROR, f"Config file not found: {config_file}")
        sys.exit(1)
    except yaml.YAMLError as e:
        log(ERROR, f"Error parsing YAML config file: {e}")
        sys.exit(1)

    algorithm = config_dict["algorithm"]
    if algorithm == "dynsl":
        initialized_server = init_dynsl_server(config_dict, model)
    elif algorithm == "ares":
        initialized_server = init_ares_server(config_dict, model)
    elif algorithm == "fedavg":
        initialized_server = init_fedavg_server(config_dict, model)
    else:
        raise Exception(f"Unknown algorithm {algorithm}")
    initialized_config = ServerConfig(num_rounds=config_dict["global_config"]["rounds"])

    # Parse IP address
    parsed_address = parse_address(server_address)
    if not parsed_address:
        sys.exit(f"Server IP address ({server_address}) cannot be parsed.")
    host, port, is_v6 = parsed_address
    address = f"[{host}]:{port}" if is_v6 else f"{host}:{port}"

    log(
        INFO,
        "Starting Flower server, config: %s",
        initialized_config,
    )

    # Start gRPC server
    grpc_server = await start_grpc_server(
        client_manager=initialized_server.client_manager(),
        server_model_manager=initialized_server.server_model_manager,
        client_layer_allocator=initialized_server.get_client_layer_allocator(),
        server_address=address,
        max_message_length=grpc_max_message_length,
        certificates=certificates,
    )
    log(
        INFO,
        "Flower ECE: gRPC server running (%s rounds), SSL is %s",
        initialized_config.num_rounds,
        "enabled" if certificates is not None else "disabled",
    )

    # Start training
    try:
        run_fl_ = partial(run_fl, server=initialized_server, config=initialized_config)
        hist = await asyncio.to_thread(run_fl_)

    except Exception as e:  # pylint: disable=broad-except
        log(ERROR, "Exception in start_server: %s", e)
        log(ERROR, traceback.format_exc())
        hist = None

    finally:
        await grpc_server.stop(grace=1)

    event(EventType.START_SERVER_LEAVE)

    return hist
