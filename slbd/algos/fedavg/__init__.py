from slbd.algos.fedavg.client import FedAvgNumpyClientAdapter
from slbd.algos.fedavg.strategy import FedAvgStrategy
from slbd.algos.fedavg.layer_allocator import FedAvgClientLayerAllocator


"""FedAvg algorithm

All computation happens fully locally
"""

__all__ = [
    "FedAvgNumpyClientAdapter",
    "FedAvgStrategy",
    "FedAvgClientLayerAllocator",
]
