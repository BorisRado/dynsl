"""Paper: Speed Up Federated Learning in Heterogeneous Environments: A Dynamic Tiering Approach

Core features:
- clients are clustered based on their compute capacity
- clients have a local classification head -- the loss on the server updates only the server model
"""
from slbd.algos.dtfl.client import DTFLClientAdapter
from slbd.algos.dtfl.strategy import DTFLStrategy
from slbd.algos.dtfl.layer_allocator_simulation import DTFLSimulationClientLayerAllocator


__all__ = [
    "DTFLClientAdapter",
    "DTFLStrategy",
    "DTFLSimulationClientLayerAllocator",
]
