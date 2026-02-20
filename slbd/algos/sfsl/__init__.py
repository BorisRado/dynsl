from slbd.algos.sfsl.client import SFSLClientAdapter
from slbd.algos.sfsl.layer_allocator import SFSLClientLayerAllocator
from slbd.algos.sfsl.strategy import SFSLStrategy

"""SFSL -- every client trains the same number of layer on all the clients, while training the encoder solely
on the client loss
"""

__all__ = [
    "SFSLClientAdapter",
    "SFSLClientLayerAllocator",
    "SFSLStrategy",
]
