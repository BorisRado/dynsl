from slbd.algos.ares.client import ARESNumpyClientAdapter
from slbd.algos.ares.strategy import ARESStrategy
from slbd.algos.ares.layer_allocator import ARESClientLayerAllocator

"""
ARES algorithm implementation.

Paper: ARES: Adaptive Resource-Aware Split Learning for Internet of Things

Core features:
- every client trains its own classification head
- every client periodically profiles the performance of the trained model on its local infrastructure
- the server *independently per every client*, based on network speed and profiles of both the
client and the server, where to split a model
"""

__all__ = [
    "ARESNumpyClientAdapter",
    "ARESStrategy",
    "ARESClientLayerAllocator",
]
