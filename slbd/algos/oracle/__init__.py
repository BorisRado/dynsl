from slbd.algos.oracle.client import OracleNumpyClientAdapter
from slbd.algos.oracle.layer_allocator import OracleClientLayerAllocator
from slbd.algos.oracle.simulation_layer_allocator import OracleSimulationClientLayerAllocator
from slbd.algos.oracle.strategy import OracleStrategy

"""Oracle -- we assign a predefined number of layers to every device type (works only for CoLExT)
"""

__all__ = [
    "OracleNumpyClientAdapter",
    "OracleClientLayerAllocator",
    "OracleSimulationClientLayerAllocator",
    "OracleStrategy"
]
