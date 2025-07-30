"""
Network and Distributed Computing Framework

This package contains distributed computing and networking components
for coordinating superintelligent entities.
"""

from .distributed import DistributedManager, ComputeNode, DistributedTask, TaskStatus, NodeType

__version__ = "1.0.0"
__author__ = "Mathematical Consciousness Framework Team"

__all__ = [
    "DistributedManager",
    "ComputeNode",
    "DistributedTask", 
    "TaskStatus",
    "NodeType"
]