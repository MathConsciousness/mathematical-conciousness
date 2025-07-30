"""Mathematical Consciousness Framework - Advanced Scientific Computing"""

__version__ = "0.1.0"
__author__ = "Mathematical Consciousness"

from .core.logic_field import LogicField
from .core.agent_factory import AgentFactory, Agent, AgentType
from .network.builder import NetworkBuilder, NetworkMetrics
from .computing.protocols import ScientificProtocols

__all__ = [
    "LogicField",
    "AgentFactory",
    "Agent", 
    "AgentType",
    "NetworkBuilder",
    "NetworkMetrics",
    "ScientificProtocols",
]