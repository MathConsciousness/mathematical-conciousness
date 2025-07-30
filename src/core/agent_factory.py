"""Agent Generation System for Mathematical Consciousness Framework

This module provides the Agent class and AgentFactory for creating specialized
computational agents for mathematical analysis and scientific computing.
"""

from typing import Dict, Any, List, Optional, Callable, Union
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from enum import Enum
import uuid

from .logic_field import LogicField


class AgentType(Enum):
    """Types of computational agents"""
    MATHEMATICAL = "mathematical"
    SCIENTIFIC = "scientific"
    ANALYTICAL = "analytical"
    QUANTUM = "quantum"


@dataclass
class AgentCapabilities:
    """Defines the capabilities of an agent"""
    mathematical_analysis: bool = False
    scientific_computing: bool = False
    quantum_simulation: bool = False
    pattern_recognition: bool = False
    optimization: bool = False
    field_operations: bool = False
    
    
@dataclass
class AgentState:
    """Represents the current state of an agent"""
    active: bool = True
    processing_load: float = 0.0
    error_count: int = 0
    last_operation: Optional[str] = None
    memory_usage: float = 0.0


class Agent:
    """Computational agent for mathematical consciousness framework
    
    An agent encapsulates computational capabilities, state management,
    and interaction protocols for the mathematical consciousness system.
    
    Attributes:
        id (str): Unique agent identifier
        agent_type (AgentType): Type of agent
        capabilities (AgentCapabilities): Agent's computational capabilities
        logic_field (LogicField): Associated mathematical field
        state (AgentState): Current agent state
    """

    def __init__(
        self,
        agent_type: AgentType,
        capabilities: AgentCapabilities,
        logic_field: LogicField,
        agent_id: Optional[str] = None,
    ) -> None:
        """Initialize computational agent
        
        Args:
            agent_type: Type of agent to create
            capabilities: Computational capabilities
            logic_field: Associated mathematical field
            agent_id: Optional custom agent ID
        """
        self.id = agent_id or str(uuid.uuid4())
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.logic_field = logic_field
        self.state = AgentState()
        self._operations: Dict[str, Callable] = {}
        self._initialize_operations()

    def _initialize_operations(self) -> None:
        """Initialize available operations based on capabilities"""
        if self.capabilities.mathematical_analysis:
            self._operations["analyze_field"] = self._analyze_field
            self._operations["compute_derivatives"] = self._compute_derivatives
            
        if self.capabilities.scientific_computing:
            self._operations["numerical_integration"] = self._numerical_integration
            self._operations["matrix_operations"] = self._matrix_operations
            
        if self.capabilities.quantum_simulation:
            self._operations["quantum_evolution"] = self._quantum_evolution
            
        if self.capabilities.pattern_recognition:
            self._operations["detect_patterns"] = self._detect_patterns
            
        if self.capabilities.optimization:
            self._operations["optimize_parameters"] = self._optimize_parameters
            
        if self.capabilities.field_operations:
            self._operations["field_manipulation"] = self._field_manipulation

    def execute_operation(self, operation_name: str, **kwargs: Any) -> Any:
        """Execute a computational operation
        
        Args:
            operation_name: Name of operation to execute
            **kwargs: Operation parameters
            
        Returns:
            Operation result
            
        Raises:
            ValueError: If operation is not available
        """
        if operation_name not in self._operations:
            available_ops = list(self._operations.keys())
            raise ValueError(
                f"Operation '{operation_name}' not available. "
                f"Available operations: {available_ops}"
            )
        
        try:
            self.state.last_operation = operation_name
            self.state.processing_load += 0.1  # Simulate load increase
            
            result = self._operations[operation_name](**kwargs)
            
            self.state.processing_load = max(0.0, self.state.processing_load - 0.1)
            return result
            
        except Exception as e:
            self.state.error_count += 1
            self.state.processing_load = max(0.0, self.state.processing_load - 0.1)
            raise RuntimeError(f"Operation '{operation_name}' failed: {e}")

    def _analyze_field(self, **kwargs: Any) -> Dict[str, float]:
        """Analyze the associated logic field"""
        intelligence = self.logic_field.calculate_intelligence_level()
        coherence = self.logic_field.compute_coherence()
        energy = self.logic_field.field_energy()
        
        return {
            "intelligence_level": intelligence,
            "coherence": coherence,
            "field_energy": energy,
            "field_stability": 1.0 - np.linalg.norm(self.logic_field.phi_grad),
        }

    def _compute_derivatives(self, data: NDArray[np.floating], **kwargs: Any) -> NDArray[np.floating]:
        """Compute numerical derivatives"""
        return np.gradient(data)

    def _numerical_integration(self, func: Callable, bounds: tuple, **kwargs: Any) -> float:
        """Perform numerical integration"""
        from scipy.integrate import quad
        result, _ = quad(func, bounds[0], bounds[1])
        return result

    def _matrix_operations(self, operation: str, matrix: NDArray[np.floating], **kwargs: Any) -> NDArray[np.floating]:
        """Perform matrix operations"""
        if operation == "eigenvalues":
            return np.linalg.eigvals(matrix)
        elif operation == "inverse":
            return np.linalg.inv(matrix)
        elif operation == "svd":
            u, s, vh = np.linalg.svd(matrix)
            return np.column_stack([u.flatten(), s, vh.flatten()])
        else:
            raise ValueError(f"Unknown matrix operation: {operation}")

    def _quantum_evolution(self, hamiltonian: NDArray[np.floating], time: float, **kwargs: Any) -> NDArray[np.floating]:
        """Simulate quantum system evolution"""
        from scipy.linalg import expm
        evolution_operator = expm(-1j * hamiltonian * time)
        return evolution_operator

    def _detect_patterns(self, data: NDArray[np.floating], **kwargs: Any) -> Dict[str, Any]:
        """Detect patterns in data"""
        # Simple pattern detection using FFT
        fft_result = np.fft.fft(data.flatten())
        frequencies = np.fft.fftfreq(len(data.flatten()))
        
        # Find dominant frequencies
        magnitude = np.abs(fft_result)
        dominant_freq_idx = np.argmax(magnitude[1:]) + 1  # Skip DC component
        
        return {
            "dominant_frequency": frequencies[dominant_freq_idx],
            "frequency_magnitude": magnitude[dominant_freq_idx],
            "periodicity_score": magnitude[dominant_freq_idx] / np.sum(magnitude),
        }

    def _optimize_parameters(self, objective: Callable, initial_params: NDArray[np.floating], **kwargs: Any) -> Dict[str, Any]:
        """Optimize parameters using scipy"""
        from scipy.optimize import minimize
        
        result = minimize(objective, initial_params, method="BFGS")
        
        return {
            "optimal_params": result.x,
            "optimal_value": result.fun,
            "success": result.success,
            "iterations": result.nit,
        }

    def _field_manipulation(self, operation: str, **kwargs: Any) -> None:
        """Manipulate the logic field"""
        if operation == "evolve":
            dt = kwargs.get("dt", 0.01)
            external_force = kwargs.get("external_force")
            self.logic_field.evolve_field(dt, external_force)
        elif operation == "reset_coherence":
            self.logic_field.sigma = kwargs.get("new_sigma", 1.0)
        else:
            raise ValueError(f"Unknown field operation: {operation}")

    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "id": self.id,
            "type": self.agent_type.value,
            "active": self.state.active,
            "processing_load": self.state.processing_load,
            "error_count": self.state.error_count,
            "last_operation": self.state.last_operation,
            "memory_usage": self.state.memory_usage,
            "available_operations": list(self._operations.keys()),
        }

    def __repr__(self) -> str:
        return f"Agent(id={self.id[:8]}..., type={self.agent_type.value}, active={self.state.active})"


class AgentFactory:
    """Factory for creating specialized computational agents
    
    The AgentFactory creates agents with specific capabilities and configurations
    for different computational tasks in the mathematical consciousness framework.
    """

    def __init__(self) -> None:
        """Initialize the agent factory"""
        self._created_agents: List[Agent] = []

    def create_mathematical_agent(self, params: Dict[str, Any]) -> Agent:
        """Create mathematical analysis agent
        
        Args:
            params: Configuration parameters including:
                - tau: Intelligence parameter (default: 1.0)
                - field_dimension: Dimension of field vectors (default: 3)
                - matrix_size: Size of matrices (default: 3x3)
                - enable_optimization: Enable optimization capabilities (default: True)
                
        Returns:
            Agent: Configured mathematical agent
        """
        # Extract parameters with defaults
        tau = params.get("tau", 1.0)
        field_dim = params.get("field_dimension", 3)
        matrix_size = params.get("matrix_size", 3)
        enable_opt = params.get("enable_optimization", True)
        
        # Create logic field components
        Q = np.eye(matrix_size) + 0.1 * np.random.random((matrix_size, matrix_size))
        Q = Q @ Q.T  # Ensure positive definite
        phi_grad = np.random.normal(0, 0.1, field_dim)
        sigma = params.get("coherence", 0.8)
        S = np.random.normal(0, 0.1, (matrix_size, matrix_size))
        
        # Create logic field
        logic_field = LogicField(tau, Q, phi_grad, sigma, S)
        
        # Define capabilities
        capabilities = AgentCapabilities(
            mathematical_analysis=True,
            scientific_computing=True,
            pattern_recognition=True,
            optimization=enable_opt,
            field_operations=True,
        )
        
        # Create agent
        agent = Agent(AgentType.MATHEMATICAL, capabilities, logic_field)
        self._created_agents.append(agent)
        
        return agent

    def create_scientific_agent(self, params: Dict[str, Any]) -> Agent:
        """Create scientific computing agent
        
        Args:
            params: Configuration parameters including:
                - tau: Intelligence parameter (default: 1.5)
                - enable_quantum: Enable quantum simulation (default: True)
                - field_dimension: Dimension of field vectors (default: 4)
                - matrix_size: Size of matrices (default: 4x4)
                
        Returns:
            Agent: Configured scientific agent
        """
        # Extract parameters with defaults
        tau = params.get("tau", 1.5)
        field_dim = params.get("field_dimension", 4)
        matrix_size = params.get("matrix_size", 4)
        enable_quantum = params.get("enable_quantum", True)
        
        # Create logic field components
        Q = np.eye(matrix_size) + 0.2 * np.random.random((matrix_size, matrix_size))
        Q = Q @ Q.T  # Ensure positive definite
        phi_grad = np.random.normal(0, 0.05, field_dim)
        sigma = params.get("coherence", 0.9)
        S = np.random.normal(0, 0.05, (matrix_size, matrix_size))
        
        # Create logic field
        logic_field = LogicField(tau, Q, phi_grad, sigma, S)
        
        # Define capabilities
        capabilities = AgentCapabilities(
            mathematical_analysis=True,
            scientific_computing=True,
            quantum_simulation=enable_quantum,
            pattern_recognition=True,
            optimization=True,
            field_operations=True,
        )
        
        # Create agent
        agent = Agent(AgentType.SCIENTIFIC, capabilities, logic_field)
        self._created_agents.append(agent)
        
        return agent

    def create_quantum_agent(self, params: Dict[str, Any]) -> Agent:
        """Create quantum simulation agent
        
        Args:
            params: Configuration parameters for quantum agent
                
        Returns:
            Agent: Configured quantum agent
        """
        tau = params.get("tau", 2.0)
        field_dim = params.get("field_dimension", 2)
        matrix_size = params.get("matrix_size", 2)
        
        # Create minimal logic field for quantum operations
        Q = np.eye(matrix_size)
        phi_grad = np.zeros(field_dim)
        sigma = params.get("coherence", 0.95)
        S = np.eye(matrix_size)
        
        logic_field = LogicField(tau, Q, phi_grad, sigma, S)
        
        capabilities = AgentCapabilities(
            quantum_simulation=True,
            scientific_computing=True,
            mathematical_analysis=True,
            field_operations=True,
        )
        
        agent = Agent(AgentType.QUANTUM, capabilities, logic_field)
        self._created_agents.append(agent)
        
        return agent

    def get_created_agents(self) -> List[Agent]:
        """Get list of all created agents"""
        return self._created_agents.copy()

    def get_agent_statistics(self) -> Dict[str, Any]:
        """Get statistics about created agents"""
        if not self._created_agents:
            return {"total_agents": 0}
        
        active_agents = sum(1 for agent in self._created_agents if agent.state.active)
        avg_load = np.mean([agent.state.processing_load for agent in self._created_agents])
        total_errors = sum(agent.state.error_count for agent in self._created_agents)
        
        type_counts = {}
        for agent in self._created_agents:
            agent_type = agent.agent_type.value
            type_counts[agent_type] = type_counts.get(agent_type, 0) + 1
        
        return {
            "total_agents": len(self._created_agents),
            "active_agents": active_agents,
            "average_processing_load": avg_load,
            "total_errors": total_errors,
            "agent_types": type_counts,
        }