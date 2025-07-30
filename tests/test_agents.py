"""Test suite for Agent and AgentFactory classes

Tests agent creation, capabilities, and operations.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from src.core.agent_factory import AgentFactory, Agent, AgentType, AgentCapabilities
from src.core.logic_field import LogicField


class TestAgent:
    """Test cases for Agent class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        # Create a simple logic field
        tau = 1.0
        Q = np.eye(2)
        phi_grad = np.array([0.1, 0.2])
        sigma = 0.8
        S = np.eye(2)
        
        self.logic_field = LogicField(tau, Q, phi_grad, sigma, S)
        
        # Create capabilities
        self.capabilities = AgentCapabilities(
            mathematical_analysis=True,
            scientific_computing=True,
            optimization=True,
        )
        
        # Create agent
        self.agent = Agent(
            AgentType.MATHEMATICAL,
            self.capabilities,
            self.logic_field,
            agent_id="test-agent"
        )
    
    def test_agent_initialization(self):
        """Test proper agent initialization"""
        assert self.agent.id == "test-agent"
        assert self.agent.agent_type == AgentType.MATHEMATICAL
        assert self.agent.capabilities == self.capabilities
        assert self.agent.logic_field == self.logic_field
        assert self.agent.state.active is True
        assert self.agent.state.processing_load == 0.0
        assert self.agent.state.error_count == 0
    
    def test_agent_auto_id_generation(self):
        """Test automatic ID generation"""
        agent = Agent(AgentType.SCIENTIFIC, self.capabilities, self.logic_field)
        assert agent.id is not None
        assert len(agent.id) > 0
        assert agent.id != "test-agent"
    
    def test_available_operations(self):
        """Test that operations are properly initialized based on capabilities"""
        operations = list(self.agent._operations.keys())
        
        # Should have mathematical operations
        assert "analyze_field" in operations
        assert "compute_derivatives" in operations
        
        # Should have scientific computing operations
        assert "numerical_integration" in operations
        assert "matrix_operations" in operations
        
        # Should have optimization
        assert "optimize_parameters" in operations
        
        # Should NOT have quantum operations (not enabled)
        assert "quantum_evolution" not in operations
    
    def test_execute_operation_field_analysis(self):
        """Test field analysis operation"""
        result = self.agent.execute_operation("analyze_field")
        
        assert isinstance(result, dict)
        assert "intelligence_level" in result
        assert "coherence" in result
        assert "field_energy" in result
        assert "field_stability" in result
        
        # Values should be reasonable
        assert result["intelligence_level"] > 0
        assert 0 <= result["coherence"] <= 1
        assert result["field_energy"] >= 0
    
    def test_execute_operation_derivatives(self):
        """Test derivative computation operation"""
        data = np.array([1, 4, 9, 16, 25])  # x^2 for x = 1,2,3,4,5
        result = self.agent.execute_operation("compute_derivatives", data=data)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(data)
    
    def test_execute_operation_numerical_integration(self):
        """Test numerical integration operation"""
        # Integrate x^2 from 0 to 1 (should be 1/3)
        func = lambda x: x**2
        result = self.agent.execute_operation(
            "numerical_integration", 
            func=func, 
            bounds=(0, 1)
        )
        
        assert isinstance(result, float)
        assert abs(result - 1/3) < 0.01  # Should be close to analytical result
    
    def test_execute_operation_matrix_operations(self):
        """Test matrix operations"""
        matrix = np.array([[1, 2], [3, 4]])
        
        # Test eigenvalues
        eigenvals = self.agent.execute_operation(
            "matrix_operations",
            operation="eigenvalues",
            matrix=matrix
        )
        assert isinstance(eigenvals, np.ndarray)
        assert len(eigenvals) == 2
        
        # Test inverse
        matrix_inv = np.array([[2, -1], [-1.5, 0.5]])  # Invertible matrix
        inv_result = self.agent.execute_operation(
            "matrix_operations",
            operation="inverse",
            matrix=matrix_inv
        )
        assert isinstance(inv_result, np.ndarray)
        assert inv_result.shape == matrix_inv.shape
    
    def test_execute_operation_optimization(self):
        """Test parameter optimization"""
        # Simple quadratic function: (x-2)^2 + (y-3)^2
        def objective(params):
            x, y = params
            return (x - 2)**2 + (y - 3)**2
        
        initial_params = np.array([0.0, 0.0])
        result = self.agent.execute_operation(
            "optimize_parameters",
            objective=objective,
            initial_params=initial_params
        )
        
        assert isinstance(result, dict)
        assert "optimal_params" in result
        assert "optimal_value" in result
        assert "success" in result
        
        # Should find minimum near (2, 3)
        optimal_params = np.array(result["optimal_params"])
        assert abs(optimal_params[0] - 2.0) < 0.1
        assert abs(optimal_params[1] - 3.0) < 0.1
    
    def test_execute_operation_processing_load(self):
        """Test that operations update processing load"""
        initial_load = self.agent.state.processing_load
        
        self.agent.execute_operation("analyze_field")
        
        # Load should be back to original after operation completes
        assert self.agent.state.processing_load == initial_load
        assert self.agent.state.last_operation == "analyze_field"
    
    def test_execute_operation_invalid(self):
        """Test executing invalid operation"""
        with pytest.raises(ValueError, match="not available"):
            self.agent.execute_operation("invalid_operation")
    
    def test_execute_operation_error_handling(self):
        """Test error handling in operations"""
        # This should cause an error in matrix inversion
        singular_matrix = np.array([[1, 1], [1, 1]])
        
        with pytest.raises(RuntimeError, match="Operation.*failed"):
            self.agent.execute_operation(
                "matrix_operations",
                operation="inverse",
                matrix=singular_matrix
            )
        
        # Error count should increase
        assert self.agent.state.error_count > 0
    
    def test_get_status(self):
        """Test agent status reporting"""
        status = self.agent.get_status()
        
        assert isinstance(status, dict)
        assert status["id"] == self.agent.id
        assert status["type"] == self.agent.agent_type.value
        assert status["active"] == self.agent.state.active
        assert "processing_load" in status
        assert "error_count" in status
        assert "available_operations" in status
        
        # Check available operations
        operations = status["available_operations"]
        assert "analyze_field" in operations
    
    def test_agent_repr(self):
        """Test agent string representation"""
        repr_str = repr(self.agent)
        assert "Agent" in repr_str
        assert self.agent.id[:8] in repr_str
        assert self.agent.agent_type.value in repr_str


class TestAgentCapabilities:
    """Test agent capabilities with different configurations"""
    
    def test_quantum_agent_capabilities(self):
        """Test agent with quantum capabilities"""
        tau = 1.5
        Q = np.eye(2)
        phi_grad = np.array([0.0, 0.0])
        sigma = 0.9
        S = np.eye(2)
        
        logic_field = LogicField(tau, Q, phi_grad, sigma, S)
        
        capabilities = AgentCapabilities(
            quantum_simulation=True,
            scientific_computing=True,
            mathematical_analysis=True,
        )
        
        agent = Agent(AgentType.QUANTUM, capabilities, logic_field)
        
        # Should have quantum operations
        operations = list(agent._operations.keys())
        assert "quantum_evolution" in operations
        
        # Test quantum evolution operation
        hamiltonian = np.array([[1, 0], [0, -1]])
        result = agent.execute_operation(
            "quantum_evolution",
            hamiltonian=hamiltonian,
            time=0.1
        )
        
        assert isinstance(result, np.ndarray)
        assert result.shape == hamiltonian.shape
    
    def test_pattern_recognition_capabilities(self):
        """Test agent with pattern recognition"""
        capabilities = AgentCapabilities(pattern_recognition=True)
        logic_field = LogicField(1.0, np.eye(2), np.array([0, 0]), 0.8, np.eye(2))
        
        agent = Agent(AgentType.ANALYTICAL, capabilities, logic_field)
        
        # Test pattern detection
        # Create data with a clear pattern
        t = np.linspace(0, 4*np.pi, 100)
        data = np.sin(t) + 0.1 * np.random.normal(size=len(t))
        
        result = agent.execute_operation("detect_patterns", data=data)
        
        assert isinstance(result, dict)
        assert "dominant_frequency" in result
        assert "periodicity_score" in result
    
    def test_field_operations_capabilities(self):
        """Test agent with field operation capabilities"""
        capabilities = AgentCapabilities(field_operations=True)
        logic_field = LogicField(1.0, np.eye(2), np.array([0.1, 0.2]), 0.8, np.eye(2))
        
        agent = Agent(AgentType.MATHEMATICAL, capabilities, logic_field)
        
        # Test field evolution
        original_grad = agent.logic_field.phi_grad.copy()
        agent.execute_operation("field_manipulation", operation="evolve", dt=0.01)
        
        # Field should have evolved
        assert not np.allclose(agent.logic_field.phi_grad, original_grad)
        
        # Test coherence reset
        agent.execute_operation("field_manipulation", operation="reset_coherence", new_sigma=0.5)
        assert agent.logic_field.sigma == 0.5


class TestAgentFactory:
    """Test cases for AgentFactory class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.factory = AgentFactory()
    
    def test_factory_initialization(self):
        """Test factory initialization"""
        assert isinstance(self.factory, AgentFactory)
        assert len(self.factory._created_agents) == 0
    
    def test_create_mathematical_agent(self):
        """Test mathematical agent creation"""
        params = {
            "tau": 1.5,
            "field_dimension": 4,
            "matrix_size": 3,
            "coherence": 0.9,
            "enable_optimization": True,
        }
        
        agent = self.factory.create_mathematical_agent(params)
        
        assert isinstance(agent, Agent)
        assert agent.agent_type == AgentType.MATHEMATICAL
        assert agent.logic_field.tau == params["tau"]
        assert len(agent.logic_field.phi_grad) == params["field_dimension"]
        assert agent.logic_field.Q.shape == (params["matrix_size"], params["matrix_size"])
        assert agent.logic_field.sigma == params["coherence"]
        
        # Check capabilities
        assert agent.capabilities.mathematical_analysis is True
        assert agent.capabilities.scientific_computing is True
        assert agent.capabilities.optimization == params["enable_optimization"]
        
        # Should be in factory's created agents list
        assert agent in self.factory._created_agents
    
    def test_create_scientific_agent(self):
        """Test scientific agent creation"""
        params = {
            "tau": 2.0,
            "field_dimension": 5,
            "matrix_size": 4,
            "coherence": 0.85,
            "enable_quantum": True,
        }
        
        agent = self.factory.create_scientific_agent(params)
        
        assert isinstance(agent, Agent)
        assert agent.agent_type == AgentType.SCIENTIFIC
        assert agent.logic_field.tau == params["tau"]
        assert len(agent.logic_field.phi_grad) == params["field_dimension"]
        assert agent.logic_field.Q.shape == (params["matrix_size"], params["matrix_size"])
        assert agent.capabilities.quantum_simulation == params["enable_quantum"]
    
    def test_create_quantum_agent(self):
        """Test quantum agent creation"""
        params = {
            "tau": 3.0,
            "field_dimension": 2,
            "matrix_size": 2,
            "coherence": 0.95,
        }
        
        agent = self.factory.create_quantum_agent(params)
        
        assert isinstance(agent, Agent)
        assert agent.agent_type == AgentType.QUANTUM
        assert agent.logic_field.tau == params["tau"]
        assert agent.capabilities.quantum_simulation is True
    
    def test_create_multiple_agents(self):
        """Test creating multiple agents"""
        params = {"tau": 1.0}
        
        agent1 = self.factory.create_mathematical_agent(params)
        agent2 = self.factory.create_scientific_agent(params)
        agent3 = self.factory.create_quantum_agent(params)
        
        created_agents = self.factory.get_created_agents()
        assert len(created_agents) == 3
        assert agent1 in created_agents
        assert agent2 in created_agents
        assert agent3 in created_agents
        
        # Each agent should have unique ID
        ids = [agent.id for agent in created_agents]
        assert len(set(ids)) == 3
    
    def test_agent_default_parameters(self):
        """Test agent creation with default parameters"""
        agent = self.factory.create_mathematical_agent({})
        
        # Should use default values
        assert agent.logic_field.tau == 1.0
        assert len(agent.logic_field.phi_grad) == 3
        assert agent.logic_field.Q.shape == (3, 3)
        assert agent.logic_field.sigma == 0.8
        assert agent.capabilities.optimization is True
    
    def test_get_agent_statistics(self):
        """Test agent statistics reporting"""
        # Initially empty
        stats = self.factory.get_agent_statistics()
        assert stats["total_agents"] == 0
        
        # Create some agents
        self.factory.create_mathematical_agent({"tau": 1.0})
        self.factory.create_scientific_agent({"tau": 1.5})
        self.factory.create_quantum_agent({"tau": 2.0})
        
        stats = self.factory.get_agent_statistics()
        assert stats["total_agents"] == 3
        assert stats["active_agents"] == 3  # All should be active initially
        assert "average_processing_load" in stats
        assert "total_errors" in stats
        assert "agent_types" in stats
        
        # Check agent type counts
        type_counts = stats["agent_types"]
        assert type_counts["mathematical"] == 1
        assert type_counts["scientific"] == 1
        assert type_counts["quantum"] == 1
    
    def test_agent_statistics_with_errors(self):
        """Test statistics with agents that have errors"""
        agent = self.factory.create_mathematical_agent({})
        
        # Cause an error
        try:
            agent.execute_operation("invalid_operation")
        except ValueError:
            pass
        
        stats = self.factory.get_agent_statistics()
        assert stats["total_errors"] > 0
    
    def test_field_component_validation(self):
        """Test that created agents have valid field components"""
        agent = self.factory.create_mathematical_agent({
            "matrix_size": 4,
            "field_dimension": 5,
        })
        
        # Q matrix should be positive definite
        eigenvals = np.linalg.eigvals(agent.logic_field.Q)
        assert np.all(eigenvals > 0)
        
        # Q matrix should be symmetric
        Q = agent.logic_field.Q
        assert np.allclose(Q, Q.T)
        
        # Field gradient should have correct dimension
        assert len(agent.logic_field.phi_grad) == 5
        
        # S matrix should be square
        S = agent.logic_field.S
        assert S.shape[0] == S.shape[1] == 4