"""
Basic tests for the Mathematical Framework System core components.
"""

import pytest
import numpy as np
import asyncio
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.logic_field import LogicField
from core.agent_factory import AgentFactory, ComputationalAgent
from network.builder import NetworkBuilder

class TestLogicField:
    """Test cases for LogicField class."""
    
    def test_logic_field_initialization(self):
        """Test LogicField initialization with valid parameters."""
        field = LogicField(
            tau=0.95,
            Q=np.eye(3),
            phi_grad=np.zeros(3),
            sigma=0.8,
            S=np.identity(4)
        )
        
        assert field.tau == 0.95
        assert field.sigma == 0.8
        assert field.Q.shape == (3, 3)
        assert field.S.shape == (4, 4)
    
    def test_logic_field_invalid_tau(self):
        """Test LogicField raises error for invalid tau."""
        with pytest.raises(ValueError):
            LogicField(
                tau=1.5,  # Invalid: > 1
                Q=np.eye(3),
                phi_grad=np.zeros(3),
                sigma=0.8,
                S=np.identity(4)
            )
    
    def test_compute_field_strength(self):
        """Test field strength computation."""
        field = LogicField(
            tau=0.95,
            Q=np.eye(3),
            phi_grad=np.zeros(3),
            sigma=0.8,
            S=np.identity(4)
        )
        
        x = np.array([1.0, 0.0, 0.0])
        t = 0.5
        strength = field.compute_field_strength(x, t)
        
        assert isinstance(strength, float)
        assert strength > 0
    
    def test_evolve_state(self):
        """Test quantum state evolution."""
        field = LogicField(
            tau=0.95,
            Q=np.eye(3),
            phi_grad=np.zeros(3),
            sigma=0.8,
            S=np.identity(4)
        )
        
        initial_state = np.array([1.0, 0.0, 0.0, 0.0])
        evolved_state = field.evolve_state(initial_state, dt=0.1)
        
        assert evolved_state.shape == initial_state.shape
        assert np.isclose(np.linalg.norm(evolved_state), 1.0, atol=1e-10)

class TestAgentFactory:
    """Test cases for AgentFactory class."""
    
    @pytest.mark.asyncio
    async def test_create_single_agent(self):
        """Test creating a single agent."""
        factory = AgentFactory()
        agent = await factory.create_agent()
        
        assert isinstance(agent, ComputationalAgent)
        assert agent.agent_id.startswith("agent_")
        assert agent.status == "active"
        assert len(agent.capabilities) > 0
    
    @pytest.mark.asyncio
    async def test_deploy_agents(self):
        """Test deploying multiple agents."""
        factory = AgentFactory()
        agents = await factory.deploy_all_agents(count=10)
        
        assert len(agents) == 10
        assert all(isinstance(agent, ComputationalAgent) for agent in agents)
        assert len(set(agent.agent_id for agent in agents)) == 10  # All unique IDs
    
    @pytest.mark.asyncio
    async def test_system_metrics(self):
        """Test getting system metrics."""
        factory = AgentFactory()
        agents = await factory.deploy_all_agents(count=5)
        metrics = await factory.get_system_metrics()
        
        assert metrics["total_agents"] == 5
        assert metrics["status"] == "operational"
        assert "average_capabilities" in metrics

class TestNetworkBuilder:
    """Test cases for NetworkBuilder class."""
    
    @pytest.mark.asyncio
    async def test_build_small_network(self):
        """Test building a small network."""
        # Create agents
        factory = AgentFactory()
        agents = await factory.deploy_all_agents(count=5)
        
        # Build network
        builder = NetworkBuilder()
        network = await builder.build_network(agents, target_density=0.8)
        
        assert len(network.graph.nodes()) == 5
        assert network.density() > 0
        assert network.density() <= 1.0
    
    @pytest.mark.asyncio
    async def test_network_density_target(self):
        """Test that network achieves approximate target density."""
        # Create agents
        factory = AgentFactory()
        agents = await factory.deploy_all_agents(count=10)
        
        # Build network with specific target
        builder = NetworkBuilder()
        target_density = 0.7
        network = await builder.build_network(agents, target_density=target_density)
        
        actual_density = network.density()
        assert abs(actual_density - target_density) < 0.1  # Within 10%
    
    @pytest.mark.asyncio
    async def test_network_metrics(self):
        """Test network metrics calculation."""
        # Create agents
        factory = AgentFactory()
        agents = await factory.deploy_all_agents(count=6)
        
        # Build network
        builder = NetworkBuilder()
        network = await builder.build_network(agents, target_density=0.6)
        
        metrics = network.get_metrics()
        
        assert "node_count" in metrics
        assert "edge_count" in metrics
        assert "density" in metrics
        assert "is_connected" in metrics
        assert metrics["node_count"] == 6

class TestSystemIntegration:
    """Integration tests for the complete system."""
    
    @pytest.mark.asyncio
    async def test_full_system_initialization(self):
        """Test complete system initialization process."""
        # Initialize LogicField
        field = LogicField(
            tau=0.95,
            Q=np.eye(3),
            phi_grad=np.zeros(3),
            sigma=0.8,
            S=np.identity(4)
        )
        
        # Deploy agents
        factory = AgentFactory()
        agents = await factory.deploy_all_agents(count=10)
        
        # Build network
        builder = NetworkBuilder()
        network = await builder.build_network(agents, target_density=0.8)
        
        # Verify all components
        assert field is not None
        assert len(agents) == 10
        assert network.density() > 0
        
        # Test field computations with agents
        for agent in agents[:3]:  # Test with first 3 agents
            strength = field.compute_field_strength(agent.position, 0.5)
            assert strength > 0
    
    @pytest.mark.asyncio
    async def test_agent_processing(self):
        """Test agent task processing capabilities."""
        factory = AgentFactory()
        agents = await factory.deploy_all_agents(count=3)
        
        # Test processing with each agent
        for agent in agents:
            result = agent.process_task({"test_data": "sample"})
            assert result["agent_id"] == agent.agent_id
            assert result["status"] == "completed"
            assert "processing_time" in result

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])