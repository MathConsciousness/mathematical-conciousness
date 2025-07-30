"""Integration tests for the Mathematical Framework System."""

import pytest
import numpy as np
import asyncio
from core.logic_field import LogicField
from core.agent_factory import AgentFactory
from network.builder import NetworkBuilder


class TestSystemIntegration:
    """Test class for complete system integration validation."""
    
    def test_system_integration(self):
        """Validate complete system integration."""
        # Initialize LogicField
        field = LogicField(
            tau=0.95, 
            Q=np.eye(3), 
            phi_grad=np.zeros(3), 
            sigma=0.8, 
            S=np.identity(4)
        )
        intelligence_level = field.calculate_intelligence_level()
        assert intelligence_level > 0.9, f"Intelligence level {intelligence_level} below threshold"
        
        # Test agent deployment
        factory = AgentFactory()
        agents = factory.deploy_all_agents(count=91)
        assert len(agents) == 91, f"Agent deployment incomplete: {len(agents)} != 91"
        
        # Validate network density
        builder = NetworkBuilder()
        network = builder.build_network(agents, target_density=0.964)
        density = network.density()
        assert abs(density - 0.964) < 0.01, f"Network density {density} outside target range"
    
    def test_logic_field_initialization(self):
        """Test LogicField initialization and basic functionality."""
        # Test with various parameter combinations
        field = LogicField(
            tau=0.95,
            Q=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            phi_grad=np.array([0.1, 0.2, 0.3]),
            sigma=0.8,
            S=np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        )
        
        assert field.tau == 0.95
        assert field.sigma == 0.8
        assert field.calculate_intelligence_level() > 0.9
        
    def test_logic_field_parameter_validation(self):
        """Test LogicField parameter validation."""
        # Test invalid tau
        with pytest.raises(ValueError, match="tau must be between 0 and 1"):
            LogicField(
                tau=1.5, 
                Q=np.eye(3), 
                phi_grad=np.zeros(3), 
                sigma=0.8, 
                S=np.identity(4)
            )
        
        # Test invalid sigma
        with pytest.raises(ValueError, match="sigma must be positive"):
            LogicField(
                tau=0.95, 
                Q=np.eye(3), 
                phi_grad=np.zeros(3), 
                sigma=-0.1, 
                S=np.identity(4)
            )
    
    def test_agent_factory_deployment(self):
        """Test AgentFactory deployment functionality."""
        factory = AgentFactory()
        
        # Test exact count requirement
        agents = factory.deploy_all_agents(count=91)
        assert len(agents) == 91
        
        # Test that all agents are unique
        agent_ids = [agent.id for agent in agents]
        assert len(set(agent_ids)) == 91, "Duplicate agent IDs found"
        
        # Test agent properties
        for i, agent in enumerate(agents):
            assert agent.name == f"Agent_{i:03d}"
            assert agent.intelligence_level >= 0.91
            assert agent.status == "active"
            assert len(agent.capabilities) >= 5
    
    def test_agent_factory_invalid_count(self):
        """Test AgentFactory with invalid agent count."""
        factory = AgentFactory()
        
        # Should raise error for non-91 count
        with pytest.raises(ValueError, match="System requires exactly 91 agents"):
            factory.deploy_all_agents(count=90)
        
        with pytest.raises(ValueError, match="System requires exactly 91 agents"):
            factory.deploy_all_agents(count=92)
    
    @pytest.mark.asyncio
    async def test_async_agent_deployment(self):
        """Test asynchronous agent deployment."""
        factory = AgentFactory()
        agents = await factory.deploy_all_agents_async(count=91)
        
        assert len(agents) == 91
        assert all(agent.status == "active" for agent in agents)
    
    def test_network_builder_basic(self):
        """Test NetworkBuilder basic functionality."""
        # Create test agents
        factory = AgentFactory()
        agents = factory.deploy_all_agents(count=91)
        
        # Build network
        builder = NetworkBuilder()
        network = builder.build_network(agents, target_density=0.964)
        
        # Validate network properties
        assert network.graph.number_of_nodes() == 91
        density = network.density()
        assert abs(density - 0.964) < 0.01, f"Density {density} not close to target 0.964"
        
        # Check connectivity
        stats = network.get_network_stats()
        assert stats["total_nodes"] == 91
        assert stats["is_connected"] is True
    
    def test_network_density_variations(self):
        """Test network building with different density targets."""
        factory = AgentFactory()
        agents = factory.deploy_all_agents(count=91)
        builder = NetworkBuilder()
        
        # Test different density targets
        for target_density in [0.5, 0.7, 0.9, 0.964]:
            network = builder.build_network(agents, target_density=target_density)
            actual_density = network.density()
            assert abs(actual_density - target_density) < 0.05, \
                f"Density {actual_density} too far from target {target_density}"
    
    @pytest.mark.asyncio
    async def test_async_network_building(self):
        """Test asynchronous network building."""
        factory = AgentFactory()
        agents = await factory.deploy_all_agents_async(count=91)
        
        builder = NetworkBuilder()
        network = await builder.build_network_async(agents, target_density=0.964)
        
        assert network.graph.number_of_nodes() == 91
        assert abs(network.density() - 0.964) < 0.01
    
    def test_network_topology_analysis(self):
        """Test network topology analysis functionality."""
        factory = AgentFactory()
        agents = factory.deploy_all_agents(count=91)
        
        builder = NetworkBuilder()
        network = builder.build_network(agents, target_density=0.964)
        
        analysis = builder.analyze_network_topology(network)
        
        # Check analysis structure
        assert "basic_stats" in analysis
        assert "centrality_measures" in analysis
        assert "topology_metrics" in analysis
        
        # Validate basic stats
        basic_stats = analysis["basic_stats"]
        assert basic_stats["total_nodes"] == 91
        assert basic_stats["density"] > 0.9
        
        # Check centrality measures exist
        centrality = analysis["centrality_measures"]
        assert "degree_centrality" in centrality
        assert "betweenness_centrality" in centrality
        assert "closeness_centrality" in centrality
    
    def test_full_system_workflow(self):
        """Test complete system workflow from start to finish."""
        # Step 1: Initialize logic field
        field = LogicField(
            tau=0.95,
            Q=np.eye(3),
            phi_grad=np.zeros(3),
            sigma=0.8,
            S=np.identity(4)
        )
        
        # Step 2: Deploy agents
        factory = AgentFactory()
        agents = factory.deploy_all_agents(count=91)
        
        # Step 3: Build network
        builder = NetworkBuilder()
        network = builder.build_network(agents, target_density=0.964)
        
        # Step 4: Validate complete system
        assert field.calculate_intelligence_level() > 0.9
        assert len(agents) == 91
        assert abs(network.density() - 0.964) < 0.01
        
        # Step 5: Get system statistics
        field_params = field.get_parameters()
        agent_stats = factory.get_deployment_stats()
        network_stats = network.get_network_stats()
        
        # Validate system is ready for deployment
        assert field_params["tau"] == 0.95
        assert agent_stats["total_agents"] == 91
        assert agent_stats["active_agents"] == 91
        assert network_stats["is_connected"] is True
        assert network_stats["density"] > 0.96
    
    def test_system_resilience(self):
        """Test system resilience and error handling."""
        # Test with edge case parameters
        field = LogicField(
            tau=0.99,  # Very high tau
            Q=np.eye(10),  # Larger matrix
            phi_grad=np.ones(10) * 0.1,  # Non-zero gradient
            sigma=1.0,  # Maximum sigma
            S=np.identity(10)
        )
        
        assert field.calculate_intelligence_level() <= 1.0  # Should not exceed 1
        
        # Test network with minimal agents
        factory = AgentFactory()
        agents = factory.deploy_all_agents(count=91)
        
        # Test with lower density
        builder = NetworkBuilder()
        network = builder.build_network(agents, target_density=0.1)
        
        # Should still be connected
        assert network.get_network_stats()["is_connected"] is True
    
    def test_performance_benchmarks(self):
        """Test system performance benchmarks."""
        import time
        
        # Benchmark agent deployment
        factory = AgentFactory()
        start_time = time.time()
        agents = factory.deploy_all_agents(count=91)
        deployment_time = time.time() - start_time
        
        assert deployment_time < 1.0, f"Agent deployment too slow: {deployment_time}s"
        
        # Benchmark network building
        builder = NetworkBuilder()
        start_time = time.time()
        network = builder.build_network(agents, target_density=0.964)
        network_time = time.time() - start_time
        
        assert network_time < 5.0, f"Network building too slow: {network_time}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])