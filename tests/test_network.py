"""Test suite for network connectivity and architecture

Tests the NetworkBuilder class and network optimization functionality.
"""

import pytest
import numpy as np
import networkx as nx
from unittest.mock import Mock

from src.network.builder import NetworkBuilder, NetworkMetrics, ConnectionWeight
from src.core.agent_factory import AgentFactory, AgentType


class TestNetworkBuilder:
    """Test cases for NetworkBuilder class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.builder = NetworkBuilder(random_seed=42)
        self.factory = AgentFactory()
        
        # Create a set of test agents
        self.agents = []
        for i in range(5):
            agent = self.factory.create_mathematical_agent({
                "tau": 1.0 + i * 0.2,
                "coherence": 0.7 + i * 0.05,
            })
            self.agents.append(agent)
    
    def test_builder_initialization(self):
        """Test NetworkBuilder initialization"""
        builder = NetworkBuilder()
        assert isinstance(builder, NetworkBuilder)
        assert len(builder.networks) == 0
        assert len(builder.agent_networks) == 0
        
        # Test with random seed
        builder_seeded = NetworkBuilder(random_seed=123)
        assert isinstance(builder_seeded, NetworkBuilder)
    
    def test_build_network_basic(self):
        """Test basic network construction"""
        network = self.builder.build_network(self.agents, target_density=0.5)
        
        assert isinstance(network, nx.Graph)
        assert network.number_of_nodes() == len(self.agents)
        
        # Check that all agents are represented as nodes
        node_ids = set(network.nodes())
        agent_ids = {agent.id for agent in self.agents}
        assert node_ids == agent_ids
        
        # Network should be connected (at least MST edges)
        assert nx.is_connected(network)
        
        # Density should be approximately correct
        actual_density = nx.density(network)
        assert 0.4 <= actual_density <= 0.6  # Allow some tolerance
    
    def test_build_network_high_density(self):
        """Test network with high target density"""
        network = self.builder.build_network(self.agents, target_density=0.9)
        
        actual_density = nx.density(network)
        assert actual_density >= 0.8  # Should be close to target
        assert nx.is_connected(network)
    
    def test_build_network_low_density(self):
        """Test network with low target density"""
        network = self.builder.build_network(self.agents, target_density=0.1)
        
        actual_density = nx.density(network)
        assert actual_density >= 0.1  # At least minimum for connectivity
        assert nx.is_connected(network)
    
    def test_build_network_validation(self):
        """Test network building parameter validation"""
        # Empty agent list
        with pytest.raises(ValueError, match="empty agent list"):
            self.builder.build_network([])
        
        # Invalid density
        with pytest.raises(ValueError, match="between 0 and 1"):
            self.builder.build_network(self.agents, target_density=1.5)
        
        with pytest.raises(ValueError, match="between 0 and 1"):
            self.builder.build_network(self.agents, target_density=-0.1)
        
        # Too few agents
        with pytest.raises(ValueError, match="at least 2 agents"):
            self.builder.build_network([self.agents[0]])
    
    def test_build_network_node_properties(self):
        """Test that nodes have correct properties"""
        network = self.builder.build_network(self.agents)
        
        for agent in self.agents:
            node_data = network.nodes[agent.id]
            
            assert node_data["agent_type"] == agent.agent_type.value
            assert "intelligence" in node_data
            assert "coherence" in node_data
            assert "capabilities" in node_data
            
            # Values should be reasonable
            assert node_data["intelligence"] > 0
            assert 0 <= node_data["coherence"] <= 1
    
    def test_build_network_edge_properties(self):
        """Test that edges have correct properties"""
        network = self.builder.build_network(self.agents, target_density=0.8)
        
        for edge in network.edges():
            edge_data = network.edges[edge]
            
            assert "weight" in edge_data
            assert "connection" in edge_data
            
            connection = edge_data["connection"]
            assert isinstance(connection, ConnectionWeight)
            assert 0 <= connection.strength <= 2  # Allow some flexibility
            assert connection.latency >= 0
            assert connection.bandwidth >= 0
            assert 0 <= connection.compatibility <= 2
    
    def test_build_multiple_networks(self):
        """Test building multiple networks"""
        network1 = self.builder.build_network(self.agents[:3], network_id="net1")
        network2 = self.builder.build_network(self.agents[2:], network_id="net2")
        
        assert len(self.builder.networks) == 2
        assert "net1" in self.builder.networks
        assert "net2" in self.builder.networks
        
        assert network1.number_of_nodes() == 3
        assert network2.number_of_nodes() == 3
    
    def test_optimize_information_flow(self):
        """Test network optimization"""
        network = self.builder.build_network(self.agents, target_density=0.3)
        
        # Get metrics before optimization
        metrics_before = self.builder.get_network_metrics(network)
        original_edge_count = network.number_of_edges()
        
        # Optimize the network
        self.builder.optimize_information_flow(network)
        
        # Network should still be valid
        assert nx.is_connected(network)
        assert network.number_of_nodes() == len(self.agents)
        
        # Get metrics after optimization
        metrics_after = self.builder.get_network_metrics(network)
        
        # Some metric should improve (efficiency is most likely)
        assert metrics_after.efficiency >= metrics_before.efficiency
    
    def test_get_network_metrics(self):
        """Test network metrics computation"""
        network = self.builder.build_network(self.agents, target_density=0.6)
        metrics = self.builder.get_network_metrics(network)
        
        assert isinstance(metrics, NetworkMetrics)
        assert metrics.node_count == len(self.agents)
        assert metrics.edge_count == network.number_of_edges()
        assert 0 <= metrics.density <= 1
        assert 0 <= metrics.average_clustering <= 1
        assert metrics.average_path_length > 0
        assert 0 <= metrics.efficiency <= 1
        assert -1 <= metrics.modularity <= 1
        assert 0 <= metrics.connectivity_score <= 1
    
    def test_get_network_metrics_edge_cases(self):
        """Test network metrics with edge cases"""
        # Single node network (degenerate case)
        single_agent = [self.agents[0]]
        single_network = nx.Graph()
        single_network.add_node(self.agents[0].id)
        
        metrics = self.builder.get_network_metrics(single_network)
        assert metrics.node_count == 1
        assert metrics.edge_count == 0
        assert metrics.density == 0.0
        
        # Two node network
        two_agents = self.agents[:2]
        two_network = self.builder.build_network(two_agents)
        
        metrics = self.builder.get_network_metrics(two_network)
        assert metrics.node_count == 2
        assert metrics.edge_count >= 1  # Should have at least one edge
    
    def test_get_network_status(self):
        """Test network status reporting"""
        network = self.builder.build_network(self.agents, network_id="test_net")
        
        status = self.builder.get_network_status("test_net")
        
        assert isinstance(status, dict)
        assert status["network_id"] == "test_net"
        assert "metrics" in status
        assert "agent_status" in status
        assert "health_score" in status
        
        # Agent status should have correct counts
        agent_status = status["agent_status"]
        assert agent_status["total_agents"] == len(self.agents)
        assert agent_status["active_agents"] <= len(self.agents)
        assert "average_load" in agent_status
        assert "total_errors" in agent_status
        
        # Health score should be reasonable
        assert 0 <= status["health_score"] <= 1
    
    def test_get_network_status_invalid(self):
        """Test network status with invalid network ID"""
        with pytest.raises(ValueError, match="not found"):
            self.builder.get_network_status("nonexistent")


class TestConnectionWeight:
    """Test ConnectionWeight functionality"""
    
    def test_connection_weight_creation(self):
        """Test ConnectionWeight creation and effective weight calculation"""
        connection = ConnectionWeight(
            strength=0.8,
            latency=0.1,
            bandwidth=0.9,
            compatibility=0.7
        )
        
        assert connection.strength == 0.8
        assert connection.latency == 0.1
        assert connection.bandwidth == 0.9
        assert connection.compatibility == 0.7
        
        # Test effective weight calculation
        effective = connection.effective_weight()
        expected = 0.8 * 0.7 / (1.0 + 0.1)
        assert abs(effective - expected) < 1e-10
    
    def test_connection_weight_edge_cases(self):
        """Test ConnectionWeight with edge case values"""
        # Zero latency
        connection_zero_latency = ConnectionWeight(1.0, 0.0, 1.0, 1.0)
        assert connection_zero_latency.effective_weight() == 1.0
        
        # High latency
        connection_high_latency = ConnectionWeight(1.0, 10.0, 1.0, 1.0)
        assert connection_high_latency.effective_weight() < 0.1


class TestNetworkCompatibility:
    """Test agent compatibility calculations"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.builder = NetworkBuilder(random_seed=42)
        self.factory = AgentFactory()
    
    def test_same_type_compatibility(self):
        """Test compatibility between agents of same type"""
        agent1 = self.factory.create_mathematical_agent({"tau": 1.0})
        agent2 = self.factory.create_mathematical_agent({"tau": 1.1})
        
        compatibility = self.builder._compute_agent_compatibility(agent1, agent2)
        
        # Same type agents should have good compatibility
        assert compatibility > 0.5
    
    def test_different_type_compatibility(self):
        """Test compatibility between agents of different types"""
        math_agent = self.factory.create_mathematical_agent({"tau": 1.0})
        science_agent = self.factory.create_scientific_agent({"tau": 1.0})
        quantum_agent = self.factory.create_quantum_agent({"tau": 1.0})
        
        math_science_compat = self.builder._compute_agent_compatibility(math_agent, science_agent)
        science_quantum_compat = self.builder._compute_agent_compatibility(science_agent, quantum_agent)
        math_quantum_compat = self.builder._compute_agent_compatibility(math_agent, quantum_agent)
        
        # All should be positive
        assert math_science_compat > 0
        assert science_quantum_compat > 0
        assert math_quantum_compat > 0
        
        # Mathematical-Scientific should have higher compatibility than others
        assert math_science_compat >= math_quantum_compat
    
    def test_intelligence_level_compatibility(self):
        """Test compatibility based on intelligence levels"""
        agent_low = self.factory.create_mathematical_agent({"tau": 0.5})
        agent_high = self.factory.create_mathematical_agent({"tau": 3.0})
        agent_medium = self.factory.create_mathematical_agent({"tau": 1.5})
        
        low_high_compat = self.builder._compute_agent_compatibility(agent_low, agent_high)
        low_medium_compat = self.builder._compute_agent_compatibility(agent_low, agent_medium)
        
        # Closer intelligence levels should have better compatibility
        assert low_medium_compat > low_high_compat
    
    def test_coherence_compatibility(self):
        """Test compatibility based on coherence levels"""
        agent_low_coh = self.factory.create_mathematical_agent({"coherence": 0.3})
        agent_high_coh = self.factory.create_mathematical_agent({"coherence": 0.9})
        agent_medium_coh = self.factory.create_mathematical_agent({"coherence": 0.6})
        
        low_high_compat = self.builder._compute_agent_compatibility(agent_low_coh, agent_high_coh)
        low_medium_compat = self.builder._compute_agent_compatibility(agent_low_coh, agent_medium_coh)
        
        # Closer coherence levels should have better compatibility
        assert low_medium_compat > low_high_compat


class TestNetworkOptimization:
    """Test network optimization algorithms"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.builder = NetworkBuilder(random_seed=42)
        self.factory = AgentFactory()
        
        # Create more agents for optimization testing
        self.agents = []
        for i in range(8):
            agent = self.factory.create_mathematical_agent({
                "tau": 1.0 + i * 0.1,
                "coherence": 0.6 + i * 0.04,
            })
            self.agents.append(agent)
    
    def test_optimization_preserves_connectivity(self):
        """Test that optimization preserves network connectivity"""
        network = self.builder.build_network(self.agents, target_density=0.4)
        
        # Ensure it's connected before optimization
        assert nx.is_connected(network)
        
        # Optimize
        self.builder.optimize_information_flow(network)
        
        # Should still be connected
        assert nx.is_connected(network)
        assert network.number_of_nodes() == len(self.agents)
    
    def test_optimization_improves_efficiency(self):
        """Test that optimization tends to improve network efficiency"""
        network = self.builder.build_network(self.agents, target_density=0.3)
        
        initial_efficiency = nx.global_efficiency(network)
        
        # Optimize
        self.builder.optimize_information_flow(network)
        
        final_efficiency = nx.global_efficiency(network)
        
        # Efficiency should not decrease significantly
        assert final_efficiency >= initial_efficiency * 0.9
    
    def test_optimization_with_small_network(self):
        """Test optimization with very small networks"""
        small_agents = self.agents[:2]
        network = self.builder.build_network(small_agents)
        
        # Should not crash with small networks
        self.builder.optimize_information_flow(network)
        
        assert nx.is_connected(network)
        assert network.number_of_nodes() == 2
    
    def test_edge_benefit_calculation(self):
        """Test edge benefit calculation"""
        network = self.builder.build_network(self.agents[:4], target_density=0.3)
        
        agent1 = self.agents[0]
        agent2 = self.agents[1]
        
        benefit = self.builder._calculate_edge_benefit(network, agent1, agent2)
        
        # Benefit should be a reasonable number
        assert isinstance(benefit, float)
        assert benefit >= 0  # Should be non-negative
    
    def test_load_balancing(self):
        """Test network load balancing functionality"""
        # Set different processing loads
        for i, agent in enumerate(self.agents):
            agent.state.processing_load = i * 0.2
        
        network = self.builder.build_network(self.agents, target_density=0.4)
        
        # Optimize (includes load balancing)
        self.builder.optimize_information_flow(network)
        
        # Network should still be valid
        assert nx.is_connected(network)
        assert network.number_of_nodes() == len(self.agents)