"""Network Architecture for Mathematical Consciousness Framework

This module provides network construction and optimization capabilities
for connecting computational agents in an efficient topology.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
from numpy.typing import NDArray
import networkx as nx
from dataclasses import dataclass
import random
from collections import defaultdict

from ..core.agent_factory import Agent, AgentType


@dataclass
class NetworkMetrics:
    """Network topology and performance metrics"""
    node_count: int
    edge_count: int
    density: float
    average_clustering: float
    average_path_length: float
    efficiency: float
    modularity: float
    connectivity_score: float


@dataclass
class ConnectionWeight:
    """Represents the weight and properties of a connection between agents"""
    strength: float
    latency: float
    bandwidth: float
    compatibility: float
    
    def effective_weight(self) -> float:
        """Calculate effective connection weight"""
        return self.strength * self.compatibility / (1.0 + self.latency)


class NetworkBuilder:
    """Builder for agent network architectures
    
    Constructs optimized networks of computational agents with target
    connectivity patterns and information flow characteristics.
    """

    def __init__(self, random_seed: Optional[int] = None) -> None:
        """Initialize network builder
        
        Args:
            random_seed: Optional seed for reproducible network generation
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            random.seed(random_seed)
        
        self.networks: Dict[str, nx.Graph] = {}
        self.agent_networks: Dict[str, List[Agent]] = {}

    def build_network(
        self, 
        agents: List[Agent], 
        target_density: float = 0.964,
        network_id: Optional[str] = None
    ) -> nx.Graph:
        """Build agent network with target connectivity
        
        Args:
            agents: List of agents to connect
            target_density: Target network density (0-1)
            network_id: Optional network identifier
            
        Returns:
            nx.Graph: Constructed network
            
        Raises:
            ValueError: If invalid parameters provided
        """
        if not agents:
            raise ValueError("Cannot build network with empty agent list")
            
        if not 0 <= target_density <= 1:
            raise ValueError("Target density must be between 0 and 1")
        
        if len(agents) < 2:
            raise ValueError("Need at least 2 agents to build a network")
        
        network_id = network_id or f"network_{len(self.networks)}"
        
        # Create graph with agents as nodes
        G = nx.Graph()
        
        # Add agents as nodes with their properties
        for agent in agents:
            G.add_node(
                agent.id,
                agent_type=agent.agent_type.value,
                intelligence=agent.logic_field.calculate_intelligence_level(),
                coherence=agent.logic_field.compute_coherence(),
                capabilities=agent.capabilities,
            )
        
        # Calculate target number of edges
        n_nodes = len(agents)
        max_edges = n_nodes * (n_nodes - 1) // 2
        target_edges = int(target_density * max_edges)
        
        # Build connections using hybrid strategy
        self._build_connections(G, agents, target_edges)
        
        # Store network
        self.networks[network_id] = G
        self.agent_networks[network_id] = agents.copy()
        
        return G

    def _build_connections(self, G: nx.Graph, agents: List[Agent], target_edges: int) -> None:
        """Build network connections using intelligent strategy"""
        agent_ids = [agent.id for agent in agents]
        n_agents = len(agents)
        
        # Phase 1: Ensure connectivity (minimum spanning tree)
        self._ensure_connectivity(G, agents)
        current_edges = G.number_of_edges()
        
        # Phase 2: Add edges based on compatibility and optimization
        remaining_edges = target_edges - current_edges
        
        if remaining_edges > 0:
            self._add_optimized_edges(G, agents, remaining_edges)

    def _ensure_connectivity(self, G: nx.Graph, agents: List[Agent]) -> None:
        """Ensure all agents are connected using MST approach"""
        agent_ids = [agent.id for agent in agents]
        n_agents = len(agents)
        
        # Create complete graph with weights based on compatibility
        compatibility_matrix = self._compute_compatibility_matrix(agents)
        
        # Use Kruskal's algorithm for MST
        edges_by_weight = []
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                weight = compatibility_matrix[i, j]
                edges_by_weight.append((weight, agent_ids[i], agent_ids[j], i, j))
        
        # Sort by compatibility (higher is better, so negate for min heap behavior)
        edges_by_weight.sort(key=lambda x: -x[0])
        
        # Add edges to ensure connectivity
        connected_components = {agent_id: {agent_id} for agent_id in agent_ids}
        
        for weight, id1, id2, i, j in edges_by_weight:
            comp1 = connected_components[id1]
            comp2 = connected_components[id2]
            
            if comp1 != comp2:
                # Add edge
                connection_weight = self._create_connection_weight(agents[i], agents[j])
                G.add_edge(id1, id2, weight=connection_weight.effective_weight(), connection=connection_weight)
                
                # Merge components
                merged = comp1.union(comp2)
                for node_id in merged:
                    connected_components[node_id] = merged
                
                # If all connected, break
                if len(merged) == n_agents:
                    break

    def _add_optimized_edges(self, G: nx.Graph, agents: List[Agent], target_edges: int) -> None:
        """Add additional edges to optimize network properties"""
        agent_ids = [agent.id for agent in agents]
        n_agents = len(agents)
        
        # Get existing edges
        existing_edges = set(G.edges())
        
        # Calculate all possible edges with their benefits
        edge_candidates = []
        
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                edge = (agent_ids[i], agent_ids[j])
                if edge not in existing_edges and (agent_ids[j], agent_ids[i]) not in existing_edges:
                    benefit = self._calculate_edge_benefit(G, agents[i], agents[j])
                    edge_candidates.append((benefit, edge, i, j))
        
        # Sort by benefit (descending)
        edge_candidates.sort(key=lambda x: -x[0])
        
        # Add top edges
        added_edges = 0
        for benefit, (id1, id2), i, j in edge_candidates:
            if added_edges >= target_edges:
                break
                
            connection_weight = self._create_connection_weight(agents[i], agents[j])
            G.add_edge(id1, id2, weight=connection_weight.effective_weight(), connection=connection_weight)
            added_edges += 1

    def _compute_compatibility_matrix(self, agents: List[Agent]) -> NDArray[np.floating]:
        """Compute compatibility matrix between agents"""
        n_agents = len(agents)
        compatibility = np.zeros((n_agents, n_agents))
        
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                comp = self._compute_agent_compatibility(agents[i], agents[j])
                compatibility[i, j] = comp
                compatibility[j, i] = comp
        
        return compatibility

    def _compute_agent_compatibility(self, agent1: Agent, agent2: Agent) -> float:
        """Compute compatibility score between two agents"""
        # Type compatibility
        type_bonus = 1.0
        if agent1.agent_type == agent2.agent_type:
            type_bonus = 1.2
        elif agent1.agent_type == AgentType.MATHEMATICAL and agent2.agent_type == AgentType.SCIENTIFIC:
            type_bonus = 1.1
        elif agent1.agent_type == AgentType.SCIENTIFIC and agent2.agent_type == AgentType.QUANTUM:
            type_bonus = 1.15
        
        # Capability overlap
        caps1_dict = agent1.capabilities.__dict__
        caps2_dict = agent2.capabilities.__dict__
        
        common_capabilities = sum(
            1 for cap in caps1_dict 
            if caps1_dict[cap] and caps2_dict[cap]
        )
        total_capabilities = sum(
            1 for cap in caps1_dict 
            if caps1_dict[cap] or caps2_dict[cap]
        )
        
        capability_score = common_capabilities / max(1, total_capabilities)
        
        # Intelligence level compatibility
        intel1 = agent1.logic_field.calculate_intelligence_level()
        intel2 = agent2.logic_field.calculate_intelligence_level()
        intel_diff = abs(intel1 - intel2)
        intel_score = np.exp(-intel_diff / 2.0)
        
        # Coherence compatibility
        coh1 = agent1.logic_field.compute_coherence()
        coh2 = agent2.logic_field.compute_coherence()
        coherence_score = 1.0 - abs(coh1 - coh2)
        
        # Combined compatibility
        compatibility = (
            type_bonus * capability_score * intel_score * coherence_score
        )
        
        return float(compatibility)

    def _create_connection_weight(self, agent1: Agent, agent2: Agent) -> ConnectionWeight:
        """Create connection weight between two agents"""
        compatibility = self._compute_agent_compatibility(agent1, agent2)
        
        # Base strength from compatibility
        strength = compatibility
        
        # Latency based on agent processing loads
        avg_load = (agent1.state.processing_load + agent2.state.processing_load) / 2
        latency = 0.01 + avg_load * 0.1
        
        # Bandwidth based on agent capabilities
        caps1 = sum(1 for cap in agent1.capabilities.__dict__.values() if cap)
        caps2 = sum(1 for cap in agent2.capabilities.__dict__.values() if cap)
        bandwidth = (caps1 + caps2) / 12.0  # Normalize by max possible capabilities
        
        return ConnectionWeight(
            strength=strength,
            latency=latency,
            bandwidth=bandwidth,
            compatibility=compatibility,
        )

    def _calculate_edge_benefit(self, G: nx.Graph, agent1: Agent, agent2: Agent) -> float:
        """Calculate benefit of adding edge between two agents"""
        # Compatibility benefit
        compatibility_benefit = self._compute_agent_compatibility(agent1, agent2)
        
        # Structural benefit (reduces path lengths)
        try:
            current_distance = nx.shortest_path_length(G, agent1.id, agent2.id)
            structural_benefit = 1.0 / current_distance
        except nx.NetworkXNoPath:
            structural_benefit = 10.0  # High benefit for connecting disconnected components
        
        # Clustering benefit
        neighbors1 = set(G.neighbors(agent1.id))
        neighbors2 = set(G.neighbors(agent2.id))
        common_neighbors = len(neighbors1.intersection(neighbors2))
        clustering_benefit = common_neighbors / max(1, len(neighbors1.union(neighbors2)))
        
        # Load balancing benefit
        load1 = agent1.state.processing_load
        load2 = agent2.state.processing_load
        load_balance_benefit = 1.0 - abs(load1 - load2)
        
        total_benefit = (
            compatibility_benefit * 2.0 +
            structural_benefit +
            clustering_benefit +
            load_balance_benefit
        )
        
        return total_benefit

    def optimize_information_flow(self, network: nx.Graph) -> None:
        """Optimize network information flow
        
        Args:
            network: Network graph to optimize
        """
        if network.number_of_nodes() < 2:
            return
        
        # Phase 1: Remove weak connections
        self._remove_weak_connections(network)
        
        # Phase 2: Add high-benefit connections
        self._add_flow_optimized_connections(network)
        
        # Phase 3: Balance loads
        self._balance_network_loads(network)

    def _remove_weak_connections(self, G: nx.Graph) -> None:
        """Remove connections that don't contribute to information flow"""
        # Calculate edge betweenness centrality
        edge_betweenness = nx.edge_betweenness_centrality(G, weight='weight')
        
        # Remove edges with very low betweenness (bottom 10%)
        if len(edge_betweenness) > 0:
            threshold = np.percentile(list(edge_betweenness.values()), 10)
            edges_to_remove = [
                edge for edge, centrality in edge_betweenness.items()
                if centrality < threshold
            ]
            
            # Only remove if it doesn't disconnect the graph
            for edge in edges_to_remove:
                G_temp = G.copy()
                G_temp.remove_edge(*edge)
                if nx.is_connected(G_temp):
                    G.remove_edge(*edge)

    def _add_flow_optimized_connections(self, G: nx.Graph) -> None:
        """Add connections that improve information flow"""
        nodes = list(G.nodes())
        
        # Calculate current flow efficiency
        current_efficiency = nx.global_efficiency(G)
        
        # Try adding connections that maximize efficiency improvement
        best_improvement = 0
        best_edge = None
        
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                if not G.has_edge(node1, node2):
                    # Temporarily add edge
                    G.add_edge(node1, node2, weight=1.0)
                    new_efficiency = nx.global_efficiency(G)
                    improvement = new_efficiency - current_efficiency
                    
                    if improvement > best_improvement:
                        best_improvement = improvement
                        best_edge = (node1, node2)
                    
                    # Remove temporary edge
                    G.remove_edge(node1, node2)
        
        # Add the best edge if it provides significant improvement
        if best_edge and best_improvement > 0.01:
            G.add_edge(*best_edge, weight=1.0)

    def _balance_network_loads(self, G: nx.Graph) -> None:
        """Balance processing loads across network"""
        # Calculate node loads (simplified)
        node_loads = {}
        for node in G.nodes():
            # Load based on degree and edge weights
            degree_load = G.degree(node, weight='weight')
            node_loads[node] = degree_load
        
        # Identify overloaded and underloaded nodes
        avg_load = np.mean(list(node_loads.values()))
        overloaded = {n: load for n, load in node_loads.items() if load > avg_load * 1.5}
        underloaded = {n: load for n, load in node_loads.items() if load < avg_load * 0.5}
        
        # Redistribute some connections
        for overloaded_node in overloaded:
            for underloaded_node in underloaded:
                if not G.has_edge(overloaded_node, underloaded_node):
                    # Add connection to balance loads
                    G.add_edge(overloaded_node, underloaded_node, weight=0.5)
                    break

    def get_network_metrics(self, network: nx.Graph) -> NetworkMetrics:
        """Calculate comprehensive network metrics
        
        Args:
            network: Network graph to analyze
            
        Returns:
            NetworkMetrics: Computed metrics
        """
        n_nodes = network.number_of_nodes()
        n_edges = network.number_of_edges()
        
        if n_nodes < 2:
            return NetworkMetrics(
                node_count=n_nodes,
                edge_count=n_edges,
                density=0.0,
                average_clustering=0.0,
                average_path_length=0.0,
                efficiency=0.0,
                modularity=0.0,
                connectivity_score=0.0,
            )
        
        # Basic metrics
        density = nx.density(network)
        
        # Clustering
        clustering = nx.average_clustering(network, weight='weight')
        
        # Path length (only for connected graphs)
        if nx.is_connected(network):
            avg_path_length = nx.average_shortest_path_length(network, weight='weight')
        else:
            # For disconnected graphs, calculate average within components
            components = list(nx.connected_components(network))
            if components:
                path_lengths = []
                for component in components:
                    if len(component) > 1:
                        subgraph = network.subgraph(component)
                        path_lengths.append(nx.average_shortest_path_length(subgraph, weight='weight'))
                avg_path_length = np.mean(path_lengths) if path_lengths else 0.0
            else:
                avg_path_length = 0.0
        
        # Efficiency
        efficiency = nx.global_efficiency(network)
        
        # Modularity (using simple community detection)
        try:
            communities = nx.algorithms.community.greedy_modularity_communities(network)
            modularity = nx.algorithms.community.modularity(network, communities, weight='weight')
        except:
            modularity = 0.0
        
        # Connectivity score (custom metric)
        connectivity_score = self._calculate_connectivity_score(network)
        
        return NetworkMetrics(
            node_count=n_nodes,
            edge_count=n_edges,
            density=density,
            average_clustering=clustering,
            average_path_length=avg_path_length,
            efficiency=efficiency,
            modularity=modularity,
            connectivity_score=connectivity_score,
        )

    def _calculate_connectivity_score(self, network: nx.Graph) -> float:
        """Calculate custom connectivity score"""
        if network.number_of_nodes() < 2:
            return 0.0
        
        # Combine multiple connectivity measures
        density = nx.density(network)
        
        # Robustness (how many nodes can be removed before disconnection)
        robustness = 0.0
        if nx.is_connected(network):
            from itertools import combinations
            nodes = list(network.nodes())
            max_removable = 0
            for k in range(1, min(len(nodes), 3)):  # Limit to avoid exponential complexity
                for node_subset in combinations(nodes, k):
                    temp_graph = network.copy()
                    temp_graph.remove_nodes_from(node_subset)
                    if nx.is_connected(temp_graph):
                        max_removable = k
                    else:
                        break
                if max_removable < k:
                    break
            robustness = max_removable / len(nodes)
        
        # Efficiency
        efficiency = nx.global_efficiency(network)
        
        # Combined score
        connectivity_score = (density + robustness + efficiency) / 3.0
        
        return connectivity_score

    def get_network_status(self, network_id: str) -> Dict[str, Any]:
        """Get comprehensive network status
        
        Args:
            network_id: ID of network to analyze
            
        Returns:
            Dict with network status information
        """
        if network_id not in self.networks:
            raise ValueError(f"Network '{network_id}' not found")
        
        network = self.networks[network_id]
        agents = self.agent_networks[network_id]
        metrics = self.get_network_metrics(network)
        
        # Agent status summary
        agent_status = {
            "total_agents": len(agents),
            "active_agents": sum(1 for agent in agents if agent.state.active),
            "average_load": np.mean([agent.state.processing_load for agent in agents]),
            "total_errors": sum(agent.state.error_count for agent in agents),
        }
        
        return {
            "network_id": network_id,
            "metrics": metrics.__dict__,
            "agent_status": agent_status,
            "last_optimization": "Not implemented",
            "health_score": min(1.0, metrics.connectivity_score * metrics.efficiency),
        }