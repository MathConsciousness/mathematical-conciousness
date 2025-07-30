"""
NetworkBuilder: Distributed computing infrastructure for building agent networks.
Targets 96.4% network density for optimal computational efficiency.
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio

# Import the ComputationalAgent from core module
from core.agent_factory import ComputationalAgent


@dataclass
class NetworkConnection:
    """Represents a connection between two agents in the network."""
    source_id: str
    target_id: str
    weight: float
    connection_type: str = "computational"
    latency: float = 0.0
    
    def __post_init__(self):
        """Validate connection parameters."""
        if self.weight < 0 or self.weight > 1:
            raise ValueError(f"Connection weight must be in [0, 1], got {self.weight}")


class AgentNetwork:
    """
    Represents the computational network of agents with graph topology.
    """
    
    def __init__(self, agents: List[ComputationalAgent]):
        """
        Initialize the network with a list of agents.
        
        Args:
            agents: List of ComputationalAgent instances
        """
        self.agents = agents
        self.graph = nx.Graph()
        self.connections: List[NetworkConnection] = []
        
        # Add agents as nodes
        for agent in agents:
            self.graph.add_node(agent.agent_id, agent=agent)
    
    def add_connection(self, connection: NetworkConnection) -> None:
        """
        Add a connection between two agents.
        
        Args:
            connection: NetworkConnection instance
        """
        self.connections.append(connection)
        self.graph.add_edge(
            connection.source_id,
            connection.target_id,
            weight=connection.weight,
            connection_type=connection.connection_type,
            latency=connection.latency
        )
    
    def density(self) -> float:
        """
        Calculate the network density.
        
        Returns:
            Network density as a float between 0 and 1
        """
        return nx.density(self.graph)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive network metrics.
        
        Returns:
            Dictionary containing network analysis metrics
        """
        if len(self.graph.nodes()) == 0:
            return {"error": "Empty network"}
        
        metrics = {
            "node_count": len(self.graph.nodes()),
            "edge_count": len(self.graph.edges()),
            "density": self.density(),
            "average_clustering": nx.average_clustering(self.graph) if len(self.graph.nodes()) > 2 else 0,
            "is_connected": nx.is_connected(self.graph)
        }
        
        # Add centrality measures for small networks
        if len(self.graph.nodes()) <= 100:
            try:
                centrality = nx.degree_centrality(self.graph)
                metrics["max_centrality"] = max(centrality.values()) if centrality else 0
                metrics["avg_centrality"] = np.mean(list(centrality.values())) if centrality else 0
            except:
                metrics["max_centrality"] = 0
                metrics["avg_centrality"] = 0
        
        return metrics
    
    def get_agent_neighbors(self, agent_id: str) -> List[str]:
        """
        Get the neighbors of a specific agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            List of neighbor agent IDs
        """
        if agent_id in self.graph:
            return list(self.graph.neighbors(agent_id))
        return []
    
    def __repr__(self) -> str:
        """String representation of the network."""
        return f"AgentNetwork(nodes={len(self.graph.nodes())}, edges={len(self.graph.edges())}, density={self.density():.3f})"


class NetworkBuilder:
    """
    Builder class for constructing distributed computing networks.
    Optimizes for 96.4% network density target.
    """
    
    def __init__(self):
        """Initialize the NetworkBuilder."""
        self.target_density = 0.964
        self.connection_strategies = {
            "distance_based": self._distance_based_connections,
            "capability_based": self._capability_based_connections,
            "random": self._random_connections,
            "hybrid": self._hybrid_connections
        }
    
    async def build_network(self, agents: List[ComputationalAgent], 
                          target_density: float = 0.964,
                          strategy: str = "hybrid") -> AgentNetwork:
        """
        Build a network with the specified target density.
        
        Args:
            agents: List of ComputationalAgent instances
            target_density: Target network density (default: 0.964)
            strategy: Connection strategy to use
            
        Returns:
            AgentNetwork instance with desired density
        """
        if len(agents) < 2:
            raise ValueError("Need at least 2 agents to build a network")
        
        print(f"Building network with {len(agents)} agents, target density: {target_density:.3f}")
        
        # Create initial network
        network = AgentNetwork(agents)
        
        # Calculate required number of edges for target density
        n_nodes = len(agents)
        max_edges = n_nodes * (n_nodes - 1) // 2
        target_edges = int(target_density * max_edges)
        
        print(f"Target edges: {target_edges} out of {max_edges} possible")
        
        # Generate connections using the specified strategy
        strategy_func = self.connection_strategies.get(strategy, self._hybrid_connections)
        connections = await strategy_func(agents, target_edges)
        
        # Add connections to network
        for connection in connections:
            network.add_connection(connection)
        
        # Verify density
        actual_density = network.density()
        print(f"Achieved density: {actual_density:.3f} (target: {target_density:.3f})")
        
        # Fine-tune if needed
        if abs(actual_density - target_density) > 0.01:
            print("Fine-tuning network density...")
            network = await self._fine_tune_density(network, target_density)
        
        return network
    
    async def _distance_based_connections(self, agents: List[ComputationalAgent], 
                                        target_edges: int) -> List[NetworkConnection]:
        """Create connections based on spatial distance between agents."""
        connections = []
        
        # Calculate distances between all agent pairs
        agent_pairs = []
        for i, agent_a in enumerate(agents):
            for j, agent_b in enumerate(agents[i+1:], i+1):
                distance = np.linalg.norm(agent_a.position - agent_b.position)
                agent_pairs.append((agent_a, agent_b, distance))
        
        # Sort by distance (closest first)
        agent_pairs.sort(key=lambda x: x[2])
        
        # Create connections starting with closest pairs
        for i, (agent_a, agent_b, distance) in enumerate(agent_pairs[:target_edges]):
            weight = max(0.1, 1.0 - distance / 2.0)  # Distance-based weight
            connection = NetworkConnection(
                source_id=agent_a.agent_id,
                target_id=agent_b.agent_id,
                weight=weight,
                connection_type="spatial",
                latency=distance * 0.1
            )
            connections.append(connection)
        
        return connections
    
    async def _capability_based_connections(self, agents: List[ComputationalAgent],
                                          target_edges: int) -> List[NetworkConnection]:
        """Create connections based on agent capability compatibility."""
        connections = []
        
        # Calculate capability similarity between agents
        agent_pairs = []
        for i, agent_a in enumerate(agents):
            for j, agent_b in enumerate(agents[i+1:], i+1):
                # Compute capability similarity
                cap_a = np.array(list(agent_a.capabilities.values()))
                cap_b = np.array(list(agent_b.capabilities.values()))
                similarity = np.dot(cap_a, cap_b) / (np.linalg.norm(cap_a) * np.linalg.norm(cap_b))
                agent_pairs.append((agent_a, agent_b, similarity))
        
        # Sort by similarity (highest first)
        agent_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Create connections for most compatible agents
        for i, (agent_a, agent_b, similarity) in enumerate(agent_pairs[:target_edges]):
            connection = NetworkConnection(
                source_id=agent_a.agent_id,
                target_id=agent_b.agent_id,
                weight=max(0.1, similarity),
                connection_type="capability",
                latency=0.05
            )
            connections.append(connection)
        
        return connections
    
    async def _random_connections(self, agents: List[ComputationalAgent],
                                target_edges: int) -> List[NetworkConnection]:
        """Create random connections between agents."""
        connections = []
        
        # Generate all possible agent pairs
        agent_pairs = []
        for i, agent_a in enumerate(agents):
            for j, agent_b in enumerate(agents[i+1:], i+1):
                agent_pairs.append((agent_a, agent_b))
        
        # Randomly select pairs for connections
        selected_pairs = np.random.choice(
            len(agent_pairs), 
            size=min(target_edges, len(agent_pairs)), 
            replace=False
        )
        
        for idx in selected_pairs:
            agent_a, agent_b = agent_pairs[idx]
            weight = np.random.uniform(0.1, 1.0)
            connection = NetworkConnection(
                source_id=agent_a.agent_id,
                target_id=agent_b.agent_id,
                weight=weight,
                connection_type="random",
                latency=np.random.uniform(0.01, 0.1)
            )
            connections.append(connection)
        
        return connections
    
    async def _hybrid_connections(self, agents: List[ComputationalAgent],
                                target_edges: int) -> List[NetworkConnection]:
        """Create connections using a hybrid approach combining multiple strategies."""
        connections = []
        
        # Allocate edges to different strategies
        distance_edges = target_edges // 3
        capability_edges = target_edges // 3
        random_edges = target_edges - distance_edges - capability_edges
        
        # Generate connections using each strategy
        distance_conns = await self._distance_based_connections(agents, distance_edges)
        capability_conns = await self._capability_based_connections(agents, capability_edges)
        random_conns = await self._random_connections(agents, random_edges)
        
        # Combine all connections
        connections.extend(distance_conns)
        connections.extend(capability_conns)
        connections.extend(random_conns)
        
        # Remove duplicates (same agent pair)
        unique_connections = {}
        for conn in connections:
            pair_key = tuple(sorted([conn.source_id, conn.target_id]))
            if pair_key not in unique_connections:
                unique_connections[pair_key] = conn
        
        return list(unique_connections.values())
    
    async def _fine_tune_density(self, network: AgentNetwork, 
                               target_density: float) -> AgentNetwork:
        """Fine-tune network density by adding or removing connections."""
        current_density = network.density()
        n_nodes = len(network.graph.nodes())
        max_edges = n_nodes * (n_nodes - 1) // 2
        target_edges = int(target_density * max_edges)
        current_edges = len(network.graph.edges())
        
        if current_edges < target_edges:
            # Add more connections
            edges_to_add = target_edges - current_edges
            print(f"Adding {edges_to_add} connections...")
            
            # Find unconnected pairs
            all_pairs = [(a, b) for a in network.graph.nodes() for b in network.graph.nodes() if a < b]
            unconnected_pairs = [pair for pair in all_pairs if not network.graph.has_edge(pair[0], pair[1])]
            
            # Add random connections from unconnected pairs
            if unconnected_pairs:
                pairs_to_connect = np.random.choice(
                    len(unconnected_pairs),
                    size=min(edges_to_add, len(unconnected_pairs)),
                    replace=False
                )
                
                for idx in pairs_to_connect:
                    agent_a_id, agent_b_id = unconnected_pairs[idx]
                    connection = NetworkConnection(
                        source_id=agent_a_id,
                        target_id=agent_b_id,
                        weight=np.random.uniform(0.1, 1.0),
                        connection_type="fine_tune"
                    )
                    network.add_connection(connection)
        
        elif current_edges > target_edges:
            # Remove some connections
            edges_to_remove = current_edges - target_edges
            print(f"Removing {edges_to_remove} connections...")
            
            # Remove random edges
            edges_list = list(network.graph.edges())
            if edges_list:
                edges_to_remove_indices = np.random.choice(
                    len(edges_list),
                    size=min(edges_to_remove, len(edges_list)),
                    replace=False
                )
                
                for idx in edges_to_remove_indices:
                    edge = edges_list[idx]
                    network.graph.remove_edge(edge[0], edge[1])
                
                # Update connections list
                network.connections = [
                    conn for conn in network.connections
                    if network.graph.has_edge(conn.source_id, conn.target_id)
                ]
        
        return network