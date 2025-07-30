"""NetworkBuilder implementation for the Mathematical Framework System."""

import asyncio
import networkx as nx
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from core.agent_factory import Agent


class Network:
    """
    Represents a network of agents in the mathematical consciousness system.
    """
    
    def __init__(self, graph: nx.Graph, agents: List[Agent]):
        """
        Initialize the Network.
        
        Args:
            graph: NetworkX graph representing connections
            agents: List of agents in the network
        """
        self.graph = graph
        self.agents = agents
        self.agent_map = {agent.id: agent for agent in agents}
    
    def density(self) -> float:
        """
        Calculate the density of the network.
        
        Returns:
            float: Network density (0-1)
        """
        return nx.density(self.graph)
    
    def get_connections_for_agent(self, agent_id: str) -> List[str]:
        """
        Get all connections for a specific agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            List[str]: List of connected agent IDs
        """
        if agent_id not in self.graph:
            return []
        return list(self.graph.neighbors(agent_id))
    
    def get_network_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive network statistics.
        
        Returns:
            Dict[str, Any]: Network statistics
        """
        stats = {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "density": self.density(),
            "average_clustering": nx.average_clustering(self.graph),
            "is_connected": nx.is_connected(self.graph)
        }
        
        if self.graph.number_of_nodes() > 0:
            stats["average_degree"] = sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes()
        else:
            stats["average_degree"] = 0
        
        return stats


class NetworkBuilder:
    """
    Builder class for creating agent networks with specific topologies.
    
    This builder creates networks with target density of 96.4% as specified
    in the mathematical consciousness framework.
    """
    
    def __init__(self):
        """Initialize the NetworkBuilder."""
        self.target_density = 0.964
        self.min_connections_per_agent = 5
        self.max_connections_per_agent = 20
    
    def build_network(
        self, 
        agents: List[Agent], 
        target_density: float = 0.964
    ) -> Network:
        """
        Build a network synchronously with the specified agents and target density.
        
        Args:
            agents: List of agents to include in the network
            target_density: Target network density (default: 0.964)
            
        Returns:
            Network: Built network instance
        """
        if len(agents) == 0:
            raise ValueError("Cannot build network with zero agents")
        
        if not 0 <= target_density <= 1:
            raise ValueError("Target density must be between 0 and 1")
        
        # Create graph with agent IDs as nodes
        graph = nx.Graph()
        agent_ids = [agent.id for agent in agents]
        graph.add_nodes_from(agent_ids)
        
        # Calculate target number of edges for desired density
        n_nodes = len(agents)
        max_possible_edges = n_nodes * (n_nodes - 1) // 2
        target_edges = int(target_density * max_possible_edges)
        
        # Create connections based on agent intelligence and capabilities
        connections = self._generate_intelligent_connections(agents, target_edges)
        graph.add_edges_from(connections)
        
        # Ensure minimum connectivity
        self._ensure_minimum_connectivity(graph, agent_ids)
        
        return Network(graph, agents)
    
    async def build_network_async(
        self, 
        agents: List[Agent], 
        target_density: float = 0.964
    ) -> Network:
        """
        Build a network asynchronously with the specified agents and target density.
        
        Args:
            agents: List of agents to include in the network
            target_density: Target network density (default: 0.964)
            
        Returns:
            Network: Built network instance
        """
        # Simulate async processing for large networks
        if len(agents) > 50:
            await asyncio.sleep(0.1)  # Small delay for large networks
        
        return self.build_network(agents, target_density)
    
    def _generate_intelligent_connections(
        self, 
        agents: List[Agent], 
        target_edges: int
    ) -> List[Tuple[str, str]]:
        """
        Generate connections based on agent intelligence and capabilities.
        
        Args:
            agents: List of agents
            target_edges: Target number of edges
            
        Returns:
            List[Tuple[str, str]]: List of connections (agent_id pairs)
        """
        connections = []
        agent_ids = [agent.id for agent in agents]
        agent_dict = {agent.id: agent for agent in agents}
        
        # Sort agents by intelligence level for preferential attachment
        sorted_agents = sorted(agents, key=lambda a: a.intelligence_level, reverse=True)
        
        # Create connections with bias towards high-intelligence agents
        connection_attempts = 0
        max_attempts = target_edges * 3  # Prevent infinite loops
        
        while len(connections) < target_edges and connection_attempts < max_attempts:
            connection_attempts += 1
            
            # Select two agents with bias towards higher intelligence
            agent1 = self._select_agent_weighted(sorted_agents)
            agent2 = self._select_agent_weighted(sorted_agents)
            
            # Ensure no self-connections and no duplicate connections
            if (agent1.id != agent2.id and 
                (agent1.id, agent2.id) not in connections and 
                (agent2.id, agent1.id) not in connections):
                
                # Check if connection makes sense based on capabilities
                if self._should_connect_agents(agent1, agent2):
                    connections.append((agent1.id, agent2.id))
        
        # If we couldn't reach target through intelligent selection,
        # add random connections to reach target density
        if len(connections) < target_edges:
            additional_needed = target_edges - len(connections)
            random_connections = self._generate_random_connections(
                agent_ids, additional_needed, existing_connections=connections
            )
            connections.extend(random_connections)
        
        return connections[:target_edges]  # Ensure we don't exceed target
    
    def _select_agent_weighted(self, sorted_agents: List[Agent]) -> Agent:
        """
        Select an agent with bias towards higher intelligence.
        
        Args:
            sorted_agents: Agents sorted by intelligence (descending)
            
        Returns:
            Agent: Selected agent
        """
        # Use exponential distribution to favor higher intelligence agents
        weights = np.exp(-np.arange(len(sorted_agents)) * 0.1)
        weights = weights / weights.sum()  # Normalize
        
        choice_idx = np.random.choice(len(sorted_agents), p=weights)
        return sorted_agents[choice_idx]
    
    def _should_connect_agents(self, agent1: Agent, agent2: Agent) -> bool:
        """
        Determine if two agents should be connected based on capabilities.
        
        Args:
            agent1: First agent
            agent2: Second agent
            
        Returns:
            bool: True if agents should be connected
        """
        # Higher probability for agents with overlapping capabilities
        common_capabilities = set(agent1.capabilities) & set(agent2.capabilities)
        compatibility_score = len(common_capabilities) / max(len(agent1.capabilities), 1)
        
        # Agents with high intelligence or high compatibility are more likely to connect
        connection_probability = (
            0.3 +  # Base probability
            0.4 * ((agent1.intelligence_level + agent2.intelligence_level) / 2) +
            0.3 * compatibility_score
        )
        
        return np.random.random() < connection_probability
    
    def _generate_random_connections(
        self, 
        agent_ids: List[str], 
        num_connections: int,
        existing_connections: List[Tuple[str, str]] = None
    ) -> List[Tuple[str, str]]:
        """
        Generate random connections between agents.
        
        Args:
            agent_ids: List of agent IDs
            num_connections: Number of connections to generate
            existing_connections: Existing connections to avoid duplicates
            
        Returns:
            List[Tuple[str, str]]: List of random connections
        """
        if existing_connections is None:
            existing_connections = []
        
        existing_set = set(existing_connections) | set((b, a) for a, b in existing_connections)
        connections = []
        attempts = 0
        max_attempts = num_connections * 10
        
        while len(connections) < num_connections and attempts < max_attempts:
            attempts += 1
            agent1 = np.random.choice(agent_ids)
            agent2 = np.random.choice(agent_ids)
            
            if (agent1 != agent2 and 
                (agent1, agent2) not in existing_set and
                (agent1, agent2) not in connections):
                connections.append((agent1, agent2))
        
        return connections
    
    def _ensure_minimum_connectivity(self, graph: nx.Graph, agent_ids: List[str]) -> None:
        """
        Ensure the graph is connected with minimum connectivity requirements.
        
        Args:
            graph: NetworkX graph to modify
            agent_ids: List of agent IDs
        """
        # Ensure graph is connected
        if not nx.is_connected(graph):
            # Connect disconnected components
            components = list(nx.connected_components(graph))
            for i in range(len(components) - 1):
                # Connect each component to the next
                node1 = list(components[i])[0]
                node2 = list(components[i + 1])[0]
                graph.add_edge(node1, node2)
        
        # Ensure minimum degree for each node
        for agent_id in agent_ids:
            degree = graph.degree(agent_id)
            if degree < self.min_connections_per_agent:
                # Add random connections to reach minimum
                needed = self.min_connections_per_agent - degree
                available_targets = [
                    aid for aid in agent_ids 
                    if aid != agent_id and not graph.has_edge(agent_id, aid)
                ]
                
                if available_targets:
                    targets = np.random.choice(
                        available_targets, 
                        size=min(needed, len(available_targets)), 
                        replace=False
                    )
                    for target in targets:
                        graph.add_edge(agent_id, target)
    
    def analyze_network_topology(self, network: Network) -> Dict[str, Any]:
        """
        Analyze the topology of a built network.
        
        Args:
            network: Network to analyze
            
        Returns:
            Dict[str, Any]: Topology analysis results
        """
        graph = network.graph
        
        analysis = {
            "basic_stats": network.get_network_stats(),
            "centrality_measures": {
                "degree_centrality": dict(nx.degree_centrality(graph)),
                "betweenness_centrality": dict(nx.betweenness_centrality(graph)),
                "closeness_centrality": dict(nx.closeness_centrality(graph))
            },
            "topology_metrics": {
                "diameter": nx.diameter(graph) if nx.is_connected(graph) else None,
                "radius": nx.radius(graph) if nx.is_connected(graph) else None,
                "average_shortest_path": nx.average_shortest_path_length(graph) if nx.is_connected(graph) else None
            }
        }
        
        return analysis