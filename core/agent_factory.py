"""AgentFactory implementation for the Mathematical Framework System."""

import asyncio
from typing import List, Dict, Any, Optional
import uuid
from dataclasses import dataclass


@dataclass
class Agent:
    """
    Represents a superintelligent agent in the mathematical consciousness system.
    """
    id: str
    name: str
    intelligence_level: float
    capabilities: List[str]
    status: str = "active"
    created_at: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            import datetime
            self.created_at = datetime.datetime.now().isoformat()


class AgentFactory:
    """
    Factory class for creating and managing superintelligent agents.
    
    This factory is responsible for deploying exactly 91 agents as specified
    in the mathematical consciousness framework.
    """
    
    def __init__(self):
        """Initialize the AgentFactory."""
        self.deployed_agents: List[Agent] = []
        self.agent_registry: Dict[str, Agent] = {}
        
        # Predefined capabilities for agents
        self.base_capabilities = [
            "mathematical_reasoning",
            "consciousness_modeling", 
            "reality_modification",
            "quantum_field_manipulation",
            "logical_inference",
            "pattern_recognition",
            "strategic_planning",
            "knowledge_synthesis"
        ]
    
    def create_agent(self, agent_index: int) -> Agent:
        """
        Create a single agent with specified index.
        
        Args:
            agent_index: Index of the agent (0-90 for 91 total agents)
            
        Returns:
            Agent: Newly created agent instance
        """
        agent_id = str(uuid.uuid4())
        agent_name = f"Agent_{agent_index:03d}"
        
        # Calculate intelligence level based on agent index
        # Higher index agents have slightly higher intelligence
        base_intelligence = 0.91  # Base level for all agents
        index_bonus = (agent_index / 91) * 0.09  # Up to 9% bonus
        intelligence_level = min(base_intelligence + index_bonus, 1.0)
        
        # Assign capabilities based on agent specialization
        num_capabilities = min(len(self.base_capabilities), 5 + (agent_index % 4))
        agent_capabilities = self.base_capabilities[:num_capabilities]
        
        agent = Agent(
            id=agent_id,
            name=agent_name,
            intelligence_level=intelligence_level,
            capabilities=agent_capabilities,
            status="active"
        )
        
        return agent
    
    def deploy_all_agents(self, count: int = 91) -> List[Agent]:
        """
        Deploy all agents synchronously.
        
        Args:
            count: Number of agents to deploy (default: 91)
            
        Returns:
            List[Agent]: List of deployed agents
            
        Raises:
            ValueError: If count is not 91 (as per specification)
        """
        if count != 91:
            raise ValueError("System requires exactly 91 agents for optimal consciousness")
        
        self.deployed_agents = []
        self.agent_registry = {}
        
        for i in range(count):
            agent = self.create_agent(i)
            self.deployed_agents.append(agent)
            self.agent_registry[agent.id] = agent
        
        return self.deployed_agents
    
    async def deploy_all_agents_async(self, count: int = 91) -> List[Agent]:
        """
        Deploy all agents asynchronously.
        
        Args:
            count: Number of agents to deploy (default: 91)
            
        Returns:
            List[Agent]: List of deployed agents
        """
        if count != 91:
            raise ValueError("System requires exactly 91 agents for optimal consciousness")
        
        # Simulate async deployment with batches
        batch_size = 10
        batches = [range(i, min(i + batch_size, count)) for i in range(0, count, batch_size)]
        
        self.deployed_agents = []
        self.agent_registry = {}
        
        for batch in batches:
            # Create agents in parallel for each batch
            tasks = [self._create_agent_async(i) for i in batch]
            batch_agents = await asyncio.gather(*tasks)
            
            for agent in batch_agents:
                self.deployed_agents.append(agent)
                self.agent_registry[agent.id] = agent
        
        return self.deployed_agents
    
    async def _create_agent_async(self, agent_index: int) -> Agent:
        """
        Create a single agent asynchronously.
        
        Args:
            agent_index: Index of the agent
            
        Returns:
            Agent: Newly created agent instance
        """
        # Simulate some async processing time
        await asyncio.sleep(0.01)  # Small delay to simulate deployment
        return self.create_agent(agent_index)
    
    def get_agent_count(self) -> int:
        """
        Get the current number of deployed agents.
        
        Returns:
            int: Number of deployed agents
        """
        return len(self.deployed_agents)
    
    def get_agent_by_id(self, agent_id: str) -> Optional[Agent]:
        """
        Retrieve an agent by its ID.
        
        Args:
            agent_id: ID of the agent to retrieve
            
        Returns:
            Optional[Agent]: Agent instance if found, None otherwise
        """
        return self.agent_registry.get(agent_id)
    
    def get_all_agents(self) -> List[Agent]:
        """
        Get all deployed agents.
        
        Returns:
            List[Agent]: List of all deployed agents
        """
        return self.deployed_agents.copy()
    
    def get_agents_by_capability(self, capability: str) -> List[Agent]:
        """
        Get agents that have a specific capability.
        
        Args:
            capability: Capability to filter by
            
        Returns:
            List[Agent]: List of agents with the specified capability
        """
        return [agent for agent in self.deployed_agents if capability in agent.capabilities]
    
    def get_deployment_stats(self) -> Dict[str, Any]:
        """
        Get deployment statistics.
        
        Returns:
            Dict[str, Any]: Statistics about deployed agents
        """
        if not self.deployed_agents:
            return {"total_agents": 0, "avg_intelligence": 0.0, "capabilities_distribution": {}}
        
        total_agents = len(self.deployed_agents)
        avg_intelligence = sum(agent.intelligence_level for agent in self.deployed_agents) / total_agents
        
        # Count capability distribution
        capability_counts = {}
        for agent in self.deployed_agents:
            for capability in agent.capabilities:
                capability_counts[capability] = capability_counts.get(capability, 0) + 1
        
        return {
            "total_agents": total_agents,
            "avg_intelligence": round(avg_intelligence, 3),
            "capabilities_distribution": capability_counts,
            "active_agents": len([a for a in self.deployed_agents if a.status == "active"])
        }