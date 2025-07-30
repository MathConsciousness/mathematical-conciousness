"""
AgentFactory: Agent generation and management system for 91 computational agents.
"""

import asyncio
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import uuid


@dataclass
class ComputationalAgent:
    """
    Represents a single computational agent in the mathematical framework.
    """
    agent_id: str
    position: np.ndarray
    state: np.ndarray
    capabilities: Dict[str, float]
    status: str = "active"
    
    def __post_init__(self):
        """Initialize agent after creation."""
        if self.status not in ["active", "idle", "processing", "disabled"]:
            self.status = "active"
    
    def process_task(self, task_data: Any) -> Dict[str, Any]:
        """
        Process a computational task.
        
        Args:
            task_data: Input data for processing
            
        Returns:
            Dictionary containing processing results
        """
        # Simulate computational processing
        complexity = np.random.random()
        processing_time = complexity * sum(self.capabilities.values())
        
        return {
            "agent_id": self.agent_id,
            "result": f"Processed by {self.agent_id}",
            "complexity": complexity,
            "processing_time": processing_time,
            "status": "completed"
        }
    
    def update_state(self, new_state: np.ndarray) -> None:
        """Update the agent's internal state."""
        self.state = new_state.copy()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get agent performance metrics."""
        return {
            "agent_id": self.agent_id,
            "position": self.position.tolist(),
            "state_norm": float(np.linalg.norm(self.state)),
            "capabilities": self.capabilities,
            "status": self.status
        }


class AgentFactory:
    """
    Factory class for creating and managing computational agents.
    Designed to deploy exactly 91 agents for the mathematical framework.
    """
    
    def __init__(self):
        """Initialize the AgentFactory."""
        self.agents: List[ComputationalAgent] = []
        self.agent_registry: Dict[str, ComputationalAgent] = {}
        
    async def create_agent(self, agent_config: Optional[Dict[str, Any]] = None) -> ComputationalAgent:
        """
        Create a single computational agent.
        
        Args:
            agent_config: Optional configuration for the agent
            
        Returns:
            Created ComputationalAgent instance
        """
        # Generate unique agent ID
        agent_id = f"agent_{uuid.uuid4().hex[:8]}"
        
        # Default configuration
        default_config = {
            "position_dim": 3,
            "state_dim": 4,
            "capabilities": {
                "computation": np.random.uniform(0.5, 1.0),
                "memory": np.random.uniform(0.5, 1.0),
                "networking": np.random.uniform(0.5, 1.0),
                "analysis": np.random.uniform(0.5, 1.0)
            }
        }
        
        # Merge with provided config
        config = {**default_config, **(agent_config or {})}
        
        # Generate random position and state
        position = np.random.uniform(-1, 1, config["position_dim"])
        state = np.random.uniform(-1, 1, config["state_dim"])
        
        # Normalize state vector
        state = state / np.linalg.norm(state)
        
        # Create agent
        agent = ComputationalAgent(
            agent_id=agent_id,
            position=position,
            state=state,
            capabilities=config["capabilities"]
        )
        
        # Small delay to simulate creation overhead
        await asyncio.sleep(0.001)
        
        return agent
    
    async def deploy_all_agents(self, count: int = 91) -> List[ComputationalAgent]:
        """
        Deploy all computational agents for the mathematical framework.
        
        Args:
            count: Number of agents to deploy (default: 91)
            
        Returns:
            List of deployed ComputationalAgent instances
        """
        print(f"Deploying {count} computational agents...")
        
        # Create agents in batches for efficiency
        batch_size = 10
        all_agents = []
        
        for i in range(0, count, batch_size):
            batch_count = min(batch_size, count - i)
            
            # Create batch of agents concurrently
            batch_tasks = [
                self.create_agent() for _ in range(batch_count)
            ]
            
            batch_agents = await asyncio.gather(*batch_tasks)
            all_agents.extend(batch_agents)
            
            print(f"Deployed batch {i//batch_size + 1}: {len(batch_agents)} agents")
        
        # Register all agents
        for agent in all_agents:
            self.agent_registry[agent.agent_id] = agent
        
        self.agents = all_agents
        
        print(f"Successfully deployed {len(all_agents)} agents")
        return all_agents
    
    async def get_agent_by_id(self, agent_id: str) -> Optional[ComputationalAgent]:
        """
        Retrieve an agent by its ID.
        
        Args:
            agent_id: Unique identifier of the agent
            
        Returns:
            ComputationalAgent instance or None if not found
        """
        return self.agent_registry.get(agent_id)
    
    async def get_all_agents(self) -> List[ComputationalAgent]:
        """
        Get all deployed agents.
        
        Returns:
            List of all ComputationalAgent instances
        """
        return self.agents.copy()
    
    async def update_agent_states(self, state_updates: Dict[str, np.ndarray]) -> None:
        """
        Update states of multiple agents.
        
        Args:
            state_updates: Dictionary mapping agent IDs to new states
        """
        for agent_id, new_state in state_updates.items():
            agent = self.agent_registry.get(agent_id)
            if agent:
                agent.update_state(new_state)
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics for all agents.
        
        Returns:
            Dictionary containing system-wide metrics
        """
        if not self.agents:
            return {"total_agents": 0, "status": "no_agents_deployed"}
        
        # Collect metrics from all agents
        agent_metrics = [agent.get_metrics() for agent in self.agents]
        
        # Compute system-wide statistics
        total_agents = len(self.agents)
        active_agents = len([a for a in self.agents if a.status == "active"])
        
        # Average capabilities
        all_capabilities = [agent.capabilities for agent in self.agents]
        avg_capabilities = {}
        if all_capabilities:
            capability_keys = all_capabilities[0].keys()
            for key in capability_keys:
                avg_capabilities[key] = np.mean([cap[key] for cap in all_capabilities])
        
        return {
            "total_agents": total_agents,
            "active_agents": active_agents,
            "average_capabilities": avg_capabilities,
            "agent_details": agent_metrics[:10],  # First 10 for brevity
            "status": "operational"
        }
    
    def __len__(self) -> int:
        """Return the number of deployed agents."""
        return len(self.agents)
    
    def __repr__(self) -> str:
        """String representation of the AgentFactory."""
        return f"AgentFactory(agents={len(self.agents)}, registered={len(self.agent_registry)})"