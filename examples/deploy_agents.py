"""
Deploy Agents Example Script
Demonstrates how to deploy computational agents using the AgentFactory.
"""

import asyncio
import sys
import os

# Add src to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.agent_factory import AgentFactory

async def deploy_agents():
    """Deploy computational agents and display results."""
    print("Mathematical Framework System - Agent Deployment Example")
    print("="*60)
    
    # Create AgentFactory
    factory = AgentFactory()
    print(f"AgentFactory created: {factory}")
    
    # Deploy 91 agents (default for the system)
    print("\nDeploying 91 computational agents...")
    agents = await factory.deploy_all_agents(count=91)
    
    print(f"\n✓ Successfully deployed {len(agents)} agents")
    
    # Display sample agent information
    print("\nSample Agent Information:")
    for i, agent in enumerate(agents[:5]):  # Show first 5 agents
        print(f"  Agent {i+1}: {agent.agent_id}")
        print(f"    Position: {agent.position}")
        print(f"    Capabilities: {agent.capabilities}")
        print(f"    Status: {agent.status}")
        print()
    
    # Get system metrics
    print("System Metrics:")
    metrics = await factory.get_system_metrics()
    print(f"  Total Agents: {metrics['total_agents']}")
    print(f"  Active Agents: {metrics['active_agents']}")
    print(f"  Average Capabilities: {metrics['average_capabilities']}")
    print(f"  Status: {metrics['status']}")
    
    print("\n" + "="*60)
    print("Agent deployment example completed successfully!")

if __name__ == "__main__":
    asyncio.run(deploy_agents())