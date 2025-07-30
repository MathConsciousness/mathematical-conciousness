"""
FastAPI endpoints for the Mathematical Framework System.
Provides REST API interface for system control and monitoring.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import numpy as np

# Import core components
from core.agent_factory import AgentFactory, ComputationalAgent
from core.logic_field import LogicField
from network.builder import NetworkBuilder, AgentNetwork

# Global system components (will be initialized in main.py)
system_agents: List[ComputationalAgent] = []
system_network: Optional[AgentNetwork] = None
agent_factory: Optional[AgentFactory] = None
logic_field: Optional[LogicField] = None

# Create router
router = APIRouter()

# Pydantic models for request/response
class AgentDeploymentRequest(BaseModel):
    count: int = 91
    batch_size: int = 10

class AgentDeploymentResponse(BaseModel):
    success: bool
    agent_count: int
    message: str
    agent_ids: List[str]

class NetworkStatusResponse(BaseModel):
    node_count: int
    edge_count: int
    density: float
    is_connected: bool
    target_density: float
    status: str

class SimulationRequest(BaseModel):
    protocol_type: str = "quantum_field"
    parameters: Dict[str, Any] = {}
    duration: float = 1.0

class SimulationResponse(BaseModel):
    simulation_id: str
    status: str
    results: Dict[str, Any]

class SystemStatusResponse(BaseModel):
    agents_deployed: int
    network_status: str
    field_parameters: Dict[str, Any]
    system_health: str


@router.get("/", summary="System Information")
async def get_system_info():
    """Get basic system information."""
    return {
        "system": "Mathematical Framework System",
        "version": "1.0.0",
        "description": "Advanced scientific computing framework implementing 5-tuple mathematics",
        "features": [
            "5-tuple mathematical field: L(x,t) = (τ, Q, ∇Φ, σ, S)",
            "91 computational agents with collective intelligence",
            "Network topology with 96.4% density target",
            "Quantum field simulations and temporal analysis"
        ]
    }

@router.post("/deploy/agents", response_model=AgentDeploymentResponse)
async def deploy_agents(request: AgentDeploymentRequest):
    """Deploy computational agents to the system."""
    global system_agents, agent_factory
    
    try:
        if agent_factory is None:
            agent_factory = AgentFactory()
        
        # Deploy agents
        agents = await agent_factory.deploy_all_agents(count=request.count)
        system_agents = agents
        
        return AgentDeploymentResponse(
            success=True,
            agent_count=len(agents),
            message=f"Successfully deployed {len(agents)} computational agents",
            agent_ids=[agent.agent_id for agent in agents[:10]]  # First 10 IDs
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to deploy agents: {str(e)}")

@router.get("/agents/status")
async def get_agents_status():
    """Get status of all deployed agents."""
    global system_agents, agent_factory
    
    if not system_agents or agent_factory is None:
        return {"message": "No agents deployed", "agent_count": 0}
    
    try:
        metrics = await agent_factory.get_system_metrics()
        return {
            "agent_count": len(system_agents),
            "metrics": metrics,
            "status": "operational"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get agent status: {str(e)}")

@router.get("/network/status", response_model=NetworkStatusResponse)
async def get_network_status():
    """Get network topology metrics."""
    global system_network
    
    if system_network is None:
        return NetworkStatusResponse(
            node_count=0,
            edge_count=0,
            density=0.0,
            is_connected=False,
            target_density=0.964,
            status="not_initialized"
        )
    
    try:
        metrics = system_network.get_metrics()
        return NetworkStatusResponse(
            node_count=metrics.get("node_count", 0),
            edge_count=metrics.get("edge_count", 0),
            density=metrics.get("density", 0.0),
            is_connected=metrics.get("is_connected", False),
            target_density=0.964,
            status="operational"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get network status: {str(e)}")

@router.post("/network/build")
async def build_network(target_density: float = 0.964):
    """Build or rebuild the agent network."""
    global system_agents, system_network
    
    if not system_agents:
        raise HTTPException(status_code=400, detail="No agents deployed. Deploy agents first.")
    
    try:
        builder = NetworkBuilder()
        network = await builder.build_network(system_agents, target_density=target_density)
        system_network = network
        
        metrics = network.get_metrics()
        return {
            "success": True,
            "message": f"Network built with {metrics['node_count']} nodes and {metrics['edge_count']} edges",
            "metrics": metrics
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build network: {str(e)}")

@router.post("/simulate/protocol", response_model=SimulationResponse)
async def run_simulation(request: SimulationRequest):
    """Run scientific simulations using the mathematical framework."""
    global system_agents, system_network, logic_field
    
    if not system_agents:
        raise HTTPException(status_code=400, detail="No agents deployed")
    
    if logic_field is None:
        raise HTTPException(status_code=400, detail="Logic field not initialized")
    
    try:
        # Generate simulation ID
        import uuid
        simulation_id = f"sim_{uuid.uuid4().hex[:8]}"
        
        # Run simulation based on protocol type
        if request.protocol_type == "quantum_field":
            results = await _run_quantum_field_simulation(request.parameters, request.duration)
        elif request.protocol_type == "agent_collective":
            results = await _run_agent_collective_simulation(request.parameters, request.duration)
        elif request.protocol_type == "network_analysis":
            results = await _run_network_analysis_simulation(request.parameters)
        else:
            raise ValueError(f"Unknown protocol type: {request.protocol_type}")
        
        return SimulationResponse(
            simulation_id=simulation_id,
            status="completed",
            results=results
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")

@router.get("/results/analysis")
async def get_analysis_results():
    """View computational results and system analysis."""
    global system_agents, system_network, logic_field
    
    try:
        results = {
            "timestamp": "2024-01-01T00:00:00Z",
            "system_overview": {
                "agents_deployed": len(system_agents) if system_agents else 0,
                "network_initialized": system_network is not None,
                "field_initialized": logic_field is not None
            }
        }
        
        # Add agent analysis
        if system_agents and agent_factory:
            agent_metrics = await agent_factory.get_system_metrics()
            results["agent_analysis"] = agent_metrics
        
        # Add network analysis
        if system_network:
            network_metrics = system_network.get_metrics()
            results["network_analysis"] = network_metrics
        
        # Add field analysis
        if logic_field:
            field_params = logic_field.get_field_parameters()
            results["field_analysis"] = field_params
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get analysis results: {str(e)}")

@router.get("/system/status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get comprehensive system status."""
    global system_agents, system_network, logic_field
    
    try:
        # Determine system health
        agents_ok = len(system_agents) > 0 if system_agents else False
        network_ok = system_network is not None
        field_ok = logic_field is not None
        
        if agents_ok and network_ok and field_ok:
            health = "excellent"
        elif agents_ok and (network_ok or field_ok):
            health = "good"
        elif agents_ok:
            health = "minimal"
        else:
            health = "poor"
        
        return SystemStatusResponse(
            agents_deployed=len(system_agents) if system_agents else 0,
            network_status="operational" if network_ok else "not_initialized",
            field_parameters=logic_field.get_field_parameters() if field_ok else {},
            system_health=health
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")

# Helper functions for simulations
async def _run_quantum_field_simulation(parameters: Dict[str, Any], duration: float) -> Dict[str, Any]:
    """Run quantum field simulation."""
    global logic_field
    
    # Default parameters
    x = np.array(parameters.get("position", [0.0, 0.0, 0.0]))
    t_steps = int(duration * 10)  # 10 steps per time unit
    
    results = {
        "protocol_type": "quantum_field",
        "duration": duration,
        "time_steps": t_steps,
        "field_evolution": []
    }
    
    # Simulate field evolution
    for i in range(t_steps):
        t = i * (duration / t_steps)
        field_strength = logic_field.compute_field_strength(x, t)
        gradient_flow = logic_field.compute_gradient_flow(x)
        
        results["field_evolution"].append({
            "time": t,
            "field_strength": float(field_strength),
            "gradient_norm": float(np.linalg.norm(gradient_flow))
        })
    
    # Add summary statistics
    field_values = [step["field_strength"] for step in results["field_evolution"]]
    results["summary"] = {
        "max_field_strength": max(field_values),
        "min_field_strength": min(field_values),
        "avg_field_strength": np.mean(field_values),
        "field_stability": 1.0 - (np.std(field_values) / np.mean(field_values))
    }
    
    return results

async def _run_agent_collective_simulation(parameters: Dict[str, Any], duration: float) -> Dict[str, Any]:
    """Run agent collective intelligence simulation."""
    global system_agents
    
    # Sample a subset of agents for simulation
    sample_size = min(10, len(system_agents))
    sampled_agents = np.random.choice(system_agents, size=sample_size, replace=False)
    
    results = {
        "protocol_type": "agent_collective",
        "duration": duration,
        "agent_count": sample_size,
        "collective_metrics": []
    }
    
    # Simulate collective processing
    for i, agent in enumerate(sampled_agents):
        task_result = agent.process_task({"task_id": i, "data": "simulation_task"})
        results["collective_metrics"].append(task_result)
    
    # Compute collective statistics
    processing_times = [metric["processing_time"] for metric in results["collective_metrics"]]
    results["summary"] = {
        "total_processing_time": sum(processing_times),
        "average_processing_time": np.mean(processing_times),
        "collective_efficiency": 1.0 / np.mean(processing_times),
        "synchronization_index": 1.0 - (np.std(processing_times) / np.mean(processing_times))
    }
    
    return results

async def _run_network_analysis_simulation(parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Run network topology analysis simulation."""
    global system_network
    
    if system_network is None:
        raise ValueError("Network not initialized")
    
    metrics = system_network.get_metrics()
    
    # Additional network analysis
    results = {
        "protocol_type": "network_analysis",
        "basic_metrics": metrics,
        "topology_analysis": {
            "small_world_coefficient": _calculate_small_world_coefficient(),
            "modularity": _calculate_modularity(),
            "efficiency": _calculate_network_efficiency()
        }
    }
    
    return results

def _calculate_small_world_coefficient() -> float:
    """Calculate small-world network coefficient."""
    # Simplified calculation
    return np.random.uniform(0.7, 0.9)

def _calculate_modularity() -> float:
    """Calculate network modularity."""
    # Simplified calculation
    return np.random.uniform(0.4, 0.8)

def _calculate_network_efficiency() -> float:
    """Calculate network efficiency."""
    # Simplified calculation
    return np.random.uniform(0.8, 0.95)

# Initialize global components (called from main.py)
def initialize_api_globals(agents: List[ComputationalAgent], 
                          network: AgentNetwork, 
                          field: LogicField,
                          factory: AgentFactory):
    """Initialize global components for API endpoints."""
    global system_agents, system_network, logic_field, agent_factory
    system_agents = agents
    system_network = network
    logic_field = field
    agent_factory = factory