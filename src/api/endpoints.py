"""FastAPI endpoints for Mathematical Consciousness Framework

This module provides REST API endpoints for agent deployment, network management,
and scientific computing operations.
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from pydantic import BaseModel, Field
import numpy as np
from datetime import datetime
import uuid
import asyncio

from ..core.agent_factory import AgentFactory, Agent, AgentType
from ..network.builder import NetworkBuilder, NetworkMetrics
from ..computing.protocols import ScientificProtocols


# Pydantic models for request/response schemas
class AgentConfig(BaseModel):
    """Configuration for agent creation"""
    agent_type: str = Field(..., description="Type of agent to create")
    tau: float = Field(1.0, ge=0.1, le=10.0, description="Intelligence parameter")
    field_dimension: int = Field(3, ge=1, le=10, description="Field vector dimension")
    matrix_size: int = Field(3, ge=1, le=10, description="Matrix size")
    coherence: float = Field(0.8, ge=0.0, le=1.0, description="Coherence parameter")
    enable_optimization: bool = Field(True, description="Enable optimization capabilities")
    enable_quantum: bool = Field(False, description="Enable quantum simulation")
    count: int = Field(1, ge=1, le=100, description="Number of agents to create")


class NetworkConfig(BaseModel):
    """Configuration for network creation"""
    target_density: float = Field(0.964, ge=0.0, le=1.0, description="Target network density")
    optimization_enabled: bool = Field(True, description="Enable network optimization")
    network_id: Optional[str] = Field(None, description="Custom network identifier")


class QuantumSimulationRequest(BaseModel):
    """Request for quantum simulation"""
    hamiltonian: List[List[complex]] = Field(..., description="Hamiltonian matrix")
    initial_state: List[complex] = Field(..., description="Initial quantum state")
    time_points: List[float] = Field(..., description="Time evolution points")
    method: str = Field("exact", description="Simulation method")
    trotter_steps: int = Field(100, ge=1, description="Trotter decomposition steps")


class TemporalAnalysisRequest(BaseModel):
    """Request for temporal analysis"""
    data: List[List[float]] = Field(..., description="Time series data")
    analysis_type: str = Field("full", description="Type of analysis to perform")


class AgentResponse(BaseModel):
    """Response containing agent information"""
    id: str
    type: str
    active: bool
    intelligence_level: float
    coherence: float
    capabilities: Dict[str, bool]
    status: Dict[str, Any]


class NetworkStatus(BaseModel):
    """Network status response"""
    network_id: str
    metrics: Dict[str, float]
    agent_status: Dict[str, Any]
    health_score: float
    last_optimization: str


class SimulationResponse(BaseModel):
    """Response for scientific simulations"""
    success: bool
    data: List[List[float]]
    metadata: Dict[str, Any]
    computation_time: float


# Global instances (in production, use dependency injection)
agent_factory = AgentFactory()
network_builder = NetworkBuilder()
scientific_protocols = ScientificProtocols()

# Store active networks and agents
active_networks: Dict[str, Any] = {}
active_agents: Dict[str, Agent] = {}

# Create API router
router = APIRouter(prefix="/api/v1", tags=["Mathematical Consciousness"])


@router.post("/deploy/agents", response_model=List[AgentResponse])
async def deploy_agents(config: AgentConfig) -> List[AgentResponse]:
    """Deploy computational agents
    
    Creates one or more computational agents based on the provided configuration.
    Supports mathematical, scientific, and quantum agent types.
    
    Args:
        config: Agent configuration parameters
        
    Returns:
        List of created agents with their properties
        
    Raises:
        HTTPException: If agent creation fails
    """
    try:
        agents = []
        agent_type = config.agent_type.lower()
        
        # Prepare parameters
        params = {
            "tau": config.tau,
            "field_dimension": config.field_dimension,
            "matrix_size": config.matrix_size,
            "coherence": config.coherence,
            "enable_optimization": config.enable_optimization,
            "enable_quantum": config.enable_quantum,
        }
        
        # Create agents based on type
        for _ in range(config.count):
            if agent_type == "mathematical":
                agent = agent_factory.create_mathematical_agent(params)
            elif agent_type == "scientific":
                agent = agent_factory.create_scientific_agent(params)
            elif agent_type == "quantum":
                agent = agent_factory.create_quantum_agent(params)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown agent type: {config.agent_type}"
                )
            
            agents.append(agent)
            active_agents[agent.id] = agent
        
        # Convert to response format
        responses = []
        for agent in agents:
            intelligence = agent.logic_field.calculate_intelligence_level()
            coherence = agent.logic_field.compute_coherence()
            
            responses.append(AgentResponse(
                id=agent.id,
                type=agent.agent_type.value,
                active=agent.state.active,
                intelligence_level=intelligence,
                coherence=coherence,
                capabilities=agent.capabilities.__dict__,
                status=agent.get_status(),
            ))
        
        return responses
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to deploy agents: {str(e)}"
        )


@router.get("/agents", response_model=List[AgentResponse])
async def list_agents() -> List[AgentResponse]:
    """List all active agents
    
    Returns:
        List of all currently active agents
    """
    responses = []
    
    for agent in active_agents.values():
        intelligence = agent.logic_field.calculate_intelligence_level()
        coherence = agent.logic_field.compute_coherence()
        
        responses.append(AgentResponse(
            id=agent.id,
            type=agent.agent_type.value,
            active=agent.state.active,
            intelligence_level=intelligence,
            coherence=coherence,
            capabilities=agent.capabilities.__dict__,
            status=agent.get_status(),
        ))
    
    return responses


@router.get("/agents/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: str) -> AgentResponse:
    """Get specific agent information
    
    Args:
        agent_id: ID of agent to retrieve
        
    Returns:
        Agent information
        
    Raises:
        HTTPException: If agent not found
    """
    if agent_id not in active_agents:
        raise HTTPException(
            status_code=404,
            detail=f"Agent {agent_id} not found"
        )
    
    agent = active_agents[agent_id]
    intelligence = agent.logic_field.calculate_intelligence_level()
    coherence = agent.logic_field.compute_coherence()
    
    return AgentResponse(
        id=agent.id,
        type=agent.agent_type.value,
        active=agent.state.active,
        intelligence_level=intelligence,
        coherence=coherence,
        capabilities=agent.capabilities.__dict__,
        status=agent.get_status(),
    )


@router.post("/agents/{agent_id}/execute")
async def execute_agent_operation(
    agent_id: str,
    operation: str,
    parameters: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Execute operation on specific agent
    
    Args:
        agent_id: ID of agent to use
        operation: Operation name to execute
        parameters: Operation parameters
        
    Returns:
        Operation result
        
    Raises:
        HTTPException: If agent not found or operation fails
    """
    if agent_id not in active_agents:
        raise HTTPException(
            status_code=404,
            detail=f"Agent {agent_id} not found"
        )
    
    agent = active_agents[agent_id]
    parameters = parameters or {}
    
    try:
        result = agent.execute_operation(operation, **parameters)
        return {
            "agent_id": agent_id,
            "operation": operation,
            "result": result,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Operation failed: {str(e)}"
        )


@router.post("/network/create")
async def create_network(config: NetworkConfig) -> Dict[str, Any]:
    """Create agent network
    
    Args:
        config: Network configuration
        
    Returns:
        Network creation result
        
    Raises:
        HTTPException: If network creation fails
    """
    if len(active_agents) < 2:
        raise HTTPException(
            status_code=400,
            detail="Need at least 2 agents to create a network"
        )
    
    try:
        agents = list(active_agents.values())
        network_id = config.network_id or f"network_{len(active_networks)}"
        
        network = network_builder.build_network(
            agents,
            target_density=config.target_density,
            network_id=network_id
        )
        
        if config.optimization_enabled:
            network_builder.optimize_information_flow(network)
        
        active_networks[network_id] = {
            "network": network,
            "agents": agents,
            "created_at": datetime.utcnow().isoformat(),
            "optimized": config.optimization_enabled,
        }
        
        metrics = network_builder.get_network_metrics(network)
        
        return {
            "network_id": network_id,
            "created": True,
            "agent_count": len(agents),
            "edge_count": network.number_of_edges(),
            "metrics": metrics.__dict__,
            "optimization_enabled": config.optimization_enabled,
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create network: {str(e)}"
        )


@router.get("/network/status", response_model=List[NetworkStatus])
async def get_network_status() -> List[NetworkStatus]:
    """Get network topology and metrics
    
    Returns:
        List of network status information for all active networks
    """
    statuses = []
    
    for network_id, network_data in active_networks.items():
        try:
            network = network_data["network"]
            agents = network_data["agents"]
            
            metrics = network_builder.get_network_metrics(network)
            
            agent_status = {
                "total_agents": len(agents),
                "active_agents": sum(1 for agent in agents if agent.state.active),
                "average_load": np.mean([agent.state.processing_load for agent in agents]),
                "total_errors": sum(agent.state.error_count for agent in agents),
            }
            
            health_score = min(1.0, metrics.connectivity_score * metrics.efficiency)
            
            statuses.append(NetworkStatus(
                network_id=network_id,
                metrics=metrics.__dict__,
                agent_status=agent_status,
                health_score=health_score,
                last_optimization=network_data.get("last_optimization", "Never"),
            ))
            
        except Exception as e:
            # Include error networks in status
            statuses.append(NetworkStatus(
                network_id=network_id,
                metrics={},
                agent_status={"error": str(e)},
                health_score=0.0,
                last_optimization="Error",
            ))
    
    return statuses


@router.get("/network/{network_id}/status", response_model=NetworkStatus)
async def get_specific_network_status(network_id: str) -> NetworkStatus:
    """Get status for specific network
    
    Args:
        network_id: ID of network to check
        
    Returns:
        Network status information
        
    Raises:
        HTTPException: If network not found
    """
    if network_id not in active_networks:
        raise HTTPException(
            status_code=404,
            detail=f"Network {network_id} not found"
        )
    
    network_data = active_networks[network_id]
    network = network_data["network"]
    agents = network_data["agents"]
    
    metrics = network_builder.get_network_metrics(network)
    
    agent_status = {
        "total_agents": len(agents),
        "active_agents": sum(1 for agent in agents if agent.state.active),
        "average_load": np.mean([agent.state.processing_load for agent in agents]),
        "total_errors": sum(agent.state.error_count for agent in agents),
    }
    
    health_score = min(1.0, metrics.connectivity_score * metrics.efficiency)
    
    return NetworkStatus(
        network_id=network_id,
        metrics=metrics.__dict__,
        agent_status=agent_status,
        health_score=health_score,
        last_optimization=network_data.get("last_optimization", "Never"),
    )


@router.post("/network/{network_id}/optimize")
async def optimize_network(network_id: str, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    """Optimize network information flow
    
    Args:
        network_id: ID of network to optimize
        background_tasks: FastAPI background tasks
        
    Returns:
        Optimization status
        
    Raises:
        HTTPException: If network not found
    """
    if network_id not in active_networks:
        raise HTTPException(
            status_code=404,
            detail=f"Network {network_id} not found"
        )
    
    def optimize_task():
        try:
            network_data = active_networks[network_id]
            network = network_data["network"]
            
            network_builder.optimize_information_flow(network)
            network_data["last_optimization"] = datetime.utcnow().isoformat()
            network_data["optimized"] = True
            
        except Exception as e:
            print(f"Network optimization failed: {e}")
    
    background_tasks.add_task(optimize_task)
    
    return {
        "network_id": network_id,
        "optimization_started": True,
        "message": "Network optimization started in background",
    }


@router.post("/compute/quantum-simulation", response_model=SimulationResponse)
async def quantum_simulation(request: QuantumSimulationRequest) -> SimulationResponse:
    """Run quantum field simulation
    
    Args:
        request: Quantum simulation parameters
        
    Returns:
        Simulation results
        
    Raises:
        HTTPException: If simulation fails
    """
    try:
        # Convert complex lists to numpy arrays
        hamiltonian = np.array(request.hamiltonian, dtype=np.complex128)
        initial_state = np.array(request.initial_state, dtype=np.complex128)
        time_points = np.array(request.time_points, dtype=np.float64)
        
        params = {
            "hamiltonian": hamiltonian,
            "initial_state": initial_state,
            "time_steps": time_points,
            "method": request.method,
            "trotter_steps": request.trotter_steps,
        }
        
        import time
        start_time = time.time()
        
        result = scientific_protocols.quantum_simulation(params)
        
        computation_time = time.time() - start_time
        
        # Convert complex result to real representation for JSON
        result_real = []
        for state in result:
            state_real = []
            for amplitude in state:
                state_real.extend([amplitude.real, amplitude.imag])
            result_real.append(state_real)
        
        return SimulationResponse(
            success=True,
            data=result_real,
            metadata={
                "method": request.method,
                "time_points": request.time_points,
                "hamiltonian_shape": list(hamiltonian.shape),
                "state_dimension": len(initial_state),
            },
            computation_time=computation_time,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Quantum simulation failed: {str(e)}"
        )


@router.post("/compute/temporal-analysis")
async def temporal_analysis(request: TemporalAnalysisRequest) -> Dict[str, Any]:
    """Perform temporal data analysis
    
    Args:
        request: Temporal analysis parameters
        
    Returns:
        Analysis results
        
    Raises:
        HTTPException: If analysis fails
    """
    try:
        data = np.array(request.data, dtype=np.float64)
        
        results = scientific_protocols.analyze_temporal_patterns(data)
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        json_results = convert_numpy(results)
        
        return {
            "success": True,
            "analysis_type": request.analysis_type,
            "data_shape": list(data.shape),
            "results": json_results,
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Temporal analysis failed: {str(e)}"
        )


@router.get("/system/status")
async def get_system_status() -> Dict[str, Any]:
    """Get overall system status
    
    Returns:
        Comprehensive system status information
    """
    agent_stats = agent_factory.get_agent_statistics()
    
    network_stats = {
        "total_networks": len(active_networks),
        "active_networks": len([n for n in active_networks.values() if n.get("optimized", False)]),
    }
    
    # Calculate overall system health
    total_agents = len(active_agents)
    active_agent_count = sum(1 for agent in active_agents.values() if agent.state.active)
    avg_intelligence = np.mean([
        agent.logic_field.calculate_intelligence_level() 
        for agent in active_agents.values()
    ]) if active_agents else 0.0
    avg_coherence = np.mean([
        agent.logic_field.compute_coherence()
        for agent in active_agents.values()
    ]) if active_agents else 0.0
    
    system_health = min(1.0, (
        (active_agent_count / max(1, total_agents)) * 0.4 +
        min(1.0, avg_intelligence / 2.0) * 0.3 +
        avg_coherence * 0.3
    ))
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "agents": agent_stats,
        "networks": network_stats,
        "system_health": system_health,
        "metrics": {
            "total_agents": total_agents,
            "active_agents": active_agent_count,
            "average_intelligence": avg_intelligence,
            "average_coherence": avg_coherence,
        },
        "status": "operational" if system_health > 0.7 else "degraded" if system_health > 0.3 else "critical",
    }


@router.delete("/agents/{agent_id}")
async def deactivate_agent(agent_id: str) -> Dict[str, Any]:
    """Deactivate specific agent
    
    Args:
        agent_id: ID of agent to deactivate
        
    Returns:
        Deactivation result
        
    Raises:
        HTTPException: If agent not found
    """
    if agent_id not in active_agents:
        raise HTTPException(
            status_code=404,
            detail=f"Agent {agent_id} not found"
        )
    
    agent = active_agents[agent_id]
    agent.state.active = False
    
    return {
        "agent_id": agent_id,
        "deactivated": True,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.delete("/network/{network_id}")
async def delete_network(network_id: str) -> Dict[str, Any]:
    """Delete network
    
    Args:
        network_id: ID of network to delete
        
    Returns:
        Deletion result
        
    Raises:
        HTTPException: If network not found
    """
    if network_id not in active_networks:
        raise HTTPException(
            status_code=404,
            detail=f"Network {network_id} not found"
        )
    
    del active_networks[network_id]
    
    return {
        "network_id": network_id,
        "deleted": True,
        "timestamp": datetime.utcnow().isoformat(),
    }