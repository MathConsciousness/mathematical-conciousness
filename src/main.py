"""
Main Application: Mathematical Framework System
Advanced scientific computing framework implementing 5-tuple mathematics and multi-agent networks.
"""

import asyncio
import uvicorn
import numpy as np
from fastapi import FastAPI
from core.logic_field import LogicField
from core.agent_factory import AgentFactory
from network.builder import NetworkBuilder
from api.endpoints import router, initialize_api_globals

# Global system components
system_field = None
system_agents = []
system_network = None
system_factory = None

async def initialize_system():
    """Initialize the Mathematical Framework System"""
    global system_field, system_agents, system_network, system_factory
    
    print("Initializing Mathematical Framework System...")
    
    # Initialize LogicField with 5-tuple mathematics
    print("Setting up LogicField with 5-tuple mathematics...")
    system_field = LogicField(
        tau=0.95, 
        Q=np.eye(3), 
        phi_grad=np.zeros(3), 
        sigma=0.8, 
        S=np.identity(4)
    )
    print(f"LogicField initialized: {system_field}")
    
    # Create and deploy 91 computational agents
    print("Creating AgentFactory and deploying computational agents...")
    system_factory = AgentFactory()
    system_agents = await system_factory.deploy_all_agents(count=91)
    print(f"Deployed {len(system_agents)} computational agents")
    
    # Build network with 96.4% density target
    print("Building agent network with target density 96.4%...")
    builder = NetworkBuilder()
    system_network = await builder.build_network(system_agents, target_density=0.964)
    print(f"Network built with density: {system_network.density():.3f}")
    
    # Initialize API globals
    initialize_api_globals(system_agents, system_network, system_field, system_factory)
    print("API endpoints initialized with system components")
    
    return system_field, system_agents, system_network

# Create FastAPI application
app = FastAPI(
    title="Mathematical Framework System",
    description="Advanced scientific computing framework implementing 5-tuple mathematics and multi-agent networks",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Include API router
app.include_router(router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    try:
        field, agents, network = await initialize_system()
        
        print("\n" + "="*60)
        print("MATHEMATICAL FRAMEWORK SYSTEM - INITIALIZATION COMPLETE")
        print("="*60)
        print(f"✓ LogicField initialized: L(x,t) = (τ={field.tau}, Q, ∇Φ, σ={field.sigma}, S)")
        print(f"✓ Agents deployed: {len(agents)} computational entities")
        print(f"✓ Network density: {network.density():.3f} (target: 0.964)")
        print(f"✓ System status: OPERATIONAL")
        print("="*60)
        print(f"API Documentation: http://localhost:8000/docs")
        print(f"System Health: http://localhost:8000/api/v1/system/status")
        print("="*60)
        
    except Exception as e:
        print(f"ERROR: Failed to initialize system: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("Shutting down Mathematical Framework System...")

@app.get("/")
async def root():
    """Root endpoint with system overview"""
    return {
        "system": "Mathematical Framework System",
        "version": "1.0.0",
        "status": "operational",
        "description": "Advanced scientific computing framework implementing 5-tuple mathematics",
        "components": {
            "logic_field": system_field is not None,
            "agents_deployed": len(system_agents) if system_agents else 0,
            "network_initialized": system_network is not None
        },
        "endpoints": {
            "api_documentation": "/docs",
            "system_status": "/api/v1/system/status",
            "deploy_agents": "/api/v1/deploy/agents",
            "network_status": "/api/v1/network/status",
            "run_simulation": "/api/v1/simulate/protocol",
            "analysis_results": "/api/v1/results/analysis"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "system_initialized": system_field is not None,
        "agents_count": len(system_agents) if system_agents else 0,
        "network_density": system_network.density() if system_network else 0.0
    }

if __name__ == "__main__":
    print("Starting Mathematical Framework System...")
    print("5-tuple mathematics: L(x,t) = (τ, Q, ∇Φ, σ, S)")
    print("Target: 91 agents with 96.4% network density")
    
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )