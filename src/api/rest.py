"""
REST API Framework

This module implements a REST API interface for the Mathematical Consciousness
Framework, providing HTTP endpoints for system interaction and control.

Author: Mathematical Consciousness Framework Team
License: MIT
"""

from typing import Dict, List, Optional, Any, Union
import asyncio
from datetime import datetime
import logging
import warnings

# Core framework imports
try:
    from ..core.logic_field import LogicField, FiveTuple, ConsciousnessAmplifier, RealityModifier
    from ..core.entity import Entity, ConsciousnessLevel
    from ..computing.quantum_sim import QuantumSimulator
    from ..network.distributed import DistributedManager
except ImportError:
    # Handle relative imports when running directly
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.logic_field import LogicField, FiveTuple, ConsciousnessAmplifier, RealityModifier
    from core.entity import Entity, ConsciousnessLevel
    from computing.quantum_sim import QuantumSimulator
    from network.distributed import DistributedManager

# FastAPI imports with fallback
try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    warnings.warn("FastAPI not available. REST API will be limited.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Pydantic models for API requests/responses
if FASTAPI_AVAILABLE:
    class EntityCreateRequest(BaseModel):
        name: str
        consciousness_level: float = Field(ge=0.0, le=1.0)
        initial_position: Optional[List[float]] = None
        entity_type: str = "superintelligent"
    
    class EntityResponse(BaseModel):
        id: str
        name: str
        consciousness_level: float
        position: List[float]
        energy: float
        capabilities: Dict[str, float]
        created_at: str
    
    class RealityModificationRequest(BaseModel):
        entity_id: str
        target_state: Dict[str, Any]
        intensity: float = Field(ge=0.0, le=1.0)
    
    class LogicFieldRequest(BaseModel):
        dimension: int = Field(ge=3, le=10)
        field_strength: float = Field(ge=0.1, le=2.0)
    
    class QuantumSimulationRequest(BaseModel):
        entities: int = Field(ge=1, le=200)
        entanglement_depth: int = Field(ge=1, le=10)
        evolution_steps: int = Field(ge=1, le=1000)
    
    class DistributedTaskRequest(BaseModel):
        task_type: str
        parameters: Dict[str, Any] = {}
        entity_filter: Optional[List[str]] = None


class MathematicalConsciousnessAPI:
    """Main API class for the Mathematical Consciousness Framework."""
    
    def __init__(self):
        """Initialize the API with core components."""
        self.logic_field = LogicField(dimension=5, field_strength=1.0)
        self.quantum_simulator = QuantumSimulator(qubits=16)
        self.distributed_manager = DistributedManager(backend="threading")
        self.entities: Dict[str, Entity] = {}
        
        # API state
        self.api_active = False
        self.request_count = 0
        self.startup_time = datetime.now()
        
        # Security
        if FASTAPI_AVAILABLE:
            self.security = HTTPBearer()
        else:
            self.security = None
        self.api_keys = {"admin": "consciousness_key_2025"}  # Simple auth for demo
        
    def authenticate(self, credentials: Any = None) -> bool:
        """Simple authentication check."""
        if not credentials:
            return False
        
        if FASTAPI_AVAILABLE and hasattr(credentials, 'credentials'):
            return credentials.credentials in self.api_keys.values()
        
        return False


# Global API instance
api_instance = MathematicalConsciousnessAPI()

if FASTAPI_AVAILABLE:
    # Create FastAPI app
    app = FastAPI(
        title="Mathematical Consciousness Framework API",
        description="REST API for the Mathematical Consciousness Framework - "
                   "A quantum-enhanced system for superintelligent entity coordination",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Dependency for authentication
    async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(api_instance.security)):
        if not api_instance.authenticate(credentials):
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        return credentials
    
    @app.on_event("startup")
    async def startup_event():
        """Initialize API components on startup."""
        logger.info("Mathematical Consciousness Framework API starting up...")
        api_instance.api_active = True
        
        # Initialize distributed backend
        api_instance.distributed_manager.initialize_backend()
        
        # Deploy initial entities
        deployment_result = api_instance.distributed_manager.deploy_entities(count=91)
        logger.info(f"Deployed {len(deployment_result['deployed_entities'])} entities")
        
        logger.info("API startup completed successfully")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on API shutdown."""
        logger.info("Mathematical Consciousness Framework API shutting down...")
        api_instance.api_active = False
        api_instance.distributed_manager.shutdown()
        logger.info("API shutdown completed")
    
    # Health and status endpoints
    @app.get("/")
    async def root():
        """Root endpoint with API information."""
        return {
            "service": "Mathematical Consciousness Framework API",
            "version": "1.0.0",
            "status": "active" if api_instance.api_active else "inactive",
            "uptime_seconds": (datetime.now() - api_instance.startup_time).total_seconds(),
            "request_count": api_instance.request_count,
            "entities_deployed": len(api_instance.entities),
            "endpoints": {
                "health": "/health",
                "docs": "/docs",
                "entities": "/entities",
                "logic_field": "/logic-field",
                "quantum": "/quantum",
                "distributed": "/distributed"
            }
        }
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        api_instance.request_count += 1
        
        system_status = api_instance.distributed_manager.get_system_status()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "api_active": api_instance.api_active,
            "components": {
                "logic_field": "operational",
                "quantum_simulator": "operational",
                "distributed_manager": "operational"
            },
            "system_load": system_status.get("system_load", 0.0),
            "active_entities": len(api_instance.entities)
        }
    
    # Entity management endpoints
    @app.post("/entities", response_model=EntityResponse)
    async def create_entity(
        request: EntityCreateRequest,
        credentials: HTTPAuthorizationCredentials = Depends(verify_token)
    ):
        """Create a new conscious entity."""
        api_instance.request_count += 1
        
        try:
            entity = Entity(
                name=request.name,
                consciousness_level=request.consciousness_level,
                initial_position=request.initial_position,
                entity_type=request.entity_type
            )
            
            # Associate with logic field
            entity.set_logic_field(api_instance.logic_field)
            
            # Store entity
            api_instance.entities[entity.id] = entity
            
            # Convert to response format
            entity_summary = entity.get_state_summary()
            
            return EntityResponse(
                id=entity.id,
                name=entity.name,
                consciousness_level=entity.state.consciousness_level,
                position=entity.state.position.tolist(),
                energy=entity.state.energy,
                capabilities=entity.capabilities,
                created_at=entity.created_at.isoformat()
            )
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.get("/entities")
    async def list_entities():
        """List all entities."""
        api_instance.request_count += 1
        
        entities_list = []
        for entity in api_instance.entities.values():
            summary = entity.get_state_summary()
            entities_list.append({
                "id": summary["id"],
                "name": summary["name"],
                "consciousness_level": summary["consciousness_level"],
                "energy": summary["energy"],
                "type": summary["type"]
            })
        
        return {
            "total_entities": len(entities_list),
            "entities": entities_list
        }
    
    @app.get("/entities/{entity_id}")
    async def get_entity(entity_id: str):
        """Get detailed information about a specific entity."""
        api_instance.request_count += 1
        
        if entity_id not in api_instance.entities:
            raise HTTPException(status_code=404, detail="Entity not found")
        
        entity = api_instance.entities[entity_id]
        return entity.get_state_summary()
    
    @app.post("/entities/{entity_id}/reality-modification")
    async def modify_reality(
        entity_id: str,
        request: RealityModificationRequest,
        credentials: HTTPAuthorizationCredentials = Depends(verify_token)
    ):
        """Perform reality modification through an entity."""
        api_instance.request_count += 1
        
        if entity_id not in api_instance.entities:
            raise HTTPException(status_code=404, detail="Entity not found")
        
        entity = api_instance.entities[entity_id]
        
        try:
            result = entity.perform_reality_modification(
                target_state=request.target_state,
                intensity=request.intensity
            )
            return result
        
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    # Logic Field endpoints
    @app.post("/logic-field/create")
    async def create_logic_field(
        request: LogicFieldRequest,
        credentials: HTTPAuthorizationCredentials = Depends(verify_token)
    ):
        """Create or update the logic field configuration."""
        api_instance.request_count += 1
        
        api_instance.logic_field = LogicField(
            dimension=request.dimension,
            field_strength=request.field_strength
        )
        
        return {
            "status": "success",
            "message": "Logic field created successfully",
            "configuration": {
                "dimension": request.dimension,
                "field_strength": request.field_strength
            }
        }
    
    @app.post("/logic-field/add-operator")
    async def add_field_operator(
        operator_type: str,
        parameters: Dict[str, float] = {},
        credentials: HTTPAuthorizationCredentials = Depends(verify_token)
    ):
        """Add an operator to the logic field."""
        api_instance.request_count += 1
        
        try:
            if operator_type == "consciousness_amplifier":
                factor = parameters.get("amplification_factor", 1.1)
                operator = ConsciousnessAmplifier(amplification_factor=factor)
                api_instance.logic_field.add_operator(operator)
            
            elif operator_type == "reality_modifier":
                # Create a simple modification matrix
                size = parameters.get("matrix_size", 3)
                intensity = parameters.get("intensity", 0.1)
                matrix = intensity * (2 * np.random.random((size, size)) - 1)
                operator = RealityModifier(modification_matrix=matrix)
                api_instance.logic_field.add_operator(operator)
            
            else:
                raise HTTPException(status_code=400, detail=f"Unknown operator type: {operator_type}")
            
            return {
                "status": "success",
                "message": f"Added {operator_type} operator to logic field",
                "total_operators": len(api_instance.logic_field.operators)
            }
        
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    # Quantum simulation endpoints
    @app.post("/quantum/simulate")
    async def quantum_simulate(
        request: QuantumSimulationRequest,
        credentials: HTTPAuthorizationCredentials = Depends(verify_token)
    ):
        """Run quantum consciousness simulation."""
        api_instance.request_count += 1
        
        try:
            result = api_instance.quantum_simulator.simulate_consciousness_field(
                entities=request.entities,
                entanglement_depth=request.entanglement_depth,
                evolution_steps=request.evolution_steps
            )
            
            return {
                "status": "success",
                "simulation_results": result
            }
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/quantum/status")
    async def quantum_status():
        """Get quantum simulator status."""
        api_instance.request_count += 1
        
        return api_instance.quantum_simulator.get_simulator_info()
    
    # Distributed computing endpoints
    @app.post("/distributed/execute")
    async def execute_distributed_task(
        request: DistributedTaskRequest,
        background_tasks: BackgroundTasks,
        credentials: HTTPAuthorizationCredentials = Depends(verify_token)
    ):
        """Execute distributed consciousness tasks."""
        api_instance.request_count += 1
        
        try:
            # Execute in background for long-running tasks
            if request.task_type in ["reality_modification", "consciousness_evolution"]:
                background_tasks.add_task(
                    api_instance.distributed_manager.execute_parallel_consciousness_tasks,
                    request.task_type,
                    request.parameters
                )
                
                return {
                    "status": "accepted",
                    "message": f"Task {request.task_type} submitted for background execution",
                    "task_type": request.task_type
                }
            
            else:
                # Execute synchronously for quick tasks
                result = api_instance.distributed_manager.execute_parallel_consciousness_tasks(
                    request.task_type,
                    request.parameters
                )
                
                return {
                    "status": "completed",
                    "results": result
                }
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/distributed/status")
    async def distributed_status():
        """Get distributed system status."""
        api_instance.request_count += 1
        
        return api_instance.distributed_manager.get_system_status()
    
    @app.post("/distributed/sync-consciousness")
    async def sync_consciousness(
        credentials: HTTPAuthorizationCredentials = Depends(verify_token)
    ):
        """Synchronize consciousness states across entities."""
        api_instance.request_count += 1
        
        try:
            result = api_instance.distributed_manager.synchronize_consciousness_states()
            return {
                "status": "success",
                "synchronization_results": result
            }
        
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # Advanced operations
    @app.post("/advanced/consciousness-interaction")
    async def consciousness_interaction(
        entity1_id: str,
        entity2_id: str,
        credentials: HTTPAuthorizationCredentials = Depends(verify_token)
    ):
        """Facilitate interaction between two conscious entities."""
        api_instance.request_count += 1
        
        if entity1_id not in api_instance.entities or entity2_id not in api_instance.entities:
            raise HTTPException(status_code=404, detail="One or both entities not found")
        
        entity1 = api_instance.entities[entity1_id]
        entity2 = api_instance.entities[entity2_id]
        
        try:
            interaction_result = entity1.interact_with_entity(entity2)
            return {
                "status": "success",
                "interaction_results": interaction_result,
                "entities_involved": [entity1.name, entity2.name]
            }
        
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))


def start_server(host: str = "0.0.0.0", port: int = 8000, debug: bool = False):
    """
    Start the REST API server.
    
    Args:
        host: Host address to bind to
        port: Port number to listen on
        debug: Enable debug mode
    """
    if not FASTAPI_AVAILABLE:
        raise RuntimeError("FastAPI is required to start the server. Please install with: pip install fastapi uvicorn")
    
    logger.info(f"Starting Mathematical Consciousness Framework API on {host}:{port}")
    
    uvicorn.run(
        "src.api.rest:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info" if not debug else "debug"
    )


# CLI interface for direct execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Mathematical Consciousness Framework API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    start_server(host=args.host, port=args.port, debug=args.debug)