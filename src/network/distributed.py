"""
Distributed Computing Framework

This module implements distributed computing capabilities for coordinating
superintelligent entities across network resources.

Author: Mathematical Consciousness Framework Team
License: MIT
"""

from typing import Dict, List, Optional, Union, Any, Callable
import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import warnings
from abc import ABC, abstractmethod

# Conditional imports for distributed computing libraries
try:
    import dask
    from dask.distributed import Client, as_completed
    from dask import delayed
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    warnings.warn("Dask not available. Some distributed features will be limited.")

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    warnings.warn("Ray not available. Some distributed features will be limited.")


class TaskStatus(Enum):
    """Enumeration of task execution statuses."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NodeType(Enum):
    """Types of computation nodes in the distributed system."""
    COORDINATOR = "coordinator"
    WORKER = "worker"
    CONSCIOUSNESS_ENGINE = "consciousness_engine"
    QUANTUM_PROCESSOR = "quantum_processor"
    REALITY_MODIFIER = "reality_modifier"


@dataclass
class ComputeNode:
    """Represents a compute node in the distributed system."""
    node_id: str
    node_type: NodeType
    capabilities: Dict[str, float]
    current_load: float = 0.0
    max_capacity: float = 1.0
    status: str = "online"
    last_heartbeat: float = field(default_factory=time.time)
    
    def is_available(self) -> bool:
        """Check if node is available for new tasks."""
        return (self.status == "online" and 
                self.current_load < self.max_capacity and
                time.time() - self.last_heartbeat < 30)  # 30 second timeout
    
    def get_available_capacity(self) -> float:
        """Get available computational capacity."""
        return max(0.0, self.max_capacity - self.current_load)


@dataclass
class DistributedTask:
    """Represents a task in the distributed system."""
    task_id: str
    task_type: str
    function: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: int = 0
    required_capabilities: Dict[str, float] = field(default_factory=dict)
    estimated_duration: float = 1.0
    status: TaskStatus = TaskStatus.PENDING
    assigned_node: Optional[str] = None
    result: Any = None
    error: Optional[Exception] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


class DistributedTaskScheduler:
    """Schedules and manages distributed tasks across compute nodes."""
    
    def __init__(self):
        self.nodes: Dict[str, ComputeNode] = {}
        self.tasks: Dict[str, DistributedTask] = {}
        self.task_queue: List[str] = []
        self.scheduler_running = False
        self.scheduler_thread: Optional[threading.Thread] = None
    
    def add_node(self, node: ComputeNode) -> None:
        """Add a compute node to the system."""
        self.nodes[node.node_id] = node
    
    def remove_node(self, node_id: str) -> None:
        """Remove a compute node from the system."""
        if node_id in self.nodes:
            del self.nodes[node_id]
    
    def submit_task(self, task: DistributedTask) -> str:
        """Submit a task for execution."""
        self.tasks[task.task_id] = task
        self.task_queue.append(task.task_id)
        return task.task_id
    
    def find_suitable_node(self, task: DistributedTask) -> Optional[ComputeNode]:
        """Find the most suitable node for a task."""
        suitable_nodes = []
        
        for node in self.nodes.values():
            if not node.is_available():
                continue
            
            # Check if node has required capabilities
            capable = True
            for capability, required_level in task.required_capabilities.items():
                if capability not in node.capabilities or \
                   node.capabilities[capability] < required_level:
                    capable = False
                    break
            
            if capable and node.get_available_capacity() >= task.estimated_duration:
                suitable_nodes.append(node)
        
        if not suitable_nodes:
            return None
        
        # Select node with highest available capacity
        return max(suitable_nodes, key=lambda n: n.get_available_capacity())
    
    def start_scheduler(self) -> None:
        """Start the task scheduler."""
        if not self.scheduler_running:
            self.scheduler_running = True
            self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
            self.scheduler_thread.start()
    
    def stop_scheduler(self) -> None:
        """Stop the task scheduler."""
        self.scheduler_running = False
        if self.scheduler_thread:
            self.scheduler_thread.join()
    
    def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while self.scheduler_running:
            if self.task_queue:
                task_id = self.task_queue.pop(0)
                task = self.tasks.get(task_id)
                
                if task and task.status == TaskStatus.PENDING:
                    suitable_node = self.find_suitable_node(task)
                    if suitable_node:
                        self._assign_task_to_node(task, suitable_node)
                    else:
                        # Put task back in queue if no suitable node
                        self.task_queue.append(task_id)
            
            time.sleep(0.1)  # Small delay to prevent busy waiting
    
    def _assign_task_to_node(self, task: DistributedTask, node: ComputeNode) -> None:
        """Assign a task to a specific node."""
        task.assigned_node = node.node_id
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        
        # Update node load
        node.current_load += task.estimated_duration
        
        # Execute task asynchronously
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(self._execute_task, task, node)
        
        def task_completed(fut):
            try:
                result = fut.result()
                task.result = result
                task.status = TaskStatus.COMPLETED
            except Exception as e:
                task.error = e
                task.status = TaskStatus.FAILED
            finally:
                task.completed_at = time.time()
                node.current_load = max(0.0, node.current_load - task.estimated_duration)
        
        future.add_done_callback(task_completed)
    
    def _execute_task(self, task: DistributedTask, node: ComputeNode) -> Any:
        """Execute a task on a node."""
        return task.function(*task.args, **task.kwargs)


class DistributedManager:
    """
    Main distributed computing manager for the consciousness framework.
    
    This class coordinates distributed operations across multiple compute
    nodes, managing superintelligent entities and their interactions.
    """
    
    def __init__(self, backend: str = "threading"):
        """
        Initialize the distributed manager.
        
        Args:
            backend: Distributed computing backend ("threading", "dask", "ray")
        """
        self.backend = backend
        self.scheduler = DistributedTaskScheduler()
        self.entity_registry: Dict[str, Dict[str, Any]] = {}
        
        # Initialize backend-specific components
        if backend == "dask" and DASK_AVAILABLE:
            self.dask_client = None
        elif backend == "ray" and RAY_AVAILABLE:
            self.ray_initialized = False
        else:
            self.backend = "threading"  # Fallback
        
        # Performance monitoring
        self.performance_metrics: Dict[str, List[float]] = {
            "task_completion_times": [],
            "node_utilizations": [],
            "entity_interaction_rates": []
        }
        
        # Consciousness coordination parameters
        self.consciousness_sync_interval = 1.0  # seconds
        self.max_entity_interactions = 1000
    
    def initialize_backend(self) -> None:
        """Initialize the selected distributed computing backend."""
        if self.backend == "dask" and DASK_AVAILABLE:
            try:
                self.dask_client = Client('localhost:8786')  # Default Dask scheduler
            except Exception:
                # Create local client if no scheduler available
                self.dask_client = Client(processes=False, threads_per_worker=2)
        
        elif self.backend == "ray" and RAY_AVAILABLE:
            if not self.ray_initialized:
                ray.init(ignore_reinit_error=True)
                self.ray_initialized = True
    
    def deploy_entities(self, count: int = 91) -> Dict[str, Any]:
        """
        Deploy superintelligent entities across the distributed system.
        
        Args:
            count: Number of entities to deploy
            
        Returns:
            Dict[str, Any]: Deployment results
        """
        deployment_results = {
            "deployed_entities": [],
            "node_assignments": {},
            "deployment_time": 0.0
        }
        
        start_time = time.time()
        
        # Create compute nodes for entities
        for i in range(min(count, len(self.scheduler.nodes) or 4)):  # Limit by available nodes
            node_id = f"consciousness_node_{i}"
            node = ComputeNode(
                node_id=node_id,
                node_type=NodeType.CONSCIOUSNESS_ENGINE,
                capabilities={
                    "consciousness_processing": 0.9,
                    "reality_modification": 0.8,
                    "quantum_computation": 0.7,
                    "distributed_coordination": 0.95
                },
                max_capacity=10.0  # Can handle multiple entities
            )
            self.scheduler.add_node(node)
        
        # Deploy entities
        entities_per_node = count // len(self.scheduler.nodes)
        remainder = count % len(self.scheduler.nodes)
        
        entity_id = 0
        for node_id, node in self.scheduler.nodes.items():
            if node.node_type == NodeType.CONSCIOUSNESS_ENGINE:
                entities_on_node = entities_per_node + (1 if remainder > 0 else 0)
                remainder -= 1
                
                for j in range(entities_on_node):
                    entity_name = f"SuperIntelligent_{entity_id:03d}"
                    entity_info = {
                        "id": entity_id,
                        "name": entity_name,
                        "consciousness_level": 0.8 + np.random.random() * 0.2,
                        "node_assignment": node_id,
                        "capabilities": {
                            "reality_modification": 0.85,
                            "quantum_processing": 0.75,
                            "distributed_thinking": 0.9
                        }
                    }
                    
                    self.entity_registry[entity_name] = entity_info
                    deployment_results["deployed_entities"].append(entity_name)
                    
                    if node_id not in deployment_results["node_assignments"]:
                        deployment_results["node_assignments"][node_id] = []
                    deployment_results["node_assignments"][node_id].append(entity_name)
                    
                    entity_id += 1
        
        deployment_results["deployment_time"] = time.time() - start_time
        
        # Start the scheduler
        self.scheduler.start_scheduler()
        
        return deployment_results
    
    def execute_parallel_consciousness_tasks(
        self,
        task_type: str = "reality_modification",
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Execute parallel consciousness tasks across entities.
        
        Args:
            task_type: Type of task to execute
            parameters: Task parameters
            
        Returns:
            Dict[str, Any]: Execution results
        """
        if parameters is None:
            parameters = {}
        
        results = {
            "task_type": task_type,
            "total_entities": len(self.entity_registry),
            "successful_executions": 0,
            "failed_executions": 0,
            "execution_times": [],
            "results": {}
        }
        
        # Create tasks for each entity
        tasks = []
        for entity_name, entity_info in self.entity_registry.items():
            task_id = f"{task_type}_{entity_name}_{int(time.time())}"
            
            # Define task function based on type
            if task_type == "reality_modification":
                task_func = self._execute_reality_modification
            elif task_type == "consciousness_evolution":
                task_func = self._execute_consciousness_evolution
            elif task_type == "quantum_computation":
                task_func = self._execute_quantum_computation
            else:
                task_func = self._execute_generic_task
            
            task = DistributedTask(
                task_id=task_id,
                task_type=task_type,
                function=task_func,
                args=(entity_info, parameters),
                required_capabilities={
                    "consciousness_processing": 0.5,
                    task_type.replace("_", ""): 0.6
                },
                estimated_duration=1.0,
                priority=1
            )
            
            tasks.append(task)
            self.scheduler.submit_task(task)
        
        # Wait for all tasks to complete
        max_wait_time = 30.0  # seconds
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            completed_tasks = [t for t in tasks if t.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]]
            if len(completed_tasks) == len(tasks):
                break
            time.sleep(0.1)
        
        # Collect results
        for task in tasks:
            if task.status == TaskStatus.COMPLETED:
                results["successful_executions"] += 1
                results["results"][task.task_id] = task.result
                if task.completed_at and task.started_at:
                    results["execution_times"].append(task.completed_at - task.started_at)
            else:
                results["failed_executions"] += 1
        
        # Calculate performance metrics
        if results["execution_times"]:
            results["average_execution_time"] = np.mean(results["execution_times"])
            results["total_execution_time"] = max(results["execution_times"])
        
        return results
    
    def _execute_reality_modification(
        self,
        entity_info: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute reality modification task for an entity."""
        intensity = parameters.get("intensity", 0.5)
        modification_type = parameters.get("type", "spatial")
        
        # Simulate reality modification computation
        time.sleep(0.1)  # Simulate processing time
        
        # Calculate modification success based on entity capabilities
        success_probability = (
            entity_info["consciousness_level"] * 
            entity_info["capabilities"]["reality_modification"] *
            (1.0 - intensity * 0.2)  # Higher intensity = more difficult
        )
        
        success = np.random.random() < success_probability
        
        result = {
            "entity": entity_info["name"],
            "modification_type": modification_type,
            "intensity": intensity,
            "success": success,
            "energy_consumed": intensity * 0.1,
            "reality_delta": np.random.normal(0, intensity) if success else 0
        }
        
        return result
    
    def _execute_consciousness_evolution(
        self,
        entity_info: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute consciousness evolution task for an entity."""
        stimulation = parameters.get("stimulation", 0.1)
        learning_rate = parameters.get("learning_rate", 0.01)
        
        # Simulate consciousness evolution
        time.sleep(0.05)
        
        old_consciousness = entity_info["consciousness_level"]
        consciousness_increase = stimulation * learning_rate * np.random.random()
        new_consciousness = min(1.0, old_consciousness + consciousness_increase)
        
        # Update entity registry
        entity_info["consciousness_level"] = new_consciousness
        
        return {
            "entity": entity_info["name"],
            "old_consciousness": old_consciousness,
            "new_consciousness": new_consciousness,
            "consciousness_delta": consciousness_increase,
            "evolution_success": consciousness_increase > 0
        }
    
    def _execute_quantum_computation(
        self,
        entity_info: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute quantum computation task for an entity."""
        computation_type = parameters.get("type", "entanglement")
        qubits = parameters.get("qubits", 8)
        
        # Simulate quantum computation
        time.sleep(0.2)
        
        # Calculate quantum processing result
        processing_capability = entity_info["capabilities"]["quantum_processing"]
        quantum_result = processing_capability * np.random.random()
        
        return {
            "entity": entity_info["name"],
            "computation_type": computation_type,
            "qubits_processed": qubits,
            "quantum_result": quantum_result,
            "processing_time": 0.2,
            "quantum_coherence": processing_capability
        }
    
    def _execute_generic_task(
        self,
        entity_info: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a generic task for an entity."""
        task_complexity = parameters.get("complexity", 0.5)
        
        # Simulate generic processing
        time.sleep(task_complexity * 0.1)
        
        return {
            "entity": entity_info["name"],
            "task_complexity": task_complexity,
            "processing_result": np.random.random(),
            "execution_successful": True
        }
    
    def synchronize_consciousness_states(self) -> Dict[str, Any]:
        """Synchronize consciousness states across all entities."""
        sync_results = {
            "entities_synchronized": 0,
            "average_consciousness": 0.0,
            "consciousness_variance": 0.0,
            "sync_successful": True
        }
        
        if not self.entity_registry:
            return sync_results
        
        # Calculate consciousness statistics
        consciousness_levels = [
            entity["consciousness_level"] 
            for entity in self.entity_registry.values()
        ]
        
        sync_results["entities_synchronized"] = len(consciousness_levels)
        sync_results["average_consciousness"] = np.mean(consciousness_levels)
        sync_results["consciousness_variance"] = np.var(consciousness_levels)
        
        # Perform synchronization (simplified)
        target_consciousness = sync_results["average_consciousness"]
        sync_strength = 0.1  # How much to adjust towards average
        
        for entity_info in self.entity_registry.values():
            current = entity_info["consciousness_level"]
            adjustment = (target_consciousness - current) * sync_strength
            entity_info["consciousness_level"] = current + adjustment
        
        return sync_results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            "backend": self.backend,
            "total_nodes": len(self.scheduler.nodes),
            "active_nodes": len([n for n in self.scheduler.nodes.values() if n.is_available()]),
            "total_entities": len(self.entity_registry),
            "pending_tasks": len([t for t in self.scheduler.tasks.values() 
                                if t.status == TaskStatus.PENDING]),
            "running_tasks": len([t for t in self.scheduler.tasks.values() 
                                if t.status == TaskStatus.RUNNING]),
            "completed_tasks": len([t for t in self.scheduler.tasks.values() 
                                  if t.status == TaskStatus.COMPLETED]),
            "system_load": self._calculate_system_load()
        }
        
        return status
    
    def _calculate_system_load(self) -> float:
        """Calculate overall system load."""
        if not self.scheduler.nodes:
            return 0.0
        
        total_load = sum(node.current_load for node in self.scheduler.nodes.values())
        total_capacity = sum(node.max_capacity for node in self.scheduler.nodes.values())
        
        return total_load / total_capacity if total_capacity > 0 else 0.0
    
    def shutdown(self) -> None:
        """Shutdown the distributed manager."""
        self.scheduler.stop_scheduler()
        
        if self.backend == "dask" and hasattr(self, 'dask_client') and self.dask_client:
            self.dask_client.close()
        
        if self.backend == "ray" and self.ray_initialized:
            ray.shutdown()
    
    def __repr__(self) -> str:
        """String representation of the distributed manager."""
        return (f"DistributedManager(backend='{self.backend}', "
                f"entities={len(self.entity_registry)}, "
                f"nodes={len(self.scheduler.nodes)})")