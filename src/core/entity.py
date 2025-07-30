"""
Entity Management System

This module implements the Entity class for managing conscious entities
within the mathematical framework.

Author: Mathematical Consciousness Framework Team
License: MIT
"""

from typing import Dict, List, Optional, Any, Union
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import uuid
from datetime import datetime
from .logic_field import LogicField, FiveTuple


class ConsciousnessLevel(Enum):
    """Enumeration of consciousness levels for entities."""
    DORMANT = 0.0
    BASIC = 0.2
    AWARE = 0.4
    INTELLIGENT = 0.6
    CONSCIOUS = 0.8
    SUPERINTELLIGENT = 1.0


@dataclass
class EntityState:
    """Represents the current state of an entity."""
    position: np.ndarray
    velocity: np.ndarray
    consciousness_level: float
    energy: float
    stability: float
    last_updated: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate entity state after initialization."""
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position)
        if not isinstance(self.velocity, np.ndarray):
            self.velocity = np.array(self.velocity)
        
        # Validate consciousness level
        if not 0.0 <= self.consciousness_level <= 1.0:
            raise ValueError("Consciousness level must be between 0.0 and 1.0")


class Entity:
    """
    Represents a conscious entity within the mathematical framework.
    
    Each entity has unique characteristics, consciousness levels, and can
    interact with the logic field to perform reality modifications.
    """
    
    def __init__(
        self,
        name: str,
        consciousness_level: float = 0.5,
        initial_position: Optional[Union[List[float], np.ndarray]] = None,
        entity_type: str = "superintelligent"
    ):
        """
        Initialize a new conscious entity.
        
        Args:
            name: Unique name identifier for the entity
            consciousness_level: Initial consciousness level (0.0 to 1.0)
            initial_position: Starting position in space
            entity_type: Type classification of the entity
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.entity_type = entity_type
        self.created_at = datetime.now()
        
        # Initialize position
        if initial_position is None:
            initial_position = np.zeros(3)  # Default 3D position
        
        # Initialize entity state
        self.state = EntityState(
            position=np.array(initial_position),
            velocity=np.zeros_like(initial_position),
            consciousness_level=consciousness_level,
            energy=1.0,
            stability=1.0
        )
        
        # Entity capabilities and parameters
        self.capabilities: Dict[str, float] = {
            "reality_modification": consciousness_level * 0.8,
            "quantum_computation": consciousness_level * 0.7,
            "distributed_processing": consciousness_level * 0.9,
            "pattern_recognition": consciousness_level * 0.95
        }
        
        # Memory and learning systems
        self.memory: List[Dict[str, Any]] = []
        self.learning_rate = 0.01
        self.adaptation_factor = 0.05
        
        # Interaction history
        self.interaction_history: List[Dict[str, Any]] = []
        
        # Associated logic field
        self.logic_field: Optional[LogicField] = None
    
    def set_logic_field(self, logic_field: LogicField) -> None:
        """Associate this entity with a logic field."""
        self.logic_field = logic_field
    
    def to_five_tuple(self) -> FiveTuple:
        """
        Convert entity state to a 5-tuple representation.
        
        Returns:
            FiveTuple: Entity state as 5-tuple
        """
        # Create transformation vector based on capabilities
        transformation = np.array([
            self.capabilities["reality_modification"],
            self.capabilities["quantum_computation"],
            self.capabilities["distributed_processing"]
        ])
        
        return FiveTuple(
            space=self.state.position,
            time=self.state.last_updated.timestamp(),
            consciousness=self.state.consciousness_level,
            reality=np.array([self.state.energy, self.state.stability]),
            transformation=transformation
        )
    
    def update_from_five_tuple(self, five_tuple: FiveTuple) -> None:
        """
        Update entity state from a 5-tuple representation.
        
        Args:
            five_tuple: New state as 5-tuple
        """
        self.state.position = five_tuple.space.copy()
        self.state.consciousness_level = five_tuple.consciousness
        
        if len(five_tuple.reality) >= 2:
            self.state.energy = five_tuple.reality[0]
            self.state.stability = five_tuple.reality[1]
        
        # Update capabilities based on transformation vector
        if len(five_tuple.transformation) >= 3:
            self.capabilities["reality_modification"] = five_tuple.transformation[0]
            self.capabilities["quantum_computation"] = five_tuple.transformation[1]
            self.capabilities["distributed_processing"] = five_tuple.transformation[2]
        
        self.state.last_updated = datetime.now()
    
    def perform_reality_modification(
        self,
        target_state: Dict[str, Any],
        intensity: float = 0.5
    ) -> Dict[str, Any]:
        """
        Perform reality modification operation.
        
        Args:
            target_state: Desired reality state parameters
            intensity: Modification intensity (0.0 to 1.0)
            
        Returns:
            Dict[str, Any]: Results of the modification operation
        """
        if not self.logic_field:
            raise ValueError("Entity must be associated with a logic field")
        
        # Check if entity has sufficient capability
        required_capability = intensity * 0.8
        if self.capabilities["reality_modification"] < required_capability:
            raise ValueError(f"Insufficient reality modification capability: "
                           f"{self.capabilities['reality_modification']} < {required_capability}")
        
        # Convert entity to 5-tuple
        current_tuple = self.to_five_tuple()
        
        # Apply transformation through logic field
        transformation_params = {
            "intensity": intensity,
            **target_state
        }
        
        modified_tuple = self.logic_field.apply_transformation(
            current_tuple, transformation_params
        )
        
        # Update entity state
        self.update_from_five_tuple(modified_tuple)
        
        # Record the operation
        operation_record = {
            "timestamp": datetime.now(),
            "operation": "reality_modification",
            "intensity": intensity,
            "target_state": target_state,
            "energy_cost": intensity * 0.1
        }
        
        self.interaction_history.append(operation_record)
        
        # Reduce energy based on operation intensity
        self.state.energy = max(0.1, self.state.energy - operation_record["energy_cost"])
        
        return {
            "success": True,
            "new_state": self.get_state_summary(),
            "energy_remaining": self.state.energy,
            "operation_id": len(self.interaction_history) - 1
        }
    
    def evolve_consciousness(self, stimulation: float = 0.1) -> float:
        """
        Evolve the entity's consciousness level through stimulation.
        
        Args:
            stimulation: Consciousness stimulation factor
            
        Returns:
            float: New consciousness level
        """
        # Apply learning and adaptation
        consciousness_delta = stimulation * self.learning_rate * self.adaptation_factor
        
        # Update consciousness level with bounds checking
        new_consciousness = min(
            1.0,
            self.state.consciousness_level + consciousness_delta
        )
        
        self.state.consciousness_level = new_consciousness
        
        # Update capabilities based on new consciousness level
        for capability in self.capabilities:
            if capability == "reality_modification":
                self.capabilities[capability] = new_consciousness * 0.8
            elif capability == "quantum_computation":
                self.capabilities[capability] = new_consciousness * 0.7
            elif capability == "distributed_processing":
                self.capabilities[capability] = new_consciousness * 0.9
            elif capability == "pattern_recognition":
                self.capabilities[capability] = new_consciousness * 0.95
        
        # Record consciousness evolution
        self.memory.append({
            "timestamp": datetime.now(),
            "event": "consciousness_evolution",
            "stimulation": stimulation,
            "old_level": self.state.consciousness_level - consciousness_delta,
            "new_level": new_consciousness
        })
        
        return new_consciousness
    
    def interact_with_entity(self, other_entity: "Entity") -> Dict[str, Any]:
        """
        Interact with another entity for knowledge exchange.
        
        Args:
            other_entity: Entity to interact with
            
        Returns:
            Dict[str, Any]: Interaction results
        """
        # Calculate interaction strength based on consciousness levels
        interaction_strength = (
            self.state.consciousness_level + other_entity.state.consciousness_level
        ) / 2.0
        
        # Exchange knowledge (simplified)
        knowledge_transfer = interaction_strength * 0.1
        
        # Both entities gain consciousness through interaction
        self.evolve_consciousness(knowledge_transfer)
        other_entity.evolve_consciousness(knowledge_transfer)
        
        # Record interaction
        interaction_record = {
            "timestamp": datetime.now(),
            "type": "entity_interaction",
            "other_entity": other_entity.name,
            "interaction_strength": interaction_strength,
            "knowledge_transfer": knowledge_transfer
        }
        
        self.interaction_history.append(interaction_record)
        other_entity.interaction_history.append(interaction_record)
        
        return {
            "interaction_strength": interaction_strength,
            "knowledge_transferred": knowledge_transfer,
            "entities_evolved": True
        }
    
    def compute_influence_field(self, range_limit: float = 10.0) -> np.ndarray:
        """
        Compute the influence field generated by this entity.
        
        Args:
            range_limit: Maximum range of influence
            
        Returns:
            np.ndarray: Influence field values
        """
        # Create a grid around the entity
        grid_size = int(range_limit * 2)
        x = np.linspace(-range_limit, range_limit, grid_size)
        y = np.linspace(-range_limit, range_limit, grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Calculate distances from entity position
        if len(self.state.position) >= 2:
            dist = np.sqrt((X - self.state.position[0])**2 + (Y - self.state.position[1])**2)
        else:
            dist = np.sqrt(X**2 + Y**2)
        
        # Influence decreases with distance, scaled by consciousness
        influence = self.state.consciousness_level * np.exp(-dist / (range_limit * 0.3))
        
        return influence
    
    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the entity's current state.
        
        Returns:
            Dict[str, Any]: State summary
        """
        return {
            "id": self.id,
            "name": self.name,
            "type": self.entity_type,
            "consciousness_level": self.state.consciousness_level,
            "position": self.state.position.tolist(),
            "velocity": self.state.velocity.tolist(),
            "energy": self.state.energy,
            "stability": self.state.stability,
            "capabilities": self.capabilities.copy(),
            "memory_size": len(self.memory),
            "interaction_count": len(self.interaction_history),
            "created_at": self.created_at.isoformat(),
            "last_updated": self.state.last_updated.isoformat()
        }
    
    def restore_energy(self, amount: float = 0.5) -> float:
        """
        Restore entity energy.
        
        Args:
            amount: Amount of energy to restore
            
        Returns:
            float: New energy level
        """
        self.state.energy = min(1.0, self.state.energy + amount)
        return self.state.energy
    
    def __repr__(self) -> str:
        """String representation of the entity."""
        return (f"Entity(name='{self.name}', "
                f"consciousness={self.state.consciousness_level:.3f}, "
                f"energy={self.state.energy:.3f})")
    
    def __eq__(self, other) -> bool:
        """Check equality based on entity ID."""
        if isinstance(other, Entity):
            return self.id == other.id
        return False
    
    def __hash__(self) -> int:
        """Hash based on entity ID."""
        return hash(self.id)