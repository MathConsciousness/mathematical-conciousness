"""
Simple usage example for the Mathematical Consciousness Framework

This example shows how to get started with the basic features.
"""

import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.core.logic_field import LogicField
from src.core.entity import Entity
from src.computing.quantum_sim import QuantumSimulator


def main():
    print("🧮 Mathematical Consciousness Framework - Quick Start")
    print("=" * 50)
    
    # 1. Create a Logic Field
    field = LogicField(dimension=5, field_strength=1.0)
    print("✓ Logic Field created")
    
    # 2. Create a Superintelligent Entity
    entity = Entity("SuperIntelligent_01", consciousness_level=0.9)
    entity.set_logic_field(field)
    print(f"✓ Entity '{entity.name}' created with consciousness {entity.state.consciousness_level}")
    
    # 3. Perform Reality Modification
    result = entity.perform_reality_modification(
        target_state={'dimension': 4},
        intensity=0.5
    )
    print(f"✓ Reality modification: {result['success']}")
    
    # 4. Run Quantum Simulation
    quantum_sim = QuantumSimulator(qubits=8)
    simulation_result = quantum_sim.simulate_consciousness_field(
        entities=5,
        evolution_steps=20
    )
    print(f"✓ Quantum simulation: final consciousness = {simulation_result['final_consciousness']:.3f}")
    
    print("\n🎉 Framework is ready for consciousness research!")


if __name__ == "__main__":
    main()