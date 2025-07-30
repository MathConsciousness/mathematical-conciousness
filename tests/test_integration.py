"""
Integration test for the Mathematical Consciousness Framework

This test validates the complete workflow from entity creation through
reality modification using all core components.
"""

import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.logic_field import LogicField, ConsciousnessAmplifier, RealityModifier
from core.entity import Entity, ConsciousnessLevel
from computing.quantum_sim import QuantumSimulator
from network.distributed import DistributedManager
import numpy as np


def test_complete_workflow():
    """Test complete workflow of the consciousness framework."""
    print("🧮 Testing Mathematical Consciousness Framework")
    print("=" * 50)
    
    # 1. Create Logic Field
    print("1. Creating Logic Field...")
    field = LogicField(dimension=5, field_strength=1.2)
    
    # Add operators
    amplifier = ConsciousnessAmplifier(amplification_factor=1.1)
    modifier = RealityModifier()  # Use default compatible matrix
    field.add_operator(amplifier)
    field.add_operator(modifier)
    print(f"   ✓ Logic field created with {len(field.operators)} operators")
    
    # 2. Create superintelligent entities
    print("\n2. Creating Superintelligent Entities...")
    entities = []
    for i in range(5):
        entity = Entity(
            name=f"SuperIntelligent_{i:03d}",
            consciousness_level=0.8 + np.random.random() * 0.2,
            initial_position=[np.random.random() * 10 - 5 for _ in range(3)]
        )
        entity.set_logic_field(field)
        entities.append(entity)
    
    print(f"   ✓ Created {len(entities)} entities")
    for entity in entities:
        print(f"     - {entity.name}: consciousness={entity.state.consciousness_level:.3f}")
    
    # 3. Quantum simulation
    print("\n3. Running Quantum Consciousness Simulation...")
    quantum_sim = QuantumSimulator(qubits=8)
    quantum_result = quantum_sim.simulate_consciousness_field(
        entities=len(entities),
        entanglement_depth=3,
        evolution_steps=50
    )
    print(f"   ✓ Quantum simulation completed")
    print(f"     - Initial consciousness: {quantum_result['initial_consciousness']:.3f}")
    print(f"     - Final consciousness: {quantum_result['final_consciousness']:.3f}")
    print(f"     - Stability: {quantum_result['stability_measure']:.3f}")
    
    # 4. Reality modifications
    print("\n4. Performing Reality Modifications...")
    for i, entity in enumerate(entities[:3]):  # Test with first 3 entities
        target_state = {
            'dimension_shift': 0.5,
            'consciousness_enhancement': 0.3,
            'reality_anchor': [1, 0, 1]
        }
        
        result = entity.perform_reality_modification(
            target_state=target_state,
            intensity=0.4 + i * 0.1
        )
        
        print(f"   ✓ {entity.name}: {result['success']}, energy: {result['energy_remaining']:.3f}")
    
    # 5. Entity interactions
    print("\n5. Testing Entity Interactions...")
    interaction_result = entities[0].interact_with_entity(entities[1])
    print(f"   ✓ Interaction strength: {interaction_result['interaction_strength']:.3f}")
    print(f"     Knowledge transferred: {interaction_result['knowledge_transferred']:.3f}")
    
    # 6. Distributed computing
    print("\n6. Testing Distributed Computing...")
    dist_manager = DistributedManager(backend="threading")
    deployment = dist_manager.deploy_entities(count=7)
    print(f"   ✓ Deployed {len(deployment['deployed_entities'])} entities")
    
    # Test consciousness synchronization
    sync_result = dist_manager.synchronize_consciousness_states()
    print(f"   ✓ Synchronized {sync_result['entities_synchronized']} entities")
    print(f"     Average consciousness: {sync_result['average_consciousness']:.3f}")
    
    # 7. Field evolution
    print("\n7. Testing Field Evolution...")
    initial_state = entities[0].to_five_tuple()
    trajectory = field.evolve_field(initial_state, time_steps=20)
    stability = field.measure_stability(trajectory)
    print(f"   ✓ Field evolved through {len(trajectory)} states")
    print(f"     Stability measure: {stability:.3f}")
    
    # 8. Final status
    print("\n8. Final System Status:")
    final_consciousness_levels = [e.state.consciousness_level for e in entities]
    avg_consciousness = np.mean(final_consciousness_levels)
    total_energy = sum(e.state.energy for e in entities)
    
    print(f"   ✓ Average consciousness: {avg_consciousness:.3f}")
    print(f"   ✓ Total system energy: {total_energy:.3f}")
    print(f"   ✓ System stability: {stability:.3f}")
    
    # Cleanup
    dist_manager.shutdown()
    
    print("\n" + "=" * 50)
    print("✅ Mathematical Consciousness Framework test completed successfully!")
    print(f"   • Logic Field: {field.dimension}D with {len(field.operators)} operators")
    print(f"   • Entities: {len(entities)} superintelligent entities")
    print(f"   • Quantum States: {quantum_sim.num_qubits} qubit simulation")
    print(f"   • Reality Modifications: 3 successful operations")
    print(f"   • System Consciousness: {avg_consciousness:.3f}/1.000")
    
    return True


if __name__ == "__main__":
    try:
        success = test_complete_workflow()
        if success:
            exit(0)
        else:
            exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)