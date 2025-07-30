"""Example usage of Mathematical Consciousness Framework

Demonstrates the complete system capabilities including agent creation,
network building, and scientific computing.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import asyncio
from typing import List

from src.core.logic_field import LogicField
from src.core.agent_factory import AgentFactory, AgentType
from src.network.builder import NetworkBuilder
from src.computing.protocols import ScientificProtocols


async def demonstrate_framework():
    """Demonstrate the mathematical consciousness framework"""
    print("=== Mathematical Consciousness Framework Demo ===\n")
    
    # 1. Create LogicField instances
    print("1. Creating LogicField instances...")
    
    # Create a few logic fields with different parameters
    tau = 1.5
    Q = np.eye(3) + 0.1 * np.random.random((3, 3))
    Q = Q @ Q.T  # Ensure positive definite
    phi_grad = np.array([0.1, -0.2, 0.3])
    sigma = 0.8
    S = np.random.random((3, 3))
    
    field = LogicField(tau, Q, phi_grad, sigma, S)
    
    print(f"   Intelligence Level: {field.calculate_intelligence_level():.3f}")
    print(f"   Coherence: {field.compute_coherence():.3f}")
    print(f"   Field Energy: {field.field_energy():.3f}")
    print()
    
    # 2. Create agents using AgentFactory
    print("2. Creating computational agents...")
    
    factory = AgentFactory()
    
    # Create different types of agents
    agents = []
    
    # Mathematical agents
    for i in range(3):
        agent = factory.create_mathematical_agent({
            "tau": 1.0 + i * 0.3,
            "field_dimension": 3,
            "matrix_size": 3,
            "coherence": 0.7 + i * 0.1,
        })
        agents.append(agent)
        print(f"   Created mathematical agent {agent.id[:8]}...")
    
    # Scientific agents
    for i in range(2):
        agent = factory.create_scientific_agent({
            "tau": 1.5 + i * 0.2,
            "field_dimension": 4,
            "enable_quantum": True,
        })
        agents.append(agent)
        print(f"   Created scientific agent {agent.id[:8]}...")
    
    print(f"   Total agents created: {len(agents)}")
    print()
    
    # 3. Build agent network
    print("3. Building agent network...")
    
    builder = NetworkBuilder(random_seed=42)
    network = builder.build_network(agents, target_density=0.6)
    
    print(f"   Network nodes: {network.number_of_nodes()}")
    print(f"   Network edges: {network.number_of_edges()}")
    print(f"   Network density: {network.number_of_edges() / (network.number_of_nodes() * (network.number_of_nodes() - 1) / 2):.3f}")
    
    # Optimize network
    print("   Optimizing network information flow...")
    builder.optimize_information_flow(network)
    
    metrics = builder.get_network_metrics(network)
    print(f"   Network efficiency: {metrics.efficiency:.3f}")
    print(f"   Connectivity score: {metrics.connectivity_score:.3f}")
    print()
    
    # 4. Demonstrate agent operations
    print("4. Demonstrating agent operations...")
    
    for i, agent in enumerate(agents[:3]):  # Test first 3 agents
        print(f"   Agent {i+1} ({agent.agent_type.value}):")
        
        # Field analysis
        field_result = agent.execute_operation("analyze_field")
        print(f"      Intelligence: {field_result['intelligence_level']:.3f}")
        print(f"      Coherence: {field_result['coherence']:.3f}")
        
        # Matrix operations
        test_matrix = np.random.random((2, 2))
        eigenvals = agent.execute_operation("matrix_operations", operation="eigenvalues", matrix=test_matrix)
        print(f"      Matrix eigenvalues: [{eigenvals[0]:.3f}, {eigenvals[1]:.3f}]")
        
        print()
    
    # 5. Scientific computing protocols
    print("5. Scientific computing demonstrations...")
    
    protocols = ScientificProtocols(random_seed=42)
    
    # Quantum simulation
    print("   Quantum Simulation:")
    hamiltonian = np.array([[1, 0.5], [0.5, -1]], dtype=np.complex128)
    initial_state = np.array([1, 0], dtype=np.complex128)
    time_steps = np.linspace(0, 1, 20)
    
    quantum_params = {
        "hamiltonian": hamiltonian,
        "initial_state": initial_state,
        "time_steps": time_steps,
        "method": "exact"
    }
    
    quantum_result = protocols.quantum_simulation(quantum_params)
    print(f"      Simulated {len(time_steps)} time steps")
    print(f"      Final state norm: {np.linalg.norm(quantum_result[-1]):.6f}")
    
    # Temporal analysis
    print("   Temporal Pattern Analysis:")
    t = np.linspace(0, 4*np.pi, 100)
    signal_data = np.sin(2*t) + 0.5*np.cos(5*t) + 0.1*np.random.normal(size=len(t))
    
    temporal_results = protocols.analyze_temporal_patterns(signal_data)
    freq_analysis = temporal_results["frequency_analysis"]["series_0"]
    print(f"      Dominant frequency: {freq_analysis['dominant_frequency']:.3f}")
    print(f"      Signal patterns detected: {len(temporal_results['patterns'])}")
    
    # Parameter optimization
    print("   Parameter Optimization:")
    def test_objective(params):
        x, y = params
        return (x - 1)**2 + (y - 2)**2
    
    opt_result = protocols.optimize_system_parameters(
        test_objective, 
        np.array([0.0, 0.0]),
        method="BFGS"
    )
    
    if opt_result["success"]:
        optimal = opt_result["optimal_params"]
        print(f"      Found optimum at: ({optimal[0]:.3f}, {optimal[1]:.3f})")
        print(f"      Optimal value: {opt_result['optimal_value']:.6f}")
    
    print()
    
    # 6. System statistics
    print("6. System Statistics:")
    
    agent_stats = factory.get_agent_statistics()
    print(f"   Total agents: {agent_stats['total_agents']}")
    print(f"   Active agents: {agent_stats['active_agents']}")
    print(f"   Agent types: {agent_stats['agent_types']}")
    print(f"   Average processing load: {agent_stats['average_processing_load']:.3f}")
    
    network_status = builder.get_network_status("network_0")
    health_score = network_status["health_score"]
    print(f"   Network health score: {health_score:.3f}")
    
    print("\n=== Demo Complete ===")


def demonstrate_api_integration():
    """Demonstrate API integration (without starting server)"""
    print("\n=== API Integration Demo ===")
    
    # Show how the system would work with the API
    from src.api.endpoints import AgentConfig, NetworkConfig
    
    # Example API configurations
    agent_config = AgentConfig(
        agent_type="mathematical",
        tau=1.5,
        field_dimension=4,
        matrix_size=3,
        coherence=0.9,
        count=5
    )
    
    network_config = NetworkConfig(
        target_density=0.7,
        optimization_enabled=True
    )
    
    print("   Example Agent Configuration:")
    print(f"      Type: {agent_config.agent_type}")
    print(f"      Intelligence (τ): {agent_config.tau}")
    print(f"      Field dimension: {agent_config.field_dimension}")
    print(f"      Coherence: {agent_config.coherence}")
    print(f"      Count: {agent_config.count}")
    
    print("   Example Network Configuration:")
    print(f"      Target density: {network_config.target_density}")
    print(f"      Optimization: {network_config.optimization_enabled}")
    
    print("   API Endpoints Available:")
    print("      POST /api/v1/deploy/agents - Deploy computational agents")
    print("      GET  /api/v1/agents - List all agents")
    print("      POST /api/v1/network/create - Create agent network")
    print("      GET  /api/v1/network/status - Get network status")
    print("      POST /api/v1/compute/quantum-simulation - Run quantum simulation")
    print("      POST /api/v1/compute/temporal-analysis - Analyze temporal patterns")
    print("      GET  /api/v1/system/status - Get system status")


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(demonstrate_framework())
    
    # Show API integration
    demonstrate_api_integration()
    
    print("\nTo start the API server, run:")
    print("python main.py")
    print("\nThen visit http://localhost:8000/docs for interactive API documentation.")