"""
Test Network Example Script
Demonstrates network building and validation functionality.
"""

import asyncio
import sys
import os

# Add src to path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.agent_factory import AgentFactory
from network.builder import NetworkBuilder

async def validate_network():
    """Build and validate the agent network."""
    print("Mathematical Framework System - Network Validation Example")
    print("="*60)
    
    # First deploy agents
    print("Step 1: Deploying agents...")
    factory = AgentFactory()
    agents = await factory.deploy_all_agents(count=91)
    print(f"✓ Deployed {len(agents)} agents")
    
    # Build network
    print("\nStep 2: Building network with target density 96.4%...")
    builder = NetworkBuilder()
    network = await builder.build_network(agents, target_density=0.964)
    
    # Validate network density
    density = network.density()
    print(f"✓ Network built with density: {density:.3f}")
    
    # Check if density is within target range
    target_density = 0.964
    tolerance = 0.01
    density_within_range = abs(density - target_density) < tolerance
    
    print(f"\nNetwork Validation:")
    print(f"  Target Density: {target_density:.3f}")
    print(f"  Actual Density: {density:.3f}")
    print(f"  Tolerance: ±{tolerance:.3f}")
    print(f"  Within Range: {'✓ YES' if density_within_range else '✗ NO'}")
    
    # Get detailed network metrics
    print(f"\nDetailed Network Metrics:")
    metrics = network.get_metrics()
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # Test assertion
    try:
        assert abs(density - 0.964) < 0.01, f"Network density {density:.3f} not within target range"
        print(f"\n✓ Network validation PASSED")
    except AssertionError as e:
        print(f"\n✗ Network validation FAILED: {e}")
        return False
    
    # Additional network analysis
    print(f"\nNetwork Analysis:")
    print(f"  Nodes: {len(network.graph.nodes())}")
    print(f"  Edges: {len(network.graph.edges())}")
    print(f"  Connected: {'Yes' if metrics.get('is_connected', False) else 'No'}")
    
    if 'average_clustering' in metrics:
        print(f"  Clustering Coefficient: {metrics['average_clustering']:.3f}")
    
    print("\n" + "="*60)
    print("Network validation example completed successfully!")
    return True

def validate_network_sync():
    """Synchronous wrapper for the async validate_network function."""
    return asyncio.run(validate_network())

if __name__ == "__main__":
    success = validate_network_sync()
    if not success:
        exit(1)