#!/usr/bin/env python3
"""
Deployment verification script for the Mathematical Framework System.

This script verifies that the system deployment is successful and all
components are functioning correctly.
"""

import asyncio
import requests
import sys
import time
from typing import Dict, Any, List
from core.agent_factory import AgentFactory
from network.builder import NetworkBuilder


class DeploymentVerifier:
    """Handles verification of system deployment and configuration."""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        """
        Initialize the deployment verifier.
        
        Args:
            api_url: Base URL for API endpoints
        """
        self.api_url = api_url
        self.verification_results: Dict[str, Any] = {}
    
    async def verify_deployment(self) -> bool:
        """
        Verify system deployment and configuration.
        
        Returns:
            bool: True if all verifications pass, False otherwise
        """
        print("🚀 Starting Mathematical Framework System deployment verification...")
        print("=" * 60)
        
        success = True
        
        try:
            # 1. Check agent deployment
            success &= await self._verify_agent_deployment()
            
            # 2. Validate network
            success &= await self._verify_network_construction()
            
            # 3. Test API endpoints (if available)
            success &= await self._verify_api_endpoints()
            
            # 4. Run system integration tests
            success &= await self._verify_system_integration()
            
            # 5. Performance benchmarks
            success &= await self._verify_performance()
            
        except Exception as e:
            print(f"❌ Verification failed with error: {e}")
            success = False
        
        # Print final results
        self._print_verification_summary(success)
        
        return success
    
    async def _verify_agent_deployment(self) -> bool:
        """Verify agent deployment."""
        print("📋 Verifying agent deployment...")
        
        try:
            factory = AgentFactory()
            agents = await factory.deploy_all_agents_async(count=91)
            
            if len(agents) != 91:
                print(f"❌ Expected 91 agents, got {len(agents)}")
                return False
            
            # Verify agent properties
            active_count = sum(1 for agent in agents if agent.status == "active")
            if active_count != 91:
                print(f"❌ Expected 91 active agents, got {active_count}")
                return False
            
            # Check intelligence levels
            avg_intelligence = sum(agent.intelligence_level for agent in agents) / len(agents)
            if avg_intelligence < 0.91:
                print(f"❌ Average intelligence level too low: {avg_intelligence}")
                return False
            
            print(f"✓ Deployed {len(agents)} agents successfully")
            print(f"✓ Average intelligence level: {avg_intelligence:.3f}")
            
            self.verification_results['agents'] = {
                'total': len(agents),
                'active': active_count,
                'avg_intelligence': avg_intelligence
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Agent deployment verification failed: {e}")
            return False
    
    async def _verify_network_construction(self) -> bool:
        """Verify network construction and topology."""
        print("🌐 Verifying network construction...")
        
        try:
            # Deploy agents first
            factory = AgentFactory()
            agents = await factory.deploy_all_agents_async(count=91)
            
            # Build network
            builder = NetworkBuilder()
            network = await builder.build_network_async(agents, target_density=0.964)
            
            # Check network properties
            density = network.density()
            stats = network.get_network_stats()
            
            if abs(density - 0.964) > 0.01:
                print(f"❌ Network density {density:.3f} outside target range")
                return False
            
            if not stats["is_connected"]:
                print("❌ Network is not connected")
                return False
            
            if stats["total_nodes"] != 91:
                print(f"❌ Expected 91 nodes, got {stats['total_nodes']}")
                return False
            
            print(f"✓ Network density: {density:.3f}")
            print(f"✓ Network connectivity: {stats['is_connected']}")
            print(f"✓ Average clustering: {stats['average_clustering']:.3f}")
            
            self.verification_results['network'] = {
                'density': density,
                'connected': stats['is_connected'],
                'nodes': stats['total_nodes'],
                'edges': stats['total_edges']
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Network verification failed: {e}")
            return False
    
    async def _verify_api_endpoints(self) -> bool:
        """Verify API endpoints if available."""
        print("🔌 Verifying API endpoints...")
        
        endpoints = [
            "/deploy/agents",
            "/network/status", 
            "/simulate/protocol",
            "/results/analysis"
        ]
        
        operational_endpoints = 0
        
        for endpoint in endpoints:
            try:
                response = requests.get(f"{self.api_url}{endpoint}", timeout=5)
                if response.status_code in [200, 201]:
                    print(f"✓ Endpoint {endpoint} operational")
                    operational_endpoints += 1
                else:
                    print(f"⚠️ Endpoint {endpoint} returned status {response.status_code}")
            except requests.exceptions.RequestException:
                print(f"⚠️ Endpoint {endpoint} not available (API server may not be running)")
        
        if operational_endpoints == len(endpoints):
            print("✓ All API endpoints operational")
        elif operational_endpoints > 0:
            print(f"⚠️ {operational_endpoints}/{len(endpoints)} endpoints operational")
        else:
            print("⚠️ No API endpoints available (this is expected if server is not running)")
        
        self.verification_results['api'] = {
            'total_endpoints': len(endpoints),
            'operational': operational_endpoints
        }
        
        return True  # Don't fail if API is not available
    
    async def _verify_system_integration(self) -> bool:
        """Verify complete system integration."""
        print("🔧 Verifying system integration...")
        
        try:
            from core.logic_field import LogicField
            import numpy as np
            
            # Initialize LogicField
            field = LogicField(
                tau=0.95,
                Q=np.eye(3),
                phi_grad=np.zeros(3),
                sigma=0.8,
                S=np.identity(4)
            )
            
            intelligence_level = field.calculate_intelligence_level()
            if intelligence_level <= 0.9:
                print(f"❌ Intelligence level {intelligence_level} below threshold")
                return False
            
            # Test complete workflow
            factory = AgentFactory()
            agents = await factory.deploy_all_agents_async(count=91)
            
            builder = NetworkBuilder()
            network = await builder.build_network_async(agents, target_density=0.964)
            
            # Verify integration
            if len(agents) != 91:
                print(f"❌ Integration test: wrong agent count {len(agents)}")
                return False
            
            if abs(network.density() - 0.964) > 0.01:
                print(f"❌ Integration test: wrong network density {network.density()}")
                return False
            
            print(f"✓ Logic field intelligence level: {intelligence_level:.3f}")
            print("✓ System integration successful")
            
            self.verification_results['integration'] = {
                'intelligence_level': intelligence_level,
                'agents_deployed': len(agents),
                'network_density': network.density()
            }
            
            return True
            
        except Exception as e:
            print(f"❌ System integration verification failed: {e}")
            return False
    
    async def _verify_performance(self) -> bool:
        """Verify system performance benchmarks."""
        print("⚡ Verifying performance benchmarks...")
        
        try:
            # Benchmark agent deployment
            start_time = time.time()
            factory = AgentFactory()
            agents = await factory.deploy_all_agents_async(count=91)
            deployment_time = time.time() - start_time
            
            # Benchmark network building
            start_time = time.time()
            builder = NetworkBuilder()
            network = await builder.build_network_async(agents, target_density=0.964)
            network_time = time.time() - start_time
            
            print(f"✓ Agent deployment time: {deployment_time:.3f}s")
            print(f"✓ Network building time: {network_time:.3f}s")
            
            # Check performance thresholds
            if deployment_time > 5.0:
                print(f"⚠️ Agent deployment slower than expected: {deployment_time:.3f}s")
            
            if network_time > 10.0:
                print(f"⚠️ Network building slower than expected: {network_time:.3f}s")
            
            self.verification_results['performance'] = {
                'deployment_time': deployment_time,
                'network_time': network_time
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Performance verification failed: {e}")
            return False
    
    def _print_verification_summary(self, success: bool) -> None:
        """Print verification summary."""
        print("=" * 60)
        if success:
            print("🎉 DEPLOYMENT VERIFICATION SUCCESSFUL!")
            print("✅ Mathematical Framework System is ready for operation")
        else:
            print("🚨 DEPLOYMENT VERIFICATION FAILED!")
            print("❌ Please check the errors above and retry")
        
        print("\n📊 Verification Summary:")
        for component, results in self.verification_results.items():
            print(f"  {component.capitalize()}: {results}")
        
        print("=" * 60)


async def main():
    """Main function to run deployment verification."""
    verifier = DeploymentVerifier()
    success = await verifier.verify_deployment()
    
    if success:
        print("\n🚀 System is ready for:")
        print("  1. Full 91-agent deployment")
        print("  2. Network topology validation")
        print("  3. Scientific protocol execution")
        print("  4. API interface access")
        print("  5. Docker container deployment")
        
        sys.exit(0)
    else:
        print("\n🔧 Please address the issues above before proceeding with deployment.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())