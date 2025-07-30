"""Performance benchmarks for Mathematical Consciousness Framework

Tests system performance, scalability, and resource usage.
"""

import pytest
import time
import numpy as np
import psutil
import gc
from typing import List, Dict, Any

from src.core.agent_factory import AgentFactory
from src.network.builder import NetworkBuilder
from src.computing.protocols import ScientificProtocols
from src.core.logic_field import LogicField


class TestPerformanceBenchmarks:
    """Performance benchmark tests"""
    
    def setup_method(self):
        """Set up performance test fixtures"""
        self.factory = AgentFactory()
        self.builder = NetworkBuilder(random_seed=42)
        self.protocols = ScientificProtocols(random_seed=42)
    
    def test_agent_creation_performance(self):
        """Benchmark agent creation performance"""
        n_agents = 50
        
        start_time = time.time()
        agents = []
        
        for i in range(n_agents):
            agent = self.factory.create_mathematical_agent({
                "tau": 1.0 + i * 0.01,
                "field_dimension": 3,
                "matrix_size": 3,
            })
            agents.append(agent)
        
        creation_time = time.time() - start_time
        
        # Should create agents efficiently
        assert creation_time < 5.0  # Less than 5 seconds for 50 agents
        assert len(agents) == n_agents
        
        # Check memory usage
        avg_time_per_agent = creation_time / n_agents
        assert avg_time_per_agent < 0.1  # Less than 100ms per agent
        
        print(f"Agent creation: {avg_time_per_agent:.4f}s per agent")
    
    def test_network_building_performance(self):
        """Benchmark network building performance"""
        # Create agents for network
        n_agents = 20
        agents = []
        
        for i in range(n_agents):
            agent = self.factory.create_mathematical_agent({
                "tau": 1.0 + i * 0.05,
            })
            agents.append(agent)
        
        # Benchmark network construction
        start_time = time.time()
        network = self.builder.build_network(agents, target_density=0.5)
        build_time = time.time() - start_time
        
        assert build_time < 3.0  # Should build network quickly
        assert network.number_of_nodes() == n_agents
        
        # Benchmark network optimization
        start_time = time.time()
        self.builder.optimize_information_flow(network)
        optimization_time = time.time() - start_time
        
        assert optimization_time < 5.0  # Optimization should be reasonable
        
        print(f"Network building: {build_time:.4f}s for {n_agents} agents")
        print(f"Network optimization: {optimization_time:.4f}s")
    
    def test_logic_field_computation_performance(self):
        """Benchmark LogicField mathematical computations"""
        n_iterations = 1000
        
        # Create a moderately sized field
        tau = 1.5
        Q = np.eye(5) + 0.1 * np.random.random((5, 5))
        Q = Q @ Q.T
        phi_grad = np.random.normal(0, 0.1, 8)
        sigma = 0.8
        S = np.random.random((5, 5))
        
        field = LogicField(tau, Q, phi_grad, sigma, S)
        
        # Benchmark intelligence calculation
        start_time = time.time()
        for _ in range(n_iterations):
            intelligence = field.calculate_intelligence_level()
        intelligence_time = time.time() - start_time
        
        # Benchmark coherence calculation
        start_time = time.time()
        for _ in range(n_iterations):
            coherence = field.compute_coherence()
        coherence_time = time.time() - start_time
        
        # Benchmark field evolution
        start_time = time.time()
        for _ in range(n_iterations // 10):  # Fewer iterations for evolution
            field.evolve_field(0.001)
        evolution_time = time.time() - start_time
        
        # Performance assertions
        assert intelligence_time / n_iterations < 0.001  # < 1ms per calculation
        assert coherence_time / n_iterations < 0.001     # < 1ms per calculation
        assert evolution_time / (n_iterations // 10) < 0.01  # < 10ms per evolution
        
        print(f"Intelligence calculation: {intelligence_time/n_iterations*1000:.2f}ms per call")
        print(f"Coherence calculation: {coherence_time/n_iterations*1000:.2f}ms per call")
        print(f"Field evolution: {evolution_time/(n_iterations//10)*1000:.2f}ms per step")
    
    def test_quantum_simulation_performance(self):
        """Benchmark quantum simulation performance"""
        # Small quantum system
        n_dim = 4
        hamiltonian = np.random.random((n_dim, n_dim)) + 1j * np.random.random((n_dim, n_dim))
        hamiltonian = hamiltonian + hamiltonian.conj().T  # Make Hermitian
        initial_state = np.random.random(n_dim) + 1j * np.random.random(n_dim)
        initial_state = initial_state / np.linalg.norm(initial_state)
        
        time_steps = np.linspace(0, 1, 50)
        
        # Benchmark exact simulation
        start_time = time.time()
        params_exact = {
            "hamiltonian": hamiltonian,
            "initial_state": initial_state,
            "time_steps": time_steps,
            "method": "exact"
        }
        result_exact = self.protocols.quantum_simulation(params_exact)
        exact_time = time.time() - start_time
        
        # Benchmark Runge-Kutta simulation
        start_time = time.time()
        params_rk = {
            "hamiltonian": hamiltonian,
            "initial_state": initial_state,
            "time_steps": time_steps,
            "method": "runge_kutta"
        }
        result_rk = self.protocols.quantum_simulation(params_rk)
        rk_time = time.time() - start_time
        
        # Performance assertions
        assert exact_time < 1.0  # Should complete in reasonable time
        assert rk_time < 2.0     # RK might be slower but should be reasonable
        
        print(f"Quantum simulation (exact): {exact_time:.4f}s")
        print(f"Quantum simulation (RK): {rk_time:.4f}s")
        
        # Results should be similar in quality
        assert result_exact.shape == result_rk.shape
    
    def test_temporal_analysis_performance(self):
        """Benchmark temporal analysis performance"""
        # Large time series
        n_points = 5000
        n_series = 5
        
        data = np.random.normal(0, 1, (n_series, n_points))
        
        # Add some patterns for realistic analysis
        t = np.arange(n_points)
        for i in range(n_series):
            data[i] += 0.5 * np.sin(0.01 * (i + 1) * t)  # Different frequencies
        
        start_time = time.time()
        results = self.protocols.analyze_temporal_patterns(data)
        analysis_time = time.time() - start_time
        
        assert analysis_time < 10.0  # Should complete within reasonable time
        
        # Check that analysis completed successfully
        assert "patterns" in results
        assert "frequency_analysis" in results
        assert len(results["frequency_analysis"]) == n_series
        
        print(f"Temporal analysis: {analysis_time:.4f}s for {n_points}×{n_series} data")
    
    def test_memory_usage_agents(self):
        """Test memory usage of agent creation"""
        process = psutil.Process()
        
        # Baseline memory
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create many agents
        n_agents = 100
        agents = []
        
        for i in range(n_agents):
            agent = self.factory.create_mathematical_agent({
                "tau": 1.0 + i * 0.01,
                "field_dimension": 4,
                "matrix_size": 4,
            })
            agents.append(agent)
        
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        memory_per_agent = (final_memory - baseline_memory) / n_agents
        
        # Each agent should use reasonable amount of memory
        assert memory_per_agent < 1.0  # Less than 1 MB per agent
        
        print(f"Memory usage: {memory_per_agent:.2f} MB per agent")
    
    def test_scalability_network_size(self):
        """Test network building scalability with size"""
        sizes = [5, 10, 20, 30]
        build_times = []
        
        for size in sizes:
            # Create agents
            agents = []
            for i in range(size):
                agent = self.factory.create_mathematical_agent({
                    "tau": 1.0 + i * 0.01,
                })
                agents.append(agent)
            
            # Time network building
            start_time = time.time()
            network = self.builder.build_network(agents, target_density=0.3)
            build_time = time.time() - start_time
            build_times.append(build_time)
            
            assert build_time < size * 0.2  # Should scale reasonably
        
        # Check that scaling is not too bad (should be roughly quadratic at worst)
        for i in range(1, len(sizes)):
            ratio = build_times[i] / build_times[i-1]
            size_ratio = sizes[i] / sizes[i-1]
            
            # Time ratio should not be much worse than quadratic scaling
            assert ratio < size_ratio ** 3  # Allow for some overhead
        
        print(f"Network build times: {list(zip(sizes, build_times))}")
    
    def test_concurrent_operations_performance(self):
        """Test performance with concurrent operations"""
        # Create several agents
        agents = []
        for i in range(10):
            agent = self.factory.create_mathematical_agent({
                "tau": 1.0 + i * 0.1,
            })
            agents.append(agent)
        
        # Simulate concurrent operations
        n_operations = 100
        start_time = time.time()
        
        for i in range(n_operations):
            agent = agents[i % len(agents)]
            
            # Perform various operations
            if i % 3 == 0:
                agent.execute_operation("analyze_field")
            elif i % 3 == 1:
                data = np.random.random(10)
                agent.execute_operation("compute_derivatives", data=data)
            else:
                matrix = np.random.random((2, 2))
                try:
                    agent.execute_operation("matrix_operations", operation="eigenvalues", matrix=matrix)
                except:
                    pass  # Some operations might fail, that's ok for this test
        
        total_time = time.time() - start_time
        
        assert total_time < 5.0  # Should handle concurrent operations efficiently
        
        time_per_operation = total_time / n_operations
        print(f"Concurrent operations: {time_per_operation*1000:.2f}ms per operation")


class TestStressTests:
    """Stress tests for system limits"""
    
    def setup_method(self):
        """Set up stress test fixtures"""
        self.factory = AgentFactory()
        self.builder = NetworkBuilder(random_seed=42)
        self.protocols = ScientificProtocols(random_seed=42)
    
    def test_large_field_dimensions(self):
        """Test with large field dimensions"""
        # Large but still reasonable dimensions
        tau = 2.0
        matrix_size = 10
        field_dim = 20
        
        Q = np.eye(matrix_size) + 0.01 * np.random.random((matrix_size, matrix_size))
        Q = Q @ Q.T
        phi_grad = np.random.normal(0, 0.01, field_dim)
        sigma = 0.9
        S = np.random.random((matrix_size, matrix_size))
        
        field = LogicField(tau, Q, phi_grad, sigma, S)
        
        # Should handle large dimensions without issues
        intelligence = field.calculate_intelligence_level()
        coherence = field.compute_coherence()
        energy = field.field_energy()
        
        assert np.isfinite(intelligence)
        assert np.isfinite(coherence)
        assert np.isfinite(energy)
        assert 0 <= coherence <= 1
    
    def test_many_agents_error_handling(self):
        """Test system behavior with many agents and errors"""
        n_agents = 50
        agents = []
        
        # Create many agents with various configurations
        for i in range(n_agents):
            try:
                agent = self.factory.create_mathematical_agent({
                    "tau": 0.5 + i * 0.02,
                    "field_dimension": 2 + i % 5,
                    "matrix_size": 2 + i % 4,
                })
                agents.append(agent)
            except Exception as e:
                # Some configurations might fail, that's ok
                pass
        
        assert len(agents) > n_agents * 0.8  # Most should succeed
        
        # Try operations that might fail
        error_count = 0
        success_count = 0
        
        for agent in agents[:20]:  # Test subset to keep test time reasonable
            try:
                result = agent.execute_operation("analyze_field")
                success_count += 1
            except Exception:
                error_count += 1
        
        # Most operations should succeed
        assert success_count > error_count
    
    def test_extreme_parameter_values(self):
        """Test with extreme but valid parameter values"""
        # Very small values
        tau_small = 1e-3
        Q_small = 1e-6 * np.eye(2)
        phi_grad_small = 1e-6 * np.array([1, 1])
        sigma_small = 1e-6
        S_small = 1e-6 * np.eye(2)
        
        field_small = LogicField(tau_small, Q_small, phi_grad_small, sigma_small, S_small)
        
        intelligence_small = field_small.calculate_intelligence_level()
        coherence_small = field_small.compute_coherence()
        
        assert np.isfinite(intelligence_small)
        assert np.isfinite(coherence_small)
        assert 0 <= coherence_small <= 1
        
        # Large values
        tau_large = 100.0
        Q_large = 100.0 * np.eye(2)
        phi_grad_large = 10.0 * np.array([1, 1])
        sigma_large = 1.0  # Sigma is bounded
        S_large = 10.0 * np.eye(2)
        
        field_large = LogicField(tau_large, Q_large, phi_grad_large, sigma_large, S_large)
        
        intelligence_large = field_large.calculate_intelligence_level()
        coherence_large = field_large.compute_coherence()
        
        assert np.isfinite(intelligence_large)
        assert np.isfinite(coherence_large)
        assert 0 <= coherence_large <= 1
    
    def test_long_running_simulation(self):
        """Test long-running quantum simulation"""
        # Small system for long simulation
        hamiltonian = np.array([[1, 0.1], [0.1, -1]], dtype=np.complex128)
        initial_state = np.array([1, 0], dtype=np.complex128)
        
        # Long time evolution with many steps
        time_steps = np.linspace(0, 10, 500)
        
        params = {
            "hamiltonian": hamiltonian,
            "initial_state": initial_state,
            "time_steps": time_steps,
            "method": "exact"
        }
        
        start_time = time.time()
        result = self.protocols.quantum_simulation(params)
        simulation_time = time.time() - start_time
        
        assert simulation_time < 5.0  # Should complete in reasonable time
        assert result.shape == (500, 2)
        
        # States should remain normalized
        for state in result[::50]:  # Check every 50th state
            norm = np.linalg.norm(state)
            assert abs(norm - 1.0) < 1e-10


class TestResourceMonitoring:
    """Monitor resource usage during operations"""
    
    def test_memory_leaks_agents(self):
        """Test for memory leaks in agent operations"""
        process = psutil.Process()
        
        # Create and destroy agents multiple times
        initial_memory = process.memory_info().rss
        
        for cycle in range(10):
            agents = []
            
            # Create agents
            for i in range(20):
                agent = AgentFactory().create_mathematical_agent({
                    "tau": 1.0 + i * 0.05,
                })
                agents.append(agent)
            
            # Use agents
            for agent in agents:
                agent.execute_operation("analyze_field")
            
            # Clean up
            del agents
            gc.collect()
        
        final_memory = process.memory_info().rss
        memory_growth = (final_memory - initial_memory) / 1024 / 1024  # MB
        
        # Should not have significant memory growth
        assert memory_growth < 50  # Less than 50 MB growth
        
        print(f"Memory growth over 10 cycles: {memory_growth:.2f} MB")
    
    def test_cpu_usage_monitoring(self):
        """Monitor CPU usage during intensive operations"""
        # Get initial CPU usage
        cpu_percent_before = psutil.cpu_percent(interval=1)
        
        # Perform CPU-intensive operations
        start_time = time.time()
        
        protocols = ScientificProtocols()
        
        # Multiple quantum simulations
        for i in range(5):
            hamiltonian = np.random.random((4, 4)) + 1j * np.random.random((4, 4))
            hamiltonian = hamiltonian + hamiltonian.conj().T
            initial_state = np.random.random(4) + 1j * np.random.random(4)
            initial_state = initial_state / np.linalg.norm(initial_state)
            
            params = {
                "hamiltonian": hamiltonian,
                "initial_state": initial_state,
                "time_steps": np.linspace(0, 1, 100),
                "method": "exact"
            }
            
            protocols.quantum_simulation(params)
        
        operation_time = time.time() - start_time
        cpu_percent_after = psutil.cpu_percent(interval=1)
        
        print(f"CPU usage before: {cpu_percent_before:.1f}%")
        print(f"CPU usage after: {cpu_percent_after:.1f}%")
        print(f"Operation time: {operation_time:.2f}s")
        
        # Should complete in reasonable time
        assert operation_time < 10.0