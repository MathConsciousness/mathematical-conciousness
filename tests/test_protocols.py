"""Test suite for scientific computing protocols

Tests quantum simulation, temporal analysis, and other scientific calculations.
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from unittest.mock import Mock, patch

from src.computing.protocols import ScientificProtocols, QuantumState, TemporalPattern, SimulationResult


class TestScientificProtocols:
    """Test cases for ScientificProtocols class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.protocols = ScientificProtocols(precision="double", random_seed=42)
    
    def test_protocols_initialization(self):
        """Test ScientificProtocols initialization"""
        protocols = ScientificProtocols()
        assert protocols.precision == "double"
        assert protocols.dtype == np.float64
        
        # Test different precisions
        protocols_single = ScientificProtocols(precision="single")
        assert protocols_single.dtype == np.float32
        
        protocols_quad = ScientificProtocols(precision="quad")
        assert protocols_quad.dtype == np.longdouble
    
    def test_protocols_invalid_precision(self):
        """Test invalid precision parameter"""
        with pytest.raises(ValueError, match="Unknown precision"):
            ScientificProtocols(precision="invalid")


class TestQuantumSimulation:
    """Test quantum simulation capabilities"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.protocols = ScientificProtocols(random_seed=42)
    
    def test_quantum_simulation_exact_simple(self):
        """Test exact quantum simulation with simple system"""
        # Simple 2-level system (Pauli-Z Hamiltonian)
        hamiltonian = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        initial_state = np.array([1, 0], dtype=np.complex128)
        time_steps = np.array([0.0, 0.1, 0.2])
        
        params = {
            "hamiltonian": hamiltonian,
            "initial_state": initial_state,
            "time_steps": time_steps,
            "method": "exact"
        }
        
        result = self.protocols.quantum_simulation(params)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 2)  # 3 time steps, 2-dimensional state
        
        # Initial state should be preserved
        assert_array_almost_equal(result[0], initial_state)
        
        # States should be normalized
        for state in result:
            norm = np.linalg.norm(state)
            assert abs(norm - 1.0) < 1e-10
    
    def test_quantum_simulation_harmonic_oscillator(self):
        """Test quantum simulation with harmonic oscillator"""
        # Use the built-in harmonic oscillator
        n_levels = 3
        ho_data = self.protocols.create_quantum_harmonic_oscillator(n_levels, omega=1.0)
        
        hamiltonian = ho_data["hamiltonian"]
        initial_state = ho_data["basis_states"][0]  # Ground state
        time_steps = np.linspace(0, 1, 10)
        
        params = {
            "hamiltonian": hamiltonian,
            "initial_state": initial_state,
            "time_steps": time_steps,
            "method": "exact"
        }
        
        result = self.protocols.quantum_simulation(params)
        
        assert result.shape == (10, n_levels)
        
        # Ground state should remain ground state (up to phase)
        final_state = result[-1]
        overlap = abs(np.vdot(initial_state, final_state))**2
        assert overlap > 0.99  # Should have high overlap (accounting for phase)
    
    def test_quantum_simulation_runge_kutta(self):
        """Test quantum simulation with Runge-Kutta method"""
        hamiltonian = np.array([[0, 1], [1, 0]], dtype=np.complex128)  # Pauli-X
        initial_state = np.array([1, 0], dtype=np.complex128)
        time_steps = np.linspace(0, np.pi, 20)
        
        params = {
            "hamiltonian": hamiltonian,
            "initial_state": initial_state,
            "time_steps": time_steps,
            "method": "runge_kutta"
        }
        
        result = self.protocols.quantum_simulation(params)
        
        assert result.shape == (20, 2)
        
        # After time π with Pauli-X, should approximately flip to |1⟩
        final_state = result[-1]
        expected_final = np.array([0, 1j], dtype=np.complex128)  # Up to phase
        
        # Check that we're close to the expected final state (up to global phase)
        overlap = abs(np.vdot(expected_final, final_state))**2
        assert overlap > 0.9
    
    def test_quantum_simulation_trotter(self):
        """Test quantum simulation with Suzuki-Trotter method"""
        hamiltonian = np.array([[1, 0.1], [0.1, -1]], dtype=np.complex128)
        initial_state = np.array([1, 0], dtype=np.complex128)
        time_steps = np.array([0.0, 0.5, 1.0])
        
        params = {
            "hamiltonian": hamiltonian,
            "initial_state": initial_state,
            "time_steps": time_steps,
            "method": "suzuki_trotter",
            "trotter_steps": 50
        }
        
        result = self.protocols.quantum_simulation(params)
        
        assert result.shape == (3, 2)
        
        # Compare with exact method for small system
        params_exact = params.copy()
        params_exact["method"] = "exact"
        result_exact = self.protocols.quantum_simulation(params_exact)
        
        # Should be reasonably close to exact result
        for i in range(len(time_steps)):
            overlap = abs(np.vdot(result_exact[i], result[i]))**2
            assert overlap > 0.95  # Allow some numerical error
    
    def test_quantum_simulation_validation(self):
        """Test quantum simulation parameter validation"""
        hamiltonian = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        initial_state = np.array([1, 0], dtype=np.complex128)
        time_steps = np.array([0.0, 0.1])
        
        # Missing required parameter
        with pytest.raises(ValueError, match="Missing required parameter"):
            self.protocols.quantum_simulation({"hamiltonian": hamiltonian})
        
        # Wrong Hamiltonian shape
        with pytest.raises(ValueError, match="square matrix"):
            self.protocols.quantum_simulation({
                "hamiltonian": np.array([[1, 0, 0], [0, -1, 0]]),
                "initial_state": initial_state,
                "time_steps": time_steps
            })
        
        # Wrong initial state dimension
        with pytest.raises(ValueError, match="dimension must match"):
            self.protocols.quantum_simulation({
                "hamiltonian": hamiltonian,
                "initial_state": np.array([1, 0, 0]),
                "time_steps": time_steps
            })
        
        # Unknown method
        with pytest.raises(ValueError, match="Unknown simulation method"):
            self.protocols.quantum_simulation({
                "hamiltonian": hamiltonian,
                "initial_state": initial_state,
                "time_steps": time_steps,
                "method": "invalid_method"
            })
    
    def test_create_quantum_harmonic_oscillator(self):
        """Test quantum harmonic oscillator creation"""
        n_levels = 4
        omega = 2.0
        
        ho_data = self.protocols.create_quantum_harmonic_oscillator(n_levels, omega)
        
        assert "hamiltonian" in ho_data
        assert "energies" in ho_data
        assert "basis_states" in ho_data
        assert "creation_operator" in ho_data
        assert "annihilation_operator" in ho_data
        assert "number_operator" in ho_data
        
        H = ho_data["hamiltonian"]
        energies = ho_data["energies"]
        basis_states = ho_data["basis_states"]
        a_dag = ho_data["creation_operator"]
        a = ho_data["annihilation_operator"]
        n_op = ho_data["number_operator"]
        
        # Check dimensions
        assert H.shape == (n_levels, n_levels)
        assert len(energies) == n_levels
        assert basis_states.shape == (n_levels, n_levels)
        
        # Check energy levels
        expected_energies = omega * (np.arange(n_levels) + 0.5)
        assert_array_almost_equal(energies, expected_energies)
        
        # Check that basis states are eigenstates of Hamiltonian
        for i, state in enumerate(basis_states.T):
            H_state = H @ state
            energy_state = energies[i] * state
            assert_array_almost_equal(H_state, energy_state, decimal=10)
        
        # Check commutation relation [a, a†] = I
        commutator = a @ a_dag - a_dag @ a
        expected_commutator = np.eye(n_levels)
        assert_array_almost_equal(commutator, expected_commutator, decimal=10)


class TestTemporalAnalysis:
    """Test temporal data analysis capabilities"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.protocols = ScientificProtocols(random_seed=42)
    
    def test_temporal_analysis_sine_wave(self):
        """Test temporal analysis with sine wave"""
        # Create a clear sine wave
        t = np.linspace(0, 4*np.pi, 200)
        frequency = 2.0
        data = np.sin(frequency * t) + 0.05 * np.random.normal(size=len(t))
        
        results = self.protocols.analyze_temporal_patterns(data)
        
        assert isinstance(results, dict)
        assert "patterns" in results
        assert "statistics" in results
        assert "frequency_analysis" in results
        assert "trend_analysis" in results
        assert "anomalies" in results
        
        # Check frequency analysis
        freq_analysis = results["frequency_analysis"]["series_0"]
        dominant_freq = freq_analysis["dominant_frequency"]
        
        # Should detect frequency close to 2.0/(2π) = 1/π
        expected_freq = frequency / (2 * np.pi)
        assert abs(dominant_freq - expected_freq) < 0.1
    
    def test_temporal_analysis_trending_data(self):
        """Test temporal analysis with trending data"""
        # Create data with linear trend
        t = np.arange(100)
        slope = 0.5
        data = slope * t + np.random.normal(0, 0.1, size=len(t))
        
        results = self.protocols.analyze_temporal_patterns(data)
        
        # Check trend analysis
        trend_analysis = results["trend_analysis"]["series_0"]
        detected_slope = trend_analysis["linear_slope"]
        
        assert abs(detected_slope - slope) < 0.1
        assert trend_analysis["linear_r_squared"] > 0.8
    
    def test_temporal_analysis_periodic_pattern(self):
        """Test detection of periodic patterns"""
        # Create periodic data
        t = np.arange(100)
        period = 10
        data = np.sin(2 * np.pi * t / period) + 0.1 * np.random.normal(size=len(t))
        
        results = self.protocols.analyze_temporal_patterns(data)
        
        patterns = results["patterns"]
        periodic_patterns = [p for p in patterns if isinstance(p, dict) and p.get("pattern_type") == "periodic"]
        
        # Should detect at least one periodic pattern
        assert len(periodic_patterns) > 0
        
        # Check if detected period is reasonable
        detected_periods = [p["period"] for p in periodic_patterns]
        assert any(abs(p - period) < 2 for p in detected_periods)
    
    def test_temporal_analysis_multidimensional(self):
        """Test temporal analysis with multidimensional data"""
        # Create 2D time series
        t = np.arange(50)
        data1 = np.sin(0.1 * t) + 0.05 * np.random.normal(size=len(t))
        data2 = np.cos(0.2 * t) + 0.05 * np.random.normal(size=len(t))
        data_2d = np.array([data1, data2])
        
        results = self.protocols.analyze_temporal_patterns(data_2d)
        
        # Should have results for both series
        assert "series_0" in results["statistics"]
        assert "series_1" in results["statistics"]
        assert "series_0" in results["frequency_analysis"]
        assert "series_1" in results["frequency_analysis"]
    
    def test_temporal_analysis_anomaly_detection(self):
        """Test anomaly detection in temporal data"""
        # Create data with outliers
        t = np.arange(100)
        data = np.sin(0.1 * t) + 0.05 * np.random.normal(size=len(t))
        
        # Add some outliers
        data[25] = 5.0  # Large positive outlier
        data[75] = -5.0  # Large negative outlier
        
        results = self.protocols.analyze_temporal_patterns(data)
        
        anomalies = results["anomalies"]["series_0"]
        anomaly_indices = anomalies["statistical_anomalies"]["indices"]
        
        # Should detect the outliers
        assert 25 in anomaly_indices
        assert 75 in anomaly_indices
        assert anomalies["anomaly_score"] > 0.01
    
    def test_temporal_statistics(self):
        """Test basic temporal statistics computation"""
        # Known statistical properties
        data = np.array([1, 2, 3, 4, 5])
        
        results = self.protocols.analyze_temporal_patterns(data)
        stats = results["statistics"]["series_0"]
        
        assert abs(stats["mean"] - 3.0) < 1e-10
        assert abs(stats["variance"] - 2.0) < 1e-10
        assert stats["trend_slope"] > 0  # Should detect positive trend


class TestScientificComputingUtilities:
    """Test additional scientific computing utilities"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.protocols = ScientificProtocols(random_seed=42)
    
    def test_optimize_system_parameters(self):
        """Test parameter optimization"""
        # Simple quadratic function: (x-2)² + (y-1)²
        def objective(params):
            x, y = params
            return (x - 2)**2 + (y - 1)**2
        
        initial_params = np.array([0.0, 0.0])
        
        result = self.protocols.optimize_system_parameters(
            objective, initial_params, method="BFGS"
        )
        
        assert isinstance(result, dict)
        assert "optimal_params" in result
        assert "optimal_value" in result
        assert "success" in result
        
        if result["success"]:
            optimal_params = np.array(result["optimal_params"])
            assert abs(optimal_params[0] - 2.0) < 0.01
            assert abs(optimal_params[1] - 1.0) < 0.01
            assert result["optimal_value"] < 0.01
    
    def test_optimize_with_bounds(self):
        """Test optimization with parameter bounds"""
        def objective(params):
            x = params[0]
            return x**2  # Minimum at x=0
        
        initial_params = np.array([5.0])
        bounds = [(1.0, 10.0)]  # Constrained to [1, 10]
        
        result = self.protocols.optimize_system_parameters(
            objective, initial_params, bounds=bounds
        )
        
        if result["success"]:
            # Should find minimum at boundary (x=1)
            optimal_x = result["optimal_params"][0]
            assert abs(optimal_x - 1.0) < 0.01
    
    def test_solve_differential_equation(self):
        """Test ODE solving"""
        # Simple exponential decay: dy/dt = -y, y(0) = 1
        def ode_func(t, y):
            return -y
        
        initial_conditions = np.array([1.0])
        time_span = (0.0, 2.0)
        time_points = np.linspace(0, 2, 20)
        
        result = self.protocols.solve_differential_equation(
            ode_func, initial_conditions, time_span, time_points
        )
        
        assert isinstance(result, SimulationResult)
        assert result.success
        assert result.data.shape == (20, 1)
        
        # Check that solution approximates exp(-t)
        final_value = result.data[-1, 0]
        expected_final = np.exp(-2.0)
        assert abs(final_value - expected_final) < 0.01
    
    def test_solve_differential_equation_system(self):
        """Test solving system of ODEs"""
        # Harmonic oscillator: d²x/dt² = -x
        # Convert to system: dx/dt = v, dv/dt = -x
        def harmonic_oscillator(t, y):
            x, v = y
            return [v, -x]
        
        initial_conditions = np.array([1.0, 0.0])  # x(0)=1, v(0)=0
        time_span = (0.0, 2*np.pi)
        time_points = np.linspace(0, 2*np.pi, 50)
        
        result = self.protocols.solve_differential_equation(
            harmonic_oscillator, initial_conditions, time_span, time_points
        )
        
        assert result.success
        assert result.data.shape == (50, 2)
        
        # After one period (2π), should return to initial condition
        final_state = result.data[-1]
        assert abs(final_state[0] - 1.0) < 0.1  # Position
        assert abs(final_state[1] - 0.0) < 0.1  # Velocity
    
    def test_differential_equation_error_handling(self):
        """Test ODE solver error handling"""
        # Problematic ODE that should fail
        def bad_ode(t, y):
            return y / 0  # Division by zero
        
        initial_conditions = np.array([1.0])
        time_span = (0.0, 1.0)
        
        result = self.protocols.solve_differential_equation(
            bad_ode, initial_conditions, time_span
        )
        
        assert not result.success
        assert "error" in result.metadata


class TestQuantumState:
    """Test QuantumState helper class"""
    
    def test_quantum_state_initialization(self):
        """Test QuantumState initialization and validation"""
        amplitudes = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=np.complex128)
        basis_labels = ["|0⟩", "|1⟩"]
        
        state = QuantumState(amplitudes, basis_labels, 2)
        
        assert_array_almost_equal(state.amplitudes, amplitudes)
        assert state.basis_labels == basis_labels
        assert state.dimension == 2
    
    def test_quantum_state_normalization(self):
        """Test automatic normalization of quantum states"""
        # Non-normalized amplitudes
        amplitudes = np.array([1, 1], dtype=np.complex128)
        
        with pytest.warns(UserWarning, match="not normalized"):
            state = QuantumState(amplitudes, ["|0⟩", "|1⟩"], 2)
        
        # Should be normalized after initialization
        norm = np.sum(np.abs(state.amplitudes)**2)
        assert abs(norm - 1.0) < 1e-10
    
    def test_quantum_state_validation_error(self):
        """Test QuantumState validation errors"""
        # Wrong amplitude length
        with pytest.raises(ValueError, match="length must match dimension"):
            QuantumState(np.array([1, 0]), ["|0⟩", "|1⟩"], 3)


class TestTemporalPattern:
    """Test TemporalPattern data class"""
    
    def test_temporal_pattern_creation(self):
        """Test TemporalPattern creation"""
        pattern = TemporalPattern(
            pattern_type="periodic",
            frequency=0.5,
            amplitude=0.8,
            phase=0.0,
            confidence=0.9,
            period=2.0
        )
        
        assert pattern.pattern_type == "periodic"
        assert pattern.frequency == 0.5
        assert pattern.amplitude == 0.8
        assert pattern.phase == 0.0
        assert pattern.confidence == 0.9
        assert pattern.period == 2.0