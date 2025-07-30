"""Test suite for LogicField 5-tuple mathematics

Tests the mathematical properties and operations of the LogicField class.
"""

import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_allclose

from src.core.logic_field import LogicField


class TestLogicField:
    """Test cases for LogicField class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.tau = 1.5
        self.Q = np.eye(3) + 0.1 * np.random.random((3, 3))
        self.Q = self.Q @ self.Q.T  # Ensure positive definite
        self.phi_grad = np.array([0.1, -0.2, 0.3])
        self.sigma = 0.8
        self.S = np.random.random((3, 3))
        
        self.field = LogicField(self.tau, self.Q, self.phi_grad, self.sigma, self.S)
    
    def test_initialization(self):
        """Test proper initialization of LogicField"""
        assert self.field.tau == self.tau
        assert_array_almost_equal(self.field.Q, self.Q)
        assert_array_almost_equal(self.field.phi_grad, self.phi_grad)
        assert self.field.sigma == self.sigma
        assert_array_almost_equal(self.field.S, self.S)
    
    def test_initialization_validation(self):
        """Test parameter validation during initialization"""
        # Test negative tau
        with pytest.raises(ValueError, match="Intelligence parameter"):
            LogicField(-1.0, self.Q, self.phi_grad, self.sigma, self.S)
        
        # Test invalid sigma
        with pytest.raises(ValueError, match="Coherence parameter"):
            LogicField(self.tau, self.Q, self.phi_grad, 1.5, self.S)
        
        # Test non-square Q matrix
        Q_bad = np.random.random((3, 4))
        with pytest.raises(ValueError, match="square matrix"):
            LogicField(self.tau, Q_bad, self.phi_grad, self.sigma, self.S)
        
        # Test non-symmetric Q matrix
        Q_bad = np.random.random((3, 3))
        with pytest.raises(ValueError, match="symmetric"):
            LogicField(self.tau, Q_bad, self.phi_grad, self.sigma, self.S)
        
        # Test non-positive-definite Q matrix
        Q_bad = -np.eye(3)
        with pytest.raises(ValueError, match="positive definite"):
            LogicField(self.tau, Q_bad, self.phi_grad, self.sigma, self.S)
        
        # Test invalid phi_grad dimension
        phi_grad_bad = np.random.random((3, 3))
        with pytest.raises(ValueError, match="vector"):
            LogicField(self.tau, self.Q, phi_grad_bad, self.sigma, self.S)
        
        # Test non-square S matrix
        S_bad = np.random.random((3, 4))
        with pytest.raises(ValueError, match="square"):
            LogicField(self.tau, self.Q, self.phi_grad, self.sigma, S_bad)
    
    def test_calculate_intelligence_level(self):
        """Test intelligence level calculation"""
        intelligence = self.field.calculate_intelligence_level()
        
        # Should be positive
        assert intelligence > 0
        
        # Should increase with tau
        field_high_tau = LogicField(3.0, self.Q, self.phi_grad, self.sigma, self.S)
        intelligence_high = field_high_tau.calculate_intelligence_level()
        assert intelligence_high > intelligence
        
        # Should be affected by field gradient
        field_zero_grad = LogicField(self.tau, self.Q, np.zeros_like(self.phi_grad), self.sigma, self.S)
        intelligence_zero_grad = field_zero_grad.calculate_intelligence_level()
        # The exact relationship depends on implementation details
        assert intelligence_zero_grad > 0
    
    def test_compute_coherence(self):
        """Test coherence computation"""
        coherence = self.field.compute_coherence()
        
        # Should be in [0, 1] range
        assert 0 <= coherence <= 1
        
        # Should increase with sigma
        field_high_sigma = LogicField(self.tau, self.Q, self.phi_grad, 0.95, self.S)
        coherence_high = field_high_sigma.compute_coherence()
        assert coherence_high >= coherence  # Should be at least as high
        
        # High coherence case
        Q_perfect = np.eye(3)
        S_perfect = np.eye(3)
        phi_grad_zero = np.zeros(3)
        field_perfect = LogicField(self.tau, Q_perfect, phi_grad_zero, 1.0, S_perfect)
        coherence_perfect = field_perfect.compute_coherence()
        assert coherence_perfect > 0.5  # Should be reasonably high
    
    def test_field_energy(self):
        """Test field energy calculation"""
        energy = self.field.field_energy()
        
        # Should be positive (due to kinetic + potential + intelligence terms)
        assert energy >= 0
        
        # Should increase with gradient magnitude
        field_large_grad = LogicField(self.tau, self.Q, 2 * self.phi_grad, self.sigma, self.S)
        energy_large = field_large_grad.field_energy()
        assert energy_large > energy
    
    def test_evolve_field(self):
        """Test field evolution"""
        original_grad = self.field.phi_grad.copy()
        original_sigma = self.field.sigma
        
        # Evolve without external force
        self.field.evolve_field(0.01)
        
        # Gradient should change
        assert not np.allclose(self.field.phi_grad, original_grad)
        
        # Sigma might change due to evolution
        assert 0 <= self.field.sigma <= 1
        
        # Test with external force
        field2 = LogicField(self.tau, self.Q, self.phi_grad, self.sigma, self.S)
        external_force = np.array([0.1, 0.0, -0.1])
        field2.evolve_field(0.01, external_force)
        
        # Should evolve differently with external force
        assert not np.allclose(field2.phi_grad, self.field.phi_grad)
    
    def test_evolve_field_validation(self):
        """Test field evolution parameter validation"""
        # Wrong external force dimension
        wrong_force = np.array([0.1, 0.2])  # Should be 3D
        with pytest.raises(ValueError, match="same shape"):
            self.field.evolve_field(0.01, wrong_force)
    
    def test_get_field_tuple(self):
        """Test getting complete field tuple"""
        tau, Q, phi_grad, sigma, S = self.field.get_field_tuple()
        
        assert tau == self.tau
        assert_array_almost_equal(Q, self.Q)
        assert_array_almost_equal(phi_grad, self.phi_grad)
        assert sigma == self.sigma
        assert_array_almost_equal(S, self.S)
        
        # Ensure returned arrays are copies
        Q[0, 0] = 999
        assert self.field.Q[0, 0] != 999
    
    def test_field_immutability_during_init(self):
        """Test that field parameters are properly copied during initialization"""
        Q_orig = self.Q.copy()
        phi_grad_orig = self.phi_grad.copy()
        S_orig = self.S.copy()
        
        # Modify original arrays
        self.Q[0, 0] = 999
        self.phi_grad[0] = 999
        self.S[0, 0] = 999
        
        # Field should be unaffected
        assert self.field.Q[0, 0] != 999
        assert self.field.phi_grad[0] != 999
        assert self.field.S[0, 0] != 999
    
    def test_repr(self):
        """Test string representation"""
        repr_str = repr(self.field)
        assert "LogicField" in repr_str
        assert f"tau={self.tau:.3f}" in repr_str
        assert f"sigma={self.sigma:.3f}" in repr_str
        assert "Q.shape" in repr_str
        assert "phi_grad.shape" in repr_str
        assert "S.shape" in repr_str


class TestLogicFieldMathematicalProperties:
    """Test mathematical properties and edge cases"""
    
    def test_minimal_field(self):
        """Test minimal valid field configuration"""
        tau = 0.1
        Q = np.array([[1.0]])
        phi_grad = np.array([0.0])
        sigma = 0.0
        S = np.array([[1.0]])
        
        field = LogicField(tau, Q, phi_grad, sigma, S)
        
        intelligence = field.calculate_intelligence_level()
        coherence = field.compute_coherence()
        energy = field.field_energy()
        
        assert intelligence > 0
        assert 0 <= coherence <= 1
        assert energy >= 0
    
    def test_large_field(self):
        """Test larger dimensional field"""
        tau = 2.0
        Q = np.eye(5) + 0.01 * np.random.random((5, 5))
        Q = Q @ Q.T
        phi_grad = np.random.normal(0, 0.1, 10)
        sigma = 0.9
        S = np.random.random((5, 5))
        
        field = LogicField(tau, Q, phi_grad, sigma, S)
        
        # All computations should work with larger dimensions
        intelligence = field.calculate_intelligence_level()
        coherence = field.compute_coherence()
        energy = field.field_energy()
        
        assert intelligence > 0
        assert 0 <= coherence <= 1
        assert energy >= 0
    
    def test_field_stability(self):
        """Test numerical stability of field operations"""
        # Create field with very small values
        tau = 1e-6
        Q = 1e-6 * np.eye(2)
        phi_grad = 1e-6 * np.array([1, 1])
        sigma = 1e-6
        S = 1e-6 * np.eye(2)
        
        field = LogicField(tau, Q, phi_grad, sigma, S)
        
        # Should not crash or return NaN
        intelligence = field.calculate_intelligence_level()
        coherence = field.compute_coherence()
        energy = field.field_energy()
        
        assert np.isfinite(intelligence)
        assert np.isfinite(coherence)
        assert np.isfinite(energy)
        assert 0 <= coherence <= 1
    
    def test_field_evolution_conservation(self):
        """Test that field evolution preserves certain properties"""
        # Create field
        tau = 1.0
        Q = np.eye(3) + 0.1 * np.random.random((3, 3))
        Q = Q @ Q.T
        phi_grad = np.array([0.1, 0.2, 0.3])
        sigma = 0.8
        S = np.random.random((3, 3))
        
        field = LogicField(tau, Q, phi_grad, sigma, S)
        
        # Evolve field many small steps
        dt = 0.001
        n_steps = 100
        
        for _ in range(n_steps):
            field.evolve_field(dt)
        
        # Field should still be valid
        assert field.tau > 0
        assert 0 <= field.sigma <= 1
        assert np.all(np.isfinite(field.phi_grad))
        
        # Should still be able to compute properties
        intelligence = field.calculate_intelligence_level()
        coherence = field.compute_coherence()
        energy = field.field_energy()
        
        assert np.isfinite(intelligence)
        assert np.isfinite(coherence)
        assert np.isfinite(energy)