"""
LogicField: Core mathematical computations implementing 5-tuple mathematics
L(x,t) = (τ, Q, ∇Φ, σ, S)
"""

import numpy as np
from typing import Optional, Tuple


class LogicField:
    """
    Mathematical field implementing 5-tuple mathematics for quantum computations.
    
    The field is defined as L(x,t) = (τ, Q, ∇Φ, σ, S) where:
    - τ (tau): Temporal coherence parameter
    - Q: Quantum state matrix
    - ∇Φ (phi_grad): Gradient of the potential field
    - σ (sigma): Field strength parameter
    - S: State transformation matrix
    """
    
    def __init__(self, tau: float, Q: np.ndarray, phi_grad: np.ndarray, 
                 sigma: float, S: np.ndarray):
        """
        Initialize the LogicField with 5-tuple parameters.
        
        Args:
            tau: Temporal coherence parameter (0 < tau <= 1)
            Q: Quantum state matrix
            phi_grad: Gradient of the potential field
            sigma: Field strength parameter
            S: State transformation matrix
        """
        self.tau = tau
        self.Q = Q
        self.phi_grad = phi_grad
        self.sigma = sigma
        self.S = S
        
        # Validate parameters
        self._validate_parameters()
        
    def _validate_parameters(self):
        """Validate the mathematical consistency of field parameters."""
        if not (0 < self.tau <= 1):
            raise ValueError(f"Tau must be in (0, 1], got {self.tau}")
        
        if self.sigma <= 0:
            raise ValueError(f"Sigma must be positive, got {self.sigma}")
            
        # Ensure matrices have appropriate dimensions
        if self.Q.ndim != 2 or self.Q.shape[0] != self.Q.shape[1]:
            raise ValueError("Q must be a square matrix")
            
        if self.S.ndim != 2 or self.S.shape[0] != self.S.shape[1]:
            raise ValueError("S must be a square matrix")
    
    def compute_field_strength(self, x: np.ndarray, t: float) -> float:
        """
        Compute the field strength at position x and time t.
        
        Args:
            x: Spatial coordinates
            t: Time parameter
            
        Returns:
            Field strength value
        """
        # Temporal evolution with tau parameter
        temporal_factor = np.exp(-t / self.tau)
        
        # Spatial field contribution
        spatial_factor = np.dot(x, self.phi_grad)
        
        # Quantum coherence from Q matrix
        quantum_factor = np.trace(self.Q)
        
        # Combined field strength
        strength = self.sigma * temporal_factor * (1 + spatial_factor + quantum_factor)
        
        return float(strength)
    
    def evolve_state(self, state: np.ndarray, dt: float) -> np.ndarray:
        """
        Evolve a quantum state using the transformation matrix S.
        
        Args:
            state: Input quantum state vector
            dt: Time step
            
        Returns:
            Evolved state vector
        """
        # Apply temporal coherence
        evolution_matrix = np.exp(-dt / self.tau) * self.S
        
        # Evolve the state
        evolved_state = np.dot(evolution_matrix, state)
        
        # Normalize if needed
        norm = np.linalg.norm(evolved_state)
        if norm > 0:
            evolved_state = evolved_state / norm
            
        return evolved_state
    
    def compute_gradient_flow(self, position: np.ndarray) -> np.ndarray:
        """
        Compute the gradient flow at a given position.
        
        Args:
            position: Spatial position vector
            
        Returns:
            Gradient flow vector
        """
        # Scale gradient by field strength
        flow = self.sigma * self.phi_grad
        
        # Apply quantum corrections from Q matrix
        if len(position) <= self.Q.shape[0]:
            quantum_correction = np.dot(self.Q[:len(position), :len(position)], position)
            flow = flow[:len(position)] + 0.1 * quantum_correction
            
        return flow
    
    def get_field_parameters(self) -> dict:
        """
        Get current field parameters as a dictionary.
        
        Returns:
            Dictionary containing all field parameters
        """
        return {
            'tau': self.tau,
            'Q_shape': self.Q.shape,
            'Q_trace': np.trace(self.Q),
            'phi_grad_norm': np.linalg.norm(self.phi_grad),
            'sigma': self.sigma,
            'S_shape': self.S.shape,
            'S_determinant': np.linalg.det(self.S)
        }
    
    def __repr__(self) -> str:
        """String representation of the LogicField."""
        return f"LogicField(tau={self.tau:.3f}, sigma={self.sigma:.3f}, Q_shape={self.Q.shape}, S_shape={self.S.shape})"