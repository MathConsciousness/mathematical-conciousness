"""Enhanced LogicField Class for Mathematical Consciousness Framework

This module implements the 5-tuple mathematical field L(x,t) = (τ, Q, ∇Φ, σ, S)
representing the core mathematical structure of the consciousness framework.
"""

from typing import Tuple, Optional
import numpy as np
from numpy.typing import NDArray


class LogicField:
    """Enhanced LogicField representing the 5-tuple mathematical field L(x,t) = (τ, Q, ∇Φ, σ, S)
    
    This class encapsulates the fundamental mathematical structure for consciousness
    modeling, combining intelligence levels, quality tensors, field gradients,
    coherence parameters, and state matrices.
    
    Attributes:
        tau (float): Intelligence level parameter
        Q (NDArray): Quality tensor
        phi_grad (NDArray): Field gradient ∇Φ
        sigma (float): Coherence parameter
        S (NDArray): State matrix
    """

    def __init__(
        self,
        tau: float,
        Q: NDArray[np.floating],
        phi_grad: NDArray[np.floating],
        sigma: float,
        S: NDArray[np.floating],
    ) -> None:
        """Initialize 5-tuple mathematical field L(x,t) = (τ, Q, ∇Φ, σ, S)
        
        Args:
            tau: Intelligence level parameter (must be positive)
            Q: Quality tensor (symmetric positive definite matrix)
            phi_grad: Field gradient vector
            sigma: Coherence parameter (0 ≤ σ ≤ 1)
            S: State matrix (square matrix)
            
        Raises:
            ValueError: If parameters don't meet mathematical constraints
        """
        self._validate_parameters(tau, Q, phi_grad, sigma, S)
        
        self.tau = tau
        self.Q = Q.copy()
        self.phi_grad = phi_grad.copy()
        self.sigma = sigma
        self.S = S.copy()
        
    def _validate_parameters(
        self,
        tau: float,
        Q: NDArray[np.floating],
        phi_grad: NDArray[np.floating],
        sigma: float,
        S: NDArray[np.floating],
    ) -> None:
        """Validate mathematical constraints on field parameters"""
        if tau <= 0:
            raise ValueError("Intelligence parameter τ must be positive")
            
        if sigma < 0 or sigma > 1:
            raise ValueError("Coherence parameter σ must be in [0, 1]")
            
        if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
            raise ValueError("Quality tensor Q must be a square matrix")
            
        if not np.allclose(Q, Q.T):
            raise ValueError("Quality tensor Q must be symmetric")
            
        # Check positive definiteness
        try:
            np.linalg.cholesky(Q)
        except np.linalg.LinAlgError:
            raise ValueError("Quality tensor Q must be positive definite")
            
        if phi_grad.ndim != 1:
            raise ValueError("Field gradient ∇Φ must be a vector")
            
        if S.ndim != 2 or S.shape[0] != S.shape[1]:
            raise ValueError("State matrix S must be square")

    def calculate_intelligence_level(self) -> float:
        """Calculate system intelligence level based on tau parameter
        
        Computes the effective intelligence level considering the interaction
        between the base intelligence parameter τ and the field characteristics.
        
        Returns:
            float: Effective intelligence level
        """
        # Intelligence level enhanced by field gradient magnitude
        grad_magnitude = np.linalg.norm(self.phi_grad)
        
        # Quality enhancement factor from trace of Q matrix
        quality_factor = np.trace(self.Q) / self.Q.shape[0]
        
        # Coherence modulation
        coherence_factor = 1 + self.sigma * np.log1p(grad_magnitude)
        
        effective_intelligence = (
            self.tau * quality_factor * coherence_factor
        )
        
        return float(effective_intelligence)

    def compute_coherence(self) -> float:
        """Compute system coherence preservation
        
        Calculates how well the system maintains coherence across its
        mathematical structure, considering field gradients and state evolution.
        
        Returns:
            float: Coherence preservation measure [0, 1]
        """
        # Base coherence from sigma parameter
        base_coherence = self.sigma
        
        # Field stability contribution
        if np.allclose(self.phi_grad, 0):
            field_stability = 1.0
        else:
            # Normalized gradient magnitude
            grad_norm = np.linalg.norm(self.phi_grad)
            field_stability = np.exp(-grad_norm / (1 + grad_norm))
        
        # State matrix stability (condition number)
        try:
            condition_num = np.linalg.cond(self.S)
            state_stability = 1.0 / (1.0 + np.log1p(condition_num))
        except np.linalg.LinAlgError:
            state_stability = 0.0
        
        # Quality tensor contribution (eigenvalue spread)
        eigenvals = np.linalg.eigvals(self.Q)
        quality_coherence = np.min(eigenvals) / np.max(eigenvals)
        
        # Combined coherence measure
        total_coherence = (
            base_coherence * field_stability * state_stability * quality_coherence
        )
        
        return float(np.clip(total_coherence, 0.0, 1.0))

    def field_energy(self) -> float:
        """Compute total field energy
        
        Returns:
            float: Total energy of the mathematical field
        """
        # Kinetic energy from gradient
        kinetic = 0.5 * np.dot(self.phi_grad, self.phi_grad)
        
        # Potential energy from quality tensor
        potential = 0.5 * np.trace(self.Q @ self.S @ self.S.T)
        
        # Intelligence contribution
        intelligence_energy = self.tau * self.sigma
        
        return float(kinetic + potential + intelligence_energy)

    def evolve_field(self, dt: float, external_force: Optional[NDArray[np.floating]] = None) -> None:
        """Evolve the field forward in time
        
        Args:
            dt: Time step
            external_force: Optional external force vector
        """
        if external_force is not None and external_force.shape != self.phi_grad.shape:
            raise ValueError("External force must have same shape as field gradient")
        
        # Update field gradient (simple evolution equation)
        acceleration = -self.sigma * self.phi_grad
        if external_force is not None:
            acceleration += external_force / self.tau
            
        self.phi_grad += acceleration * dt
        
        # Update coherence based on field evolution
        energy_change = np.linalg.norm(acceleration) * dt
        self.sigma *= np.exp(-energy_change * 0.01)  # Small damping
        self.sigma = np.clip(self.sigma, 0.0, 1.0)

    def get_field_tuple(self) -> Tuple[float, NDArray[np.floating], NDArray[np.floating], float, NDArray[np.floating]]:
        """Get the complete 5-tuple field representation
        
        Returns:
            Tuple containing (τ, Q, ∇Φ, σ, S)
        """
        return (self.tau, self.Q.copy(), self.phi_grad.copy(), self.sigma, self.S.copy())

    def __repr__(self) -> str:
        return (
            f"LogicField(tau={self.tau:.3f}, Q.shape={self.Q.shape}, "
            f"phi_grad.shape={self.phi_grad.shape}, sigma={self.sigma:.3f}, "
            f"S.shape={self.S.shape})"
        )