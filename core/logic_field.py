"""LogicField implementation for the Mathematical Framework System."""

import numpy as np
from typing import Optional, Union


class LogicField:
    """
    Logic field implementing mathematical consciousness framework.
    
    This class represents the core logic field that calculates intelligence
    levels based on mathematical parameters including tau, Q matrix,
    phi gradients, sigma, and S matrix.
    """
    
    def __init__(
        self,
        tau: float,
        Q: np.ndarray,
        phi_grad: np.ndarray,
        sigma: float,
        S: np.ndarray
    ):
        """
        Initialize the LogicField.
        
        Args:
            tau: Coherence parameter (should be between 0 and 1)
            Q: Quantum matrix representation
            phi_grad: Phi gradient vector
            sigma: Sigma parameter for field strength
            S: State matrix
        """
        self.tau = tau
        self.Q = Q
        self.phi_grad = phi_grad
        self.sigma = sigma
        self.S = S
        
        # Validate parameters
        self._validate_parameters()
    
    def _validate_parameters(self) -> None:
        """Validate input parameters."""
        if not 0 <= self.tau <= 1:
            raise ValueError("tau must be between 0 and 1")
        
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        
        if not isinstance(self.Q, np.ndarray) or not isinstance(self.S, np.ndarray):
            raise ValueError("Q and S must be numpy arrays")
    
    def calculate_intelligence_level(self) -> float:
        """
        Calculate the intelligence level of the logic field.
        
        This method computes the intelligence level based on the
        mathematical consciousness framework using the field parameters.
        
        Returns:
            float: Intelligence level (typically between 0 and 1)
        """
        # Core intelligence calculation based on mathematical consciousness
        # Using tau as base coherence, enhanced by matrix operations
        
        # Calculate quantum coherence contribution
        if self.Q.size > 0:
            q_trace = np.trace(self.Q) if self.Q.ndim == 2 else np.sum(self.Q)
            q_contribution = min(abs(q_trace) / (self.Q.size), 1.0)
        else:
            q_contribution = 0.0
        
        # Calculate phi gradient contribution
        if self.phi_grad.size > 0:
            phi_magnitude = np.linalg.norm(self.phi_grad)
            phi_contribution = min(phi_magnitude / (1 + phi_magnitude), 0.3)
        else:
            phi_contribution = 0.0
        
        # Calculate state matrix contribution
        if self.S.size > 0:
            s_trace = np.trace(self.S) if self.S.ndim == 2 else np.sum(self.S)
            s_contribution = min(abs(s_trace) / (self.S.size * 10), 0.2)
        else:
            s_contribution = 0.0
        
        # Combine contributions with tau as primary factor
        base_intelligence = self.tau * (1 + q_contribution * 0.1)
        field_enhancement = self.sigma * (phi_contribution + s_contribution)
        
        intelligence_level = base_intelligence + field_enhancement
        
        # Ensure result is within reasonable bounds
        return min(max(intelligence_level, 0.0), 1.0)
    
    def get_field_strength(self) -> float:
        """
        Get the current field strength.
        
        Returns:
            float: Field strength value
        """
        return self.sigma
    
    def update_tau(self, new_tau: float) -> None:
        """
        Update the tau parameter.
        
        Args:
            new_tau: New tau value (must be between 0 and 1)
        """
        if not 0 <= new_tau <= 1:
            raise ValueError("tau must be between 0 and 1")
        self.tau = new_tau
    
    def get_parameters(self) -> dict:
        """
        Get current field parameters.
        
        Returns:
            dict: Dictionary containing all field parameters
        """
        return {
            'tau': self.tau,
            'Q_shape': self.Q.shape,
            'phi_grad_shape': self.phi_grad.shape,
            'sigma': self.sigma,
            'S_shape': self.S.shape
        }