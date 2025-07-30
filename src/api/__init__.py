"""
API Framework

This package contains REST API components for the mathematical
consciousness framework.
"""

from .rest import MathematicalConsciousnessAPI, start_server

__version__ = "1.0.0"
__author__ = "Mathematical Consciousness Framework Team"

__all__ = [
    "MathematicalConsciousnessAPI",
    "start_server"
]