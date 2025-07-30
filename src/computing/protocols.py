"""Scientific Computing Protocols for Mathematical Consciousness Framework

This module provides advanced scientific computing capabilities including
quantum simulation, temporal analysis, and specialized computational protocols.
"""

from typing import Dict, Any, List, Optional, Callable, Tuple, Union
import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass
from scipy import integrate, optimize, signal, linalg
from scipy.special import hermite, factorial
import warnings


@dataclass
class QuantumState:
    """Represents a quantum state with associated properties"""
    amplitudes: NDArray[np.complex128]
    basis_labels: List[str]
    dimension: int
    
    def __post_init__(self) -> None:
        """Validate quantum state after initialization"""
        if len(self.amplitudes) != self.dimension:
            raise ValueError("Amplitudes length must match dimension")
        if not np.isclose(np.sum(np.abs(self.amplitudes)**2), 1.0):
            warnings.warn("State is not normalized, normalizing automatically")
            norm = np.sqrt(np.sum(np.abs(self.amplitudes)**2))
            self.amplitudes = self.amplitudes / norm


@dataclass
class TemporalPattern:
    """Represents detected temporal patterns in data"""
    pattern_type: str
    frequency: float
    amplitude: float
    phase: float
    confidence: float
    period: float
    
    
@dataclass
class SimulationResult:
    """Contains results from scientific simulations"""
    data: NDArray[np.floating]
    metadata: Dict[str, Any]
    success: bool
    computation_time: float
    convergence_info: Optional[Dict[str, Any]] = None


class ScientificProtocols:
    """Advanced scientific computing protocols
    
    Provides quantum simulation, temporal analysis, optimization,
    and other specialized scientific computing capabilities.
    """

    def __init__(self, precision: str = "double", random_seed: Optional[int] = None) -> None:
        """Initialize scientific protocols
        
        Args:
            precision: Numerical precision ("single", "double", "quad")
            random_seed: Optional seed for reproducible computations
        """
        self.precision = precision
        self.dtype = self._get_dtype(precision)
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Cache for expensive computations
        self._quantum_cache: Dict[str, Any] = {}
        self._optimization_cache: Dict[str, Any] = {}

    def _get_dtype(self, precision: str) -> np.dtype:
        """Get numpy dtype for specified precision"""
        if precision == "single":
            return np.float32
        elif precision == "double":
            return np.float64
        elif precision == "quad":
            return np.longdouble
        else:
            raise ValueError(f"Unknown precision: {precision}")

    def quantum_simulation(self, params: Dict[str, Any]) -> NDArray[np.complex128]:
        """Run quantum field simulation
        
        Args:
            params: Simulation parameters including:
                - hamiltonian: Hamiltonian operator (matrix)
                - initial_state: Initial quantum state vector
                - time_steps: Array of time points
                - method: Integration method ("exact", "runge_kutta", "suzuki_trotter")
                - trotter_steps: Steps for Trotter decomposition (if applicable)
                
        Returns:
            NDArray: Quantum state evolution over time
            
        Raises:
            ValueError: If required parameters are missing or invalid
        """
        # Validate required parameters
        required_params = ["hamiltonian", "initial_state", "time_steps"]
        for param in required_params:
            if param not in params:
                raise ValueError(f"Missing required parameter: {param}")
        
        hamiltonian = np.array(params["hamiltonian"], dtype=np.complex128)
        initial_state = np.array(params["initial_state"], dtype=np.complex128)
        time_steps = np.array(params["time_steps"], dtype=self.dtype)
        method = params.get("method", "exact")
        
        # Validate dimensions
        if hamiltonian.ndim != 2 or hamiltonian.shape[0] != hamiltonian.shape[1]:
            raise ValueError("Hamiltonian must be a square matrix")
        
        if initial_state.ndim != 1 or len(initial_state) != hamiltonian.shape[0]:
            raise ValueError("Initial state dimension must match Hamiltonian")
        
        # Normalize initial state
        initial_state = initial_state / np.linalg.norm(initial_state)
        
        # Choose simulation method
        if method == "exact":
            return self._quantum_simulation_exact(hamiltonian, initial_state, time_steps)
        elif method == "runge_kutta":
            return self._quantum_simulation_rk(hamiltonian, initial_state, time_steps)
        elif method == "suzuki_trotter":
            trotter_steps = params.get("trotter_steps", 100)
            return self._quantum_simulation_trotter(hamiltonian, initial_state, time_steps, trotter_steps)
        else:
            raise ValueError(f"Unknown simulation method: {method}")

    def _quantum_simulation_exact(
        self, 
        hamiltonian: NDArray[np.complex128], 
        initial_state: NDArray[np.complex128], 
        time_steps: NDArray[np.floating]
    ) -> NDArray[np.complex128]:
        """Exact quantum simulation using matrix exponentiation"""
        n_steps = len(time_steps)
        n_dim = len(initial_state)
        
        evolution = np.zeros((n_steps, n_dim), dtype=np.complex128)
        evolution[0] = initial_state
        
        for i, t in enumerate(time_steps[1:], 1):
            dt = t - time_steps[i-1]
            evolution_operator = linalg.expm(-1j * hamiltonian * dt)
            evolution[i] = evolution_operator @ evolution[i-1]
        
        return evolution

    def _quantum_simulation_rk(
        self, 
        hamiltonian: NDArray[np.complex128], 
        initial_state: NDArray[np.complex128], 
        time_steps: NDArray[np.floating]
    ) -> NDArray[np.complex128]:
        """Quantum simulation using Runge-Kutta integration"""
        def schrodinger_eq(t: float, psi: NDArray[np.complex128]) -> NDArray[np.complex128]:
            return -1j * hamiltonian @ psi
        
        # Flatten for scipy integration
        psi_0_real = np.concatenate([initial_state.real, initial_state.imag])
        
        def real_schrodinger(t: float, psi_real: NDArray[np.floating]) -> NDArray[np.floating]:
            n = len(psi_real) // 2
            psi = psi_real[:n] + 1j * psi_real[n:]
            dpsi_dt = schrodinger_eq(t, psi)
            return np.concatenate([dpsi_dt.real, dpsi_dt.imag])
        
        sol = integrate.solve_ivp(
            real_schrodinger, 
            [time_steps[0], time_steps[-1]], 
            psi_0_real,
            t_eval=time_steps,
            method='RK45',
            rtol=1e-8,
            atol=1e-10
        )
        
        n_dim = len(initial_state)
        evolution = np.zeros((len(time_steps), n_dim), dtype=np.complex128)
        
        for i in range(len(time_steps)):
            real_part = sol.y[:n_dim, i]
            imag_part = sol.y[n_dim:, i]
            evolution[i] = real_part + 1j * imag_part
        
        return evolution

    def _quantum_simulation_trotter(
        self, 
        hamiltonian: NDArray[np.complex128], 
        initial_state: NDArray[np.complex128], 
        time_steps: NDArray[np.floating],
        trotter_steps: int
    ) -> NDArray[np.complex128]:
        """Quantum simulation using Suzuki-Trotter decomposition"""
        # For simplicity, implement first-order Trotter decomposition
        # H = H1 + H2 + ... -> exp(-iHt) ≈ [exp(-iH1*dt/n) * exp(-iH2*dt/n) * ...]^n
        
        # Decompose Hamiltonian (simplified: just use full Hamiltonian)
        n_steps = len(time_steps)
        n_dim = len(initial_state)
        
        evolution = np.zeros((n_steps, n_dim), dtype=np.complex128)
        evolution[0] = initial_state
        
        for i, t in enumerate(time_steps[1:], 1):
            dt = t - time_steps[i-1]
            small_dt = dt / trotter_steps
            
            state = evolution[i-1].copy()
            for _ in range(trotter_steps):
                evolution_operator = linalg.expm(-1j * hamiltonian * small_dt)
                state = evolution_operator @ state
            
            evolution[i] = state
        
        return evolution

    def analyze_temporal_patterns(self, data: NDArray[np.floating]) -> Dict[str, Any]:
        """Perform temporal data analysis
        
        Args:
            data: Time series data (1D or 2D array)
            
        Returns:
            Dict containing detected patterns and analysis results
        """
        if data.ndim > 2:
            raise ValueError("Data must be 1D or 2D array")
        
        if data.ndim == 1:
            data = data.reshape(1, -1)
        
        results = {
            "patterns": [],
            "statistics": {},
            "frequency_analysis": {},
            "trend_analysis": {},
            "anomalies": {},
        }
        
        for i, series in enumerate(data):
            # Basic statistics
            stats = self._compute_temporal_statistics(series)
            results["statistics"][f"series_{i}"] = stats
            
            # Frequency analysis
            freq_analysis = self._analyze_frequencies(series)
            results["frequency_analysis"][f"series_{i}"] = freq_analysis
            
            # Pattern detection
            patterns = self._detect_temporal_patterns(series)
            results["patterns"].extend(patterns)
            
            # Trend analysis
            trend = self._analyze_trends(series)
            results["trend_analysis"][f"series_{i}"] = trend
            
            # Anomaly detection
            anomalies = self._detect_anomalies(series)
            results["anomalies"][f"series_{i}"] = anomalies
        
        return results

    def _compute_temporal_statistics(self, series: NDArray[np.floating]) -> Dict[str, float]:
        """Compute basic temporal statistics"""
        return {
            "mean": float(np.mean(series)),
            "std": float(np.std(series)),
            "variance": float(np.var(series)),
            "skewness": float(self._compute_skewness(series)),
            "kurtosis": float(self._compute_kurtosis(series)),
            "autocorrelation_lag1": float(self._autocorrelation(series, 1)),
            "trend_slope": float(self._linear_trend_slope(series)),
        }

    def _compute_skewness(self, data: NDArray[np.floating]) -> float:
        """Compute skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)

    def _compute_kurtosis(self, data: NDArray[np.floating]) -> float:
        """Compute kurtosis of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3.0

    def _autocorrelation(self, series: NDArray[np.floating], lag: int) -> float:
        """Compute autocorrelation at specified lag"""
        if lag >= len(series):
            return 0.0
        
        n = len(series)
        mean = np.mean(series)
        
        numerator = np.sum((series[:-lag] - mean) * (series[lag:] - mean))
        denominator = np.sum((series - mean) ** 2)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator

    def _linear_trend_slope(self, series: NDArray[np.floating]) -> float:
        """Compute slope of linear trend"""
        x = np.arange(len(series))
        return np.polyfit(x, series, 1)[0]

    def _analyze_frequencies(self, series: NDArray[np.floating]) -> Dict[str, Any]:
        """Analyze frequency content of time series"""
        # FFT analysis
        fft_result = np.fft.fft(series)
        frequencies = np.fft.fftfreq(len(series))
        magnitude = np.abs(fft_result)
        
        # Find dominant frequencies
        positive_freq_idx = frequencies > 0
        pos_frequencies = frequencies[positive_freq_idx]
        pos_magnitudes = magnitude[positive_freq_idx]
        
        if len(pos_magnitudes) > 0:
            dominant_idx = np.argmax(pos_magnitudes)
            dominant_frequency = pos_frequencies[dominant_idx]
            dominant_magnitude = pos_magnitudes[dominant_idx]
        else:
            dominant_frequency = 0.0
            dominant_magnitude = 0.0
        
        # Power spectral density
        power_spectrum = magnitude ** 2
        total_power = np.sum(power_spectrum)
        
        return {
            "dominant_frequency": float(dominant_frequency),
            "dominant_magnitude": float(dominant_magnitude),
            "total_power": float(total_power),
            "frequency_distribution": {
                "frequencies": frequencies.tolist(),
                "magnitudes": magnitude.tolist(),
            },
            "spectral_centroid": float(np.sum(frequencies * magnitude) / np.sum(magnitude)) if np.sum(magnitude) > 0 else 0.0,
        }

    def _detect_temporal_patterns(self, series: NDArray[np.floating]) -> List[TemporalPattern]:
        """Detect temporal patterns in time series"""
        patterns = []
        
        # Periodic pattern detection using autocorrelation
        max_lag = min(len(series) // 4, 100)
        autocorr = [self._autocorrelation(series, lag) for lag in range(1, max_lag)]
        
        # Find peaks in autocorrelation (indicating periodicity)
        if len(autocorr) > 2:
            peaks, properties = signal.find_peaks(autocorr, height=0.3, distance=5)
            
            for peak in peaks:
                period = peak + 1  # +1 because we started from lag=1
                confidence = autocorr[peak]
                frequency = 1.0 / period if period > 0 else 0.0
                
                patterns.append(TemporalPattern(
                    pattern_type="periodic",
                    frequency=frequency,
                    amplitude=confidence,
                    phase=0.0,  # Simplified
                    confidence=confidence,
                    period=float(period),
                ))
        
        # Trend pattern detection
        slope = self._linear_trend_slope(series)
        if abs(slope) > 0.01 * np.std(series) / len(series):
            patterns.append(TemporalPattern(
                pattern_type="trend",
                frequency=0.0,
                amplitude=abs(slope),
                phase=0.0,
                confidence=min(1.0, abs(slope) / np.std(series)),
                period=float('inf'),
            ))
        
        return patterns

    def _analyze_trends(self, series: NDArray[np.floating]) -> Dict[str, Any]:
        """Analyze trends in time series"""
        # Linear trend
        x = np.arange(len(series))
        linear_fit = np.polyfit(x, series, 1)
        linear_trend = np.polyval(linear_fit, x)
        
        # Quadratic trend
        quad_fit = np.polyfit(x, series, 2)
        quad_trend = np.polyval(quad_fit, x)
        
        # Detrended series
        detrended_linear = series - linear_trend
        detrended_quad = series - quad_trend
        
        return {
            "linear_slope": float(linear_fit[0]),
            "linear_intercept": float(linear_fit[1]),
            "linear_r_squared": float(1 - np.var(detrended_linear) / np.var(series)) if np.var(series) > 0 else 0.0,
            "quadratic_coefficients": quad_fit.tolist(),
            "quadratic_r_squared": float(1 - np.var(detrended_quad) / np.var(series)) if np.var(series) > 0 else 0.0,
            "trend_strength": float(abs(linear_fit[0]) / np.std(series)) if np.std(series) > 0 else 0.0,
        }

    def _detect_anomalies(self, series: NDArray[np.floating]) -> Dict[str, Any]:
        """Detect anomalies in time series"""
        # Statistical anomaly detection using z-score
        mean = np.mean(series)
        std = np.std(series)
        
        if std == 0:
            z_scores = np.zeros_like(series)
        else:
            z_scores = np.abs((series - mean) / std)
        
        anomaly_threshold = 3.0
        anomaly_indices = np.where(z_scores > anomaly_threshold)[0]
        
        # Change point detection (simplified)
        change_points = []
        if len(series) > 10:
            window_size = max(5, len(series) // 20)
            for i in range(window_size, len(series) - window_size):
                before = series[i-window_size:i]
                after = series[i:i+window_size]
                
                # Statistical test for difference in means
                if len(before) > 0 and len(after) > 0:
                    mean_diff = abs(np.mean(after) - np.mean(before))
                    pooled_std = np.sqrt((np.var(before) + np.var(after)) / 2)
                    
                    if pooled_std > 0 and mean_diff / pooled_std > 2.0:
                        change_points.append(i)
        
        return {
            "statistical_anomalies": {
                "indices": anomaly_indices.tolist(),
                "values": series[anomaly_indices].tolist(),
                "z_scores": z_scores[anomaly_indices].tolist(),
            },
            "change_points": change_points,
            "anomaly_score": float(len(anomaly_indices) / len(series)),
        }

    def optimize_system_parameters(
        self, 
        objective_function: Callable,
        initial_params: NDArray[np.floating],
        bounds: Optional[List[Tuple[float, float]]] = None,
        method: str = "L-BFGS-B",
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Optimize system parameters using advanced algorithms
        
        Args:
            objective_function: Function to minimize
            initial_params: Initial parameter values
            bounds: Optional parameter bounds
            method: Optimization method
            **kwargs: Additional optimization parameters
            
        Returns:
            Dict containing optimization results
        """
        try:
            result = optimize.minimize(
                objective_function,
                initial_params,
                method=method,
                bounds=bounds,
                **kwargs
            )
            
            return {
                "optimal_params": result.x.tolist(),
                "optimal_value": float(result.fun),
                "success": bool(result.success),
                "message": str(result.message),
                "iterations": int(result.nit) if hasattr(result, 'nit') else -1,
                "function_evaluations": int(result.nfev) if hasattr(result, 'nfev') else -1,
            }
        
        except Exception as e:
            return {
                "optimal_params": initial_params.tolist(),
                "optimal_value": float('inf'),
                "success": False,
                "message": f"Optimization failed: {str(e)}",
                "iterations": 0,
                "function_evaluations": 0,
            }

    def solve_differential_equation(
        self,
        ode_function: Callable,
        initial_conditions: NDArray[np.floating],
        time_span: Tuple[float, float],
        time_points: Optional[NDArray[np.floating]] = None,
        method: str = "RK45",
        **kwargs: Any
    ) -> SimulationResult:
        """Solve ordinary differential equations
        
        Args:
            ode_function: ODE function dy/dt = f(t, y)
            initial_conditions: Initial values
            time_span: (t_start, t_end)
            time_points: Optional specific time points to evaluate
            method: Integration method
            **kwargs: Additional solver parameters
            
        Returns:
            SimulationResult with solution data
        """
        import time
        start_time = time.time()
        
        try:
            if time_points is None:
                time_points = np.linspace(time_span[0], time_span[1], 100)
            
            solution = integrate.solve_ivp(
                ode_function,
                time_span,
                initial_conditions,
                t_eval=time_points,
                method=method,
                **kwargs
            )
            
            computation_time = time.time() - start_time
            
            return SimulationResult(
                data=solution.y.T,  # Transpose to have time as first axis
                metadata={
                    "time_points": solution.t.tolist(),
                    "method": method,
                    "success": solution.success,
                    "message": solution.message,
                    "nfev": solution.nfev,
                },
                success=solution.success,
                computation_time=computation_time,
                convergence_info={
                    "final_time": float(solution.t[-1]),
                    "final_state": solution.y[:, -1].tolist(),
                }
            )
        
        except Exception as e:
            computation_time = time.time() - start_time
            return SimulationResult(
                data=np.array([]),
                metadata={"error": str(e)},
                success=False,
                computation_time=computation_time,
            )

    def create_quantum_harmonic_oscillator(self, n_levels: int, omega: float = 1.0) -> Dict[str, NDArray[np.floating]]:
        """Create quantum harmonic oscillator Hamiltonian and states
        
        Args:
            n_levels: Number of energy levels
            omega: Oscillator frequency
            
        Returns:
            Dict containing Hamiltonian, energies, and basis states
        """
        # Creation and annihilation operators
        a_dag = np.zeros((n_levels, n_levels), dtype=np.complex128)
        a = np.zeros((n_levels, n_levels), dtype=np.complex128)
        
        for n in range(n_levels - 1):
            a_dag[n+1, n] = np.sqrt(n + 1)
            a[n, n+1] = np.sqrt(n + 1)
        
        # Number operator
        number_op = a_dag @ a
        
        # Hamiltonian H = ħω(a†a + 1/2)
        hamiltonian = omega * (number_op + 0.5 * np.eye(n_levels))
        
        # Energy eigenvalues
        energies = omega * (np.arange(n_levels) + 0.5)
        
        # Basis states (Fock states)
        basis_states = np.eye(n_levels, dtype=np.complex128)
        
        return {
            "hamiltonian": hamiltonian,
            "energies": energies,
            "basis_states": basis_states,
            "creation_operator": a_dag,
            "annihilation_operator": a,
            "number_operator": number_op,
        }