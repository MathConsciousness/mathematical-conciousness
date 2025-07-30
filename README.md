# Mathematical Consciousness Framework

An advanced scientific computing framework for mathematical consciousness modeling, featuring quantum simulation, agent-based computing, and network optimization.

## Features

### Core Components

- **LogicField**: Enhanced 5-tuple mathematical field L(x,t) = (τ, Q, ∇Φ, σ, S)
  - Intelligence level calculation with τ parameter
  - Coherence preservation with σ parameter  
  - Field gradient evolution and energy computation
  - Mathematical validation and stability analysis

- **Agent System**: Computational agents with specialized capabilities
  - Mathematical analysis agents
  - Scientific computing agents
  - Quantum simulation agents
  - Configurable capabilities and operations

- **Network Architecture**: Intelligent agent network construction
  - Target density optimization (default 0.964)
  - Information flow optimization
  - Compatibility-based connections
  - Load balancing and robustness metrics

- **Scientific Protocols**: Advanced computational capabilities
  - Quantum field simulation (exact, Runge-Kutta, Suzuki-Trotter)
  - Temporal pattern analysis and anomaly detection
  - Parameter optimization and differential equation solving
  - Harmonic oscillator and quantum state modeling

### FastAPI Interface

Complete REST API with endpoints for:
- Agent deployment and management
- Network creation and optimization
- Scientific computing operations
- System monitoring and health checks

## Installation

```bash
# Clone the repository
git clone https://github.com/MathConsciousness/mathematical-conciousness.git
cd mathematical-conciousness

# Install dependencies
pip install -e .

# Install development dependencies (optional)
pip install pytest pytest-cov pytest-asyncio psutil
```

## Quick Start

### Basic Usage

```python
import numpy as np
from src.core.logic_field import LogicField
from src.core.agent_factory import AgentFactory
from src.network.builder import NetworkBuilder
from src.computing.protocols import ScientificProtocols

# Create a LogicField
tau = 1.5
Q = np.eye(3)
phi_grad = np.array([0.1, 0.2, 0.3])
sigma = 0.8
S = np.eye(3)

field = LogicField(tau, Q, phi_grad, sigma, S)
print(f"Intelligence Level: {field.calculate_intelligence_level():.3f}")
print(f"Coherence: {field.compute_coherence():.3f}")

# Create agents
factory = AgentFactory()
agent = factory.create_mathematical_agent({"tau": 1.0})

# Execute operations
result = agent.execute_operation("analyze_field")
print(f"Field analysis: {result}")

# Build networks
agents = [factory.create_mathematical_agent({"tau": 1.0 + i*0.1}) for i in range(5)]
builder = NetworkBuilder()
network = builder.build_network(agents, target_density=0.6)

# Scientific computing
protocols = ScientificProtocols()
hamiltonian = np.array([[1, 0.5], [0.5, -1]], dtype=np.complex128)
initial_state = np.array([1, 0], dtype=np.complex128)
time_steps = np.linspace(0, 1, 20)

quantum_result = protocols.quantum_simulation({
    "hamiltonian": hamiltonian,
    "initial_state": initial_state,
    "time_steps": time_steps,
    "method": "exact"
})
```

### API Server

```bash
# Start the FastAPI server
python main.py

# Access interactive documentation at:
# http://localhost:8000/docs
```

### Run Demo

```bash
# Run the comprehensive demonstration
python examples/demo.py
```

## API Endpoints

### Agent Management
- `POST /api/v1/deploy/agents` - Deploy computational agents
- `GET /api/v1/agents` - List all active agents
- `GET /api/v1/agents/{agent_id}` - Get specific agent details
- `POST /api/v1/agents/{agent_id}/execute` - Execute agent operations

### Network Management
- `POST /api/v1/network/create` - Create agent networks
- `GET /api/v1/network/status` - Get network topology and metrics
- `POST /api/v1/network/{network_id}/optimize` - Optimize network flow

### Scientific Computing
- `POST /api/v1/compute/quantum-simulation` - Run quantum simulations
- `POST /api/v1/compute/temporal-analysis` - Analyze temporal patterns

### System Monitoring
- `GET /api/v1/system/status` - Get overall system status
- `GET /health` - Health check endpoint

## Testing

```bash
# Run all tests
python -m pytest

# Run specific test modules
python -m pytest tests/test_logic_field.py
python -m pytest tests/test_agents.py
python -m pytest tests/test_network.py
python -m pytest tests/test_protocols.py
python -m pytest tests/test_performance.py

# Run with coverage
python -m pytest --cov=src
```

## Architecture

### Mathematical Foundation

The framework is built on the 5-tuple mathematical field:

**L(x,t) = (τ, Q, ∇Φ, σ, S)**

Where:
- **τ**: Intelligence level parameter
- **Q**: Quality tensor (symmetric positive definite)
- **∇Φ**: Field gradient vector
- **σ**: Coherence parameter [0,1]
- **S**: State matrix

### Agent Types

1. **Mathematical Agents**: Specialized for mathematical analysis
   - Field operations and derivatives
   - Matrix computations
   - Optimization algorithms

2. **Scientific Agents**: Advanced scientific computing
   - Numerical integration
   - Pattern recognition
   - Quantum simulation capabilities

3. **Quantum Agents**: Quantum system modeling
   - Quantum state evolution
   - Hamiltonian simulation
   - Coherence analysis

### Network Optimization

- **Compatibility-based connections**: Agents connect based on type, capabilities, and field characteristics
- **Information flow optimization**: Dynamic topology adjustment for efficiency
- **Load balancing**: Distribute computational load across network
- **Robustness analysis**: Network resilience to node failures

## Performance

The framework is optimized for:
- **Agent Creation**: < 100ms per agent
- **Network Building**: < 3s for 20 agents  
- **Quantum Simulation**: < 1s for small systems
- **Memory Usage**: < 1MB per agent
- **Mathematical Computations**: < 1ms per field operation

## Dependencies

- **numpy** >= 1.21.0: Numerical computing
- **scipy** >= 1.7.0: Scientific algorithms
- **networkx** >= 2.6.0: Graph algorithms
- **fastapi** >= 0.68.0: API framework
- **pydantic** >= 1.8.0: Data validation
- **uvicorn** >= 0.15.0: ASGI server

## Development

### Project Structure

```
mathematical-conciousness/
├── src/
│   ├── core/           # Core mathematical components
│   ├── network/        # Network architecture
│   ├── computing/      # Scientific protocols
│   └── api/           # FastAPI endpoints
├── tests/             # Comprehensive test suite
├── examples/          # Usage examples
├── main.py           # API server entry point
└── pyproject.toml    # Project configuration
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Citation

If you use this framework in research, please cite:

```
Mathematical Consciousness Framework
Version 0.1.0
https://github.com/MathConsciousness/mathematical-conciousness
```
