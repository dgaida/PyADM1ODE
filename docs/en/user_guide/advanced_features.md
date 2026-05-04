# Advanced Features

PyADM1ODE provides advanced features for large-scale studies and configuration management.

## Parallel Simulation

Run multiple scenarios concurrently to speed up parameter sweeps or Monte Carlo simulations.

```python
from pyadm1.simulation import ParallelSimulator

# Parameter sweep
parallel = ParallelSimulator(adm1, n_workers=4)
scenarios = [
    {"k_dis": 0.5, "Q": [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]},
    {"k_dis": 0.6, "Q": [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]},
    {"k_dis": 0.7, "Q": [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]}
]

results = parallel.run_scenarios(scenarios, duration=30, initial_state=state)
```

See the [Parallel Simulation Example](../examples/parallel_simulation.md) for more details.

## Configuration Management

Save and reuse plant designs using JSON serialization.

```python
# Save configuration
plant.to_json("two_stage_plant.json")

# Load later
from pyadm1.configurator import BiogasPlant
plant = BiogasPlant.from_json("two_stage_plant.json", feedstock)
plant.initialize()
results = plant.simulate(duration=30, dt=1/24)
```

## PyADM1ODE Ecosystem

PyADM1ODE is part of a larger ecosystem of tools for biogas plant optimization and control:

- **[PyADM1ODE_calibration](https://github.com/dgaida/PyADM1ODE_calibration)**: Tools for automated calibration of ADM1 parameters based on measurement data.
- **[PyADM1ODE_mcp](https://github.com/dgaida/PyADM1ODE_mcp)**: Model Context Protocol (MCP) server that enables controlling PyADM1ODE with LLMs (like Claude or ChatGPT).
- **[PyADM1ODE_estimate](https://github.com/dgaida/PyADM1ODE_estimate)**: Tools for real-time estimation of state variables and parameters.
