# Simulation Engine

Simulation Engine

Core simulation functionality for single and parallel execution of biogas plant models.

The Simulator class has been refactored to use the new solver architecture from
pyadm1.core.solver, providing better separation of concerns and improved testability.

Modules:

    simulator: Main Simulator class orchestrating single plant simulation runs, managing
              time stepping using the ODESolver interface, state updates, and result
              collection with progress reporting and error handling.

    parallel: ParallelSimulator for concurrent execution of multiple scenarios with
             different parameter sets, substrate mixtures, or operating conditions
             using multiprocessing for efficient parameter sweeps and Monte Carlo
             simulations.

    scenarios: Scenario management system for defining, organizing, and comparing
              different simulation configurations including parameter variations,
              substrate schedules, and operational strategies with metadata tracking.

    time_series: Time series data handling for simulation inputs (substrate feeds,
                temperatures, prices) and outputs (gas production, concentrations,
                power) with interpolation, resampling, and statistical analysis.

    results: Result management and analysis including data extraction, aggregation,
            comparison between scenarios, statistical summaries, and export in
            various formats (CSV, JSON, HDF5) for further processing.

Example:

```python
    >>> from pyadm1.simulation import Simulator, ParallelSimulator
    >>> from pyadm1.core import ADM1, create_solver
    >>> from pyadm1.substrates import Feedstock
    >>>
    >>> # Create model
    >>> feedstock = Feedstock(feeding_freq=48)
    >>> adm1 = ADM1(feedstock)
    >>>
    >>> # Single simulation
    >>> solver = create_solver(method='BDF', rtol=1e-7)
    >>> simulator = Simulator(adm1, solver=solver)
    >>> initial_state = [0.01] * 37
    >>> final_state = simulator.simulate_AD_plant([0, 30], initial_state)
    >>>
    >>> # Parallel simulations
    >>> parallel = ParallelSimulator(adm1, n_workers=4)
    >>> scenarios = [{"k_dis": 0.5, "Q": [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]}]
    >>> results = parallel.run_scenarios(scenarios, duration=30, initial_state=initial_state)
```

## Base Classes

- [MonteCarloConfig](#montecarloconfig)
- [ParallelSimulator](#parallelsimulator)
- [ParameterSweepConfig](#parametersweepconfig)
- [ScenarioResult](#scenarioresult)
- [Simulator](#simulator)

### MonteCarloConfig

```python
from pyadm1.simulation import MonteCarloConfig
```

Configuration for Monte Carlo simulation.

Attributes:

    n_samples: Number of Monte Carlo samples
    parameter_distributions: Dict mapping parameter names to (mean, std) tuples
    fixed_params: Parameters to keep fixed
    seed: Random seed for reproducibility

**Signature:**

```python
MonteCarloConfig(
    n_samples,
    parameter_distributions,
    fixed_params=<factory>,
    seed=None
)
```

**Methods:**

**Attributes:**

- n_samples: Number of Monte Carlo samples
- parameter_distributions: Dict mapping parameter names to (mean, std) tuples
- fixed_params: Parameters to keep fixed
- seed: Random seed for reproducibility


### ParallelSimulator

```python
from pyadm1.simulation import ParallelSimulator
```

Parallel simulator for running multiple ADM1 scenarios concurrently.

Uses multiprocessing to distribute scenarios across CPU cores for efficient
parameter sweeps, sensitivity analysis, and Monte Carlo simulations.

Attributes:

    adm1: Base ADM1 model instance (will be copied for each worker)
    n_workers: Number of parallel worker processes
    verbose: Enable progress reporting

Example:

```python
    >>> parallel = ParallelSimulator(adm1, n_workers=4, verbose=True)
    >>> results = parallel.run_scenarios(scenarios, duration=30)
```

**Signature:**

```python
ParallelSimulator(
    adm1,
    n_workers=None,
    verbose=True
)
```

**Methods:**

#### `monte_carlo()`

```python
monte_carlo(config, duration, initial_state, kwargs)
```

Run Monte Carlo simulation with parameter uncertainty.

Samples parameters from normal distributions and runs multiple scenarios
to quantify uncertainty in predictions.

Args:

    config: MonteCarloConfig with distributions and sample count
    duration: Simulation duration [days]
    initial_state: Initial ADM1 state vector
    **kwargs: Additional arguments for run_scenarios

Returns:

    List of ScenarioResult objects

Example:

```python
    >>> config = MonteCarloConfig(
    ...     n_samples=100,
    ...     parameter_distributions={
    ...         "k_dis": (0.5, 0.05),  # mean=0.5, std=0.05
    ...         "Y_su": (0.10, 0.01)
    ...     },
    ...     fixed_params={"Q": [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]},
    ...     seed=42
    ... )
    >>> results = parallel.monte_carlo(config, duration=30, initial_state=state)
```

#### `multi_parameter_sweep()`

```python
multi_parameter_sweep(parameter_configs, duration, initial_state, fixed_params=None, kwargs)
```

Run multi-parameter sweep (full factorial design).

Tests all combinations of provided parameter values.

Args:

    parameter_configs: Dict mapping parameter names to value lists
    duration: Simulation duration [days]
    initial_state: Initial ADM1 state vector
    fixed_params: Parameters to keep fixed
    **kwargs: Additional arguments for run_scenarios

Returns:

    List of ScenarioResult objects

Example:

```python
    >>> parameter_configs = {
    ...     "k_dis": [0.4, 0.5, 0.6],
    ...     "Y_su": [0.09, 0.10, 0.11]
    ... }
    >>> results = parallel.multi_parameter_sweep(
    ...     parameter_configs,
    ...     duration=30,
    ...     initial_state=state,
    ...     fixed_params={"Q": [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]}
    ... )
```

#### `parameter_sweep()`

```python
parameter_sweep(config, duration, initial_state, kwargs)
```

Run parameter sweep for a single parameter.

Tests multiple values of one parameter while keeping others fixed.

Args:

    config: ParameterSweepConfig with parameter and values
    duration: Simulation duration [days]
    initial_state: Initial ADM1 state vector
    **kwargs: Additional arguments for run_scenarios

Returns:

    List of ScenarioResult objects

Example:

```python
    >>> config = ParameterSweepConfig(
    ...     parameter_name="k_dis",
    ...     values=[0.3, 0.4, 0.5, 0.6, 0.7],
    ...     other_params={"Q": [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]}
    ... )
    >>> results = parallel.parameter_sweep(config, duration=30, initial_state=state)
```

#### `run_scenarios()`

```python
run_scenarios(scenarios, duration, initial_state, dt=0.041666666666666664, compute_metrics=True, save_time_series=False)
```

Run multiple simulation scenarios in parallel.

Each scenario is a dictionary containing parameter values and substrate
feed rates. The simulator will run all scenarios concurrently and
collect results.

Args:

    scenarios: List of scenario dictionaries with parameters
    duration: Simulation duration [days]
    initial_state: Initial ADM1 state vector
    dt: Time step [days]
    compute_metrics: Calculate performance metrics
    save_time_series: Save full time series data

Returns:

    List of ScenarioResult objects

Example:

```python
    >>> scenarios = [
    ...     {"k_dis": 0.5, "Q": [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]},
    ...     {"k_dis": 0.6, "Q": [20, 10, 0, 0, 0, 0, 0, 0, 0, 0]},
    ... ]
    >>> results = parallel.run_scenarios(scenarios, duration=30, initial_state=state)
```

#### `summarize_results()`

```python
summarize_results(results, metrics=None)
```

Summarize results from multiple scenarios.

Computes statistics (mean, std, min, max) for each metric across
all successful scenarios.

Args:

    results: List of ScenarioResult objects
    metrics: List of metric names to summarize (default: all)

Returns:

    Dictionary with summary statistics

Example:

```python
    >>> summary = parallel.summarize_results(results)
    >>> print(f"Mean CH4: {summary['Q_ch4']['mean']:.1f} m³/d")
```

**Attributes:**

- adm1: Base ADM1 model instance (will be copied for each worker)
- n_workers: Number of parallel worker processes
- verbose: Enable progress reporting


### ParameterSweepConfig

```python
from pyadm1.simulation import ParameterSweepConfig
```

Configuration for parameter sweep.

Attributes:

    parameter_name: Name of parameter to sweep
    values: List of values to test
    other_params: Fixed parameters for all scenarios

**Signature:**

```python
ParameterSweepConfig(
    parameter_name,
    values,
    other_params=<factory>
)
```

**Methods:**

**Attributes:**

- parameter_name: Name of parameter to sweep
- values: List of values to test
- other_params: Fixed parameters for all scenarios


### ScenarioResult

```python
from pyadm1.simulation import ScenarioResult
```

Result from a single simulation scenario.

Attributes:

    scenario_id: Unique identifier for this scenario
    parameters: Parameter values used in this scenario
    success: Whether simulation completed successfully
    duration: Simulation duration [days]
    final_state: Final ADM1 state vector
    time_series: Optional time series data
    metrics: Computed performance metrics
    error: Error message if simulation failed
    execution_time: Wall clock time for execution [seconds]

**Signature:**

```python
ScenarioResult(
    scenario_id,
    parameters,
    success,
    duration,
    final_state=None,
    time_series=None,
    metrics=<factory>,
    error=None,
    execution_time=0.0
)
```

**Methods:**

**Attributes:**

- scenario_id: Unique identifier for this scenario
- parameters: Parameter values used in this scenario
- success: Whether simulation completed successfully
- duration: Simulation duration [days]
- final_state: Final ADM1 state vector
- time_series: Optional time series data
- metrics: Computed performance metrics
- error: Error message if simulation failed
- execution_time: Wall clock time for execution [seconds]


### Simulator

```python
from pyadm1.simulation import Simulator
```

Handles ADM1 simulation runs with various configurations.

This class provides high-level interfaces for running ADM1 simulations,
including single runs and multi-scenario optimization for substrate feed
determination.

Attributes:

    adm1: ADM1 model instance
    solver: ODE solver instance

Example:

```python
    >>> simulator = Simulator(adm1)
    >>> result = simulator.simulate_AD_plant([0, 10], initial_state)
```

**Signature:**

```python
Simulator(
    adm1,
    solver=None
)
```

**Methods:**

#### `determine_best_feed_by_n_sims()`

```python
determine_best_feed_by_n_sims(state_zero, Q, Qch4sp, feeding_freq, n=13)
```

Determine optimal substrate feed by running n simulations.

Runs n simulations with varying substrate feed rates around Q and
returns the feed rate yielding methane production closest to setpoint.

The first simulation uses Q, the 2nd and 3rd use Q ± 1.5 m³/d,
and remaining simulations use random variations.

Args:

    state_zero: Initial ADM1 state vector (37 elements)
    Q: Initial volumetric flow rates [m³/d], e.g. [15, 10, 0, ...]
    Qch4sp: Methane flow rate setpoint [m³/d]
    feeding_freq: Feeding frequency [hours]
    n: Number of simulations to run (default: 13, minimum: 3)

Returns:

    Tuple containing:
        - Q_Gas_7d_best: Best biogas production after 7 days [m³/d]
        - Q_CH4_7d_best: Best methane production after 7 days [m³/d]
        - Qbest: Best substrate feed rates [m³/d]
        - Q_Gas_7d_initial: Initial biogas production after 7 days [m³/d]
        - Q_CH4_7d_initial: Initial methane production after 7 days [m³/d]
        - Q_initial: Initial substrate feed rates [m³/d]
        - q_gas_best_2d: Best biogas after feeding_freq/24 days [m³/d]
        - q_ch4_best_2d: Best methane after feeding_freq/24 days [m³/d]
        - q_gas_2d: Initial biogas after feeding_freq/24 days [m³/d]
        - q_ch4_2d: Initial methane after feeding_freq/24 days [m³/d]

Example:

```python
    >>> result = simulator.determine_best_feed_by_n_sims(
    ...     state, [15, 10, 0, 0, 0, 0, 0, 0, 0, 0], 900, 48, n=13
    ... )
    >>> Q_best = result[2]
```

#### `simulate_AD_plant()`

```python
simulate_AD_plant(tstep, state_zero)
```

Simulate ADM1 for specified time span and return final state.

This is the main simulation method that integrates the ADM1 ODEs
and tracks process values for operator information.

Args:

    tstep: Time span [t_start, t_end] in days
    state_zero: Initial ADM1 state vector (37 elements)

Returns:

    Final ADM1 state vector after simulation (37 elements)

Example:

```python
    >>> final_state = simulator.simulate_AD_plant([0, 1], initial_state)
    >>> print(f"Final pH: {final_state[...])
```

**Attributes:**

- adm1: ADM1 model instance
- solver: ODE solver instance


