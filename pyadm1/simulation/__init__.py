# pyadm1/simulation/__init__.py
"""
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
"""

from simulator import Simulator
from parallel import ParallelSimulator, ScenarioResult, ParameterSweepConfig, MonteCarloConfig

# Future imports (currently stubs)
# from pyadm1.simulation.scenarios import (
#     ScenarioManager,
#     Scenario,
#     ParameterSweep,
# )
# from pyadm1.simulation.time_series import (
#     TimeSeries,
#     TimeSeriesInterpolator,
# )
# from pyadm1.simulation.results import (
#     SimulationResults,
#     ResultsAnalyzer,
#     ResultsExporter,
# )

__all__ = [
    "Simulator",
    "ParallelSimulator",
    "ScenarioResult",
    "ParameterSweepConfig",
    "MonteCarloConfig",
    # "ScenarioManager",
    # "Scenario",
    # "ParameterSweep",
    # "TimeSeries",
    # "TimeSeriesInterpolator",
    # "SimulationResults",
    # "ResultsAnalyzer",
    # "ResultsExporter",
]
