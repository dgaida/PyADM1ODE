"""
Simulation Engine

Core simulation functionality for single and parallel execution of biogas plant models.

Modules:
    simulator: Main Simulator class orchestrating single plant simulation runs, managing
              time stepping, component execution order, state updates, and result
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
    >>> from pyadm1.configurator import BiogasPlant
    >>>
    >>> # Single simulation
    >>> plant = BiogasPlant.from_json("plant.json", feedstock)
    >>> simulator = Simulator(plant)
    >>> results = simulator.simulate(duration=30, dt=1/24)
    >>>
    >>> # Parallel parameter sweep
    >>> scenarios = [
    ...     {"k_dis": 0.5, "Y_su": 0.1},
    ...     {"k_dis": 0.6, "Y_su": 0.11},
    ...     {"k_dis": 0.7, "Y_su": 0.12},
    ... ]
    >>> parallel_sim = ParallelSimulator(plant, n_workers=4)
    >>> results = parallel_sim.run_scenarios(scenarios, duration=30)
"""

from pyadm1.simulation.simulator import Simulator

# from pyadm1.simulation.parallel import ParallelSimulator, ScenarioResult
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
    # "ParallelSimulator",
    # "ScenarioResult",
    # "ScenarioManager",
    # "Scenario",
    # "ParameterSweep",
    # "TimeSeries",
    # "TimeSeriesInterpolator",
    # "SimulationResults",
    # "ResultsAnalyzer",
    # "ResultsExporter",
]
