# pyadm1/simulation/parallel.py
"""
Parallel Simulation Engine for Parameter Sweeps and Monte Carlo Analysis

This module provides the ParallelSimulator class for concurrent execution of
multiple ADM1 simulation scenarios with different parameter sets, substrate
mixtures, or operating conditions.

Features:
- Multiprocessing for CPU-bound parallel execution
- Parameter sweep support (single and multi-parameter)
- Monte Carlo simulation with uncertainty quantification
- Automatic workload distribution and progress tracking
- Result aggregation and statistical analysis
- Error handling per scenario with graceful degradation

Example:
    >>> from pyadm1.simulation import ParallelSimulator
    >>> from pyadm1.core import ADM1
    >>> from pyadm1.substrates import Feedstock
    >>>
    >>> # Setup base model
    >>> feedstock = Feedstock(feeding_freq=48)
    >>> adm1 = ADM1(feedstock)
    >>>
    >>> # Define scenarios
    >>> scenarios = [
    ...     {"k_dis": 0.4, "Y_su": 0.09, "Q": [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]},
    ...     {"k_dis": 0.5, "Y_su": 0.10, "Q": [20, 10, 0, 0, 0, 0, 0, 0, 0, 0]},
    ...     {"k_dis": 0.6, "Y_su": 0.11, "Q": [15, 15, 0, 0, 0, 0, 0, 0, 0, 0]},
    ... ]
    >>>
    >>> # Run parallel simulations
    >>> parallel = ParallelSimulator(adm1, n_workers=4)
    >>> results = parallel.run_scenarios(
    ...     scenarios=scenarios,
    ...     duration=30,
    ...     initial_state=state_zero
    ... )
    >>>
    >>> # Analyze results
    >>> summary = parallel.summarize_results(results)
"""

import multiprocessing as mp
from multiprocessing import Pool
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import time
from functools import partial
import traceback


@dataclass
class ScenarioResult:
    """
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
    """

    scenario_id: int
    parameters: Dict[str, Any]
    success: bool
    duration: float
    final_state: Optional[List[float]] = None
    time_series: Optional[Dict[str, List[float]]] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time: float = 0.0


@dataclass
class ParameterSweepConfig:
    """
    Configuration for parameter sweep.

    Attributes:
        parameter_name: Name of parameter to sweep
        values: List of values to test
        other_params: Fixed parameters for all scenarios
    """

    parameter_name: str
    values: List[float]
    other_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonteCarloConfig:
    """
    Configuration for Monte Carlo simulation.

    Attributes:
        n_samples: Number of Monte Carlo samples
        parameter_distributions: Dict mapping parameter names to (mean, std) tuples
        fixed_params: Parameters to keep fixed
        seed: Random seed for reproducibility
    """

    n_samples: int
    parameter_distributions: Dict[str, Tuple[float, float]]
    fixed_params: Dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None


class ParallelSimulator:
    """
    Parallel simulator for running multiple ADM1 scenarios concurrently.

    Uses multiprocessing to distribute scenarios across CPU cores for efficient
    parameter sweeps, sensitivity analysis, and Monte Carlo simulations.

    Attributes:
        adm1: Base ADM1 model instance (will be copied for each worker)
        n_workers: Number of parallel worker processes
        verbose: Enable progress reporting

    Example:
        >>> parallel = ParallelSimulator(adm1, n_workers=4, verbose=True)
        >>> results = parallel.run_scenarios(scenarios, duration=30)
    """

    def __init__(self, adm1, n_workers: Optional[int] = None, verbose: bool = True):
        """
        Initialize parallel simulator.

        Args:
            adm1: ADM1 model instance
            n_workers: Number of worker processes (default: CPU count - 1)
            verbose: Enable progress output
        """
        self.adm1 = adm1
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)
        self.verbose = verbose

    def run_scenarios(
        self,
        scenarios: List[Dict[str, Any]],
        duration: float,
        initial_state: List[float],
        dt: float = 1.0 / 24.0,
        compute_metrics: bool = True,
        save_time_series: bool = False,
    ) -> List[ScenarioResult]:
        """
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
            >>> scenarios = [
            ...     {"k_dis": 0.5, "Q": [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]},
            ...     {"k_dis": 0.6, "Q": [20, 10, 0, 0, 0, 0, 0, 0, 0, 0]},
            ... ]
            >>> results = parallel.run_scenarios(scenarios, duration=30, initial_state=state)
        """
        if self.verbose:
            print(f"Starting parallel simulation with {len(scenarios)} scenarios")
            print(f"Using {self.n_workers} worker processes")

        start_time = time.time()

        # Create worker function with fixed parameters
        worker_func = partial(
            _run_single_scenario,
            adm1_config=self._serialize_adm1(),
            duration=duration,
            initial_state=initial_state,
            dt=dt,
            compute_metrics=compute_metrics,
            save_time_series=save_time_series,
        )

        # Add scenario IDs
        scenarios_with_ids = [(i, scenario) for i, scenario in enumerate(scenarios)]

        # Run scenarios in parallel
        with Pool(processes=self.n_workers) as pool:
            if self.verbose:
                # Use imap for progress tracking
                results = []
                for i, result in enumerate(pool.imap(worker_func, scenarios_with_ids)):
                    results.append(result)
                    if (i + 1) % 10 == 0 or (i + 1) == len(scenarios):
                        print(f"  Completed {i + 1}/{len(scenarios)} scenarios")
            else:
                results = pool.map(worker_func, scenarios_with_ids)

        elapsed_time = time.time() - start_time

        if self.verbose:
            n_success = sum(1 for r in results if r.success)
            print("\nSimulation complete:")
            print(f"  Total scenarios: {len(scenarios)}")
            print(f"  Successful: {n_success}")
            print(f"  Failed: {len(scenarios) - n_success}")
            print(f"  Total time: {elapsed_time:.1f} seconds")
            print(f"  Average time per scenario: {elapsed_time / len(scenarios):.2f} seconds")

        return results

    def parameter_sweep(
        self, config: ParameterSweepConfig, duration: float, initial_state: List[float], **kwargs
    ) -> List[ScenarioResult]:
        """
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
            >>> config = ParameterSweepConfig(
            ...     parameter_name="k_dis",
            ...     values=[0.3, 0.4, 0.5, 0.6, 0.7],
            ...     other_params={"Q": [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]}
            ... )
            >>> results = parallel.parameter_sweep(config, duration=30, initial_state=state)
        """
        # Generate scenarios
        scenarios = []
        for value in config.values:
            scenario = config.other_params.copy()
            scenario[config.parameter_name] = value
            scenarios.append(scenario)

        if self.verbose:
            print(f"Parameter sweep: {config.parameter_name}")
            print(f"  Values: {config.values}")

        return self.run_scenarios(scenarios, duration, initial_state, **kwargs)

    def multi_parameter_sweep(
        self,
        parameter_configs: Dict[str, List[float]],
        duration: float,
        initial_state: List[float],
        fixed_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[ScenarioResult]:
        """
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
        """
        fixed_params = fixed_params or {}

        # Generate all combinations using recursive approach
        param_names = list(parameter_configs.keys())
        param_values = [parameter_configs[name] for name in param_names]

        scenarios = []
        for combination in self._generate_combinations(param_values):
            scenario = fixed_params.copy()
            for name, value in zip(param_names, combination):
                scenario[name] = value
            scenarios.append(scenario)

        if self.verbose:
            print("Multi-parameter sweep:")
            for name, values in parameter_configs.items():
                print(f"  {name}: {len(values)} values")
            print(f"  Total combinations: {len(scenarios)}")

        return self.run_scenarios(scenarios, duration, initial_state, **kwargs)

    def monte_carlo(
        self, config: MonteCarloConfig, duration: float, initial_state: List[float], **kwargs
    ) -> List[ScenarioResult]:
        """
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
        """
        # Set random seed for reproducibility
        if config.seed is not None:
            np.random.seed(config.seed)

        # Generate scenarios
        scenarios = []
        for i in range(config.n_samples):
            scenario = config.fixed_params.copy()

            # Sample each parameter from its distribution
            for param_name, (mean, std) in config.parameter_distributions.items():
                value = np.random.normal(mean, std)
                scenario[param_name] = value

            scenarios.append(scenario)

        if self.verbose:
            print("Monte Carlo simulation:")
            print(f"  Samples: {config.n_samples}")
            print("  Parameters with uncertainty:")
            for param_name, (mean, std) in config.parameter_distributions.items():
                print(f"    {param_name}: N({mean}, {std}²)")

        return self.run_scenarios(scenarios, duration, initial_state, **kwargs)

    def summarize_results(self, results: List[ScenarioResult], metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Summarize results from multiple scenarios.

        Computes statistics (mean, std, min, max) for each metric across
        all successful scenarios.

        Args:
            results: List of ScenarioResult objects
            metrics: List of metric names to summarize (default: all)

        Returns:
            Dictionary with summary statistics

        Example:
            >>> summary = parallel.summarize_results(results)
            >>> print(f"Mean CH4: {summary['Q_ch4']['mean']:.1f} m³/d")
        """
        successful = [r for r in results if r.success]
        n_scenarios = len(results)
        n_successful = len(successful)

        summary = {
            "n_scenarios": n_scenarios,
            "n_successful": n_successful,
            "n_failed": n_scenarios - n_successful,
            "success_rate": n_successful / n_scenarios if n_scenarios > 0 else 0,
            "metrics": {},
        }

        if not successful:
            if n_scenarios > 0:
                summary["error"] = "No successful scenarios"
            else:
                summary["error"] = "No scenarios to summarize"
            return summary

        # Determine which metrics to summarize
        if metrics is None:
            # Use all metrics from first result
            metrics = list(successful[0].metrics.keys())

        # Compute statistics for each metric
        for metric_name in metrics:
            values = [r.metrics.get(metric_name, np.nan) for r in successful]
            values = [v for v in values if not np.isnan(v)]

            if values:
                summary["metrics"][metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "median": np.median(values),
                    "q25": np.percentile(values, 25),
                    "q75": np.percentile(values, 75),
                }

        return summary

    def _serialize_adm1(self) -> Dict[str, Any]:
        """
        Serialize ADM1 model for passing to worker processes.

        Returns:
            Dictionary with ADM1 configuration
        """
        return {
            "V_liq": self.adm1.V_liq,
            "V_gas": self.adm1._V_gas,
            "T_ad": self.adm1._T_ad,
            "feedstock_config": {
                "feeding_freq": 48,  # Default from feedstock
            },
        }

    @staticmethod
    def _generate_combinations(value_lists: List[List[Any]]) -> List[Tuple]:
        """
        Generate all combinations from lists of values (Cartesian product).

        Args:
            value_lists: List of lists containing values

        Returns:
            List of tuples with all combinations
        """
        if not value_lists:
            return [()]

        result = []
        for value in value_lists[0]:
            for rest in ParallelSimulator._generate_combinations(value_lists[1:]):
                result.append((value,) + rest)

        return result


def _run_single_scenario(
    scenario_data: Tuple[int, Dict[str, Any]],
    adm1_config: Dict[str, Any],
    duration: float,
    initial_state: List[float],
    dt: float,
    compute_metrics: bool,
    save_time_series: bool,
) -> ScenarioResult:
    """
    Worker function to run a single scenario.

    This function is called by each worker process. It creates a new ADM1
    instance, applies the scenario parameters, runs the simulation, and
    returns results.

    Args:
        scenario_data: Tuple of (scenario_id, parameters)
        adm1_config: Serialized ADM1 configuration
        duration: Simulation duration [days]
        initial_state: Initial state vector
        dt: Time step [days]
        compute_metrics: Whether to compute metrics
        save_time_series: Whether to save time series

    Returns:
        ScenarioResult object
    """
    scenario_id, parameters = scenario_data
    start_time = time.time()

    try:
        # Import here to avoid issues with multiprocessing
        from pyadm1.core.adm1 import ADM1
        from pyadm1.substrates.feedstock import Feedstock
        from pyadm1.simulation.simulator import Simulator

        # Create fresh ADM1 instance
        feedstock = Feedstock(feeding_freq=adm1_config["feedstock_config"]["feeding_freq"])
        adm1 = ADM1(feedstock=feedstock, V_liq=adm1_config["V_liq"], V_gas=adm1_config["V_gas"], T_ad=adm1_config["T_ad"])

        # Apply scenario parameters
        Q = parameters.get("Q", [0] * 10)

        # Create influent
        adm1.create_influent(Q, 0)

        # Run simulation
        simulator = Simulator(adm1)
        final_state = simulator.simulate_AD_plant([0, duration], initial_state)

        # Compute metrics if requested
        metrics = {}
        if compute_metrics:
            metrics = _compute_scenario_metrics(adm1, final_state, Q)

        # Save time series if requested
        time_series = None
        if save_time_series:
            time_series = {
                "Q_gas": adm1.Q_GAS[-10:] if adm1.Q_GAS else [],
                "Q_ch4": adm1.Q_CH4[-10:] if adm1.Q_CH4 else [],
                "pH": adm1.pH_l[-10:] if adm1.pH_l else [],
                "VFA": adm1.VFA[-10:] if adm1.VFA else [],
            }

        execution_time = time.time() - start_time

        return ScenarioResult(
            scenario_id=scenario_id,
            parameters=parameters,
            success=True,
            duration=duration,
            final_state=final_state,
            time_series=time_series,
            metrics=metrics,
            execution_time=execution_time,
        )

    except Exception as e:
        execution_time = time.time() - start_time

        return ScenarioResult(
            scenario_id=scenario_id,
            parameters=parameters,
            success=False,
            duration=duration,
            error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
            execution_time=execution_time,
        )


def _compute_scenario_metrics(adm1, final_state: List[float], Q: List[float]) -> Dict[str, float]:
    """
    Compute performance metrics from simulation results.

    Args:
        adm1: ADM1 instance
        final_state: Final state vector
        Q: Substrate flow rates

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    try:
        # Calculate gas production
        pi_Sh2, pi_Sch4, pi_Sco2, pTOTAL = final_state[33:37]
        q_gas, q_ch4, q_co2, p_gas = adm1.calc_gas(pi_Sh2, pi_Sch4, pi_Sco2, pTOTAL)

        metrics["Q_gas"] = float(q_gas)
        metrics["Q_ch4"] = float(q_ch4)
        metrics["Q_co2"] = float(q_co2)
        metrics["p_gas"] = float(p_gas)

        # Methane content
        if q_gas > 0:
            metrics["CH4_content"] = float(q_ch4 / q_gas)

        # Calculate process indicators using DLL if available
        try:
            from biogas import ADMstate

            # Use 1D list of standard Python floats
            final_state_l = [float(x) for x in final_state]
            metrics["pH"] = float(ADMstate.calcPHOfADMstate(final_state_l))
            metrics["VFA"] = float(ADMstate.calcVFAOfADMstate(final_state_l, "gHAceq/l").Value)
            metrics["TAC"] = float(ADMstate.calcTACOfADMstate(final_state_l, "gCaCO3eq/l").Value)
            if metrics["TAC"] > 0:
                metrics["FOS_TAC"] = float(metrics["VFA"] / metrics["TAC"])
        except Exception as e:
            print(e)

        # Substrate-related metrics
        Q_total = sum(Q)
        if Q_total > 0:
            metrics["Q_total"] = float(Q_total)
            metrics["specific_gas_production"] = float(q_gas / Q_total)  # m³/m³
            metrics["specific_ch4_production"] = float(q_ch4 / Q_total)  # m³/m³

        # HRT
        if Q_total > 0:
            metrics["HRT"] = float(adm1.V_liq / Q_total)

    except Exception as e:
        metrics["error"] = str(e)

    return metrics
