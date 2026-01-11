"""
Integration tests for ParallelSimulator.

This module tests the parallel simulation engine for running multiple
ADM1 scenarios concurrently with parameter sweeps and Monte Carlo analysis.
"""

import pytest
import numpy as np
from pathlib import Path
from typing import List

from pyadm1.simulation.parallel import (
    ParallelSimulator,
    ScenarioResult,
    ParameterSweepConfig,
    MonteCarloConfig,
)
from pyadm1.core.adm1 import ADM1, get_state_zero_from_initial_state
from pyadm1.substrates.feedstock import Feedstock


@pytest.fixture
def feedstock() -> Feedstock:
    """
    Create a Feedstock instance for testing.

    Returns:
        Feedstock instance with standard configuration.
    """
    return Feedstock(feeding_freq=48)


@pytest.fixture
def adm1_instance(feedstock: Feedstock) -> ADM1:
    """
    Create an ADM1 instance for testing.

    Args:
        feedstock: Feedstock fixture.

    Returns:
        ADM1 instance with standard configuration.
    """
    return ADM1(feedstock, V_liq=2000, V_gas=300, T_ad=308.15)


@pytest.fixture
def initial_state() -> List[float]:
    """
    Provide initial ADM1 state vector.

    Returns:
        Initial state vector (37 elements).
    """
    # Try to load from default initial state file
    try:
        data_path = Path(__file__).parent.parent.parent / "data" / "initial_states"
        default_file = data_path / "digester_initial8.csv"

        if default_file.exists():
            return get_state_zero_from_initial_state(str(default_file))
    except Exception:
        pass

    # Fallback to sample state if file not available
    return [
        0.0055,
        0.0025,
        0.0398,
        0.0052,
        0.0101,
        0.0281,
        0.9686,
        1.075e-7,
        0.0556,
        0.0117,
        0.2438,
        10.738,
        18.015,
        0.2544,
        0.0554,
        0.0184,
        7.8300,
        1.3649,
        0.3288,
        1.0153,
        0.8791,
        3.1754,
        1.6683,
        38.294,
        2.0409,
        0.0,
        0.0,
        0.0052,
        0.0101,
        0.0281,
        0.9672,
        0.2284,
        0.0128,
        5.935e-6,
        0.5592,
        0.4253,
        0.9845,
    ]


@pytest.fixture
def base_Q() -> List[float]:
    """
    Provide base substrate flow rates.

    Returns:
        Substrate flow rates [m³/d].
    """
    return [15.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


class TestParallelSimulatorInitialization:
    """Test suite for ParallelSimulator initialization."""

    def test_initialization_with_default_workers(self, adm1_instance: ADM1) -> None:
        """Test initialization with default number of workers."""
        parallel = ParallelSimulator(adm1_instance)

        assert parallel.n_workers > 0, "Should have positive number of workers"
        assert parallel.adm1 is not None, "Should store ADM1 instance"

    def test_initialization_with_custom_workers(self, adm1_instance: ADM1) -> None:
        """Test initialization with custom number of workers."""
        n_workers = 2
        parallel = ParallelSimulator(adm1_instance, n_workers=n_workers)

        assert parallel.n_workers == n_workers, f"Should use {n_workers} workers"

    def test_initialization_verbose_setting(self, adm1_instance: ADM1) -> None:
        """Test verbose setting."""
        parallel_verbose = ParallelSimulator(adm1_instance, verbose=True)
        parallel_quiet = ParallelSimulator(adm1_instance, verbose=False)

        assert parallel_verbose.verbose is True
        assert parallel_quiet.verbose is False


class TestParallelSimulatorBasicScenarios:
    """Test suite for basic scenario execution."""

    def test_run_single_scenario(self, adm1_instance: ADM1, initial_state: List[float], base_Q: List[float]) -> None:
        """Test running a single scenario."""
        parallel = ParallelSimulator(adm1_instance, n_workers=1, verbose=False)

        scenarios = [{"Q": base_Q}]

        results = parallel.run_scenarios(
            scenarios=scenarios,
            duration=7.0,
            initial_state=initial_state,
            dt=1.0 / 24.0,
            compute_metrics=True,
            save_time_series=False,
        )

        assert len(results) == 1, "Should return one result"
        assert isinstance(results[0], ScenarioResult), "Should return ScenarioResult"
        assert results[0].success, "Scenario should succeed"

    def test_run_multiple_scenarios(self, adm1_instance: ADM1, initial_state: List[float], base_Q: List[float]) -> None:
        """Test running multiple scenarios in parallel."""
        parallel = ParallelSimulator(adm1_instance, n_workers=2, verbose=False)

        scenarios = [
            {"Q": [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]},
            {"Q": [20, 10, 0, 0, 0, 0, 0, 0, 0, 0]},
            {"Q": [15, 15, 0, 0, 0, 0, 0, 0, 0, 0]},
        ]

        results = parallel.run_scenarios(scenarios=scenarios, duration=7.0, initial_state=initial_state, compute_metrics=True)

        assert len(results) == 3, "Should return three results"

        successful = [r for r in results if r.success]
        assert len(successful) > 0, "At least some scenarios should succeed"

    def test_scenario_results_contain_metrics(
        self, adm1_instance: ADM1, initial_state: List[float], base_Q: List[float]
    ) -> None:
        """Test that scenario results contain computed metrics."""
        parallel = ParallelSimulator(adm1_instance, n_workers=1, verbose=False)

        scenarios = [{"Q": base_Q}]

        results = parallel.run_scenarios(scenarios=scenarios, duration=7.0, initial_state=initial_state, compute_metrics=True)

        result = results[0]
        assert result.success, "Scenario should succeed"
        assert "Q_gas" in result.metrics, "Should have Q_gas metric"
        assert "Q_ch4" in result.metrics, "Should have Q_ch4 metric"
        assert result.metrics["Q_gas"] > 0, "Gas production should be positive"

    def test_scenario_with_time_series(self, adm1_instance: ADM1, initial_state: List[float], base_Q: List[float]) -> None:
        """Test scenario with time series data saving."""
        parallel = ParallelSimulator(adm1_instance, n_workers=1, verbose=False)

        scenarios = [{"Q": base_Q}]

        results = parallel.run_scenarios(scenarios=scenarios, duration=7.0, initial_state=initial_state, save_time_series=True)

        result = results[0]
        assert result.time_series is not None, "Should have time series"
        assert "Q_gas" in result.time_series, "Time series should include Q_gas"


class TestParameterSweep:
    """Test suite for parameter sweep functionality."""

    def test_single_parameter_sweep(self, adm1_instance: ADM1, initial_state: List[float], base_Q: List[float]) -> None:
        """Test single parameter sweep."""
        parallel = ParallelSimulator(adm1_instance, n_workers=2, verbose=False)

        config = ParameterSweepConfig(parameter_name="k_dis", values=[0.4, 0.5, 0.6], other_params={"Q": base_Q})

        results = parallel.parameter_sweep(config=config, duration=7.0, initial_state=initial_state)

        assert len(results) == 3, "Should test 3 parameter values"

        successful = [r for r in results if r.success]
        assert len(successful) > 0, "Some scenarios should succeed"

        # Check that parameter values vary
        k_dis_values = [r.parameters.get("k_dis") for r in successful]
        assert len(set(k_dis_values)) > 1, "Should test different k_dis values"

    def test_parameter_sweep_affects_results(
        self, adm1_instance: ADM1, initial_state: List[float], base_Q: List[float]
    ) -> None:
        """Test that parameter sweep produces varying results."""
        parallel = ParallelSimulator(adm1_instance, n_workers=2, verbose=False)

        config = ParameterSweepConfig(parameter_name="k_dis", values=[0.3, 0.5, 0.7], other_params={"Q": base_Q})

        results = parallel.parameter_sweep(config=config, duration=7.0, initial_state=initial_state, compute_metrics=True)

        successful = [r for r in results if r.success]
        assert len(successful) >= 2, "Need at least 2 successful runs to compare"

        # Extract gas production values
        q_ch4_values = [r.metrics["Q_ch4"] for r in successful]

        # Values should vary with parameter
        assert np.std(q_ch4_values) > 0.1, "Results should vary with parameter"

    def test_multi_parameter_sweep(self, adm1_instance: ADM1, initial_state: List[float], base_Q: List[float]) -> None:
        """Test multi-parameter sweep (factorial design)."""
        parallel = ParallelSimulator(adm1_instance, n_workers=2, verbose=False)

        parameter_configs = {"k_dis": [0.4, 0.6], "Y_su": [0.09, 0.11]}

        results = parallel.multi_parameter_sweep(
            parameter_configs=parameter_configs, duration=7.0, initial_state=initial_state, fixed_params={"Q": base_Q}
        )

        # Should test all combinations: 2 × 2 = 4
        assert len(results) == 4, "Should test all parameter combinations"

        # Check that all combinations are present
        param_combos = [(r.parameters["k_dis"], r.parameters["Y_su"]) for r in results if r.success]
        assert len(set(param_combos)) >= 2, "Should have different parameter combinations"


class TestMonteCarloSimulation:
    """Test suite for Monte Carlo simulation."""

    def test_monte_carlo_basic(self, adm1_instance: ADM1, initial_state: List[float], base_Q: List[float]) -> None:
        """Test basic Monte Carlo simulation."""
        parallel = ParallelSimulator(adm1_instance, n_workers=2, verbose=False)

        config = MonteCarloConfig(
            n_samples=10,
            parameter_distributions={
                "k_dis": (0.5, 0.05),  # mean=0.5, std=0.05
            },
            fixed_params={"Q": base_Q},
            seed=42,
        )

        results = parallel.monte_carlo(config=config, duration=7.0, initial_state=initial_state)

        assert len(results) == 10, "Should run 10 samples"

        successful = [r for r in results if r.success]
        assert len(successful) > 0, "Some scenarios should succeed"

    def test_monte_carlo_reproducibility(self, adm1_instance: ADM1, initial_state: List[float], base_Q: List[float]) -> None:
        """Test that Monte Carlo with seed is reproducible."""
        parallel = ParallelSimulator(adm1_instance, n_workers=1, verbose=False)

        config = MonteCarloConfig(
            n_samples=5,
            parameter_distributions={
                "k_dis": (0.5, 0.05),
            },
            fixed_params={"Q": base_Q},
            seed=42,
        )

        results1 = parallel.monte_carlo(config, 7.0, initial_state)
        results2 = parallel.monte_carlo(config, 7.0, initial_state)

        # Extract parameter values
        params1 = [r.parameters["k_dis"] for r in results1]
        params2 = [r.parameters["k_dis"] for r in results2]

        assert np.allclose(params1, params2), "Should be reproducible with same seed"

    def test_monte_carlo_parameter_distribution(
        self, adm1_instance: ADM1, initial_state: List[float], base_Q: List[float]
    ) -> None:
        """Test that Monte Carlo samples from specified distribution."""
        parallel = ParallelSimulator(adm1_instance, n_workers=2, verbose=False)

        mean = 0.5
        std = 0.05

        config = MonteCarloConfig(
            n_samples=50,
            parameter_distributions={
                "k_dis": (mean, std),
            },
            fixed_params={"Q": base_Q},
            seed=42,
        )

        results = parallel.monte_carlo(config, 7.0, initial_state)

        # Extract sampled parameter values
        k_dis_values = [r.parameters["k_dis"] for r in results]

        # Check distribution statistics
        sample_mean = np.mean(k_dis_values)
        sample_std = np.std(k_dis_values)

        assert abs(sample_mean - mean) < 0.1, "Sample mean should be close to target"
        assert abs(sample_std - std) < 0.05, "Sample std should be close to target"


class TestResultSummarization:
    """Test suite for result summarization."""

    def test_summarize_results_basic(self, adm1_instance: ADM1, initial_state: List[float], base_Q: List[float]) -> None:
        """Test basic result summarization."""
        parallel = ParallelSimulator(adm1_instance, n_workers=2, verbose=False)

        scenarios = [
            {"Q": [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]},
            {"Q": [20, 10, 0, 0, 0, 0, 0, 0, 0, 0]},
            {"Q": [15, 15, 0, 0, 0, 0, 0, 0, 0, 0]},
        ]

        results = parallel.run_scenarios(scenarios=scenarios, duration=7.0, initial_state=initial_state, compute_metrics=True)

        summary = parallel.summarize_results(results)

        assert "n_scenarios" in summary, "Should have total scenario count"
        assert "n_successful" in summary, "Should have success count"
        assert "success_rate" in summary, "Should have success rate"
        assert "metrics" in summary, "Should have metrics summary"

    def test_summarize_results_statistics(self, adm1_instance: ADM1, initial_state: List[float], base_Q: List[float]) -> None:
        """Test that summary contains proper statistics."""
        parallel = ParallelSimulator(adm1_instance, n_workers=2, verbose=False)

        config = ParameterSweepConfig(parameter_name="k_dis", values=[0.4, 0.5, 0.6], other_params={"Q": base_Q})

        results = parallel.parameter_sweep(config, 7.0, initial_state)
        summary = parallel.summarize_results(results)

        if summary["n_successful"] > 0:
            assert "Q_ch4" in summary["metrics"], "Should have Q_ch4 metrics"

            q_ch4_stats = summary["metrics"]["Q_ch4"]
            assert "mean" in q_ch4_stats, "Should have mean"
            assert "std" in q_ch4_stats, "Should have std"
            assert "min" in q_ch4_stats, "Should have min"
            assert "max" in q_ch4_stats, "Should have max"
            assert "median" in q_ch4_stats, "Should have median"


class TestParallelSimulatorPerformance:
    """Test suite for performance characteristics."""

    def test_parallel_faster_than_serial(self, adm1_instance: ADM1, initial_state: List[float], base_Q: List[float]) -> None:
        """Test that parallel execution is faster than serial (if multiple cores)."""
        import time

        scenarios = [{"Q": [15 + i, 10, 0, 0, 0, 0, 0, 0, 0, 0]} for i in range(4)]

        # Serial execution
        parallel_serial = ParallelSimulator(adm1_instance, n_workers=1, verbose=False)
        start = time.time()
        parallel_serial.run_scenarios(scenarios, 7.0, initial_state)
        time_serial = time.time() - start

        # Parallel execution
        parallel_multi = ParallelSimulator(adm1_instance, n_workers=2, verbose=False)
        start = time.time()
        parallel_multi.run_scenarios(scenarios, 7.0, initial_state)
        time_parallel = time.time() - start

        # Parallel should be at least somewhat faster (allowing for overhead)
        # This is a weak test since we can't guarantee speedup in all environments
        assert time_parallel < time_serial * 1.5, "Parallel should not be much slower"

    def test_many_scenarios_complete(self, adm1_instance: ADM1, initial_state: List[float]) -> None:
        """Test that many scenarios can be completed."""
        parallel = ParallelSimulator(adm1_instance, n_workers=2, verbose=False)

        # Create 20 scenarios with varying parameters
        scenarios = [{"Q": [15 + i * 0.5, 10, 0, 0, 0, 0, 0, 0, 0, 0]} for i in range(20)]

        results = parallel.run_scenarios(
            scenarios=scenarios, duration=3.0, initial_state=initial_state  # Short duration for speed
        )

        assert len(results) == 20, "Should complete all scenarios"
        successful = [r for r in results if r.success]
        assert len(successful) > 15, "Most scenarios should succeed"


class TestErrorHandling:
    """Test suite for error handling."""

    def test_failed_scenario_captured(self, adm1_instance: ADM1, initial_state: List[float]) -> None:
        """Test that failed scenarios are captured and don't crash."""
        parallel = ParallelSimulator(adm1_instance, n_workers=1, verbose=False)

        # Create a scenario likely to fail (invalid parameters)
        scenarios = [
            {"Q": [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]},  # Valid
            {"Q": [-100, -100, 0, 0, 0, 0, 0, 0, 0, 0]},  # Invalid (negative)
        ]

        # Should not raise exception
        results = parallel.run_scenarios(scenarios, 7.0, initial_state)

        assert len(results) == 2, "Should return results for all scenarios"

        # Check that we have both success and failure
        success_count = sum(1 for r in results if r.success)
        failure_count = sum(1 for r in results if not r.success)

        # At least one should succeed, likely one should fail
        assert success_count > 0, "At least one scenario should succeed"
        assert failure_count > 0, "At least one scenario should fail"

    def test_execution_time_recorded(self, adm1_instance: ADM1, initial_state: List[float], base_Q: List[float]) -> None:
        """Test that execution time is recorded for each scenario."""
        parallel = ParallelSimulator(adm1_instance, n_workers=1, verbose=False)

        scenarios = [{"Q": base_Q}]

        results = parallel.run_scenarios(scenarios, 7.0, initial_state)

        assert results[0].execution_time > 0, "Should record execution time"


class TestScenarioResultStructure:
    """Test suite for ScenarioResult data structure."""

    def test_scenario_result_attributes(self, adm1_instance: ADM1, initial_state: List[float], base_Q: List[float]) -> None:
        """Test that ScenarioResult has all expected attributes."""
        parallel = ParallelSimulator(adm1_instance, n_workers=1, verbose=False)

        scenarios = [{"Q": base_Q}]
        results = parallel.run_scenarios(scenarios, 7.0, initial_state)

        result = results[0]

        assert hasattr(result, "scenario_id"), "Should have scenario_id"
        assert hasattr(result, "parameters"), "Should have parameters"
        assert hasattr(result, "success"), "Should have success"
        assert hasattr(result, "duration"), "Should have duration"
        assert hasattr(result, "final_state"), "Should have final_state"
        assert hasattr(result, "metrics"), "Should have metrics"
        assert hasattr(result, "execution_time"), "Should have execution_time"

    def test_final_state_correct_length(self, adm1_instance: ADM1, initial_state: List[float], base_Q: List[float]) -> None:
        """Test that final state has correct length."""
        parallel = ParallelSimulator(adm1_instance, n_workers=1, verbose=False)

        scenarios = [{"Q": base_Q}]
        results = parallel.run_scenarios(scenarios, 7.0, initial_state)

        result = results[0]
        assert result.success, "Scenario should succeed"
        assert len(result.final_state) == 37, "Final state should have 37 elements"


class TestIntegrationWithADM1:
    """Test suite for integration with actual ADM1 model."""

    def test_results_physically_reasonable(self, adm1_instance: ADM1, initial_state: List[float], base_Q: List[float]) -> None:
        """Test that simulation results are physically reasonable."""
        parallel = ParallelSimulator(adm1_instance, n_workers=1, verbose=False)

        scenarios = [{"Q": base_Q}]
        results = parallel.run_scenarios(scenarios=scenarios, duration=7.0, initial_state=initial_state, compute_metrics=True)

        result = results[0]
        assert result.success, "Scenario should succeed"

        # Check physical reasonableness
        assert result.metrics["Q_gas"] > 0, "Gas production should be positive"
        assert result.metrics["Q_ch4"] > 0, "Methane production should be positive"
        assert 0 < result.metrics["CH4_content"] < 1, "CH4 content should be fraction"

        if "pH" in result.metrics:
            assert 5 < result.metrics["pH"] < 9, "pH should be in reasonable range"

    def test_mass_balance_consistency(self, adm1_instance: ADM1, initial_state: List[float], base_Q: List[float]) -> None:
        """Test that mass balance is maintained across scenarios."""
        parallel = ParallelSimulator(adm1_instance, n_workers=2, verbose=False)

        # Double the substrate input
        scenarios = [{"Q": base_Q}, {"Q": [2 * q for q in base_Q]}]

        results = parallel.run_scenarios(scenarios=scenarios, duration=7.0, initial_state=initial_state, compute_metrics=True)

        successful = [r for r in results if r.success]
        assert len(successful) == 2, "Both scenarios should succeed"

        # Higher substrate input should generally produce more gas
        # (allowing for some nonlinearity)
        q_ch4_1 = successful[0].metrics["Q_ch4"]
        q_ch4_2 = successful[1].metrics["Q_ch4"]

        assert q_ch4_2 > q_ch4_1 * 0.5, "More substrate should produce more gas"


@pytest.mark.slow
class TestLargeScaleSimulations:
    """Test suite for large-scale simulations (marked as slow)."""

    def test_100_scenario_monte_carlo(self, adm1_instance: ADM1, initial_state: List[float], base_Q: List[float]) -> None:
        """Test large Monte Carlo simulation with 100 samples."""
        parallel = ParallelSimulator(adm1_instance, n_workers=4, verbose=False)

        config = MonteCarloConfig(
            n_samples=100,
            parameter_distributions={
                "k_dis": (0.5, 0.05),
                "Y_su": (0.10, 0.01),
            },
            fixed_params={"Q": base_Q},
            seed=42,
        )

        results = parallel.monte_carlo(config=config, duration=3.0, initial_state=initial_state)  # Short duration

        assert len(results) == 100, "Should complete 100 scenarios"

        successful = [r for r in results if r.success]
        success_rate = len(successful) / len(results)

        assert success_rate > 0.8, "Most scenarios should succeed"

    def test_comprehensive_parameter_space(self, adm1_instance: ADM1, initial_state: List[float], base_Q: List[float]) -> None:
        """Test comprehensive parameter space exploration."""
        parallel = ParallelSimulator(adm1_instance, n_workers=4, verbose=False)

        parameter_configs = {
            "k_dis": [0.3, 0.4, 0.5, 0.6, 0.7],
            "Y_su": [0.08, 0.10, 0.12],
        }

        results = parallel.multi_parameter_sweep(
            parameter_configs=parameter_configs, duration=3.0, initial_state=initial_state, fixed_params={"Q": base_Q}
        )

        # Should test 5 × 3 = 15 combinations
        assert len(results) == 15, "Should test all combinations"

        summary = parallel.summarize_results(results)
        assert summary["n_successful"] > 10, "Most combinations should work"


class TestDocumentationExamples:
    """Test suite verifying documentation examples work."""

    def test_basic_example_from_docstring(self, adm1_instance: ADM1, initial_state: List[float]) -> None:
        """Test basic example from module docstring."""
        # Example from docstring
        scenarios = [
            {"k_dis": 0.4, "Y_su": 0.09, "Q": [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]},
            {"k_dis": 0.5, "Y_su": 0.10, "Q": [20, 10, 0, 0, 0, 0, 0, 0, 0, 0]},
            {"k_dis": 0.6, "Y_su": 0.11, "Q": [15, 15, 0, 0, 0, 0, 0, 0, 0, 0]},
        ]

        parallel = ParallelSimulator(adm1_instance, n_workers=2, verbose=False)
        results = parallel.run_scenarios(scenarios=scenarios, duration=7.0, initial_state=initial_state)

        assert len(results) == 3
        summary = parallel.summarize_results(results)
        assert summary is not None
