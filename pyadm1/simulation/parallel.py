# pyadm1/simulation/parallel.py
"""
Parallel simulation engine for parameter sweeps and Monte Carlo analysis.

Run multiple :class:`pyadm1.core.adm1.ADM1` simulation scenarios concurrently,
each with its own parameter set or substrate feed.  Designed for parameter
sweeps, sensitivity analysis, and Monte Carlo uncertainty quantification.

Each scenario is a dictionary that may contain:
  - ``Q`` (list of float): substrate feed rates [m³/d].
  - ``T_ad`` (float, optional): reactor temperature [K]. Overrides the
    base ADM1 instance's temperature for this scenario; useful for
    temperature sweeps that previously required one ParallelSimulator
    per temperature.
  - any subset of the calibration parameter keys (see
    :data:`_CALIBRATION_PARAM_KEYS`).
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from functools import partial
from itertools import pairwise
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pyadm1.core.adm1 import ADM1

#: Scenario keys that are not calibration parameters.
_SCENARIO_KEYS = frozenset(("Q", "T_ad"))

_CALIBRATION_PARAM_KEYS = frozenset(
    (
        "k_dis_PS",
        "k_dis_PF",
        "k_hyd_ch",
        "k_hyd_pr",
        "k_hyd_li",
        "k_m_su",
        "k_m_aa",
        "k_m_fa",
        "k_m_c4",
        "k_m_pro",
        "k_m_ac",
        "k_m_h2",
        "k_p",
        "k_L_a",
        "K_H_co2",
        "K_H_ch4",
        "K_H_h2",
    )
)


@dataclass
class ScenarioResult:
    """Result from a single simulation scenario."""

    scenario_id: int
    parameters: dict[str, Any]
    success: bool
    duration: float
    final_state: list[float] | None = None
    time_series: dict[str, list[float]] | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    error: str | None = None
    execution_time: float = 0.0


@dataclass
class ParameterSweepConfig:
    """Configuration for a single-parameter sweep."""

    parameter_name: str
    values: list[float]
    other_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation."""

    n_samples: int
    parameter_distributions: dict[str, tuple[float, float]]
    fixed_params: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None


def _get_mp_context() -> mp.context.BaseContext:
    """
    Choose a safe multiprocessing start method.

    Override via ``PYADM1_MP_START_METHOD`` (e.g. ``spawn``).
    """
    method = os.getenv("PYADM1_MP_START_METHOD")
    if method:
        return mp.get_context(method)

    if sys.platform.startswith("linux"):
        return mp.get_context("forkserver")

    return mp.get_context("spawn")


class ParallelSimulator:
    """
    Parallel simulator for running multiple ADM1 scenarios concurrently.
    """

    def __init__(self, adm1: ADM1, n_workers: int | None = None, verbose: bool = True):
        """
        Parameters
        ----------
        adm1 : ADM1
            Base model instance — its V_liq, V_gas, T_ad and feedstock
            substrate IDs are copied to each worker.
        n_workers : int, optional
            Worker process count (default = ``cpu_count() - 1``).
        verbose : bool
            Whether to print progress.
        """
        self.adm1 = adm1
        self.n_workers = n_workers or max(1, mp.cpu_count() - 1)
        self.verbose = verbose

    def run_scenarios(
        self,
        scenarios: list[dict[str, Any]],
        duration: float,
        initial_state: list[float],
        dt: float = 1.0 / 24.0,
        compute_metrics: bool = True,
        save_time_series: bool = False,
    ) -> list[ScenarioResult]:
        """
        Run multiple scenarios in parallel.

        Args:
            scenarios (List[Dict[str, Any]]): One parameter dict per scenario.
                Allowed keys are ``Q`` (feed rates), ``T_ad`` [K] and the
                calibration parameters in ``_CALIBRATION_PARAM_KEYS``; they
                are applied on top of the base model's own calibration.
            duration (float): Simulated time per scenario [days].
            initial_state (List[float]): Start state (41 elements).
            dt (float): Resolution of the saved time series [days]. Only used
                with ``save_time_series=True``; the final state does not
                depend on it.
            compute_metrics (bool): Compute gas, pH and loading metrics.
            save_time_series (bool): Keep the last ten points of gas and pH.

        Returns:
            List[ScenarioResult]: One result per scenario, in input order.

        Raises:
            ValueError: A scenario carries a key that is not a known
                parameter. A misspelt name would otherwise be ignored and
                every scenario of a sweep would return the same result.
        """
        allowed = _SCENARIO_KEYS | _CALIBRATION_PARAM_KEYS
        for index, scenario in enumerate(scenarios):
            unknown = sorted(set(scenario) - allowed)
            if unknown:
                raise ValueError(f"Scenario {index}: unknown parameter(s) {unknown}; allowed are {sorted(allowed)}.")

        if self.verbose:
            print(f"Starting parallel simulation with {len(scenarios)} scenarios")
            print(f"Using {self.n_workers} worker processes")

        start_time = time.time()

        worker_func = partial(
            _run_single_scenario,
            adm1_config=self._serialize_adm1(),
            duration=duration,
            initial_state=initial_state,
            dt=dt,
            compute_metrics=compute_metrics,
            save_time_series=save_time_series,
            verbose=self.verbose,
        )

        scenarios_with_ids = [(i, scenario) for i, scenario in enumerate(scenarios)]

        use_sequential = self.n_workers <= 1 or len(scenarios_with_ids) <= 1

        if use_sequential:
            results = []
            for i, scenario_data in enumerate(scenarios_with_ids):
                results.append(worker_func(scenario_data))
                if self.verbose and ((i + 1) % 10 == 0 or (i + 1) == len(scenarios)):
                    print(f"  Completed {i + 1}/{len(scenarios)} scenarios")
        else:
            ctx = _get_mp_context()
            with ctx.Pool(processes=self.n_workers) as pool:
                if self.verbose:
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
        self,
        config: ParameterSweepConfig,
        duration: float,
        initial_state: list[float],
        **kwargs: Any,
    ) -> list[ScenarioResult]:
        """Run a single-parameter sweep."""
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
        parameter_configs: dict[str, list[float]],
        duration: float,
        initial_state: list[float],
        fixed_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[ScenarioResult]:
        """Run a multi-parameter sweep (full factorial design)."""
        fixed_params = fixed_params or {}

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
        self,
        config: MonteCarloConfig,
        duration: float,
        initial_state: list[float],
        **kwargs: Any,
    ) -> list[ScenarioResult]:
        """Run Monte Carlo simulation with parameter uncertainty."""
        if config.seed is not None:
            np.random.seed(config.seed)

        scenarios = []
        for _ in range(config.n_samples):
            scenario = config.fixed_params.copy()
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

    def summarize_results(self, results: list[ScenarioResult], metrics: list[str] | None = None) -> dict[str, Any]:
        """Compute summary statistics across multiple scenarios."""
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

        if metrics is None:
            metrics = list(successful[0].metrics.keys())

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

    def _serialize_adm1(self) -> dict[str, Any]:
        """Serialize ADM1 model configuration for the worker pool."""
        feedstock = self.adm1.feedstock
        substrate_ids: list[Any] = []
        feeding_freq = 24
        if feedstock is not None:
            # The parameter objects travel to the workers, not the IDs: an ID
            # is only a file stem, so a substrate loaded from a user path
            # would be replaced by the library file of the same name.
            substrate_ids = list(getattr(feedstock, "substrates", None) or getattr(feedstock, "substrate_ids", []))
            feeding_freq = int(getattr(feedstock, "feeding_freq", 24))

        return {
            "V_liq": self.adm1.V_liq,
            "V_gas": self.adm1._V_gas,
            "T_ad": self.adm1._T_ad,
            "feedstock_substrates": substrate_ids,
            "feeding_freq": feeding_freq,
            "simba_q_convention": bool(getattr(feedstock, "_simba_q_convention", True)),
            "calibration": dict(self.adm1.get_calibration_parameters()),
        }

    @staticmethod
    def _generate_combinations(value_lists: list[list[Any]]) -> list[tuple]:
        """Cartesian product of a list of value lists."""
        if not value_lists:
            return [()]

        result = []
        for value in value_lists[0]:
            for rest in ParallelSimulator._generate_combinations(value_lists[1:]):
                result.append((value, *rest))

        return result


def _run_single_scenario(
    scenario_data: tuple[int, dict[str, Any]],
    adm1_config: dict[str, Any],
    duration: float,
    initial_state: list[float],
    dt: float,
    compute_metrics: bool,
    save_time_series: bool,
    verbose: bool,
) -> ScenarioResult:
    """Worker function: run a single scenario in its own process."""
    scenario_id, parameters = scenario_data
    start_time = time.time()

    try:
        from pyadm1.core.adm1 import ADM1, STATE_SIZE
        from pyadm1.simulation.simulator import Simulator
        from pyadm1.substrates.feedstock import Feedstock

        substrate_ids = adm1_config.get("feedstock_substrates") or []
        feeding_freq = adm1_config.get("feeding_freq", 24)
        if substrate_ids:
            feedstock = Feedstock(
                substrate_ids,
                feeding_freq=feeding_freq,
                total_simtime=int(duration) + 1,
                simba_q_convention=adm1_config.get("simba_q_convention", True),
            )
        else:
            feedstock = None

        # Per-scenario T_ad override: the base ADM1 fixes T at construction,
        # but a sweep over temperature would otherwise need one
        # ParallelSimulator per T. Allowing the scenario to override T_ad
        # lets temperature live alongside Q and the kinetic constants as a
        # first-class parameter axis.
        T_ad = float(parameters.get("T_ad", adm1_config["T_ad"]))
        adm1 = ADM1(
            feedstock=feedstock,
            V_liq=adm1_config["V_liq"],
            V_gas=adm1_config["V_gas"],
            T_ad=T_ad,
        )

        Q = parameters.get("Q", [0.0] * 10)
        # The base model's calibration applies to every scenario; a scenario
        # value for the same parameter wins.
        calibration_params = dict(adm1_config.get("calibration") or {})
        calibration_params.update({key: value for key, value in parameters.items() if key in _CALIBRATION_PARAM_KEYS})
        if calibration_params:
            adm1.set_calibration_parameters(calibration_params)

        # Without substrate ids there is no influent composition to build, and
        # ``create_influent`` would dereference the absent feedstock. ADM_ODE
        # treats a missing influent as an autonomous run (q_ad = 0, s_in = 0).
        if feedstock is not None:
            adm1.create_influent(Q, 0)

        if len(initial_state) != STATE_SIZE:
            raise ValueError(f"initial_state must have {STATE_SIZE} elements; got {len(initial_state)}")

        simulator = Simulator(adm1)
        if save_time_series and 0.0 < dt < duration:
            # One solver call per ``dt`` so the tracked gas and pH series get
            # one point per step.
            n_full = int(duration / dt + 1.0e-9)
            edges = [i * dt for i in range(n_full + 1)]
            if duration - edges[-1] > 1.0e-9:
                edges.append(duration)
            final_state = list(initial_state)
            for t_start, t_end in pairwise(edges):
                final_state = simulator.simulate_AD_plant([t_start, t_end], final_state)
        else:
            final_state = simulator.simulate_AD_plant([0.0, duration], list(initial_state))

        metrics = {}
        if compute_metrics:
            metrics = _compute_scenario_metrics(adm1, final_state, Q)

        time_series = None
        if save_time_series:
            time_series = {
                "Q_gas": adm1.Q_GAS[-10:] if adm1.Q_GAS else [],
                "Q_ch4": adm1.Q_CH4[-10:] if adm1.Q_CH4 else [],
                "pH": adm1.pH_l[-10:] if adm1.pH_l else [],
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

    except Exception as e:  # noqa: BLE001 - worker boundary: report any scenario failure instead of crashing the batch
        execution_time = time.time() - start_time

        return ScenarioResult(
            scenario_id=scenario_id,
            parameters=parameters,
            success=False,
            duration=duration,
            error=f"{type(e).__name__}: {e!s}\n{traceback.format_exc()}",
            execution_time=execution_time,
        )


def _compute_scenario_metrics(adm1: ADM1, final_state: list[float], Q: list[float]) -> dict[str, float]:
    """Compute performance metrics from simulation results."""
    metrics = {}

    try:
        pi_Sh2, pi_Sch4, pi_Sco2, pTOTAL = final_state[37:41]
        q_gas, q_ch4, q_co2, _, p_gas = adm1.calc_gas(pi_Sh2, pi_Sch4, pi_Sco2, pTOTAL)

        metrics["Q_gas"] = float(q_gas)
        metrics["Q_ch4"] = float(q_ch4)
        metrics["Q_co2"] = float(q_co2)
        metrics["p_gas"] = float(p_gas)

        if q_gas > 0:
            metrics["CH4_content"] = float(q_ch4 / q_gas)

        # pH from the same charge-balance solver used by ADM1.
        ip = adm1._inhib_params
        S_H = adm1._calc_ph(
            float(final_state[10]),  # S_nh4
            float(final_state[36]),  # S_nh3
            float(final_state[35]),  # S_hco3
            float(final_state[34]),  # S_ac_ion
            float(final_state[33]),  # S_pro_ion
            float(final_state[32]),  # S_bu_ion
            float(final_state[31]),  # S_va_ion
            float(final_state[29]),  # S_cation
            float(final_state[30]),  # S_anion
            ip["K_w"],
        )
        ph = float(-np.log10(max(S_H, 1.0e-14)))
        if 0.0 < ph < 14.0:
            metrics["pH"] = ph

        # ``Q_total`` and the specific yields refer to the feed rates as entered
        # (tonnes of fresh matter per day under the default Feedstock
        # convention); the retention time needs the volume flow the model uses.
        Q_total = float(np.sum(Q))
        if Q_total > 0:
            metrics["Q_total"] = Q_total
            metrics["specific_gas_production"] = float(q_gas / Q_total)
            metrics["specific_ch4_production"] = float(q_ch4 / Q_total)
            feedstock = getattr(adm1, "feedstock", None)
            q_volume = float(np.sum(feedstock.actual_Q(Q))) if hasattr(feedstock, "actual_Q") else Q_total
            if q_volume > 0:
                metrics["HRT"] = float(adm1.V_liq / q_volume)

    except Exception as e:  # noqa: BLE001 - capture any metric-computation error into the result
        metrics["error"] = str(e)

    return metrics
