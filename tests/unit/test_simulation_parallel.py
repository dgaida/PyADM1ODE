"""Unit tests for ParallelSimulator (sequential mode + helpers)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import pytest

import pyadm1.simulation.parallel as parallel_mod
from pyadm1.simulation.parallel import (
    MonteCarloConfig,
    ParallelSimulator,
    ParameterSweepConfig,
    ScenarioResult,
)


class _DummyADM1Config:
    """Minimal stand-in for ADM1 that satisfies _serialize_adm1's attribute access."""

    V_liq = 1200.0
    _V_gas = 216.0
    _T_ad = 315.15
    feedstock = None  # _serialize_adm1 handles this gracefully


def _fake_worker(scenario_data, **kwargs):
    scenario_id, parameters = scenario_data
    return ScenarioResult(
        scenario_id=scenario_id,
        parameters=parameters,
        success=(scenario_id % 2 == 0),
        duration=kwargs["duration"],
        metrics={"m": float(scenario_id)},
        execution_time=0.01,
    )


def test_get_mp_context_env_override(monkeypatch):
    called = {}

    monkeypatch.setattr(parallel_mod.os, "getenv", lambda key: "spawn")
    monkeypatch.setattr(
        parallel_mod.mp,
        "get_context",
        lambda method: called.setdefault("method", method) or "ctx",
    )

    parallel_mod._get_mp_context()
    assert called["method"] == "spawn"


def test_get_mp_context_linux_default(monkeypatch):
    called = {}

    monkeypatch.setattr(parallel_mod.os, "getenv", lambda key: None)
    monkeypatch.setattr(parallel_mod.sys, "platform", "linux")
    monkeypatch.setattr(
        parallel_mod.mp,
        "get_context",
        lambda method: called.setdefault("method", method) or "ctx",
    )

    parallel_mod._get_mp_context()
    assert called["method"] == "forkserver"


def test_run_scenarios_sequential_verbose_progress_and_summary(monkeypatch, capsys):
    sim = ParallelSimulator(_DummyADM1Config(), n_workers=1, verbose=True)
    monkeypatch.setattr(parallel_mod, "_run_single_scenario", _fake_worker)

    times = iter([100.0, 102.0])
    monkeypatch.setattr(parallel_mod.time, "time", lambda: next(times))

    results = sim.run_scenarios([{"Q": [1] * 10}], duration=1.0, initial_state=[0.0] * 41)

    assert len(results) == 1
    out = capsys.readouterr().out
    assert "Starting parallel simulation with 1 scenarios" in out
    assert "Using 1 worker processes" in out
    assert "Completed 1/1 scenarios" in out
    assert "Simulation complete:" in out
    assert "Successful: 1" in out


def test_parameter_sweep_verbose_prints(monkeypatch, capsys):
    sim = ParallelSimulator(_DummyADM1Config(), n_workers=1, verbose=True)
    captured = {}

    def fake_run(scenarios, duration, initial_state, **kwargs):
        captured["scenarios"] = scenarios
        return ["ok"]

    monkeypatch.setattr(sim, "run_scenarios", fake_run)
    cfg = ParameterSweepConfig(parameter_name="k_m_ac", values=[7.0, 8.0], other_params={"Q": [1] * 10})

    result = sim.parameter_sweep(cfg, duration=1.0, initial_state=[0.0] * 41)

    assert result == ["ok"]
    assert captured["scenarios"][0]["k_m_ac"] == 7.0
    out = capsys.readouterr().out
    assert "Parameter sweep: k_m_ac" in out
    assert "Values: [7.0, 8.0]" in out


def test_multi_parameter_sweep_verbose_prints(monkeypatch, capsys):
    sim = ParallelSimulator(_DummyADM1Config(), n_workers=1, verbose=True)
    captured = {}

    def fake_run(scenarios, duration, initial_state, **kwargs):
        captured["count"] = len(scenarios)
        return scenarios

    monkeypatch.setattr(sim, "run_scenarios", fake_run)

    results = sim.multi_parameter_sweep(
        {"k_m_ac": [7.0, 8.0], "k_m_pro": [13.0]},
        1.0,
        [0.0] * 41,
        fixed_params={"Q": [1] * 10},
    )

    assert len(results) == 2
    assert captured["count"] == 2
    out = capsys.readouterr().out
    assert "Multi-parameter sweep:" in out
    assert "k_m_ac: 2 values" in out
    assert "Total combinations: 2" in out


def test_monte_carlo_verbose_prints(monkeypatch, capsys):
    sim = ParallelSimulator(_DummyADM1Config(), n_workers=1, verbose=True)
    monkeypatch.setattr(
        sim,
        "run_scenarios",
        lambda scenarios, duration, initial_state, **kwargs: scenarios,
    )

    cfg = MonteCarloConfig(
        n_samples=2,
        parameter_distributions={"k_m_ac": (8.0, 0.5)},
        fixed_params={"Q": [1] * 10},
        seed=1,
    )

    results = sim.monte_carlo(cfg, 1.0, [0.0] * 41)

    assert len(results) == 2
    out = capsys.readouterr().out
    assert "Monte Carlo simulation:" in out
    assert "Samples: 2" in out
    assert "k_m_ac: N(8.0, 0.5" in out


@pytest.mark.parametrize(
    "results, expected_error",
    [
        ([ScenarioResult(0, {}, False, 1.0, error="x")], "No successful scenarios"),
        ([], "No scenarios to summarize"),
    ],
)
def test_summarize_results_no_success_cases(results, expected_error):
    sim = ParallelSimulator(_DummyADM1Config(), n_workers=1, verbose=False)
    summary = sim.summarize_results(results)
    assert summary["error"] == expected_error


def test_compute_scenario_metrics_returns_gas_metrics():
    """End-to-end call against a real ADM1 instance."""
    from pyadm1 import Feedstock
    from pyadm1.core.adm1 import ADM1, STATE_SIZE

    fs = Feedstock(
        ["maize_silage_milk_ripeness", "swine_manure"],
        feeding_freq=24,
        total_simtime=3,
    )
    adm = ADM1(fs, V_liq=1200.0, V_gas=216.0, T_ad=315.15)
    state = [0.01] * STATE_SIZE
    state[37:41] = [1.0e-5, 0.65, 0.33, 0.65 + 0.33 + 1.0e-5]

    metrics = parallel_mod._compute_scenario_metrics(adm, state, [11.4, 6.1])

    assert "Q_gas" in metrics
    assert "Q_ch4" in metrics
    assert "pH" in metrics
    assert "HRT" in metrics


def test_compute_scenario_metrics_handles_invalid_input():
    """Outer try/except should wrap any error into the metrics dict."""

    class BrokenADM1:
        V_liq = 1.0

        def calc_gas(self, *args):
            raise RuntimeError("boom")

    metrics = parallel_mod._compute_scenario_metrics(BrokenADM1(), [0.0] * 10, [1.0])
    assert "error" in metrics


def test_get_mp_context_non_linux_default(monkeypatch):
    """Windows / macOS have no fork, so the pool must be spawned."""
    called = {}

    monkeypatch.setattr(parallel_mod.os, "getenv", lambda key: None)
    monkeypatch.setattr(parallel_mod.sys, "platform", "win32")
    monkeypatch.setattr(
        parallel_mod.mp,
        "get_context",
        lambda method: called.setdefault("method", method) or "ctx",
    )

    parallel_mod._get_mp_context()
    assert called["method"] == "spawn"


class _InlinePool:
    """Stand-in for ``multiprocessing.Pool`` that runs the work in this process.

    Keeps the worker-pool branch of ``run_scenarios`` testable without paying
    for process spawn (and without needing picklable fixtures on Windows).
    """

    instances: ClassVar[list[_InlinePool]] = []

    def __init__(self, processes=None):
        self.processes = processes
        self.map_calls = 0
        self.imap_calls = 0
        _InlinePool.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def map(self, func, iterable):
        self.map_calls += 1
        return [func(item) for item in iterable]

    def imap(self, func, iterable):
        self.imap_calls += 1
        for item in iterable:
            yield func(item)


def _use_inline_pool(monkeypatch):
    _InlinePool.instances = []
    monkeypatch.setattr(parallel_mod, "_get_mp_context", lambda: SimpleNamespace(Pool=_InlinePool))
    monkeypatch.setattr(parallel_mod, "_run_single_scenario", _fake_worker)


def test_run_scenarios_uses_a_worker_pool_for_multiple_scenarios(monkeypatch):
    _use_inline_pool(monkeypatch)
    sim = ParallelSimulator(_DummyADM1Config(), n_workers=3, verbose=False)
    scenarios = [{"Q": [float(i)] * 10} for i in range(4)]

    results = sim.run_scenarios(scenarios, duration=1.0, initial_state=[0.0] * 41)

    assert [r.scenario_id for r in results] == [0, 1, 2, 3]
    pool = _InlinePool.instances[0]
    assert pool.processes == 3
    assert pool.map_calls == 1  # non-verbose takes the plain map path
    assert pool.imap_calls == 0


def test_pooled_run_reports_progress_when_verbose(monkeypatch, capsys):
    _use_inline_pool(monkeypatch)
    sim = ParallelSimulator(_DummyADM1Config(), n_workers=2, verbose=True)
    scenarios = [{"Q": [float(i)] * 10} for i in range(12)]

    results = sim.run_scenarios(scenarios, duration=1.0, initial_state=[0.0] * 41)

    assert len(results) == 12
    pool = _InlinePool.instances[0]
    assert pool.imap_calls == 1  # verbose streams results to report progress
    assert pool.map_calls == 0
    out = capsys.readouterr().out
    assert "Completed 10/12 scenarios" in out
    assert "Completed 12/12 scenarios" in out


def test_a_single_scenario_stays_sequential_even_with_many_workers(monkeypatch):
    """Spawning a pool for one scenario would cost more than it saves."""
    _use_inline_pool(monkeypatch)
    sim = ParallelSimulator(_DummyADM1Config(), n_workers=8, verbose=False)

    sim.run_scenarios([{"Q": [1.0] * 10}], duration=1.0, initial_state=[0.0] * 41)

    assert _InlinePool.instances == []


def test_summarize_results_defaults_to_the_metrics_of_the_first_success():
    sim = ParallelSimulator(_DummyADM1Config(), n_workers=1, verbose=False)
    results = [
        ScenarioResult(0, {}, False, 1.0, error="boom"),
        ScenarioResult(1, {}, True, 1.0, metrics={"Q_gas": 100.0, "pH": 7.4}),
        ScenarioResult(2, {}, True, 1.0, metrics={"Q_gas": 200.0, "pH": 7.6}),
    ]

    summary = sim.summarize_results(results)

    assert "error" not in summary
    assert set(summary["metrics"]) == {"Q_gas", "pH"}
    assert summary["metrics"]["Q_gas"]["mean"] == pytest.approx(150.0)
    assert summary["metrics"]["pH"]["mean"] == pytest.approx(7.5)


def test_summarize_results_honours_an_explicit_metric_selection():
    sim = ParallelSimulator(_DummyADM1Config(), n_workers=1, verbose=False)
    results = [ScenarioResult(0, {}, True, 1.0, metrics={"Q_gas": 100.0, "pH": 7.4})]

    summary = sim.summarize_results(results, metrics=["pH"])

    assert set(summary["metrics"]) == {"pH"}


# --------------------------------------------------------------------------
# The worker function itself (runs in a child process in production)
# --------------------------------------------------------------------------
def _worker_config(substrates: list[str] | None = None) -> dict:
    return {
        "V_liq": 1200.0,
        "V_gas": 216.0,
        "T_ad": 315.15,
        "feedstock_substrates": substrates or [],
        "feeding_freq": 24,
    }


def _worker_initial_state() -> list[float]:
    from pyadm1.core.adm1 import STATE_SIZE

    state = [0.01] * STATE_SIZE
    state[37:41] = [1.0e-5, 0.65, 0.33, 0.65 + 0.33 + 1.0e-5]
    return state


def _run_worker(config, initial_state, **overrides):
    kwargs = {
        "adm1_config": config,
        "duration": 1.0,
        "initial_state": initial_state,
        "dt": 1.0 / 24.0,
        "compute_metrics": True,
        "save_time_series": False,
        "verbose": False,
    }
    kwargs.update(overrides)
    return parallel_mod._run_single_scenario((0, {"Q": [10.0, 5.0] + [0.0] * 8}), **kwargs)


def test_worker_runs_an_autonomous_scenario_without_a_feedstock():
    """No substrate ids -> no influent at all; the digester runs autonomously.

    Regression guard: the worker used to call ``create_influent`` unconditionally
    and crashed on the absent feedstock, so every scenario of a feedstock-less
    ParallelSimulator failed.
    """
    result = _run_worker(_worker_config(), _worker_initial_state())

    if not result.success:
        pytest.fail(f"worker failed: {result.error}")
    assert result.scenario_id == 0
    assert result.metrics["Q_gas"] >= 0.0
    assert result.time_series is None
    assert result.execution_time >= 0.0


def test_worker_runs_a_scenario_with_a_feedstock():
    config = _worker_config(["maize_silage_milk_ripeness", "swine_manure"])

    result = _run_worker(config, _worker_initial_state())

    if not result.success:
        pytest.fail(f"worker failed: {result.error}")
    assert result.metrics["Q_gas"] >= 0.0


def test_the_influent_wiring_actually_changes_the_trajectory():
    """A fed run must not end up in the same state as an autonomous one."""
    initial_state = _worker_initial_state()
    fed = _run_worker(_worker_config(["maize_silage_milk_ripeness", "swine_manure"]), initial_state)
    autonomous = _run_worker(_worker_config(), initial_state)

    assert fed.success and autonomous.success
    assert fed.final_state != autonomous.final_state
    # The feed dilutes/loads the reactor, so the soluble sugar pool differs.
    assert fed.final_state[0] != pytest.approx(autonomous.final_state[0])


def test_worker_can_return_a_time_series_tail():
    result = _run_worker(_worker_config(), _worker_initial_state(), save_time_series=True)

    if not result.success:
        pytest.fail(f"worker failed: {result.error}")
    assert set(result.time_series) == {"Q_gas", "Q_ch4", "pH"}
    assert len(result.time_series["Q_gas"]) <= 10
    assert all(isinstance(v, float) for v in result.time_series["Q_gas"])


def test_worker_can_skip_metric_computation():
    result = _run_worker(_worker_config(), _worker_initial_state(), compute_metrics=False)

    assert result.success
    assert result.metrics == {}


def test_worker_reports_a_wrong_sized_initial_state_instead_of_crashing():
    """A bad scenario must be reported back, never take the whole batch down."""
    result = _run_worker(_worker_config(), [0.01] * 10)

    assert result.success is False
    assert "initial_state must have 41 elements; got 10" in result.error
    assert result.scenario_id == 0
    assert result.parameters == {"Q": [10.0, 5.0] + [0.0] * 8}
    assert result.execution_time >= 0.0


def test_worker_applies_scenario_calibration_and_temperature_overrides():
    config = _worker_config()
    scenario = (1, {"Q": [10.0, 5.0] + [0.0] * 8, "T_ad": 311.15, "k_m_ac": 6.0})

    result = parallel_mod._run_single_scenario(
        scenario,
        adm1_config=config,
        duration=1.0,
        initial_state=_worker_initial_state(),
        dt=1.0 / 24.0,
        compute_metrics=True,
        save_time_series=False,
        verbose=False,
    )

    if not result.success:
        pytest.fail(f"worker failed: {result.error}")
    assert result.scenario_id == 1
    assert result.parameters["T_ad"] == 311.15
