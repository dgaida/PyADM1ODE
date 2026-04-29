# -*- coding: utf-8 -*-
"""Unit tests for ParallelSimulator (sequential mode + helpers)."""

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
