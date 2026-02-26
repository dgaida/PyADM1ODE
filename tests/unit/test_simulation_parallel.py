import sys
import types

import pytest

import pyadm1.simulation.parallel as parallel_mod
from pyadm1.simulation.parallel import MonteCarloConfig, ParallelSimulator, ParameterSweepConfig, ScenarioResult


class _DummyADM1Config:
    V_liq = 100.0
    _V_gas = 20.0
    _T_ad = 308.15


class _FakePool:
    def __init__(self):
        self.imap_calls = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def imap(self, func, items):
        self.imap_calls += 1
        for item in items:
            yield func(item)

    def map(self, func, items):
        return [func(item) for item in items]


class _FakeContext:
    def __init__(self):
        self.pool = _FakePool()
        self.processes = None

    def Pool(self, processes):
        self.processes = processes
        return self.pool


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
    monkeypatch.setattr(parallel_mod.mp, "get_context", lambda method: called.setdefault("method", method) or "ctx")

    parallel_mod._get_mp_context()

    assert called["method"] == "spawn"


def test_get_mp_context_linux_default(monkeypatch):
    called = {}

    monkeypatch.setattr(parallel_mod.os, "getenv", lambda key: None)
    monkeypatch.setattr(parallel_mod.sys, "platform", "linux")
    monkeypatch.setattr(parallel_mod.mp, "get_context", lambda method: called.setdefault("method", method) or "ctx")

    parallel_mod._get_mp_context()

    assert called["method"] == "forkserver"


def test_run_scenarios_sequential_verbose_progress_and_summary(monkeypatch, capsys):
    sim = ParallelSimulator(_DummyADM1Config(), n_workers=1, verbose=True)
    monkeypatch.setattr(parallel_mod, "_run_single_scenario", _fake_worker)

    times = iter([100.0, 102.0])
    monkeypatch.setattr(parallel_mod.time, "time", lambda: next(times))

    results = sim.run_scenarios([{"Q": [1] * 10}], duration=1.0, initial_state=[0.0] * 37)

    assert len(results) == 1
    out = capsys.readouterr().out
    assert "Starting parallel simulation with 1 scenarios" in out
    assert "Using 1 worker processes" in out
    assert "Completed 1/1 scenarios" in out
    assert "Simulation complete:" in out
    assert "Successful: 1" in out


def test_run_scenarios_parallel_verbose_uses_imap_and_progress(monkeypatch, capsys):
    sim = ParallelSimulator(_DummyADM1Config(), n_workers=2, verbose=True)
    fake_ctx = _FakeContext()

    monkeypatch.setattr(parallel_mod, "_run_single_scenario", _fake_worker)
    monkeypatch.setattr(parallel_mod, "_get_mp_context", lambda: fake_ctx)
    times = iter([1.0, 2.5])
    monkeypatch.setattr(parallel_mod.time, "time", lambda: next(times))

    results = sim.run_scenarios([{"a": 1}, {"a": 2}], duration=2.0, initial_state=[0.0] * 37)

    assert len(results) == 2
    assert fake_ctx.processes == 2
    assert fake_ctx.pool.imap_calls == 1
    out = capsys.readouterr().out
    assert "Completed 2/2 scenarios" in out
    assert "Failed: 1" in out


def test_parameter_sweep_verbose_prints(monkeypatch, capsys):
    sim = ParallelSimulator(_DummyADM1Config(), n_workers=1, verbose=True)
    captured = {}

    def fake_run(scenarios, duration, initial_state, **kwargs):
        captured["scenarios"] = scenarios
        return ["ok"]

    monkeypatch.setattr(sim, "run_scenarios", fake_run)
    cfg = ParameterSweepConfig(parameter_name="k_dis", values=[0.4, 0.5], other_params={"Q": [1] * 10})

    result = sim.parameter_sweep(cfg, duration=1.0, initial_state=[0.0] * 37)

    assert result == ["ok"]
    assert captured["scenarios"][0]["k_dis"] == 0.4
    out = capsys.readouterr().out
    assert "Parameter sweep: k_dis" in out
    assert "Values: [0.4, 0.5]" in out


def test_multi_parameter_sweep_verbose_prints(monkeypatch, capsys):
    sim = ParallelSimulator(_DummyADM1Config(), n_workers=1, verbose=True)
    captured = {}

    def fake_run(scenarios, duration, initial_state, **kwargs):
        captured["count"] = len(scenarios)
        return scenarios

    monkeypatch.setattr(sim, "run_scenarios", fake_run)

    results = sim.multi_parameter_sweep({"k_dis": [0.4, 0.5], "Y_su": [0.1]}, 1.0, [0.0] * 37, fixed_params={"Q": [1] * 10})

    assert len(results) == 2
    assert captured["count"] == 2
    out = capsys.readouterr().out
    assert "Multi-parameter sweep:" in out
    assert "k_dis: 2 values" in out
    assert "Total combinations: 2" in out


def test_monte_carlo_verbose_prints(monkeypatch, capsys):
    sim = ParallelSimulator(_DummyADM1Config(), n_workers=1, verbose=True)
    monkeypatch.setattr(sim, "run_scenarios", lambda scenarios, duration, initial_state, **kwargs: scenarios)

    cfg = MonteCarloConfig(
        n_samples=2,
        parameter_distributions={"k_dis": (0.5, 0.05)},
        fixed_params={"Q": [1] * 10},
        seed=1,
    )

    results = sim.monte_carlo(cfg, 1.0, [0.0] * 37)

    assert len(results) == 2
    out = capsys.readouterr().out
    assert "Monte Carlo simulation:" in out
    assert "Samples: 2" in out
    assert "Parameters with uncertainty:" in out
    assert "k_dis: N(0.5, 0.05" in out


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


def test_compute_scenario_metrics_ignores_dll_errors(monkeypatch):
    class FakeADM1:
        V_liq = 100.0

        @staticmethod
        def calc_gas(*args):
            return (10.0, 6.0, 4.0, 1.2)

    class FakeADMState:
        @staticmethod
        def calcPHOfADMstate(_state):
            raise RuntimeError("dll failed")

    monkeypatch.setattr(parallel_mod.mp, "current_process", lambda: types.SimpleNamespace(name="MainProcess"))
    monkeypatch.setitem(sys.modules, "biogas", types.SimpleNamespace(ADMstate=FakeADMState))

    state = [0.0] * 37
    state[33:37] = [1.0, 2.0, 3.0, 4.0]
    metrics = parallel_mod._compute_scenario_metrics(FakeADM1(), state, [5.0, 0.0])

    assert metrics["Q_gas"] == 10.0
    assert metrics["HRT"] == 20.0
    assert "pH" not in metrics
    assert "error" not in metrics


def test_compute_scenario_metrics_outer_exception_sets_error():
    class FakeADM1:
        V_liq = 100.0

        @staticmethod
        def calc_gas(*args):
            return (1.0, 1.0, 0.0, 1.0)

    metrics = parallel_mod._compute_scenario_metrics(FakeADM1(), [0.0] * 10, [1.0])

    assert "error" in metrics
