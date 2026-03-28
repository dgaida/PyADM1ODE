"""Unit tests for solver helpers and adaptive solver behavior."""

from types import SimpleNamespace

import numpy as np
import pytest

import pyadm1.core.solver as solver_module
from pyadm1.core.solver import AdaptiveODESolver, ODESolver, SolverConfig, create_solver


def _ok_result(t, y):
    return SimpleNamespace(
        success=True,
        message="ok",
        t=np.asarray(t, dtype=float),
        y=np.asarray(y, dtype=float),
    )


class TestODESolverSolve:
    def test_solve_inserts_interval_bounds_when_default_t_eval_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = {}

        def fake_solve_ivp(**kwargs):  # noqa: ANN001
            captured.update(kwargs)
            return _ok_result(kwargs["t_eval"], [[1.0] * len(kwargs["t_eval"])])

        monkeypatch.setattr(solver_module.scipy.integrate, "solve_ivp", fake_solve_ivp)
        solver = ODESolver()
        solver.solve(lambda t, y: y, (0.0, 0.01), [1.0])  # noqa: ARG005

        assert np.allclose(captured["t_eval"], np.array([0.0, 0.01]))

    def test_solve_passes_lsoda_min_step_and_first_step(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = {}

        def fake_solve_ivp(**kwargs):  # noqa: ANN001
            captured.update(kwargs)
            return _ok_result([0.0, 1.0], [[1.0, 0.5]])

        monkeypatch.setattr(solver_module.scipy.integrate, "solve_ivp", fake_solve_ivp)
        cfg = SolverConfig(method="LSODA", min_step=1e-4, max_step=0.2, first_step=1e-3)
        ODESolver(cfg).solve(lambda t, y: y, (0.0, 1.0), [1.0], t_eval=np.array([0.0, 1.0]))  # noqa: ARG005

        assert captured["min_step"] == 1e-4
        assert captured["max_step"] == 0.2
        assert captured["first_step"] == 1e-3

    def test_solve_inserts_t_start_when_default_arange_misses_start(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured = {}

        def fake_arange(*args, **kwargs):  # noqa: ANN001
            return np.array([0.1, 0.15], dtype=float)

        def fake_solve_ivp(**kwargs):  # noqa: ANN001
            captured.update(kwargs)
            return _ok_result(kwargs["t_eval"], [[1.0] * len(kwargs["t_eval"])])

        monkeypatch.setattr(solver_module.np, "arange", fake_arange)
        monkeypatch.setattr(solver_module.scipy.integrate, "solve_ivp", fake_solve_ivp)

        ODESolver().solve(lambda t, y: y, (0.0, 0.2), [1.0])  # noqa: ARG005

        assert np.isclose(captured["t_eval"][0], 0.0)


class TestODESolverIterativeMethods:
    def test_solve_to_steady_state_converges(self, monkeypatch: pytest.MonkeyPatch) -> None:
        solver = ODESolver()
        calls = []

        def fake_solve(fun, t_span, y0, t_eval=None, dense_output=False):  # noqa: ANN001, ARG001
            calls.append((t_span, list(y0)))
            return _ok_result([t_span[0], t_span[1]], [[1.0, 1.0], [2.0, 2.0]])

        monkeypatch.setattr(solver, "solve", fake_solve)
        state, final_time, converged = solver.solve_to_steady_state(
            lambda t, y: y, [1.0, 2.0], check_interval=5.0
        )  # noqa: ARG005

        assert converged is True
        assert final_time == 5.0
        assert state == [1.0, 2.0]
        assert calls

    def test_solve_to_steady_state_returns_false_after_max_time(self, monkeypatch: pytest.MonkeyPatch) -> None:
        solver = ODESolver()
        values = iter(
            [
                np.array([[0.0, 1.0], [0.0, 1.0]], dtype=float),
                np.array([[1.0, 2.0], [1.0, 2.0]], dtype=float),
            ]
        )

        def fake_solve(fun, t_span, y0, t_eval=None, dense_output=False):  # noqa: ANN001, ARG001
            return _ok_result([t_span[0], t_span[1]], next(values))

        monkeypatch.setattr(solver, "solve", fake_solve)
        state, final_time, converged = solver.solve_to_steady_state(
            lambda t, y: y,
            [0.0, 0.0],
            max_time=20.0,
            check_interval=10.0,
            steady_state_tol=1e-12,  # noqa: ARG005
        )

        assert converged is False
        assert final_time == 20.0
        assert state == [2.0, 2.0]

    def test_solve_sequential_returns_state_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        solver = ODESolver()
        terminal_states = iter(
            [
                np.array([[1.0, 2.0], [0.0, 1.0]], dtype=float),
                np.array([[2.0, 3.0], [1.0, 2.0]], dtype=float),
            ]
        )

        def fake_solve(fun, t_span, y0, t_eval=None, dense_output=False):  # noqa: ANN001, ARG001
            return _ok_result([t_span[0], t_span[1]], next(terminal_states))

        monkeypatch.setattr(solver, "solve", fake_solve)
        states = solver.solve_sequential(lambda t, y: y, [0.0, 1.0, 2.0], [1.0, 0.0])  # noqa: ARG005

        assert states[0] == [1.0, 0.0]
        assert states[1] == [2.0, 1.0]
        assert states[2] == [3.0, 2.0]


class TestAdaptiveODESolver:
    def test_adaptive_solver_init_sets_fields(self) -> None:
        solver = AdaptiveODESolver(adaptive=True, min_rtol=1e-9, max_rtol=1e-3)
        assert solver.adaptive is True
        assert solver.min_rtol == 1e-9
        assert solver.max_rtol == 1e-3
        assert solver._solution_history == []

    def test_adaptive_solve_calls_update_tolerances_when_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        result = _ok_result([0.0, 1.0], [[1.0, 0.5]])
        monkeypatch.setattr(ODESolver, "solve", lambda self, *args, **kwargs: result)

        solver = AdaptiveODESolver(adaptive=True)
        called = []
        monkeypatch.setattr(solver, "_update_tolerances", lambda r: called.append(r))

        out = solver.solve(lambda t, y: y, (0.0, 1.0), [1.0])  # noqa: ARG005

        assert out is result
        assert called == [result]

    def test_update_tolerances_tightens_for_high_curvature(self) -> None:
        solver = AdaptiveODESolver(config=SolverConfig(rtol=1e-6, atol=1e-8))
        result = _ok_result([0.0, 0.5, 1.0], [[0.0, 1.0, 0.0]])

        solver._update_tolerances(result)

        assert solver.config.rtol < 1e-6
        assert solver.config.atol < 1e-8

    def test_update_tolerances_relaxes_for_smooth_solution(self) -> None:
        solver = AdaptiveODESolver(config=SolverConfig(rtol=1e-6, atol=1e-8), max_rtol=1e-4)
        result = _ok_result([0.0, 0.5, 1.0, 1.5], [[1.0, 2.0, 3.0, 4.0]])

        solver._update_tolerances(result)

        assert solver.config.rtol > 1e-6
        assert solver.config.atol > 1e-8


class TestCreateSolverFactory:
    def test_create_solver_returns_adaptive_solver_when_requested(self) -> None:
        solver = create_solver(method="BDF", adaptive=True)
        assert isinstance(solver, AdaptiveODESolver)
