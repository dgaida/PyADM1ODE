"""Parity tests: the PyTorch ADM1da right-hand side must match ``ADM1.ADM_ODE``.

The torch backend is only useful if it reproduces the numpy model's values
bit-closely (differences bounded by the closed-form vs. Newton pH solve and
float rounding). These tests lock that equivalence at float64 before the
backend is wired into the digester.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pyadm1 import Feedstock
from pyadm1.core.adm1 import _IDX_P_TOTAL, ADM1, STATE_SIZE
from pyadm1.core.adm1_torch import (
    _GAS_LEAK_SLOPE,
    Adm1TorchParams,
    adm1da_rhs_torch,
    calc_gas_quasi_steady_torch,
    calc_gas_torch,
    gas_equilibrium_torch,
    ph_torch,
    tac_torch,
    ts_torch,
    vfa_torch,
)

# Plausible order-of-magnitude base state (units per the ADM1da index map).
_BASE_STATE = np.array(
    [
        0.012,
        0.005,
        0.10,
        0.012,
        0.013,
        0.016,
        0.20,
        2.4e-7,
        0.05,
        0.15,  # 0-9
        0.13,
        0.02,  # 10-11
        2.0,
        2.0,
        2.0,
        0.5,
        0.5,
        0.5,
        2.0,
        2.0,
        2.0,
        5.0,  # 12-21
        0.5,
        0.3,
        0.2,
        0.3,
        0.4,
        0.6,
        0.3,  # 22-28
        0.04,
        0.02,
        0.011,
        0.013,
        0.016,
        0.20,
        0.12,
        0.004,  # 29-36
        1.0e-5,
        0.55,
        0.45,
        1.05,  # 37-40
    ],
    dtype=np.float64,
)


def _make_adm1(T_ad: float = 308.15) -> ADM1:
    """Build a bare ADM1 (no feedstock needed; we set influent fields directly)."""
    return ADM1(feedstock=None, V_liq=1977.0, V_gas=304.0, T_ad=T_ad)


def _random_state(rng: np.random.Generator) -> np.ndarray:
    """Perturb the base state by a lognormal factor, staying strictly positive."""
    factor = np.exp(rng.normal(0.0, 0.35, size=STATE_SIZE))
    state = _BASE_STATE * factor
    # Keep the total gas pressure safely positive for the p/pTOTAL terms.
    state[40] = max(state[40], 0.3)
    return state


def _numpy_dxdt(adm1: ADM1, state: np.ndarray) -> np.ndarray:
    return np.asarray(adm1.ADM_ODE(0.0, list(state)), dtype=np.float64)


def _torch_dxdt(adm1: ADM1, state: np.ndarray) -> np.ndarray:
    params = Adm1TorchParams.from_adm1(adm1)
    x = torch.tensor(state, dtype=torch.float64)
    return adm1da_rhs_torch(x, params).detach().numpy()


def _assert_parity(adm1: ADM1, state: np.ndarray, rtol: float = 1e-6, atol: float = 1e-9) -> None:
    ref = _numpy_dxdt(adm1, state)
    got = _torch_dxdt(adm1, state)
    assert got.shape == (STATE_SIZE,)
    np.testing.assert_allclose(got, ref, rtol=rtol, atol=atol)


# --------------------------------------------------------------------------
# Parity across scenarios
# --------------------------------------------------------------------------
def test_parity_no_feed():
    """Autonomous case: no influent (q_ad = 0, s_in = 0)."""
    adm1 = _make_adm1()
    rng = np.random.default_rng(0)
    for _ in range(25):
        _assert_parity(adm1, _random_state(rng))


def test_parity_with_feed():
    """With a non-trivial influent composition and flow."""
    adm1 = _make_adm1()
    rng = np.random.default_rng(1)
    adm1._state_input = list(np.abs(rng.normal(0.5, 0.5, size=37)))
    adm1._Q = [120.0]
    for _ in range(25):
        _assert_parity(adm1, _random_state(rng))


def test_parity_with_q_out_override():
    """Dynamic-volume path: outflow is driven by an external override."""
    adm1 = _make_adm1()
    rng = np.random.default_rng(2)
    adm1._state_input = list(np.abs(rng.normal(0.5, 0.5, size=37)))
    adm1._Q = [90.0]
    adm1._Q_out_override = 42.0
    for _ in range(15):
        _assert_parity(adm1, _random_state(rng))


def test_parity_with_calibration_overrides():
    """k_L_a / k_p / Henry-constant calibration overrides must be picked up."""
    adm1 = _make_adm1()
    adm1.set_calibration_parameters({"k_L_a": 150.0, "k_p": 5.0e3, "K_H_co2": adm1._K_H_co2 * 1.1})
    rng = np.random.default_rng(3)
    adm1._Q = [60.0]
    adm1._state_input = list(np.abs(rng.normal(0.4, 0.4, size=37)))
    for _ in range(15):
        _assert_parity(adm1, _random_state(rng))


def test_parity_co2_free_clamp():
    """Exercise the S_co2 < S_hco3 branch (S_co2_free clamped to zero)."""
    adm1 = _make_adm1()
    rng = np.random.default_rng(4)
    state = _random_state(rng)
    state[9] = 0.05  # S_co2
    state[35] = 0.20  # S_hco3 > S_co2 -> free CO2 clamped
    _assert_parity(adm1, state)


def test_parity_other_temperature():
    """Temperature-corrected kinetics / inhibition must also match."""
    adm1 = _make_adm1(T_ad=313.15)
    rng = np.random.default_rng(5)
    adm1._Q = [75.0]
    adm1._state_input = list(np.abs(rng.normal(0.5, 0.5, size=37)))
    for _ in range(15):
        _assert_parity(adm1, _random_state(rng))


# --------------------------------------------------------------------------
# Shape / batching
# --------------------------------------------------------------------------
def test_batched_shapes_and_values():
    """A batch [B, 41] must return [B, 41] and match per-row parity."""
    adm1 = _make_adm1()
    adm1._Q = [100.0]
    rng = np.random.default_rng(6)
    adm1._state_input = list(np.abs(rng.normal(0.5, 0.5, size=37)))
    batch = np.stack([_random_state(rng) for _ in range(8)], axis=0)
    params = Adm1TorchParams.from_adm1(adm1)
    out = adm1da_rhs_torch(torch.tensor(batch, dtype=torch.float64), params).detach().numpy()
    assert out.shape == (8, STATE_SIZE)
    for i in range(8):
        ref = _numpy_dxdt(adm1, batch[i])
        np.testing.assert_allclose(out[i], ref, rtol=1e-6, atol=1e-9)


# --------------------------------------------------------------------------
# Differentiability (the whole point of the torch backend)
# --------------------------------------------------------------------------
def test_autograd_flows_through_rhs():
    """dx/dt must be differentiable w.r.t. the state with finite gradients."""
    adm1 = _make_adm1()
    adm1._Q = [100.0]
    rng = np.random.default_rng(7)
    adm1._state_input = list(np.abs(rng.normal(0.5, 0.5, size=37)))
    params = Adm1TorchParams.from_adm1(adm1)

    x = torch.tensor(_random_state(rng), dtype=torch.float64, requires_grad=True)
    dxdt = adm1da_rhs_torch(x, params)
    dxdt.sum().backward()

    assert x.grad is not None
    assert x.grad.shape == (STATE_SIZE,)
    assert torch.isfinite(x.grad).all()


# --------------------------------------------------------------------------
# Backend selection (rhs_callable) + step equivalence through the solver
# --------------------------------------------------------------------------
def test_rhs_callable_numpy_is_adm_ode():
    """Default backend must return the untouched numpy ADM_ODE."""
    adm1 = _make_adm1()
    assert adm1.backend == "numpy"
    assert adm1.rhs_callable() == adm1.ADM_ODE


def test_invalid_backend_rejected():
    """An unknown backend must be rejected at construction time."""
    with pytest.raises(ValueError):
        ADM1(feedstock=None, backend="jax")


def _configure(adm1: ADM1, rng: np.random.Generator) -> None:
    adm1._Q = [110.0]
    adm1._state_input = list(np.abs(rng.normal(0.5, 0.5, size=37)))


def test_step_equivalence_through_solver():
    """Integrating with backend='numpy' vs 'torch' must give the same trajectory."""
    from scipy.integrate import solve_ivp

    rng = np.random.default_rng(11)
    y0 = _random_state(rng)

    adm1_np = _make_adm1()
    adm1_pt = ADM1(feedstock=None, V_liq=1977.0, V_gas=304.0, backend="torch")
    _configure(adm1_np, np.random.default_rng(99))
    _configure(adm1_pt, np.random.default_rng(99))  # identical influent

    dt = 1.0 / 24.0
    y_np = y0.copy()
    y_pt = y0.copy()
    for _ in range(5):
        r_np = solve_ivp(adm1_np.rhs_callable(), (0.0, dt), y_np, method="BDF", rtol=1e-6, atol=1e-8)
        r_pt = solve_ivp(adm1_pt.rhs_callable(), (0.0, dt), y_pt, method="BDF", rtol=1e-6, atol=1e-8)
        assert r_np.success and r_pt.success
        y_np = r_np.y[:, -1]
        y_pt = r_pt.y[:, -1]
        np.testing.assert_allclose(y_pt, y_np, rtol=1e-5, atol=1e-8)


def test_torch_backend_sets_q_s_loss_last():
    """The torch adapter must mirror ADM_ODE's cached sludge-volume side effect."""
    rng = np.random.default_rng(12)
    state = _random_state(rng)

    adm1_np = _make_adm1()
    adm1_pt = ADM1(feedstock=None, V_liq=1977.0, V_gas=304.0, backend="torch")

    adm1_np.ADM_ODE(0.0, list(state))
    adm1_pt.rhs_callable()(0.0, state)

    assert adm1_pt._q_S_loss_last == pytest.approx(adm1_np._q_S_loss_last, rel=1e-9)


# --------------------------------------------------------------------------
# Differentiable measurement map h(x): gas flows, pH, VFA, TAC
# --------------------------------------------------------------------------
def test_calc_gas_torch_parity():
    """calc_gas_torch must match ADM1.calc_gas over random gas-phase states."""
    adm1 = _make_adm1()
    params = Adm1TorchParams.from_adm1(adm1)
    rng = np.random.default_rng(20)
    for _ in range(25):
        state = _random_state(rng)
        q_gas, q_ch4, q_co2, _, _ = adm1.calc_gas(state[37], state[38], state[39], state[40])
        x = torch.tensor(state, dtype=torch.float64)
        g, c, o = (t.item() for t in calc_gas_torch(x, params))
        assert g == pytest.approx(q_gas, rel=1e-9, abs=1e-9)
        assert c == pytest.approx(q_ch4, rel=1e-9, abs=1e-9)
        assert o == pytest.approx(q_co2, rel=1e-9, abs=1e-9)


def test_calc_gas_parity_when_the_headspace_cannot_vent():
    """pTOTAL below the outlet balance: both backends floor the flow at zero."""
    adm1 = _make_adm1()
    params = Adm1TorchParams.from_adm1(adm1)
    state = _random_state(np.random.default_rng(23))
    state[40] = 0.5  # < p_ext - p_gas_h2o -> negative raw flow

    ref = adm1.calc_gas(state[37], state[38], state[39], state[40])[:3]
    got = [t.item() for t in calc_gas_torch(torch.tensor(state, dtype=torch.float64), params)]

    assert ref == (0.0, 0.0, 0.0)
    assert got == [0.0, 0.0, 0.0]


def test_calc_gas_parity_for_a_non_positive_wet_pressure():
    """Mirror of the numpy degenerate branch: species flows vanish, total does not."""
    adm1 = _make_adm1()
    params = Adm1TorchParams.from_adm1(adm1)
    state = _random_state(np.random.default_rng(24))
    state[37] = state[38] = state[39] = -0.05  # p_gas = -0.15 < -p_gas_h2o
    state[40] = 1.3
    assert -0.15 + adm1._p_gas_h2o < 0.0  # precondition for the branch

    q_gas_ref, q_ch4_ref, q_co2_ref, _q_h2o, _p = adm1.calc_gas(-0.05, -0.05, -0.05, 1.3)
    q_gas, q_ch4, q_co2 = (t.item() for t in calc_gas_torch(torch.tensor(state, dtype=torch.float64), params))

    assert q_gas == pytest.approx(q_gas_ref, rel=1e-12) and q_gas > 0.0
    assert (q_ch4, q_co2) == (q_ch4_ref, q_co2_ref) == (0.0, 0.0)


# --------------------------------------------------------------------------
# Leaky floor (soft=True): same values where gas flows, live gradient below
# --------------------------------------------------------------------------
def test_soft_and_hard_floor_agree_wherever_gas_actually_flows():
    adm1 = _make_adm1()
    params = Adm1TorchParams.from_adm1(adm1)
    rng = np.random.default_rng(25)
    for _ in range(10):
        state = _random_state(rng)
        state[40] = 1.3  # well above the outlet balance -> positive flow
        x = torch.tensor(state, dtype=torch.float64)
        hard = [t.item() for t in calc_gas_torch(x, params)]
        soft = [t.item() for t in calc_gas_torch(x, params, soft=True)]
        assert hard[0] > 0.0
        assert soft == hard


def test_soft_floor_replaces_the_zero_dead_zone_with_a_small_negative_slope():
    """Below zero flow the hard floor kills the gradient; the leaky one does not."""
    adm1 = _make_adm1()
    params = Adm1TorchParams.from_adm1(adm1)
    state = _random_state(np.random.default_rng(26))
    state[40] = 0.5  # no flow

    x = torch.tensor(state, dtype=torch.float64, requires_grad=True)
    q_hard = calc_gas_torch(x, params)[0]
    q_soft = calc_gas_torch(x, params, soft=True)[0]

    assert q_hard.item() == 0.0
    assert q_soft.item() < 0.0
    # The leak is the raw (negative) flow scaled by the documented slope.
    p = params
    raw = p.k_p * (state[40] + p.p_gas_h2o - p.p_ext) / (p.RT / 1000.0 * p.NQ) * p.V_liq
    assert q_soft.item() == pytest.approx(_GAS_LEAK_SLOPE * raw)

    q_soft.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert x.grad[_IDX_P_TOTAL] != 0.0


def test_hard_floor_has_no_gradient_below_zero_flow():
    """Contrast case documenting why the leaky floor exists."""
    adm1 = _make_adm1()
    params = Adm1TorchParams.from_adm1(adm1)
    state = _random_state(np.random.default_rng(27))
    state[40] = 0.5

    x = torch.tensor(state, dtype=torch.float64, requires_grad=True)
    calc_gas_torch(x, params)[0].backward()

    assert x.grad[_IDX_P_TOTAL] == 0.0


# --------------------------------------------------------------------------
# Feed-aware parameter copies
# --------------------------------------------------------------------------
def test_with_q_ad_matches_the_numpy_model_at_the_new_flow():
    """A rescaled q_ad must reproduce ADM_ODE with the same total influent flow."""
    adm1 = _make_adm1()
    rng = np.random.default_rng(30)
    adm1._state_input = list(np.abs(rng.normal(0.5, 0.5, size=37)))
    adm1._Q = [100.0]
    params = Adm1TorchParams.from_adm1(adm1)
    state = _random_state(rng)

    for q_new in (0.0, 45.0, 250.0):
        adm1._Q = [q_new]
        ref = _numpy_dxdt(adm1, state)
        got = adm1da_rhs_torch(torch.tensor(state, dtype=torch.float64), params.with_q_ad(q_new)).detach().numpy()
        np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-9)


def test_with_q_ad_returns_a_copy_and_keeps_the_composition():
    adm1 = _make_adm1()
    adm1._state_input = list(np.abs(np.random.default_rng(31).normal(0.5, 0.5, size=37)))
    adm1._Q = [100.0]
    params = Adm1TorchParams.from_adm1(adm1)

    rescaled = params.with_q_ad(250.0)

    assert rescaled is not params
    assert params.q_ad == pytest.approx(100.0)  # original untouched
    assert rescaled.q_ad == pytest.approx(250.0)
    assert rescaled.s_in == params.s_in  # only the rate changes, not the mix
    assert rescaled.V_liq == params.V_liq


# --------------------------------------------------------------------------
# Quasi-steady gas phase (the well-conditioned alternative to calc_gas_torch)
# --------------------------------------------------------------------------
def _qss_setup():
    """A deterministically gas-bearing liquid state.

    The quasi-steady solve only has a positive root when the dissolved gases can
    sustain the ambient pressure, so the free-CO2 margin (``S_co2 > S_hco3``) and
    the dissolved CH4 are pinned here rather than left to a random draw.
    """
    adm1 = _make_adm1()
    params = Adm1TorchParams.from_adm1(adm1)
    state = _BASE_STATE.copy()
    state[7] = 2.4e-7  # S_h2  (trace)
    state[8] = 0.055  # S_ch4
    state[9] = 0.20  # S_co2
    state[35] = 0.10  # S_hco3 -> free CO2 = 0.10
    return adm1, params, torch.tensor(state, dtype=torch.float64)


def test_quasi_steady_partial_pressures_sum_to_the_total():
    _adm1, params, x = _qss_setup()

    p_h2, p_ch4, p_co2, pTOTAL = gas_equilibrium_torch(x, params).tolist()

    assert p_h2 + p_ch4 + p_co2 == pytest.approx(pTOTAL, rel=1e-12)


def test_quasi_steady_total_pressure_is_pinned_to_the_outlet_balance():
    """The sum constraint drives pTOTAL to ``p_ext - p_h2o`` for a gas-bearing liquid."""
    _adm1, params, x = _qss_setup()

    pTOTAL = gas_equilibrium_torch(x, params)[3].item()

    assert pTOTAL == pytest.approx(params.p_ext - params.p_gas_h2o, rel=1e-10)


def test_quasi_steady_solution_ignores_the_stored_gas_phase():
    """Only the liquid slots are read -- the headspace state is slaved, not an input."""
    _adm1, params, x = _qss_setup()
    perturbed = x.clone()
    perturbed[37:41] = torch.tensor([9.9, 8.8, 7.7, 6.6], dtype=torch.float64)

    assert torch.allclose(gas_equilibrium_torch(x, params), gas_equilibrium_torch(perturbed, params))
    for a, b in zip(calc_gas_quasi_steady_torch(x, params), calc_gas_quasi_steady_torch(perturbed, params)):
        assert torch.allclose(a, b)


def test_quasi_steady_species_flows_follow_the_partial_pressures():
    _adm1, params, x = _qss_setup()

    _p_h2, p_ch4, p_co2, pTOTAL = gas_equilibrium_torch(x, params).tolist()
    q_gas, q_ch4, q_co2 = (t.item() for t in calc_gas_quasi_steady_torch(x, params))

    p_gas_wet = pTOTAL + params.p_gas_h2o
    assert q_ch4 == pytest.approx(q_gas * p_ch4 / p_gas_wet, rel=1e-12)
    assert q_co2 == pytest.approx(q_gas * p_co2 / p_gas_wet, rel=1e-12)


def test_quasi_steady_methane_flow_scales_with_dissolved_methane():
    _adm1, params, x = _qss_setup()

    flows = []
    for factor in (0.5, 1.0, 2.0):
        scaled = x.clone()
        scaled[8] = x[8] * factor  # S_ch4
        flows.append(calc_gas_quasi_steady_torch(scaled, params)[1].item())

    assert 0.0 < flows[0] < flows[1] < flows[2]
    # Transfer is linear in the dissolved concentration at fixed pTOTAL.
    assert flows[2] == pytest.approx(2.0 * flows[1], rel=1e-3)
    assert flows[0] == pytest.approx(0.5 * flows[1], rel=1e-3)


def test_quasi_steady_hydrogen_stays_a_trace_component():
    _adm1, params, x = _qss_setup()

    p_h2, p_ch4, p_co2, _pTOTAL = gas_equilibrium_torch(x, params).tolist()

    assert p_h2 < 1.0e-4 * min(p_ch4, p_co2)
    assert p_ch4 > 0.0 and p_co2 > 0.0


def test_quasi_steady_solution_is_converged_at_the_default_iteration_count():
    _adm1, params, x = _qss_setup()

    assert torch.allclose(
        gas_equilibrium_torch(x, params, n_iter=25),
        gas_equilibrium_torch(x, params, n_iter=80),
        rtol=1e-10,
    )


def test_quasi_steady_avoids_the_knife_edge_of_the_pressure_driven_form():
    """At the pinned pTOTAL the k_p form cancels to zero; the transfer form does not.

    This is the whole reason the quasi-steady solver exists: it reads the total
    flow from the gas transfer instead of from ``k_p * (pTOTAL + p_h2o - p_ext)``.
    """
    _adm1, params, x = _qss_setup()
    equilibrated = x.clone()
    equilibrated[37:41] = gas_equilibrium_torch(x, params)

    q_pressure_driven = calc_gas_torch(equilibrated, params)[0].item()
    q_transfer_driven = calc_gas_quasi_steady_torch(equilibrated, params)[0].item()

    assert q_pressure_driven == 0.0  # catastrophic cancellation
    assert q_transfer_driven > 0.0


def test_quasi_steady_depleted_liquid_gives_a_leaky_no_flow():
    """No dissolved gas -> a tiny negative flow, keeping the gradient alive."""
    _adm1, params, x = _qss_setup()
    depleted = x.clone()
    depleted[7] = depleted[8] = depleted[9] = 1.0e-12  # S_h2, S_ch4, S_co2
    depleted[35] = 0.0  # S_hco3

    q_gas, q_ch4, q_co2 = (t.item() for t in calc_gas_quasi_steady_torch(depleted, params))

    assert np.isfinite([q_gas, q_ch4, q_co2]).all()
    assert q_gas <= 0.0
    assert abs(q_gas) < 1.0e-3  # negligible in magnitude, not a real backflow


def test_quasi_steady_batches_row_by_row():
    _adm1, params, _x = _qss_setup()
    rng = np.random.default_rng(41)
    batch = np.stack([_random_state(rng) for _ in range(6)], axis=0)
    xb = torch.tensor(batch, dtype=torch.float64)

    q_gas, q_ch4, q_co2 = calc_gas_quasi_steady_torch(xb, params)
    pressures = gas_equilibrium_torch(xb, params)

    assert q_gas.shape == (6,)
    assert pressures.shape == (6, 4)
    for i in range(6):
        xi = torch.tensor(batch[i], dtype=torch.float64)
        gi, ci, oi = calc_gas_quasi_steady_torch(xi, params)
        assert q_gas[i].item() == pytest.approx(gi.item(), rel=1e-12)
        assert q_ch4[i].item() == pytest.approx(ci.item(), rel=1e-12)
        assert q_co2[i].item() == pytest.approx(oi.item(), rel=1e-12)


def test_quasi_steady_gradients_flow_back_to_the_liquid_state():
    _adm1, params, x0 = _qss_setup()
    x = x0.clone().requires_grad_(True)

    calc_gas_quasi_steady_torch(x, params)[1].backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert x.grad[8] != 0.0  # dissolved CH4 drives the methane flow
    # The headspace slots are not inputs to the solve, so they carry no gradient.
    assert torch.all(x.grad[37:41] == 0.0)


def test_quasi_steady_gradients_stay_finite_at_a_fully_zero_state():
    """The Newton derivative is floored so a flat spot cannot poison the backward pass."""
    _adm1, params, _x = _qss_setup()
    x = torch.zeros(STATE_SIZE, dtype=torch.float64, requires_grad=True)

    q_gas = calc_gas_quasi_steady_torch(x, params)[0]
    q_gas.backward()

    assert np.isfinite(q_gas.item())
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_quasi_steady_helpers_share_a_single_solve():
    """``gas_equilibrium_torch`` and ``calc_gas_quasi_steady_torch`` must stay consistent."""
    _adm1, params, x = _qss_setup()

    _p_h2, _p_ch4, _p_co2, pTOTAL = gas_equilibrium_torch(x, params).tolist()
    q_gas = calc_gas_quasi_steady_torch(x, params)[0].item()

    # Q_gas = Rho_T_11 * V_gas / (RT/1000 * NQ) with Rho_T_11 = B * pTOTAL; the
    # ratio therefore recovers B, which must be strictly positive when gas flows.
    B = q_gas * (params.RT / 1000.0 * params.NQ) / (params.V_gas * pTOTAL)
    assert B > 0.0


def _indicator_reference(feedstock, state):
    """pH / VFA / TAC as the Digester component actually computes them."""
    from pyadm1.components.biological import Digester

    d = Digester("d", feedstock, V_liq=1200.0, V_gas=216.0, T_ad=315.15)
    d.adm1_state = list(state)
    return d._compute_indicators()


def test_ph_vfa_tac_ts_torch_parity():
    """pH / VFA / TAC / TS torch maps must match the Digester indicator formulas."""
    feedstock = Feedstock(["maize_silage_milk_ripeness", "swine_manure"], feeding_freq=24, total_simtime=10)
    adm1 = ADM1(feedstock=None, V_liq=1200.0, V_gas=216.0, T_ad=315.15)
    params = Adm1TorchParams.from_adm1(adm1)
    rng = np.random.default_rng(21)
    for _ in range(20):
        state = _random_state(rng)
        ref = _indicator_reference(feedstock, state)
        x = torch.tensor(state, dtype=torch.float64)
        # pH: closed-form [H+] vs. the numpy Newton iteration differ only at the
        # ~1e-5 pH level (the closed form is in fact the more accurate root),
        # far below sensor relevance (~0.05 pH).
        assert ph_torch(x, params).item() == pytest.approx(ref["pH"], rel=1e-6, abs=1e-4)
        assert vfa_torch(x).item() == pytest.approx(ref["VFA"], rel=1e-9, abs=1e-9)
        assert tac_torch(x, params).item() == pytest.approx(ref["TAC"], rel=1e-7, abs=1e-9)
        assert ts_torch(x).item() == pytest.approx(ref["TS"], rel=1e-12, abs=1e-12)


def test_measurement_map_batched_and_differentiable():
    """h(x) must batch and be differentiable w.r.t. the state."""
    adm1 = _make_adm1()
    params = Adm1TorchParams.from_adm1(adm1)
    rng = np.random.default_rng(22)
    batch = np.stack([_random_state(rng) for _ in range(6)], axis=0)
    x = torch.tensor(batch, dtype=torch.float64, requires_grad=True)

    q_gas, _q_ch4, _q_co2 = calc_gas_torch(x, params)
    y = q_gas.sum() + ph_torch(x, params).sum() + vfa_torch(x).sum() + tac_torch(x, params).sum()
    y.backward()

    assert q_gas.shape == (6,)
    assert x.grad is not None and torch.isfinite(x.grad).all()
