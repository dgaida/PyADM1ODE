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
from pyadm1.core.adm1 import ADM1, STATE_SIZE
from pyadm1.core.adm1_torch import (
    Adm1TorchParams,
    adm1da_rhs_torch,
    calc_gas_torch,
    ph_torch,
    tac_torch,
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


def _indicator_reference(feedstock, state):
    """pH / VFA / TAC as the Digester component actually computes them."""
    from pyadm1.components.biological import Digester

    d = Digester("d", feedstock, V_liq=1200.0, V_gas=216.0, T_ad=315.15)
    d.adm1_state = list(state)
    return d._compute_indicators()


def test_ph_vfa_tac_torch_parity():
    """pH / VFA / TAC torch maps must match the Digester indicator formulas."""
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


def test_measurement_map_batched_and_differentiable():
    """h(x) must batch and be differentiable w.r.t. the state."""
    adm1 = _make_adm1()
    params = Adm1TorchParams.from_adm1(adm1)
    rng = np.random.default_rng(22)
    batch = np.stack([_random_state(rng) for _ in range(6)], axis=0)
    x = torch.tensor(batch, dtype=torch.float64, requires_grad=True)

    q_gas, q_ch4, q_co2 = calc_gas_torch(x, params)
    y = q_gas.sum() + ph_torch(x, params).sum() + vfa_torch(x).sum() + tac_torch(x, params).sum()
    y.backward()

    assert q_gas.shape == (6,)
    assert x.grad is not None and torch.isfinite(x.grad).all()
