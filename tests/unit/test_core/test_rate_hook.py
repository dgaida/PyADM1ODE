"""The rate-family hook: one multiplicative factor per rate family, both backends."""

from __future__ import annotations

import numpy as np
import pytest

from pyadm1.core.adm1 import RATE_FAMILIES

torch = pytest.importorskip("torch")
import importlib.util  # noqa: E402
from pathlib import Path  # noqa: E402

from pyadm1.core.adm1_torch import Adm1TorchParams, adm1da_rhs_torch, make_scipy_rhs  # noqa: E402

# Reuse the parity test's bare model and base state without making the test
# directory a package.
_spec = importlib.util.spec_from_file_location("_adm1_torch_helpers", Path(__file__).with_name("test_adm1_torch.py"))
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
_BASE_STATE, _make_adm1 = _helpers._BASE_STATE, _helpers._make_adm1

_IDX_S_AC, _IDX_X_AC = 6, 27


def _state() -> np.ndarray:
    return np.array(_BASE_STATE, dtype=np.float64)


def test_rate_families_has_thirteen_entries():
    assert len(RATE_FAMILIES) == 13
    assert RATE_FAMILIES[0] == "dis_PS" and RATE_FAMILIES[-1] == "dec"


def test_unit_factors_leave_the_numpy_rhs_unchanged():
    adm1 = _make_adm1()
    ref = np.asarray(adm1.ADM_ODE(0.0, list(_state())))
    adm1.rate_hook = lambda state, S_H, q_ad: np.ones(13)
    got = np.asarray(adm1.ADM_ODE(0.0, list(_state())))
    np.testing.assert_allclose(got, ref, rtol=0, atol=0)


def test_hook_receives_proton_concentration_and_feed():
    adm1 = _make_adm1()
    seen = {}

    def hook(state, S_H, q_ad):
        seen["S_H"], seen["q_ad"], seen["n"] = S_H, q_ad, len(state)
        return np.ones(13)

    adm1.rate_hook = hook
    adm1.ADM_ODE(0.0, list(_state()))
    assert seen["n"] == 41
    assert 0.0 < seen["S_H"] < 1.0  # a proton concentration [kmol/m3], not a pH
    assert seen["q_ad"] == 0.0  # no feed configured on the bare model


def test_acetate_factor_scales_only_the_acetate_uptake():
    """Doubling ``ac`` must change d S_ac/dt by exactly one extra Rho_ac."""
    adm1 = _make_adm1()
    x = _state()
    ref = np.asarray(adm1.ADM_ODE(0.0, list(x)))
    f = np.ones(13)
    f[RATE_FAMILIES.index("ac")] = 2.0
    adm1.rate_hook = lambda state, S_H, q_ad: f
    got = np.asarray(adm1.ADM_ODE(0.0, list(x)))
    d = got - ref
    # S_ac is consumed by Rho_ac, X_ac grows with Y_ac * Rho_ac: both move,
    # the substrate rows that do not involve acetate uptake do not.
    assert d[_IDX_S_AC] < 0.0 and d[_IDX_X_AC] > 0.0
    for i in (0, 1, 2, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21):
        assert d[i] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("seed", [0, 1])
def test_torch_factors_match_the_numpy_hook(seed):
    rng = np.random.default_rng(seed)
    factors = np.exp(rng.normal(0.0, 0.3, size=13))
    adm1 = _make_adm1()
    x = _state()
    adm1.rate_hook = lambda state, S_H, q_ad: factors
    ref = np.asarray(adm1.ADM_ODE(0.0, list(x)))
    params = Adm1TorchParams.from_adm1(adm1)
    got = adm1da_rhs_torch(torch.tensor(x), params, torch.tensor(factors)).numpy()
    np.testing.assert_allclose(got, ref, rtol=1e-6, atol=1e-9)
    # the scipy adapter of the torch backend applies the same hook
    fun = make_scipy_rhs(adm1)
    np.testing.assert_allclose(fun(0.0, x), ref, rtol=1e-6, atol=1e-9)


def test_torch_factors_broadcast_over_a_batch():
    adm1 = _make_adm1()
    params = Adm1TorchParams.from_adm1(adm1)
    x = torch.tensor(np.stack([_state(), _state() * 1.1]))
    rf = torch.ones(2, 13, dtype=torch.float64)
    rf[1, RATE_FAMILIES.index("hyd_ch")] = 0.5
    out = adm1da_rhs_torch(x, params, rf)
    ref = adm1da_rhs_torch(x, params)
    assert out.shape == (2, 41)
    torch.testing.assert_close(out[0], ref[0])
    assert not torch.allclose(out[1], ref[1])


def test_torch_factors_carry_gradients():
    adm1 = _make_adm1()
    params = Adm1TorchParams.from_adm1(adm1)
    rf = torch.ones(13, dtype=torch.float64, requires_grad=True)
    out = adm1da_rhs_torch(torch.tensor(_state()), params, rf)
    out[_IDX_S_AC].backward()
    assert rf.grad is not None
    assert rf.grad[RATE_FAMILIES.index("ac")] < 0.0  # more uptake, less acetate
