"""
Acid capacity (TAC) and the charge bookkeeping the ADM1da formula relies on.

The TAC formula of ADM1da (Schlattmann 2011) subtracts the strong-ion
difference, because the substrate characterisation fixes ``S_cation = 0`` and
lets ``S_anion`` absorb the charge balance. A pre-inoculated state that carries
its buffer as strong ions instead of ammonium therefore reported far too little
acid capacity — negative in long-retention stores, which turned FOS/TAC into a
division by a value drifting through zero.
"""

from __future__ import annotations

import math

import pytest

from pyadm1 import Feedstock
from pyadm1.components.biological.digester import _ADM1DA_INOCULUM_S_NH4, Digester

_IDX_S_NH4, _IDX_S_CATION, _IDX_S_ANION = 10, 29, 30

#: Strong-ion difference of the ADM1da / SIMBA# reference state
#: (S_cation 0.038553 - S_anion 0.021132).
_REFERENCE_ION_DIFFERENCE = 0.017420


@pytest.fixture
def feedstock():
    return Feedstock(["maize_silage_milk_ripeness", "cattle_manure"], feeding_freq=24, total_simtime=5)


def _digester(feedstock, **kwargs):
    return Digester(component_id="d", feedstock=feedstock, V_liq=2000.0, V_gas=400.0, **kwargs)


class TestInoculumChargeBookkeeping:
    def test_cation_follows_the_adm1da_convention(self, feedstock):
        """``S_cation`` is fixed at zero, exactly as the influent characterisation does."""
        state = _digester(feedstock)._build_pre_inoculated_state([20.0, 10.0])
        assert state[_IDX_S_CATION] == pytest.approx(0.0, abs=1e-12)

    def test_tank_without_substrate_still_carries_ammonium(self, feedstock):
        """A post-digester or store is seeded with the reference inoculum, not with zero N."""
        state = _digester(feedstock)._build_pre_inoculated_state([0.0, 0.0])
        assert state[_IDX_S_NH4] == pytest.approx(_ADM1DA_INOCULUM_S_NH4, rel=1e-9)

    def test_tank_without_substrate_matches_the_reference_ion_difference(self, feedstock):
        """Its buffer is carried by ammonium, so the strong-ion difference stays small."""
        state = _digester(feedstock)._build_pre_inoculated_state([0.0, 0.0])
        difference = state[_IDX_S_CATION] - state[_IDX_S_ANION]
        assert difference == pytest.approx(_REFERENCE_ION_DIFFERENCE, abs=5e-4)

    def test_fed_tank_keeps_its_substrate_derived_ammonium(self, feedstock):
        """The fallback must not override a blend that has ammonium of its own.

        Maize silage alone carries little N, so the seeded value has to stay
        below the reference inoculum. (With maize + cattle slurry at 20:10 the
        two coincide — that is the very blend the ADM1da reference state was
        generated from, 1.5x its blended ammonium.)
        """
        state = _digester(feedstock)._build_pre_inoculated_state([20.0, 0.0])
        assert 0.0 < state[_IDX_S_NH4] < _ADM1DA_INOCULUM_S_NH4


class TestTacIndicator:
    def test_positive_for_a_freshly_inoculated_tank(self, feedstock):
        for feed in ([20.0, 10.0], [0.0, 0.0]):
            d = _digester(feedstock)
            d.initialize({"Q_substrates": feed})
            tac = d._compute_indicators()["TAC"]
            assert tac > 0.0, f"TAC must be positive for feed {feed}, got {tac}"

    def test_plausible_magnitude(self, feedstock):
        """Agricultural digestate titrates to roughly 5-20 kg CaCO3/m3."""
        d = _digester(feedstock)
        d.initialize({"Q_substrates": [0.0, 0.0]})
        assert 3.0 < d._compute_indicators()["TAC"] < 20.0

    def test_guard_reports_nan_and_warns_once(self, feedstock, monkeypatch):
        """A state whose strong ions exceed the buffer yields NaN, not a huge FOS/TAC."""
        monkeypatch.setattr(Digester, "_tac_warned", False, raising=False)
        d = _digester(feedstock)
        d.initialize({"Q_substrates": [0.0, 0.0]})
        d.adm1_state[_IDX_S_CATION] = 0.5  # far beyond the carbonate buffer

        with pytest.warns(RuntimeWarning, match="TAC <= 0"):
            first = d._compute_indicators()["TAC"]
        assert math.isnan(first)

        # second call stays silent but keeps reporting NaN
        with _no_warning():
            second = d._compute_indicators()["TAC"]
        assert math.isnan(second)


class _no_warning:
    """Context manager asserting that no RuntimeWarning is raised."""

    def __enter__(self):
        import warnings

        self._ctx = warnings.catch_warnings(record=True)
        self._records = self._ctx.__enter__()
        import warnings as _w

        _w.simplefilter("always")
        return self

    def __exit__(self, *exc):
        runtime = [r for r in self._records if issubclass(r.category, RuntimeWarning)]
        self._ctx.__exit__(*exc)
        assert not runtime, f"expected no RuntimeWarning, got {[str(r.message) for r in runtime]}"
        return False
