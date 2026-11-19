"""The influent composition follows the substrate mix of every step.

The digester used to cache the blended influent when it was initialised.
Feed rates passed later changed only the flow: a digester switched from a
maize/manure mix to manure alone kept receiving the composition of the mix and
produced *more* methane than one switched to maize alone. These tests pin the
fixed behaviour.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyadm1 import Feedstock
from pyadm1.components.biological import Digester
from pyadm1.core.adm1 import ADM1, INFLUENT_COLUMNS

KEYS = INFLUENT_COLUMNS[:-1]

MIX = [15.0, 10.0]
MAIZE_ONLY = [25.0, 0.0]
MANURE_ONLY = [0.0, 25.0]


@pytest.fixture
def feedstock() -> Feedstock:
    return Feedstock(["maize_silage_milk_ripeness", "swine_manure"], feeding_freq=24, total_simtime=10)


def _digester(feedstock: Feedstock, Q: list[float]) -> Digester:
    digester = Digester("d", feedstock, V_liq=2000.0, V_gas=300.0)
    digester.initialize({"Q_substrates": Q})
    return digester


def _influent(digester: Digester) -> dict[str, float]:
    return dict(zip(KEYS, digester.adm1._state_input))


class TestCompositionFollowsFeed:
    def test_step_input_changes_the_composition(self, feedstock: Feedstock) -> None:
        digester = _digester(feedstock, MIX)

        digester.step(0.0, 1.0, {"Q_substrates": MANURE_ONLY})

        assert _influent(digester) == pytest.approx(feedstock.blended_concentrations(MANURE_ONLY))
        assert _influent(digester)["X_PF_ch"] != pytest.approx(feedstock.blended_concentrations(MIX)["X_PF_ch"])

    def test_attribute_assignment_changes_the_composition(self, feedstock: Feedstock) -> None:
        digester = _digester(feedstock, MIX)

        digester.Q_substrates = MAIZE_ONLY
        digester.step(0.0, 1.0, {})

        assert _influent(digester) == pytest.approx(feedstock.blended_concentrations(MAIZE_ONLY))

    def test_flow_and_density_follow_the_feed(self, feedstock: Feedstock) -> None:
        digester = _digester(feedstock, MIX)

        digester.step(0.0, 1.0, {"Q_substrates": MAIZE_ONLY})

        assert sum(digester.adm1._Q) == pytest.approx(sum(feedstock.actual_Q(MAIZE_ONLY)))
        assert digester.adm1._rho_in == pytest.approx(feedstock.blended_density(MAIZE_ONLY))

    def test_unchanged_mix_gives_the_initial_blend(self, feedstock: Feedstock) -> None:
        digester = _digester(feedstock, MIX)

        digester.step(0.0, 1.0, {})

        assert _influent(digester) == pytest.approx(feedstock.blended_concentrations(MIX))

    def test_switched_digester_equals_one_started_on_the_new_mix(self, feedstock: Feedstock) -> None:
        """Same state and same feed must give the same trajectory, whatever the mix at start-up."""
        switched = _digester(feedstock, MIX)
        reference = _digester(feedstock, MAIZE_ONLY)
        reference.adm1_state = list(switched.adm1_state)

        for day in range(3):
            out_switched = switched.step(float(day), 1.0, {"Q_substrates": MAIZE_ONLY})
            out_reference = reference.step(float(day), 1.0, {})

        assert out_switched["Q_ch4"] == pytest.approx(out_reference["Q_ch4"], rel=1e-9)
        assert switched.adm1_state == pytest.approx(reference.adm1_state, rel=1e-9)

    def test_energy_rich_substrate_gives_more_methane(self, feedstock: Feedstock) -> None:
        """The symptom of the bug: manure alone out-produced maize alone."""
        maize = _digester(feedstock, MIX)
        manure = _digester(feedstock, MIX)

        for day in range(20):
            ch4_maize = maize.step(float(day), 1.0, {"Q_substrates": MAIZE_ONLY})["Q_ch4"]
            ch4_manure = manure.step(float(day), 1.0, {"Q_substrates": MANURE_ONLY})["Q_ch4"]

        assert ch4_maize > 3.0 * ch4_manure

    def test_zero_feed_gives_an_empty_influent(self, feedstock: Feedstock) -> None:
        digester = _digester(feedstock, MIX)

        digester.step(0.0, 1.0, {"Q_substrates": [0.0, 0.0]})

        assert sum(digester.adm1._Q) == 0.0
        assert all(value == 0.0 for value in digester.adm1._state_input)


class TestExternalInfluentFrame:
    """Only an explicitly supplied DataFrame overrides the feedstock blend."""

    def _frame(self, feedstock: Feedstock, Q: list[float]) -> pd.DataFrame:
        return feedstock.get_influent_dataframe(Q)

    def test_initialize_does_not_cache_a_frame(self, feedstock: Feedstock) -> None:
        assert _digester(feedstock, MIX).adm1._influent_df is None

    def test_external_frame_takes_precedence(self, feedstock: Feedstock) -> None:
        digester = _digester(feedstock, MIX)
        digester.adm1.set_influent_dataframe(self._frame(feedstock, MIX))

        digester.step(0.0, 1.0, {"Q_substrates": MANURE_ONLY})

        assert _influent(digester) == pytest.approx(feedstock.blended_concentrations(MIX))
        assert sum(digester.adm1._Q) == pytest.approx(sum(feedstock.actual_Q(MANURE_ONLY)))

    def test_clearing_the_frame_restores_the_blend(self, feedstock: Feedstock) -> None:
        digester = _digester(feedstock, MIX)
        digester.adm1.set_influent_dataframe(self._frame(feedstock, MIX))
        digester.adm1.clear_influent_dataframe()

        digester.step(0.0, 1.0, {"Q_substrates": MANURE_ONLY})

        assert _influent(digester) == pytest.approx(feedstock.blended_concentrations(MANURE_ONLY))

    def test_set_feedstock_drops_an_external_frame(self, feedstock: Feedstock) -> None:
        digester = _digester(feedstock, MIX)
        digester.adm1.set_influent_dataframe(self._frame(feedstock, MIX))

        digester.set_feedstock(feedstock, Q_substrates=MAIZE_ONLY, rebuild_state=False)
        digester.step(0.0, 1.0, {})

        assert digester.adm1._influent_df is None
        assert _influent(digester) == pytest.approx(feedstock.blended_concentrations(MAIZE_ONLY))

    def test_frame_without_feedstock_still_works(self, feedstock: Feedstock) -> None:
        model = ADM1(None, V_liq=2000.0, V_gas=300.0)
        model.set_influent_dataframe(self._frame(feedstock, MIX))

        model.create_influent([20.0], 0)

        assert model._Q == [20.0]
        assert dict(zip(KEYS, model._state_input)) == pytest.approx(feedstock.blended_concentrations(MIX))


class TestCoreModel:
    def test_create_influent_blends_from_the_feedstock(self, feedstock: Feedstock) -> None:
        model = ADM1(feedstock, V_liq=2000.0, V_gas=300.0)

        model.create_influent(MIX, 0)
        mix = list(model._state_input)
        model.create_influent(MANURE_ONLY, 0)

        assert mix == pytest.approx([feedstock.blended_concentrations(MIX)[k] for k in KEYS])
        assert model._state_input == pytest.approx([feedstock.blended_concentrations(MANURE_ONLY)[k] for k in KEYS])
        assert not np.allclose(mix, model._state_input)

    def test_feedstock_that_only_offers_a_frame_is_still_supported(self, feedstock: Feedstock) -> None:
        """Duck-typed feedstocks without ``blended_concentrations`` fall back to their DataFrame."""

        class FrameOnly:
            def get_influent_dataframe(self, Q: list[float]) -> pd.DataFrame:
                return feedstock.get_influent_dataframe(Q)

        model = ADM1(FrameOnly(), V_liq=2000.0, V_gas=300.0)

        model.create_influent(MANURE_ONLY, 0)

        assert model._Q == MANURE_ONLY
        assert model._state_input == pytest.approx([feedstock.blended_concentrations(MANURE_ONLY)[k] for k in KEYS])

    def test_explicit_densities_are_used(self, feedstock: Feedstock) -> None:
        model = ADM1(feedstock, V_liq=2000.0, V_gas=300.0)

        model.create_influent(MIX, 0, rho=[1200.0, 1000.0])

        q_maize, q_manure = feedstock.actual_Q(MIX)
        assert model._rho_in == pytest.approx((q_maize * 1200.0 + q_manure * 1000.0) / (q_maize + q_manure))
