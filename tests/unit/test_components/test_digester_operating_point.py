"""Operating point of a digester: temperature and feedstock after construction.

Both used to be write-once. ``T_ad`` looked writable but was not: assigning it
changed the reported value while the simulation kept running on the constructor
temperature. ``feedstock`` was required up front and could not be supplied later.
These tests pin the fixed behaviour.
"""

from __future__ import annotations

import pytest

from pyadm1 import Feedstock
from pyadm1.components.biological import Digester
from pyadm1.components.energy.heating import HeatingSystem
from pyadm1.configurator.plant_builder import BiogasPlant
from pyadm1.configurator.plant_configurator import PlantConfigurator
from pyadm1.core.adm1 import ADM1

# ADM1 attributes whose VALUE depends on T_ad -- these must move on a change.
TEMPERATURE_DEPENDENT = ("_T_ad", "_RT", "_p_ext", "_p_gas_h2o", "_K_H_co2", "_K_H_ch4", "_K_H_h2")
# Recomputed alongside them, but constant in ADMParams.getADMgasparams.
TEMPERATURE_INDEPENDENT = ("_k_p", "_k_L_a")
DERIVED = TEMPERATURE_DEPENDENT + TEMPERATURE_INDEPENDENT

MESO, THERMO = 313.15, 328.15


@pytest.fixture
def feedstock() -> Feedstock:
    return Feedstock(["maize_silage_milk_ripeness", "swine_manure"], feeding_freq=24, total_simtime=10)


@pytest.fixture
def Q() -> list[float]:
    return [10.0, 5.0] + [0.0] * 8


class TestTemperatureChange:
    """A: ADM1 recomputes everything derived from T_ad; B: Digester stays in sync."""

    def test_reported_temperature_matches_simulated(self, feedstock: Feedstock, Q: list[float]) -> None:
        d = Digester("d", feedstock, V_liq=1000.0, V_gas=150.0, T_ad=MESO)
        d.initialize({"Q_substrates": Q})

        d.T_ad = THERMO

        assert d.T_ad == THERMO
        assert d.adm1.T_ad == THERMO
        assert d.to_dict()["T_ad"] == THERMO

    def test_all_derived_constants_move(self, feedstock: Feedstock) -> None:
        d = Digester("d", feedstock, V_liq=1000.0, V_gas=150.0, T_ad=MESO)
        before = {k: getattr(d.adm1, k) for k in DERIVED}
        before_kinetic = dict(d.adm1._kinetic)
        before_inhib = dict(d.adm1._inhib_params)

        d.T_ad = THERMO

        for key in TEMPERATURE_DEPENDENT:
            assert getattr(d.adm1, key) != before[key], f"{key} did not follow the temperature change"
        for key in TEMPERATURE_INDEPENDENT:
            assert getattr(d.adm1, key) == before[key], f"{key} is not temperature-dependent and must not move"
        assert d.adm1._kinetic != before_kinetic
        assert d.adm1._inhib_params != before_inhib

    def test_setting_equals_constructing_at_that_temperature(self, feedstock: Feedstock) -> None:
        changed = ADM1(feedstock=feedstock, V_liq=1000.0, V_gas=150.0, T_ad=MESO)
        changed.T_ad = THERMO
        fresh = ADM1(feedstock=feedstock, V_liq=1000.0, V_gas=150.0, T_ad=THERMO)

        for key in DERIVED:
            assert getattr(changed, key) == pytest.approx(getattr(fresh, key))
        assert changed._kinetic == pytest.approx(fresh._kinetic)
        assert changed._inhib_params == pytest.approx(fresh._inhib_params)

    def test_calibration_survives_and_rests_on_the_new_baseline(self, feedstock: Feedstock) -> None:
        d = Digester("d", feedstock, V_liq=1000.0, V_gas=150.0, T_ad=MESO)
        d.adm1.set_calibration_parameters({"k_m_ac": 7.5, "K_H_ch4": 30.0, "k_L_a": 210.0})

        d.T_ad = THERMO

        assert d.adm1._kinetic["k_m_ac"] == 7.5
        assert d.adm1._K_H_ch4 == 30.0
        assert d.adm1.get_calibration_parameters()["k_L_a"] == 210.0

        # Clearing must fall back to the defaults at the NEW temperature.
        d.adm1.clear_calibration_parameters()
        fresh = ADM1(feedstock=feedstock, V_liq=1000.0, V_gas=150.0, T_ad=THERMO)
        assert d.adm1._kinetic["k_m_ac"] == pytest.approx(fresh._kinetic["k_m_ac"])
        assert d.adm1._K_H_ch4 == pytest.approx(fresh._K_H_ch4)

    def test_rebuild_state_matches_direct_construction(self, feedstock: Feedstock, Q: list[float]) -> None:
        changed = Digester("c", feedstock, V_liq=1000.0, V_gas=150.0, T_ad=MESO)
        changed.initialize({"Q_substrates": Q})
        changed.set_temperature(THERMO, rebuild_state=True)

        fresh = Digester("f", feedstock, V_liq=1000.0, V_gas=150.0, T_ad=THERMO)
        fresh.initialize({"Q_substrates": Q})

        assert changed.adm1_state == pytest.approx(fresh.adm1_state)

    def test_state_is_kept_without_rebuild(self, feedstock: Feedstock, Q: list[float]) -> None:
        d = Digester("d", feedstock, V_liq=1000.0, V_gas=150.0, T_ad=MESO)
        d.initialize({"Q_substrates": Q})
        before = list(d.adm1_state)

        d.set_temperature(THERMO)

        assert d.adm1_state == before

    def test_rebuild_state_without_feedstock_is_refused(self, Q: list[float]) -> None:
        d = Digester("d", None, V_liq=1000.0, V_gas=150.0, T_ad=MESO)
        d.initialize({"Q_substrates": Q})

        with pytest.raises(RuntimeError, match="needs a feedstock"):
            d.set_temperature(THERMO, rebuild_state=True)


class TestLateFeedstock:
    """C: build the structure first, supply the substrates later."""

    def test_configurator_works_without_feedstock(self) -> None:
        cfg = PlantConfigurator(BiogasPlant("P"))
        digester, info = cfg.add_digester("F1", V_liq=1000.0, V_gas=150.0)

        assert digester.feedstock is None
        assert "set_feedstock" in info

    def test_simulating_without_feedstock_raises_a_clear_error(self, Q: list[float]) -> None:
        d = Digester("d", None, V_liq=1000.0, V_gas=150.0)
        d.initialize({"Q_substrates": Q})

        with pytest.raises(RuntimeError, match="no influent source"):
            d.step(0.0, 1.0, {})

    def test_late_feedstock_equals_from_the_start(self, feedstock: Feedstock, Q: list[float]) -> None:
        early = Digester("a", feedstock, V_liq=1000.0, V_gas=150.0, T_ad=MESO)
        early.initialize({"Q_substrates": Q})

        late = Digester("b", None, V_liq=1000.0, V_gas=150.0, T_ad=MESO)
        late.initialize({"Q_substrates": Q})
        late.set_feedstock(feedstock, Q_substrates=Q)

        assert late.adm1_state == pytest.approx(early.adm1_state)
        assert late.adm1._influent_df.equals(early.adm1._influent_df)
        assert late.adm1._rho_in == pytest.approx(early.adm1._rho_in)
        assert late.step(0.0, 1.0, {})["Q_gas"] == pytest.approx(early.step(0.0, 1.0, {})["Q_gas"])

    def test_plain_assignment_still_reaches_the_solver(self, feedstock: Feedstock) -> None:
        """``feedstock`` is a property now, so it can no longer desynchronise."""
        d = Digester("d", None, V_liq=1000.0, V_gas=150.0)
        d.feedstock = feedstock

        assert d.adm1.feedstock is feedstock

    def test_configurator_set_feedstock_updates_all_digesters(self, feedstock: Feedstock, Q: list[float]) -> None:
        cfg = PlantConfigurator(BiogasPlant("P"))
        cfg.add_digester("F1", V_liq=1000.0, V_gas=150.0)
        cfg.add_digester("F2", V_liq=1000.0, V_gas=150.0)

        updated = cfg.set_feedstock(feedstock, Q_substrates={"F1": Q})

        assert sorted(updated) == ["F1", "F2"]
        assert cfg.feedstock is feedstock
        assert cfg.plant.components["F1"].Q_substrates == Q
        for cid in ("F1", "F2"):
            assert cfg.plant.components[cid].adm1._influent_df is not None

    def test_configurator_set_feedstock_can_be_restricted(self, feedstock: Feedstock) -> None:
        cfg = PlantConfigurator(BiogasPlant("P"))
        cfg.add_digester("F1", V_liq=1000.0, V_gas=150.0)
        cfg.add_digester("F2", V_liq=1000.0, V_gas=150.0)

        updated = cfg.set_feedstock(feedstock, digester_ids=["F1"])

        assert updated == ["F1"]
        assert cfg.plant.components["F2"].feedstock is None

    def test_per_digester_feedstock_overrides_the_configurator(self, feedstock: Feedstock) -> None:
        other = Feedstock(["cattle_manure"], feeding_freq=24, total_simtime=10)
        cfg = PlantConfigurator(BiogasPlant("P"), feedstock)

        digester, _ = cfg.add_digester("F1", V_liq=1000.0, V_gas=150.0, feedstock=other)

        assert digester.feedstock is other


class TestDefaultTemperature:
    """D: one default temperature across every entry point."""

    DEFAULT = 315.15

    def test_all_digester_entry_points_agree(self, feedstock: Feedstock) -> None:
        cfg = PlantConfigurator(BiogasPlant("P"), feedstock)

        assert ADM1(feedstock=feedstock).T_ad == self.DEFAULT
        assert Digester("d", feedstock).T_ad == self.DEFAULT
        assert cfg.add_digester("F1")[0].T_ad == self.DEFAULT
        assert Digester.from_dict({"component_id": "x"}, feedstock=feedstock).T_ad == self.DEFAULT

    def test_heating_setpoint_matches_the_digester_default(self, feedstock: Feedstock) -> None:
        """A heating system left at its default must not fight the digester."""
        cfg = PlantConfigurator(BiogasPlant("P"), feedstock)

        assert HeatingSystem("h").target_temperature == self.DEFAULT
        assert cfg.add_heating("heat1").target_temperature == self.DEFAULT
        assert HeatingSystem.from_dict({"component_id": "h2"}).target_temperature == self.DEFAULT
        assert cfg.add_digester("F1")[0].T_ad == cfg.add_heating("heat2").target_temperature
