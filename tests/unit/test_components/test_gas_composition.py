"""Gas composition from the digester through the storage to the CHP, and T_ground.

The CHP used to assume 60 % methane for every gas, because the storage tracked
volume only and the digester's actual ``Q_ch4`` / ``Q_gas`` was lost on the
way. The storage now keeps the composition of its (ideally mixed) inventory and
the CHP burns what it is actually given.
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from pyadm1.components.biological import Digester
from pyadm1.components.energy.chp import CHP
from pyadm1.components.energy.gas_storage import GasStorage
from pyadm1.components.energy.heating import HeatingSystem
from pyadm1.configurator.connection_manager import Connection
from pyadm1.configurator.plant_builder import BiogasPlant


class TestGasStorageComposition:
    def test_initial_composition(self) -> None:
        gs = GasStorage("gs", initial_ch4_fraction=0.52, initial_co2_fraction=0.45)

        assert (gs.ch4_fraction, gs.co2_fraction) == (0.52, 0.45)
        assert gs.outputs_data["CH4_fraction"] == 0.52

    @pytest.mark.parametrize(("ch4", "co2"), [(-0.1, 0.4), (1.2, 0.0), (0.7, 0.5)])
    def test_invalid_composition_is_rejected(self, ch4: float, co2: float) -> None:
        with pytest.raises(ValueError, match="fractions"):
            GasStorage("gs", initial_ch4_fraction=ch4, initial_co2_fraction=co2)

    def test_inflow_is_blended_into_the_inventory(self) -> None:
        gs = GasStorage("gs", capacity_m3=1000.0, initial_fill_fraction=0.1)  # 100 m³ at 60 % CH4

        out = gs.step(0.0, 1.0, {"Q_gas_in_m3_per_day": 100.0, "Q_ch4_in_m3_per_day": 50.0, "Q_co2_in_m3_per_day": 48.0})

        assert out["CH4_fraction"] == pytest.approx((100.0 * 0.60 + 50.0) / 200.0)
        assert out["CO2_fraction"] == pytest.approx((100.0 * 0.40 + 48.0) / 200.0)

    def test_inflow_without_composition_leaves_it_unchanged(self) -> None:
        gs = GasStorage("gs", capacity_m3=1000.0, initial_ch4_fraction=0.52, initial_co2_fraction=0.45)

        out = gs.step(0.0, 1.0, {"Q_gas_in_m3_per_day": 100.0})

        assert out["CH4_fraction"] == pytest.approx(0.52)

    def test_supply_carries_the_stored_composition(self) -> None:
        gs = GasStorage("gs", capacity_m3=1000.0, initial_fill_fraction=0.5, initial_ch4_fraction=0.52)

        out = gs.step(0.0, 1.0, {"Q_gas_out_m3_per_day": 100.0})

        assert out["Q_gas_supplied_m3_per_day"] > 0.0
        assert out["Q_ch4_supplied_m3_per_day"] == pytest.approx(out["Q_gas_supplied_m3_per_day"] * 0.52)
        assert out["CH4_fraction"] == pytest.approx(0.52)  # withdrawing does not change the mix

    def test_long_run_converges_to_the_inflow_composition(self) -> None:
        gs = GasStorage("gs", capacity_m3=1000.0, initial_fill_fraction=0.5)
        inputs = {"Q_gas_in_m3_per_day": 500.0, "Q_ch4_in_m3_per_day": 260.0, "Q_gas_out_m3_per_day": 500.0}

        for _ in range(60):
            out = gs.step(0.0, 1.0, inputs)

        assert out["CH4_fraction"] == pytest.approx(0.52, abs=1e-6)

    def test_composition_round_trips(self) -> None:
        gs = GasStorage("gs", initial_ch4_fraction=0.52, initial_co2_fraction=0.45)

        restored = GasStorage.from_dict(gs.to_dict())

        assert (restored.ch4_fraction, restored.co2_fraction) == (0.52, 0.45)

    def test_config_without_composition_still_loads(self) -> None:
        config = GasStorage("gs").to_dict()
        del config["ch4_fraction"], config["co2_fraction"]

        assert GasStorage.from_dict(config).ch4_fraction == 0.60

    def test_digester_passes_its_composition_to_the_plant_storage(self) -> None:
        from pyadm1 import Feedstock

        fs = Feedstock(["maize_silage_milk_ripeness", "swine_manure"], feeding_freq=24, total_simtime=5)
        plant = BiogasPlant("composition")
        d = Digester("d", fs, V_liq=1000.0, V_gas=150.0)
        d.initialize({"Q_substrates": [10.0, 5.0] + [0.0] * 8})
        storage = GasStorage("s", capacity_m3=150.0)
        plant.add_component(d)
        plant.add_component(storage)
        plant.add_connection(Connection("d", "s", "gas"))
        before = storage.ch4_fraction

        for _ in range(24):
            out = plant.step(1.0 / 24.0)["d"]

        assert storage.ch4_fraction != before
        # Moves towards what the digester actually produces.
        produced = out["Q_ch4"] / out["Q_gas"]
        assert abs(storage.ch4_fraction - produced) < abs(before - produced)
        assert out["CH4_fraction"] == pytest.approx(produced)


class TestCHPMethaneContent:
    """Same biogas volume, different methane content -> different power."""

    @staticmethod
    def _power(inputs: dict) -> dict:
        return CHP("chp", P_el_nom=500.0, eta_el=0.40).step(0.0, 1.0 / 24.0, inputs)

    def test_methane_flow_from_the_storage_is_used(self) -> None:
        out = self._power({"Q_gas_supplied_m3_per_day": 1000.0, "Q_ch4_supplied_m3_per_day": 500.0})

        assert out["CH4_fraction"] == pytest.approx(0.50)
        assert out["P_el"] == pytest.approx(500.0 / 24.0 * 10.0 * 0.40)

    def test_fraction_input_is_used(self) -> None:
        rich = self._power({"Q_gas_supplied_m3_per_day": 1000.0, "CH4_fraction": 0.65})
        lean = self._power({"Q_gas_supplied_m3_per_day": 1000.0, "CH4_fraction": 0.50})

        assert rich["P_el"] == pytest.approx(lean["P_el"] * 0.65 / 0.50)

    def test_lean_gas_needs_more_volume_for_the_same_power(self) -> None:
        rich = self._power({"Q_gas_supplied_m3_per_day": 8000.0, "CH4_fraction": 0.65})
        lean = self._power({"Q_gas_supplied_m3_per_day": 8000.0, "CH4_fraction": 0.50})

        assert rich["P_el"] == lean["P_el"] == 500.0  # both reach nominal load
        assert lean["Q_gas_consumed"] == pytest.approx(rich["Q_gas_consumed"] * 0.65 / 0.50)

    def test_direct_digester_stream_gives_its_own_fraction(self) -> None:
        out = self._power({"Q_ch4": 3000.0, "Q_gas": 5500.0})

        assert out["CH4_fraction"] == pytest.approx(3000.0 / 5500.0)

    def test_default_applies_only_without_any_composition(self) -> None:
        out = self._power({"Q_gas_supplied_m3_per_day": 1000.0})

        assert out["CH4_fraction"] == 0.60

    def test_default_is_configurable_and_round_trips(self) -> None:
        chp = CHP("chp", ch4_content_default=0.55)

        assert CHP.from_dict(chp.to_dict()).ch4_content_default == 0.55
        with pytest.raises(ValueError, match="ch4_content_default"):
            CHP("chp2", ch4_content_default=0.0)


class TestPlantGasPath:
    @staticmethod
    def _plant(ch4_fraction: float) -> BiogasPlant:
        plant = BiogasPlant("p")
        plant.add_component(
            GasStorage(
                "gs",
                capacity_m3=50000.0,
                initial_fill_fraction=0.5,
                initial_ch4_fraction=ch4_fraction,
                initial_co2_fraction=1.0 - ch4_fraction,
            )
        )
        plant.add_component(CHP("chp", P_el_nom=500.0, eta_el=0.40))
        plant.add_connection(Connection("gs", "chp", "gas"))
        return plant

    def test_chp_burns_the_stored_methane_content(self) -> None:
        results = self._plant(0.50).step(1.0 / 24.0)

        assert results["chp"]["CH4_fraction"] == pytest.approx(0.50)

    def test_lean_storage_is_asked_for_more_gas(self) -> None:
        rich = self._plant(0.65).step(1.0 / 24.0)
        lean = self._plant(0.50).step(1.0 / 24.0)

        assert rich["chp"]["P_el"] == pytest.approx(500.0)
        assert lean["chp"]["P_el"] == pytest.approx(500.0)
        ratio = lean["gs"]["Q_gas_supplied_m3_per_day"] / rich["gs"]["Q_gas_supplied_m3_per_day"]
        assert ratio == pytest.approx(0.65 / 0.50)


class TestGroundTemperature:
    KW: ClassVar[dict] = dict(
        V_liq=3000.0, V_gas=175.0, diameter=20.0, roof_type="concrete", k_wall=0.4, k_roof=0.25, k_ground=1.9
    )

    def test_ground_share_of_the_coefficient(self) -> None:
        d = Digester("d", None, **self.KW)

        assert d.heat_loss_coefficient_ground == pytest.approx(1.9 * d.base_area / 1000.0)
        assert d.heat_loss_coefficient_ground < d.heat_loss_coefficient
        assert Digester("d2", None, V_liq=1000.0, V_gas=150.0).heat_loss_coefficient_ground is None

    def test_slab_loses_against_the_ground_temperature(self) -> None:
        heating = HeatingSystem("h", target_temperature=313.15)
        inputs = {
            "T_ambient": 263.15,
            "T_ground": 283.15,
            "heat_loss_coefficient": 0.9,
            "heat_loss_coefficient_ground": 0.6,
        }

        heating.step(0.0, 1.0 / 24.0, inputs)

        assert heating.state["Q_heat_demand"] == pytest.approx(0.3 * 50.0 + 0.6 * 30.0)

    def test_flat_coefficient_is_not_split(self) -> None:
        heating = HeatingSystem("h", target_temperature=313.15, heat_loss_coefficient=1.0)

        heating.step(0.0, 1.0 / 24.0, {"T_ambient": 263.15, "T_ground": 283.15, "heat_loss_coefficient_ground": 0.6})

        assert heating.state["Q_heat_demand"] == pytest.approx(50.0)

    def test_plant_default_and_override(self, tmp_path) -> None:
        assert BiogasPlant("p").T_ground == 283.15
        plant = BiogasPlant("p", T_ground=281.15)
        path = tmp_path / "plant.json"

        plant.to_json(str(path))

        assert BiogasPlant.from_json(str(path)).T_ground == 281.15

    def test_plant_value_reaches_a_connected_heating(self) -> None:
        from pyadm1 import Feedstock
        from pyadm1.configurator.plant_configurator import PlantConfigurator

        fs = Feedstock(["maize_silage_milk_ripeness", "swine_manure"], feeding_freq=24, total_simtime=5)

        def demand(T_ground: float) -> float:
            plant = BiogasPlant("p", T_ambient=263.15, T_ground=T_ground)
            cfg = PlantConfigurator(plant, fs)
            cfg.add_digester("dig", T_ad=313.15, Q_substrates=[0.0] * 10, **self.KW)
            heating = cfg.add_heating("h", target_temperature=313.15, digester_id="dig")
            plant.step(1.0 / 24.0)
            return heating.state["Q_heat_demand"]

        UA_ground = Digester("d", None, **self.KW).heat_loss_coefficient_ground
        assert demand(278.15) - demand(283.15) == pytest.approx(UA_ground * 5.0)
