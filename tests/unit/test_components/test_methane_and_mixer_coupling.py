"""Methane content for boiler and flare, methane heating value, mixer coupling.

The boiler burnt every biogas at a flat 6.0 kWh/m³ and the flare assumed 60 %
methane, although the gas storage reports the actual content. The mixer
estimated its own tank geometry and never saw the digester's temperature or
total solids, because the input keys did not match.
"""

from __future__ import annotations

import math

import pytest

from pyadm1 import Feedstock
from pyadm1.components.energy.boiler import Boiler
from pyadm1.components.energy.chp import CHP
from pyadm1.components.energy.flare import Flare
from pyadm1.components.energy.gas_storage import GasStorage
from pyadm1.components.mechanical.mixer import Mixer
from pyadm1.configurator.connection_manager import Connection
from pyadm1.configurator.plant_builder import BiogasPlant
from pyadm1.configurator.plant_configurator import PlantConfigurator


class TestBoilerMethaneContent:
    @staticmethod
    def _gas(inputs: dict) -> dict:
        return Boiler("b", P_th_nom=200.0, fuel_type="biogas").step(0.0, 1.0 / 24.0, {"P_th_demand": 200.0, **inputs})

    def test_flat_heating_value_without_composition(self) -> None:
        out = self._gas({})

        assert out["lhv_biogas_actual"] == 6.0

    def test_lean_gas_needs_more_volume(self) -> None:
        lean = self._gas({"CH4_fraction": 0.50})
        rich = self._gas({"CH4_fraction": 0.65})

        assert lean["lhv_biogas_actual"] == pytest.approx(5.0)
        assert lean["Q_gas_consumed_m3_per_day"] == pytest.approx(rich["Q_gas_consumed_m3_per_day"] * 0.65 / 0.50)

    def test_sixty_percent_matches_the_flat_value(self) -> None:
        assert self._gas({"CH4_fraction": 0.60})["Q_gas_consumed_m3_per_day"] == pytest.approx(
            self._gas({})["Q_gas_consumed_m3_per_day"]
        )

    def test_storage_composition_reaches_the_boiler_in_a_plant(self) -> None:
        plant = BiogasPlant("p")
        plant.add_component(GasStorage("gs", initial_ch4_fraction=0.50, initial_co2_fraction=0.50))
        plant.add_component(Boiler("b", fuel_type="biogas"))
        plant.add_connection(Connection("gs", "b", "gas"))

        results = plant.step(1.0 / 24.0)

        assert results["b"]["lhv_biogas_actual"] == pytest.approx(5.0)

    def test_lhv_ch4_round_trips(self) -> None:
        assert Boiler.from_dict(Boiler("b", lhv_ch4=9.97).to_dict()).lhv_ch4 == 9.97


class TestFlareMethaneContent:
    def test_slip_is_reported(self) -> None:
        out = Flare("f", destruction_efficiency=0.98).step(0.0, 1.0, {"Q_gas_in_m3_per_day": 100.0, "CH4_fraction": 0.5})

        assert out["CH4_destroyed_m3"] == pytest.approx(49.0)
        assert out["CH4_slip_m3"] == pytest.approx(1.0)

    def test_default_fraction_is_a_property(self) -> None:
        flare = Flare("f", ch4_fraction_default=0.52)

        out = flare.step(0.0, 1.0, {"Q_gas_in_m3_per_day": 100.0})

        assert out["CH4_destroyed_m3"] == pytest.approx(100.0 * 0.52 * 0.98)
        assert Flare.from_dict(flare.to_dict()).ch4_fraction_default == 0.52


class TestMethaneHeatingValue:
    def test_power_scales_with_the_heating_value(self) -> None:
        inputs = {"Q_gas_supplied_m3_per_day": 1000.0, "CH4_fraction": 0.5}

        low = CHP("a", lhv_ch4=9.97).step(0.0, 1.0 / 24.0, inputs)
        high = CHP("b", lhv_ch4=10.0).step(0.0, 1.0 / 24.0, inputs)

        assert low["P_el"] == pytest.approx(high["P_el"] * 9.97 / 10.0)

    def test_round_trip_and_validation(self) -> None:
        assert CHP.from_dict(CHP("c", lhv_ch4=9.97).to_dict()).lhv_ch4 == 9.97
        with pytest.raises(ValueError, match="lhv_ch4"):
            CHP("c", lhv_ch4=0.0)

    def test_builder_sizes_the_gas_demand_with_it(self) -> None:
        def supplied(lhv: float) -> float:
            plant = BiogasPlant("p")
            plant.add_component(GasStorage("gs", capacity_m3=50000.0, initial_fill_fraction=0.5))
            plant.add_component(CHP("chp", lhv_ch4=lhv))
            plant.add_connection(Connection("gs", "chp", "gas"))
            return plant.step(1.0 / 24.0)["gs"]["Q_gas_supplied_m3_per_day"]

        assert supplied(8.0) == pytest.approx(supplied(10.0) * 10.0 / 8.0)


class TestMixerFluidModel:
    def test_fluid_properties_are_constructor_arguments_and_round_trip(self) -> None:
        mixer = Mixer("m", fluid_density=1050.0, fluid_viscosity=0.08, ts_reference=8.0, viscosity_ts_coefficient=0.3)

        restored = Mixer.from_dict(mixer.to_dict())

        assert restored.fluid_density == 1050.0
        assert restored.viscosity_reference == 0.08
        assert (restored.ts_reference, restored.viscosity_ts_coefficient) == (8.0, 0.3)

    def test_reference_state_gives_the_reference_viscosity(self) -> None:
        mixer = Mixer("m", fluid_viscosity=0.05, ts_reference=6.0)

        mixer.step(0.0, 1.0 / 24.0, {"T_digester": 308.15, "TS": 6.0})

        assert mixer.fluid_viscosity == pytest.approx(0.05)

    def test_thicker_and_colder_sludge_is_more_viscous(self) -> None:
        mixer = Mixer("m")
        mixer.step(0.0, 1.0 / 24.0, {"T_digester": 308.15, "TS": 6.0})
        base = mixer.fluid_viscosity

        mixer.step(0.0, 1.0 / 24.0, {"TS": 10.0})
        thick = mixer.fluid_viscosity
        mixer.step(0.0, 1.0 / 24.0, {"T_digester": 298.15})

        assert thick == pytest.approx(base * math.exp(0.5 * 4.0))
        assert mixer.fluid_viscosity > thick  # TS is remembered, cooling adds on top

    def test_repeated_updates_do_not_compound(self) -> None:
        mixer = Mixer("m")
        for _ in range(5):
            mixer.step(0.0, 1.0 / 24.0, {"T_digester": 313.15, "TS": 9.0})
        once = Mixer("m2")
        once.step(0.0, 1.0 / 24.0, {"T_digester": 313.15, "TS": 9.0})

        assert mixer.fluid_viscosity == pytest.approx(once.fluid_viscosity)

    def test_ts_coupling_can_be_switched_off(self) -> None:
        mixer = Mixer("m", viscosity_ts_coefficient=0.0)

        mixer.step(0.0, 1.0 / 24.0, {"T_digester": 308.15, "TS": 12.0})

        assert mixer.fluid_viscosity == pytest.approx(0.05)

    def test_explicit_viscosity_wins(self) -> None:
        mixer = Mixer("m")

        mixer.step(0.0, 1.0 / 24.0, {"T_digester": 298.15, "TS": 12.0, "fluid_viscosity": 0.123})

        assert mixer.fluid_viscosity == 0.123


class TestMixerDigesterCoupling:
    @staticmethod
    def _cfg() -> tuple[BiogasPlant, PlantConfigurator]:
        fs = Feedstock(["maize_silage_milk_ripeness", "swine_manure"], feeding_freq=24, total_simtime=5)
        plant = BiogasPlant("p")
        cfg = PlantConfigurator(plant, fs)
        cfg.add_digester("dig", V_liq=3000.0, V_gas=175.0, diameter=20.0, T_ad=313.15, Q_substrates=[10.0, 5.0] + [0.0] * 8)
        return plant, cfg

    def test_geometry_comes_from_the_digester(self) -> None:
        plant, cfg = self._cfg()
        digester = plant.components["dig"]

        mixer = cfg.add_mixer("mix", "dig")

        assert mixer.tank_volume == 3000.0
        assert mixer.tank_diameter == 20.0
        assert mixer.tank_height == pytest.approx(digester.liquid_height)
        # The mixer's own estimate would have been a tall, narrow tank.
        assert Mixer("alone", tank_volume=3000.0).tank_diameter == pytest.approx(13.66, abs=0.01)

    def test_tank_arguments_cannot_be_overridden(self) -> None:
        _, cfg = self._cfg()

        with pytest.raises(ValueError, match="tank_diameter"):
            cfg.add_mixer("mix", "dig", tank_diameter=5.0)

    def test_unknown_digester_is_rejected(self) -> None:
        _, cfg = self._cfg()

        with pytest.raises(ValueError, match="ghost"):
            cfg.add_mixer("mix", "ghost")

    def test_mixer_follows_the_digester_state(self) -> None:
        plant, cfg = self._cfg()
        mixer = cfg.add_mixer("mix", "dig", power_installed=15.0)

        plant.step(1.0 / 24.0)
        plant.step(1.0 / 24.0)

        digester_out = plant.components["dig"].outputs_data
        assert mixer._fluid_temperature == 313.15
        assert mixer._fluid_ts == pytest.approx(digester_out["TS"])
