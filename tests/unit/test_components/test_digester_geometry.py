"""Tank geometry of a digester and the plant-level ambient temperature.

The digester used to know only its volumes. It now carries a diameter (given
or estimated from ``V_liq + V_gas``) and a roof type; every height is derived.
``BiogasPlant.T_ambient`` replaces the fixed 15 °C the heating system fell back
to because nothing ever supplied the ``T_ambient`` input.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from pyadm1.components.biological import Digester
from pyadm1.components.energy.heating import HeatingSystem
from pyadm1.configurator.connection_manager import Connection
from pyadm1.configurator.plant_builder import BiogasPlant


class TestDiameter:
    def test_given_diameter_is_kept(self) -> None:
        d = Digester("d", None, V_liq=3000.0, V_gas=175.0, diameter=20.0)

        assert d.diameter == 20.0
        assert d.diameter_estimated is False

    def test_missing_diameter_is_estimated_from_both_volumes(self) -> None:
        d = Digester("d", None, V_liq=3000.0, V_gas=175.0)

        assert d.diameter_estimated is True
        # Cylinder of V_liq + V_gas with H = 0.4 * D.
        assert np.pi / 4.0 * d.diameter**2 * (0.4 * d.diameter) == pytest.approx(3175.0)
        assert d.diameter > Digester("d2", None, V_liq=3000.0, V_gas=0.0).diameter

    def test_estimate_is_realistic_for_an_agricultural_digester(self) -> None:
        # Gummersbach main digester: 20 m measured.
        assert Digester("d", None, V_liq=3000.0, V_gas=175.0).diameter == pytest.approx(20.0, abs=2.0)

    @pytest.mark.parametrize("diameter", [0.0, -5.0])
    def test_non_positive_diameter_is_rejected(self, diameter: float) -> None:
        with pytest.raises(ValueError, match="diameter"):
            Digester("d", None, V_liq=1000.0, V_gas=150.0, diameter=diameter)


class TestRoofType:
    def test_unknown_roof_type_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="roof_type"):
            Digester("d", None, V_liq=1000.0, V_gas=150.0, roof_type="thatched")

    def test_flat_roof_keeps_headspace_inside_the_cylinder(self) -> None:
        d = Digester("d", None, V_liq=3000.0, V_gas=175.0, diameter=20.0, roof_type="concrete")

        assert d.height == pytest.approx(3175.0 / d.base_area)
        assert d.roof_height == 0.0
        assert d.roof_area == pytest.approx(d.base_area)

    def test_membrane_dome_holds_the_gas_volume(self) -> None:
        d = Digester("d", None, V_liq=3000.0, V_gas=175.0, diameter=20.0, roof_type="membrane")

        assert d.height == pytest.approx(d.liquid_height) == pytest.approx(3000.0 / d.base_area)
        h, a = d.roof_height, d.diameter / 2.0
        assert np.pi * h / 6.0 * (3.0 * a**2 + h**2) == pytest.approx(175.0)
        assert d.roof_area > d.base_area
        assert d.total_height == pytest.approx(d.height + h)

    def test_open_tank_has_no_headspace(self) -> None:
        d = Digester("d", None, V_liq=3000.0, V_gas=175.0, roof_type="open")

        assert d.diameter == pytest.approx(Digester("d2", None, V_liq=3000.0, V_gas=0.0).diameter)
        assert d.roof_height == 0.0
        assert d.height == pytest.approx(d.liquid_height)


class TestSerialization:
    def test_given_diameter_and_roof_round_trip(self) -> None:
        d = Digester("d", None, V_liq=3000.0, V_gas=175.0, diameter=20.0, roof_type="concrete")

        restored = Digester.from_dict(d.to_dict())

        assert restored.diameter == 20.0
        assert restored.diameter_estimated is False
        assert restored.roof_type == "concrete"

    def test_estimated_diameter_stays_an_estimate(self) -> None:
        d = Digester("d", None, V_liq=3000.0, V_gas=175.0)

        assert d.to_dict()["diameter"] is None
        restored = Digester.from_dict(d.to_dict())
        assert restored.diameter_estimated is True
        assert restored.diameter == pytest.approx(d.diameter)

    def test_config_without_geometry_still_loads(self) -> None:
        config = Digester("d", None, V_liq=1000.0, V_gas=150.0).to_dict()
        del config["diameter"], config["roof_type"]

        restored = Digester.from_dict(config)

        assert restored.roof_type == "membrane"
        assert restored.diameter_estimated is True


class TestPlantAmbientTemperature:
    @staticmethod
    def _plant(**kwargs) -> tuple[BiogasPlant, HeatingSystem]:
        plant = BiogasPlant("p", **kwargs)
        heating = HeatingSystem("h", target_temperature=313.15, heat_loss_coefficient=1.0)
        plant.add_component(heating)
        return plant, heating

    def test_default_is_15_degrees(self) -> None:
        plant, heating = self._plant()

        plant.step(1.0 / 24.0)

        assert plant.T_ambient == 288.15
        assert heating.state["Q_heat_demand"] == pytest.approx(313.15 - 288.15)

    def test_plant_value_reaches_the_heating_system(self) -> None:
        plant, heating = self._plant(T_ambient=273.15)

        plant.step(1.0 / 24.0)

        assert heating.state["Q_heat_demand"] == pytest.approx(40.0)

    def test_value_can_be_changed_between_steps(self) -> None:
        plant, heating = self._plant()
        plant.step(1.0 / 24.0)

        plant.T_ambient = 263.15
        plant.step(1.0 / 24.0)

        assert heating.state["Q_heat_demand"] == pytest.approx(50.0)

    def test_value_reaches_a_chp_fed_heating_system(self) -> None:
        from pyadm1.components.energy.chp import CHP

        plant, heating = self._plant(T_ambient=273.15)
        plant.add_component(CHP("chp"))
        plant.add_connection(Connection("chp", "h", "heat"))

        plant.step(1.0 / 24.0)

        assert heating.state["Q_heat_demand"] == pytest.approx(40.0)

    def test_round_trips_through_json(self, tmp_path) -> None:
        plant, _ = self._plant(T_ambient=278.15)
        path = tmp_path / "plant.json"

        plant.to_json(str(path))

        assert BiogasPlant.from_json(str(path)).T_ambient == 278.15


class TestHeatLossCoefficient:
    """UA = Σ k_i · A_i from the optional U-values and the derived surfaces."""

    GUMMERSBACH: ClassVar[dict] = dict(V_liq=3000.0, V_gas=175.0, diameter=20.0, roof_type="concrete")

    def test_none_without_k_values(self) -> None:
        assert Digester("d", None, **self.GUMMERSBACH).heat_loss_coefficient is None

    def test_sums_wall_roof_and_ground(self) -> None:
        d = Digester("d", None, k_wall=0.4, k_roof=0.25, k_ground=1.9, **self.GUMMERSBACH)

        expected_w_per_k = 0.4 * d.wall_area + 0.25 * d.roof_area + 1.9 * d.base_area
        assert d.heat_loss_coefficient == pytest.approx(expected_w_per_k / 1000.0)
        # Legacy Gummersbach values: roughly twice the flat 0.5 kW/K default.
        assert d.heat_loss_coefficient == pytest.approx(0.93, abs=0.02)

    def test_partial_k_values_warn_and_count_only_what_is_given(self) -> None:
        with pytest.warns(UserWarning, match="only some"):
            d = Digester("d", None, k_wall=0.4, **self.GUMMERSBACH)

        assert d.heat_loss_coefficient == pytest.approx(0.4 * d.wall_area / 1000.0)

    def test_open_tank_needs_no_roof_value(self, recwarn) -> None:
        d = Digester("d", None, V_liq=3000.0, V_gas=0.0, roof_type="open", k_wall=0.4, k_roof=9.9, k_ground=1.9)

        assert not [w for w in recwarn if issubclass(w.category, UserWarning)]
        assert d.heat_loss_coefficient == pytest.approx((0.4 * d.wall_area + 1.9 * d.base_area) / 1000.0)

    def test_negative_k_value_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="k_roof"):
            Digester("d", None, k_roof=-0.1, **self.GUMMERSBACH)

    def test_k_values_round_trip(self) -> None:
        d = Digester("d", None, k_wall=0.4, k_roof=0.25, k_ground=1.9, **self.GUMMERSBACH)

        restored = Digester.from_dict(d.to_dict())

        assert (restored.k_wall, restored.k_roof, restored.k_ground) == (0.4, 0.25, 1.9)
        assert restored.heat_loss_coefficient == pytest.approx(d.heat_loss_coefficient)


class TestDigesterHeatingLink:
    """A heating system connected to a digester takes UA, temperature and feed from it."""

    @staticmethod
    def _plant(heat_loss_coefficient=None, **digester_kwargs):
        from pyadm1 import Feedstock
        from pyadm1.configurator.plant_configurator import PlantConfigurator

        fs = Feedstock(["maize_silage_milk_ripeness", "swine_manure"], feeding_freq=24, total_simtime=5)
        plant = BiogasPlant("p", T_ambient=278.15)
        cfg = PlantConfigurator(plant, fs)
        cfg.add_digester(
            "dig", V_liq=3000.0, V_gas=175.0, T_ad=313.15, Q_substrates=[10.0, 5.0] + [0.0] * 8, **digester_kwargs
        )
        heating = cfg.add_heating(
            "h", target_temperature=313.15, heat_loss_coefficient=heat_loss_coefficient, digester_id="dig"
        )
        return plant, heating

    def test_heating_uses_the_digester_coefficient(self) -> None:
        plant, heating = self._plant(diameter=20.0, roof_type="concrete", k_wall=0.4, k_roof=0.25, k_ground=1.9)

        plant.step(1.0 / 24.0)

        UA = plant.components["dig"].heat_loss_coefficient
        assert heating.state["heat_loss_coefficient_used"] == pytest.approx(UA)

    def test_explicit_heating_coefficient_wins(self) -> None:
        plant, heating = self._plant(heat_loss_coefficient=0.7, k_wall=0.4, k_roof=0.25, k_ground=1.9)

        plant.step(1.0 / 24.0)

        assert heating.state["heat_loss_coefficient_used"] == 0.7

    def test_flat_default_without_k_values(self) -> None:
        plant, heating = self._plant()

        plant.step(1.0 / 24.0)

        assert heating.state["heat_loss_coefficient_used"] == 0.5

    def test_substrate_feed_reaches_the_heating(self) -> None:
        plant, heating = self._plant()

        plant.step(1.0 / 24.0)

        Q_loss = 0.5 * (313.15 - 278.15)
        assert heating.state["Q_heat_demand"] > Q_loss  # process heat on top of the losses

    def test_feed_signal_does_not_override_a_downstream_digester(self) -> None:
        assert "Q_substrates" not in Digester("d", None).outputs_data
        plant, _ = self._plant()
        plant.step(1.0 / 24.0)
        assert "Q_substrates" not in plant.components["dig"].outputs_data
