# -*- coding: utf-8 -*-
"""Unit tests for PlantConfigurator (high-level builder helpers)."""

import pytest

from pyadm1 import BiogasPlant, Feedstock
from pyadm1.components.biological import Digester
from pyadm1.components.energy.chp import CHP
from pyadm1.components.energy.heating import HeatingSystem
from pyadm1.configurator.plant_configurator import PlantConfigurator


@pytest.fixture
def feedstock() -> Feedstock:
    return Feedstock(
        ["maize_silage_milk_ripeness", "swine_manure"],
        feeding_freq=24,
        total_simtime=5,
    )


@pytest.fixture
def cfg(feedstock: Feedstock) -> PlantConfigurator:
    plant = BiogasPlant("Test")
    return PlantConfigurator(plant, feedstock)


class TestAddDigester:
    def test_creates_and_registers_digester(self, cfg: PlantConfigurator) -> None:
        digester, info = cfg.add_digester(
            digester_id="d1",
            V_liq=1200.0,
            V_gas=216.0,
            T_ad=315.15,
            Q_substrates=[11.4, 6.1, 0, 0, 0, 0, 0, 0, 0, 0],
        )

        assert isinstance(digester, Digester)
        assert "d1" in cfg.plant.components
        assert "d1_storage" in cfg.plant.components
        assert "Auto-built" in info

    def test_user_supplied_initial_state_is_used(self, cfg: PlantConfigurator) -> None:
        custom_state = [0.001 * (i + 1) for i in range(41)]
        digester, info = cfg.add_digester(
            digester_id="d_custom",
            adm1_state=custom_state,
        )
        assert digester.adm1_state == custom_state
        assert "User-supplied" in info

    def test_kla_override_applied_to_underlying_model(self, cfg: PlantConfigurator) -> None:
        digester, _ = cfg.add_digester(digester_id="d1", k_L_a=123.0)
        assert digester.adm1.get_calibration_parameters()["k_L_a"] == 123.0

    def test_default_q_substrates_is_zero_list(self, cfg: PlantConfigurator) -> None:
        digester, _ = cfg.add_digester(digester_id="d1")
        assert digester.Q_substrates == [0.0] * 10


class TestAddChpAndHeating:
    def test_add_chp_creates_chp_and_flare(self, cfg: PlantConfigurator) -> None:
        chp = cfg.add_chp(chp_id="chp1", P_el_nom=500.0)
        assert isinstance(chp, CHP)
        assert "chp1" in cfg.plant.components
        assert "chp1_flare" in cfg.plant.components

    def test_add_heating_registers_component(self, cfg: PlantConfigurator) -> None:
        heating = cfg.add_heating("heat1", target_temperature=308.15)
        assert isinstance(heating, HeatingSystem)
        assert "heat1" in cfg.plant.components


class TestAutoConnections:
    def test_auto_connect_digester_to_chp_via_storage(self, cfg: PlantConfigurator) -> None:
        cfg.add_digester("d1")
        cfg.add_chp("chp1")

        cfg.auto_connect_digester_to_chp("d1", "chp1")

        assert any(c.from_component == "d1_storage" and c.to_component == "chp1" for c in cfg.plant.connections)

    def test_auto_connect_missing_storage_raises(self, cfg: PlantConfigurator) -> None:
        cfg.add_chp("chp1")
        with pytest.raises(ValueError, match="Gas storage"):
            cfg.auto_connect_digester_to_chp("not_added", "chp1")

    def test_auto_connect_chp_to_heating_creates_heat_link(self, cfg: PlantConfigurator) -> None:
        cfg.add_chp("chp1")
        cfg.add_heating("heat1")
        cfg.auto_connect_chp_to_heating("chp1", "heat1")

        assert any(
            c.from_component == "chp1" and c.to_component == "heat1" and c.connection_type == "heat"
            for c in cfg.plant.connections
        )


class TestSingleStagePlant:
    def test_creates_digester_chp_heating(self, cfg: PlantConfigurator) -> None:
        components = cfg.create_single_stage_plant(
            digester_config={"digester_id": "main", "V_liq": 1200.0, "V_gas": 216.0, "T_ad": 315.15},
            chp_config={"chp_id": "chp1", "P_el_nom": 500.0},
            heating_config={"heating_id": "heat1", "target_temperature": 315.15},
        )

        assert components["digester"] == "main"
        assert components["chp"] == "chp1"
        assert components["heating"] == "heat1"
        assert "flare" in components


class TestTwoStagePlant:
    def test_creates_hydrolysis_and_main_digester(self, cfg: PlantConfigurator) -> None:
        components = cfg.create_two_stage_plant(
            hydrolysis_config={"V_liq": 500.0, "V_gas": 75.0, "T_ad": 318.15},
            digester_config={"V_liq": 1200.0, "V_gas": 216.0, "T_ad": 315.15},
        )

        assert "hydrolysis" in components
        assert "digester" in components
        assert any(
            c.from_component == components["hydrolysis"]
            and c.to_component == components["digester"]
            and c.connection_type == "liquid"
            for c in cfg.plant.connections
        )

    def test_thermophilic_hydrolysis_temperature(self, cfg: PlantConfigurator) -> None:
        components = cfg.create_two_stage_plant(
            hydrolysis_config={"T_ad": 328.15},  # 55 °C
        )
        hydro = cfg.plant.components[components["hydrolysis"]]
        assert hydro.T_ad == 328.15
