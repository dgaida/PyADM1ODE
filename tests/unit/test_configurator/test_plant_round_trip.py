"""Saving and loading a plant with every component kind.

Pumps and feeders used to register as ``mixer`` and the substrate storage as
``storage``; ``from_json`` knew neither ``mixer`` nor ``sensor``. A plant with
any of these could be saved but not loaded, and a substrate storage came back
as a gas storage.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyadm1 import Feedstock
from pyadm1.components.base import ComponentType
from pyadm1.components.energy.biogas_upgrading import BiogasUpgrading
from pyadm1.components.energy.boiler import Boiler
from pyadm1.components.energy.chp import CHP
from pyadm1.components.energy.flare import Flare
from pyadm1.components.energy.gas_storage import GasStorage
from pyadm1.components.energy.heating import HeatingSystem
from pyadm1.components.feeding.feeder import Feeder
from pyadm1.components.feeding.substrate_storage import SubstrateStorage
from pyadm1.components.mechanical.mixer import Mixer
from pyadm1.components.mechanical.pump import Pump
from pyadm1.components.registry import get_registry
from pyadm1.components.sensors.chemical import ChemicalSensor
from pyadm1.components.sensors.gas import GasSensor
from pyadm1.components.sensors.physical import PhysicalSensor
from pyadm1.configurator.connection_manager import Connection
from pyadm1.configurator.plant_builder import BiogasPlant
from pyadm1.configurator.plant_configurator import PlantConfigurator


@pytest.fixture
def feedstock() -> Feedstock:
    return Feedstock(["maize_silage_milk_ripeness", "swine_manure"], feeding_freq=24, total_simtime=5)


class TestComponentTypes:
    def test_each_kind_has_its_own_type(self) -> None:
        assert Pump("p").component_type is ComponentType.PUMP
        assert Feeder("f").component_type is ComponentType.FEEDER
        assert SubstrateStorage("s").component_type is ComponentType.SUBSTRATE_STORAGE
        assert Mixer("m").component_type is ComponentType.MIXER
        assert GasStorage("g").component_type is ComponentType.STORAGE

    def test_all_component_classes_are_registered(self) -> None:
        assert {"Pump", "Mixer", "Feeder", "SubstrateStorage"} <= set(get_registry().get_registered_components())

    def test_substrate_storage_is_not_stepped_as_a_gas_storage(self) -> None:
        plant = BiogasPlant("p")
        silo = SubstrateStorage("silo", capacity=1000.0, initial_level=500.0)
        plant.add_component(silo)

        plant.step(1.0)

        # The gas-storage pass would have fed it gas inputs; the generic pass
        # runs the silo's own balance, so storage losses accrue.
        assert silo.state["current_level"] < 500.0


class TestRoundTrip:
    @staticmethod
    def _save_and_load(plant: BiogasPlant, tmp_path: Path, feedstock=None) -> BiogasPlant:
        path = tmp_path / "plant.json"
        plant.to_json(str(path))
        return BiogasPlant.from_json(str(path), feedstock=feedstock)

    def test_every_component_kind_survives(self, tmp_path: Path, feedstock: Feedstock) -> None:
        plant = BiogasPlant("all")
        cfg = PlantConfigurator(plant, feedstock)
        cfg.add_digester("dig", V_liq=1000.0, V_gas=150.0, Q_substrates=[10.0, 5.0] + [0.0] * 8)
        for component in (
            CHP("chp"),
            HeatingSystem("heat"),
            Boiler("boiler"),
            Flare("flare"),
            BiogasUpgrading("bgaa"),
            Pump("pump"),
            Mixer("mixer"),
            Feeder("feeder"),
            SubstrateStorage("silo", storage_type="bunker_silo", capacity=800.0),
            PhysicalSensor("ph", sensor_type="pH"),
            ChemicalSensor("vfa", sensor_type="VFA"),
            GasSensor("ch4", sensor_type="CH4"),
        ):
            plant.add_component(component)

        restored = self._save_and_load(plant, tmp_path, feedstock)

        assert {cid: type(c) for cid, c in restored.components.items()} == {
            cid: type(c) for cid, c in plant.components.items()
        }
        assert restored.components["silo"].capacity == 800.0

    def test_file_without_component_class_still_loads(self, tmp_path: Path) -> None:
        import json

        plant = BiogasPlant("legacy")
        plant.add_component(Pump("pump"))
        plant.add_component(GasStorage("gs"))
        path = tmp_path / "plant.json"
        plant.to_json(str(path))
        config = json.loads(path.read_text())
        for component in config["components"]:
            del component["component_class"]
        path.write_text(json.dumps(config))

        restored = BiogasPlant.from_json(str(path))

        assert isinstance(restored.components["pump"], Pump)
        assert isinstance(restored.components["gs"], GasStorage)

    def test_heating_gets_its_feedstock_back(self, tmp_path: Path, feedstock: Feedstock) -> None:
        plant = BiogasPlant("heat")
        cfg = PlantConfigurator(plant, feedstock)
        cfg.add_digester("dig", V_liq=1000.0, V_gas=150.0, T_ad=313.15, Q_substrates=[10.0, 5.0] + [0.0] * 8)
        cfg.add_heating("heat", target_temperature=313.15, digester_id="dig")

        restored = self._save_and_load(plant, tmp_path, feedstock)
        plant.step(1.0 / 24.0)
        restored.step(1.0 / 24.0)

        assert restored.components["heat"].feedstock is feedstock
        demand = plant.components["heat"].state["Q_heat_demand"]
        assert demand > 0.5 * (313.15 - 288.15)  # process heat on top of the losses
        assert restored.components["heat"].state["Q_heat_demand"] == pytest.approx(demand)

    def test_cumulative_counters_survive(self) -> None:
        heating = HeatingSystem("h", heat_loss_coefficient=1.0)
        heating.step(0.0, 1.0, {"T_ambient": 278.15})
        chp = CHP("chp")
        chp.step(0.0, 1.0, {"Q_ch4": 3000.0})

        assert HeatingSystem.from_dict(heating.to_dict()).state["energy_consumed"] == heating.state["energy_consumed"] > 0.0
        assert CHP.from_dict(chp.to_dict()).state["operating_hours"] == 24.0

    def test_heating_is_marked_initialized(self) -> None:
        assert HeatingSystem("h")._initialized is True

    def test_flare_and_upgrading_keep_their_connections(self) -> None:
        plant = BiogasPlant("p")
        plant.add_component(BiogasUpgrading("bgaa"))
        plant.add_component(Flare("flare"))
        plant.add_connection(Connection("bgaa", "flare", "gas"))

        flare = Flare.from_dict(plant.components["flare"].to_dict())
        bgaa = BiogasUpgrading.from_dict(plant.components["bgaa"].to_dict())

        assert flare.inputs == ["bgaa"]
        assert bgaa.outputs == ["flare"]
