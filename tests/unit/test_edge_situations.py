"""Situations at the edge of normal operation that the focused suites did not reach.

Each test describes a state a plant can really be in (an idle upstream tank, a
restored snapshot, a mis-typed configuration value) and asserts what the model
must do there, not merely that it runs.
"""

from __future__ import annotations

import pytest

from pyadm1 import BiogasPlant, Feedstock
from pyadm1.components.biological.separator import Separator
from pyadm1.components.energy.biogas_upgrading import BiogasUpgrading
from pyadm1.components.energy.gas_storage import GasStorage
from pyadm1.components.feeding.feeder import Feeder
from pyadm1.components.feeding.substrate_storage import SubstrateStorage
from pyadm1.configurator.plant_configurator import PlantConfigurator


@pytest.fixture(scope="module")
def feedstock() -> Feedstock:
    return Feedstock(["maize_silage_milk_ripeness", "cattle_manure"], feeding_freq=24, total_simtime=5)


class TestRestoredSnapshot:
    def test_computed_values_in_a_snapshot_are_ignored(self) -> None:
        """A snapshot may report a computed value; restoring it must not try to set it."""
        storage = SubstrateStorage("silo", capacity=500.0, initial_level=200.0)
        storage.initialize()
        snapshot = storage.get_state()
        snapshot["available_stock"] = 9999.0  # read-only property of the storage
        snapshot["current_level"] = 120.0

        storage.set_state(snapshot)

        assert storage.current_level == pytest.approx(120.0)
        assert storage.available_stock == pytest.approx(120.0)

    def test_gas_storage_takes_its_gas_composition_from_the_initial_state(self) -> None:
        storage = GasStorage("gs", capacity_m3=1000.0)

        storage.initialize({"ch4_fraction": 0.62, "co2_fraction": 0.36})

        assert storage.ch4_fraction == pytest.approx(0.62)
        assert storage.co2_fraction == pytest.approx(0.36)

    def test_gas_storage_rejects_a_composition_above_one_hundred_percent(self) -> None:
        storage = GasStorage("gs", capacity_m3=1000.0)

        with pytest.raises(ValueError, match="sum to at most 1"):
            storage.initialize({"ch4_fraction": 0.7, "co2_fraction": 0.4})


class TestInvalidConfiguration:
    def test_upgrading_rejects_a_negative_electricity_demand(self) -> None:
        with pytest.raises(ValueError, match="specific_electricity"):
            BiogasUpgrading("up", specific_electricity=-0.1)

    @pytest.mark.parametrize("factor", [0.0, -1.0])
    def test_feeder_rejects_a_non_positive_density_factor(self, factor: float) -> None:
        """The factor divides the stock of the storage; zero would mean infinite stock."""
        with pytest.raises(ValueError, match="flow_to_mass_factor"):
            Feeder("f", flow_to_mass_factor=factor)

    def test_storage_for_a_substrate_needs_a_feedstock(self) -> None:
        """Without a feedstock the substrate name cannot be mapped to a feed slot."""
        configurator = PlantConfigurator(BiogasPlant("p"))

        with pytest.raises(ValueError, match="feedstock is required"):
            configurator.add_substrate_storage("silo", substrate="maize_silage_milk_ripeness")


class TestSeparatorWithoutUsableState:
    def test_non_numeric_upstream_state_is_not_passed_on(self) -> None:
        """A state that is no list of numbers yields no liquid-fraction state, not a crash."""
        separator = Separator("sep")

        assert separator._split_state(["not", "numbers"], Q_in=10.0, Q_liquid=8.0) is None
        assert separator._split_state(object(), Q_in=10.0, Q_liquid=8.0) is None


class TestIdleUpstreamDigester:
    def test_second_digester_receives_nothing_while_the_first_is_not_fed(self, feedstock: Feedstock) -> None:
        """An unfed first stage has no outflow, so the second stage keeps its volume and gets no liquid."""
        plant = BiogasPlant("two-stage")
        configurator = PlantConfigurator(plant, feedstock)
        configurator.add_digester("first", V_liq=1000.0, V_gas=150.0, Q_substrates=[0.0, 0.0])
        configurator.add_digester("second", V_liq=1000.0, V_gas=150.0, Q_substrates=[0.0, 0.0])
        configurator.connect("first", "second", "liquid")
        plant.initialize()

        results = plant.step(1.0)

        assert results["first"]["Q_out"] == pytest.approx(0.0, abs=1e-9)
        assert results["second"]["Q_out"] == pytest.approx(0.0, abs=1e-9)
        assert results["second"]["V_liq"] == pytest.approx(1000.0)
