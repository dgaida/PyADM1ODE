"""Gas train of a plant: storage, consumers, flare and boiler act on one balance.

The storage used to be stepped twice per plant step, first with the production
and then with the demand. A whole step's production was vented before any
consumer could draw it, so the gas burnt depended on step length and storage
size; the vented gas never reached the flare; and every CHP was stepped twice.
"""

from __future__ import annotations

from typing import Any

import pytest

from pyadm1.components.base import Component, ComponentType
from pyadm1.components.energy.biogas_upgrading import BiogasUpgrading
from pyadm1.components.energy.boiler import Boiler
from pyadm1.components.energy.chp import CHP
from pyadm1.components.energy.flare import Flare
from pyadm1.components.energy.gas_storage import GasStorage
from pyadm1.components.energy.heating import HeatingSystem
from pyadm1.components.mechanical.pump import Pump
from pyadm1.configurator.connection_manager import Connection
from pyadm1.configurator.plant_builder import BiogasPlant

# A CHP of this size burns 2000 m³/d of biogas with 60 % methane.
_P_EL_FOR_2000_M3 = 2000.0 * 0.6 * 10.0 / 24.0 * 0.4


class _GasSource(Component):
    """Stands in for a digester: constant gas production and effluent."""

    def __init__(self, component_id: str, q_gas: float = 2400.0, ch4: float = 0.6, q_out: float = 0.0):
        super().__init__(component_id, ComponentType.DIGESTER)
        self.q_gas = q_gas
        self.ch4 = ch4
        self.q_out = q_out
        self.Q_substrates = [0.0]
        self.received: list[dict[str, Any]] = []
        self.initialize()

    def initialize(self, initial_state: dict[str, Any] | None = None) -> None:
        self.outputs_data = {}
        self._initialized = True

    def step(self, t: float, dt: float, inputs: dict[str, Any]) -> dict[str, Any]:
        self.received.append(dict(inputs))
        self.outputs_data = {
            "Q_gas": self.q_gas,
            "Q_ch4": self.q_gas * self.ch4,
            "Q_co2": self.q_gas * (1.0 - self.ch4),
            "Q_out": self.q_out,
            "state_out": [1.0] * 41,
            "T_digester": 315.15,
        }
        return self.outputs_data

    def to_dict(self) -> dict[str, Any]:
        return {"component_id": self.component_id, "component_type": "digester"}

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> _GasSource:
        return cls(config["component_id"])


def _chp_plant(capacity_m3: float = 300.0, q_gas: float = 2400.0, P_el_nom: float = _P_EL_FOR_2000_M3) -> BiogasPlant:
    plant = BiogasPlant("gas train")
    plant.add_component(_GasSource("dig", q_gas=q_gas))
    plant.add_component(GasStorage("store", capacity_m3=capacity_m3, initial_fill_fraction=0.1))
    plant.add_component(CHP("chp", P_el_nom=P_el_nom, eta_el=0.4, eta_th=0.45))
    plant.add_component(Flare("flare"))
    plant.add_connection(Connection("dig", "store", "gas"))
    plant.add_connection(Connection("store", "chp", "gas"))
    plant.add_connection(Connection("chp", "flare", "gas"))
    return plant


def _run(plant: BiogasPlant, days: float, dt: float) -> dict[str, float]:
    """Simulate and integrate the gas flows of the plant."""
    stored_before = plant.components["store"].stored_volume_m3
    burnt = 0.0
    for _ in range(round(days / dt)):
        results = plant.step(dt)
        burnt += sum(out.get("Q_gas_consumed", 0.0) for cid, out in results.items() if cid.startswith("chp")) * dt
    store = plant.components["store"]
    return {
        "burnt": burnt,
        "vented": store.outputs_data["cumulative_vented_m3"],
        "stored_change": store.stored_volume_m3 - stored_before,
    }


class TestGasBalance:
    @pytest.mark.parametrize("dt", [1.0 / 24.0, 0.25, 1.0])
    def test_produced_equals_burnt_plus_vented_plus_stored(self, dt: float) -> None:
        totals = _run(_chp_plant(), days=2.0, dt=dt)

        assert totals["burnt"] + totals["vented"] + totals["stored_change"] == pytest.approx(2.0 * 2400.0)

    def test_gas_burnt_does_not_depend_on_the_step_length(self) -> None:
        hourly = _run(_chp_plant(), days=2.0, dt=1.0 / 24.0)
        daily = _run(_chp_plant(), days=2.0, dt=1.0)

        assert hourly["burnt"] == pytest.approx(4000.0, rel=0.02)
        assert daily["burnt"] == pytest.approx(hourly["burnt"], rel=0.02)

    def test_a_storage_smaller_than_one_step_of_production_still_feeds_the_chp(self) -> None:
        """300 m³ of storage, 2400 m³ produced per daily step: the CHP runs at full load."""
        plant = _chp_plant(capacity_m3=300.0)

        results = plant.step(1.0)

        assert results["chp"]["P_el"] == pytest.approx(_P_EL_FOR_2000_M3)
        assert results["store"]["vented_volume_m3"] == pytest.approx(30.0 + 2400.0 - 2000.0 - 300.0)


class TestFlare:
    def test_vented_gas_reaches_the_flare_of_the_chp(self) -> None:
        plant = _chp_plant()

        results = plant.step(1.0)

        vented = results["store"]["vented_volume_m3"]
        assert vented > 0.0
        assert results["flare"]["vented_volume_m3"] == pytest.approx(vented)
        assert results["flare"]["CH4_slip_m3"] == pytest.approx(vented * 0.6 * 0.02)

    def test_a_flare_on_the_storage_takes_precedence(self) -> None:
        plant = _chp_plant()
        plant.add_component(Flare("store_flare"))
        plant.add_connection(Connection("store", "store_flare", "gas"))

        results = plant.step(1.0)

        assert results["store_flare"]["vented_volume_m3"] == pytest.approx(results["store"]["vented_volume_m3"])
        assert results["flare"]["vented_volume_m3"] == 0.0

    def test_without_surplus_the_flare_stays_cold(self) -> None:
        plant = _chp_plant(q_gas=1000.0)

        results = plant.step(1.0 / 24.0)

        assert results["flare"]["vented_volume_m3"] == 0.0

    def test_storage_without_a_flare_just_vents(self) -> None:
        plant = BiogasPlant("no flare")
        plant.add_component(_GasSource("dig"))
        plant.add_component(GasStorage("store", capacity_m3=100.0))
        plant.add_connection(Connection("dig", "store", "gas"))

        results = plant.step(1.0)

        assert results["store"]["vented_volume_m3"] > 0.0


class TestConsumers:
    def test_a_chp_is_stepped_once_per_plant_step(self) -> None:
        plant = _chp_plant()

        for _ in range(48):
            plant.step(1.0 / 24.0)

        assert plant.components["chp"].state["operating_hours"] == pytest.approx(48.0)

    def test_a_chp_without_gas_collects_no_operating_hours(self) -> None:
        plant = _chp_plant(q_gas=0.0)
        plant.components["store"].stored_volume_m3 = 0.0

        results = plant.step(1.0)

        assert results["chp"]["P_el"] == 0.0
        assert plant.components["chp"].state["operating_hours"] == 0.0

    def test_two_chps_share_one_storage_in_proportion_to_their_demand(self) -> None:
        plant = _chp_plant(q_gas=1500.0, P_el_nom=_P_EL_FOR_2000_M3)
        plant.components["store"].stored_volume_m3 = 0.0
        plant.add_component(CHP("chp2", P_el_nom=_P_EL_FOR_2000_M3 / 2.0, eta_el=0.4, eta_th=0.45))
        plant.add_connection(Connection("store", "chp2", "gas"))

        results = plant.step(1.0)

        # 1500 m³/d for a demand of 2000 + 1000 m³/d: both run at half load.
        assert results["chp"]["Q_gas_consumed"] == pytest.approx(1000.0)
        assert results["chp2"]["Q_gas_consumed"] == pytest.approx(500.0)
        assert results["store"]["Q_gas_supplied_m3_per_day"] == pytest.approx(1500.0)

    def test_upgrading_overflow_and_storage_vent_go_to_its_flare(self) -> None:
        plant = BiogasPlant("upgrading")
        plant.add_component(_GasSource("dig", q_gas=2400.0))
        plant.add_component(GasStorage("store", capacity_m3=300.0, initial_fill_fraction=0.0))
        plant.add_component(BiogasUpgrading("bgaa", capacity_m3h=50.0))
        plant.add_component(Flare("flare"))
        plant.add_connection(Connection("dig", "store", "gas"))
        plant.add_connection(Connection("store", "bgaa", "gas"))
        plant.add_connection(Connection("bgaa", "flare", "gas"))

        results = plant.step(1.0)

        assert results["bgaa"]["utilization"] == pytest.approx(1.0)
        # 2400 produced, 1200 upgraded, 300 stored, the rest flared.
        assert results["flare"]["vented_volume_m3"] == pytest.approx(900.0)

    def test_upgrading_fed_beyond_its_capacity_passes_the_surplus_to_the_flare(self) -> None:
        plant = BiogasPlant("upgrading overflow")
        plant.add_component(_GasSource("dig", q_gas=0.0))
        plant.add_component(GasStorage("store", capacity_m3=5000.0, initial_fill_fraction=1.0))
        bgaa = BiogasUpgrading("bgaa", capacity_m3h=50.0)
        plant.add_component(bgaa)
        plant.add_component(Flare("flare"))
        plant.add_connection(Connection("dig", "store", "gas"))
        plant.add_connection(Connection("store", "bgaa", "gas"))
        plant.add_connection(Connection("bgaa", "flare", "gas"))
        # The unit loses capacity after the storage was asked for the full amount.
        original_step = bgaa.step

        def derated_step(t, dt, inputs):
            bgaa.capacity_m3h = 25.0
            return original_step(t, dt, inputs)

        bgaa.step = derated_step

        results = plant.step(1.0)

        assert results["flare"]["vented_volume_m3"] == pytest.approx(600.0)


class TestBoiler:
    @staticmethod
    def _plant(fuel_type: str, stored_fraction: float, P_th_nom: float = 500.0) -> BiogasPlant:
        plant = BiogasPlant("boiler")
        plant.add_component(_GasSource("dig", q_gas=0.0))
        plant.add_component(GasStorage("store", capacity_m3=1000.0, initial_fill_fraction=stored_fraction))
        # 2 kW/K against 15 °C ambient at 42 °C: 54 kW of losses, no CHP heat.
        plant.add_component(HeatingSystem("heat", target_temperature=315.15, heat_loss_coefficient=2.0))
        plant.add_component(Boiler("boiler", P_th_nom=P_th_nom, fuel_type=fuel_type))
        plant.add_connection(Connection("dig", "store", "gas"))
        plant.add_connection(Connection("dig", "heat", "control"))
        plant.add_connection(Connection("heat", "boiler", "heat"))
        plant.add_connection(Connection("store", "boiler", "gas"))
        return plant

    def test_boiler_covers_the_auxiliary_heat_in_the_same_step(self) -> None:
        plant = self._plant("biogas", stored_fraction=0.5)

        results = plant.step(1.0 / 24.0)

        assert results["heat"]["P_aux_heat"] == pytest.approx(54.0)
        assert results["boiler"]["P_th_supplied"] == pytest.approx(54.0)
        assert results["boiler"]["P_th_unmet"] == 0.0

    def test_boiler_gas_is_taken_out_of_the_storage(self) -> None:
        plant = self._plant("biogas", stored_fraction=0.5)

        results = plant.step(1.0)

        burnt = results["boiler"]["Q_gas_consumed_m3_per_day"]
        assert burnt > 0.0
        assert plant.components["store"].stored_volume_m3 == pytest.approx(500.0 - burnt)
        assert results["store"]["stored_volume_m3"] == pytest.approx(500.0 - burnt)

    def test_biogas_boiler_is_limited_by_the_gas_in_the_storage(self) -> None:
        plant = self._plant("biogas", stored_fraction=0.01)  # 10 m³

        results = plant.step(1.0)

        assert results["boiler"]["Q_gas_consumed_m3_per_day"] == pytest.approx(10.0)
        assert results["boiler"]["P_th_unmet"] > 0.0
        assert plant.components["store"].stored_volume_m3 == pytest.approx(0.0)

    def test_empty_storage_supplies_no_gas(self) -> None:
        plant = self._plant("biogas", stored_fraction=0.0)

        results = plant.step(1.0)

        assert results["boiler"]["P_th_supplied"] == 0.0
        assert results["boiler"]["P_th_unmet"] == pytest.approx(54.0)

    def test_demand_above_the_nominal_output_is_reported_as_unmet(self) -> None:
        plant = self._plant("natural_gas", stored_fraction=0.5, P_th_nom=40.0)

        results = plant.step(1.0 / 24.0)

        assert results["boiler"]["P_th_supplied"] == pytest.approx(40.0)
        assert results["boiler"]["P_th_unmet"] == pytest.approx(14.0)


class TestConnections:
    def test_a_duplicate_connection_is_ignored_with_a_warning(self) -> None:
        plant = _chp_plant()

        with pytest.warns(UserWarning, match="already exists"):
            plant.add_connection(Connection("dig", "store", "gas"))

        assert len([c for c in plant.connections if c.from_component == "dig"]) == 1
        # The production is counted once.
        assert plant.step(1.0)["store"]["vented_volume_m3"] == pytest.approx(30.0 + 2400.0 - 2000.0 - 300.0)

    def test_untyped_connections_get_the_only_type_that_fits(self) -> None:
        plant = BiogasPlant("untyped")
        for component in (
            _GasSource("d1"),
            _GasSource("d2"),
            GasStorage("store"),
            CHP("chp"),
            HeatingSystem("heat"),
            Pump("pump"),
        ):
            plant.add_component(component)
        for source, target in (("d1", "d2"), ("d1", "store"), ("store", "chp"), ("chp", "heat"), ("d2", "pump")):
            plant.add_connection(Connection(source, target))
        plant.add_connection(Connection("pump", "heat"))

        types = {(c.from_component, c.to_component): c.connection_type for c in plant.connections}
        assert types[("d1", "d2")] == "liquid"
        assert types[("d2", "pump")] == "liquid"
        assert types[("d1", "store")] == "gas"
        assert types[("store", "chp")] == "gas"
        assert types[("chp", "heat")] == "heat"
        assert types[("pump", "heat")] == "default"

    def test_an_untyped_digester_cascade_carries_the_flow(self) -> None:
        plant = BiogasPlant("cascade")
        plant.add_component(_GasSource("d1", q_out=50.0))
        plant.add_component(_GasSource("d2"))
        plant.add_connection(Connection("d1", "d2"))

        plant.step(1.0)

        assert plant.components["d2"].received[-1]["Q_in"] == pytest.approx(50.0)

    def test_a_pump_between_two_digesters_passes_flow_and_state_on(self) -> None:
        plant = BiogasPlant("pumped cascade")
        plant.add_component(_GasSource("d1", q_out=50.0))
        plant.add_component(Pump("pump", Q_nom=60.0))
        plant.add_component(_GasSource("d2"))
        plant.add_connection(Connection("d1", "pump", "liquid"))
        plant.add_connection(Connection("pump", "d2", "liquid"))

        results = plant.step(1.0)

        assert results["pump"]["Q_out"] == pytest.approx(50.0)
        assert results["pump"]["P_consumed"] > 0.0
        received = plant.components["d2"].received[-1]
        assert received["Q_in"] == pytest.approx(50.0)
        assert received["state_in"] == [1.0] * 41


class TestSimulate:
    def test_no_step_is_lost_to_floating_point(self) -> None:
        plant = _chp_plant()

        results = plant.simulate(duration=0.3, dt=0.1)

        assert len(results) == 3
        assert plant.simulation_time == pytest.approx(0.3)

    def test_a_duration_that_is_no_multiple_of_dt_ends_with_a_short_step(self) -> None:
        plant = _chp_plant()

        plant.simulate(duration=0.25, dt=0.1)

        assert plant.simulation_time == pytest.approx(0.25)

    def test_results_are_saved_on_the_interval(self) -> None:
        plant = _chp_plant()

        results = plant.simulate(duration=3.0, dt=1.0 / 24.0, save_interval=1.0)

        assert [entry["time"] for entry in results] == pytest.approx([1.0 / 24.0, 1.0, 2.0, 3.0])

    def test_a_non_positive_step_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="dt must be positive"):
            _chp_plant().simulate(duration=1.0, dt=0.0)
