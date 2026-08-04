"""Unit and integration tests for the BiogasUpgrading component (BGAA).

Covers the CH4 mass balance of a single ``step``, the capacity/overflow split,
serialization, and the plant-level Pass 4 that couples gas storages, the BGAA
and its emergency flare.
"""

from __future__ import annotations

import pytest

from pyadm1 import BiogasPlant, Feedstock
from pyadm1.components.base import ComponentType
from pyadm1.components.energy.biogas_upgrading import BiogasUpgrading
from pyadm1.components.energy.flare import Flare
from pyadm1.components.energy.gas_storage import GasStorage
from pyadm1.components.registry import ComponentRegistry
from pyadm1.configurator.connection_manager import Connection
from pyadm1.configurator.plant_configurator import PlantConfigurator


@pytest.fixture
def bgaa() -> BiogasUpgrading:
    """Default 500 m³/h unit: 55 % CH4 in, 97 % CH4 out, 98 % recovery."""
    return BiogasUpgrading("bgaa_1")


class TestConstruction:
    def test_registers_as_upgrading_component(self, bgaa: BiogasUpgrading) -> None:
        assert bgaa.component_type is ComponentType.UPGRADING
        assert bgaa.component_id == "bgaa_1"

    def test_capacity_converts_hourly_rating_to_daily(self) -> None:
        unit = BiogasUpgrading("b", capacity_m3h=250.0)
        assert unit.capacity_m3_per_day == pytest.approx(6000.0)

    def test_is_initialized_with_zeroed_outputs(self, bgaa: BiogasUpgrading) -> None:
        assert bgaa._initialized
        assert all(value == 0.0 for value in bgaa.outputs_data.values())


class TestMethaneBalance:
    """The core process: CH4 is concentrated from raw-gas to grid quality."""

    def test_biomethane_volume_follows_ch4_concentration_ratio(self, bgaa: BiogasUpgrading) -> None:
        # 1000 m³/d raw gas at 55 % CH4 -> 550 m³ CH4, 98 % recovered = 539 m³ CH4,
        # delivered at 97 % grade -> 539 / 0.97 m³ biomethane.
        out = bgaa.step(t=0.0, dt=1.0, inputs={"Q_gas_in_m3_per_day": 1000.0})

        assert out["Q_biomethane_m3_per_day"] == pytest.approx(1000.0 * 0.55 * 0.98 / 0.97)
        # Upgrading strips CO2, so the product stream is much smaller than the feed.
        assert out["Q_biomethane_m3_per_day"] < 1000.0

    def test_ch4_leaving_in_the_product_equals_recovered_ch4(self, bgaa: BiogasUpgrading) -> None:
        """CH4 bookkeeping must close: product CH4 = feed CH4 x recovery."""
        Q_in = 1000.0
        out = bgaa.step(t=0.0, dt=1.0, inputs={"Q_gas_in_m3_per_day": Q_in})

        ch4_in_product = out["Q_biomethane_m3_per_day"] * bgaa.ch4_content_out
        assert ch4_in_product == pytest.approx(Q_in * 0.55 * 0.98)

    def test_volume_balance_closes_product_plus_offgas_equals_feed(self, bgaa: BiogasUpgrading) -> None:
        out = bgaa.step(t=0.0, dt=1.0, inputs={"Q_gas_in_m3_per_day": 1000.0})

        assert out["Q_biomethane_m3_per_day"] + out["Q_offgas_m3_per_day"] == pytest.approx(1000.0)
        assert out["Q_gas_out_m3_per_day"] == 0.0  # below capacity -> no overflow

    def test_measured_ch4_fraction_overrides_the_design_value(self, bgaa: BiogasUpgrading) -> None:
        """Raw-gas quality varies; the ``CH4_fraction`` input must win."""
        rich = bgaa.step(t=0.0, dt=1.0, inputs={"Q_gas_in_m3_per_day": 1000.0, "CH4_fraction": 0.62})
        bgaa.initialize()
        lean = bgaa.step(t=0.0, dt=1.0, inputs={"Q_gas_in_m3_per_day": 1000.0, "CH4_fraction": 0.48})

        assert rich["Q_biomethane_m3_per_day"] == pytest.approx(1000.0 * 0.62 * 0.98 / 0.97)
        assert lean["Q_biomethane_m3_per_day"] == pytest.approx(1000.0 * 0.48 * 0.98 / 0.97)
        # Richer raw gas yields more biomethane and less CO2 reject.
        assert rich["Q_biomethane_m3_per_day"] > lean["Q_biomethane_m3_per_day"]
        assert rich["Q_offgas_m3_per_day"] < lean["Q_offgas_m3_per_day"]

    def test_offgas_cannot_go_negative_for_near_pure_feed_gas(self) -> None:
        """A feed richer than the product grade would give Q_bm > Q_in; offgas floors at 0."""
        unit = BiogasUpgrading("pure", ch4_recovery=0.99, ch4_content_out=0.97)

        out = unit.step(t=0.0, dt=1.0, inputs={"Q_gas_in_m3_per_day": 100.0, "CH4_fraction": 0.99})

        assert out["Q_biomethane_m3_per_day"] > 100.0
        assert out["Q_offgas_m3_per_day"] == 0.0

    def test_zero_inflow_produces_zero_outputs(self, bgaa: BiogasUpgrading) -> None:
        out = bgaa.step(t=0.0, dt=1.0, inputs={})

        assert out["Q_biomethane_m3_per_day"] == 0.0
        assert out["Q_offgas_m3_per_day"] == 0.0
        assert out["Q_gas_out_m3_per_day"] == 0.0
        assert out["utilization"] == 0.0

    def test_degenerate_product_grade_does_not_divide_by_zero(self) -> None:
        unit = BiogasUpgrading("degenerate", ch4_content_out=0.0)

        out = unit.step(t=0.0, dt=1.0, inputs={"Q_gas_in_m3_per_day": 1.0})

        assert out["Q_biomethane_m3_per_day"] > 0.0  # finite, not NaN/inf


class TestCapacityAndOverflow:
    def test_throughput_is_capped_and_surplus_is_passed_downstream(self) -> None:
        unit = BiogasUpgrading("small", capacity_m3h=10.0)  # 240 m³/d
        out = unit.step(t=0.0, dt=1.0, inputs={"Q_gas_in_m3_per_day": 400.0})

        assert out["Q_gas_out_m3_per_day"] == pytest.approx(160.0)
        # Only the 240 m³/d actually processed are upgraded.
        assert out["Q_biomethane_m3_per_day"] == pytest.approx(240.0 * 0.55 * 0.98 / 0.97)
        assert out["Q_biomethane_m3_per_day"] + out["Q_offgas_m3_per_day"] == pytest.approx(240.0)

    def test_utilization_reports_load_and_saturates_at_full_capacity(self) -> None:
        unit = BiogasUpgrading("util", capacity_m3h=10.0)  # 240 m³/d

        assert unit.step(0.0, 1.0, {"Q_gas_in_m3_per_day": 120.0})["utilization"] == pytest.approx(0.5)
        unit.initialize()
        assert unit.step(0.0, 1.0, {"Q_gas_in_m3_per_day": 240.0})["utilization"] == pytest.approx(1.0)
        unit.initialize()
        assert unit.step(0.0, 1.0, {"Q_gas_in_m3_per_day": 1000.0})["utilization"] == pytest.approx(1.0)

    def test_only_processed_gas_counts_towards_cumulative_input(self) -> None:
        unit = BiogasUpgrading("cum_cap", capacity_m3h=10.0)  # 240 m³/d

        out = unit.step(t=0.0, dt=1.0, inputs={"Q_gas_in_m3_per_day": 400.0})

        assert out["cumulative_gas_in_m3"] == pytest.approx(240.0)
        assert out["cumulative_overflow_m3"] == pytest.approx(160.0)


class TestCumulativeTotals:
    def test_totals_integrate_flows_over_sub_daily_timesteps(self, bgaa: BiogasUpgrading) -> None:
        dt = 1.0 / 24.0
        for i in range(24):
            out = bgaa.step(t=i * dt, dt=dt, inputs={"Q_gas_in_m3_per_day": 1200.0})

        # 24 hourly steps at a constant rate integrate to one full day of flow.
        assert out["cumulative_gas_in_m3"] == pytest.approx(1200.0)
        assert out["cumulative_biomethane_m3"] == pytest.approx(1200.0 * 0.55 * 0.98 / 0.97)
        assert out["cumulative_offgas_m3"] == pytest.approx(1200.0 - 1200.0 * 0.55 * 0.98 / 0.97)
        assert out["cumulative_overflow_m3"] == 0.0

    def test_initialize_without_state_resets_the_counters(self, bgaa: BiogasUpgrading) -> None:
        bgaa.step(t=0.0, dt=1.0, inputs={"Q_gas_in_m3_per_day": 1000.0})
        assert bgaa._cum_biomethane_m3 > 0.0

        bgaa.initialize()

        assert bgaa._cum_gas_in_m3 == 0.0
        assert bgaa._cum_biomethane_m3 == 0.0
        assert bgaa.outputs_data["cumulative_biomethane_m3"] == 0.0

    def test_initialize_with_state_restores_the_counters(self, bgaa: BiogasUpgrading) -> None:
        bgaa.initialize(
            {
                "cumulative_gas_in_m3": 100.0,
                "cumulative_biomethane_m3": 55.0,
                "cumulative_offgas_m3": 45.0,
                "cumulative_overflow_m3": 7.0,
            }
        )

        assert bgaa._cum_gas_in_m3 == 100.0
        assert bgaa._cum_biomethane_m3 == 55.0
        assert bgaa._cum_offgas_m3 == 45.0
        assert bgaa._cum_overflow_m3 == 7.0

    def test_restored_counters_continue_accumulating(self, bgaa: BiogasUpgrading) -> None:
        bgaa.initialize({"cumulative_gas_in_m3": 100.0})

        out = bgaa.step(t=0.0, dt=1.0, inputs={"Q_gas_in_m3_per_day": 50.0})

        assert out["cumulative_gas_in_m3"] == pytest.approx(150.0)


class TestSerialization:
    def test_to_dict_captures_configuration_and_state(self) -> None:
        unit = BiogasUpgrading("serde", capacity_m3h=300.0, ch4_recovery=0.95, name="BGAA Nord")
        unit.add_input("stor_1")
        unit.add_output("bgaa_flare")
        unit.step(t=0.0, dt=1.0, inputs={"Q_gas_in_m3_per_day": 1000.0})

        data = unit.to_dict()

        assert data["component_id"] == "serde"
        assert data["component_type"] == "upgrading"
        assert data["name"] == "BGAA Nord"
        assert data["capacity_m3h"] == 300.0
        assert data["ch4_recovery"] == 0.95
        assert data["inputs"] == ["stor_1"]
        assert data["outputs"] == ["bgaa_flare"]
        assert data["cumulative_biomethane_m3"] > 0.0

    def test_round_trip_preserves_configuration_and_totals(self) -> None:
        unit = BiogasUpgrading("rt", capacity_m3h=42.0, ch4_recovery=0.93, ch4_content_in=0.6, ch4_content_out=0.96)
        unit.step(t=0.0, dt=1.0, inputs={"Q_gas_in_m3_per_day": 500.0})

        clone = BiogasUpgrading.from_dict(unit.to_dict())

        assert clone.capacity_m3h == 42.0
        assert clone.ch4_recovery == 0.93
        assert clone.ch4_content_in == 0.6
        assert clone.ch4_content_out == 0.96
        assert clone._cum_biomethane_m3 == pytest.approx(unit._cum_biomethane_m3)
        assert clone._cum_overflow_m3 == pytest.approx(unit._cum_overflow_m3)

    def test_restored_unit_reproduces_the_original_step(self) -> None:
        unit = BiogasUpgrading("repro", capacity_m3h=20.0, ch4_recovery=0.9)
        clone = BiogasUpgrading.from_dict(unit.to_dict())

        inputs = {"Q_gas_in_m3_per_day": 600.0, "CH4_fraction": 0.53}
        assert clone.step(0.0, 1.0, inputs) == unit.step(0.0, 1.0, inputs)

    def test_from_dict_falls_back_to_defaults_for_a_minimal_config(self) -> None:
        clone = BiogasUpgrading.from_dict({"component_id": "minimal"})

        assert clone.capacity_m3h == 500.0
        assert clone.ch4_recovery == 0.98
        assert clone.ch4_content_out == 0.97
        assert clone._cum_gas_in_m3 == 0.0

    def test_registry_can_build_the_component_by_name(self) -> None:
        registry = ComponentRegistry()

        unit = registry.create("BiogasUpgrading", "via_registry", capacity_m3h=123.0)

        assert isinstance(unit, BiogasUpgrading)
        assert unit.capacity_m3_per_day == pytest.approx(123.0 * 24.0)


class TestPlantConfiguratorIntegration:
    @pytest.fixture
    def cfg(self) -> PlantConfigurator:
        feedstock = Feedstock(["maize_silage_milk_ripeness", "swine_manure"], feeding_freq=24, total_simtime=5)
        return PlantConfigurator(BiogasPlant("BGAA Plant"), feedstock)

    def test_add_bgaa_also_creates_and_wires_its_emergency_flare(self, cfg: PlantConfigurator) -> None:
        unit = cfg.add_bgaa("bgaa1", capacity_m3h=300.0, ch4_recovery=0.96)

        assert isinstance(unit, BiogasUpgrading)
        assert cfg.plant.components["bgaa1"] is unit
        assert unit.ch4_recovery == 0.96
        assert isinstance(cfg.plant.components["bgaa1_flare"], Flare)
        assert any(
            c.from_component == "bgaa1" and c.to_component == "bgaa1_flare" and c.connection_type == "gas"
            for c in cfg.plant.connections
        )

    def test_auto_connect_routes_the_digester_gas_storage_into_the_bgaa(self, cfg: PlantConfigurator) -> None:
        cfg.add_digester("d1")
        cfg.add_bgaa("bgaa1")

        cfg.auto_connect_digester_to_bgaa("d1", "bgaa1")

        assert any(
            c.from_component == "d1_storage" and c.to_component == "bgaa1" and c.connection_type == "gas"
            for c in cfg.plant.connections
        )

    def test_auto_connect_reports_a_missing_gas_storage(self, cfg: PlantConfigurator) -> None:
        cfg.add_bgaa("bgaa1")

        with pytest.raises(ValueError, match="Gas storage 'ghost_storage' not found"):
            cfg.auto_connect_digester_to_bgaa("ghost", "bgaa1")


class TestPlantStepIntegration:
    """Pass 4 of ``BiogasPlant.step``: storage -> BGAA -> flare."""

    @staticmethod
    def _plant(capacity_m3h: float = 10.0, n_storages: int = 1, fill: float = 1.0) -> BiogasPlant:
        plant = BiogasPlant("BGAA Step")
        plant.add_component(BiogasUpgrading("bgaa", capacity_m3h=capacity_m3h))
        plant.add_component(Flare("bgaa_flare"))
        plant.add_connection(Connection("bgaa", "bgaa_flare", "gas"))
        for i in range(n_storages):
            sid = f"stor{i}"
            plant.add_component(GasStorage(sid, capacity_m3=5000.0, initial_fill_fraction=fill))
            plant.add_connection(Connection(sid, "bgaa", "gas"))
        return plant

    def test_bgaa_draws_gas_from_storage_and_delivers_biomethane(self) -> None:
        plant = self._plant(capacity_m3h=10.0)  # demand 240 m³/d
        storage = plant.components["stor0"]
        before = storage.stored_volume_m3

        results = plant.step(dt=1.0)

        supplied = results["stor0"]["Q_gas_supplied_m3_per_day"]
        assert supplied > 0.0
        # Regression guard: without Pass 4 the storage would only ever fill up.
        assert storage.stored_volume_m3 == pytest.approx(before - supplied)
        assert results["bgaa"]["Q_biomethane_m3_per_day"] == pytest.approx(supplied * 0.55 * 0.98 / 0.97)

    def test_demand_is_capped_at_capacity_so_the_flare_stays_idle(self) -> None:
        plant = self._plant(capacity_m3h=10.0)

        results = plant.step(dt=1.0)

        assert results["stor0"]["Q_gas_supplied_m3_per_day"] <= plant.components["bgaa"].capacity_m3_per_day + 1e-9
        assert results["bgaa"]["Q_gas_out_m3_per_day"] == pytest.approx(0.0)
        assert results["bgaa_flare"]["vented_volume_m3"] == pytest.approx(0.0)

    def test_demand_is_split_evenly_across_connected_storages(self) -> None:
        plant = self._plant(capacity_m3h=10.0, n_storages=2)

        results = plant.step(dt=1.0)

        s0 = results["stor0"]["Q_gas_supplied_m3_per_day"]
        s1 = results["stor1"]["Q_gas_supplied_m3_per_day"]
        assert s0 == pytest.approx(s1)
        assert results["bgaa"]["Q_biomethane_m3_per_day"] == pytest.approx((s0 + s1) * 0.55 * 0.98 / 0.97)

    def test_an_empty_storage_yields_no_biomethane(self) -> None:
        plant = self._plant(capacity_m3h=10.0, fill=0.0)

        results = plant.step(dt=1.0)

        assert results["stor0"]["Q_gas_supplied_m3_per_day"] == pytest.approx(0.0)
        assert results["bgaa"]["Q_biomethane_m3_per_day"] == pytest.approx(0.0)

    def test_an_unconnected_bgaa_is_left_at_its_idle_state(self) -> None:
        plant = BiogasPlant("Lonely BGAA")
        plant.add_component(BiogasUpgrading("bgaa", capacity_m3h=10.0))

        plant.step(dt=1.0)

        assert plant.components["bgaa"].outputs_data["Q_biomethane_m3_per_day"] == 0.0
        assert plant.components["bgaa"].outputs_data["cumulative_gas_in_m3"] == 0.0

    def test_totals_accumulate_across_several_plant_steps(self) -> None:
        plant = self._plant(capacity_m3h=10.0)
        unit = plant.components["bgaa"]

        for _ in range(5):
            results = plant.step(dt=1.0)

        assert results["bgaa"]["cumulative_biomethane_m3"] == pytest.approx(unit._cum_biomethane_m3)
        assert unit._cum_biomethane_m3 > results["bgaa"]["Q_biomethane_m3_per_day"]
