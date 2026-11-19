"""Feed chain storage -> feeder -> digester, and the plant's electricity balance.

Storage and feeder could not be wired: the feeder asked for
``substrate_available`` while the storage offered ``available_mass``, so the
feeder drew from an infinite stock; the storage waited for ``withdrawal_rate``
while the feeder reported ``Q_actual``, so it never emptied; and neither reached
the digester, whose feed came straight from the user.

Electricity: only some consumers reported ``P_consumed`` and nothing summed
them, so plant variants could only be compared by gross CHP output.
"""

from __future__ import annotations

import pytest

from pyadm1 import Feedstock
from pyadm1.components.energy.biogas_upgrading import BiogasUpgrading
from pyadm1.components.energy.chp import CHP
from pyadm1.components.energy.gas_storage import GasStorage
from pyadm1.components.feeding.feeder import Feeder
from pyadm1.components.feeding.substrate_storage import SubstrateStorage
from pyadm1.components.mechanical.pump import Pump
from pyadm1.configurator.connection_manager import Connection
from pyadm1.configurator.plant_builder import BiogasPlant
from pyadm1.configurator.plant_configurator import PlantConfigurator

SUBSTRATES = ["maize_silage_milk_ripeness", "swine_manure"]
Q = [10.0, 5.0] + [0.0] * 8
DT = 1.0 / 24.0


@pytest.fixture
def feedstock() -> Feedstock:
    return Feedstock(SUBSTRATES, feeding_freq=24, total_simtime=5)


def _plant(feedstock: Feedstock) -> tuple[BiogasPlant, PlantConfigurator]:
    plant = BiogasPlant("p")
    cfg = PlantConfigurator(plant, feedstock)
    cfg.add_digester("dig", V_liq=1000.0, V_gas=150.0, Q_substrates=list(Q))
    return plant, cfg


class TestStorageFromFeedstock:
    def test_properties_come_from_the_substrate_definition(self, feedstock: Feedstock) -> None:
        _, cfg = _plant(feedstock)

        silo = cfg.add_substrate_storage("silo", "maize_silage_milk_ripeness", capacity=800.0)

        params = feedstock._subs[0]
        assert silo.density == pytest.approx(feedstock._densities[0])
        assert silo.dry_matter == pytest.approx(params.TS / 10.0)
        assert silo.vs_content == pytest.approx((1.0 - params.fRA) * 100.0)
        assert silo.unlimited is True  # no stock given: always enough substrate

    def test_substrate_by_index_and_unknown_substrate(self, feedstock: Feedstock) -> None:
        _, cfg = _plant(feedstock)

        assert cfg.add_substrate_storage("tank", 1).density == pytest.approx(feedstock._densities[1])
        with pytest.raises(ValueError, match="grass"):
            cfg.add_substrate_storage("x", "grass")
        with pytest.raises(ValueError, match="index 7"):
            cfg.add_substrate_storage("y", 7)

    def test_measured_properties_round_trip(self) -> None:
        silo = SubstrateStorage("s", density=712.0, dry_matter=33.0, vs_content=96.0)

        restored = SubstrateStorage.from_dict(silo.to_dict())

        assert (restored.density, restored.dry_matter, restored.vs_content) == (712.0, 33.0, 96.0)

    def test_given_degradation_rate_survives_a_temperature_input(self) -> None:
        given = SubstrateStorage("s", degradation_rate=0.01)
        estimated = SubstrateStorage("e")
        before = estimated.degradation_rate

        given.step(0.0, DT, {"temperature": 303.15})
        estimated.step(0.0, DT, {"temperature": 303.15})

        assert given.degradation_rate == 0.01
        assert estimated.degradation_rate > before
        assert SubstrateStorage.from_dict(estimated.to_dict())._degradation_rate_given is False


class TestFeeder:
    def test_standing_setpoint_and_input_override(self) -> None:
        feeder = Feeder("f", Q_setpoint=8.0, enable_dosing_noise=False)

        assert feeder.step(0.0, DT, {})["Q_actual"] == pytest.approx(8.0)
        assert feeder.step(0.0, DT, {"Q_setpoint": 3.0})["Q_actual"] == pytest.approx(3.0)
        assert Feeder("idle", enable_dosing_noise=False).step(0.0, DT, {})["Q_actual"] == 0.0

    def test_specific_energy_per_tonne(self) -> None:
        feeder = Feeder("f", Q_setpoint=12.0, specific_energy=0.92, enable_dosing_noise=False)

        out = feeder.step(0.0, DT, {})

        assert out["P_consumed"] == pytest.approx(0.92 * 12.0 / 24.0)

    def test_volume_flow_is_converted_to_mass(self) -> None:
        feeder = Feeder("f", Q_setpoint=10.0, flow_to_mass_factor=0.7, specific_energy=1.0, enable_dosing_noise=False)

        out = feeder.step(0.0, DT, {})

        assert out["mass_flow_t_per_day"] == pytest.approx(7.0)
        assert out["P_consumed"] == pytest.approx(7.0 / 24.0)

    def test_new_parameters_round_trip(self) -> None:
        feeder = Feeder("f", Q_setpoint=8.0, substrate_index=1, flow_to_mass_factor=0.7, specific_energy=0.92)

        restored = Feeder.from_dict(feeder.to_dict())

        assert (restored.Q_setpoint, restored.substrate_index) == (8.0, 1)
        assert (restored.flow_to_mass_factor, restored.specific_energy) == (0.7, 0.92)


class TestFeedChain:
    def test_setpoint_defaults_to_the_digester_rate(self, feedstock: Feedstock) -> None:
        _, cfg = _plant(feedstock)

        feeder = cfg.add_feeder("feed", "dig", "swine_manure", enable_dosing_noise=False)

        assert feeder.substrate_index == 1
        assert feeder.Q_setpoint == 5.0
        assert feeder.flow_to_mass_factor == pytest.approx(1.0)  # ADM1da mass convention: Q in t/d

    def test_volume_convention_uses_the_density(self) -> None:
        fs = Feedstock(SUBSTRATES, feeding_freq=24, total_simtime=5, simba_q_convention=False)
        _, cfg = _plant(fs)

        feeder = cfg.add_feeder("feed", "dig", 0)

        assert feeder.flow_to_mass_factor == pytest.approx(fs._densities[0] / 1000.0)

    def test_storage_is_drawn_down_by_the_dosed_flow(self, feedstock: Feedstock) -> None:
        plant, cfg = _plant(feedstock)
        silo = cfg.add_substrate_storage("silo", 0, capacity=500.0, initial_level=500.0, degradation_rate=1e-12)
        cfg.add_feeder("feed", "dig", 0, storage_id="silo", enable_dosing_noise=False)

        results = plant.step(DT)

        assert results["feed"]["Q_actual"] == pytest.approx(10.0)
        assert silo.current_level == pytest.approx(500.0 - 10.0 * DT, abs=1e-6)
        assert results["silo"]["withdrawn_this_step"] == pytest.approx(10.0 * DT)

    def test_digester_follows_the_feeder_not_the_configured_rate(self, feedstock: Feedstock) -> None:
        plant, cfg = _plant(feedstock)
        cfg.add_feeder("feed", "dig", 0, Q_setpoint=4.0, enable_dosing_noise=False)

        plant.step(DT)

        digester = plant.components["dig"]
        assert digester.Q_substrates[0] == pytest.approx(4.0)
        assert digester.Q_substrates[1] == 5.0  # slot without a feeder is untouched

    def test_empty_storage_stops_the_feeding(self, feedstock: Feedstock) -> None:
        plant, cfg = _plant(feedstock)
        cfg.add_substrate_storage("silo", 0, capacity=500.0, initial_level=0.1, degradation_rate=1e-12)
        cfg.add_feeder("feed", "dig", 0, storage_id="silo", enable_dosing_noise=False)

        first = plant.step(DT)  # 0.1 t in stock, 10 t/d asked: all of it goes in one step
        second = plant.step(DT)

        assert first["feed"]["Q_actual"] == pytest.approx(0.1 / DT)
        assert second["feed"]["Q_actual"] == 0.0
        assert plant.components["dig"].Q_substrates[0] == 0.0
        assert plant.components["silo"].current_level == pytest.approx(0.0, abs=1e-9)

    def test_missing_digester_or_storage_is_rejected(self, feedstock: Feedstock) -> None:
        _, cfg = _plant(feedstock)

        with pytest.raises(ValueError, match="ghost"):
            cfg.add_feeder("feed", "ghost", 0)
        with pytest.raises(ValueError, match="nosilo"):
            cfg.add_feeder("feed", "dig", 0, storage_id="nosilo")

    def test_chain_survives_save_and_load(self, feedstock: Feedstock, tmp_path) -> None:
        plant, cfg = _plant(feedstock)
        cfg.add_substrate_storage("silo", 0, capacity=500.0, initial_level=500.0, degradation_rate=1e-12)
        cfg.add_feeder("feed", "dig", 0, storage_id="silo", Q_setpoint=6.0, enable_dosing_noise=False)
        path = tmp_path / "plant.json"
        plant.to_json(str(path))

        restored = BiogasPlant.from_json(str(path), feedstock=feedstock)
        restored.step(DT)

        assert restored.components["dig"].Q_substrates[0] == pytest.approx(6.0)
        assert restored.components["silo"].current_level == pytest.approx(500.0 - 6.0 * DT, abs=1e-6)


class TestElectricityBalance:
    def test_chp_reports_gross_own_and_net(self) -> None:
        out = CHP("chp", P_el_nom=500.0, own_consumption_fraction=0.04).step(0.0, DT, {"Q_ch4": 5000.0})

        assert out["P_el"] == 500.0
        assert out["P_consumed"] == pytest.approx(20.0)
        assert out["P_el_net"] == pytest.approx(480.0)

    def test_chp_fraction_round_trips_and_is_validated(self) -> None:
        assert CHP.from_dict(CHP("c", own_consumption_fraction=0.05).to_dict()).own_consumption_fraction == 0.05
        with pytest.raises(ValueError, match="own_consumption_fraction"):
            CHP("c", own_consumption_fraction=1.0)

    def test_upgrading_electricity_demand(self) -> None:
        bgaa = BiogasUpgrading("b", capacity_m3h=500.0, specific_electricity=0.3)

        out = bgaa.step(0.0, DT, {"Q_gas_in_m3_per_day": 2400.0})

        assert out["P_consumed"] == pytest.approx(0.3 * 2400.0 / 24.0)
        assert BiogasUpgrading.from_dict(bgaa.to_dict()).specific_electricity == 0.3

    def test_upgrading_demand_follows_the_treated_not_the_offered_gas(self) -> None:
        bgaa = BiogasUpgrading("b", capacity_m3h=50.0, specific_electricity=0.25)  # 1200 m³/d

        out = bgaa.step(0.0, DT, {"Q_gas_in_m3_per_day": 5000.0})

        assert out["P_consumed"] == pytest.approx(0.25 * 1200.0 / 24.0)

    def test_plant_balance_sums_every_consumer(self) -> None:
        plant = BiogasPlant("p")
        plant.add_component(GasStorage("gs", capacity_m3=50000.0, initial_fill_fraction=0.5))
        plant.add_component(CHP("chp", P_el_nom=500.0, own_consumption_fraction=0.03))
        plant.add_connection(Connection("gs", "chp", "gas"))
        plant.add_component(Pump("pump", Q_nom=100.0, pressure_head=10.0))
        plant.components["pump"].step(0.0, DT, {"Q_setpoint": 100.0})

        plant.step(DT)
        balance = plant.get_electricity_balance()

        assert balance["P_el_produced"] == pytest.approx(500.0)
        assert set(balance["consumers"]) >= {"chp"}
        assert balance["P_el_consumed"] == pytest.approx(sum(balance["consumers"].values()))
        assert balance["P_el_net"] == pytest.approx(balance["P_el_produced"] - balance["P_el_consumed"])
        assert balance["own_consumption_fraction"] == pytest.approx(balance["P_el_consumed"] / 500.0)

    def test_energy_accumulates_and_round_trips(self, tmp_path) -> None:
        plant = BiogasPlant("p")
        plant.add_component(GasStorage("gs", capacity_m3=50000.0, initial_fill_fraction=0.5))
        plant.add_component(CHP("chp", P_el_nom=500.0, own_consumption_fraction=0.03))
        plant.add_connection(Connection("gs", "chp", "gas"))

        for _ in range(24):
            plant.step(DT)

        assert plant.energy_produced_kwh == pytest.approx(500.0 * 24.0)
        assert plant.energy_consumed_kwh == pytest.approx(15.0 * 24.0)
        path = tmp_path / "plant.json"
        plant.to_json(str(path))
        restored = BiogasPlant.from_json(str(path))
        assert restored.get_electricity_balance()["energy_net_kwh"] == pytest.approx(485.0 * 24.0)

    def test_plant_without_a_chp_has_no_fraction(self) -> None:
        assert BiogasPlant("p").get_electricity_balance()["own_consumption_fraction"] == 0.0
