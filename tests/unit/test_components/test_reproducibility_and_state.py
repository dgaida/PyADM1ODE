"""Reproducible noise, state snapshots and step-length independence of components."""

from __future__ import annotations

import pytest

from pyadm1.components.energy.biogas_upgrading import BiogasUpgrading
from pyadm1.components.energy.boiler import Boiler
from pyadm1.components.energy.chp import CHP
from pyadm1.components.energy.flare import Flare
from pyadm1.components.energy.gas_storage import GasStorage
from pyadm1.components.energy.heating import HeatingSystem, _calc_process_heat_kw
from pyadm1.components.feeding.feeder import Feeder
from pyadm1.components.feeding.substrate_storage import SubstrateStorage
from pyadm1.components.mechanical.mixer import Mixer
from pyadm1.components.mechanical.pump import Pump
from pyadm1.components.sensors.gas import GasSensor
from pyadm1.components.sensors.physical import PhysicalSensor


class TestFeederRandomness:
    @staticmethod
    def _flows(feeder: Feeder, n: int = 20) -> list[float]:
        return [feeder.step(i / 24.0, 1.0 / 24.0, {"Q_setpoint": 15.0})["Q_actual"] for i in range(n)]

    def test_two_runs_of_the_same_feeder_give_the_same_flows(self) -> None:
        assert self._flows(Feeder("feeder_1")) == self._flows(Feeder("feeder_1"))

    def test_two_feeders_have_different_noise(self) -> None:
        assert self._flows(Feeder("feeder_1")) != self._flows(Feeder("feeder_2"))

    def test_an_explicit_seed_survives_a_dict_round_trip(self) -> None:
        feeder = Feeder("feeder_1", seed=7)

        restored = Feeder.from_dict(feeder.to_dict())

        assert restored.seed == 7
        assert self._flows(restored) == self._flows(Feeder("feeder_1", seed=7))

    def test_noise_never_doses_more_than_the_storage_holds(self) -> None:
        feeder = Feeder("feeder_1", dosing_accuracy=0.5)

        for i in range(200):
            out = feeder.step(i / 24.0, 1.0 / 24.0, {"Q_setpoint": 15.0, "substrate_available": 0.5})
            assert out["Q_actual"] * (1.0 / 24.0) <= 0.5 + 1e-12

    def test_an_exact_feeder_can_be_configured(self) -> None:
        feeder = Feeder("feeder_1", dosing_accuracy=0.0)

        assert feeder.dosing_accuracy == 0.0
        assert self._flows(feeder, n=5) == [15.0] * 5

    def test_event_counters_are_restored(self) -> None:
        feeder = Feeder("feeder_1")
        feeder.step(0.0, 1.0 / 24.0, {"Q_setpoint": 15.0})
        feeder.n_blockages = 3
        feeder.state["n_blockages"] = 3

        restored = Feeder.from_dict(feeder.to_dict())

        assert restored.n_starts == 1
        assert restored.n_blockages == 3


class TestSubstrateStorage:
    def test_a_loss_free_storage_can_be_configured(self) -> None:
        silo = SubstrateStorage("silo", capacity=1000.0, initial_level=500.0, degradation_rate=0.0)

        silo.step(0.0, 10.0, {})

        assert silo.degradation_rate == 0.0
        assert silo.state["current_level"] == pytest.approx(500.0)


class TestSensorRandomness:
    @staticmethod
    def _readings(sensor: PhysicalSensor, start: int = 0, n: int = 10) -> list[float]:
        return [sensor.step((start + i) / 24.0, 1.0 / 24.0, {"pH": 7.2})["measurement"] for i in range(n)]

    def test_two_runs_of_the_same_sensor_give_the_same_readings(self) -> None:
        first = PhysicalSensor("ph_1", sensor_type="pH", measurement_noise=0.05, accuracy=0.1)
        second = PhysicalSensor("ph_1", sensor_type="pH", measurement_noise=0.05, accuracy=0.1)

        assert first.calibration_offset == second.calibration_offset
        assert self._readings(first) == self._readings(second)

    def test_a_restored_sensor_continues_its_noise_sequence(self) -> None:
        reference = PhysicalSensor("ph_1", sensor_type="pH", measurement_noise=0.05, accuracy=0.1)
        expected = self._readings(reference, n=20)[10:]

        sensor = PhysicalSensor("ph_1", sensor_type="pH", measurement_noise=0.05, accuracy=0.1)
        self._readings(sensor, n=10)
        restored = PhysicalSensor.from_dict(sensor.to_dict())

        assert restored.calibration_offset == sensor.calibration_offset
        assert self._readings(restored, start=10) == expected

    def test_a_restored_sensor_still_accepts_its_alias_keys(self) -> None:
        sensor = GasSensor("ch4_1", sensor_type="CH4", measurement_noise=0.0, accuracy=0.0)

        restored = GasSensor.from_dict(sensor.to_dict())
        out = restored.step(0.0, 1.0 / 24.0, {"CH4_fraction": 0.55})

        assert out["measurement"] == pytest.approx(55.0)


class TestMethaneSensorUnits:
    def test_reads_a_gas_storage_in_percent(self) -> None:
        storage = GasStorage("store", initial_ch4_fraction=0.55, initial_co2_fraction=0.43)
        sensor = GasSensor("ch4_1", sensor_type="CH4", measurement_noise=0.0, accuracy=0.0)

        out = sensor.step(0.0, 1.0 / 24.0, storage.outputs_data)

        assert out["measurement"] == pytest.approx(55.0)
        assert out["in_range"] is True

    def test_a_value_already_in_percent_is_not_scaled(self) -> None:
        sensor = GasSensor("ch4_1", sensor_type="CH4", measurement_noise=0.0, accuracy=0.0)

        assert sensor.step(0.0, 1.0 / 24.0, {"CH4": 55.0})["measurement"] == pytest.approx(55.0)

    def test_co2_sensor_reads_a_chp_free_storage_fraction(self) -> None:
        sensor = GasSensor("co2_1", sensor_type="CO2", measurement_noise=0.0, accuracy=0.0)

        assert sensor.step(0.0, 1.0 / 24.0, {"CO2_fraction": 0.4})["measurement"] == pytest.approx(40.0)


class TestMixerDutyCycle:
    @pytest.mark.parametrize("dt", [1.0 / 288.0, 1.0 / 24.0, 0.25, 1.0])
    def test_one_day_costs_the_same_energy_at_any_step_length(self, dt: float) -> None:
        mixer = Mixer("mixer_1", intermittent=True, on_time_fraction=0.25)

        P_running = max(mixer.step(i * dt, dt, {})["P_running"] for i in range(round(1.0 / dt)))

        assert mixer.operating_hours == pytest.approx(6.0)
        assert mixer.energy_consumed == pytest.approx(P_running * 6.0)

    def test_a_daily_step_reports_the_mean_power(self) -> None:
        mixer = Mixer("mixer_1", intermittent=True, on_time_fraction=0.25)

        out = mixer.step(0.0, 1.0, {})

        assert out["P_consumed"] == pytest.approx(0.25 * out["P_running"])
        assert out["P_average"] == pytest.approx(out["P_consumed"])

    def test_the_cycle_is_one_hour(self) -> None:
        mixer = Mixer("mixer_1", intermittent=True, on_time_fraction=0.25)
        minute = 1.0 / 1440.0

        assert mixer.step(10 * minute, minute, {})["is_running"] is True
        assert mixer.step(20 * minute, minute, {})["is_running"] is False
        assert mixer.step(70 * minute, minute, {})["is_running"] is True

    def test_a_zero_length_step_reports_the_momentary_state(self) -> None:
        mixer = Mixer("mixer_1", intermittent=True, on_time_fraction=0.25)

        assert mixer.step(0.0, 0.0, {})["is_running"] is True
        assert mixer.step(0.5 / 24.0, 0.0, {})["is_running"] is False

    def test_a_disabled_mixer_stays_off(self) -> None:
        mixer = Mixer("mixer_1", intermittent=False)

        out = mixer.step(0.0, 1.0, {"enable_mixing": False})

        assert out["P_consumed"] == 0.0
        assert mixer.operating_hours == 0.0


class TestStateSnapshots:
    def test_pump_snapshot_keeps_the_nominal_efficiency(self) -> None:
        pump = Pump("pump", Q_nom=10.0, efficiency=0.6)
        pump.step(0.0, 1.0, {"Q_setpoint": 3.0})
        assert pump.actual_efficiency != pytest.approx(0.6)

        pump.set_state(pump.get_state())

        assert pump.efficiency == 0.6

    def test_heating_snapshot_keeps_the_configured_coefficient(self) -> None:
        heating = HeatingSystem("heat")  # no own coefficient: follows the digester
        heating.step(0.0, 1.0, {"heat_loss_coefficient": 0.9})

        heating.set_state(heating.get_state())

        assert heating.heat_loss_coefficient is None
        assert heating.state["heat_loss_coefficient_used"] == pytest.approx(0.9)

    def test_flare_snapshot_round_trip(self) -> None:
        flare = Flare("flare")
        flare.step(0.0, 1.0, {"Q_gas_in_m3_per_day": 100.0})
        snapshot = flare.get_state()
        flare.step(1.0, 1.0, {"Q_gas_in_m3_per_day": 100.0})

        flare.set_state(snapshot)

        assert snapshot == {"cumulative_vented_m3": 100.0}
        assert flare.outputs_data["cumulative_vented_m3"] == pytest.approx(100.0)

    def test_upgrading_snapshot_round_trip(self) -> None:
        unit = BiogasUpgrading("bgaa", capacity_m3h=50.0)
        unit.step(0.0, 1.0, {"Q_gas_in_m3_per_day": 600.0})
        snapshot = unit.get_state()
        unit.step(1.0, 1.0, {"Q_gas_in_m3_per_day": 600.0})

        unit.set_state(snapshot)

        assert snapshot["cumulative_gas_in_m3"] == pytest.approx(600.0)
        assert unit.outputs_data["cumulative_gas_in_m3"] == pytest.approx(600.0)

    def test_gas_storage_dict_round_trip_keeps_the_vented_total_and_wiring(self) -> None:
        storage = GasStorage("store", capacity_m3=100.0, initial_fill_fraction=1.0)
        storage.add_input("dig")
        storage.add_output("chp")
        storage.step(0.0, 1.0, {"Q_gas_in_m3_per_day": 40.0})

        restored = GasStorage.from_dict(storage.to_dict())

        assert restored.outputs_data["cumulative_vented_m3"] == pytest.approx(40.0)
        assert restored.inputs == ["dig"]
        assert restored.outputs == ["chp"]

    def test_chp_counts_hours_only_while_running(self) -> None:
        chp = CHP("chp", P_el_nom=100.0)

        chp.step(0.0, 1.0, {"Q_gas_supplied_m3_per_day": 0.0})
        assert chp.state["operating_hours"] == 0.0

        chp.step(1.0, 1.0, {"Q_gas_supplied_m3_per_day": 1000.0, "Q_ch4_supplied_m3_per_day": 600.0})
        assert chp.state["operating_hours"] == pytest.approx(24.0)


class TestBoilerSupply:
    def test_without_a_stated_supply_a_biogas_boiler_is_not_limited(self) -> None:
        boiler = Boiler("boiler", P_th_nom=100.0, fuel_type="biogas")

        out = boiler.step(0.0, 1.0, {"P_th_demand": 60.0})

        assert out["P_th_supplied"] == pytest.approx(60.0)
        assert out["P_th_unmet"] == 0.0


class TestProcessHeat:
    def test_feed_rates_are_tonnes_per_day(self) -> None:
        """10 t/d of water warmed by 20 K: 10 000 kg · 4.18 kJ/(kg·K) · 20 K per day."""
        from pyadm1 import Feedstock
        from pyadm1.components.energy.heating import _substrate_cp

        feedstock = Feedstock(["maize_silage_milk_ripeness"], feeding_freq=24, total_simtime=2)
        cp = _substrate_cp(feedstock.substrates[0])

        heat = _calc_process_heat_kw([10.0], feedstock, target_temperature=313.15, t_inlet=293.15)

        assert heat == pytest.approx(10.0 * 1000.0 * cp * 20.0 / 86400.0)
