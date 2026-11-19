"""Inputs that used to be accepted silently and produced plausible-looking nonsense."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml

from pyadm1 import BiogasPlant, Feedstock
from pyadm1.components.biological import Digester
from pyadm1.configurator.plant_configurator import PlantConfigurator
from pyadm1.core.adm1 import ADM1, STATE_SIZE
from pyadm1.simulation.parallel import ParallelSimulator, _run_single_scenario
from pyadm1.substrates.feedstock import SubstrateRegistry, load_substrate

_LEGACY_MAIZE = Path(__file__).resolve().parents[2] / "data" / "substrates" / "legacy" / "maize_silage_milk_ripeness.yaml"


@pytest.fixture
def feedstock() -> Feedstock:
    return Feedstock(["maize_silage_milk_ripeness", "cattle_manure"], feeding_freq=24, total_simtime=3)


class TestTemperature:
    def test_celsius_instead_of_kelvin_is_rejected(self, feedstock: Feedstock) -> None:
        with pytest.raises(ValueError, match="Kelvin"):
            ADM1(feedstock, T_ad=42.0)

    def test_setting_celsius_later_is_rejected_too(self, feedstock: Feedstock) -> None:
        adm1 = ADM1(feedstock, T_ad=315.15)

        with pytest.raises(ValueError, match="Kelvin"):
            adm1.T_ad = 38.0


class TestFeedRates:
    @pytest.mark.parametrize("rates", [[-5.0, 10.0], [float("nan"), 10.0], [float("inf"), 10.0]])
    def test_negative_or_non_finite_rates_are_rejected(self, feedstock: Feedstock, rates: list[float]) -> None:
        with pytest.raises(ValueError, match="finite and not negative"):
            feedstock.blended_concentrations(rates)

    def test_zero_feed_is_allowed(self, feedstock: Feedstock) -> None:
        assert feedstock.mass_flows([0.0, 0.0]) == [0.0, 0.0]


class TestSubstrateValues:
    @staticmethod
    def _valid():
        return SubstrateRegistry().get("maize_silage_milk_ripeness")

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("TS", 1200.0, "TS="),
            ("TS", -1.0, "TS="),
            ("pH", -3.0, "pH="),
            ("fRP", 8.5, "fraction of TS"),
            ("fRA", 0.99, "more than the dry matter holds"),
            ("aXI", 1.5, "aXI="),
            ("NH4", -0.1, "NH4="),
        ],
    )
    def test_out_of_range_values_are_rejected(self, field: str, value: float, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            dataclasses.replace(self._valid(), **{field: value})

    def test_water_without_solids_is_a_valid_substrate(self) -> None:
        water = dataclasses.replace(self._valid(), TS=0.0)

        assert water.TS == 0.0

    def test_a_misspelt_key_in_a_standard_file_is_rejected(self, tmp_path: Path) -> None:
        raw = yaml.safe_load(_LEGACY_MAIZE.read_text(encoding="utf-8"))
        raw["aXi"] = raw.pop("aXI")
        path = tmp_path / "typo.yaml"
        path.write_text(yaml.safe_dump(raw), encoding="utf-8")

        with pytest.raises(ValueError, match="Unknown parameter"):
            load_substrate(path)


class TestSubstrateIds:
    def test_the_same_substrate_twice_gets_two_ids(self) -> None:
        feedstock = Feedstock(["cattle_manure", "cattle_manure", "cattle_manure"], feeding_freq=24, total_simtime=2)

        assert feedstock.substrate_ids == ["cattle_manure", "cattle_manure_2", "cattle_manure_3"]

    def test_costs_of_a_repeated_substrate_are_not_merged(self) -> None:
        feedstock = Feedstock(["cattle_manure", "cattle_manure"], feeding_freq=24, total_simtime=2)
        feedstock.set_substrate_costs({"cattle_manure": 10.0, "cattle_manure_2": 30.0})

        costs = feedstock.calc_substrate_costs([1.0, 2.0])

        assert costs["by_substrate"] == pytest.approx({"cattle_manure": 10.0, "cattle_manure_2": 60.0})


class TestParallelScenarios:
    @staticmethod
    def _state() -> list[float]:
        """Inoculated start state of a digester fed with the test mix."""
        feedstock = Feedstock(["maize_silage_milk_ripeness", "cattle_manure"], feeding_freq=24, total_simtime=3)
        digester = Digester("d", feedstock, V_liq=1977.0, V_gas=304.0)
        digester.initialize({"Q_substrates": [10.0, 5.0]})
        assert len(digester.adm1_state) == STATE_SIZE
        return list(digester.adm1_state)

    def test_a_misspelt_scenario_parameter_is_rejected(self, feedstock: Feedstock) -> None:
        parallel = ParallelSimulator(ADM1(feedstock), n_workers=1, verbose=False)

        with pytest.raises(ValueError, match=r"unknown parameter\(s\) \['k_dis_Ps'\]"):
            parallel.run_scenarios([{"Q": [10.0, 5.0], "k_dis_Ps": 0.1}], duration=1.0, initial_state=self._state())

    def test_the_base_calibration_reaches_the_workers(self, feedstock: Feedstock) -> None:
        adm1 = ADM1(feedstock)
        adm1.set_calibration_parameters({"k_L_a": 50.0})

        config = ParallelSimulator(adm1, n_workers=1, verbose=False)._serialize_adm1()

        assert config["calibration"] == {"k_L_a": 50.0}

    def test_a_scenario_runs_with_the_base_calibration(self, feedstock: Feedstock) -> None:
        adm1 = ADM1(feedstock)
        plain = ParallelSimulator(adm1, n_workers=1, verbose=False)
        reference = plain.run_scenarios([{"Q": [10.0, 5.0]}], duration=1.0, initial_state=self._state())[0]

        adm1.set_calibration_parameters({"k_m_ac": 0.1})
        calibrated = ParallelSimulator(adm1, n_workers=1, verbose=False)
        result = calibrated.run_scenarios([{"Q": [10.0, 5.0]}], duration=1.0, initial_state=self._state())[0]

        assert reference.success and result.success
        assert reference.metrics["Q_ch4"] > 0.0
        assert result.metrics["Q_ch4"] != pytest.approx(reference.metrics["Q_ch4"], rel=1e-3)

    def test_substrates_travel_to_the_workers_as_parameters(self, tmp_path: Path) -> None:
        """A user file sharing its name with a library substrate must not be swapped for it."""
        raw = yaml.safe_load(_LEGACY_MAIZE.read_text(encoding="utf-8"))
        raw["TS"] = 200.0
        own = tmp_path / "maize_silage_milk_ripeness.yaml"
        own.write_text(yaml.safe_dump(raw), encoding="utf-8")
        adm1 = ADM1(Feedstock([own], feeding_freq=24, total_simtime=2))

        config = ParallelSimulator(adm1, n_workers=1, verbose=False)._serialize_adm1()

        assert config["feedstock_substrates"][0].TS == 200.0

    def _series_length(self, feedstock: Feedstock, dt: float, verbose: bool = False) -> int:
        config = ParallelSimulator(ADM1(feedstock), n_workers=1, verbose=False)._serialize_adm1()
        result = _run_single_scenario(
            (0, {"Q": [10.0, 5.0]}),
            adm1_config=config,
            duration=0.25,
            initial_state=self._state(),
            dt=dt,
            compute_metrics=False,
            save_time_series=True,
            verbose=verbose,
        )
        assert result.success, result.error
        assert all(q > 0.0 for q in result.time_series["Q_ch4"])
        return len(result.time_series["Q_gas"])

    def test_the_time_series_is_filled_without_verbose_output(self, feedstock: Feedstock) -> None:
        assert self._series_length(feedstock, dt=0.1) > 0

    def test_dt_sets_the_resolution_of_the_time_series(self, feedstock: Feedstock) -> None:
        # 0.25 d in steps of 0.1 d (three solver calls) and of 0.05 d (five).
        assert self._series_length(feedstock, dt=0.05) == self._series_length(feedstock, dt=0.1) + 2


class TestCalibrationRoundTrip:
    def test_calibration_parameters_survive_saving_and_loading_a_plant(self, feedstock: Feedstock, tmp_path: Path) -> None:
        plant = BiogasPlant("calibrated")
        PlantConfigurator(plant, feedstock).add_digester(
            "dig", V_liq=1200.0, V_gas=216.0, Q_substrates=[10.0, 5.0], k_L_a=50.0
        )
        path = tmp_path / "plant.json"
        plant.to_json(str(path))

        restored = BiogasPlant.from_json(str(path), feedstock=feedstock)

        assert restored.components["dig"].get_calibration_parameters() == {"k_L_a": 50.0}
