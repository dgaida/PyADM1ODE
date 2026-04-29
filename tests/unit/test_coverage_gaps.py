# -*- coding: utf-8 -*-
"""
Targeted tests for coverage gaps identified by ``pytest --cov``.

This module bundles tests that exercise specific uncovered branches across the
codebase rather than belonging to one module's behavioural suite.  Keeping
them here avoids polluting the focused suites and makes it obvious why the
tests exist when the gap they fill is closed for some other reason.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from pyadm1 import BiogasPlant, Feedstock
from pyadm1.components.biological import Digester
from pyadm1.configurator.connection_manager import Connection
from pyadm1.configurator.plant_configurator import PlantConfigurator
from pyadm1.core.adm1 import ADM1, STATE_SIZE
from pyadm1.simulation.parallel import (
    ParallelSimulator,
    ScenarioResult,
    _compute_scenario_metrics,
)
from pyadm1.substrates.feedstock import (
    SubstrateRegistry,
    load_substrate_xml,
)

# ---------------------------------------------------------------------------
# plant_builder.py — CHP gas-demand pass 3, simulate save logic, from_json
# ---------------------------------------------------------------------------


@pytest.fixture
def feedstock() -> Feedstock:
    return Feedstock(
        ["maize_silage_milk_ripeness", "swine_manure"],
        feeding_freq=24,
        total_simtime=5,
    )


class TestPlantBuilderCHPDemandPass:
    """Exercise the Pass 3 CHP gas-demand → storage feedback loop in BiogasPlant.step."""

    def test_step_re_executes_storage_with_chp_demand(self, feedstock: Feedstock) -> None:
        plant = BiogasPlant("Demand Test")
        cfg = PlantConfigurator(plant, feedstock)

        cfg.add_digester(
            "main_digester",
            V_liq=1200.0,
            V_gas=216.0,
            T_ad=315.15,
            Q_substrates=[11.4, 6.1],
        )
        cfg.add_chp("chp_1", P_el_nom=200.0, eta_el=0.40, eta_th=0.45)
        cfg.auto_connect_digester_to_chp("main_digester", "chp_1")

        plant.initialize()

        results = plant.step(dt=1.0)

        assert "chp_1" in results
        assert "main_digester_storage" in results
        # Pass 3 fires the CHP step with Q_gas_supplied → P_el non-zero on operation
        assert "P_el" in results["chp_1"]

    def test_step_skips_chp_with_no_storage_attached(self, feedstock: Feedstock) -> None:
        plant = BiogasPlant("No-Storage Test")
        cfg = PlantConfigurator(plant, feedstock)

        cfg.add_chp("chp_isolated", P_el_nom=100.0)
        plant.initialize()

        results = plant.step(dt=1.0)

        # CHP is in the plant but has no incoming gas connection — Pass 3
        # `continue` branch should be hit and CHP stays at idle.
        assert "chp_isolated" in results


class TestPlantBuilderSimulateBranches:
    def test_simulate_default_save_interval_uses_dt(self, feedstock: Feedstock) -> None:
        plant = BiogasPlant("Default Save Test")
        cfg = PlantConfigurator(plant, feedstock)
        cfg.add_digester(
            "d1",
            V_liq=1200.0,
            V_gas=216.0,
            T_ad=315.15,
            Q_substrates=[11.4, 6.1],
        )
        plant.initialize()

        # Without save_interval kwarg → save every step.
        results = plant.simulate(duration=2.0, dt=1.0)
        assert len(results) == 2

    def test_simulate_emits_progress_every_100_steps(self, feedstock: Feedstock, capsys) -> None:
        plant = BiogasPlant("Progress Test")
        cfg = PlantConfigurator(plant, feedstock)
        cfg.add_digester(
            "d1",
            V_liq=1200.0,
            V_gas=216.0,
            T_ad=315.15,
            Q_substrates=[11.4, 6.1],
        )
        plant.initialize()

        # Use a stub component so 100 steps run quickly: replace ``step`` to be a no-op.
        d = plant.components["d1"]
        d.step = lambda t, dt, inputs: {"Q_gas": 0.0, "Q_ch4": 0.0, "pH": 7.0}
        # Storage stays a real GasStorage.

        plant.simulate(duration=120.0, dt=1.0, save_interval=10.0)
        out = capsys.readouterr().out
        assert "Simulated 100/120 steps" in out


class TestPlantBuilderJsonRoundTrip:
    def test_from_json_recreates_components_and_connections(self, feedstock: Feedstock, tmp_path: Path) -> None:
        plant = BiogasPlant("Round Trip")
        cfg = PlantConfigurator(plant, feedstock)
        cfg.add_digester(
            "main_digester",
            V_liq=1200.0,
            V_gas=216.0,
            T_ad=315.15,
            Q_substrates=[11.4, 6.1],
        )
        cfg.add_chp("chp_1", P_el_nom=300.0)
        cfg.add_heating("heating_1", target_temperature=315.15)
        cfg.auto_connect_digester_to_chp("main_digester", "chp_1")
        cfg.auto_connect_chp_to_heating("chp_1", "heating_1")

        path = tmp_path / "plant.json"
        plant.to_json(str(path))

        loaded = BiogasPlant.from_json(str(path), feedstock=feedstock)

        assert loaded.plant_name == "Round Trip"
        assert "main_digester" in loaded.components
        assert "chp_1" in loaded.components
        assert "heating_1" in loaded.components
        # Connections were restored (digester→storage, storage→chp, chp→heating, chp→flare)
        assert len(loaded.connections) == len(plant.connections)

    def test_from_json_raises_when_digester_present_without_feedstock(self, feedstock: Feedstock, tmp_path: Path) -> None:
        plant = BiogasPlant("Missing Feedstock")
        cfg = PlantConfigurator(plant, feedstock)
        cfg.add_digester("d1", V_liq=1200.0, V_gas=216.0, T_ad=315.15)

        path = tmp_path / "plant.json"
        plant.to_json(str(path))

        with pytest.raises(ValueError, match="Feedstock required"):
            BiogasPlant.from_json(str(path), feedstock=None)

    def test_from_json_raises_for_unknown_component_type(self, tmp_path: Path) -> None:
        path = tmp_path / "plant.json"
        path.write_text(
            json.dumps(
                {
                    "plant_name": "Bogus",
                    "components": [
                        # ``mixer`` is a valid ComponentType but plant_builder.from_json
                        # only knows DIGESTER / CHP / HEATING — it should reject anything else.
                        {"component_type": "mixer", "component_id": "mx1", "name": "Mixer"}
                    ],
                    "connections": [],
                }
            )
        )

        with pytest.raises(ValueError, match="Unknown component type"):
            BiogasPlant.from_json(str(path), feedstock=None)


# ---------------------------------------------------------------------------
# Digester — state_in / Q_in chaining and helper edge cases
# ---------------------------------------------------------------------------


class TestDigesterChaining:
    def test_step_with_state_in_and_q_in_mixes_influents(self, feedstock: Feedstock) -> None:
        upstream = Digester("upstream", feedstock, V_liq=600.0, V_gas=100.0, T_ad=315.15)
        upstream.initialize({"Q_substrates": [11.4, 6.1]})

        downstream = Digester("downstream", feedstock, V_liq=1200.0, V_gas=216.0, T_ad=308.15)
        downstream.initialize({"Q_substrates": [0.0, 0.0]})

        # First, advance the upstream stage one day so it has a non-trivial state.
        up_out = upstream.step(t=0.0, dt=1.0, inputs={"Q_substrates": [11.4, 6.1]})

        out = downstream.step(
            t=0.0,
            dt=1.0,
            inputs={"Q_in": up_out["Q_out"], "state_in": up_out["state_out"]},
        )
        # The chained step should produce gas and a sensible pH.
        assert out["Q_out"] >= up_out["Q_out"]  # at least the upstream effluent flow
        assert 5.0 < out["pH"] < 9.0


class TestDigesterHelpersAndCalibration:
    def test_has_valid_state_input_false_before_create_influent(self, feedstock: Feedstock) -> None:
        d = Digester("d1", feedstock, V_liq=1200.0, V_gas=216.0)
        d.initialize()  # No Q_substrates → no _state_input populated yet

        # ``adm1._state_input`` is None before create_influent runs.
        assert d._has_valid_state_input() is False

    def test_calibration_helpers_round_trip(self, feedstock: Feedstock) -> None:
        d = Digester("d1", feedstock)
        d.apply_calibration_parameters({"k_p": 5.0e3})
        assert d.get_calibration_parameters()["k_p"] == 5.0e3

        d.clear_calibration_parameters()
        assert d.get_calibration_parameters() == {}

    def test_initialize_falls_back_to_default_state_when_feedstock_lacks_blending(
        self,
    ) -> None:
        # A feedstock-less Digester with explicit Q_substrates → uses the
        # plain [0.01]*STATE_SIZE default (no pre-inoculation).
        d = Digester("d1", feedstock=None, V_liq=1200.0, V_gas=216.0)
        d.initialize({"Q_substrates": [10.0]})

        assert d.adm1_state == [0.01] * STATE_SIZE


# ---------------------------------------------------------------------------
# Feedstock — single-substrate accessors, error branches, helpers
# ---------------------------------------------------------------------------


class TestFeedstockSingleSubstrateAccessors:
    @pytest.fixture
    def fs(self) -> Feedstock:
        return Feedstock("maize_silage_milk_ripeness", feeding_freq=24, total_simtime=3)

    def test_substrate_property(self, fs: Feedstock) -> None:
        assert fs.substrate.name  # human-readable name from XML

    def test_concentrations_property_returns_dict(self, fs: Feedstock) -> None:
        c = fs.concentrations
        assert isinstance(c, dict)
        assert "X_PS_ch" in c

    def test_density_property(self, fs: Feedstock) -> None:
        assert fs.density > 1000.0  # solid substrate

    def test_q_conversion_factors(self, fs: Feedstock) -> None:
        factors = fs.q_conversion_factors
        assert len(factors) == 1
        # Solid maize → factor < 1.0 (mass-equivalent → smaller actual volume)
        assert factors[0] < 1.0

    def test_concentrations_list_returns_copies(self, fs: Feedstock) -> None:
        out = fs.concentrations_list
        out[0]["S_ac"] = -999.0
        assert fs.concentrations_list[0]["S_ac"] != -999.0


class TestFeedstockMultiSubstrateAccessors:
    @pytest.fixture
    def fs(self) -> Feedstock:
        return Feedstock(
            ["maize_silage_milk_ripeness", "swine_manure"],
            feeding_freq=24,
            total_simtime=3,
        )

    def test_single_substrate_accessor_raises(self, fs: Feedstock) -> None:
        with pytest.raises(ValueError, match="single-substrate accessor"):
            _ = fs.substrate

        with pytest.raises(ValueError, match="single-substrate accessor"):
            _ = fs.density

        with pytest.raises(ValueError, match="single-substrate accessor"):
            _ = fs.concentrations

    def test_blended_accessors_at_zero_flow_return_safe_defaults(self, fs: Feedstock) -> None:
        # All-zero feed → density falls back to water, blended concs are zero.
        assert fs.blended_density([0.0, 0.0]) == 1000.0
        assert fs.blended_vs_content([0.0, 0.0]) == 0.0
        blended = fs.blended_concentrations([0.0, 0.0])
        assert all(v == 0.0 for v in blended.values())

    def test_blended_vs_content_uses_weighted_average(self, fs: Feedstock) -> None:
        vs_pure_maize = fs.blended_vs_content([5.0, 0.0])
        vs_pure_manure = fs.blended_vs_content([0.0, 5.0])
        vs_mix = fs.blended_vs_content([5.0, 5.0])
        # Mix should fall between the two pure values (weighted by actual_Q)
        lo, hi = min(vs_pure_maize, vs_pure_manure), max(vs_pure_maize, vs_pure_manure)
        assert lo - 1e-9 <= vs_mix <= hi + 1e-9

    def test_bmp_theoretical_returns_positive_for_real_substrate(self, fs: Feedstock) -> None:
        assert fs.bmp_theoretical(0) > 0.0


class TestFeedstockEdgeCases:
    def test_empty_substrate_list_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one substrate"):
            Feedstock([])

    def test_unknown_substrate_id_raises(self) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            Feedstock("not_a_real_substrate_xyz")

    def test_extra_nonzero_q_raises(self) -> None:
        fs = Feedstock("maize_silage_milk_ripeness", feeding_freq=24, total_simtime=3)
        with pytest.raises(ValueError, match="non-zero values beyond"):
            fs.actual_Q([5.0, 1.0])  # 1.0 is past the only-1-substrate limit

    def test_load_substrate_xml_missing_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Substrate XML not found"):
            load_substrate_xml(tmp_path / "no_such_file.xml")

    def test_load_substrate_xml_missing_required_param_raises(self, tmp_path: Path) -> None:
        # Build an incomplete XML: only ``name`` attribute, no params at all.
        xml_path = tmp_path / "incomplete.xml"
        xml_path.write_text('<?xml version="1.0"?><substrate name="incomplete"/>')

        with pytest.raises(ValueError, match="Required parameter"):
            load_substrate_xml(xml_path)

    def test_registry_unknown_id_raises(self) -> None:
        reg = SubstrateRegistry()
        with pytest.raises(KeyError, match="not found"):
            reg.get("not_a_real_id_xyz")

    def test_registry_load_all_returns_full_dict(self) -> None:
        reg = SubstrateRegistry()
        all_subs = reg.load_all()
        assert set(all_subs.keys()) == set(reg.available())

    def test_registry_returns_empty_for_missing_directory(self, tmp_path: Path) -> None:
        reg = SubstrateRegistry(xml_dir=tmp_path / "does_not_exist")
        assert reg.available() == []

    def test_feedstock_default_loads_all_bundled_xmls(self) -> None:
        fs = Feedstock(feeding_freq=24, total_simtime=3)
        # The bundled directory ships with at least the three SIMBA# substrates.
        ids = set(fs.substrate_ids)
        assert {"cattle_manure", "maize_silage_milk_ripeness", "swine_manure"} <= ids


# ---------------------------------------------------------------------------
# ParallelSimulator — initialization, default workers, and metric edge cases
# ---------------------------------------------------------------------------


class _StubADM1:
    """Cheap stub satisfying ParallelSimulator._serialize_adm1's attribute access."""

    V_liq = 1200.0
    _V_gas = 216.0
    _T_ad = 315.15
    feedstock = None


class TestParallelSimulatorInit:
    def test_default_workers_is_cpu_count_minus_one(self) -> None:
        sim = ParallelSimulator(_StubADM1(), verbose=False)
        # Whatever the host CPU count, workers must be at least 1
        assert sim.n_workers >= 1

    def test_serialize_adm1_handles_missing_feedstock(self) -> None:
        sim = ParallelSimulator(_StubADM1(), n_workers=1, verbose=False)
        cfg = sim._serialize_adm1()
        assert cfg["V_liq"] == 1200.0
        assert cfg["feedstock_substrates"] == []
        assert cfg["feeding_freq"] == 24


class TestParallelSimulatorSummary:
    def test_summarize_results_computes_statistics_for_successful_metrics(self) -> None:
        sim = ParallelSimulator(_StubADM1(), n_workers=1, verbose=False)
        results = [
            ScenarioResult(
                scenario_id=i,
                parameters={},
                success=True,
                duration=1.0,
                metrics={"Q_gas": float(100 + i * 50)},
            )
            for i in range(4)
        ]
        summary = sim.summarize_results(results, metrics=["Q_gas"])

        assert summary["n_successful"] == 4
        stats = summary["metrics"]["Q_gas"]
        assert stats["mean"] == pytest.approx(np.mean([100.0, 150.0, 200.0, 250.0]))
        assert stats["min"] == 100.0
        assert stats["max"] == 250.0

    def test_summarize_results_skips_nan_metric_values(self) -> None:
        sim = ParallelSimulator(_StubADM1(), n_workers=1, verbose=False)
        results = [
            ScenarioResult(0, {}, True, 1.0, metrics={"Q_gas": float("nan")}),
            ScenarioResult(1, {}, True, 1.0, metrics={"Q_gas": 100.0}),
        ]
        summary = sim.summarize_results(results, metrics=["Q_gas"])
        # NaN dropped → only one valid value → mean equals it.
        assert summary["metrics"]["Q_gas"]["mean"] == 100.0


class TestComputeScenarioMetrics:
    def test_metrics_includes_specific_yields_when_q_total_positive(self) -> None:
        fs = Feedstock(["maize_silage_milk_ripeness"], feeding_freq=24, total_simtime=2)
        adm = ADM1(fs, V_liq=1200.0, V_gas=216.0, T_ad=315.15)
        state = [0.01] * STATE_SIZE
        state[37:41] = [1.0e-5, 0.65, 0.33, 0.65 + 0.33 + 1.0e-5]

        metrics = _compute_scenario_metrics(adm, state, [10.0])

        assert metrics["Q_total"] == 10.0
        assert metrics["specific_gas_production"] >= 0.0
        assert metrics["HRT"] == pytest.approx(120.0)


# ---------------------------------------------------------------------------
# Core ADM1 — set_influent_density, set_calibration_parameters edge cases
# ---------------------------------------------------------------------------


class TestADM1IndividualHelpers:
    @pytest.fixture
    def adm1(self) -> ADM1:
        fs = Feedstock(["maize_silage_milk_ripeness"], feeding_freq=24, total_simtime=2)
        return ADM1(fs, V_liq=1200.0, V_gas=216.0, T_ad=315.15)

    def test_set_influent_density_stores_values(self, adm1: ADM1) -> None:
        adm1.set_influent_density(rho_in=1100.0, rho_sludge=900.0)
        assert adm1._rho_in == 1100.0
        assert adm1._rho_sludge == 900.0

    def test_resume_from_broken_simulation_appends_q_ch4(self, adm1: ADM1) -> None:
        adm1.resume_from_broken_simulation([100.0, 110.0, 120.0])
        assert list(adm1.Q_CH4) == [100.0, 110.0, 120.0]

    def test_print_params_at_current_state_populates_tracking_lists(self, adm1: ADM1) -> None:
        state = [0.01] * STATE_SIZE
        state[37:41] = [1.0e-5, 0.65, 0.33, 0.65 + 0.33 + 1.0e-5]

        adm1.print_params_at_current_state(state)

        # Both lists got bootstrapped to length >= 2 (see _track_pH / _track_gas).
        assert len(adm1.pH_l) >= 2
        assert len(adm1.Q_GAS) >= 3

    def test_create_influent_handles_density_kwarg(self, adm1: ADM1) -> None:
        # rho is honoured: the weighted average is stored in _rho_in.
        adm1.create_influent([10.0], 0, rho=[1100.0])
        assert adm1._rho_in == pytest.approx(1100.0)

    def test_clear_then_set_calibration_parameters(self, adm1: ADM1) -> None:
        adm1.set_calibration_parameters({"k_p": 5.0e3, "k_L_a": 250.0})
        adm1.clear_calibration_parameters()
        assert adm1.get_calibration_parameters() == {}

        adm1.set_calibration_parameters({"k_L_a": 300.0})
        assert adm1.get_calibration_parameters() == {"k_L_a": 300.0}


# ---------------------------------------------------------------------------
# PlantConfigurator — create_two_stage_plant heating list path
# ---------------------------------------------------------------------------


class TestPlantConfiguratorCreateTwoStagePlantHeating:
    def test_heating_configs_attaches_per_stage_heating_to_chp(self, feedstock: Feedstock) -> None:
        plant = BiogasPlant("Two-Stage With Heating")
        cfg = PlantConfigurator(plant, feedstock)

        components = cfg.create_two_stage_plant(
            hydrolysis_config={"V_liq": 500.0, "V_gas": 75.0, "T_ad": 318.15},
            digester_config={"V_liq": 1200.0, "V_gas": 216.0, "T_ad": 315.15},
            heating_configs=[
                {"target_temperature": 318.15},
                {"target_temperature": 315.15},
            ],
        )

        assert "heating" in components
        assert len(components["heating"]) == 2
        # Heating connections were made.
        assert any(c.connection_type == "heat" for c in plant.connections)


# ---------------------------------------------------------------------------
# Connection.from_dict — covers the key-mapping branch
# ---------------------------------------------------------------------------


def test_connection_from_dict_round_trip() -> None:
    original = Connection("a", "b", "gas")
    restored = Connection.from_dict(original.to_dict())
    assert restored.from_component == "a"
    assert restored.to_component == "b"
    assert restored.connection_type == "gas"


# ---------------------------------------------------------------------------
# PhysicalSensor — Nernst pH temperature compensation branches
# ---------------------------------------------------------------------------


class TestPhysicalSensorNernst:
    def _ph_sensor(self, **kwargs):
        from pyadm1.components.sensors.physical import PhysicalSensor

        defaults = dict(
            component_id="ph_sensor",
            sensor_type="pH",
            measurement_range=(0.0, 14.0),
            measurement_noise=0.0,
            accuracy=0.0,
            drift_rate=0.0,
            response_time=0.0,
            sample_interval=0.0,
            temperature_signal_key="temperature",
            temperature_reference=298.15,
            pH_isopotential=7.0,
        )
        defaults.update(kwargs)
        return PhysicalSensor(**defaults)

    def _step_sensor(self, sensor, inputs):
        sensor.initialize()
        return sensor.step(t=0.0, dt=1.0, inputs=inputs)

    def test_nernst_returns_unchanged_when_temperature_key_missing(self) -> None:
        sensor = self._ph_sensor()
        # No "temperature" key in inputs → correction is a no-op.
        out = self._step_sensor(sensor, {"pH": 8.0})
        # Without correction the reading equals the input pH.
        assert out["measurement"] == pytest.approx(8.0, abs=1e-6)

    def test_nernst_returns_unchanged_for_non_numeric_temperature(self) -> None:
        sensor = self._ph_sensor()
        out = self._step_sensor(sensor, {"pH": 8.0, "temperature": "not-a-number"})
        assert out["measurement"] == pytest.approx(8.0, abs=1e-6)

    def test_nernst_returns_unchanged_for_non_positive_temperature(self) -> None:
        sensor = self._ph_sensor()
        out = self._step_sensor(sensor, {"pH": 8.0, "temperature": 0.0})
        assert out["measurement"] == pytest.approx(8.0, abs=1e-6)

    def test_nernst_correction_at_higher_temperature_shifts_reading(self) -> None:
        # T_actual > T_ref scales the deviation from the isopotential point by
        # T_ref / T_actual, so the reading is *closer* to pH 7 at higher T.
        sensor = self._ph_sensor()
        out = self._step_sensor(sensor, {"pH": 8.0, "temperature": 333.15})  # 60 °C

        # 7.0 + (8.0 - 7.0) * (298.15 / 333.15) ≈ 7.895
        expected = 7.0 + 1.0 * (298.15 / 333.15)
        assert out["measurement"] == pytest.approx(expected, rel=1e-5)


class TestPhysicalSensorInvalidWhenSourceMissing:
    def test_is_valid_false_when_signal_key_absent(self) -> None:
        from pyadm1.components.sensors.physical import PhysicalSensor

        sensor = PhysicalSensor(
            component_id="t1",
            sensor_type="temperature",
            sample_interval=0.0,
        )
        sensor.initialize()
        out = sensor.step(t=0.0, dt=1.0, inputs={})  # No "temperature" key
        # Sensor falls back to invalid when it can't resolve a signal.
        assert out["is_valid"] is False or sensor.is_valid is False


# ---------------------------------------------------------------------------
# GasSensor — cross-sensitivity, batch (GC) mode, sample maturation
# ---------------------------------------------------------------------------


class TestGasSensorCrossSensitivity:
    def test_cross_sensitivity_adds_bias_from_other_inputs(self) -> None:
        from pyadm1.components.sensors.gas import GasSensor

        sensor = GasSensor(
            component_id="ch4_ndir",
            sensor_type="CH4",
            analyzer_method="infrared",
            measurement_range=(0.0, 100.0),
            measurement_noise=0.0,
            accuracy=0.0,
            drift_rate=0.0,
            response_time=0.0,
            measurement_delay=0.0,
            detection_limit=0.0,
            sample_interval=0.0,
            cross_sensitivity={"CO2": 0.5},  # +0.5 %-CH4 per %-CO2
        )
        sensor.initialize()

        baseline = sensor.step(t=0.0, dt=1.0, inputs={"CH4": 50.0})

        # Reset state and re-step with CO2 interference.
        sensor.initialize()
        biased = sensor.step(t=0.0, dt=1.0, inputs={"CH4": 50.0, "CO2": 30.0})

        # Bias adds 0.5 × 30 = +15 to the CH4 reading.
        assert biased["measurement"] > baseline["measurement"]

    def test_cross_sensitivity_skips_non_numeric_inputs(self) -> None:
        from pyadm1.components.sensors.gas import GasSensor

        sensor = GasSensor(
            component_id="ch4_ndir",
            sensor_type="CH4",
            analyzer_method="infrared",
            measurement_noise=0.0,
            response_time=0.0,
            measurement_delay=0.0,
            detection_limit=0.0,
            sample_interval=0.0,
            cross_sensitivity={"CO2": 0.5, "H2S": 0.1},
        )
        sensor.initialize()

        # Pass a non-numeric value for one of the cross-sensitivity inputs.
        out = sensor.step(t=0.0, dt=1.0, inputs={"CH4": 50.0, "CO2": 30.0, "H2S": "n/a"})
        # Should not raise; CO2 path adds 15, H2S path is skipped.
        assert out is not None


class TestGasSensorBatchMode:
    def _gc_sensor(self):
        from pyadm1.components.sensors.gas import GasSensor

        sensor = GasSensor(
            component_id="gc_h2s",
            sensor_type="H2S",
            analyzer_method="gas_chromatography",
            measurement_range=(0.0, 1000.0),
            measurement_noise=0.0,
            accuracy=0.0,
            drift_rate=0.0,
            response_time=0.0,
            measurement_delay=2.0 / 24.0,  # 2 hours analysis delay
            detection_limit=0.0,
            sample_interval=1.0 / 24.0,  # sample every hour
        )
        sensor.initialize()
        return sensor

    def test_pending_sample_matures_after_delay(self) -> None:
        sensor = self._gc_sensor()
        dt = 1.0 / 24.0  # 1 hour

        # First step: queues a sample (release at t = 2/24).
        sensor.step(t=0.0, dt=dt, inputs={"H2S": 200.0})
        assert sensor._pending_samples, "First step should queue a pending sample"

        # Second step: another sample queued, no mature one yet.
        sensor.step(t=dt, dt=dt, inputs={"H2S": 220.0})

        # Third step (t = 2/24): the first sample matures and is reported.
        out_t2 = sensor.step(t=2 * dt, dt=dt, inputs={"H2S": 240.0})

        # By the third step the sensor should have finalised at least one sample.
        assert sensor.measured_value is not None
        # Outputs dict carries some value
        assert out_t2 is not None

    def test_pop_latest_ready_sample_returns_newest_mature_value(self) -> None:
        sensor = self._gc_sensor()
        # Manually plant two pending samples that both released by t = 1.0.
        sensor._pending_samples = [(0.5, 100.0), (0.9, 200.0)]
        latest = sensor._pop_latest_ready_sample(t=1.0)

        assert latest == 200.0
        assert sensor._pending_samples == []  # both removed

    def test_pop_latest_ready_sample_keeps_future_samples(self) -> None:
        sensor = self._gc_sensor()
        sensor._pending_samples = [(0.5, 100.0), (5.0, 200.0)]
        latest = sensor._pop_latest_ready_sample(t=1.0)

        assert latest == 100.0
        # The future sample remains queued.
        assert sensor._pending_samples == [(5.0, 200.0)]

    def test_invalid_when_sample_due_but_signal_is_nan(self) -> None:
        sensor = self._gc_sensor()
        # Force true_value to NaN by passing a non-resolvable signal.
        sensor.is_valid = True
        sensor.step(t=0.0, dt=1.0 / 24.0, inputs={})  # No H2S key → NaN

        # Sample due but signal NaN → is_valid set to False inside batch branch.
        assert sensor.is_valid is False
