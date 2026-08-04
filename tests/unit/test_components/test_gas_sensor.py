"""Unit tests for gas sensor components."""

import math

import pytest

from pyadm1.components.base import ComponentType
from pyadm1.components.sensors import GasSensor


class TestGasSensorInitialization:
    """Construction and initialization behavior."""

    def test_initialization_sets_expected_defaults(self) -> None:
        sensor = GasSensor("ch4_1")

        assert sensor.component_id == "ch4_1"
        assert sensor.component_type == ComponentType.SENSOR
        assert sensor.sensor_type.value == "CH4"
        assert sensor.signal_key == "CH4"
        assert sensor.analyzer_method.value == "infrared"
        assert sensor.output_key == "ch4_1_measurement"


class TestGasSensorStep:
    """Gas sensor signal processing behavior."""

    def test_methane_sensor_reads_fraction_without_noise(self) -> None:
        sensor = GasSensor("ch4_1", sensor_type="CH4", measurement_noise=0.0, accuracy=0.0)

        result = sensor.step(t=0.0, dt=1.0 / 24.0, inputs={"methane_fraction": 57.5})

        assert result["measurement"] == 57.5
        assert result["true_value"] == 57.5

    def test_h2s_sensor_applies_detection_limit(self) -> None:
        sensor = GasSensor(
            "h2s_1",
            sensor_type="H2S",
            measurement_noise=0.0,
            accuracy=0.0,
            detection_limit=5.0,
        )

        result = sensor.step(t=0.0, dt=1.0 / 24.0, inputs={"H2S_ppm": 2.0})

        assert result["measurement"] == 0.0
        assert result["is_detected"] is False

    def test_response_time_applies_first_order_lag(self) -> None:
        sensor = GasSensor(
            "co2_1",
            sensor_type="CO2",
            response_time=1.0,
            measurement_noise=0.0,
            accuracy=0.0,
        )

        first = sensor.step(t=0.0, dt=0.5, inputs={"CO2": 35.0})
        second = sensor.step(t=0.5, dt=0.5, inputs={"CO2": 45.0})

        assert first["measurement"] == 35.0
        assert math.isclose(second["measurement"], 40.0, rel_tol=0.0, abs_tol=1e-9)

    def test_sample_interval_holds_last_measurement_between_updates(self) -> None:
        sensor = GasSensor(
            "o2_1",
            sensor_type="O2",
            sample_interval=1.0,
            measurement_noise=0.0,
            accuracy=0.0,
        )

        first = sensor.step(t=0.0, dt=0.25, inputs={"O2": 0.4})
        second = sensor.step(t=0.25, dt=0.25, inputs={"O2": 0.8})
        third = sensor.step(t=1.0, dt=0.25, inputs={"O2": 0.8})

        assert first["measurement"] == 0.4
        assert second["measurement"] == 0.4
        assert third["measurement"] == 0.8

    def test_trace_gas_sensor_accepts_custom_key(self) -> None:
        sensor = GasSensor(
            "trace_1",
            sensor_type="trace_gas",
            signal_key="siloxanes",
            measurement_noise=0.0,
            accuracy=0.0,
        )

        result = sensor.step(t=0.0, dt=1.0 / 24.0, inputs={"siloxanes": 14.0})

        assert result["measurement"] == 14.0
        assert result["true_value"] == 14.0


class TestGasSensorSerialization:
    """Serialization helpers."""

    def test_roundtrip_from_dict_restores_configuration(self) -> None:
        original = GasSensor(
            "co2_1",
            sensor_type="CO2",
            analyzer_method="infrared",
            signal_key="CO2_fraction",
            measurement_range=(0.0, 100.0),
            measurement_noise=0.0,
            accuracy=0.0,
            drift_rate=0.02,
            response_time=0.5,
            sample_interval=0.25,
            detection_limit=0.1,
            unit="%",
            output_key="co2_signal",
            name="CO2 Analyzer",
        )
        original.add_input("gas_stream_1")
        original.add_output("controller_1")

        restored = GasSensor.from_dict(original.to_dict())

        assert restored.component_id == "co2_1"
        assert restored.sensor_type.value == "CO2"
        assert restored.analyzer_method.value == "infrared"
        assert restored.signal_key == "CO2_fraction"
        assert restored.measurement_range == (0.0, 100.0)
        assert restored.drift_rate == 0.02
        assert restored.response_time == 0.5
        assert restored.sample_interval == 0.25
        assert restored.detection_limit == 0.1
        assert restored.unit == "%"
        assert restored.output_key == "co2_signal"
        assert restored.inputs == ["gas_stream_1"]
        assert restored.outputs == ["controller_1"]


class TestGasSensorValidity:
    """``is_valid`` reports whether the last due sample produced a reading."""

    def test_a_sample_without_an_upstream_signal_is_marked_invalid(self) -> None:
        """Continuous mode: the sample is due but no signal ever arrived."""
        sensor = GasSensor("ch4_1", sensor_type="CH4", measurement_noise=0.0, accuracy=0.0)

        sensor.step(t=0.0, dt=1.0 / 24.0, inputs={"unrelated": 1.0})

        assert sensor.is_valid is False
        assert math.isnan(sensor.true_value)
        assert math.isnan(sensor.measured_value)

    def test_validity_is_regained_once_the_signal_appears(self) -> None:
        sensor = GasSensor("ch4_1", sensor_type="CH4", measurement_noise=0.0, accuracy=0.0, response_time=0.0)

        sensor.step(t=0.0, dt=1.0 / 24.0, inputs={})
        assert sensor.is_valid is False

        sensor.step(t=1.0 / 24.0, dt=1.0 / 24.0, inputs={"methane_fraction": 55.0})
        assert sensor.is_valid is True
        assert sensor.measured_value > 0.0

    def test_a_batch_analyzer_without_a_signal_is_marked_invalid(self) -> None:
        """Batch (GC) mode: nothing can be queued, so the due sample is invalid."""
        sensor = GasSensor(
            "gc_1",
            sensor_type="CH4",
            analyzer_method="gas_chromatography",
            measurement_delay=0.05,
            sample_interval=0.1,
            measurement_noise=0.0,
            accuracy=0.0,
        )

        sensor.step(t=0.0, dt=1.0 / 24.0, inputs={})

        assert sensor.is_valid is False
        assert sensor._pending_samples == []

    def test_a_stale_reading_between_samples_stays_valid(self) -> None:
        """Between two due samples the sensor holds its last result rather than flagging invalid."""
        sensor = GasSensor(
            "ch4_1",
            sensor_type="CH4",
            sample_interval=1.0,
            measurement_noise=0.0,
            accuracy=0.0,
            response_time=0.0,
        )

        sensor.step(t=0.0, dt=1.0 / 24.0, inputs={"methane_fraction": 55.0})
        assert sensor.is_valid is True

        # Not due yet -> the invalid branch must not fire.
        sensor.step(t=0.1, dt=1.0 / 24.0, inputs={})

        assert sensor.is_valid is True
        assert sensor.measured_value == pytest.approx(55.0)
