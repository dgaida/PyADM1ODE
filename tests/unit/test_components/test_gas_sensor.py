# -*- coding: utf-8 -*-
"""Unit tests for gas sensor components."""

import math

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
        sensor = GasSensor("co2_1", sensor_type="CO2", response_time=1.0, measurement_noise=0.0, accuracy=0.0)

        first = sensor.step(t=0.0, dt=0.5, inputs={"CO2": 35.0})
        second = sensor.step(t=0.5, dt=0.5, inputs={"CO2": 45.0})

        assert first["measurement"] == 35.0
        assert math.isclose(second["measurement"], 40.0, rel_tol=0.0, abs_tol=1e-9)

    def test_sample_interval_holds_last_measurement_between_updates(self) -> None:
        sensor = GasSensor("o2_1", sensor_type="O2", sample_interval=1.0, measurement_noise=0.0, accuracy=0.0)

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
