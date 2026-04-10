# -*- coding: utf-8 -*-
"""Unit tests for physical sensor components."""

import math

from pyadm1.components.base import ComponentType
from pyadm1.components.sensors import PhysicalSensor


class TestPhysicalSensorInitialization:
    """Construction and initialization behavior."""

    def test_initialization_sets_expected_defaults(self) -> None:
        sensor = PhysicalSensor("temp_1")

        assert sensor.component_id == "temp_1"
        assert sensor.component_type == ComponentType.SENSOR
        assert sensor.sensor_type.value == "temperature"
        assert sensor.signal_key == "temperature"
        assert sensor.output_key == "temp_1_measurement"


class TestPhysicalSensorStep:
    """Sensor signal processing behavior."""

    def test_temperature_sensor_reads_input_without_noise(self) -> None:
        sensor = PhysicalSensor("temp_1", sensor_type="temperature")

        result = sensor.step(t=0.0, dt=1.0 / 24.0, inputs={"temperature": 308.15})

        assert result["measurement"] == 308.15
        assert result["temp_1_measurement"] == 308.15
        assert result["true_value"] == 308.15
        assert result["is_valid"] is True

    def test_flow_sensor_falls_back_to_digester_outflow(self) -> None:
        sensor = PhysicalSensor("flow_1", sensor_type="flow")

        result = sensor.step(t=0.0, dt=1.0 / 24.0, inputs={"Q_out": 12.5})

        assert result["measurement"] == 12.5
        assert result["true_value"] == 12.5

    def test_response_time_applies_first_order_lag(self) -> None:
        sensor = PhysicalSensor("temp_1", sensor_type="temperature", response_time=1.0)

        first = sensor.step(t=0.0, dt=0.5, inputs={"temperature": 300.0})
        second = sensor.step(t=0.5, dt=0.5, inputs={"temperature": 320.0})

        assert first["measurement"] == 300.0
        assert math.isclose(second["measurement"], 310.0, rel_tol=0.0, abs_tol=1e-9)

    def test_sample_interval_holds_last_measurement_between_updates(self) -> None:
        sensor = PhysicalSensor("temp_1", sensor_type="temperature", sample_interval=1.0)

        first = sensor.step(t=0.0, dt=0.25, inputs={"temperature": 300.0})
        second = sensor.step(t=0.25, dt=0.25, inputs={"temperature": 320.0})
        third = sensor.step(t=1.0, dt=0.25, inputs={"temperature": 320.0})

        assert first["measurement"] == 300.0
        assert second["measurement"] == 300.0
        assert third["measurement"] == 320.0

    def test_measurement_is_clamped_to_range(self) -> None:
        sensor = PhysicalSensor("ph_1", sensor_type="pH", measurement_range=(0.0, 14.0))

        result = sensor.step(t=0.0, dt=1.0 / 24.0, inputs={"pH": 16.0})

        assert result["measurement"] == 14.0
        assert result["in_range"] is False

    def test_drift_rate_accumulates_over_time(self) -> None:
        sensor = PhysicalSensor("press_1", sensor_type="pressure", drift_rate=0.2)

        first = sensor.step(t=0.0, dt=1.0, inputs={"pressure_bar": 1.5})
        second = sensor.step(t=1.0, dt=1.0, inputs={"pressure_bar": 1.5})

        assert math.isclose(first["measurement"], 1.7, rel_tol=0.0, abs_tol=1e-9)
        assert math.isclose(second["measurement"], 1.9, rel_tol=0.0, abs_tol=1e-9)


class TestPhysicalSensorSerialization:
    """Serialization helpers."""

    def test_roundtrip_from_dict_restores_configuration(self) -> None:
        original = PhysicalSensor(
            "level_1",
            sensor_type="level",
            signal_key="current_level",
            measurement_range=(0.0, 100.0),
            measurement_noise=0.0,
            accuracy=0.0,
            drift_rate=0.01,
            response_time=0.5,
            sample_interval=0.25,
            unit="m3",
            output_key="level_signal",
            name="Level Sensor",
        )
        original.add_input("tank_1")
        original.add_output("controller_1")

        restored = PhysicalSensor.from_dict(original.to_dict())

        assert restored.component_id == "level_1"
        assert restored.sensor_type.value == "level"
        assert restored.signal_key == "current_level"
        assert restored.measurement_range == (0.0, 100.0)
        assert restored.drift_rate == 0.01
        assert restored.response_time == 0.5
        assert restored.sample_interval == 0.25
        assert restored.unit == "m3"
        assert restored.output_key == "level_signal"
        assert restored.inputs == ["tank_1"]
        assert restored.outputs == ["controller_1"]
