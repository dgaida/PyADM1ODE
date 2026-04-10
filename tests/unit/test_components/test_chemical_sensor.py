# -*- coding: utf-8 -*-
"""Unit tests for chemical sensor components."""

import math

from pyadm1.components.base import ComponentType
from pyadm1.components.sensors import ChemicalSensor


class TestChemicalSensorInitialization:
    """Construction and initialization behavior."""

    def test_initialization_sets_expected_defaults(self) -> None:
        sensor = ChemicalSensor("vfa_1")

        assert sensor.component_id == "vfa_1"
        assert sensor.component_type == ComponentType.SENSOR
        assert sensor.sensor_type.value == "VFA"
        assert sensor.signal_key == "VFA"
        assert sensor.analyzer_method.value == "online_titration"
        assert sensor.output_key == "vfa_1_measurement"


class TestChemicalSensorStep:
    """Analyzer signal processing behavior."""

    def test_vfa_sensor_applies_measurement_delay(self) -> None:
        sensor = ChemicalSensor(
            "vfa_1",
            sensor_type="VFA",
            measurement_delay=0.5,
            sample_interval=0.5,
            measurement_noise=0.0,
            accuracy=0.0,
        )

        first = sensor.step(t=0.0, dt=0.25, inputs={"VFA": 4.2})
        second = sensor.step(t=0.25, dt=0.25, inputs={"VFA": 6.0})
        third = sensor.step(t=0.5, dt=0.25, inputs={"VFA": 6.0})

        assert math.isnan(first["measurement"])
        assert math.isnan(second["measurement"])
        assert math.isclose(third["measurement"], 4.2, rel_tol=0.0, abs_tol=1e-9)

    def test_ammonia_sensor_accepts_tan_input(self) -> None:
        sensor = ChemicalSensor("nh3_1", sensor_type="ammonia", measurement_noise=0.0, accuracy=0.0)

        result = sensor.step(t=0.0, dt=1.0 / 24.0, inputs={"TAN": 1.75})

        assert result["measurement"] == 1.75
        assert result["true_value"] == 1.75

    def test_below_detection_limit_reports_zero(self) -> None:
        sensor = ChemicalSensor(
            "cod_1",
            sensor_type="COD",
            measurement_noise=0.0,
            accuracy=0.0,
            detection_limit=0.5,
        )

        result = sensor.step(t=0.0, dt=1.0 / 24.0, inputs={"COD": 0.2})

        assert result["measurement"] == 0.0
        assert result["is_detected"] is False

    def test_drift_rate_accumulates_over_time(self) -> None:
        sensor = ChemicalSensor(
            "nut_1",
            sensor_type="nutrients",
            signal_key="phosphate",
            measurement_noise=0.0,
            accuracy=0.0,
            drift_rate=2.0,
            detection_limit=0.0,
        )

        first = sensor.step(t=0.0, dt=1.0, inputs={"phosphate": 10.0})
        second = sensor.step(t=1.0, dt=1.0, inputs={"phosphate": 10.0})

        assert math.isclose(first["measurement"], 12.0, rel_tol=0.0, abs_tol=1e-9)
        assert math.isclose(second["measurement"], 14.0, rel_tol=0.0, abs_tol=1e-9)


class TestChemicalSensorSerialization:
    """Serialization helpers."""

    def test_roundtrip_from_dict_restores_configuration(self) -> None:
        original = ChemicalSensor(
            "vfa_1",
            sensor_type="VFA",
            analyzer_method="gc",
            signal_key="VFA",
            measurement_range=(0.0, 20.0),
            measurement_noise=0.0,
            accuracy=0.0,
            drift_rate=0.02,
            sample_interval=0.5,
            measurement_delay=0.25,
            detection_limit=0.1,
            unit="g/L",
            output_key="vfa_signal",
            name="VFA Analyzer",
        )
        original.add_input("digester_1")
        original.add_output("controller_1")
        original.step(t=0.0, dt=0.25, inputs={"VFA": 3.5})

        restored = ChemicalSensor.from_dict(original.to_dict())

        assert restored.component_id == "vfa_1"
        assert restored.sensor_type.value == "VFA"
        assert restored.analyzer_method.value == "gas_chromatography"
        assert restored.signal_key == "VFA"
        assert restored.measurement_range == (0.0, 20.0)
        assert restored.drift_rate == 0.02
        assert restored.sample_interval == 0.5
        assert restored.measurement_delay == 0.25
        assert restored.detection_limit == 0.1
        assert restored.unit == "g/L"
        assert restored.output_key == "vfa_signal"
        assert restored.inputs == ["digester_1"]
        assert restored.outputs == ["controller_1"]
        assert len(restored.state["pending_samples"]) == 1
