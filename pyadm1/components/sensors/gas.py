"""
Gas composition sensors for biogas plant simulations.

This module provides configurable virtual gas analyzers for common biogas
components such as methane, carbon dioxide, hydrogen sulfide, oxygen, and
trace gases. The sensor model includes:

- calibration offset within the stated accuracy
- Gaussian measurement noise
- linear calibration drift over time
- first-order response lag
- discrete sampling intervals
- detection limits and range checks
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..base import Component, ComponentType


class GasSensorType(str, Enum):
    """Supported gas sensor types."""

    CH4 = "CH4"
    CO2 = "CO2"
    H2S = "H2S"
    O2 = "O2"
    TRACE_GAS = "trace_gas"


class GasAnalyzerMethod(str, Enum):
    """Representative gas analyzer methods."""

    INFRARED = "infrared"
    CALORIMETRIC = "calorimetric"
    ELECTROCHEMICAL = "electrochemical"
    PARAMAGNETIC = "paramagnetic"
    PHOTOIONIZATION = "photoionization"


_DEFAULT_SENSOR_CONFIG: Dict[GasSensorType, Dict[str, Any]] = {
    GasSensorType.CH4: {
        "signal_key": "CH4",
        "candidate_keys": ("CH4", "ch4", "methane_fraction", "CH4_fraction"),
        "measurement_range": (0.0, 100.0),
        "unit": "%",
        "analyzer_method": GasAnalyzerMethod.INFRARED.value,
        "detection_limit": 0.1,
    },
    GasSensorType.CO2: {
        "signal_key": "CO2",
        "candidate_keys": ("CO2", "co2", "co2_fraction", "CO2_fraction"),
        "measurement_range": (0.0, 100.0),
        "unit": "%",
        "analyzer_method": GasAnalyzerMethod.INFRARED.value,
        "detection_limit": 0.1,
    },
    GasSensorType.H2S: {
        "signal_key": "H2S",
        "candidate_keys": ("H2S", "h2s", "H2S_ppm", "h2s_ppm"),
        "measurement_range": (0.0, 10000.0),
        "unit": "ppm",
        "analyzer_method": GasAnalyzerMethod.ELECTROCHEMICAL.value,
        "detection_limit": 1.0,
    },
    GasSensorType.O2: {
        "signal_key": "O2",
        "candidate_keys": ("O2", "o2", "oxygen_fraction", "O2_fraction"),
        "measurement_range": (0.0, 25.0),
        "unit": "%",
        "analyzer_method": GasAnalyzerMethod.PARAMAGNETIC.value,
        "detection_limit": 0.01,
    },
    GasSensorType.TRACE_GAS: {
        "signal_key": "trace_gas",
        "candidate_keys": ("trace_gas", "trace_gas_ppm", "siloxanes", "VOC"),
        "measurement_range": (0.0, 5000.0),
        "unit": "ppm",
        "analyzer_method": GasAnalyzerMethod.PHOTOIONIZATION.value,
        "detection_limit": 0.5,
    },
}


class GasSensor(Component):
    """
    Generic gas composition sensor with detection limit and response lag.

    Args:
        component_id: Unique component identifier.
        sensor_type: One of ``CH4``, ``CO2``, ``H2S``, ``O2``, or ``trace_gas``.
        analyzer_method: Measurement principle such as infrared or electrochemical.
        signal_key: Input key to read from upstream outputs. If omitted,
            a type-specific default is used.
        measurement_range: Inclusive valid measurement range.
        measurement_noise: Gaussian noise standard deviation in engineering units.
        accuracy: Maximum fixed calibration offset in engineering units.
        drift_rate: Linear drift rate in engineering units per day.
        response_time: First-order lag time constant in days.
        sample_interval: Sampling interval in days. The output is held between samples.
        detection_limit: Values below this threshold are reported as zero.
        unit: Engineering unit label.
        output_key: Namespaced output key for the measured value.
        rng_seed: Optional random seed for deterministic runs.
        name: Human-readable component name.
    """

    def __init__(
        self,
        component_id: str,
        sensor_type: str = "CH4",
        analyzer_method: Optional[str] = None,
        signal_key: Optional[str] = None,
        measurement_range: Optional[Tuple[float, float]] = None,
        measurement_noise: float = 0.0,
        accuracy: float = 0.0,
        drift_rate: float = 0.0,
        response_time: float = 0.0,
        sample_interval: float = 0.0,
        detection_limit: Optional[float] = None,
        unit: Optional[str] = None,
        output_key: Optional[str] = None,
        rng_seed: Optional[int] = None,
        name: Optional[str] = None,
    ):
        super().__init__(component_id, ComponentType.SENSOR, name)

        self.sensor_type = self._parse_sensor_type(sensor_type)
        defaults = _DEFAULT_SENSOR_CONFIG[self.sensor_type]

        self.analyzer_method = self._parse_analyzer_method(analyzer_method or defaults["analyzer_method"])
        self.signal_key = signal_key or defaults["signal_key"]
        self._candidate_keys = (self.signal_key,) if signal_key else defaults["candidate_keys"]
        self.measurement_range = tuple(measurement_range or defaults["measurement_range"])
        self.measurement_noise = float(max(0.0, measurement_noise))
        self.accuracy = float(max(0.0, accuracy))
        self.drift_rate = float(drift_rate)
        self.response_time = float(max(0.0, response_time))
        self.sample_interval = float(max(0.0, sample_interval))
        self.detection_limit = float(max(0.0, detection_limit if detection_limit is not None else defaults["detection_limit"]))
        self.unit = unit or defaults["unit"]
        self.output_key = output_key or f"{component_id}_measurement"

        self._rng = np.random.default_rng(rng_seed)
        self.calibration_offset = self._rng.uniform(-self.accuracy, self.accuracy) if self.accuracy > 0 else 0.0

        self.true_value = np.nan
        self.filtered_value = np.nan
        self.measured_value = np.nan
        self.reported_value = np.nan
        self.drift_offset = 0.0
        self.last_sample_time = -np.inf
        self.is_valid = False
        self.in_range = True
        self.is_detected = False

        self.initialize()

    @staticmethod
    def _parse_sensor_type(sensor_type: str) -> GasSensorType:
        """Normalize user input into a supported gas sensor type."""
        normalized = sensor_type.strip().lower()
        aliases = {
            "ch4": GasSensorType.CH4,
            "methane": GasSensorType.CH4,
            "co2": GasSensorType.CO2,
            "carbon_dioxide": GasSensorType.CO2,
            "h2s": GasSensorType.H2S,
            "hydrogen_sulfide": GasSensorType.H2S,
            "o2": GasSensorType.O2,
            "oxygen": GasSensorType.O2,
            "trace_gas": GasSensorType.TRACE_GAS,
            "tracegas": GasSensorType.TRACE_GAS,
            "trace": GasSensorType.TRACE_GAS,
        }
        if normalized not in aliases:
            raise ValueError(f"Unsupported gas sensor type '{sensor_type}'")
        return aliases[normalized]

    @staticmethod
    def _parse_analyzer_method(analyzer_method: str) -> GasAnalyzerMethod:
        """Normalize analyzer method input."""
        normalized = analyzer_method.strip().lower()
        aliases = {
            "infrared": GasAnalyzerMethod.INFRARED,
            "ndir": GasAnalyzerMethod.INFRARED,
            "calorimetric": GasAnalyzerMethod.CALORIMETRIC,
            "electrochemical": GasAnalyzerMethod.ELECTROCHEMICAL,
            "paramagnetic": GasAnalyzerMethod.PARAMAGNETIC,
            "photoionization": GasAnalyzerMethod.PHOTOIONIZATION,
            "pid": GasAnalyzerMethod.PHOTOIONIZATION,
        }
        if normalized not in aliases:
            raise ValueError(f"Unsupported analyzer method '{analyzer_method}'")
        return aliases[normalized]

    def initialize(self, initial_state: Optional[Dict[str, Any]] = None) -> None:
        """Initialize or restore sensor state."""
        self.true_value = np.nan
        self.filtered_value = np.nan
        self.measured_value = np.nan
        self.reported_value = np.nan
        self.drift_offset = 0.0
        self.last_sample_time = -np.inf
        self.is_valid = False
        self.in_range = True
        self.is_detected = False

        if initial_state:
            self.true_value = float(initial_state.get("true_value", np.nan))
            self.filtered_value = float(initial_state.get("filtered_value", self.true_value))
            self.measured_value = float(initial_state.get("measured_value", self.filtered_value))
            self.reported_value = float(initial_state.get("reported_value", self.measured_value))
            self.drift_offset = float(initial_state.get("drift_offset", 0.0))
            self.last_sample_time = float(initial_state.get("last_sample_time", -np.inf))
            self.is_valid = bool(initial_state.get("is_valid", False))
            self.in_range = bool(initial_state.get("in_range", True))
            self.is_detected = bool(initial_state.get("is_detected", False))

        self.state = {
            "true_value": self.true_value,
            "filtered_value": self.filtered_value,
            "measured_value": self.measured_value,
            "reported_value": self.reported_value,
            "drift_offset": self.drift_offset,
            "calibration_offset": self.calibration_offset,
            "last_sample_time": self.last_sample_time,
            "is_valid": self.is_valid,
            "in_range": self.in_range,
            "is_detected": self.is_detected,
        }

        self.outputs_data = {
            "measurement": float(self.reported_value),
            self.output_key: float(self.reported_value),
            "true_value": float(self.true_value),
            "sensor_type": self.sensor_type.value,
            "analyzer_method": self.analyzer_method.value,
            "signal_key": self.signal_key,
            "unit": self.unit,
            "is_valid": self.is_valid,
            "in_range": self.in_range,
            "is_detected": self.is_detected,
        }

        self._initialized = True

    def step(self, t: float, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Advance the sensor by one simulation step.

        The sensor reads the configured gas signal from upstream outputs, then
        applies response lag, calibration error, drift, noise, detection limit,
        and range limits.
        """
        self.drift_offset += self.drift_rate * dt

        true_value = self._read_true_value(inputs)
        if true_value is not None:
            self.true_value = true_value

        should_sample = self._should_sample(t)
        if should_sample and not np.isnan(self.true_value):
            self.filtered_value = self._apply_response_lag(self.true_value, dt)
            raw_value = self.filtered_value + self.calibration_offset + self.drift_offset
            if self.measurement_noise > 0:
                raw_value += float(self._rng.normal(0.0, self.measurement_noise))

            min_value, max_value = self.measurement_range
            self.in_range = min_value <= raw_value <= max_value
            self.measured_value = float(np.clip(raw_value, min_value, max_value))
            self.is_detected = self.measured_value >= self.detection_limit
            self.reported_value = float(self.measured_value if self.is_detected else 0.0)
            self.last_sample_time = t
            self.is_valid = True
        elif should_sample:
            self.is_valid = False

        self.state.update(
            {
                "true_value": float(self.true_value),
                "filtered_value": float(self.filtered_value),
                "measured_value": float(self.measured_value),
                "reported_value": float(self.reported_value),
                "drift_offset": float(self.drift_offset),
                "calibration_offset": float(self.calibration_offset),
                "last_sample_time": float(self.last_sample_time),
                "is_valid": bool(self.is_valid),
                "in_range": bool(self.in_range),
                "is_detected": bool(self.is_detected),
            }
        )

        self.outputs_data = {
            "measurement": float(self.reported_value),
            self.output_key: float(self.reported_value),
            "true_value": float(self.true_value),
            "sensor_type": self.sensor_type.value,
            "analyzer_method": self.analyzer_method.value,
            "signal_key": self.signal_key,
            "unit": self.unit,
            "drift_offset": float(self.drift_offset),
            "is_valid": bool(self.is_valid),
            "in_range": bool(self.in_range),
            "is_detected": bool(self.is_detected),
        }
        return self.outputs_data

    def _read_true_value(self, inputs: Dict[str, Any]) -> Optional[float]:
        """Resolve the measured gas variable from upstream inputs."""
        for key in self._candidate_keys:
            if key in inputs:
                try:
                    return float(inputs[key])
                except (TypeError, ValueError):
                    return None
        return None

    def _should_sample(self, t: float) -> bool:
        """Return True when the next sample is due."""
        if self.sample_interval <= 0:
            return True
        return (t - self.last_sample_time) >= self.sample_interval

    def _apply_response_lag(self, true_value: float, dt: float) -> float:
        """Apply first-order lag to the true gas signal."""
        if np.isnan(self.filtered_value) or self.response_time <= 0:
            return true_value

        alpha = min(1.0, dt / max(self.response_time, 1.0e-12))
        return self.filtered_value + alpha * (true_value - self.filtered_value)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize sensor configuration and state."""
        return {
            "component_id": self.component_id,
            "component_type": self.component_type.value,
            "name": self.name,
            "sensor_type": self.sensor_type.value,
            "analyzer_method": self.analyzer_method.value,
            "signal_key": self.signal_key,
            "measurement_range": list(self.measurement_range),
            "measurement_noise": self.measurement_noise,
            "accuracy": self.accuracy,
            "drift_rate": self.drift_rate,
            "response_time": self.response_time,
            "sample_interval": self.sample_interval,
            "detection_limit": self.detection_limit,
            "unit": self.unit,
            "output_key": self.output_key,
            "state": self.state,
            "inputs": self.inputs,
            "outputs": self.outputs,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "GasSensor":
        """Create sensor from serialized configuration."""
        sensor = cls(
            component_id=config["component_id"],
            sensor_type=config.get("sensor_type", "CH4"),
            analyzer_method=config.get("analyzer_method"),
            signal_key=config.get("signal_key"),
            measurement_range=tuple(config.get("measurement_range", (0.0, 100.0))),
            measurement_noise=config.get("measurement_noise", 0.0),
            accuracy=config.get("accuracy", 0.0),
            drift_rate=config.get("drift_rate", 0.0),
            response_time=config.get("response_time", 0.0),
            sample_interval=config.get("sample_interval", 0.0),
            detection_limit=config.get("detection_limit"),
            unit=config.get("unit"),
            output_key=config.get("output_key"),
            name=config.get("name"),
        )

        if "state" in config:
            sensor.initialize(config["state"])

        for input_id in config.get("inputs", []):
            sensor.add_input(input_id)
        for output_id in config.get("outputs", []):
            sensor.add_output(output_id)

        return sensor
