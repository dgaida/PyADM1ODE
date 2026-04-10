"""
Chemical process sensors for biogas plant simulations.

This module provides configurable virtual analyzers for liquid-phase chemical
measurements such as volatile fatty acids, ammonia, COD, and nutrients.
Compared to physical sensors, chemical analyzers commonly have an explicit
sampling and analysis delay, so this model includes:

- calibration offset within the stated accuracy
- Gaussian measurement noise
- linear calibration drift over time
- discrete sampling intervals
- configurable analysis delay
- detection limits and range checks
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..base import Component, ComponentType


class ChemicalSensorType(str, Enum):
    """Supported chemical sensor types."""

    VFA = "VFA"
    AMMONIA = "ammonia"
    COD = "COD"
    NUTRIENTS = "nutrients"


class ChemicalAnalyzerMethod(str, Enum):
    """Representative analyzer methods."""

    ONLINE_TITRATION = "online_titration"
    GAS_CHROMATOGRAPHY = "gas_chromatography"
    ION_SELECTIVE = "ion_selective"
    SPECTROSCOPY = "spectroscopy"
    COLORIMETRIC = "colorimetric"


_DEFAULT_SENSOR_CONFIG: Dict[ChemicalSensorType, Dict[str, Any]] = {
    ChemicalSensorType.VFA: {
        "signal_key": "VFA",
        "candidate_keys": ("VFA",),
        "measurement_range": (0.0, 25.0),
        "unit": "g/L",
        "analyzer_method": ChemicalAnalyzerMethod.ONLINE_TITRATION.value,
        "detection_limit": 0.05,
    },
    ChemicalSensorType.AMMONIA: {
        "signal_key": "ammonia",
        "candidate_keys": ("ammonia", "NH3", "TAN", "S_nh3", "S_nh4"),
        "measurement_range": (0.0, 10.0),
        "unit": "g/L",
        "analyzer_method": ChemicalAnalyzerMethod.ION_SELECTIVE.value,
        "detection_limit": 0.01,
    },
    ChemicalSensorType.COD: {
        "signal_key": "COD",
        "candidate_keys": ("COD", "cod"),
        "measurement_range": (0.0, 200.0),
        "unit": "g/L",
        "analyzer_method": ChemicalAnalyzerMethod.SPECTROSCOPY.value,
        "detection_limit": 0.10,
    },
    ChemicalSensorType.NUTRIENTS: {
        "signal_key": "nutrients",
        "candidate_keys": ("nutrients", "nitrogen", "phosphorus", "phosphate", "NH4_N", "PO4_P"),
        "measurement_range": (0.0, 10000.0),
        "unit": "mg/L",
        "analyzer_method": ChemicalAnalyzerMethod.COLORIMETRIC.value,
        "detection_limit": 1.0,
    },
}


class ChemicalSensor(Component):
    """
    Generic chemical analyzer with sampling delay and drift.

    Args:
        component_id: Unique component identifier.
        sensor_type: One of ``VFA``, ``ammonia``, ``COD``, or ``nutrients``.
        analyzer_method: Measurement principle such as titration or spectroscopy.
        signal_key: Input key to read from upstream outputs. If omitted,
            a type-specific default is used.
        measurement_range: Inclusive valid measurement range.
        measurement_noise: Gaussian noise standard deviation in engineering units.
        accuracy: Maximum fixed calibration offset in engineering units.
        drift_rate: Linear drift rate in engineering units per day.
        sample_interval: Sampling interval in days.
        measurement_delay: Delay between sampling and reported result in days.
        detection_limit: Values below this threshold are reported as zero.
        unit: Engineering unit label.
        output_key: Namespaced output key for the measured value.
        rng_seed: Optional random seed for deterministic runs.
        name: Human-readable component name.
    """

    def __init__(
        self,
        component_id: str,
        sensor_type: str = "VFA",
        analyzer_method: Optional[str] = None,
        signal_key: Optional[str] = None,
        measurement_range: Optional[Tuple[float, float]] = None,
        measurement_noise: float = 0.0,
        accuracy: float = 0.0,
        drift_rate: float = 0.0,
        sample_interval: float = 0.0,
        measurement_delay: float = 0.0,
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
        self.sample_interval = float(max(0.0, sample_interval))
        self.measurement_delay = float(max(0.0, measurement_delay))
        self.detection_limit = float(max(0.0, detection_limit if detection_limit is not None else defaults["detection_limit"]))
        self.unit = unit or defaults["unit"]
        self.output_key = output_key or f"{component_id}_measurement"

        self._rng = np.random.default_rng(rng_seed)
        self.calibration_offset = self._rng.uniform(-self.accuracy, self.accuracy) if self.accuracy > 0 else 0.0

        self.true_value = np.nan
        self.measured_value = np.nan
        self.reported_value = np.nan
        self.drift_offset = 0.0
        self.last_sample_time = -np.inf
        self.last_result_time = -np.inf
        self.is_valid = False
        self.in_range = True
        self.is_detected = False
        self._pending_samples: List[Tuple[float, float]] = []

        self.initialize()

    @staticmethod
    def _parse_sensor_type(sensor_type: str) -> ChemicalSensorType:
        """Normalize user input into a supported chemical sensor type."""
        normalized = sensor_type.strip().lower()
        aliases = {
            "vfa": ChemicalSensorType.VFA,
            "volatile_fatty_acids": ChemicalSensorType.VFA,
            "ammonia": ChemicalSensorType.AMMONIA,
            "nh3": ChemicalSensorType.AMMONIA,
            "tan": ChemicalSensorType.AMMONIA,
            "cod": ChemicalSensorType.COD,
            "nutrients": ChemicalSensorType.NUTRIENTS,
            "nutrient": ChemicalSensorType.NUTRIENTS,
        }
        if normalized not in aliases:
            raise ValueError(f"Unsupported chemical sensor type '{sensor_type}'")
        return aliases[normalized]

    @staticmethod
    def _parse_analyzer_method(analyzer_method: str) -> ChemicalAnalyzerMethod:
        """Normalize analyzer method input."""
        normalized = analyzer_method.strip().lower()
        aliases = {
            "online_titration": ChemicalAnalyzerMethod.ONLINE_TITRATION,
            "titration": ChemicalAnalyzerMethod.ONLINE_TITRATION,
            "gas_chromatography": ChemicalAnalyzerMethod.GAS_CHROMATOGRAPHY,
            "gc": ChemicalAnalyzerMethod.GAS_CHROMATOGRAPHY,
            "ion_selective": ChemicalAnalyzerMethod.ION_SELECTIVE,
            "ion_selective_electrode": ChemicalAnalyzerMethod.ION_SELECTIVE,
            "ise": ChemicalAnalyzerMethod.ION_SELECTIVE,
            "spectroscopy": ChemicalAnalyzerMethod.SPECTROSCOPY,
            "spectroscopic": ChemicalAnalyzerMethod.SPECTROSCOPY,
            "colorimetric": ChemicalAnalyzerMethod.COLORIMETRIC,
        }
        if normalized not in aliases:
            raise ValueError(f"Unsupported analyzer method '{analyzer_method}'")
        return aliases[normalized]

    def initialize(self, initial_state: Optional[Dict[str, Any]] = None) -> None:
        """Initialize or restore analyzer state."""
        self.true_value = np.nan
        self.measured_value = np.nan
        self.reported_value = np.nan
        self.drift_offset = 0.0
        self.last_sample_time = -np.inf
        self.last_result_time = -np.inf
        self.is_valid = False
        self.in_range = True
        self.is_detected = False
        self._pending_samples = []

        if initial_state:
            self.true_value = float(initial_state.get("true_value", np.nan))
            self.measured_value = float(initial_state.get("measured_value", np.nan))
            self.reported_value = float(initial_state.get("reported_value", self.measured_value))
            self.drift_offset = float(initial_state.get("drift_offset", 0.0))
            self.last_sample_time = float(initial_state.get("last_sample_time", -np.inf))
            self.last_result_time = float(initial_state.get("last_result_time", -np.inf))
            self.is_valid = bool(initial_state.get("is_valid", False))
            self.in_range = bool(initial_state.get("in_range", True))
            self.is_detected = bool(initial_state.get("is_detected", False))
            pending = initial_state.get("pending_samples", [])
            self._pending_samples = [(float(release_t), float(sample_value)) for release_t, sample_value in pending]

        self.state = {
            "true_value": self.true_value,
            "measured_value": self.measured_value,
            "reported_value": self.reported_value,
            "drift_offset": self.drift_offset,
            "calibration_offset": self.calibration_offset,
            "last_sample_time": self.last_sample_time,
            "last_result_time": self.last_result_time,
            "is_valid": self.is_valid,
            "in_range": self.in_range,
            "is_detected": self.is_detected,
            "pending_samples": list(self._pending_samples),
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
        """Advance the analyzer by one simulation step."""
        self.drift_offset += self.drift_rate * dt

        true_value = self._read_true_value(inputs)
        if true_value is not None:
            self.true_value = true_value

        if self._should_sample(t) and not np.isnan(self.true_value):
            self._pending_samples.append((t + self.measurement_delay, self.true_value))
            self.last_sample_time = t

        matured_sample = self._pop_latest_ready_sample(t)
        if matured_sample is not None:
            raw_value = matured_sample + self.calibration_offset + self.drift_offset
            if self.measurement_noise > 0:
                raw_value += float(self._rng.normal(0.0, self.measurement_noise))

            min_value, max_value = self.measurement_range
            self.in_range = min_value <= raw_value <= max_value
            self.measured_value = float(np.clip(raw_value, min_value, max_value))
            self.is_detected = self.measured_value >= self.detection_limit
            self.reported_value = float(self.measured_value if self.is_detected else 0.0)
            self.last_result_time = t
            self.is_valid = True

        self.state.update(
            {
                "true_value": float(self.true_value),
                "measured_value": float(self.measured_value),
                "reported_value": float(self.reported_value),
                "drift_offset": float(self.drift_offset),
                "calibration_offset": float(self.calibration_offset),
                "last_sample_time": float(self.last_sample_time),
                "last_result_time": float(self.last_result_time),
                "is_valid": bool(self.is_valid),
                "in_range": bool(self.in_range),
                "is_detected": bool(self.is_detected),
                "pending_samples": list(self._pending_samples),
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
        """Resolve the measured analyte from upstream inputs."""
        for key in self._candidate_keys:
            if key in inputs:
                try:
                    return float(inputs[key])
                except (TypeError, ValueError):
                    return None
        return None

    def _should_sample(self, t: float) -> bool:
        """Return True when a new sample is due."""
        if self.sample_interval <= 0:
            return True
        return (t - self.last_sample_time) >= self.sample_interval

    def _pop_latest_ready_sample(self, t: float) -> Optional[float]:
        """Return the latest sample whose analysis delay has elapsed."""
        latest_ready: Optional[float] = None
        pending: List[Tuple[float, float]] = []
        for release_time, sample_value in self._pending_samples:
            if release_time <= t + 1e-12:
                latest_ready = sample_value
            else:
                pending.append((release_time, sample_value))
        self._pending_samples = pending
        return latest_ready

    def to_dict(self) -> Dict[str, Any]:
        """Serialize analyzer configuration and state."""
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
            "sample_interval": self.sample_interval,
            "measurement_delay": self.measurement_delay,
            "detection_limit": self.detection_limit,
            "unit": self.unit,
            "output_key": self.output_key,
            "state": self.state,
            "inputs": self.inputs,
            "outputs": self.outputs,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ChemicalSensor":
        """Create analyzer from serialized configuration."""
        sensor = cls(
            component_id=config["component_id"],
            sensor_type=config.get("sensor_type", "VFA"),
            analyzer_method=config.get("analyzer_method"),
            signal_key=config.get("signal_key"),
            measurement_range=tuple(config.get("measurement_range", (0.0, 25.0))),
            measurement_noise=config.get("measurement_noise", 0.0),
            accuracy=config.get("accuracy", 0.0),
            drift_rate=config.get("drift_rate", 0.0),
            sample_interval=config.get("sample_interval", 0.0),
            measurement_delay=config.get("measurement_delay", 0.0),
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
