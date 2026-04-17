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
- optional analysis delay (batch-mode GC analyzers)
- detection limits and range checks
- additive cross-sensitivity bias from interfering gas signals

References (default measurement parameters):
    [1]  QED Technologies BIOGAS 5000 datasheet — NDIR CH4/CO2, ±0.5% accuracy.
    [2]  Bürkert Type 8025 NDIR gas module datasheet — ±2% FS, T90 ≤ 30 s.
    [3]  ABB AO2000 / LIMAS11 process analyzer spec sheet — NDIR T90 < 30 s (typ).
    [4]  Awite AwiFLEX calorimetric/TCD biogas analyzer — ±0.5 vol%, response 10–20 s.
    [5]  Agilent 490 Micro GC biogas application note (pub. 5990-9517EN) —
         full biogas cycle (CH4, CO2, H2S, N2, O2) in < 3–5 min.
    [6]  Aeroqual EHT electrochemical H2S sensor datasheet — 0–100 ppm range,
         ±0.1 ppm / ±10% reading, T90 ≤ 30 s.
    [7]  ATI B12 Series H2S transmitter — 0–50 ppm, 0.1 ppm resolution, T90 ≤ 30 s.
    [8]  EPA Method 15 (H2S/sulfur by GC-FPD); Lamichhane et al. (2021)
         PMC8037836 — biogas GC-FPD, RSD 2.78–5.68%, LOD ~0.2–0.5 ppm.
    [9]  ABB Magnos28 paramagnetic O2 analyzer — T90 3–10 s, ±0.5% of span.
    [10] Siemens OXYMAT 6 paramagnetic O2 — T90 2.5–35 s (typ. 2.5 s), ±0.5% FS.
    [11] Honeywell/Semeatech electrochemical O2 cell, Technical Note 114 (2018)
         — ±5% FS, T90 < 10 s.
    [12] ION Science Tiger PID — detection from 0.01 ppm, T90 ≈ 2–3 s;
         RAE Systems MiniRAE 3000 — T90 2 s, ±2–5 ppm field accuracy.
    [13] García-Pérez et al. (2021) PMC8037836 — GC-MS siloxane analysis in
         biogas, RSD 2.78–9.52%, cycle time 10–30 min.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._base import AbstractSensor


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
    GAS_CHROMATOGRAPHY = "gas_chromatography"


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

# Measurement characteristics per (sensor_type, analyzer_method).
# response_time and measurement_delay are in days.
#   response_time    — first-order lag on the live signal (continuous) or on
#                      the matured result (batch); e.g. 10 s = 10.0 / 86400.
#   measurement_delay — time from sampling to result release (batch only);
#                       > 0 activates the pending-samples queue.
# These values apply when the user does not supply an explicit override.
_ANALYZER_METHOD_DEFAULTS: Dict[GasSensorType, Dict[GasAnalyzerMethod, Dict[str, float]]] = {
    GasSensorType.CH4: {
        GasAnalyzerMethod.INFRARED: {
            "measurement_noise": 0.3,  # % std dev — NDIR accuracy ±0.5% FS → σ ≈ 0.3% [1,2,3]
            "response_time": 30.0 / 86400,  # T90 ≤ 30 s — Bürkert 8025, ABB AO2000 [2,3]
            "measurement_delay": 0.0,
            "detection_limit": 0.1,  # % — typical NDIR baseline threshold [1]
        },
        GasAnalyzerMethod.CALORIMETRIC: {
            "measurement_noise": 1.0,  # % std dev — TCD/WLD ±2% FS → σ ≈ 1% [4]
            "response_time": 20.0 / 86400,  # 20 s — thermal conductivity cell equilibration [4]
            "measurement_delay": 0.0,
            "detection_limit": 0.5,  # % — TCD baseline noise limit [4]
        },
        GasAnalyzerMethod.GAS_CHROMATOGRAPHY: {
            "measurement_noise": 0.1,  # % std dev — micro-GC repeatability RSD < 1% [5]
            "response_time": 0.0,  # result is instantaneous once run completes
            "measurement_delay": 5.0 / 1440,  # 5 min — Agilent 490 Micro GC biogas cycle [5]
            "detection_limit": 0.05,
        },
    },
    GasSensorType.CO2: {
        GasAnalyzerMethod.INFRARED: {
            "measurement_noise": 0.3,  # % std dev — same NDIR instrument as CH4 [1,2,3]
            "response_time": 30.0 / 86400,  # T90 ≤ 30 s [2,3]
            "measurement_delay": 0.0,
            "detection_limit": 0.1,
        },
        GasAnalyzerMethod.CALORIMETRIC: {
            "measurement_noise": 1.0,  # % std dev [4]
            "response_time": 20.0 / 86400,  # 20 s [4]
            "measurement_delay": 0.0,
            "detection_limit": 0.5,
        },
        GasAnalyzerMethod.GAS_CHROMATOGRAPHY: {
            "measurement_noise": 0.1,  # % std dev [5]
            "response_time": 0.0,
            "measurement_delay": 5.0 / 1440,  # 5 min [5]
            "detection_limit": 0.05,
        },
    },
    GasSensorType.H2S: {
        GasAnalyzerMethod.ELECTROCHEMICAL: {
            "measurement_noise": 10.0,  # ppm std dev — ≈10% at 100 ppm; ±10% reading spec [6,7]
            "response_time": 30.0 / 86400,  # T90 ≤ 30 s — Aeroqual EHT, ATI B12, Dräger [6,7]
            "measurement_delay": 0.0,
            "detection_limit": 1.0,  # ppm — practical process biogas threshold [6,7]
        },
        GasAnalyzerMethod.GAS_CHROMATOGRAPHY: {
            "measurement_noise": 5.0,  # ppm std dev — GC-FPD RSD ~3% at 150 ppm H2S [8]
            "response_time": 0.0,
            "measurement_delay": 15.0 / 1440,  # 15 min — GC-FPD with dedicated sulfur column [8]
            "detection_limit": 0.5,  # ppm — FPD LOD ~0.2–0.5 ppm per EPA Method 15 [8]
        },
    },
    GasSensorType.O2: {
        GasAnalyzerMethod.PARAMAGNETIC: {
            "measurement_noise": 0.05,  # % std dev — ±0.5% FS at 0–25% → σ ≈ 0.06% [9,10]
            "response_time": 10.0 / 86400,  # T90 3–10 s — ABB Magnos28, Siemens OXYMAT 6 [9,10]
            "measurement_delay": 0.0,
            "detection_limit": 0.01,
        },
        GasAnalyzerMethod.ELECTROCHEMICAL: {
            "measurement_noise": 0.5,  # % std dev — ±5% FS at 0–25% → σ ≈ 0.6% [11]
            "response_time": 10.0 / 86400,  # T90 < 10 s [11]
            "measurement_delay": 0.0,
            "detection_limit": 0.05,
        },
    },
    GasSensorType.TRACE_GAS: {
        GasAnalyzerMethod.PHOTOIONIZATION: {
            "measurement_noise": 2.0,  # ppm std dev — PID field accuracy ±2–5 ppm [12]
            "response_time": 5.0 / 86400,  # T90 ≈ 2–3 s — ION Science Tiger, RAE MiniRAE [12]
            "measurement_delay": 0.0,
            "detection_limit": 0.5,
        },
        GasAnalyzerMethod.GAS_CHROMATOGRAPHY: {
            "measurement_noise": 0.2,  # ppm std dev — GC-MS RSD ~5% [13]
            "response_time": 0.0,
            "measurement_delay": 10.0 / 1440,  # 10–30 min run — online GC siloxane analysis [13]
            "detection_limit": 0.1,
        },
    },
}


class GasSensor(AbstractSensor):
    """
    Generic gas composition sensor with detection limit and response lag.

    Supports two operating modes selected automatically by *measurement_delay*:

    **Continuous mode** (``measurement_delay = 0``, default for NDIR,
    electrochemical, paramagnetic, PID): the response lag is applied to the
    live input signal every step, matching how analog gas analyzers smooth
    their output.

    **Batch mode** (``measurement_delay > 0``, e.g. gas chromatographs):
    a sample is queued at each sampling interval and released after the
    analysis delay elapses.  The response lag is then applied continuously
    toward the latest released result, modelling the instrument output settling.

    When *measurement_noise*, *response_time*, *measurement_delay*, or
    *detection_limit* are omitted, they are taken from
    ``_ANALYZER_METHOD_DEFAULTS`` for the chosen *analyzer_method*.

    Args:
        component_id: Unique component identifier.
        sensor_type: One of ``CH4``, ``CO2``, ``H2S``, ``O2``, or ``trace_gas``.
        analyzer_method: Measurement principle — ``infrared``, ``calorimetric``,
            ``electrochemical``, ``paramagnetic``, ``photoionization``, or
            ``gas_chromatography``.
        signal_key: Input key to read from upstream outputs.
        measurement_range: Inclusive valid measurement range.
        measurement_noise: Gaussian noise std dev in engineering units.
            ``None`` uses the analyzer-method default.
        accuracy: Maximum fixed calibration offset in engineering units.
        drift_rate: Linear drift rate in engineering units per day.
        response_time: First-order lag time constant in days.
            ``None`` uses the analyzer-method default.
        sample_interval: Sampling interval in days.
        measurement_delay: Delay from sampling to result release in days.
            ``None`` uses the analyzer-method default.  Values > 0 activate
            batch mode.
        detection_limit: Values below this threshold are reported as zero.
            ``None`` uses the analyzer-method default.
        cross_sensitivity: Mapping from input signal key to interference
            coefficient (same engineering units as the sensor output per unit
            of the interfering gas).  The total additive bias is
            ``sum(coeff * inputs[key] for key, coeff in cross_sensitivity.items())``.
            Applied after response lag and before calibration errors.
            Example — CO2 interference on a CH4 NDIR channel:
            ``{"CO2": -0.003}`` (−0.3 % CH4 per 1 % CO2).
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
        measurement_noise: Optional[float] = None,
        accuracy: float = 0.0,
        drift_rate: float = 0.0,
        response_time: Optional[float] = None,
        sample_interval: float = 0.0,
        measurement_delay: Optional[float] = None,
        detection_limit: Optional[float] = None,
        cross_sensitivity: Optional[Dict[str, float]] = None,
        unit: Optional[str] = None,
        output_key: Optional[str] = None,
        rng_seed: Optional[int] = None,
        name: Optional[str] = None,
    ):
        self.sensor_type = self._parse_sensor_type(sensor_type)
        defaults = _DEFAULT_SENSOR_CONFIG[self.sensor_type]

        self.analyzer_method = self._parse_analyzer_method(analyzer_method or defaults["analyzer_method"])

        method_defaults = _ANALYZER_METHOD_DEFAULTS.get(self.sensor_type, {}).get(self.analyzer_method, {})

        resolved_signal_key = signal_key or defaults["signal_key"]
        resolved_candidate_keys = (resolved_signal_key,) if signal_key else defaults["candidate_keys"]
        resolved_noise = measurement_noise if measurement_noise is not None else method_defaults.get("measurement_noise", 0.0)
        resolved_response_time = response_time if response_time is not None else method_defaults.get("response_time", 0.0)
        resolved_delay = measurement_delay if measurement_delay is not None else method_defaults.get("measurement_delay", 0.0)
        resolved_detection_limit = (
            detection_limit
            if detection_limit is not None
            else method_defaults.get("detection_limit", defaults["detection_limit"])
        )

        super().__init__(
            component_id=component_id,
            signal_key=resolved_signal_key,
            candidate_keys=resolved_candidate_keys,
            measurement_range=measurement_range or defaults["measurement_range"],
            measurement_noise=resolved_noise,
            accuracy=accuracy,
            drift_rate=drift_rate,
            sample_interval=sample_interval,
            unit=unit or defaults["unit"],
            output_key=output_key,
            rng_seed=rng_seed,
            name=name,
        )

        self.response_time = float(max(0.0, resolved_response_time))
        self.measurement_delay = float(max(0.0, resolved_delay))
        self.detection_limit = float(max(0.0, resolved_detection_limit))
        self.cross_sensitivity: Dict[str, float] = dict(cross_sensitivity) if cross_sensitivity else {}

        self.filtered_value: float = np.nan
        self.reported_value: float = np.nan
        self.is_detected: bool = False
        self._pending_samples: List[Tuple[float, float]] = []

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
            "gas_chromatography": GasAnalyzerMethod.GAS_CHROMATOGRAPHY,
            "gc": GasAnalyzerMethod.GAS_CHROMATOGRAPHY,
        }
        if normalized not in aliases:
            raise ValueError(f"Unsupported analyzer method '{analyzer_method}'")
        return aliases[normalized]

    def _apply_cross_sensitivity(self, value: float, inputs: Dict[str, Any]) -> float:
        """Add interference bias from other signals in *inputs*."""
        if not self.cross_sensitivity:
            return value
        bias = 0.0
        for key, coefficient in self.cross_sensitivity.items():
            raw = inputs.get(key)
            if raw is None:
                continue
            try:
                bias += coefficient * float(raw)
            except (TypeError, ValueError):
                continue
        return value + bias

    def initialize(self, initial_state: Optional[Dict[str, Any]] = None) -> None:
        """Initialize or restore sensor state."""
        super().initialize(initial_state)

        self.filtered_value = np.nan
        self.reported_value = np.nan
        self.is_detected = False
        self._pending_samples = []

        if initial_state:
            self.filtered_value = float(initial_state.get("filtered_value", self.true_value))
            self.reported_value = float(initial_state.get("reported_value", self.measured_value))
            self.is_detected = bool(initial_state.get("is_detected", False))
            pending = initial_state.get("pending_samples", [])
            self._pending_samples = [(float(r), float(s)) for r, s in pending]

        self.state = {
            **self._base_state_dict(),
            "filtered_value": float(self.filtered_value),
            "reported_value": float(self.reported_value),
            "is_detected": bool(self.is_detected),
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
        """
        Advance the sensor by one simulation step.

        In **continuous mode** (``measurement_delay = 0``) the response lag is
        applied to the live input signal before adding errors, matching the
        analog behaviour of NDIR and electrochemical sensors.

        In **batch mode** (``measurement_delay > 0``) a sample is queued and
        released after the analysis delay, then the response lag tracks the
        released result, matching the discrete output of a GC analyzer.
        """
        self.drift_offset += self.drift_rate * dt

        true_value = self._read_true_value(inputs)
        if true_value is not None:
            self.true_value = true_value

        should_sample = self._should_sample(t)

        if self.measurement_delay > 0:
            # ---- Batch mode (GC-type) ----------------------------------------
            if should_sample and not np.isnan(self.true_value):
                self._pending_samples.append((t + self.measurement_delay, self.true_value))
                self.last_sample_time = t
            elif should_sample:
                self.is_valid = False

            matured = self._pop_latest_ready_sample(t)
            if matured is not None:
                cross_corrected = self._apply_cross_sensitivity(matured, inputs)
                raw_value = self._apply_errors(cross_corrected)
                self.measured_value, self.in_range = self._clamp_to_range(raw_value)
                self.is_valid = True

            # Lag applied continuously toward the latest released result.
            if not np.isnan(self.measured_value):
                self.filtered_value = self._apply_response_lag(
                    self.measured_value, self.filtered_value, self.response_time, dt
                )
                self.is_detected = self.filtered_value >= self.detection_limit
                self.reported_value = float(self.filtered_value if self.is_detected else 0.0)

        else:
            # ---- Continuous mode (NDIR, electrochemical, PID, …) --------------
            if should_sample and not np.isnan(self.true_value):
                self.filtered_value = self._apply_response_lag(self.true_value, self.filtered_value, self.response_time, dt)
                cross_corrected = self._apply_cross_sensitivity(self.filtered_value, inputs)
                raw_value = self._apply_errors(cross_corrected)
                self.measured_value, self.in_range = self._clamp_to_range(raw_value)
                self.is_detected = self.measured_value >= self.detection_limit
                self.reported_value = float(self.measured_value if self.is_detected else 0.0)
                self.last_sample_time = t
                self.is_valid = True
            elif should_sample:
                self.is_valid = False

        self.state.update(
            {
                **self._base_state_dict(),
                "filtered_value": float(self.filtered_value),
                "reported_value": float(self.reported_value),
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
        """Serialize sensor configuration and state."""
        return {
            **self._base_config_dict(),
            "sensor_type": self.sensor_type.value,
            "analyzer_method": self.analyzer_method.value,
            "response_time": self.response_time,
            "measurement_delay": self.measurement_delay,
            "detection_limit": self.detection_limit,
            "cross_sensitivity": dict(self.cross_sensitivity),
            "state": self.state,
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
            measurement_noise=config.get("measurement_noise"),
            accuracy=config.get("accuracy", 0.0),
            drift_rate=config.get("drift_rate", 0.0),
            response_time=config.get("response_time"),
            sample_interval=config.get("sample_interval", 0.0),
            measurement_delay=config.get("measurement_delay"),
            detection_limit=config.get("detection_limit"),
            cross_sensitivity=config.get("cross_sensitivity"),
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
