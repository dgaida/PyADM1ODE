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

References (default measurement parameters):
    [1]  Hach EZ7200 / EZ7250 VFA online analyzer datasheet — FOS/TAC titration,
         ±3% FS, cycle time 3–9 min, detection limit ≤ 10 mg/L.
    [2]  Diamantis et al. (2014) Water Research 48:266 — online FOS/TAC comparison
         in full-scale digesters; RSD < 5%, analysis cycle 5–15 min.
    [3]  Ramos-Suárez et al. (2022) PMC9412176 — GC-FID VFA in biogas effluent;
         LOD 0.91–2.25 mg/L, RSD < 3%, run time 25–40 min.
    [4]  YSI ProDSS NH4-ISE manual (Dec 2015) — ±10% of reading, LOD 0.01 mg/L,
         response ≤ 30 s; APHA Standard Methods 4500-NH3 (23rd ed., 2017).
    [5]  EPA Method 350.1 — semi-automated colorimetric ammonia (Indophenol Blue);
         online heated manifold analyzers (Hach Amtax, Seal AA500) complete
         color development in 8–15 min.
    [6]  Peng et al. (2022) PMC9054276 — UV-Vis COD proxy in wastewater, ±4%
         for 0–60 mg/L range, ±3% for 0–1000 mg/L; online measurement < 3 min.
    [7]  EPA Methods 365.2 (phosphorus) and 353.2 (nitrate/nitrite colorimetry);
         Molybdenum Blue PO4 reaction ~25 min; Seal Analytical AA500 / SYSTEA
         online nutrient analyzers complete cycle in 8–30 min.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._base import AbstractSensor


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

# Measurement characteristics per (sensor_type, analyzer_method).
# measurement_delay is in days (e.g. 20.0 / 1440 = 20 minutes).
# measurement_noise is the Gaussian std dev in engineering units.
# These values apply when the user does not supply an explicit override.
_ANALYZER_METHOD_DEFAULTS: Dict[ChemicalSensorType, Dict[ChemicalAnalyzerMethod, Dict[str, float]]] = {
    ChemicalSensorType.VFA: {
        ChemicalAnalyzerMethod.ONLINE_TITRATION: {
            "measurement_noise": 0.06,  # g/L std dev — FOS/TAC ±3% FS → σ ≈ 5 mg/L [1,2]
            "measurement_delay": 10.0 / 1440,  # 10 min — Hach EZ7200 cycle 3–9 min + flushing [1]
            "response_time": 1.0 / 1440,  # 1 min — titration endpoint settling
            "detection_limit": 0.01,  # g/L — Hach EZ7200 LOD ≤ 10 mg/L [1]
        },
        ChemicalAnalyzerMethod.GAS_CHROMATOGRAPHY: {
            "measurement_noise": 0.02,  # g/L std dev — GC-FID RSD < 3% [3]
            "measurement_delay": 45.0 / 1440,  # 45 min — run 25–40 min + sample prep [3]
            "response_time": 0.0,  # GC result is instantaneous once run completes
            "detection_limit": 0.005,  # g/L — practical online GC LOD ~5 mg/L [3]
        },
    },
    ChemicalSensorType.AMMONIA: {
        ChemicalAnalyzerMethod.ION_SELECTIVE: {
            "measurement_noise": 0.03,  # g/L std dev — ISE ±10% reading + K+/Na+ interference [4]
            "measurement_delay": 10.0 / 1440,  # 10 min — sample transport + conditioning [4]
            "response_time": 3.0 / 1440,  # 3 min — ISE electrode equilibration [4]
            "detection_limit": 0.01,  # g/L — ISE LOD 0.01 mg/L (practical process LOD higher) [4]
        },
        ChemicalAnalyzerMethod.SPECTROSCOPY: {
            "measurement_noise": 0.02,  # g/L std dev — colorimetric ±5% → σ ≈ 0.02 g/L [5]
            "measurement_delay": 15.0 / 1440,  # 15 min — Indophenol Blue reaction in heated manifold [5]
            "response_time": 0.5 / 1440,  # 30 s — photometer settling
            "detection_limit": 0.005,  # g/L — Hach Amtax / Seal AA500 LOD ~0.03 mg/L [5]
        },
    },
    ChemicalSensorType.COD: {
        ChemicalAnalyzerMethod.SPECTROSCOPY: {
            "measurement_noise": 2.0,  # g/L std dev — UV-Vis proxy ±4–10% FS [6]
            "measurement_delay": 3.0 / 1440,  # 3 min — online UV cell measurement [6]
            "response_time": 0.0,  # UV cell reads in milliseconds
            "detection_limit": 0.10,  # g/L [6]
        },
    },
    ChemicalSensorType.NUTRIENTS: {
        ChemicalAnalyzerMethod.COLORIMETRIC: {
            "measurement_noise": 3.0,  # mg/L std dev — colorimetric nutrient analysis [7]
            "measurement_delay": 30.0 / 1440,  # 30 min — phosphorus Molybdenum Blue: ~25 min [7]
            "response_time": 1.0 / 1440,  # 1 min — photometer settling
            "detection_limit": 1.0,  # mg/L [7]
        },
    },
}


class ChemicalSensor(AbstractSensor):
    """
    Generic chemical analyzer with sampling delay and drift.

    When *measurement_noise*, *measurement_delay*, or *detection_limit* are
    omitted, they are taken from ``_ANALYZER_METHOD_DEFAULTS`` for the chosen
    *analyzer_method*, so different analyzer types produce different
    measurement fidelity and timing out of the box.

    Args:
        component_id: Unique component identifier.
        sensor_type: One of ``VFA``, ``ammonia``, ``COD``, or ``nutrients``.
        analyzer_method: Measurement principle such as titration or spectroscopy.
        signal_key: Input key to read from upstream outputs. If omitted,
            a type-specific default is used.
        measurement_range: Inclusive valid measurement range.
        measurement_noise: Gaussian noise standard deviation in engineering units.
            ``None`` uses the analyzer-method default.
        accuracy: Maximum fixed calibration offset in engineering units.
        drift_rate: Linear drift rate in engineering units per day.
        sample_interval: Sampling interval in days.
        measurement_delay: Delay between sampling and reported result in days.
            ``None`` uses the analyzer-method default.
        response_time: First-order lag time constant in days applied to the
            analytical result as it settles to its final reading (e.g. ISE
            electrode equilibration). Applied every step toward the latest
            matured result, independently of the sample queue.
            ``None`` uses the analyzer-method default.
        detection_limit: Values below this threshold are reported as zero.
            ``None`` uses the analyzer-method default.
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
        measurement_noise: Optional[float] = None,
        accuracy: float = 0.0,
        drift_rate: float = 0.0,
        sample_interval: float = 0.0,
        measurement_delay: Optional[float] = None,
        response_time: Optional[float] = None,
        detection_limit: Optional[float] = None,
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
        resolved_delay = measurement_delay if measurement_delay is not None else method_defaults.get("measurement_delay", 0.0)
        resolved_response_time = response_time if response_time is not None else method_defaults.get("response_time", 0.0)
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

        self.measurement_delay = float(max(0.0, resolved_delay))
        self.response_time = float(max(0.0, resolved_response_time))
        self.detection_limit = float(max(0.0, resolved_detection_limit))

        self.filtered_value: float = np.nan
        self.reported_value: float = np.nan
        self.last_result_time: float = -np.inf
        self.is_detected: bool = False
        self._pending_samples: List[Tuple[float, float]] = []

        self.initialize()

    @staticmethod
    def _parse_sensor_type(sensor_type: str) -> ChemicalSensorType:
        """Normalize user input into a supported chemical sensor type."""
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
        return AbstractSensor._parse_enum(sensor_type, aliases, ChemicalSensorType, "chemical sensor type")

    @staticmethod
    def _parse_analyzer_method(analyzer_method: str) -> ChemicalAnalyzerMethod:
        """Normalize analyzer method input."""
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
        return AbstractSensor._parse_enum(analyzer_method, aliases, ChemicalAnalyzerMethod, "analyzer method")

    def _initialize_subclass(self, initial_state: Optional[Dict[str, Any]]) -> None:
        """Reset / restore chemical-analyzer extras and build state + outputs."""
        self.filtered_value = np.nan
        self.reported_value = np.nan
        self.last_result_time = -np.inf
        self.is_detected = False
        self._pending_samples = []

        if initial_state:
            self.filtered_value = float(initial_state.get("filtered_value", self.measured_value))
            self.reported_value = float(initial_state.get("reported_value", self.filtered_value))
            self.last_result_time = float(initial_state.get("last_result_time", -np.inf))
            self.is_detected = bool(initial_state.get("is_detected", False))
            pending = initial_state.get("pending_samples", [])
            self._pending_samples = [(float(r), float(s)) for r, s in pending]

        self.state = {
            **self._base_state_dict(),
            "filtered_value": float(self.filtered_value),
            "reported_value": float(self.reported_value),
            "last_result_time": float(self.last_result_time),
            "is_detected": bool(self.is_detected),
            "pending_samples": list(self._pending_samples),
        }
        self.outputs_data = self._build_outputs(
            self.reported_value,
            extras={
                "analyzer_method": self.analyzer_method.value,
                "is_detected": bool(self.is_detected),
            },
        )

    def step(self, t: float, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Advance the analyzer by one simulation step."""
        self._advance_drift_and_read(dt, inputs)

        if self._should_sample(t) and not np.isnan(self.true_value):
            self._pending_samples.append((t + self.measurement_delay, self.true_value))
            self.last_sample_time = t

        matured_sample = self._pop_latest_ready_sample(t + dt)
        if matured_sample is not None:
            raw_value = self._apply_errors(matured_sample)
            self.measured_value, self.in_range = self._clamp_to_range(raw_value)
            self.last_result_time = t
            self.is_valid = True

        # Apply response lag toward the latest analytical result every step.
        # With response_time=0 this is a direct assignment (backward compatible).
        if not np.isnan(self.measured_value):
            self.filtered_value = self._apply_response_lag(self.measured_value, self.filtered_value, self.response_time, dt)
            self.is_detected = self.filtered_value >= self.detection_limit
            self.reported_value = float(self.filtered_value if self.is_detected else 0.0)

        self.state.update(
            {
                **self._base_state_dict(),
                "filtered_value": float(self.filtered_value),
                "reported_value": float(self.reported_value),
                "last_result_time": float(self.last_result_time),
                "is_detected": bool(self.is_detected),
                "pending_samples": list(self._pending_samples),
            }
        )
        self.outputs_data = self._build_outputs(
            self.reported_value,
            extras={
                "analyzer_method": self.analyzer_method.value,
                "is_detected": bool(self.is_detected),
            },
            include_drift=True,
        )
        return self.outputs_data

    def _pop_latest_ready_sample(self, t_end: float) -> Optional[float]:
        """
        Return the latest sample whose analysis delay completes before *t_end*.

        A sample added at time ``t_sample`` with delay ``d`` has release time
        ``t_sample + d``; it matures during a simulation step covering the
        half-open interval ``[t, t + dt)``.  Callers pass ``t_end = t + dt``
        so that samples whose analysis finishes within the current step are
        reported at the end of that step (strict ``<`` preserves the
        half-open convention — a sample releasing exactly at ``t + dt``
        matures on the next step).
        """
        latest_ready: Optional[float] = None
        pending: List[Tuple[float, float]] = []
        for release_time, sample_value in self._pending_samples:
            if release_time < t_end:
                latest_ready = sample_value
            else:
                pending.append((release_time, sample_value))
        self._pending_samples = pending
        return latest_ready

    def to_dict(self) -> Dict[str, Any]:
        """Serialize analyzer configuration and state."""
        return {
            **self._base_config_dict(),
            "sensor_type": self.sensor_type.value,
            "analyzer_method": self.analyzer_method.value,
            "measurement_delay": self.measurement_delay,
            "response_time": self.response_time,
            "detection_limit": self.detection_limit,
            "state": self.state,
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
            measurement_noise=config.get("measurement_noise"),
            accuracy=config.get("accuracy", 0.0),
            drift_rate=config.get("drift_rate", 0.0),
            sample_interval=config.get("sample_interval", 0.0),
            measurement_delay=config.get("measurement_delay"),
            response_time=config.get("response_time"),
            detection_limit=config.get("detection_limit"),
            unit=config.get("unit"),
            output_key=config.get("output_key"),
            name=config.get("name"),
        )
        cls._restore_io(sensor, config)
        return sensor
