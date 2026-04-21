"""
Physical process sensors for biogas plant simulations.

This module provides a configurable virtual sensor for common physical
measurements such as pH, temperature, pressure, level, and flow. The sensor
models practical measurement effects including:

- calibration offset within the stated accuracy
- Gaussian measurement noise
- linear calibration drift over time
- first-order response lag
- discrete sampling intervals
- Nernst temperature compensation for pH sensors
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ._base import AbstractSensor

# Physical constants for the Nernst equation
_R = 8.314  # J / (mol·K)  — ideal gas constant
_F = 96485.0  # C / mol      — Faraday constant


class PhysicalSensorType(str, Enum):
    """Supported physical sensor types."""

    PH = "pH"
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    LEVEL = "level"
    FLOW = "flow"


_DEFAULT_SENSOR_CONFIG: Dict[PhysicalSensorType, Dict[str, Any]] = {
    PhysicalSensorType.PH: {
        "signal_key": "pH",
        "candidate_keys": ("pH",),
        "measurement_range": (0.0, 14.0),
        "unit": "pH",
    },
    PhysicalSensorType.TEMPERATURE: {
        "signal_key": "temperature",
        "candidate_keys": ("temperature", "T_digester", "T_ad"),
        "measurement_range": (250.0, 400.0),
        "unit": "K",
    },
    PhysicalSensorType.PRESSURE: {
        "signal_key": "pressure_bar",
        "candidate_keys": ("pressure_bar", "pressure_actual", "pressure"),
        "measurement_range": (0.0, 100.0),
        "unit": "bar",
    },
    PhysicalSensorType.LEVEL: {
        "signal_key": "current_level",
        "candidate_keys": ("current_level", "level", "stored_volume_m3", "utilization"),
        "measurement_range": (0.0, 1.0e9),
        "unit": "arb",
    },
    PhysicalSensorType.FLOW: {
        "signal_key": "Q_actual",
        "candidate_keys": ("Q_actual", "Q_out", "Q_gas", "Q_gas_supplied_m3_per_day"),
        "measurement_range": (0.0, 1.0e9),
        "unit": "m3/d",
    },
}


class PhysicalSensor(AbstractSensor):
    """
    Generic physical sensor with realistic measurement behavior.

    For ``sensor_type="pH"`` an optional Nernst temperature compensation can
    be enabled by supplying *temperature_signal_key*.  Without it the sensor
    behaves as if the process temperature always equals the calibration
    reference (no systematic temperature error).

    The Nernst correction models what an un-compensated pH electrode reads at
    the actual process temperature::

        pH_apparent = pH_iso + (pH_true − pH_iso) × (T_ref / T_actual)

    This systematic shift is applied after the response lag and before the
    noise / calibration errors, because it is a physical electrode property,
    not measurement uncertainty.

    Args:
        component_id: Unique component identifier.
        sensor_type: One of ``pH``, ``temperature``, ``pressure``, ``level``,
            or ``flow``.
        signal_key: Input key to read from connected upstream outputs. If
            omitted a type-specific default is used.
        measurement_range: Inclusive valid measurement range.
        measurement_noise: Gaussian noise standard deviation in engineering
            units.
        accuracy: Maximum fixed calibration offset in engineering units.
        drift_rate: Linear drift rate in engineering units per day.
        response_time: First-order lag time constant in days.
        sample_interval: Sampling interval in days. The output is held between
            samples.
        unit: Engineering unit label.
        output_key: Namespaced output key for the measured value.
        rng_seed: Optional random seed for deterministic runs.
        name: Human-readable component name.
        temperature_signal_key: Input key that provides the process temperature
            in Kelvin.  Only used when ``sensor_type="pH"``.  When ``None``
            (default) no Nernst correction is applied.
        temperature_reference: Electrode calibration temperature in Kelvin.
            Defaults to 298.15 K (25 °C).
        pH_isopotential: pH at which the electrode gives the same reading
            regardless of temperature.  Defaults to 7.0.
    """

    def __init__(
        self,
        component_id: str,
        sensor_type: str = "temperature",
        signal_key: Optional[str] = None,
        measurement_range: Optional[Tuple[float, float]] = None,
        measurement_noise: float = 0.0,
        accuracy: float = 0.0,
        drift_rate: float = 0.0,
        response_time: float = 0.0,
        sample_interval: float = 0.0,
        unit: Optional[str] = None,
        output_key: Optional[str] = None,
        rng_seed: Optional[int] = None,
        name: Optional[str] = None,
        temperature_signal_key: Optional[str] = None,
        temperature_reference: float = 298.15,
        pH_isopotential: float = 7.0,
    ):
        self.sensor_type = self._parse_sensor_type(sensor_type)
        defaults = _DEFAULT_SENSOR_CONFIG[self.sensor_type]

        resolved_signal_key = signal_key or defaults["signal_key"]
        resolved_candidate_keys = (resolved_signal_key,) if signal_key else defaults["candidate_keys"]

        super().__init__(
            component_id=component_id,
            signal_key=resolved_signal_key,
            candidate_keys=resolved_candidate_keys,
            measurement_range=measurement_range or defaults["measurement_range"],
            measurement_noise=measurement_noise,
            accuracy=accuracy,
            drift_rate=drift_rate,
            sample_interval=sample_interval,
            unit=unit or defaults["unit"],
            output_key=output_key,
            rng_seed=rng_seed,
            name=name,
        )

        self.response_time = float(max(0.0, response_time))

        # Nernst compensation — only active for pH sensors with a temperature key
        self.temperature_signal_key = temperature_signal_key if self.sensor_type == PhysicalSensorType.PH else None
        self.temperature_reference = float(temperature_reference)
        self.pH_isopotential = float(pH_isopotential)

        self.filtered_value: float = np.nan
        self._temperature_value: float = np.nan

        self.initialize()

    @staticmethod
    def _parse_sensor_type(sensor_type: str) -> PhysicalSensorType:
        """Normalize user input into a supported sensor type."""
        aliases = {
            "ph": PhysicalSensorType.PH,
            "p_h": PhysicalSensorType.PH,
            "temperature": PhysicalSensorType.TEMPERATURE,
            "temp": PhysicalSensorType.TEMPERATURE,
            "pressure": PhysicalSensorType.PRESSURE,
            "level": PhysicalSensorType.LEVEL,
            "flow": PhysicalSensorType.FLOW,
        }
        return AbstractSensor._parse_enum(sensor_type, aliases, PhysicalSensorType, "physical sensor type")

    def _apply_nernst_correction(self, pH_value: float, inputs: Dict[str, Any]) -> float:
        """Apply Nernst temperature correction to *pH_value*.

        Returns *pH_value* unchanged when *temperature_signal_key* is not set,
        when the temperature cannot be read, or when the temperature is
        non-positive.

        Side-effect: updates ``self._temperature_value`` with the temperature
        that was used.
        """
        if self.temperature_signal_key is None:
            return pH_value

        raw_temp = inputs.get(self.temperature_signal_key)
        if raw_temp is None:
            return pH_value
        try:
            T_actual = float(raw_temp)
        except (TypeError, ValueError):
            return pH_value
        if T_actual <= 0.0:
            return pH_value

        self._temperature_value = T_actual

        # Without ATC, an electrode calibrated at T_ref reads at T_actual:
        #   pH_apparent = pH_iso + (pH_true − pH_iso) × (T_ref / T_actual)
        # The Nernst slope S(T) = RT·ln(10)/F scales linearly with T, so the
        # electrode over-reports the deviation from the isopotential point when
        # T_actual > T_ref.
        return self.pH_isopotential + (pH_value - self.pH_isopotential) * (self.temperature_reference / T_actual)

    def _initialize_subclass(self, initial_state: Optional[Dict[str, Any]]) -> None:
        """Reset / restore physical-sensor extras and build state + outputs."""
        self.filtered_value = np.nan
        self._temperature_value = np.nan
        if initial_state:
            self.filtered_value = float(initial_state.get("filtered_value", self.true_value))
            self._temperature_value = float(initial_state.get("temperature_value", np.nan))

        self.state = {
            **self._base_state_dict(),
            "filtered_value": float(self.filtered_value),
            "temperature_value": float(self._temperature_value),
        }
        self.outputs_data = self._build_outputs(
            self.measured_value,
            extras={"temperature_value": float(self._temperature_value)},
        )

    def step(self, t: float, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Advance the sensor by one simulation step.

        Reads the configured signal from upstream outputs, then applies
        response lag, Nernst temperature correction (pH only), calibration
        error, drift, noise, and range limits.
        """
        self._advance_drift_and_read(dt, inputs)

        should_sample = self._should_sample(t)
        if should_sample and not np.isnan(self.true_value):
            self.filtered_value = self._apply_response_lag(self.true_value, self.filtered_value, self.response_time, dt)
            # Nernst correction: applied after lag, before noise/calibration errors.
            # Only active for pH sensors when temperature_signal_key is set.
            corrected_value = self._apply_nernst_correction(self.filtered_value, inputs)
            raw_value = self._apply_errors(corrected_value)
            self.measured_value, self.in_range = self._clamp_to_range(raw_value)
            self.last_sample_time = t
            self.is_valid = True
        elif should_sample:
            self.is_valid = False

        self.state.update(
            {
                **self._base_state_dict(),
                "filtered_value": float(self.filtered_value),
                "temperature_value": float(self._temperature_value),
            }
        )
        self.outputs_data = self._build_outputs(
            self.measured_value,
            extras={"temperature_value": float(self._temperature_value)},
            include_drift=True,
        )
        return self.outputs_data

    def to_dict(self) -> Dict[str, Any]:
        """Serialize sensor configuration and state."""
        return {
            **self._base_config_dict(),
            "sensor_type": self.sensor_type.value,
            "response_time": self.response_time,
            "temperature_signal_key": self.temperature_signal_key,
            "temperature_reference": self.temperature_reference,
            "pH_isopotential": self.pH_isopotential,
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "PhysicalSensor":
        """Create sensor from serialized configuration."""
        sensor = cls(
            component_id=config["component_id"],
            sensor_type=config.get("sensor_type", "temperature"),
            signal_key=config.get("signal_key"),
            measurement_range=tuple(config.get("measurement_range", (250.0, 400.0))),
            measurement_noise=config.get("measurement_noise", 0.0),
            accuracy=config.get("accuracy", 0.0),
            drift_rate=config.get("drift_rate", 0.0),
            response_time=config.get("response_time", 0.0),
            sample_interval=config.get("sample_interval", 0.0),
            unit=config.get("unit"),
            output_key=config.get("output_key"),
            name=config.get("name"),
            temperature_signal_key=config.get("temperature_signal_key"),
            temperature_reference=config.get("temperature_reference", 298.15),
            pH_isopotential=config.get("pH_isopotential", 7.0),
        )
        cls._restore_io(sensor, config)
        return sensor
