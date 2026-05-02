"""Abstract base class for all sensor components."""

from __future__ import annotations

from abc import abstractmethod
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Tuple, Type, TypeVar

import numpy as np

from ..base import Component, ComponentType

_E = TypeVar("_E", bound=Enum)


class AbstractSensor(Component):
    """
    Abstract base class for all biogas plant sensor components.

    Handles the constructor parameters, state variables, and helper methods
    that are identical across PhysicalSensor, ChemicalSensor, and GasSensor.

    Subclasses must implement ``step()``, ``to_dict()``, and ``from_dict()``.
    The shared ``initialize()`` calls the ``_initialize_subclass()`` hook to
    let subclasses reset their own state and build ``state`` / ``outputs_data``
    dicts; the ``_initialized`` flag is set automatically afterwards.

    Subclasses must set ``sensor_type`` (a string-valued ``Enum``) before
    calling ``super().__init__()``, since shared helpers read its ``.value``.
    """

    sensor_type: Enum

    def __init__(
        self,
        component_id: str,
        signal_key: str,
        candidate_keys: Tuple[str, ...],
        measurement_range: Tuple[float, float],
        measurement_noise: float = 0.0,
        accuracy: float = 0.0,
        drift_rate: float = 0.0,
        sample_interval: float = 0.0,
        unit: str = "",
        output_key: Optional[str] = None,
        rng_seed: Optional[int] = None,
        name: Optional[str] = None,
    ):
        super().__init__(component_id, ComponentType.SENSOR, name)

        self.signal_key = signal_key
        self._candidate_keys = candidate_keys
        self.measurement_range: Tuple[float, float] = tuple(measurement_range)  # type: ignore[assignment]
        self.measurement_noise = float(max(0.0, measurement_noise))
        self.accuracy = float(max(0.0, accuracy))
        self.drift_rate = float(drift_rate)
        self.sample_interval = float(max(0.0, sample_interval))
        self.unit = unit
        self.output_key = output_key or f"{component_id}_measurement"

        self._rng = np.random.default_rng(rng_seed)
        self.calibration_offset = float(self._rng.uniform(-self.accuracy, self.accuracy)) if self.accuracy > 0 else 0.0

        # Common state — properly initialised by each subclass's initialize()
        self.true_value: float = np.nan
        self.measured_value: float = np.nan
        self.drift_offset: float = 0.0
        self.last_sample_time: float = -np.inf
        self.is_valid: bool = False
        self.in_range: bool = True

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def step(self, t: float, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Advance the sensor by one timestep and return its output dictionary."""
        ...

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize the sensor to a configuration dictionary."""
        ...

    @classmethod
    @abstractmethod
    def from_dict(cls, config: Dict[str, Any]) -> "AbstractSensor":
        """Reconstruct a sensor instance from a configuration dictionary."""
        ...

    # ------------------------------------------------------------------
    # Shared initialize — template-method pattern
    # ------------------------------------------------------------------

    def initialize(self, initial_state: Optional[Dict[str, Any]] = None) -> None:
        """Reset common sensor state, restore from *initial_state*, run subclass hook."""
        self.true_value = np.nan
        self.measured_value = np.nan
        self.drift_offset = 0.0
        self.last_sample_time = -np.inf
        self.is_valid = False
        self.in_range = True

        if initial_state:
            self.true_value = float(initial_state.get("true_value", np.nan))
            self.measured_value = float(initial_state.get("measured_value", np.nan))
            self.drift_offset = float(initial_state.get("drift_offset", 0.0))
            self.last_sample_time = float(initial_state.get("last_sample_time", -np.inf))
            self.is_valid = bool(initial_state.get("is_valid", False))
            self.in_range = bool(initial_state.get("in_range", True))

        self._initialize_subclass(initial_state)
        self._initialized = True

    def _initialize_subclass(self, initial_state: Optional[Dict[str, Any]]) -> None:
        """Hook for subclass-specific reset/restore and ``state`` / ``outputs_data`` build.

        Default no-op so subclasses without extra state need not override.
        """

    # ------------------------------------------------------------------
    # Common measurement helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_enum(
        value: str,
        aliases: Mapping[str, _E],
        enum_cls: Type[_E],  # noqa: ARG004 — kept for type symmetry / future use
        label: str,
    ) -> _E:
        """Normalize *value* (case- and whitespace-insensitive) via *aliases*."""
        normalized = value.strip().lower()
        if normalized not in aliases:
            raise ValueError(f"Unsupported {label} '{value}'")
        return aliases[normalized]

    def _advance_drift_and_read(self, dt: float, inputs: Dict[str, Any]) -> None:
        """Step preamble: integrate drift over *dt* and refresh ``true_value``."""
        self.drift_offset += self.drift_rate * dt
        true_value = self._read_true_value(inputs)
        if true_value is not None:
            self.true_value = true_value

    def _read_true_value(self, inputs: Dict[str, Any]) -> Optional[float]:
        """Resolve the measured signal from upstream component outputs."""
        for key in self._candidate_keys:
            if key in inputs:
                try:
                    return float(inputs[key])
                except (TypeError, ValueError):
                    return None
        return None

    def _should_sample(self, t: float) -> bool:
        """Return True when the next discrete sample is due."""
        if self.sample_interval <= 0:
            return True
        return (t - self.last_sample_time) >= self.sample_interval

    @staticmethod
    def _apply_response_lag(
        true_value: float,
        filtered_value: float,
        response_time: float,
        dt: float,
    ) -> float:
        """Apply a first-order lag filter and return the updated filtered value."""
        if np.isnan(filtered_value) or response_time <= 0:
            return true_value
        alpha = min(1.0, dt / max(response_time, 1.0e-12))
        return filtered_value + alpha * (true_value - filtered_value)

    def _apply_errors(self, value: float) -> float:
        """Apply calibration offset, accumulated drift, and Gaussian noise."""
        value = value + self.calibration_offset + self.drift_offset
        if self.measurement_noise > 0:
            value += float(self._rng.normal(0.0, self.measurement_noise))
        return value

    def _clamp_to_range(self, value: float) -> Tuple[float, bool]:
        """Clamp *value* to the measurement range. Returns ``(clamped_value, in_range)``."""
        min_v, max_v = self.measurement_range
        in_range = min_v <= value <= max_v
        return float(np.clip(value, min_v, max_v)), in_range

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def _base_state_dict(self) -> Dict[str, Any]:
        """Common state fields shared by all sensor types."""
        return {
            "true_value": float(self.true_value),
            "measured_value": float(self.measured_value),
            "drift_offset": float(self.drift_offset),
            "calibration_offset": float(self.calibration_offset),
            "last_sample_time": float(self.last_sample_time),
            "is_valid": bool(self.is_valid),
            "in_range": bool(self.in_range),
        }

    def _base_config_dict(self) -> Dict[str, Any]:
        """Common configuration fields for ``to_dict()``."""
        return {
            "component_id": self.component_id,
            "component_type": self.component_type.value,
            "name": self.name,
            "signal_key": self.signal_key,
            "measurement_range": list(self.measurement_range),
            "measurement_noise": self.measurement_noise,
            "accuracy": self.accuracy,
            "drift_rate": self.drift_rate,
            "sample_interval": self.sample_interval,
            "unit": self.unit,
            "output_key": self.output_key,
            "inputs": self.inputs,
            "outputs": self.outputs,
        }

    def _build_outputs(
        self,
        measurement: float,
        extras: Optional[Dict[str, Any]] = None,
        include_drift: bool = False,
    ) -> Dict[str, Any]:
        """Build the common ``outputs_data`` dict; *extras* are merged on top.

        Includes ``drift_offset`` only when *include_drift* is True (step phase).
        Subclasses pass type-specific keys (``analyzer_method``, ``is_detected``,
        ``temperature_value``, …) via *extras*.
        """
        out: Dict[str, Any] = {
            "measurement": float(measurement),
            self.output_key: float(measurement),
            "true_value": float(self.true_value),
            "sensor_type": self.sensor_type.value,
            "signal_key": self.signal_key,
            "unit": self.unit,
            "is_valid": bool(self.is_valid),
            "in_range": bool(self.in_range),
        }
        if include_drift:
            out["drift_offset"] = float(self.drift_offset)
        if extras:
            out.update(extras)
        return out

    @staticmethod
    def _restore_io(sensor: "AbstractSensor", config: Dict[str, Any]) -> None:
        """Restore ``state`` and wire ``inputs`` / ``outputs`` from a serialized config."""
        if "state" in config:
            sensor.initialize(config["state"])
        for input_id in config.get("inputs", []):
            sensor.add_input(input_id)
        for output_id in config.get("outputs", []):
            sensor.add_output(output_id)
