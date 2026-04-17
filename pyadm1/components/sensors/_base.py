"""Abstract base class for all sensor components."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..base import Component, ComponentType


class AbstractSensor(Component):
    """
    Abstract base class for all biogas plant sensor components.

    Handles the constructor parameters, state variables, and helper methods
    that are identical across PhysicalSensor, ChemicalSensor, and GasSensor.

    Subclasses must implement ``step()``, ``to_dict()``, and ``from_dict()``.
    Their ``initialize()`` override should call ``super().initialize(initial_state)``
    first to reset the common state, then handle subclass-specific state.
    """

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
    def step(self, t: float, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]: ...

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]: ...

    @classmethod
    @abstractmethod
    def from_dict(cls, config: Dict[str, Any]) -> "AbstractSensor": ...

    # ------------------------------------------------------------------
    # Shared initialize — subclasses call super().initialize() first
    # ------------------------------------------------------------------

    def initialize(self, initial_state: Optional[Dict[str, Any]] = None) -> None:
        """Reset common sensor state, optionally restoring from *initial_state*."""
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

    # ------------------------------------------------------------------
    # Common measurement helpers
    # ------------------------------------------------------------------

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
