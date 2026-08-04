"""Unit tests for behaviour shared by every sensor (``sensors/_base.py``).

Exercised through the concrete sensor classes rather than the private helpers,
so the tests describe what a user of the components actually observes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyadm1.components.sensors import ChemicalSensor, GasSensor, PhysicalSensor


class TestEnumParsing:
    """Sensor type / analyzer names are normalized case- and whitespace-insensitively."""

    @pytest.mark.parametrize("spelling", ["pH", "PH", "  ph  ", "pH "])
    def test_known_names_are_accepted_in_any_spelling(self, spelling: str) -> None:
        assert PhysicalSensor("s", sensor_type=spelling).sensor_type.value == "pH"

    @pytest.mark.parametrize("alias", ["temperature", "TEMP", " Temp "])
    def test_aliases_map_onto_the_same_type(self, alias: str) -> None:
        assert PhysicalSensor("s", sensor_type=alias).sensor_type.value == "temperature"

    def test_unknown_physical_sensor_type_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsupported physical sensor type 'unobtainium'"):
            PhysicalSensor("s", sensor_type="unobtainium")

    def test_unknown_chemical_sensor_type_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsupported chemical sensor type"):
            ChemicalSensor("s", sensor_type="not_a_probe")

    def test_unknown_gas_sensor_type_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsupported gas sensor type"):
            GasSensor("s", sensor_type="radon")

    def test_unknown_analyzer_method_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unsupported analyzer method"):
            GasSensor("s", sensor_type="CH4", analyzer_method="telepathy")

    def test_the_original_spelling_is_quoted_back_in_the_error(self) -> None:
        with pytest.raises(ValueError, match=r"Unsupported physical sensor type '  Nonsense  '"):
            PhysicalSensor("s", sensor_type="  Nonsense  ")


class TestSignalResolution:
    """``_read_true_value`` picks the first known key and must survive junk values."""

    @staticmethod
    def _sensor() -> PhysicalSensor:
        return PhysicalSensor("t1", sensor_type="temperature", measurement_noise=0.0, accuracy=0.0)

    def test_a_readable_signal_is_picked_up(self) -> None:
        sensor = self._sensor()

        sensor.step(t=0.0, dt=1.0, inputs={sensor.signal_key: 311.15})

        assert sensor.true_value == pytest.approx(311.15)

    def test_a_numeric_string_is_still_converted(self) -> None:
        sensor = self._sensor()

        sensor.step(t=0.0, dt=1.0, inputs={sensor.signal_key: "311.15"})

        assert sensor.true_value == pytest.approx(311.15)

    @pytest.mark.parametrize("junk", ["not a number", None, object(), [1.0]])
    def test_an_unconvertible_value_leaves_the_reading_untouched(self, junk) -> None:
        """A malformed upstream value must not crash the sensor nor be half-applied."""
        sensor = self._sensor()
        sensor.step(t=0.0, dt=1.0, inputs={sensor.signal_key: 300.0})
        good = sensor.true_value

        sensor.step(t=1.0, dt=1.0, inputs={sensor.signal_key: junk})

        assert sensor.true_value == pytest.approx(good)

    def test_a_missing_signal_leaves_the_reading_untouched(self) -> None:
        sensor = self._sensor()

        sensor.step(t=0.0, dt=1.0, inputs={"something_else": 42.0})

        assert math.isnan(sensor.true_value)


class TestMeasurementErrors:
    """Calibration offset, drift and Gaussian noise applied on top of the truth."""

    @staticmethod
    def _sensor(**kwargs) -> PhysicalSensor:
        defaults = {"sensor_type": "temperature", "accuracy": 0.0, "response_time": 0.0}
        return PhysicalSensor("t1", **{**defaults, **kwargs})

    def test_a_clean_sensor_reports_the_true_value(self) -> None:
        sensor = self._sensor(measurement_noise=0.0)

        sensor.step(t=0.0, dt=1.0, inputs={sensor.signal_key: 310.0})

        assert sensor.measured_value == pytest.approx(310.0)

    def test_noise_perturbs_the_reading_around_the_truth(self) -> None:
        sensor = self._sensor(measurement_noise=0.5, rng_seed=0)

        readings = []
        for i in range(200):
            sensor.step(t=float(i), dt=1.0, inputs={sensor.signal_key: 310.0})
            readings.append(sensor.measured_value)

        assert not all(r == 310.0 for r in readings)  # noise is actually applied
        assert np.mean(readings) == pytest.approx(310.0, abs=0.15)  # unbiased
        assert np.std(readings) == pytest.approx(0.5, rel=0.25)  # at the requested level

    def test_noise_is_reproducible_for_a_fixed_seed(self) -> None:
        def run(seed: int) -> float:
            sensor = self._sensor(measurement_noise=0.5, rng_seed=seed)
            sensor.step(t=0.0, dt=1.0, inputs={sensor.signal_key: 310.0})
            return sensor.measured_value

        assert run(42) == run(42)
        assert run(42) != run(43)

    def test_drift_accumulates_over_time(self) -> None:
        sensor = self._sensor(measurement_noise=0.0, drift_rate=0.5)

        sensor.step(t=0.0, dt=2.0, inputs={sensor.signal_key: 310.0})
        assert sensor.measured_value == pytest.approx(310.0 + 1.0)  # drift_rate * dt

        sensor.step(t=2.0, dt=2.0, inputs={sensor.signal_key: 310.0})
        assert sensor.measured_value == pytest.approx(310.0 + 2.0)  # drift keeps growing

    def test_a_calibration_offset_biases_every_reading_by_the_same_amount(self) -> None:
        sensor = self._sensor(measurement_noise=0.0)
        sensor.calibration_offset = 1.25  # drawn from ``accuracy`` in production

        for t in (0.0, 1.0, 2.0):
            sensor.step(t=t, dt=1.0, inputs={sensor.signal_key: 310.0})
            assert sensor.measured_value == pytest.approx(311.25)

    def test_accuracy_draws_a_fixed_offset_within_its_bounds(self) -> None:
        sensor = self._sensor(measurement_noise=0.0, accuracy=2.0, rng_seed=7)

        assert abs(sensor.calibration_offset) <= 2.0
        assert sensor.calibration_offset != 0.0
