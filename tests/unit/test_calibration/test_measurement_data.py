# -*- coding: utf-8 -*-
"""
Unit tests

- Test CSV loading
- Test validation
- Test outlier detection
- Test gap filling
"""

import pandas as pd
import numpy as np

# import pytest
from pyadm1.io.measurement_data import (
    MeasurementData,
    OutlierDetector,
    # DataValidator,
)


def test_from_csv_loading(tmp_path):
    # Create sample CSV
    csv = tmp_path / "test.csv"
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=5, freq="H"), "Q_ch4": [1, 2, 3, 4, 5]})
    df.to_csv(csv, index=False)

    data = MeasurementData.from_csv(str(csv))

    assert isinstance(data, MeasurementData)
    assert "Q_ch4" in data.data.columns
    assert isinstance(data.data.index, pd.DatetimeIndex)
    assert len(data.data) == 5


def test_validation_detects_missing_values():
    df = pd.DataFrame({"timestamp": pd.date_range("2024-01-01", periods=5, freq="H"), "Q_ch4": [1, np.nan, 3, np.nan, 5]})

    data = MeasurementData(df)
    result = data.validate()

    assert not result.is_valid
    assert "Q_ch4" in result.missing_data
    assert result.missing_data["Q_ch4"] > 0


def test_outlier_detection_iqr():
    s = pd.Series([1, 2, 3, 100])  # 100 is an outlier
    outliers = OutlierDetector.detect_iqr(s, multiplier=1.5)

    assert outliers.iloc[-1]
    assert outliers.sum() == 1


def test_outlier_detection_moving_window():
    # as moving average is calculated around center, the last window-2 elements in the outliers object are NaN.
    # window of 3 is very small to detect outliers with z-score, because z-score is calculated using mean, so not robust
    s = pd.Series([1, 2, 3, 4, 80, 2, 4])  # last entry is outlier
    outliers = OutlierDetector.detect_moving_window(s, window=5, threshold=1.5)

    # last value is NaN
    assert outliers.iloc[-3]


def test_remove_outliers_zscore():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="H"),
            "Q_ch4": [1, 2, 1000, 3, 4, 3],
        }
    )
    data = MeasurementData(df)

    removed = data.remove_outliers(method="zscore", threshold=2.0)

    assert removed == 1
    assert data.data["Q_ch4"].isna().sum() == 1


def test_fill_gaps_interpolate():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="H"),
            "Q_ch4": [1, np.nan, np.nan, 4, 5],
        }
    )
    data = MeasurementData(df)

    data.fill_gaps(method="interpolate", limit=2)

    assert data.data["Q_ch4"].isna().sum() == 0
    assert np.isclose(data.data["Q_ch4"].iloc[1], 2.0, atol=0.1)


def test_get_measurement():
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="H"),
            "pH": [7.0, 7.1, 7.2],
        }
    )
    data = MeasurementData(df)

    pH_series = data.get_measurement("pH")

    assert len(pH_series) == 3
    assert list(pH_series.values) == [7.0, 7.1, 7.2]


def test_get_time_window():
    timestamps = pd.date_range("2024-01-01", periods=10, freq="H")
    df = pd.DataFrame({"timestamp": timestamps, "Q_ch4": range(10)})
    data = MeasurementData(df)

    window = data.get_time_window(timestamps[3], timestamps[6])

    assert len(window.data) == 4
    assert window.data["Q_ch4"].iloc[0] == 3
