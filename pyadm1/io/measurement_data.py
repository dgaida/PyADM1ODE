# pyadm1/io/measurement_data.py
"""
Measurement Data Management for Biogas Plant Calibration

This module provides classes for loading, validating, and processing real plant
measurement data for model calibration and validation.

Features:
- CSV/Excel data import with automatic type detection
- Time series resampling and interpolation
- Outlier detection and removal
- Data validation with quality checks
- Gap filling with multiple methods
- Statistical summaries and quality metrics

Example:
    >>> from pyadm1.io.measurement_data import MeasurementData
    >>>
    >>> # Load measurement data
    >>> data = MeasurementData.from_csv(
    ...     "plant_data.csv",
    ...     timestamp_column="time",
    ...     resample="1H"
    ... )
    >>>
    >>> # Validate data quality
    >>> validation = data.validate()
    >>> print(f"Data quality: {validation.quality_score:.2f}")
    >>>
    >>> # Remove outliers
    >>> data.remove_outliers(method="zscore", threshold=3.0)
    >>>
    >>> # Fill gaps
    >>> data.fill_gaps(method="interpolate")
    >>>
    >>> # Get substrate feeds
    >>> Q = data.get_substrate_feeds()
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime  # , timedelta

# from pathlib import Path
# import warnings


@dataclass
class ValidationResult:
    """
    Result from data validation.

    Attributes:
        is_valid: Overall validation status
        quality_score: Overall quality score (0-1)
        issues: List of identified issues
        warnings: List of warnings
        statistics: Dictionary of data statistics
        missing_data: Dictionary of missing data percentages per column
    """

    is_valid: bool
    quality_score: float
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    missing_data: Dict[str, float] = field(default_factory=dict)

    def print_report(self) -> None:
        """Print validation report."""
        print("=" * 70)
        print("Data Validation Report")
        print("=" * 70)
        print(f"Status: {'✓ Valid' if self.is_valid else '✗ Invalid'}")
        print(f"Quality Score: {self.quality_score:.2f}")

        if self.issues:
            print(f"\nIssues ({len(self.issues)}):")
            for issue in self.issues:
                print(f"  - {issue}")

        if self.warnings:
            print(f"\nWarnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning}")

        if self.missing_data:
            print("\nMissing Data:")
            for col, pct in self.missing_data.items():
                if pct > 0:
                    print(f"  {col}: {pct:.1f}%")

        print("=" * 70)


class DataValidator:
    """
    Validator for biogas plant measurement data.

    Checks data quality, identifies issues, and provides statistics.
    """

    @staticmethod
    def validate(
        data: pd.DataFrame,
        required_columns: Optional[List[str]] = None,
        expected_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> ValidationResult:
        """
        Validate measurement data.

        Args:
            data: DataFrame to validate
            required_columns: List of required column names
            expected_ranges: Dictionary mapping columns to (min, max) tuples

        Returns:
            ValidationResult object
        """
        issues = []
        warnings_list = []

        # Check for required columns
        if required_columns:
            missing_cols = set(required_columns) - set(data.columns)
            if missing_cols:
                issues.append(f"Missing required columns: {missing_cols}")

        # Calculate missing data percentages
        missing_data = {}
        for col in data.columns:
            pct_missing = (data[col].isna().sum() / len(data)) * 100
            missing_data[col] = pct_missing

            if pct_missing > 30:
                issues.append(f"Column '{col}' has {pct_missing:.1f}% missing data")
            elif pct_missing > 5:
                warnings_list.append(f"Column '{col}' has {pct_missing:.1f}% missing data")

        # Check for expected ranges
        if expected_ranges:
            for col, (min_val, max_val) in expected_ranges.items():
                if col in data.columns:
                    values = data[col].dropna()
                    if len(values) > 0:
                        actual_min = values.min()
                        actual_max = values.max()

                        if actual_min < min_val or actual_max > max_val:
                            warnings_list.append(
                                f"Column '{col}' has values outside expected range "
                                f"[{min_val}, {max_val}]: actual [{actual_min:.2f}, {actual_max:.2f}]"
                            )

        # Check for duplicate timestamps
        if "timestamp" in data.columns:
            duplicates = data["timestamp"].duplicated().sum()
            if duplicates > 0:
                warnings_list.append(f"Found {duplicates} duplicate timestamps")

        # Calculate statistics
        statistics = {
            "n_rows": len(data),
            "n_columns": len(data.columns),
            "total_missing": data.isna().sum().sum(),
            "pct_missing": (data.isna().sum().sum() / (len(data) * len(data.columns))) * 100,
        }

        # Calculate quality score
        quality_score = DataValidator._calculate_quality_score(data, len(issues), len(warnings_list), statistics)

        is_valid = len(issues) == 0

        return ValidationResult(
            is_valid=is_valid,
            quality_score=quality_score,
            issues=issues,
            warnings=warnings_list,
            statistics=statistics,
            missing_data=missing_data,
        )

    @staticmethod
    def _calculate_quality_score(data: pd.DataFrame, n_issues: int, n_warnings: int, statistics: Dict[str, Any]) -> float:
        """Calculate overall data quality score (0-1)."""
        score = 1.0

        # Penalize for issues
        score -= min(0.5, n_issues * 0.1)

        # Penalize for warnings
        score -= min(0.3, n_warnings * 0.05)

        # Penalize for missing data
        pct_missing = statistics["pct_missing"]
        score -= min(0.2, pct_missing / 100 * 0.5)

        return max(0.0, score)


class OutlierDetector:
    """
    Outlier detection for time series data.

    Supports multiple detection methods for identifying anomalous values.
    """

    @staticmethod
    def detect_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
        """
        Detect outliers using z-score method.

        Args:
            series: Pandas Series
            threshold: Z-score threshold

        Returns:
            Boolean Series indicating outliers
        """
        if len(series.dropna()) < 2:
            return pd.Series([False] * len(series), index=series.index)

        mean = series.mean()
        std = series.std()

        if std == 0:
            return pd.Series([False] * len(series), index=series.index)

        z_scores = np.abs((series - mean) / std)
        return z_scores > threshold

    @staticmethod
    def detect_iqr(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
        """
        Detect outliers using IQR (Interquartile Range) method.

        Args:
            series: Pandas Series
            multiplier: IQR multiplier (1.5 for outliers, 3.0 for extreme)

        Returns:
            Boolean Series indicating outliers
        """
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        return (series < lower_bound) | (series > upper_bound)

    @staticmethod
    def detect_moving_window(series: pd.Series, window: int = 5, threshold: float = 3.0) -> pd.Series:
        """
        Detect outliers using moving window method.

        Args:
            series: Pandas Series
            window: Window size for rolling statistics
            threshold: Number of standard deviations

        Returns:
            Boolean Series indicating outliers
        """
        rolling_mean = series.rolling(window=window, center=True).mean()
        rolling_std = series.rolling(window=window, center=True).std()

        z_scores = np.abs((series - rolling_mean) / rolling_std)
        return z_scores > threshold


class MeasurementData:
    """
    Container for biogas plant measurement data.

    Handles loading, validation, preprocessing, and access to time series
    measurements from biogas plants.

    Attributes:
        data: Main DataFrame with measurements
        metadata: Dictionary with metadata

    Example:
        >>> data = MeasurementData.from_csv("plant_data.csv")
        >>> Q = data.get_substrate_feeds()
        >>> pH = data.get_measurement("pH")
    """

    def __init__(self, data: pd.DataFrame, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize measurement data.

        Args:
            data: DataFrame with measurements
            metadata: Optional metadata dictionary
        """
        self.data = data
        self.metadata = metadata or {}

        # Ensure timestamp column is datetime
        if "timestamp" in self.data.columns:
            if not pd.api.types.is_datetime64_any_dtype(self.data["timestamp"]):
                self.data["timestamp"] = pd.to_datetime(self.data["timestamp"])
            self.data = self.data.set_index("timestamp").sort_index()

    @classmethod
    def from_csv(
        cls,
        filepath: str,
        timestamp_column: str = "timestamp",
        sep: str = ",",
        parse_dates: bool = True,
        resample: Optional[str] = None,
        **kwargs,
    ) -> "MeasurementData":
        """
        Load measurement data from CSV file.

        Args:
            filepath: Path to CSV file
            timestamp_column: Name of timestamp column
            sep: Column separator
            parse_dates: Automatically parse dates
            resample: Resample frequency (e.g., "1H", "15min")
            **kwargs: Additional arguments for pd.read_csv

        Returns:
            MeasurementData instance

        Example:
            >>> data = MeasurementData.from_csv(
            ...     "plant_data.csv",
            ...     timestamp_column="time",
            ...     resample="1H"
            ... )
        """
        # Read CSV
        data = pd.read_csv(filepath, sep=sep, **kwargs)

        # Parse timestamp
        if timestamp_column in data.columns:
            data["timestamp"] = pd.to_datetime(data[timestamp_column])
            if timestamp_column != "timestamp":
                data = data.drop(columns=[timestamp_column])

        # Create instance
        instance = cls(data)

        # Resample if requested
        if resample is not None:
            instance.resample(resample)

        return instance

    def validate(
        self, required_columns: Optional[List[str]] = None, expected_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> ValidationResult:
        """
        Validate measurement data.

        Args:
            required_columns: List of required columns
            expected_ranges: Expected ranges for columns

        Returns:
            ValidationResult object
        """
        # Default expected ranges for biogas plant measurements
        if expected_ranges is None:
            expected_ranges = {
                "pH": (5.0, 9.0),
                "VFA": (0.0, 20.0),  # g/L
                "TAC": (0.0, 50.0),  # g CaCO3/L
                "Q_gas": (0.0, 5000.0),  # m³/d
                "Q_ch4": (0.0, 3000.0),  # m³/d
                "T_digester": (273.15, 333.15),  # K (0-60°C)
            }

        return DataValidator.validate(self.data, required_columns=required_columns, expected_ranges=expected_ranges)

    def remove_outliers(
        self, columns: Optional[List[str]] = None, method: str = "zscore", threshold: float = 3.0, **kwargs
    ) -> int:
        """
        Remove outliers from specified columns.

        Args:
            columns: List of columns to check (None = all numeric)
            method: Detection method ("zscore", "iqr", "moving_window")
            threshold: Threshold for detection
            **kwargs: Additional method-specific arguments

        Returns:
            Number of outliers removed
        """
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()

        n_outliers = 0

        for col in columns:
            if col not in self.data.columns:
                continue

            # Detect outliers
            if method == "zscore":
                is_outlier = OutlierDetector.detect_zscore(self.data[col], threshold=threshold)
            elif method == "iqr":
                is_outlier = OutlierDetector.detect_iqr(self.data[col], multiplier=threshold)
            elif method == "moving_window":
                window = kwargs.get("window", 5)
                is_outlier = OutlierDetector.detect_moving_window(self.data[col], window=window, threshold=threshold)
            else:
                raise ValueError(f"Unknown outlier detection method: {method}")

            # Remove outliers (set to NaN)
            n_col_outliers = is_outlier.sum()
            self.data.loc[is_outlier, col] = np.nan
            n_outliers += n_col_outliers

        return n_outliers

    def fill_gaps(self, columns: Optional[List[str]] = None, method: str = "interpolate", **kwargs) -> None:
        """
        Fill missing values in time series.

        Args:
            columns: List of columns to fill (None = all)
            method: Fill method:
                - "interpolate": Linear interpolation
                - "forward": Forward fill
                - "backward": Backward fill
                - "mean": Fill with column mean
                - "median": Fill with column median
            **kwargs: Additional method-specific arguments
        """
        if columns is None:
            columns = self.data.columns.tolist()

        for col in columns:
            if col not in self.data.columns:
                continue

            if method == "interpolate":
                limit = kwargs.get("limit", None)
                self.data[col] = self.data[col].interpolate(method="linear", limit=limit)
            elif method == "forward":
                limit = kwargs.get("limit", None)
                self.data[col] = self.data[col].fillna(method="ffill", limit=limit)
            elif method == "backward":
                limit = kwargs.get("limit", None)
                self.data[col] = self.data[col].fillna(method="bfill", limit=limit)
            elif method == "mean":
                self.data[col] = self.data[col].fillna(self.data[col].mean())
            elif method == "median":
                self.data[col] = self.data[col].fillna(self.data[col].median())
            else:
                raise ValueError(f"Unknown fill method: {method}")

    def resample(self, freq: str, aggregation: str = "mean") -> None:
        """
        Resample time series to different frequency.

        Args:
            freq: Pandas frequency string (e.g., "1H", "15min", "1D")
            aggregation: Aggregation method ("mean", "sum", "first", "last")
        """
        if aggregation == "mean":
            self.data = self.data.resample(freq).mean()
        elif aggregation == "sum":
            self.data = self.data.resample(freq).sum()
        elif aggregation == "first":
            self.data = self.data.resample(freq).first()
        elif aggregation == "last":
            self.data = self.data.resample(freq).last()
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

    def get_measurement(
        self, column: str, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None
    ) -> pd.Series:
        """
        Get measurement time series.

        Args:
            column: Column name
            start_time: Start time for slice
            end_time: End time for slice

        Returns:
            Pandas Series with measurements
        """
        if column not in self.data.columns:
            raise ValueError(f"Column '{column}' not found")

        series = self.data[column]

        if start_time is not None or end_time is not None:
            series = series.loc[start_time:end_time]

        return series

    def get_substrate_feeds(self, substrate_columns: Optional[List[str]] = None) -> np.ndarray:
        """
        Get substrate feed rates as array.

        Args:
            substrate_columns: List of substrate column names
                             If None, looks for columns matching "Q_*"

        Returns:
            Array of shape (n_timesteps, n_substrates)
        """
        if substrate_columns is None:
            # Find all Q_* columns
            substrate_columns = [col for col in self.data.columns if col.startswith("Q_sub")]

        if not substrate_columns:
            raise ValueError("No substrate columns found")

        return self.data[substrate_columns].values

    def get_time_window(self, start_time: datetime, end_time: datetime) -> "MeasurementData":
        """
        Get data for specific time window.

        Args:
            start_time: Start time
            end_time: End time

        Returns:
            New MeasurementData instance with windowed data
        """
        windowed_data = self.data.loc[start_time:end_time].copy()
        return MeasurementData(windowed_data, metadata=self.metadata.copy())

    def summary(self) -> pd.DataFrame:
        """
        Get statistical summary of measurements.

        Returns:
            DataFrame with summary statistics
        """
        return self.data.describe()

    def to_csv(self, filepath: str, **kwargs) -> None:
        """
        Save measurement data to CSV.

        Args:
            filepath: Output file path
            **kwargs: Additional arguments for DataFrame.to_csv
        """
        self.data.to_csv(filepath, **kwargs)

    def __len__(self) -> int:
        """Return number of time steps."""
        return len(self.data)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"MeasurementData(n_rows={len(self.data)}, "
            f"n_columns={len(self.data.columns)}, "
            f"time_range={self.data.index[0]} to {self.data.index[-1]})"
        )
