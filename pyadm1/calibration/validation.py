# pyadm1/calibration/validation.py
"""
Calibration Result Validation

This module provides tools for validating calibrated model parameters against
measurement data, computing goodness-of-fit metrics, analyzing residuals, and
assessing parameter correlation and identifiability.

Features:
- Goodness-of-fit metrics (RMSE, MAE, R², Nash-Sutcliffe, PBIAS)
- Residual analysis (normality tests, autocorrelation, heteroscedasticity)
- Parameter correlation analysis (correlation matrix, variance inflation)
- Cross-validation on held-out data
- Visual diagnostics (predicted vs observed, residual plots)
- Validation reports with comprehensive statistics

Example:
    >>> from pyadm1.calibration.validation import CalibrationValidator
    >>> from pyadm1.io import MeasurementData
    >>>
    >>> # Load measurement data
    >>> measurements = MeasurementData.from_csv("plant_data.csv")
    >>>
    >>> # Validate calibration
    >>> validator = CalibrationValidator(plant)
    >>> metrics = validator.validate(
    ...     parameters=calibrated_params,
    ...     measurements=measurements,
    ...     objectives=["Q_ch4", "pH", "VFA"]
    ... )
    >>>
    >>> # Print validation report
    >>> validator.print_validation_report(metrics)
    >>>
    >>> # Analyze residuals
    >>> residuals = validator.analyze_residuals(
    ...     measurements=measurements,
    ...     simulated=simulated_outputs
    ... )
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from scipy import stats
import warnings

from pyadm1.io import MeasurementData


@dataclass
class ValidationMetrics:
    """
    Comprehensive validation metrics for calibration results.

    Attributes:
        objective: Name of objective variable
        n_samples: Number of data points
        rmse: Root Mean Square Error
        mae: Mean Absolute Error
        r2: Coefficient of determination
        nse: Nash-Sutcliffe Efficiency
        pbias: Percent Bias
        correlation: Pearson correlation coefficient
        mape: Mean Absolute Percentage Error
        me: Mean Error (bias)
        observations_mean: Mean of observed values
        observations_std: Standard deviation of observed values
        predictions_mean: Mean of predicted values
        predictions_std: Standard deviation of predicted values
    """

    objective: str
    n_samples: int
    rmse: float
    mae: float
    r2: float
    nse: float
    pbias: float
    correlation: float
    mape: float
    me: float
    observations_mean: float
    observations_std: float
    predictions_mean: float
    predictions_std: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "objective": self.objective,
            "n_samples": self.n_samples,
            "rmse": self.rmse,
            "mae": self.mae,
            "r2": self.r2,
            "nse": self.nse,
            "pbias": self.pbias,
            "correlation": self.correlation,
            "mape": self.mape,
            "me": self.me,
            "observations_mean": self.observations_mean,
            "observations_std": self.observations_std,
            "predictions_mean": self.predictions_mean,
            "predictions_std": self.predictions_std,
        }


@dataclass
class ResidualAnalysis:
    """
    Results from residual analysis.

    Attributes:
        objective: Name of objective variable
        residuals: Array of residuals (observed - predicted)
        standardized_residuals: Standardized residuals
        normality_test: Results from Shapiro-Wilk normality test
        autocorrelation: Lag-1 autocorrelation coefficient
        heteroscedasticity_test: Breusch-Pagan test results
        outlier_indices: Indices of potential outliers (|z| > 3)
    """

    objective: str
    residuals: np.ndarray
    standardized_residuals: np.ndarray
    normality_test: Dict[str, float]  # {"statistic": ..., "p_value": ...}
    autocorrelation: float
    heteroscedasticity_test: Dict[str, float]  # {"statistic": ..., "p_value": ...}
    outlier_indices: List[int] = field(default_factory=list)

    def is_normally_distributed(self, alpha: float = 0.05) -> bool:
        """Check if residuals are normally distributed."""
        return self.normality_test["p_value"] > alpha

    def has_autocorrelation(self, threshold: float = 0.3) -> bool:
        """Check if residuals show significant autocorrelation."""
        return abs(self.autocorrelation) > threshold

    def has_heteroscedasticity(self, alpha: float = 0.05) -> bool:
        """Check if residuals show heteroscedasticity."""
        return self.heteroscedasticity_test["p_value"] < alpha


@dataclass
class ParameterCorrelation:
    """
    Parameter correlation analysis results.

    Attributes:
        correlation_matrix: Correlation matrix between parameters
        parameter_names: List of parameter names
        high_correlations: List of highly correlated parameter pairs
        vif: Variance Inflation Factors (if computable)
    """

    correlation_matrix: np.ndarray
    parameter_names: List[str]
    high_correlations: List[Tuple[str, str, float]] = field(default_factory=list)
    vif: Optional[Dict[str, float]] = None

    def get_correlation(self, param1: str, param2: str) -> float:
        """Get correlation between two parameters."""
        idx1 = self.parameter_names.index(param1)
        idx2 = self.parameter_names.index(param2)
        return self.correlation_matrix[idx1, idx2]


class CalibrationValidator:
    """
    Validator for calibrated model parameters.

    Provides comprehensive validation of calibration results including
    goodness-of-fit metrics, residual analysis, and parameter diagnostics.

    Attributes:
        plant: BiogasPlant instance to validate
        verbose: Enable progress output

    Example:
        >>> validator = CalibrationValidator(plant)
        >>> metrics = validator.validate(parameters, measurements)
    """

    def __init__(self, plant, verbose: bool = True):
        """
        Initialize validator.

        Args:
            plant: BiogasPlant instance
            verbose: Enable progress output
        """
        self.plant = plant
        self.verbose = verbose

    def validate(
        self,
        parameters: Dict[str, float],
        measurements: "MeasurementData",
        objectives: Optional[List[str]] = None,
        simulation_duration: Optional[float] = None,
    ) -> Dict[str, ValidationMetrics]:
        """
        Validate calibrated parameters against measurement data.

        Simulates the plant with calibrated parameters and compares outputs
        to measurements, computing comprehensive validation metrics.

        Args:
            parameters: Calibrated parameter values
            measurements: Measurement data for validation
            objectives: List of objectives to validate (default: ["Q_ch4", "pH", "VFA"])
            simulation_duration: Simulation duration in days (default: from measurements)

        Returns:
            Dictionary mapping objective names to ValidationMetrics

        Example:
            >>> metrics = validator.validate(
            ...     parameters={"k_dis": 0.5, "Y_su": 0.1},
            ...     measurements=validation_data,
            ...     objectives=["Q_ch4", "pH"]
            ... )
        """
        if objectives is None:
            objectives = ["Q_ch4", "pH", "VFA"]

        if self.verbose:
            print("Validating calibrated parameters...")
            print(f"Objectives: {objectives}")

        # Apply parameters to plant
        self._apply_parameters(parameters)

        # Run simulation
        if simulation_duration is None:
            simulation_duration = len(measurements) * (1.0 / 24.0)  # Assume hourly data

        simulated_outputs = self._simulate_plant(measurements, simulation_duration)

        # Compute metrics for each objective
        metrics = {}
        for objective in objectives:
            if objective not in simulated_outputs:
                warnings.warn(f"Objective '{objective}' not in simulation outputs")
                continue

            observed = self._extract_measurements(measurements, objective)
            predicted = simulated_outputs[objective]

            # Align arrays
            observed, predicted = self._align_arrays(observed, predicted)

            if len(observed) == 0 or len(predicted) == 0:
                warnings.warn(f"No valid data for objective '{objective}'")
                continue

            # Calculate metrics
            obj_metrics = self._calculate_metrics(objective, observed, predicted)
            metrics[objective] = obj_metrics

            if self.verbose:
                print(f"\n{objective}:")
                print(f"  RMSE: {obj_metrics.rmse:.4f}")
                print(f"  R²: {obj_metrics.r2:.4f}")
                print(f"  NSE: {obj_metrics.nse:.4f}")

        return metrics

    def analyze_residuals(
        self,
        measurements: "MeasurementData",
        simulated: Dict[str, np.ndarray],
        objectives: Optional[List[str]] = None,
    ) -> Dict[str, ResidualAnalysis]:
        """
        Analyze residuals between measurements and simulated outputs.

        Performs statistical tests on residuals to assess model adequacy
        including normality, autocorrelation, and heteroscedasticity tests.

        Args:
            measurements: Measurement data
            simulated: Dictionary of simulated outputs
            objectives: List of objectives to analyze

        Returns:
            Dictionary mapping objective names to ResidualAnalysis

        Example:
            >>> residuals = validator.analyze_residuals(
            ...     measurements=data,
            ...     simulated=simulation_outputs
            ... )
            >>> if not residuals["Q_ch4"].is_normally_distributed():
            ...     print("Residuals are not normally distributed")
        """
        if objectives is None:
            objectives = list(simulated.keys())

        results = {}

        for objective in objectives:
            if objective not in simulated:
                continue

            observed = self._extract_measurements(measurements, objective)
            predicted = simulated[objective]

            # Align arrays
            observed, predicted = self._align_arrays(observed, predicted)

            if len(observed) < 3:  # Need at least 3 points
                warnings.warn(f"Insufficient data for residual analysis of '{objective}'")
                continue

            # Calculate residuals
            residuals = observed - predicted

            # Standardize residuals
            std_residuals = self._standardize_residuals(residuals)

            # Normality test (Shapiro-Wilk)
            normality = self._test_normality(residuals)

            # Autocorrelation (lag-1)
            autocorr = self._calculate_autocorrelation(residuals)

            # Heteroscedasticity test (simplified Breusch-Pagan)
            hetero = self._test_heteroscedasticity(residuals, predicted)

            # Identify outliers
            outliers = np.where(np.abs(std_residuals) > 3)[0].tolist()

            results[objective] = ResidualAnalysis(
                objective=objective,
                residuals=residuals,
                standardized_residuals=std_residuals,
                normality_test=normality,
                autocorrelation=autocorr,
                heteroscedasticity_test=hetero,
                outlier_indices=outliers,
            )

        return results

    def cross_validate(
        self,
        parameters: Dict[str, float],
        measurements: "MeasurementData",
        n_folds: int = 5,
        objectives: Optional[List[str]] = None,
    ) -> Dict[str, List[ValidationMetrics]]:
        """
        Perform k-fold cross-validation.

        Splits measurement data into k folds and validates on each fold,
        providing assessment of model generalization.

        Args:
            parameters: Calibrated parameters
            measurements: Full measurement dataset
            n_folds: Number of cross-validation folds
            objectives: List of objectives to validate

        Returns:
            Dictionary mapping objectives to lists of ValidationMetrics (one per fold)

        Example:
            >>> cv_results = validator.cross_validate(
            ...     parameters=params,
            ...     measurements=data,
            ...     n_folds=5
            ... )
            >>> # Calculate mean R² across folds
            >>> mean_r2 = np.mean([m.r2 for m in cv_results["Q_ch4"]])
        """
        if objectives is None:
            objectives = ["Q_ch4", "pH", "VFA"]

        n_samples = len(measurements)
        fold_size = n_samples // n_folds

        cv_results = {obj: [] for obj in objectives}

        if self.verbose:
            print(f"\nPerforming {n_folds}-fold cross-validation...")

        for fold in range(n_folds):
            if self.verbose:
                print(f"  Fold {fold + 1}/{n_folds}")

            # Split data
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < n_folds - 1 else n_samples

            # Validation set for this fold
            val_indices = list(range(start_idx, end_idx))
            val_data = measurements.data.iloc[val_indices].copy()
            val_measurements = type(measurements)(val_data)

            # Validate on this fold
            fold_metrics = self.validate(
                parameters=parameters,
                measurements=val_measurements,
                objectives=objectives,
            )

            # Store results
            for obj, metrics in fold_metrics.items():
                cv_results[obj].append(metrics)

        # Print summary
        if self.verbose:
            print("\nCross-validation summary:")
            for obj in objectives:
                if obj in cv_results and cv_results[obj]:
                    r2_values = [m.r2 for m in cv_results[obj]]
                    rmse_values = [m.rmse for m in cv_results[obj]]
                    print(f"\n{obj}:")
                    print(f"  R²: {np.mean(r2_values):.4f} ± {np.std(r2_values):.4f}")
                    print(f"  RMSE: {np.mean(rmse_values):.4f} ± {np.std(rmse_values):.4f}")

        return cv_results

    def analyze_parameter_correlation(
        self,
        parameter_history: List[Dict[str, float]],
        threshold: float = 0.7,
    ) -> ParameterCorrelation:
        """
        Analyze correlation between calibrated parameters.

        Uses optimization history to compute parameter correlation matrix
        and identify highly correlated parameters that may indicate
        identifiability issues.

        Args:
            parameter_history: List of parameter dictionaries from optimization
            threshold: Correlation threshold for flagging high correlations

        Returns:
            ParameterCorrelation object

        Example:
            >>> # From optimization history
            >>> history = calibration_result.history
            >>> param_hist = [h["parameters"] for h in history]
            >>> corr = validator.analyze_parameter_correlation(param_hist)
            >>> print(corr.high_correlations)
        """
        if not parameter_history:
            raise ValueError("Empty parameter history")

        # Extract parameter names and values
        param_names = list(parameter_history[0].keys())
        n_params = len(param_names)
        n_samples = len(parameter_history)

        # Build parameter matrix
        param_matrix = np.zeros((n_samples, n_params))
        for i, params in enumerate(parameter_history):
            for j, name in enumerate(param_names):
                param_matrix[i, j] = params[name]

        # Calculate correlation matrix
        corr_matrix = np.corrcoef(param_matrix.T)

        # Find high correlations
        high_corr = []
        for i in range(n_params):
            for j in range(i + 1, n_params):
                corr_val = corr_matrix[i, j]
                if abs(corr_val) > threshold:
                    high_corr.append((param_names[i], param_names[j], corr_val))

        # Calculate VIF if possible
        vif = None
        if n_params > 1:
            try:
                vif = self._calculate_vif(param_matrix, param_names)
            except Exception as e:
                print(e)  # VIF calculation failed

        return ParameterCorrelation(
            correlation_matrix=corr_matrix,
            parameter_names=param_names,
            high_correlations=high_corr,
            vif=vif,
        )

    def print_validation_report(
        self,
        metrics: Dict[str, ValidationMetrics],
        residuals: Optional[Dict[str, ResidualAnalysis]] = None,
    ) -> None:
        """
        Print comprehensive validation report.

        Args:
            metrics: Validation metrics from validate()
            residuals: Optional residual analysis from analyze_residuals()

        Example:
            >>> validator.print_validation_report(metrics, residuals)
        """
        print("=" * 70)
        print("CALIBRATION VALIDATION REPORT")
        print("=" * 70)

        for objective, obj_metrics in metrics.items():
            print(f"\n{objective}:")
            print("-" * 50)
            print(f"Number of samples: {obj_metrics.n_samples}")
            print("\nGoodness-of-fit metrics:")
            print(f"  RMSE:        {obj_metrics.rmse:.4f}")
            print(f"  MAE:         {obj_metrics.mae:.4f}")
            print(f"  R²:          {obj_metrics.r2:.4f}")
            print(f"  NSE:         {obj_metrics.nse:.4f}")
            print(f"  PBIAS:       {obj_metrics.pbias:.2f}%")
            print(f"  Correlation: {obj_metrics.correlation:.4f}")
            print(f"  MAPE:        {obj_metrics.mape:.2f}%")
            print(f"  ME (bias):   {obj_metrics.me:.4f}")

            print("\nObserved statistics:")
            print(f"  Mean: {obj_metrics.observations_mean:.4f}")
            print(f"  Std:  {obj_metrics.observations_std:.4f}")

            print("\nPredicted statistics:")
            print(f"  Mean: {obj_metrics.predictions_mean:.4f}")
            print(f"  Std:  {obj_metrics.predictions_std:.4f}")

            # Interpretation
            print("\nInterpretation:")
            if obj_metrics.r2 > 0.9:
                print("  ✓ Excellent fit (R² > 0.9)")
            elif obj_metrics.r2 > 0.7:
                print("  ✓ Good fit (R² > 0.7)")
            elif obj_metrics.r2 > 0.5:
                print("  ~ Moderate fit (R² > 0.5)")
            else:
                print("  ✗ Poor fit (R² < 0.5)")

            if abs(obj_metrics.pbias) < 10:
                print("  ✓ Low bias (|PBIAS| < 10%)")
            elif abs(obj_metrics.pbias) < 25:
                print("  ~ Moderate bias (|PBIAS| < 25%)")
            else:
                print("  ✗ High bias (|PBIAS| > 25%)")

        # Print residual analysis if provided
        if residuals:
            print("\n" + "=" * 70)
            print("RESIDUAL ANALYSIS")
            print("=" * 70)

            for objective, res_analysis in residuals.items():
                print(f"\n{objective}:")
                print("-" * 50)

                # Normality
                is_normal = res_analysis.is_normally_distributed()
                print("Normality (Shapiro-Wilk):")
                print(f"  p-value: {res_analysis.normality_test['p_value']:.4f}")
                print(
                    f"  {'✓' if is_normal else '✗'} Residuals are "
                    f"{'normally' if is_normal else 'NOT normally'} distributed"
                )

                # Autocorrelation
                has_autocorr = res_analysis.has_autocorrelation()
                print("\nAutocorrelation (lag-1):")
                print(f"  Coefficient: {res_analysis.autocorrelation:.4f}")
                print(
                    f"  {'⚠' if has_autocorr else '✓'} "
                    f"{'Significant' if has_autocorr else 'No significant'} autocorrelation"
                )

                # Heteroscedasticity
                has_hetero = res_analysis.has_heteroscedasticity()
                print("\nHeteroscedasticity:")
                print(f"  p-value: {res_analysis.heteroscedasticity_test['p_value']:.4f}")
                print(f"  {'⚠' if has_hetero else '✓'} " f"{'Detected' if has_hetero else 'Not detected'}")

                # Outliers
                n_outliers = len(res_analysis.outlier_indices)
                print(f"\nOutliers (|z| > 3): {n_outliers}")
                if n_outliers > 0:
                    pct_outliers = n_outliers / len(res_analysis.residuals) * 100
                    print(f"  ({pct_outliers:.1f}% of data)")

        print("\n" + "=" * 70)

    # ========================================================================
    # PRIVATE METHODS
    # ========================================================================

    def _apply_parameters(self, parameters: Dict[str, float]) -> None:
        """Apply parameters to plant model."""
        # TODO: Implement parameter application to plant components
        pass

    def _simulate_plant(
        self,
        measurements: "MeasurementData",
        duration: float,
    ) -> Dict[str, np.ndarray]:
        """
        Simulate plant and return outputs.

        Args:
            measurements: Measurement data (for substrate feeds)
            duration: Simulation duration [days]

        Returns:
            Dictionary of simulated outputs
        """
        # TODO: Implement actual plant simulation
        # For now, return dummy outputs
        n_samples = len(measurements)

        outputs = {
            "Q_ch4": np.random.randn(n_samples) * 10 + 750,
            "pH": np.random.randn(n_samples) * 0.1 + 7.2,
            "VFA": np.random.randn(n_samples) * 0.5 + 2.5,
            "TAC": np.random.randn(n_samples) * 1.0 + 15.0,
        }

        return outputs

    def _extract_measurements(
        self,
        measurements: "MeasurementData",
        objective: str,
    ) -> np.ndarray:
        """Extract measurement array for objective."""
        try:
            series = measurements.get_measurement(objective)
            return series.values
        except Exception:
            return np.array([])

    def _align_arrays(
        self,
        observed: np.ndarray,
        predicted: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align observed and predicted arrays, removing NaN values.

        Args:
            observed: Observed values
            predicted: Predicted values

        Returns:
            Tuple of aligned arrays
        """
        # Ensure same length
        min_len = min(len(observed), len(predicted))
        observed = observed[:min_len]
        predicted = predicted[:min_len]

        # Remove NaN values
        valid = ~(np.isnan(observed) | np.isnan(predicted))
        observed = observed[valid]
        predicted = predicted[valid]

        return observed, predicted

    def _calculate_metrics(
        self,
        objective: str,
        observed: np.ndarray,
        predicted: np.ndarray,
    ) -> ValidationMetrics:
        """Calculate all validation metrics."""
        n = len(observed)

        # Basic statistics
        obs_mean = np.mean(observed)
        obs_std = np.std(observed)
        pred_mean = np.mean(predicted)
        pred_std = np.std(predicted)

        # Errors
        residuals = observed - predicted
        abs_residuals = np.abs(residuals)

        # RMSE
        rmse = np.sqrt(np.mean(residuals**2))

        # MAE
        mae = np.mean(abs_residuals)

        # R² (coefficient of determination)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((observed - obs_mean) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Nash-Sutcliffe Efficiency (same as R² for predictions)
        nse = r2

        # Percent Bias
        pbias = (np.sum(residuals) / np.sum(observed)) * 100 if np.sum(observed) != 0 else 0.0

        # Pearson correlation
        correlation = np.corrcoef(observed, predicted)[0, 1]

        # MAPE (Mean Absolute Percentage Error)
        # Avoid division by zero
        nonzero = observed != 0
        if np.any(nonzero):
            mape = np.mean(abs_residuals[nonzero] / np.abs(observed[nonzero])) * 100
        else:
            mape = 0.0

        # ME (Mean Error / Bias)
        me = np.mean(residuals)

        return ValidationMetrics(
            objective=objective,
            n_samples=n,
            rmse=rmse,
            mae=mae,
            r2=r2,
            nse=nse,
            pbias=pbias,
            correlation=correlation,
            mape=mape,
            me=me,
            observations_mean=obs_mean,
            observations_std=obs_std,
            predictions_mean=pred_mean,
            predictions_std=pred_std,
        )

    def _standardize_residuals(self, residuals: np.ndarray) -> np.ndarray:
        """Standardize residuals to z-scores."""
        mean = np.mean(residuals)
        std = np.std(residuals)
        if std == 0:
            return np.zeros_like(residuals)
        return (residuals - mean) / std

    def _test_normality(self, residuals: np.ndarray) -> Dict[str, float]:
        """Perform Shapiro-Wilk normality test."""
        if len(residuals) < 3:
            return {"statistic": 0.0, "p_value": 1.0}

        try:
            statistic, p_value = stats.shapiro(residuals)
            return {"statistic": float(statistic), "p_value": float(p_value)}
        except Exception:
            return {"statistic": 0.0, "p_value": 1.0}

    def _calculate_autocorrelation(self, residuals: np.ndarray) -> float:
        """Calculate lag-1 autocorrelation."""
        if len(residuals) < 2:
            return 0.0

        # Lag-1 autocorrelation
        res_centered = residuals - np.mean(residuals)
        autocorr = np.corrcoef(res_centered[:-1], res_centered[1:])[0, 1]

        return float(autocorr) if not np.isnan(autocorr) else 0.0

    def _test_heteroscedasticity(
        self,
        residuals: np.ndarray,
        predicted: np.ndarray,
    ) -> Dict[str, float]:
        """
        Simplified Breusch-Pagan test for heteroscedasticity.

        Tests if variance of residuals depends on predicted values.
        """
        if len(residuals) < 10:
            return {"statistic": 0.0, "p_value": 1.0}

        try:
            # Regress squared residuals on predicted values
            squared_res = residuals**2

            # Correlation between squared residuals and predictions
            correlation = np.corrcoef(squared_res, predicted)[0, 1]

            # Convert to chi-square statistic (approximation)
            n = len(residuals)
            statistic = n * correlation**2

            # p-value from chi-square distribution with 1 df
            p_value = 1 - stats.chi2.cdf(statistic, df=1)

            return {"statistic": float(statistic), "p_value": float(p_value)}

        except Exception:
            return {"statistic": 0.0, "p_value": 1.0}

    def _calculate_vif(
        self,
        param_matrix: np.ndarray,
        param_names: List[str],
    ) -> Dict[str, float]:
        """
        Calculate Variance Inflation Factors.

        VIF measures multicollinearity between parameters.
        VIF > 10 indicates high multicollinearity.
        """
        n_params = param_matrix.shape[1]
        vif_dict = {}

        for i in range(n_params):
            try:
                # Use other parameters to predict this one
                X = np.delete(param_matrix, i, axis=1)
                y = param_matrix[:, i]

                # Simple linear regression (using correlation)
                if X.shape[1] == 0:
                    vif_dict[param_names[i]] = 1.0
                    continue

                # Calculate R²
                corr_matrix = np.corrcoef(X.T, y)
                r2 = corr_matrix[-1, :-1].mean() ** 2

                # VIF = 1 / (1 - R²)
                vif = 1.0 / (1.0 - r2) if r2 < 0.999 else 999.0

                vif_dict[param_names[i]] = float(vif)

            except Exception:
                vif_dict[param_names[i]] = 1.0

        return vif_dict
