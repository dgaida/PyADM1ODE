# pyadm1/calibration/optimization/objective.py
"""
Objective Functions for Parameter Calibration

This module provides various objective function classes for calibration optimization.
Supports single and multi-objective formulations with different error metrics.

Available objective functions:
- SingleObjective: Minimize error for one output (e.g., Q_ch4)
- MultiObjectiveFunction: Weighted combination of multiple outputs
- WeightedSumObjective: Alternative multi-objective with flexible weights
- LikelihoodObjective: Maximum likelihood estimation
- CustomObjective: User-defined objective functions

Example:
    >>> from pyadm1.calibration.optimization import MultiObjectiveFunction
    >>>
    >>> objective = MultiObjectiveFunction(
    ...     simulator=simulator,
    ...     measurements=measurements,
    ...     objectives=["Q_ch4", "pH", "VFA"],
    ...     weights={"Q_ch4": 0.6, "pH": 0.2, "VFA": 0.2},
    ...     error_metric="rmse"
    ... )
    >>>
    >>> # Evaluate objective at parameter values
    >>> params = np.array([0.5, 0.10])  # [k_dis, Y_su]
    >>> error = objective(params)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Callable
import numpy as np
from dataclasses import dataclass


@dataclass
class ErrorMetrics:
    """Container for different error metrics."""

    mse: float  # Mean Squared Error
    rmse: float  # Root Mean Squared Error
    mae: float  # Mean Absolute Error
    mape: float  # Mean Absolute Percentage Error
    me: float  # Mean Error (bias)
    r2: float  # R-squared
    nse: float  # Nash-Sutcliffe Efficiency

    @classmethod
    def compute(cls, observed: np.ndarray, predicted: np.ndarray) -> "ErrorMetrics":
        """
        Compute all error metrics.

        Args:
            observed: Observed/measured values
            predicted: Predicted/simulated values

        Returns:
            ErrorMetrics object
        """
        # Ensure arrays are valid
        observed = np.atleast_1d(observed)
        predicted = np.atleast_1d(predicted)

        # Remove NaN
        valid = ~(np.isnan(observed) | np.isnan(predicted))
        if not np.any(valid):
            return cls(
                mse=float("inf"),
                rmse=float("inf"),
                mae=float("inf"),
                mape=float("inf"),
                me=float("inf"),
                r2=-float("inf"),
                nse=-float("inf"),
            )

        observed = observed[valid]
        predicted = predicted[valid]

        # Calculate metrics
        residuals = observed - predicted
        abs_residuals = np.abs(residuals)

        mse = float(np.mean(residuals**2))
        rmse = float(np.sqrt(mse))
        mae = float(np.mean(abs_residuals))
        me = float(np.mean(residuals))

        # MAPE (avoid division by zero)
        nonzero = observed != 0
        if np.any(nonzero):
            mape = float(np.mean(abs_residuals[nonzero] / np.abs(observed[nonzero])) * 100)
        else:
            mape = float("inf")

        # R² and NSE
        obs_mean = np.mean(observed)
        ss_tot = np.sum((observed - obs_mean) ** 2)
        ss_res = np.sum(residuals**2)

        if ss_tot > 0:
            r2 = float(1 - ss_res / ss_tot)
            nse = r2  # NSE is equivalent to R² for predictions
        else:
            r2 = -float("inf")
            nse = -float("inf")

        return cls(mse=mse, rmse=rmse, mae=mae, mape=mape, me=me, r2=r2, nse=nse)


class ObjectiveFunction(ABC):
    """
    Abstract base class for objective functions.

    All objective functions must implement __call__ to evaluate the
    objective for given parameter values.

    Attributes:
        parameter_names: Names of parameters being optimized
        lower_is_better: Whether lower objective values are better (default: True)
    """

    def __init__(self, parameter_names: List[str], lower_is_better: bool = True):
        """
        Initialize objective function.

        Args:
            parameter_names: Names of parameters in order
            lower_is_better: Whether to minimize (True) or maximize (False)
        """
        self.parameter_names = parameter_names
        self.lower_is_better = lower_is_better

    @abstractmethod
    def __call__(self, x: np.ndarray) -> float:
        """
        Evaluate objective function.

        Args:
            x: Parameter values

        Returns:
            Objective function value
        """
        pass

    def _params_to_dict(self, x: np.ndarray) -> Dict[str, float]:
        """Convert parameter array to dictionary."""
        return {name: float(val) for name, val in zip(self.parameter_names, x)}


class SingleObjective(ObjectiveFunction):
    """
    Single-objective function for one output variable.

    Minimizes error between simulated and measured values for a single
    output (e.g., methane production).

    Example:
        >>> objective = SingleObjective(
        ...     simulator=simulator,
        ...     measurements=measurements,
        ...     objective_name="Q_ch4",
        ...     parameter_names=["k_dis", "Y_su"],
        ...     error_metric="rmse"
        ... )
    """

    def __init__(
        self,
        simulator: Callable[[Dict[str, float]], Dict[str, np.ndarray]],
        measurements: np.ndarray,
        objective_name: str,
        parameter_names: List[str],
        error_metric: str = "rmse",
    ):
        """
        Initialize single objective.

        Args:
            simulator: Function that takes parameters dict and returns simulated outputs
            measurements: Measured values for the objective
            objective_name: Name of output to match
            parameter_names: Names of parameters
            error_metric: Error metric ("mse", "rmse", "mae", "mape")
        """
        super().__init__(parameter_names)

        self.simulator = simulator
        self.measurements = measurements
        self.objective_name = objective_name
        self.error_metric = error_metric.lower()

    def __call__(self, x: np.ndarray) -> float:
        """
        Evaluate objective.

        Args:
            x: Parameter values

        Returns:
            Error value
        """
        # Convert to parameter dict
        params = self._params_to_dict(x)

        try:
            # Run simulation
            outputs = self.simulator(params)

            # Get simulated values for this objective
            if self.objective_name not in outputs:
                return 1e10  # Penalty if output not available

            simulated = outputs[self.objective_name]

            # Compute error metrics
            metrics = ErrorMetrics.compute(self.measurements, simulated)

            # Return selected metric
            if self.error_metric == "mse":
                return metrics.mse
            elif self.error_metric == "rmse":
                return metrics.rmse
            elif self.error_metric == "mae":
                return metrics.mae
            elif self.error_metric == "mape":
                return metrics.mape
            elif self.error_metric == "nse":
                return -metrics.nse  # Maximize NSE = minimize -NSE
            elif self.error_metric == "r2":
                return -metrics.r2  # Maximize R²
            else:
                return metrics.rmse

        except Exception as e:
            print(f"Simulation error: {e}")
            return 1e10


class MultiObjectiveFunction(ObjectiveFunction):
    """
    Multi-objective function with weighted combination.

    Combines errors from multiple outputs using weights to create
    a single scalar objective.

    Example:
        >>> objective = MultiObjectiveFunction(
        ...     simulator=simulator,
        ...     measurements_dict={
        ...         "Q_ch4": measured_ch4,
        ...         "pH": measured_ph,
        ...         "VFA": measured_vfa
        ...     },
        ...     objectives=["Q_ch4", "pH", "VFA"],
        ...     weights={"Q_ch4": 0.6, "pH": 0.2, "VFA": 0.2},
        ...     parameter_names=["k_dis", "Y_su", "k_hyd_ch"],
        ...     error_metric="rmse"
        ... )
    """

    def __init__(
        self,
        simulator: Callable[[Dict[str, float]], Dict[str, np.ndarray]],
        measurements_dict: Dict[str, np.ndarray],
        objectives: List[str],
        weights: Dict[str, float],
        parameter_names: List[str],
        error_metric: str = "rmse",
        normalize: bool = True,
    ):
        """
        Initialize multi-objective function.

        Args:
            simulator: Function that takes parameters and returns outputs
            measurements_dict: Dictionary mapping objective names to measurements
            objectives: List of objective names
            weights: Dictionary of weights for each objective
            parameter_names: Names of parameters
            error_metric: Error metric to use
            normalize: Normalize errors by mean of measurements
        """
        super().__init__(parameter_names)

        self.simulator = simulator
        self.measurements_dict = measurements_dict
        self.objectives = objectives
        self.weights = weights
        self.error_metric = error_metric.lower()
        self.normalize = normalize

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in weights.items()}

    def __call__(self, x: np.ndarray) -> float:
        """
        Evaluate multi-objective function.

        Args:
            x: Parameter values

        Returns:
            Weighted sum of errors
        """
        params = self._params_to_dict(x)

        try:
            # Run simulation
            outputs = self.simulator(params)

            # Calculate weighted error
            total_error = 0.0
            n_valid = 0

            for obj_name in self.objectives:
                if obj_name not in outputs or obj_name not in self.measurements_dict:
                    continue

                simulated = outputs[obj_name]
                measured = self.measurements_dict[obj_name]

                # Compute error
                metrics = ErrorMetrics.compute(measured, simulated)

                # Get error value
                if self.error_metric == "mse":
                    error = metrics.mse
                elif self.error_metric == "mae":
                    error = metrics.mae
                elif self.error_metric == "mape":
                    error = metrics.mape
                elif self.error_metric == "nse":
                    error = -metrics.nse
                elif self.error_metric == "r2":
                    error = -metrics.r2
                else:  # Default to RMSE
                    error = metrics.rmse

                # Normalize by mean of measurements if requested
                if self.normalize:
                    mean_measured = np.mean(np.abs(measured))
                    if mean_measured > 1e-10:
                        error = error / mean_measured

                # Add weighted error
                weight = self.weights.get(obj_name, 0.0)
                total_error += weight * error
                n_valid += 1

            if n_valid == 0:
                return 1e10

            return total_error

        except Exception as e:
            print(f"Simulation error: {e}")
            return 1e10


class WeightedSumObjective(MultiObjectiveFunction):
    """
    Alias for MultiObjectiveFunction with equal weights by default.

    Convenient constructor for common use case.
    """

    def __init__(
        self,
        simulator: Callable[[Dict[str, float]], Dict[str, np.ndarray]],
        measurements_dict: Dict[str, np.ndarray],
        objectives: List[str],
        parameter_names: List[str],
        weights: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        """Initialize with equal weights if not specified."""
        if weights is None:
            # Equal weights
            weights = {obj: 1.0 / len(objectives) for obj in objectives}

        super().__init__(simulator, measurements_dict, objectives, weights, parameter_names, **kwargs)


class LikelihoodObjective(ObjectiveFunction):
    """
    Maximum likelihood objective function.

    Assumes Gaussian errors and maximizes likelihood (minimizes negative
    log-likelihood). Useful for statistical parameter estimation.

    Example:
        >>> objective = LikelihoodObjective(
        ...     simulator=simulator,
        ...     measurements_dict=measurements,
        ...     objectives=["Q_ch4"],
        ...     parameter_names=["k_dis", "Y_su"]
        ... )
    """

    def __init__(
        self,
        simulator: Callable[[Dict[str, float]], Dict[str, np.ndarray]],
        measurements_dict: Dict[str, np.ndarray],
        objectives: List[str],
        parameter_names: List[str],
        sigma: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize likelihood objective.

        Args:
            simulator: Simulator function
            measurements_dict: Measurements for each objective
            objectives: List of objectives
            parameter_names: Parameter names
            sigma: Standard deviations for each objective (estimated if None)
        """
        super().__init__(parameter_names)

        self.simulator = simulator
        self.measurements_dict = measurements_dict
        self.objectives = objectives
        self.sigma = sigma or {}

        # Estimate sigma if not provided
        for obj_name in objectives:
            if obj_name not in self.sigma:
                measured = measurements_dict[obj_name]
                self.sigma[obj_name] = float(np.std(measured) + 1e-10)

    def __call__(self, x: np.ndarray) -> float:
        """
        Evaluate negative log-likelihood.

        Args:
            x: Parameter values

        Returns:
            Negative log-likelihood
        """
        params = self._params_to_dict(x)

        try:
            outputs = self.simulator(params)

            # Calculate negative log-likelihood
            neg_log_likelihood = 0.0
            n_total = 0

            for obj_name in self.objectives:
                if obj_name not in outputs or obj_name not in self.measurements_dict:
                    continue

                simulated = outputs[obj_name]
                measured = self.measurements_dict[obj_name]

                # Align arrays
                measured = np.atleast_1d(measured)
                simulated = np.atleast_1d(simulated)
                min_len = min(len(measured), len(simulated))
                measured = measured[:min_len]
                simulated = simulated[:min_len]

                # Remove NaN
                valid = ~(np.isnan(measured) | np.isnan(simulated))
                if not np.any(valid):
                    continue

                measured = measured[valid]
                simulated = simulated[valid]

                # Calculate log-likelihood
                sigma = self.sigma[obj_name]
                residuals = measured - simulated

                # -log(L) = 0.5 * sum((residuals/sigma)^2) + n*log(sigma) + constants
                n = len(residuals)
                nll = 0.5 * np.sum((residuals / sigma) ** 2) + n * np.log(sigma)

                neg_log_likelihood += nll
                n_total += n

            if n_total == 0:
                return 1e10

            return neg_log_likelihood

        except Exception as e:
            print(f"Simulation error: {e}")
            return 1e10


class CustomObjective(ObjectiveFunction):
    """
    Custom user-defined objective function.

    Allows users to define their own objective function logic while
    maintaining compatibility with the optimization framework.

    Example:
        >>> def my_objective(simulated, measured):
        ...     # Custom error calculation
        ...     return np.sum((simulated - measured)**4)
        >>>
        >>> objective = CustomObjective(
        ...     simulator=simulator,
        ...     measurements_dict=measurements,
        ...     objectives=["Q_ch4"],
        ...     parameter_names=["k_dis"],
        ...     custom_func=my_objective
        ... )
    """

    def __init__(
        self,
        simulator: Callable[[Dict[str, float]], Dict[str, np.ndarray]],
        measurements_dict: Dict[str, np.ndarray],
        objectives: List[str],
        parameter_names: List[str],
        custom_func: Callable[[np.ndarray, np.ndarray], float],
    ):
        """
        Initialize custom objective.

        Args:
            simulator: Simulator function
            measurements_dict: Measurements
            objectives: Objectives to evaluate
            parameter_names: Parameter names
            custom_func: Custom function(simulated, measured) -> error
        """
        super().__init__(parameter_names)

        self.simulator = simulator
        self.measurements_dict = measurements_dict
        self.objectives = objectives
        self.custom_func = custom_func

    def __call__(self, x: np.ndarray) -> float:
        """Evaluate custom objective."""
        params = self._params_to_dict(x)

        try:
            outputs = self.simulator(params)

            total_error = 0.0
            n_valid = 0

            for obj_name in self.objectives:
                if obj_name not in outputs or obj_name not in self.measurements_dict:
                    continue

                simulated = outputs[obj_name]
                measured = self.measurements_dict[obj_name]

                # Apply custom function
                error = self.custom_func(simulated, measured)
                total_error += error
                n_valid += 1

            if n_valid == 0:
                return 1e10

            return total_error / n_valid

        except Exception as e:
            print(f"Simulation error: {e}")
            return 1e10


# Convenience function for creating common objectives
def create_objective(
    objective_type: str,
    simulator: Callable[[Dict[str, float]], Dict[str, np.ndarray]],
    measurements_dict: Dict[str, np.ndarray],
    objectives: List[str],
    parameter_names: List[str],
    **kwargs,
) -> ObjectiveFunction:
    """
    Factory function to create objective functions.

    Args:
        objective_type: Type of objective ("single", "multi", "weighted", "likelihood")
        simulator: Simulator function
        measurements_dict: Measurements dictionary
        objectives: List of objectives
        parameter_names: Parameter names
        **kwargs: Additional arguments for specific objective types

    Returns:
        ObjectiveFunction instance

    Example:
        >>> objective = create_objective(
        ...     objective_type="multi",
        ...     simulator=simulator,
        ...     measurements_dict=measurements,
        ...     objectives=["Q_ch4", "pH"],
        ...     parameter_names=["k_dis", "Y_su"],
        ...     weights={"Q_ch4": 0.8, "pH": 0.2}
        ... )
    """
    objective_type = objective_type.lower()

    if objective_type == "single":
        if len(objectives) != 1:
            raise ValueError("Single objective requires exactly one objective")
        return SingleObjective(
            simulator=simulator,
            measurements=measurements_dict[objectives[0]],
            objective_name=objectives[0],
            parameter_names=parameter_names,
            **kwargs,
        )

    elif objective_type in ["multi", "weighted"]:
        weights = kwargs.pop("weights", None)
        return MultiObjectiveFunction(
            simulator=simulator,
            measurements_dict=measurements_dict,
            objectives=objectives,
            weights=weights or {obj: 1.0 / len(objectives) for obj in objectives},
            parameter_names=parameter_names,
            **kwargs,
        )

    elif objective_type == "likelihood":
        return LikelihoodObjective(
            simulator=simulator,
            measurements_dict=measurements_dict,
            objectives=objectives,
            parameter_names=parameter_names,
            **kwargs,
        )

    else:
        raise ValueError(f"Unknown objective type: {objective_type}")
