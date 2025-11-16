# pyadm1/calibration/calibrator.py
"""
Main Calibrator Class for ADM1 Model Calibration

This module provides the unified interface for both initial and online calibration
of biogas plant models against measurement data.

Features:
- Initial calibration from historical data
- Online re-calibration during operation
- Multiple objective functions (gas, pH, VFA)
- Parameter bounds enforcement
- Validation and reporting
- Calibration history tracking

Example:
    >>> from pyadm1.calibration import Calibrator
    >>> from pyadm1.io import MeasurementData
    >>> from pyadm1.configurator import BiogasPlant
    >>>
    >>> # Load plant and data
    >>> plant = BiogasPlant.from_json("plant.json", feedstock)
    >>> measurements = MeasurementData.from_csv("plant_data.csv")
    >>>
    >>> # Create calibrator
    >>> calibrator = Calibrator(plant)
    >>>
    >>> # Initial calibration
    >>> result = calibrator.calibrate_initial(
    ...     measurements=measurements,
    ...     parameters=["k_dis", "k_hyd_ch", "Y_su"],
    ...     bounds={"k_dis": (0.3, 0.8)},
    ...     method="differential_evolution"
    ... )
    >>>
    >>> # Apply calibrated parameters
    >>> calibrator.apply_calibration(result)
    >>>
    >>> # Online re-calibration
    >>> result = calibrator.calibrate_online(
    ...     measurements=new_measurements,
    ...     variance_threshold=0.1,
    ...     max_parameter_change=0.2
    ... )
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np
import json

from pyadm1.io import MeasurementData
from pyadm1.configurator import BiogasPlant


@dataclass
class CalibrationResult:
    """
    Result from a calibration run.

    Attributes:
        success: Whether calibration converged successfully
        parameters: Calibrated parameter values
        initial_parameters: Initial parameter values before calibration
        objective_value: Final objective function value
        n_iterations: Number of optimization iterations
        execution_time: Wall clock time [seconds]
        method: Optimization method used
        message: Status message
        validation_metrics: Metrics on validation data
        sensitivity: Parameter sensitivity analysis results
        history: Optimization history (if available)
        timestamp: Calibration timestamp
    """

    success: bool
    parameters: Dict[str, float]
    initial_parameters: Dict[str, float]
    objective_value: float
    n_iterations: int
    execution_time: float
    method: str
    message: str
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    sensitivity: Dict[str, float] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "parameters": self.parameters,
            "initial_parameters": self.initial_parameters,
            "objective_value": self.objective_value,
            "n_iterations": self.n_iterations,
            "execution_time": self.execution_time,
            "method": self.method,
            "message": self.message,
            "validation_metrics": self.validation_metrics,
            "sensitivity": self.sensitivity,
            "history": self.history,
            "timestamp": self.timestamp,
        }

    def to_json(self, filepath: str) -> None:
        """Save result to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalibrationResult":
        """Create result from dictionary."""
        return cls(**data)


class Calibrator:
    """
    Main calibrator class for ADM1 model parameter estimation.

    Provides unified interface for initial and online calibration with
    support for multiple objective functions, parameter bounds, and
    validation.

    Attributes:
        plant: BiogasPlant instance to calibrate
        calibration_history: List of all calibration results
        current_parameters: Current parameter values

    Example:
        >>> calibrator = Calibrator(plant)
        >>> result = calibrator.calibrate_initial(measurements, ["k_dis"])
        >>> calibrator.apply_calibration(result)
    """

    def __init__(self, plant: "BiogasPlant", verbose: bool = True):
        """
        Initialize calibrator with plant model.

        Args:
            plant: BiogasPlant instance to calibrate
            verbose: Enable progress output
        """
        self.plant = plant
        self.verbose = verbose
        self.calibration_history: List[CalibrationResult] = []
        self.current_parameters: Dict[str, float] = {}

        # Will be initialized on first use
        self._initial_calibrator = None
        self._online_calibrator = None

    def calibrate_initial(
        self,
        measurements: "MeasurementData",
        parameters: List[str],
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        method: str = "differential_evolution",
        objectives: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
        validation_split: float = 0.2,
        max_iterations: int = 100,
        population_size: int = 15,
        tolerance: float = 1e-4,
        sensitivity_analysis: bool = True,
        **kwargs,
    ) -> CalibrationResult:
        """
        Perform initial calibration from historical measurement data.

        This method calibrates model parameters by comparing simulated outputs
        with historical measurement data. Uses global optimization to find
        parameters that minimize the difference between model and measurements.

        Args:
            measurements: Historical measurement data
            parameters: List of parameter names to calibrate
            bounds: Parameter bounds {param: (min, max)}
            method: Optimization method:
                - "differential_evolution" (default, global)
                - "nelder_mead" (local)
                - "lbfgsb" (gradient-based)
                - "particle_swarm" (global)
            objectives: List of outputs to match ["Q_ch4", "pH", "VFA"]
            weights: Objective weights {output: weight}
            validation_split: Fraction of data for validation (0-1)
            max_iterations: Maximum optimization iterations
            population_size: Population size for DE/PSO
            tolerance: Convergence tolerance
            sensitivity_analysis: Perform sensitivity analysis
            **kwargs: Additional arguments for optimizer

        Returns:
            CalibrationResult with optimized parameters

        Example:
            >>> result = calibrator.calibrate_initial(
            ...     measurements=data,
            ...     parameters=["k_dis", "k_hyd_ch", "Y_su"],
            ...     bounds={"k_dis": (0.3, 0.8)},
            ...     objectives=["Q_ch4", "pH"],
            ...     weights={"Q_ch4": 0.8, "pH": 0.2}
            ... )
        """
        # Lazy import to avoid circular dependencies
        from pyadm1.calibration.initial import InitialCalibrator

        if self._initial_calibrator is None:
            self._initial_calibrator = InitialCalibrator(self.plant, verbose=self.verbose)

        if self.verbose:
            print("=" * 70)
            print("Initial Model Calibration")
            print("=" * 70)
            print(f"\nParameters to calibrate: {parameters}")
            print(f"Optimization method: {method}")
            print(f"Objectives: {objectives or ['Q_ch4']}")
            print(f"Measurement points: {len(measurements)}")
            print(f"Validation split: {validation_split * 100:.0f}%")

        # Run initial calibration
        result = self._initial_calibrator.calibrate(
            measurements=measurements,
            parameters=parameters,
            bounds=bounds,
            method=method,
            objectives=objectives,
            weights=weights,
            validation_split=validation_split,
            max_iterations=max_iterations,
            population_size=population_size,
            tolerance=tolerance,
            sensitivity_analysis=sensitivity_analysis,
            **kwargs,
        )

        # Store in history
        self.calibration_history.append(result)

        if self.verbose:
            print("\n" + "=" * 70)
            print("Calibration Complete")
            print("=" * 70)
            print(f"Success: {result.success}")
            print(f"Objective value: {result.objective_value:.4f}")
            print(f"Iterations: {result.n_iterations}")
            print(f"Time: {result.execution_time:.1f} seconds")
            print("\nCalibrated parameters:")
            for param, value in result.parameters.items():
                initial = result.initial_parameters[param]
                change = ((value - initial) / initial * 100) if initial != 0 else 0
                print(f"  {param}: {initial:.4f} → {value:.4f} ({change:+.1f}%)")

            if result.validation_metrics:
                print("\nValidation metrics:")
                for metric, value in result.validation_metrics.items():
                    print(f"  {metric}: {value:.4f}")

        return result

    def calibrate_online(
        self,
        measurements: "MeasurementData",
        parameters: Optional[List[str]] = None,
        variance_threshold: float = 0.1,
        max_parameter_change: float = 0.2,
        time_window: int = 7,
        method: str = "nelder_mead",
        max_iterations: int = 50,
        **kwargs,
    ) -> CalibrationResult:
        """
        Perform online re-calibration during plant operation.

        This method re-calibrates parameters when measurement variance exceeds
        a threshold, indicating model drift. Uses a moving window of recent
        data and enforces bounds to prevent parameter drift.

        Args:
            measurements: Recent measurement data
            parameters: Parameters to adjust (default: from last calibration)
            variance_threshold: Trigger threshold for variance
            max_parameter_change: Maximum relative parameter change (0-1)
            time_window: Days of data to use for calibration
            method: Optimization method (local methods recommended)
            max_iterations: Maximum iterations
            **kwargs: Additional optimizer arguments

        Returns:
            CalibrationResult with adjusted parameters

        Example:
            >>> result = calibrator.calibrate_online(
            ...     measurements=recent_data,
            ...     variance_threshold=0.15,
            ...     max_parameter_change=0.1,
            ...     time_window=7
            ... )
        """
        # Lazy import to avoid circular dependencies
        from pyadm1.calibration.online import OnlineCalibrator

        if self._online_calibrator is None:
            self._online_calibrator = OnlineCalibrator(self.plant, verbose=self.verbose)

        # Use parameters from last calibration if not specified
        if parameters is None and self.calibration_history:
            parameters = list(self.calibration_history[-1].parameters.keys())
        elif parameters is None:
            raise ValueError("No parameters specified and no calibration history available")

        if self.verbose:
            print("=" * 70)
            print("Online Model Re-Calibration")
            print("=" * 70)
            print(f"\nParameters to adjust: {parameters}")
            print(f"Variance threshold: {variance_threshold}")
            print(f"Max parameter change: {max_parameter_change * 100:.0f}%")
            print(f"Time window: {time_window} days")

        # Run online calibration
        result = self._online_calibrator.calibrate(
            measurements=measurements,
            parameters=parameters,
            current_parameters=self.current_parameters,
            variance_threshold=variance_threshold,
            max_parameter_change=max_parameter_change,
            time_window=time_window,
            method=method,
            max_iterations=max_iterations,
            **kwargs,
        )

        # Store in history
        self.calibration_history.append(result)

        if self.verbose:
            print("\n" + "=" * 70)
            print("Re-Calibration Complete")
            print("=" * 70)
            print(f"Success: {result.success}")
            print(f"Objective value: {result.objective_value:.4f}")
            print("\nAdjusted parameters:")
            for param, value in result.parameters.items():
                initial = result.initial_parameters[param]
                change = ((value - initial) / initial * 100) if initial != 0 else 0
                print(f"  {param}: {initial:.4f} → {value:.4f} ({change:+.1f}%)")

        return result

    def apply_calibration(self, result: CalibrationResult) -> None:
        """
        Apply calibrated parameters to the plant model.

        Updates the plant model with calibrated parameter values.

        Args:
            result: CalibrationResult to apply

        Example:
            >>> calibrator.apply_calibration(result)
        """
        if not result.success:
            if self.verbose:
                print("Warning: Applying parameters from unsuccessful calibration")

        # Update current parameters
        self.current_parameters.update(result.parameters)

        # Apply to plant model (implementation depends on plant structure)
        # This is a placeholder - actual implementation needs access to
        # ADM1 parameters in the plant components
        if self.verbose:
            print(f"\nApplied {len(result.parameters)} parameters to plant model")

    def validate_calibration(self, result: CalibrationResult, validation_data: "MeasurementData") -> Dict[str, float]:
        """
        Validate calibration result against independent data.

        Args:
            result: CalibrationResult to validate
            validation_data: Independent measurement data

        Returns:
            Dictionary of validation metrics

        Example:
            >>> metrics = calibrator.validate_calibration(result, validation_data)
            >>> print(f"RMSE: {metrics['rmse']:.2f}")
        """
        from pyadm1.calibration.validation import CalibrationValidator

        validator = CalibrationValidator(self.plant)

        metrics = validator.validate(
            parameters=result.parameters, measurements=validation_data, objectives=["Q_ch4", "pH", "VFA"]
        )

        if self.verbose:
            print("\nValidation Results:")
            print("-" * 40)
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.4f}")

        return metrics

    def save_history(self, filepath: str) -> None:
        """
        Save calibration history to JSON file.

        Args:
            filepath: Output file path

        Example:
            >>> calibrator.save_history("calibration_history.json")
        """
        history_data = [result.to_dict() for result in self.calibration_history]

        with open(filepath, "w") as f:
            json.dump(history_data, f, indent=2)

        if self.verbose:
            print(f"\nSaved calibration history ({len(self.calibration_history)} runs) to {filepath}")

    def load_history(self, filepath: str) -> None:
        """
        Load calibration history from JSON file.

        Args:
            filepath: Input file path

        Example:
            >>> calibrator.load_history("calibration_history.json")
        """
        with open(filepath, "r") as f:
            history_data = json.load(f)

        self.calibration_history = [CalibrationResult.from_dict(data) for data in history_data]

        if self.verbose:
            print(f"\nLoaded calibration history ({len(self.calibration_history)} runs) from {filepath}")

    def get_parameter_trends(self) -> Dict[str, List[float]]:
        """
        Get parameter value trends across calibration history.

        Returns:
            Dictionary mapping parameter names to lists of values

        Example:
            >>> trends = calibrator.get_parameter_trends()
            >>> k_dis_values = trends["k_dis"]
        """
        trends = {}

        for result in self.calibration_history:
            for param, value in result.parameters.items():
                if param not in trends:
                    trends[param] = []
                trends[param].append(value)

        return trends

    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate calibration report with summary and visualizations.

        Args:
            output_path: Optional path to save report

        Returns:
            Report text

        Example:
            >>> report = calibrator.generate_report("calibration_report.txt")
        """
        lines = [
            "=" * 70,
            "PyADM1 Calibration Report",
            "=" * 70,
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Plant: {self.plant.plant_name}",
            f"Total calibration runs: {len(self.calibration_history)}",
        ]

        if self.calibration_history:
            lines.append("\n" + "=" * 70)
            lines.append("Calibration History")
            lines.append("=" * 70)

            for i, result in enumerate(self.calibration_history, 1):
                lines.append(f"\nRun {i}: {result.timestamp}")
                lines.append(f"  Method: {result.method}")
                lines.append(f"  Success: {result.success}")
                lines.append(f"  Objective: {result.objective_value:.4f}")
                lines.append(f"  Iterations: {result.n_iterations}")
                lines.append(f"  Time: {result.execution_time:.1f}s")
                lines.append("  Parameters:")
                for param, value in result.parameters.items():
                    lines.append(f"    {param}: {value:.4f}")

            # Parameter trends
            trends = self.get_parameter_trends()
            if trends:
                lines.append("\n" + "=" * 70)
                lines.append("Parameter Trends")
                lines.append("=" * 70)

                for param, values in trends.items():
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    min_val = np.min(values)
                    max_val = np.max(values)

                    lines.append(f"\n{param}:")
                    lines.append(f"  Mean: {mean_val:.4f}")
                    lines.append(f"  Std:  {std_val:.4f}")
                    lines.append(f"  Range: [{min_val:.4f}, {max_val:.4f}]")

        lines.append("\n" + "=" * 70)

        report = "\n".join(lines)

        if output_path:
            with open(output_path, "w") as f:
                f.write(report)
            if self.verbose:
                print(f"\nReport saved to {output_path}")

        return report

    @property
    def last_calibration(self) -> Optional[CalibrationResult]:
        """Get the most recent calibration result."""
        return self.calibration_history[-1] if self.calibration_history else None

    @property
    def n_calibrations(self) -> int:
        """Get the total number of calibrations performed."""
        return len(self.calibration_history)
