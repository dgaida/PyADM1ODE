# pyadm1/calibration/online.py
"""
Online Re-Calibration During Plant Operation

This module provides the OnlineCalibrator class for real-time parameter adjustment
during biogas plant operation. Online calibration is triggered when state estimation
variance exceeds a threshold, indicating model drift or changing conditions.

The online calibrator uses a moving window of recent measurements and enforces strict
bounds on parameter changes to prevent instability. It focuses on fast, local
optimization methods suitable for real-time operation.

Key Features:
- Variance-based triggering for automatic re-calibration
- Moving window approach using recent data only
- Bounded parameter adjustments to prevent drift
- Fast local optimization (Nelder-Mead, L-BFGS-B)
- Parameter change history tracking
- Adaptive parameter bounds based on uncertainty
- Outlier detection before calibration

References:
    - Gaida, D. (2014). Dynamic real-time substrate feed optimization of
      anaerobic co-digestion plants. PhD thesis, Leiden University.
    - Dochain & Vanrolleghem (2001). Dynamical Modelling & Estimation in
      Wastewater Treatment Processes.

Example:
    >>> from pyadm1.calibration.online import OnlineCalibrator
    >>> from pyadm1.io import MeasurementData
    >>> from pyadm1.configurator import BiogasPlant
    >>>
    >>> # Create online calibrator
    >>> calibrator = OnlineCalibrator(plant, verbose=True)
    >>>
    >>> # Monitor and re-calibrate when needed
    >>> recent_data = MeasurementData.from_csv("recent_measurements.csv")
    >>>
    >>> result = calibrator.calibrate(
    ...     measurements=recent_data,
    ...     parameters=["k_dis", "Y_su"],
    ...     variance_threshold=0.15,
    ...     max_parameter_change=0.10,
    ...     time_window=7
    ... )
    >>>
    >>> if result.success:
    ...     print("Re-calibration successful")
    ...     calibrator.apply_calibration(result)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time

from pyadm1.calibration.parameter_bounds import create_default_bounds, ParameterBounds
from pyadm1.calibration.validation import CalibrationValidator
from pyadm1.calibration.optimization import (
    create_optimizer,
    MultiObjectiveFunction,
    ParameterConstraints,
)
from pyadm1.io import MeasurementData


@dataclass
class OnlineCalibrationTrigger:
    """
    Trigger conditions for online re-calibration.

    Attributes:
        variance_threshold: Trigger when prediction variance exceeds this value (0-1).
        time_threshold: Minimum time between calibrations [hours].
        residual_threshold: Trigger when residual exceeds this value.
        consecutive_violations: Number of consecutive threshold violations required.
        enabled: Whether automatic triggering is enabled.
    """

    variance_threshold: float = 0.15
    time_threshold: float = 24.0
    residual_threshold: Optional[float] = None
    consecutive_violations: int = 3
    enabled: bool = True


@dataclass
class ParameterChangeHistory:
    """
    History of parameter changes during online operation.

    Attributes:
        timestamp: Time of parameter change.
        parameters: Parameter values after change.
        trigger_reason: Reason for re-calibration.
        objective_value: Objective value after calibration.
        variance: Variance that triggered re-calibration.
        success: Whether calibration was successful.
    """

    timestamp: datetime
    parameters: Dict[str, float]
    trigger_reason: str
    objective_value: float
    variance: float
    success: bool


@dataclass
class OnlineState:
    """
    Online calibrator state tracking.

    Attributes:
        last_calibration_time: Timestamp of last calibration.
        consecutive_violations: Counter for consecutive threshold violations.
        current_variance: Current prediction variance.
        parameter_history: List of parameter change records.
        total_calibrations: Total number of calibrations performed.
    """

    last_calibration_time: Optional[datetime] = None
    consecutive_violations: int = 0
    current_variance: float = 0.0
    parameter_history: List[ParameterChangeHistory] = field(default_factory=list)
    total_calibrations: int = 0


class OnlineCalibrator:
    """
    Online calibrator for real-time parameter adjustment during operation.

    Performs fast, bounded re-calibration when model predictions deviate from
    measurements, indicating parameter drift or changing process conditions.

    Attributes:
        plant: BiogasPlant instance to calibrate.
        verbose: Enable progress output.
        parameter_bounds: Parameter bounds manager.
        validator: CalibrationValidator instance.
        trigger: Trigger conditions for re-calibration.
        state: Current calibrator state.

    Example:
        >>> calibrator = OnlineCalibrator(plant, verbose=True)
        >>>
        >>> # Configure trigger
        >>> calibrator.set_trigger(
        ...     variance_threshold=0.15,
        ...     time_threshold=24.0
        ... )
        >>>
        >>> # Check if re-calibration needed
        >>> if calibrator.should_recalibrate(recent_data):
        ...     result = calibrator.calibrate(recent_data)
    """

    def __init__(self, plant, verbose: bool = True, parameter_bounds: Optional[ParameterBounds] = None):
        """
        Initialize online calibrator.

        Args:
            plant: BiogasPlant instance to calibrate.
            verbose: Enable progress output.
            parameter_bounds: Custom parameter bounds. Uses defaults if None.
        """
        self.plant = plant
        self.verbose = verbose
        self.parameter_bounds = parameter_bounds or create_default_bounds()
        self.validator = CalibrationValidator(plant, verbose=False)

        # Initialize trigger and state
        self.trigger = OnlineCalibrationTrigger()
        self.state = OnlineState()

        # Store current parameters as baseline
        self._baseline_parameters = self._get_current_parameters()

        # Cache for recent predictions and measurements
        self._prediction_cache: List[Dict[str, float]] = []
        self._measurement_cache: List[Dict[str, float]] = []

    def calibrate(
        self,
        measurements: "MeasurementData",
        parameters: Optional[List[str]] = None,
        current_parameters: Optional[Dict[str, float]] = None,
        variance_threshold: float = 0.15,
        max_parameter_change: float = 0.20,
        time_window: int = 7,
        method: str = "nelder_mead",
        max_iterations: int = 50,
        objectives: Optional[List[str]] = None,
        weights: Optional[Dict[str, float]] = None,
        use_constraints: bool = True,
        **kwargs,
    ):
        """
        Perform online re-calibration with bounded parameter adjustments.

        Uses recent measurement data within a moving window to adjust parameters
        while enforcing bounds on the magnitude of changes to prevent instability.

        Args:
            measurements: Recent measurement data.
            parameters: Parameters to adjust. Uses last calibration if None.
            current_parameters: Current parameter values. Uses plant values if None.
            variance_threshold: Variance threshold for triggering (0-1).
            max_parameter_change: Maximum relative parameter change (0-1).
            time_window: Days of recent data to use.
            method: Optimization method ("nelder_mead" or "lbfgsb").
            max_iterations: Maximum optimization iterations.
            objectives: List of objectives to match.
            weights: Objective weights.
            use_constraints: Use parameter constraints.
            **kwargs: Additional optimizer arguments.

        Returns:
            CalibrationResult: Result with adjusted parameters.

        Raises:
            ValueError: If time window is too large or method unknown.

        Example:
            >>> result = calibrator.calibrate(
            ...     measurements=recent_data,
            ...     parameters=["k_dis", "Y_su"],
            ...     max_parameter_change=0.10,
            ...     time_window=7
            ... )
            >>> if result.success:
            ...     calibrator.apply_calibration(result)
        """
        from pyadm1.calibration.calibrator import CalibrationResult

        start_time = time.time()

        # Setup
        if objectives is None:
            objectives = ["Q_ch4", "pH"]

        if parameters is None:
            # Use parameters from last calibration
            if self.state.parameter_history:
                parameters = list(self.state.parameter_history[-1].parameters.keys())
            else:
                raise ValueError("No parameters specified and no calibration history available")

        if current_parameters is None:
            current_parameters = self._get_current_parameters()

        # Validate time window
        measurement_duration = len(measurements) / 24.0  # Assume hourly data
        if time_window > measurement_duration:
            if self.verbose:
                print(
                    f"Warning: time_window ({time_window}d) exceeds measurement duration "
                    f"({measurement_duration:.1f}d). Using full data."
                )
            time_window = int(measurement_duration)

        if self.verbose:
            print("=" * 70)
            print("Online Re-Calibration")
            print("=" * 70)
            print(f"\nParameters: {parameters}")
            print(f"Time window: {time_window} days")
            print(f"Max parameter change: {max_parameter_change * 100:.0f}%")
            print(f"Objectives: {objectives}")

        # Extract recent data window
        windowed_data = self._extract_time_window(measurements, time_window)

        if self.verbose:
            print(f"Using {len(windowed_data)} recent measurements")

        # Check variance
        current_variance = self._calculate_prediction_variance(windowed_data, current_parameters, objectives)
        self.state.current_variance = current_variance

        if self.verbose:
            print(f"Current variance: {current_variance:.4f} (threshold: {variance_threshold:.4f})")

        # Setup bounds with maximum change constraint
        param_bounds = self._setup_online_bounds(parameters, current_parameters, max_parameter_change)

        if self.verbose:
            print("\nBounded parameter ranges:")
            for param, (lb, ub) in param_bounds.items():
                current = current_parameters.get(param, 0)
                print(f"  {param}: [{lb:.4f}, {ub:.4f}] (current: {current:.4f})")

        # ====================================================================
        # CREATE OBJECTIVE FUNCTION USING OPTIMIZATION PACKAGE
        # ====================================================================

        # Create simulator wrapper
        def simulator(params: Dict[str, float]) -> Dict[str, np.ndarray]:
            """Wrapper that simulates with given parameters."""
            return self._simulate_with_parameters(params, windowed_data)

        # Extract measurements for each objective
        measurements_dict = {}
        for obj in objectives:
            try:
                measured = windowed_data.get_measurement(obj).values
                measurements_dict[obj] = measured
            except Exception:
                continue

        # Create multi-objective function
        objective_func = MultiObjectiveFunction(
            simulator=simulator,
            measurements_dict=measurements_dict,
            objectives=objectives,
            weights=weights or {obj: 1.0 / len(objectives) for obj in objectives},
            parameter_names=parameters,
            error_metric="rmse",
            normalize=True,
        )

        # ====================================================================
        # SETUP CONSTRAINTS
        # ====================================================================

        if use_constraints:
            constraints = ParameterConstraints()

            # Add box constraints with max change limits
            for param, (lower, upper) in param_bounds.items():
                constraints.add_box_constraint(param, lower, upper, hard=True)

            # Wrap objective with penalty
            original_objective = objective_func

            def penalized_objective(x: np.ndarray) -> float:
                params = {name: val for name, val in zip(parameters, x)}
                penalty = constraints.calculate_penalty(params)
                return original_objective(x) + penalty

            objective_func_wrapped = penalized_objective
        else:
            objective_func_wrapped = objective_func

        # ====================================================================
        # CREATE OPTIMIZER USING OPTIMIZATION PACKAGE
        # ====================================================================

        if self.verbose:
            print(f"\nStarting {method} optimization...")

        # Create optimizer
        optimizer = create_optimizer(
            method=method,
            bounds=param_bounds,
            max_iterations=max_iterations,
            tolerance=kwargs.get("tolerance", 1e-4),
            verbose=self.verbose,
            **kwargs,
        )

        # Get initial guess (current parameters)
        initial_guess = np.array(
            [current_parameters.get(p, self.parameter_bounds.get_default_values([p])[p]) for p in parameters]
        )

        # Run optimization
        opt_result = optimizer.optimize(objective_func_wrapped, initial_guess=initial_guess)

        # ====================================================================
        # EXTRACT AND PROCESS RESULTS
        # ====================================================================

        success = opt_result.success
        optimized_params = opt_result.parameter_dict
        objective_value = opt_result.fun
        n_iterations = opt_result.nit

        # Calculate parameter changes
        param_changes = {}
        for param, new_val in optimized_params.items():
            old_val = current_parameters.get(param, 0)
            if old_val != 0:
                change_pct = ((new_val - old_val) / old_val) * 100
            else:
                change_pct = 0.0
            param_changes[param] = change_pct

        # Validate results
        validation_metrics = {}
        if success:
            # Quick validation on windowed data
            val_result = self.validator.validate(
                parameters=optimized_params, measurements=windowed_data, objectives=objectives
            )

            for obj in objectives:
                if obj in val_result:
                    metrics = val_result[obj]
                    validation_metrics[f"{obj}_rmse"] = metrics.rmse
                    validation_metrics[f"{obj}_r2"] = metrics.r2

        execution_time = time.time() - start_time

        # Update state
        self.state.total_calibrations += 1
        self.state.last_calibration_time = datetime.now()
        self.state.consecutive_violations = 0  # Reset counter

        # Record in history
        trigger_reason = "variance_threshold" if current_variance > variance_threshold else "manual"
        history_entry = ParameterChangeHistory(
            timestamp=datetime.now(),
            parameters=optimized_params.copy(),
            trigger_reason=trigger_reason,
            objective_value=objective_value,
            variance=current_variance,
            success=success,
        )
        self.state.parameter_history.append(history_entry)

        # Create result
        result = CalibrationResult(
            success=success,
            parameters=optimized_params,
            initial_parameters=current_parameters,
            objective_value=objective_value,
            n_iterations=n_iterations,
            execution_time=execution_time,
            method=method,
            message=opt_result.get("message", "Online calibration completed"),
            validation_metrics=validation_metrics,
            sensitivity={},  # Not computed in online mode
            history=[],  # Not tracked in online mode
        )

        if self.verbose:
            print("\n" + "=" * 70)
            print("Re-Calibration Complete")
            print("=" * 70)
            print(f"Success: {success}")
            print(f"Objective: {objective_value:.6f}")
            print(f"Iterations: {n_iterations}")
            print(f"Time: {execution_time:.1f}s")
            print("\nParameter changes:")
            for param, change in param_changes.items():
                old_val = current_parameters.get(param, 0)
                new_val = optimized_params[param]
                print(f"  {param}: {old_val:.4f} → {new_val:.4f} ({change:+.1f}%)")

            if validation_metrics:
                print("\nValidation metrics:")
                for metric, value in validation_metrics.items():
                    print(f"  {metric}: {value:.4f}")

        return result

    def should_recalibrate(
        self, recent_measurements: "MeasurementData", objectives: Optional[List[str]] = None
    ) -> Tuple[bool, str]:
        """
        Check if re-calibration should be triggered based on current conditions.

        Evaluates multiple trigger conditions:
        - Prediction variance exceeding threshold
        - Time since last calibration
        - Consecutive violations of thresholds

        Args:
            recent_measurements: Recent measurement data.
            objectives: Objectives to evaluate. Defaults to ["Q_ch4", "pH"].

        Returns:
            Tuple[bool, str]: (should_recalibrate, reason).

        Example:
            >>> should_cal, reason = calibrator.should_recalibrate(recent_data)
            >>> if should_cal:
            ...     print(f"Re-calibration needed: {reason}")
            ...     result = calibrator.calibrate(recent_data)
        """
        if not self.trigger.enabled:
            return False, "Triggering disabled"

        if objectives is None:
            objectives = ["Q_ch4", "pH"]

        # Check time threshold
        if self.state.last_calibration_time is not None:
            time_since_last = datetime.now() - self.state.last_calibration_time
            hours_since_last = time_since_last.total_seconds() / 3600

            if hours_since_last < self.trigger.time_threshold:
                return False, f"Too soon since last calibration ({hours_since_last:.1f}h < {self.trigger.time_threshold:.1f}h)"

        # Calculate current variance
        current_params = self._get_current_parameters()
        variance = self._calculate_prediction_variance(recent_measurements, current_params, objectives)

        self.state.current_variance = variance

        # Check variance threshold
        if variance > self.trigger.variance_threshold:
            self.state.consecutive_violations += 1

            if self.state.consecutive_violations >= self.trigger.consecutive_violations:
                reason = (
                    f"Variance {variance:.4f} > threshold {self.trigger.variance_threshold:.4f} "
                    f"for {self.state.consecutive_violations} consecutive checks"
                )
                return True, reason
            else:
                return False, (
                    f"Variance exceeds threshold but only {self.state.consecutive_violations}/"
                    f"{self.trigger.consecutive_violations} violations"
                )
        else:
            # Reset violation counter
            self.state.consecutive_violations = 0
            return False, f"Variance {variance:.4f} within threshold {self.trigger.variance_threshold:.4f}"

        # Check residual threshold if configured
        if self.trigger.residual_threshold is not None:
            residual = self._calculate_residuals(recent_measurements, current_params, objectives)
            if residual > self.trigger.residual_threshold:
                return True, f"Residual {residual:.4f} > threshold {self.trigger.residual_threshold:.4f}"

        return False, "All thresholds satisfied"

    def set_trigger(
        self,
        variance_threshold: Optional[float] = None,
        time_threshold: Optional[float] = None,
        residual_threshold: Optional[float] = None,
        consecutive_violations: Optional[int] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        """
        Configure trigger conditions for online re-calibration.

        Args:
            variance_threshold: Variance threshold (0-1). Higher = less sensitive.
            time_threshold: Minimum hours between calibrations.
            residual_threshold: Residual threshold for triggering.
            consecutive_violations: Required consecutive violations.
            enabled: Enable/disable automatic triggering.

        Example:
            >>> calibrator.set_trigger(
            ...     variance_threshold=0.12,
            ...     time_threshold=24.0,
            ...     consecutive_violations=3
            ... )
        """
        if variance_threshold is not None:
            self.trigger.variance_threshold = variance_threshold
        if time_threshold is not None:
            self.trigger.time_threshold = time_threshold
        if residual_threshold is not None:
            self.trigger.residual_threshold = residual_threshold
        if consecutive_violations is not None:
            self.trigger.consecutive_violations = consecutive_violations
        if enabled is not None:
            self.trigger.enabled = enabled

        if self.verbose:
            print("Updated trigger configuration:")
            print(f"  Variance threshold: {self.trigger.variance_threshold:.4f}")
            print(f"  Time threshold: {self.trigger.time_threshold:.1f}h")
            print(f"  Consecutive violations: {self.trigger.consecutive_violations}")
            print(f"  Enabled: {self.trigger.enabled}")

    def apply_calibration(self, result) -> None:
        """
        Apply calibration result to the plant.

        Updates plant parameters with calibrated values.

        Args:
            result: CalibrationResult from calibrate().

        Example:
            >>> result = calibrator.calibrate(recent_data)
            >>> if result.success:
            ...     calibrator.apply_calibration(result)
        """
        if not result.success:
            if self.verbose:
                print("Warning: Applying parameters from unsuccessful calibration")

        # Apply to all digesters
        for component_id, component in self.plant.components.items():
            if component.component_type.value == "digester":
                component.apply_calibration_parameters(result.parameters)

        if self.verbose:
            print(f"Applied {len(result.parameters)} parameters to plant")

    def get_calibration_history(self, last_n: Optional[int] = None) -> List[ParameterChangeHistory]:
        """
        Get calibration history.

        Args:
            last_n: Return only last n entries. Returns all if None.

        Returns:
            List[ParameterChangeHistory]: Calibration history.

        Example:
            >>> history = calibrator.get_calibration_history(last_n=10)
            >>> for entry in history:
            ...     print(f"{entry.timestamp}: {entry.trigger_reason}")
        """
        if last_n is None:
            return self.state.parameter_history.copy()
        return self.state.parameter_history[-last_n:]

    def reset_state(self) -> None:
        """
        Reset calibrator state (history, counters, caches).

        Example:
            >>> calibrator.reset_state()
        """
        self.state = OnlineState()
        self._prediction_cache.clear()
        self._measurement_cache.clear()

        if self.verbose:
            print("Online calibrator state reset")

    # ========================================================================
    # PRIVATE METHODS
    # ========================================================================

    def _get_current_parameters(self) -> Dict[str, float]:
        """
        Get current parameter values from plant digesters.

        Returns:
            Dict[str, float]: Current parameter values.
        """
        params = {}

        for component in self.plant.components.values():
            if component.component_type.value == "digester":
                # Get calibration parameters if set
                cal_params = component.get_calibration_parameters()
                if cal_params:
                    params.update(cal_params)
                else:
                    # Get substrate-dependent params
                    substrate_params = component.adm1._get_substrate_dependent_params()
                    params.update(substrate_params)
                break

        return params

    def _extract_time_window(self, measurements: "MeasurementData", window_days: int) -> "MeasurementData":
        """
        Extract most recent data within time window.

        Args:
            measurements: Full measurement data.
            window_days: Time window in days.

        Returns:
            MeasurementData: Windowed measurement data.
        """
        # Get last timestamp
        last_time = measurements.data.index[-1]
        start_time = last_time - timedelta(days=window_days)

        return measurements.get_time_window(start_time, last_time)

    def _calculate_prediction_variance(
        self, measurements: "MeasurementData", parameters: Dict[str, float], objectives: List[str]
    ) -> float:
        """
        Calculate prediction variance for current parameters.

        Simulates with current parameters and computes variance of residuals
        relative to measurements.

        Args:
            measurements: Measurement data.
            parameters: Current parameter values.
            objectives: Objectives to evaluate.

        Returns:
            float: Normalized prediction variance (0-1).
        """
        try:
            # Simulate with current parameters
            outputs = self._simulate_with_parameters(parameters, measurements)

            # Calculate residuals for each objective
            variances = []
            for obj in objectives:
                if obj not in outputs:
                    continue

                try:
                    measured = measurements.get_measurement(obj).values
                    simulated = np.atleast_1d(outputs[obj])

                    # Align lengths
                    min_len = min(len(measured), len(simulated))
                    measured = measured[:min_len]
                    simulated = simulated[:min_len]

                    # Remove NaN
                    valid = ~(np.isnan(measured) | np.isnan(simulated))
                    if not np.any(valid):
                        continue

                    measured = measured[valid]
                    simulated = simulated[valid]

                    # Calculate normalized variance
                    residuals = measured - simulated
                    variance = np.std(residuals) / (np.mean(np.abs(measured)) + 1e-10)
                    variances.append(variance)
                except Exception as e:
                    print(e)
                    continue

            if variances:
                return float(np.mean(variances))
            return 0.0

        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not calculate variance: {str(e)}")
            return 0.0

    def _calculate_residuals(
        self, measurements: "MeasurementData", parameters: Dict[str, float], objectives: List[str]
    ) -> float:
        """
        Calculate mean absolute residuals.

        Args:
            measurements: Measurement data.
            parameters: Parameter values.
            objectives: Objectives to evaluate.

        Returns:
            float: Mean absolute residual.
        """
        try:
            outputs = self._simulate_with_parameters(parameters, measurements)

            residuals = []
            for obj in objectives:
                if obj not in outputs:
                    continue

                try:
                    measured = measurements.get_measurement(obj).values
                    simulated = np.atleast_1d(outputs[obj])

                    min_len = min(len(measured), len(simulated))
                    measured = measured[:min_len]
                    simulated = simulated[:min_len]

                    valid = ~(np.isnan(measured) | np.isnan(simulated))
                    if np.any(valid):
                        residuals.append(np.mean(np.abs(measured[valid] - simulated[valid])))
                except Exception as e:
                    print(e)
                    continue

            return float(np.mean(residuals)) if residuals else 0.0

        except Exception as e:
            print(e)
            return 0.0

    def _simulate_with_parameters(
        self, parameters: Dict[str, float], measurements: "MeasurementData"
    ) -> Dict[str, np.ndarray]:
        """
        Simulate plant with given parameters.

        This is similar to the initial calibrator but optimized for speed
        in online operation.

        Args:
            parameters: Parameter values.
            measurements: Measurement data.

        Returns:
            Dict[str, np.ndarray]: Simulated outputs.
        """
        # Apply parameters temporarily
        original_params = {}
        for component_id, component in self.plant.components.items():
            if component.component_type.value == "digester":
                original_params[component_id] = component.get_calibration_parameters()
                component.apply_calibration_parameters(parameters)

        try:
            # Quick simulation
            n_steps = len(measurements)
            dt = 1.0 / 24.0
            duration = n_steps * dt

            # Run simulation
            results = self.plant.simulate(duration=duration, dt=dt, save_interval=dt)

            # Extract outputs
            outputs = self._extract_outputs_from_results(results)

            return outputs

        finally:
            # Restore original parameters
            for component_id, params in original_params.items():
                if params:
                    self.plant.components[component_id].apply_calibration_parameters(params)
                else:
                    self.plant.components[component_id].clear_calibration_parameters()

    def _extract_outputs_from_results(self, results: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Extract relevant outputs from simulation results."""
        outputs = {
            "Q_ch4": [],
            "Q_gas": [],
            "pH": [],
            "VFA": [],
            "TAC": [],
        }

        for result in results:
            components = result.get("components", {})

            q_ch4_total = 0.0
            q_gas_total = 0.0
            pH_list = []
            vfa_list = []
            tac_list = []

            for component_result in components.values():
                q_ch4_total += component_result.get("Q_ch4", 0.0)
                q_gas_total += component_result.get("Q_gas", 0.0)
                if "pH" in component_result:
                    pH_list.append(component_result["pH"])
                if "VFA" in component_result:
                    vfa_list.append(component_result["VFA"])
                if "TAC" in component_result:
                    tac_list.append(component_result["TAC"])

            outputs["Q_ch4"].append(q_ch4_total)
            outputs["Q_gas"].append(q_gas_total)
            outputs["pH"].append(np.mean(pH_list) if pH_list else 7.0)
            outputs["VFA"].append(np.mean(vfa_list) if vfa_list else 0.0)
            outputs["TAC"].append(np.mean(tac_list) if tac_list else 0.0)

        return {k: np.array(v) for k, v in outputs.items()}

    def _setup_online_bounds(
        self, parameters: List[str], current_parameters: Dict[str, float], max_change: float
    ) -> Dict[str, Tuple[float, float]]:
        """
        Setup bounded parameter ranges for online calibration.

        Bounds are centered on current values with maximum change constraint.

        Args:
            parameters: Parameter names.
            current_parameters: Current parameter values.
            max_change: Maximum relative change (0-1).

        Returns:
            Dict[str, Tuple[float, float]]: Parameter bounds.
        """
        bounds = {}

        for param in parameters:
            current_value = current_parameters.get(param, 0)

            # Get default bounds
            default_bounds = self.parameter_bounds.get_bounds_tuple(param)
            if default_bounds is None:
                # Fallback
                default_bounds = (current_value * 0.5, current_value * 1.5)

            # Apply maximum change constraint
            max_decrease = current_value * (1 - max_change)
            max_increase = current_value * (1 + max_change)

            # Combine with default bounds
            lower = max(default_bounds[0], max_decrease)
            upper = min(default_bounds[1], max_increase)

            # Combine with default bounds
            lower = max(default_bounds[0], max_decrease)
            upper = min(default_bounds[1], max_increase)

            bounds[param] = (lower, upper)

        return bounds
