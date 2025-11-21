# ============================================================================
# pyadm1/calibration/initial.py - Complete Implementation
# ============================================================================
"""
Initial Calibration from Historical Measurement Data

This module provides the InitialCalibrator class for batch optimization of ADM1
parameters using historical plant measurement data. Supports multiple objectives
(gas production, pH, VFA) with weighted combinations, sensitivity analysis for
parameter importance, and identifiability assessment to detect over-parameterization.

The calibration uses global optimization methods (differential evolution, particle
swarm) to find parameter values that minimize the difference between simulated
and measured outputs. Local sensitivity analysis quantifies parameter influence,
and identifiability analysis detects parameters that cannot be reliably estimated
from the available data.

Example:
    >>> from pyadm1.calibration.initial import InitialCalibrator
    >>> from pyadm1.io import MeasurementData
    >>> from pyadm1.configurator import BiogasPlant
    >>>
    >>> # Load plant and measurements
    >>> plant = BiogasPlant.from_json("plant.json", feedstock)
    >>> measurements = MeasurementData.from_csv("plant_data.csv")
    >>>
    >>> # Create calibrator
    >>> calibrator = InitialCalibrator(plant, verbose=True)
    >>>
    >>> # Calibrate with multiple objectives
    >>> result = calibrator.calibrate(
    ...     measurements=measurements,
    ...     parameters=["k_dis", "k_hyd_ch", "Y_su"],
    ...     bounds={"k_dis": (0.3, 0.8), "Y_su": (0.05, 0.15)},
    ...     objectives=["Q_ch4", "pH", "VFA"],
    ...     weights={"Q_ch4": 0.6, "pH": 0.2, "VFA": 0.2},
    ...     method="differential_evolution"
    ... )
    >>>
    >>> # Analyze sensitivity
    >>> sensitivity = calibrator.sensitivity_analysis(
    ...     parameters=result.parameters,
    ...     measurements=measurements
    ... )
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
from scipy.optimize import differential_evolution, minimize
import time

from pyadm1.calibration.parameter_bounds import create_default_bounds
from pyadm1.calibration.validation import CalibrationValidator
from pyadm1.calibration.optimization import (
    create_optimizer,
    MultiObjectiveFunction,
    WeightedSumObjective,
    ParameterConstraints,
)
from pyadm1.io import MeasurementData


@dataclass
class ObjectiveWeights:
    """Weights for multi-objective calibration."""

    Q_ch4: float = 1.0
    Q_gas: float = 0.0
    pH: float = 0.0
    VFA: float = 0.0
    TAC: float = 0.0

    def normalize(self) -> "ObjectiveWeights":
        """Normalize weights to sum to 1.0."""
        total = self.Q_ch4 + self.Q_gas + self.pH + self.VFA + self.TAC
        if total > 0:
            return ObjectiveWeights(
                Q_ch4=self.Q_ch4 / total,
                Q_gas=self.Q_gas / total,
                pH=self.pH / total,
                VFA=self.VFA / total,
                TAC=self.TAC / total,
            )
        return self


@dataclass
class SensitivityResult:
    """Result from sensitivity analysis."""

    parameter: str
    base_value: float
    sensitivity_indices: Dict[str, float]
    local_gradient: Dict[str, float]
    normalized_sensitivity: Dict[str, float]
    variance_contribution: float


@dataclass
class IdentifiabilityResult:
    """Result from identifiability analysis."""

    parameter: str
    is_identifiable: bool
    confidence_interval: Tuple[float, float]
    correlation_with: Dict[str, float]
    objective_sensitivity: float
    reason: str


class InitialCalibrator:
    """
    Initial calibrator for ADM1 parameters from historical data.

    Performs batch optimization to find parameter values that minimize the
    difference between simulated and measured plant outputs. Supports
    multiple objectives, sensitivity analysis, and identifiability assessment.

    Attributes:
        plant: BiogasPlant instance to calibrate
        verbose: Enable progress output
        parameter_bounds: Parameter bounds manager
        validator: CalibrationValidator instance for result validation

    Example:
        >>> calibrator = InitialCalibrator(plant)
        >>> result = calibrator.calibrate(
        ...     measurements=data,
        ...     parameters=["k_dis", "Y_su"],
        ...     objectives=["Q_ch4", "pH"]
        ... )
    """

    def __init__(self, plant, verbose: bool = True):
        """
        Initialize initial calibrator.

        Args:
            plant: BiogasPlant instance to calibrate.
            verbose: Enable progress output.
        """
        self.plant = plant
        self.verbose = verbose
        self.parameter_bounds = create_default_bounds()
        self.validator = CalibrationValidator(plant, verbose=False)

        # Storage for optimization history
        self._optimization_history: List[Dict[str, Any]] = []
        self._best_objective_value = float("inf")

        # Store original parameters for restoration
        self._original_parameters = self._get_current_parameters()

    def calibrate(
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
        use_constraints: bool = False,
        **kwargs,
    ):
        """
        Calibrate model parameters from historical measurements.

        Args:
            measurements: Historical measurement data with columns for substrates and outputs.
            parameters: List of parameter names to calibrate (e.g., ["k_dis", "Y_su"]).
            bounds: Parameter bounds as {param: (min, max)}. Uses defaults if None.
            method: Optimization method ("differential_evolution", "nelder_mead", "lbfgsb").
            objectives: List of outputs to match (e.g., ["Q_ch4", "pH", "VFA"]).
            weights: Objective weights as {output: weight}. Equal weights if None.
            validation_split: Fraction of data reserved for validation (0-1).
            max_iterations: Maximum optimization iterations.
            population_size: Population size for differential evolution.
            tolerance: Convergence tolerance.
            sensitivity_analysis: Perform sensitivity analysis on results.
            use_constraints: Add parameter constraints beyond bounds.
            **kwargs: Additional optimizer arguments.

        Returns:
            CalibrationResult: Object containing optimized parameters, metrics, and history.

        Raises:
            ValueError: If method is unknown or parameters/objectives are invalid.

        Example:
            >>> result = calibrator.calibrate(
            ...     measurements=data,
            ...     parameters=["k_dis", "k_hyd_ch", "Y_su"],
            ...     objectives=["Q_ch4", "pH"],
            ...     weights={"Q_ch4": 0.8, "pH": 0.2},
            ...     method="differential_evolution"
            ... )
            >>> print(f"Success: {result.success}, Best obj: {result.objective_value:.4f}")
        """
        from pyadm1.calibration.calibrator import CalibrationResult

        start_time = time.time()

        # Setup objectives and weights
        if objectives is None:
            objectives = ["Q_ch4"]

        objective_weights = self._setup_objective_weights(objectives, weights)

        # Split data
        train_data, val_data = self._split_data(measurements, validation_split)

        if self.verbose:
            print(f"\nTraining samples: {len(train_data)}")
            print(f"Validation samples: {len(val_data)}")
            print(f"\nOptimizing parameters: {parameters}")
            print(f"Objectives: {objectives}")
            print(f"Weights: {objective_weights}")

        # Get initial parameter values
        initial_params = self._get_initial_parameters(parameters)

        # Setup bounds
        param_bounds = self._setup_bounds(parameters, bounds)

        if self.verbose:
            print("\nParameter bounds:")
            for param, (lb, ub) in param_bounds.items():
                init_val = initial_params[param]
                print(f"  {param}: [{lb:.4f}, {ub:.4f}] (initial: {init_val:.4f})")

        # Reset optimization history
        self._optimization_history = []
        self._best_objective_value = float("inf")

        # ====================================================================
        # CREATE OBJECTIVE FUNCTION USING OPTIMIZATION PACKAGE
        # ====================================================================

        # Create simulator wrapper
        def simulator(params: Dict[str, float]) -> Dict[str, np.ndarray]:
            """Wrapper that simulates with given parameters."""
            return self._simulate_with_parameters(params, train_data)

        # Extract measurements for each objective
        measurements_dict = {}
        for obj in objectives:
            try:
                measured = train_data.get_measurement(obj).values
                measurements_dict[obj] = measured
            except Exception:
                continue

        # Create multi-objective function
        if weights is None:
            # Use WeightedSumObjective with equal weights
            objective_func = WeightedSumObjective(
                simulator=simulator,
                measurements_dict=measurements_dict,
                objectives=objectives,
                parameter_names=parameters,
                error_metric="rmse",
                normalize=True,
            )
        else:
            # Use MultiObjectiveFunction with custom weights
            objective_func = MultiObjectiveFunction(
                simulator=simulator,
                measurements_dict=measurements_dict,
                objectives=objectives,
                weights=weights,
                parameter_names=parameters,
                error_metric="rmse",
                normalize=True,
            )

        # ====================================================================
        # SETUP CONSTRAINTS (OPTIONAL)
        # ====================================================================

        constraints = None
        if use_constraints:
            constraints = ParameterConstraints()

            # Add box constraints
            for param, (lower, upper) in param_bounds.items():
                constraints.add_box_constraint(param, lower, upper, hard=True)

            # Add example linear constraint: Y_su + 0.1*k_dis <= 0.2
            # (Only if both parameters are being calibrated)
            if "Y_su" in parameters and "k_dis" in parameters:
                constraints.add_linear_inequality(
                    coefficients={"Y_su": 1.0, "k_dis": 0.1},
                    upper_bound=0.2,
                    weight=1.0,
                )

            # Wrap objective with penalty
            original_objective = objective_func

            def penalized_objective(x: np.ndarray) -> float:
                params = {name: val for name, val in zip(parameters, x)}
                penalty = constraints.calculate_penalty(params)
                return original_objective(x) + penalty

            # Use penalized objective
            objective_func_wrapped = penalized_objective
        else:
            objective_func_wrapped = objective_func

        # ====================================================================
        # CREATE OPTIMIZER USING OPTIMIZATION PACKAGE
        # ====================================================================

        if self.verbose:
            print(f"\nStarting {method} optimization...")
            print(f"Max iterations: {max_iterations}")

        # Create optimizer via factory
        optimizer = create_optimizer(
            method=method,
            bounds=param_bounds,
            max_iterations=max_iterations,
            tolerance=tolerance,
            verbose=self.verbose,
            population_size=population_size if method in ["differential_evolution", "de"] else None,
            **kwargs,
        )

        # Get initial guess for local methods
        initial_guess = None
        if method in ["nelder_mead", "nm", "lbfgsb", "l_bfgs_b", "powell"]:
            initial_guess = np.array([initial_params[p] for p in parameters])

        # Run optimization
        opt_result = optimizer.optimize(objective_func_wrapped, initial_guess=initial_guess)

        # ====================================================================
        # EXTRACT AND PROCESS RESULTS
        # ====================================================================

        success = opt_result.success
        optimized_params = opt_result.parameter_dict
        objective_value = opt_result.fun
        n_iterations = opt_result.nit

        # Validate on validation set
        validation_metrics = {}
        if len(val_data) > 0:
            val_result = self.validator.validate(parameters=optimized_params, measurements=val_data, objectives=objectives)

            # Convert ValidationMetrics to dict format
            for obj in objectives:
                # The validator returns a dict mapping objectives to ValidationMetrics
                if obj in val_result:
                    metrics = val_result[obj]
                    validation_metrics[f"{obj}_rmse"] = metrics.rmse
                    validation_metrics[f"{obj}_r2"] = metrics.r2
                    validation_metrics[f"{obj}_nse"] = metrics.nse

        # Sensitivity analysis
        sensitivity_results = {}
        if sensitivity_analysis and success:
            if self.verbose:
                print("\nPerforming sensitivity analysis...")

            sensitivity_results = self._perform_sensitivity_analysis(
                parameters=optimized_params, measurements=train_data, objectives=objectives
            )

        execution_time = time.time() - start_time

        # Restore original parameters
        self._apply_parameters_to_plant(self._original_parameters)

        # Store optimization history from optimizer
        self._optimization_history = opt_result.history

        # Create result object
        result = CalibrationResult(
            success=success,
            parameters=optimized_params,
            initial_parameters=initial_params,
            objective_value=objective_value,
            n_iterations=n_iterations,
            execution_time=execution_time,
            method=method,
            message=opt_result.message if hasattr(opt_result, "message") else "Optimization completed",
            validation_metrics=validation_metrics,
            sensitivity=sensitivity_results,
            history=self._optimization_history,
        )

        return result

    def sensitivity_analysis(
        self,
        parameters: Dict[str, float],
        measurements: "MeasurementData",
        objectives: Optional[List[str]] = None,
        perturbation: float = 0.01,
    ) -> Dict[str, SensitivityResult]:
        """
        Perform local sensitivity analysis at given parameter values.

        Computes local gradients by finite differences and calculates
        normalized sensitivity indices to identify influential parameters.

        Args:
            parameters: Parameter values for analysis as {param: value}.
            measurements: Measurement data for simulation.
            objectives: List of outputs to analyze. Defaults to ["Q_ch4"].
            perturbation: Relative perturbation for finite differences (0-1).

        Returns:
            Dict[str, SensitivityResult]: Mapping parameter names to sensitivity results.

        Example:
            >>> sensitivity = calibrator.sensitivity_analysis(
            ...     parameters={"k_dis": 0.5, "Y_su": 0.1},
            ...     measurements=data
            ... )
            >>> for param, result in sensitivity.items():
            ...     print(f"{param}: sensitivity = {result.sensitivity_indices}")
        """
        if objectives is None:
            objectives = ["Q_ch4"]

        if self.verbose:
            print(f"\nSensitivity analysis for {len(parameters)} parameters")
            print(f"Objectives: {objectives}")
            print(f"Perturbation: {perturbation * 100:.1f}%")

        results = {}

        for param_name, base_value in parameters.items():
            # Calculate perturbed values
            delta = base_value * perturbation
            if delta == 0:
                delta = perturbation

            param_plus = parameters.copy()
            param_plus[param_name] = base_value + delta

            param_minus = parameters.copy()
            param_minus[param_name] = base_value - delta

            # Simulate with perturbed parameters
            outputs_base = self._simulate_with_parameters(parameters, measurements)
            outputs_plus = self._simulate_with_parameters(param_plus, measurements)
            outputs_minus = self._simulate_with_parameters(param_minus, measurements)

            # Calculate gradients
            local_gradient = {}
            sensitivity_indices = {}
            normalized_sensitivity = {}

            for obj in objectives:
                if obj in outputs_base and obj in outputs_plus and obj in outputs_minus:
                    # Extract mean values
                    base_val = np.mean(outputs_base[obj])
                    plus_val = np.mean(outputs_plus[obj])
                    minus_val = np.mean(outputs_minus[obj])

                    # Local gradient (finite difference)
                    gradient = (plus_val - minus_val) / (2 * delta)
                    local_gradient[obj] = gradient

                    # Sensitivity index (relative)
                    if base_val != 0:
                        sens_idx = gradient * (base_value / base_val)
                        sensitivity_indices[obj] = sens_idx
                    else:
                        sensitivity_indices[obj] = 0.0

                    # Normalized sensitivity
                    base_std = np.std(outputs_base[obj])
                    if base_std > 0:
                        norm_sens = abs(gradient * delta / base_std)
                        normalized_sensitivity[obj] = norm_sens
                    else:
                        normalized_sensitivity[obj] = 0.0

            # Variance contribution (sum of squared sensitivities)
            variance_contrib = sum(s**2 for s in sensitivity_indices.values())

            results[param_name] = SensitivityResult(
                parameter=param_name,
                base_value=base_value,
                sensitivity_indices=sensitivity_indices,
                local_gradient=local_gradient,
                normalized_sensitivity=normalized_sensitivity,
                variance_contribution=variance_contrib,
            )

            if self.verbose:
                print(f"\n{param_name} (value: {base_value:.4f}):")
                for obj, sens in sensitivity_indices.items():
                    print(f"  {obj}: {sens:.4e}")

        return results

    def identifiability_analysis(
        self,
        parameters: Dict[str, float],
        measurements: "MeasurementData",
        confidence_level: float = 0.95,
        correlation_threshold: float = 0.8,
    ) -> Dict[str, IdentifiabilityResult]:
        """
        Assess parameter identifiability from available measurements.

        Identifies parameters that cannot be reliably estimated due to
        insufficient data, low sensitivity, or high correlation with other parameters.

        Args:
            parameters: Parameter values to analyze as {param: value}.
            measurements: Measurement data for analysis.
            confidence_level: Confidence level for intervals (0-1).
            correlation_threshold: Threshold for high correlation (0-1).

        Returns:
            Dict[str, IdentifiabilityResult]: Mapping parameter names to identifiability results.

        Example:
            >>> identifiability = calibrator.identifiability_analysis(
            ...     parameters={"k_dis": 0.5, "k_hyd_ch": 10.0, "Y_su": 0.1},
            ...     measurements=data
            ... )
            >>> for param, result in identifiability.items():
            ...     if not result.is_identifiable:
            ...         print(f"{param}: {result.reason}")
        """
        if self.verbose:
            print("\nIdentifiability analysis:")
            print(f"Parameters: {list(parameters.keys())}")
            print(f"Confidence level: {confidence_level * 100:.0f}%")

        # Perform sensitivity analysis
        sensitivity = self.sensitivity_analysis(parameters, measurements, objectives=["Q_ch4", "pH", "VFA"])

        # Calculate correlation matrix from optimization history
        correlation_matrix = self._calculate_parameter_correlation_from_history()

        results = {}

        for param_name, param_value in parameters.items():
            # Check sensitivity
            if param_name in sensitivity:
                sens_result = sensitivity[param_name]
                max_sensitivity = max(abs(s) for s in sens_result.sensitivity_indices.values())
            else:
                max_sensitivity = 0.0

            # Check correlations
            correlations = {}
            max_correlation = 0.0
            highly_correlated_with = None

            if correlation_matrix and param_name in correlation_matrix:
                for other_param, corr_value in correlation_matrix[param_name].items():
                    if other_param != param_name:
                        correlations[other_param] = corr_value
                        if abs(corr_value) > max_correlation:
                            max_correlation = abs(corr_value)
                            highly_correlated_with = other_param

            # Determine identifiability
            is_identifiable = True
            reason = "Parameter is identifiable"

            if max_sensitivity < 1e-6:
                is_identifiable = False
                reason = f"Very low sensitivity (max: {max_sensitivity:.2e})"
            elif max_correlation > correlation_threshold:
                is_identifiable = False
                reason = f"High correlation with {highly_correlated_with} ({max_correlation:.3f})"

            # Estimate confidence interval
            if is_identifiable:
                uncertainty = 0.1 * param_value / max(max_sensitivity, 1e-6)
                ci_lower = max(0, param_value - uncertainty)
                ci_upper = param_value + uncertainty
            else:
                ci_lower = 0
                ci_upper = param_value * 10

            results[param_name] = IdentifiabilityResult(
                parameter=param_name,
                is_identifiable=is_identifiable,
                confidence_interval=(ci_lower, ci_upper),
                correlation_with=correlations,
                objective_sensitivity=max_sensitivity,
                reason=reason,
            )

            if self.verbose:
                status = "✓" if is_identifiable else "✗"
                print(f"  {status} {param_name}: {reason}")
                print(f"    CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

        return results

    # ========================================================================
    # PRIVATE METHODS
    # ========================================================================

    def _objective_function(
        self,
        param_values: List[float],
        param_names: List[str],
        measurements: "MeasurementData",
        objectives: List[str],
        weights: ObjectiveWeights,
    ) -> float:
        """
        Objective function for optimization.

        Computes weighted sum of squared errors between simulated and measured outputs.

        Args:
            param_values: Parameter values to evaluate.
            param_names: Parameter names corresponding to param_values.
            measurements: Measurement data with expected outputs.
            objectives: List of objective names to evaluate.
            weights: Objective weights for multi-objective optimization.

        Returns:
            float: Objective function value (lower is better).
        """
        # Create parameter dictionary
        parameters = {name: value for name, value in zip(param_names, param_values)}

        # Simulate with these parameters
        try:
            outputs = self._simulate_with_parameters(parameters, measurements)
        except Exception as e:
            if self.verbose:
                print(f"Simulation failed: {str(e)}")
            return 1e10

        # Calculate weighted objective
        objective = 0.0
        n_objectives = 0

        for obj in objectives:
            if obj not in outputs:
                continue

            # Get measured values
            try:
                measured = measurements.get_measurement(obj).values
            except Exception as e:
                print(e)
                continue

            simulated = outputs[obj]

            # Ensure arrays
            measured = np.atleast_1d(measured)
            simulated = np.atleast_1d(simulated)

            # Handle length mismatch
            min_len = min(len(measured), len(simulated))
            measured = measured[:min_len]
            simulated = simulated[:min_len]

            # Remove NaN values
            valid = ~(np.isnan(measured) | np.isnan(simulated))
            if not np.any(valid):
                continue

            measured = measured[valid]
            simulated = simulated[valid]

            # Calculate MSE
            mse = np.mean((measured - simulated) ** 2)

            # Get weight
            weight = getattr(weights, obj, 0.0)

            objective += weight * mse
            n_objectives += 1

        if n_objectives == 0:
            return 1e10

        # Normalize by number of objectives
        objective = objective / n_objectives

        # Track best
        if objective < self._best_objective_value:
            self._best_objective_value = objective
            if self.verbose:
                param_str = ", ".join([f"{p}={v:.4f}" for p, v in zip(param_names, param_values)])
                print(f"  New best: {objective:.6f} | {param_str}")

        # Store in history
        self._optimization_history.append({"parameters": parameters.copy(), "objective": objective})

        return objective

    def _simulate_with_parameters(
        self, parameters: Dict[str, float], measurements: "MeasurementData"
    ) -> Dict[str, np.ndarray]:
        """
        Simulate plant with given parameters and extract outputs.

        This method:
        1. Applies parameters to all digester components in the plant
        2. Extracts substrate feed rates from measurements
        3. Runs plant simulation for the measurement duration
        4. Extracts and returns relevant outputs

        Args:
            parameters: Parameter values to apply as {param: value}.
            measurements: Measurement data containing substrate feeds and duration.

        Returns:
            Dict[str, np.ndarray]: Simulated outputs as {output_name: array}.

        Raises:
            ValueError: If no digesters found in plant or substrate feeds missing.
            RuntimeError: If simulation fails.
        """
        # Apply parameters to plant
        self._apply_parameters_to_plant(parameters)

        # Determine simulation settings
        n_steps = len(measurements)
        dt = 1.0 / 24.0  # 1 hour timestep
        duration = n_steps * dt

        # Extract substrate feeds from measurements
        Q_substrates = self._extract_substrate_feeds(measurements)

        # Apply substrate feeds to digesters
        for component_id, component in self.plant.components.items():
            if component.component_type.value == "digester":
                # Update substrate feeds for this digester
                component.Q_substrates = Q_substrates
                component.adm1.create_influent(Q_substrates, 0)

        # Run simulation
        try:
            results = self.plant.simulate(duration=duration, dt=dt, save_interval=dt)
        except Exception as e:
            raise RuntimeError(f"Simulation failed: {str(e)}")

        # Extract outputs
        outputs = self._extract_outputs_from_results(results)

        return outputs

    def _get_current_parameters(self) -> Dict[str, float]:
        """
        Get current parameter values from all plant digesters.

        Returns:
            Dict[str, float]: Current parameter values as {param: value}.
        """
        params = {}

        # Get parameters from first digester
        for component in self.plant.components.values():
            if component.component_type.value == "digester":
                # Get substrate-dependent params
                substrate_params = component.adm1._get_substrate_dependent_params()
                params.update(substrate_params)
                break  # Use first digester's parameters

        return params

    def _apply_parameters_to_plant(self, parameters: Dict[str, float]) -> None:
        """
        Apply parameter values to all digester components in the plant.

        Substrate-dependent parameters (k_dis, k_hyd_*, k_m_*) are applied
        by updating the substrate properties, which are then used during
        simulation to calculate these parameters dynamically.

        Args:
            parameters: Parameter values to apply as {param: value}.

        Raises:
            ValueError: If no digesters found in plant.
        """
        digester_count = 0

        for component_id, component in self.plant.components.items():
            if component.component_type.value == "digester":
                digester_count += 1

                # Access the substrates object from feedstock
                # substrates = component.feedstock.mySubstrates()

                # Apply substrate-dependent parameters
                # These parameters affect the substrate characterization
                for param_name, param_value in parameters.items():
                    if param_name in ["k_dis", "k_hyd_ch", "k_hyd_pr", "k_hyd_li", "k_m_c4", "k_m_pro", "k_m_ac", "k_m_h2"]:
                        # Store parameter for later use in get_substrate_dependent_params
                        if not hasattr(component, "_calibration_params"):
                            component._calibration_params = {}
                        component._calibration_params[param_name] = param_value

                # For yield and other ADM1 parameters, we need to modify them
                # in the parameter calculation. Since these are calculated each time
                # in ADM1_ODE from ADMParams, we store them for retrieval
                for param_name, param_value in parameters.items():
                    if param_name.startswith("Y_") or param_name.startswith("K_"):
                        if not hasattr(component, "_calibration_params"):
                            component._calibration_params = {}
                        component._calibration_params[param_name] = param_value

        if digester_count == 0:
            raise ValueError("No digesters found in plant to apply parameters")

    def _extract_substrate_feeds(self, measurements: "MeasurementData") -> List[float]:
        """
        Extract substrate feed rates from measurements.

        Looks for columns with names like 'Q_sub1', 'Q_sub2', etc., or
        uses a default substrate mix if not found.

        Args:
            measurements: Measurement data potentially containing substrate feeds.

        Returns:
            List[float]: Substrate feed rates in m³/d for each substrate.
        """
        try:
            # Try to get substrate feeds from measurements
            Q = measurements.get_substrate_feeds()
            # Use mean values over the measurement period
            return list(np.mean(Q, axis=0))
        except Exception:
            # Default substrate mix if not found in measurements
            # Assumes 2 main substrates (corn silage + cattle manure)
            return [15.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def _extract_outputs_from_results(self, results: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """
        Extract relevant outputs from simulation results.

        Args:
            results: List of simulation results from plant.simulate().

        Returns:
            Dict[str, np.ndarray]: Extracted outputs as {output_name: array}.
        """
        outputs = {
            "Q_ch4": [],
            "Q_gas": [],
            "pH": [],
            "VFA": [],
            "TAC": [],
            "Q_co2": [],
        }

        for result in results:
            components = result.get("components", {})

            # Sum outputs from all digesters
            q_ch4_total = 0.0
            q_gas_total = 0.0
            q_co2_total = 0.0
            pH_list = []
            vfa_list = []
            tac_list = []

            for component_result in components.values():
                q_ch4_total += component_result.get("Q_ch4", 0.0)
                q_gas_total += component_result.get("Q_gas", 0.0)
                q_co2_total += component_result.get("Q_co2", 0.0)
                if "pH" in component_result:
                    pH_list.append(component_result["pH"])
                if "VFA" in component_result:
                    vfa_list.append(component_result["VFA"])
                if "TAC" in component_result:
                    tac_list.append(component_result["TAC"])

            outputs["Q_ch4"].append(q_ch4_total)
            outputs["Q_gas"].append(q_gas_total)
            outputs["Q_co2"].append(q_co2_total)
            outputs["pH"].append(np.mean(pH_list) if pH_list else 7.0)
            outputs["VFA"].append(np.mean(vfa_list) if vfa_list else 0.0)
            outputs["TAC"].append(np.mean(tac_list) if tac_list else 0.0)

        # Convert to numpy arrays
        return {k: np.array(v) for k, v in outputs.items()}

    def _setup_objective_weights(self, objectives: List[str], weights: Optional[Dict[str, float]]) -> ObjectiveWeights:
        """
        Setup and normalize objective weights.

        Args:
            objectives: List of objective names.
            weights: Optional weights as {objective: weight}.

        Returns:
            ObjectiveWeights: Normalized objective weights.
        """
        if weights is None:
            # Equal weights
            weight_value = 1.0 / len(objectives)
            weights = {obj: weight_value for obj in objectives}

        obj_weights = ObjectiveWeights(
            Q_ch4=weights.get("Q_ch4", 0.0),
            Q_gas=weights.get("Q_gas", 0.0),
            pH=weights.get("pH", 0.0),
            VFA=weights.get("VFA", 0.0),
            TAC=weights.get("TAC", 0.0),
        )

        return obj_weights.normalize()

    def _split_data(self, measurements: "MeasurementData", split_ratio: float) -> Tuple["MeasurementData", "MeasurementData"]:
        """
        Split measurements into training and validation sets.

        Uses time-based splitting (chronological order).

        Args:
            measurements: Full measurement dataset.
            split_ratio: Fraction of data for validation (0-1).

        Returns:
            Tuple[MeasurementData, MeasurementData]: (train_data, validation_data).
        """
        n_total = len(measurements)
        n_train = int(n_total * (1 - split_ratio))

        # Time-based split
        train_data = MeasurementData(measurements.data.iloc[:n_train].copy(), metadata=measurements.metadata.copy())

        val_data = MeasurementData(measurements.data.iloc[n_train:].copy(), metadata=measurements.metadata.copy())

        return train_data, val_data

    def _get_initial_parameters(self, parameters: List[str]) -> Dict[str, float]:
        """
        Get initial parameter values from bounds manager.

        Args:
            parameters: List of parameter names.

        Returns:
            Dict[str, float]: Initial parameter values.
        """
        return self.parameter_bounds.get_default_values(parameters)

    def _setup_bounds(
        self, parameters: List[str], custom_bounds: Optional[Dict[str, Tuple[float, float]]]
    ) -> Dict[str, Tuple[float, float]]:
        """
        Setup parameter bounds for optimization.

        Args:
            parameters: List of parameter names.
            custom_bounds: Optional custom bounds as {param: (min, max)}.

        Returns:
            Dict[str, Tuple[float, float]]: Parameter bounds.
        """
        bounds = {}

        for param in parameters:
            if custom_bounds and param in custom_bounds:
                bounds[param] = custom_bounds[param]
            else:
                # Get default bounds from manager
                bounds_tuple = self.parameter_bounds.get_bounds_tuple(param)
                if bounds_tuple:
                    bounds[param] = bounds_tuple
                else:
                    # Fallback: ±50% of default value
                    default = self.parameter_bounds.get_default_values([param])[param]
                    bounds[param] = (default * 0.5, default * 1.5)

        return bounds

    def _optimize_differential_evolution(
        self,
        objective_func: Callable,
        bounds: Dict[str, Tuple[float, float]],
        parameters: List[str],
        max_iterations: int,
        population_size: int,
        tolerance: float,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run differential evolution optimization.

        Args:
            objective_func: Objective function to minimize.
            bounds: Parameter bounds as {param: (min, max)}.
            parameters: Parameter names in order.
            max_iterations: Maximum iterations.
            population_size: Population size.
            tolerance: Convergence tolerance.
            **kwargs: Additional scipy arguments.

        Returns:
            Dict[str, Any]: Optimization result with keys 'success', 'x', 'fun', 'nit', 'message'.
        """
        # Convert bounds to scipy format
        bounds_list = [bounds[param] for param in parameters]

        # Run optimization
        result = differential_evolution(
            func=objective_func,
            bounds=bounds_list,
            maxiter=max_iterations,
            popsize=population_size,
            tol=tolerance,
            disp=self.verbose,
            **kwargs,
        )

        return {
            "success": result.success,
            "x": result.x,
            "fun": result.fun,
            "nit": result.nit,
            "message": result.message,
        }

    def _optimize_nelder_mead(
        self,
        objective_func: Callable,
        initial_guess: List[float],
        bounds: Dict[str, Tuple[float, float]],
        parameters: List[str],
        max_iterations: int,
        tolerance: float,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run Nelder-Mead optimization.

        Args:
            objective_func: Objective function to minimize.
            initial_guess: Initial parameter values.
            bounds: Parameter bounds (not strictly enforced by Nelder-Mead).
            parameters: Parameter names.
            max_iterations: Maximum iterations.
            tolerance: Convergence tolerance.
            **kwargs: Additional scipy arguments.

        Returns:
            Dict[str, Any]: Optimization result.
        """
        result = minimize(
            fun=objective_func,
            x0=initial_guess,
            method="Nelder-Mead",
            options={"maxiter": max_iterations, "xatol": tolerance, "disp": self.verbose},
            **kwargs,
        )

        return {
            "success": result.success,
            "x": result.x,
            "fun": result.fun,
            "nit": result.nit,
            "message": result.message,
        }

    def _optimize_lbfgsb(
        self,
        objective_func: Callable,
        initial_guess: List[float],
        bounds: Dict[str, Tuple[float, float]],
        parameters: List[str],
        max_iterations: int,
        tolerance: float,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run L-BFGS-B optimization.

        Args:
            objective_func: Objective function to minimize.
            initial_guess: Initial parameter values.
            bounds: Parameter bounds.
            parameters: Parameter names in order.
            max_iterations: Maximum iterations.
            tolerance: Convergence tolerance.
            **kwargs: Additional scipy arguments.

        Returns:
            Dict[str, Any]: Optimization result.
        """
        bounds_list = [bounds[param] for param in parameters]

        result = minimize(
            fun=objective_func,
            x0=initial_guess,
            method="L-BFGS-B",
            bounds=bounds_list,
            options={"maxiter": max_iterations, "ftol": tolerance, "disp": self.verbose},
            **kwargs,
        )

        return {
            "success": result.success,
            "x": result.x,
            "fun": result.fun,
            "nit": result.nit,
            "message": result.message,
        }

    def _perform_sensitivity_analysis(
        self, parameters: Dict[str, float], measurements: "MeasurementData", objectives: List[str]
    ) -> Dict[str, float]:
        """
        Perform sensitivity analysis and return simplified metrics.

        Args:
            parameters: Parameter values for analysis.
            measurements: Measurement data.
            objectives: List of objectives.

        Returns:
            Dict[str, float]: Simplified sensitivity metrics as {param: max_sensitivity}.
        """
        sensitivity = self.sensitivity_analysis(parameters, measurements, objectives)

        # Return simplified sensitivity metrics
        results = {}
        for param_name, sens_result in sensitivity.items():
            # Max absolute sensitivity across objectives
            max_sens = max(abs(s) for s in sens_result.sensitivity_indices.values())
            results[param_name] = max_sens

        return results

    def _calculate_parameter_correlation_from_history(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate correlation matrix from optimization history.

        Uses the stored optimization history to compute correlations between parameters.

        Returns:
            Dict[str, Dict[str, float]]: Correlation matrix as nested dict.
        """
        if not self._optimization_history or len(self._optimization_history) < 2:
            return {}

        # Extract parameter names from first history entry
        param_names = list(self._optimization_history[0]["parameters"].keys())
        n_samples = len(self._optimization_history)

        # Build parameter matrix
        param_matrix = np.zeros((n_samples, len(param_names)))
        for i, entry in enumerate(self._optimization_history):
            for j, name in enumerate(param_names):
                param_matrix[i, j] = entry["parameters"][name]

        # Calculate correlation matrix
        try:
            corr_matrix = np.corrcoef(param_matrix.T)
        except Exception as e:
            print(e)
            return {}

        # Convert to dictionary format
        correlation = {}
        for i, param1 in enumerate(param_names):
            correlation[param1] = {}
            for j, param2 in enumerate(param_names):
                correlation[param1][param2] = float(corr_matrix[i, j])

        return correlation


# ============================================================================
# Example Usage
# ============================================================================
"""
Complete example showing calibration workflow:

>>> from pyadm1.configurator import BiogasPlant
>>> from pyadm1.substrates import Feedstock
>>> from pyadm1.calibration import InitialCalibrator
>>> from pyadm1.io import MeasurementData
>>>
>>> # 1. Load or create plant
>>> feedstock = Feedstock(feeding_freq=48)
>>> plant = BiogasPlant("Calibration Test Plant")
>>>
>>> from pyadm1.components.biological import Digester
>>> digester = Digester("main_dig", feedstock, V_liq=2000, V_gas=300)
>>> plant.add_component(digester)
>>> plant.initialize()
>>>
>>> # 2. Load measurement data
>>> measurements = MeasurementData.from_csv(
>>>     "plant_measurements.csv",
>>>     timestamp_column="time",
>>>     resample="1H"
>>> )
>>>
>>> # 3. Validate data
>>> validation = measurements.validate()
>>> if not validation.is_valid:
>>>     print("Data quality issues found:")
>>>     validation.print_report()
>>>
>>> # 4. Create calibrator
>>> calibrator = InitialCalibrator(plant, verbose=True)
>>>
>>> # 5. Run calibration
>>> result = calibrator.calibrate(
>>>     measurements=measurements,
>>>     parameters=["k_dis", "k_hyd_ch", "Y_su"],
>>>     bounds={
>>>         "k_dis": (0.3, 0.8),
>>>         "Y_su": (0.05, 0.15)
>>>     },
>>>     objectives=["Q_ch4", "pH"],
>>>     weights={"Q_ch4": 0.8, "pH": 0.2},
>>>     method="differential_evolution",
>>>     validation_split=0.2,
>>>     max_iterations=100
>>> )
>>>
>>> # 6. Check results
>>> if result.success:
>>>     print(f"Calibration successful!")
>>>     print(f"Objective value: {result.objective_value:.4f}")
>>>     print(f"Optimized parameters:")
>>>     for param, value in result.parameters.items():
>>>         print(f"  {param}: {value:.4f}")
>>>
>>>     # Validation metrics
>>>     print(f"\nValidation metrics:")
>>>     for metric, value in result.validation_metrics.items():
>>>         print(f"  {metric}: {value:.4f}")
>>> else:
>>>     print(f"Calibration failed: {result.message}")
>>>
>>> # 7. Apply calibrated parameters
>>> calibrator.plant.components["main_dig"].apply_calibration_parameters(
>>>     result.parameters
>>> )
>>>
>>> # 8. Perform sensitivity analysis
>>> sensitivity = calibrator.sensitivity_analysis(
>>>     parameters=result.parameters,
>>>     measurements=measurements
>>> )
>>>
>>> print("\nParameter sensitivities:")
>>> for param, sens_result in sensitivity.items():
>>>     print(f"\n{param}:")
>>>     for obj, sens in sens_result.sensitivity_indices.items():
>>>         print(f"  {obj}: {sens:.4e}")
>>>
>>> # 9. Check identifiability
>>> identifiability = calibrator.identifiability_analysis(
>>>     parameters=result.parameters,
>>>     measurements=measurements
>>> )
>>>
>>> print("\nParameter identifiability:")
>>> for param, ident_result in identifiability.items():
>>>     status = "✓" if ident_result.is_identifiable else "✗"
>>>     print(f"{status} {param}: {ident_result.reason}")
>>>
>>> # 10. Save results
>>> result.to_json("calibration_result.json")
"""
