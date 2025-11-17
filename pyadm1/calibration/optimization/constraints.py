# pyadm1/calibration/optimization/constraints.py
"""
Constraint Handling for Parameter Optimization

This module provides classes for handling various types of constraints during
parameter optimization, including:
- Box constraints (bounds)
- Linear equality and inequality constraints
- Nonlinear constraints
- Penalty functions for soft constraints

Example:
    >>> from pyadm1.calibration.optimization import ParameterConstraints
    >>>
    >>> # Create constraints
    >>> constraints = ParameterConstraints()
    >>>
    >>> # Add box constraints
    >>> constraints.add_box_constraint("k_dis", 0.3, 0.8)
    >>> constraints.add_box_constraint("Y_su", 0.05, 0.15)
    >>>
    >>> # Add linear constraint: k_dis + Y_su <= 1.0
    >>> constraints.add_linear_inequality(
    ...     coefficients={"k_dis": 1.0, "Y_su": 1.0},
    ...     upper_bound=1.0
    ... )
    >>>
    >>> # Check validity
    >>> params = {"k_dis": 0.5, "Y_su": 0.1}
    >>> is_valid = constraints.is_feasible(params)
    >>> penalty = constraints.calculate_penalty(params)
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
from dataclasses import dataclass


@dataclass
class BoxConstraint:
    """
    Box constraint (bounds) for a single parameter.

    Attributes:
        parameter_name: Name of parameter
        lower: Lower bound
        upper: Upper bound
        hard: If True, violations are not allowed (infinite penalty)
    """

    parameter_name: str
    lower: float
    upper: float
    hard: bool = True

    def is_feasible(self, value: float) -> bool:
        """Check if value satisfies constraint."""
        return self.lower <= value <= self.upper

    def project(self, value: float) -> float:
        """Project value to feasible region."""
        return np.clip(value, self.lower, self.upper)

    def violation(self, value: float) -> float:
        """Calculate constraint violation amount."""
        if value < self.lower:
            return self.lower - value
        elif value > self.upper:
            return value - self.upper
        return 0.0


@dataclass
class LinearConstraint:
    """
    Linear constraint: sum(coefficients[i] * x[i]) <= upper_bound
                      or sum(coefficients[i] * x[i]) >= lower_bound
                      or sum(coefficients[i] * x[i]) == target

    Attributes:
        coefficients: Dictionary mapping parameter names to coefficients
        lower_bound: Lower bound (None = -inf)
        upper_bound: Upper bound (None = +inf)
        constraint_type: "inequality" or "equality"
    """

    coefficients: Dict[str, float]
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    constraint_type: str = "inequality"  # "inequality" or "equality"

    def evaluate(self, parameters: Dict[str, float]) -> float:
        """Evaluate left-hand side of constraint."""
        return sum(coef * parameters.get(name, 0.0) for name, coef in self.coefficients.items())

    def is_feasible(self, parameters: Dict[str, float]) -> bool:
        """Check if parameters satisfy constraint."""
        value = self.evaluate(parameters)

        if self.constraint_type == "equality":
            # For equality, check if close to bound (within small tolerance)
            if self.upper_bound is not None:
                return abs(value - self.upper_bound) < 1e-6
            return True

        # Inequality constraints
        if self.lower_bound is not None and value < self.lower_bound:
            return False
        if self.upper_bound is not None and value > self.upper_bound:
            return False

        return True

    def violation(self, parameters: Dict[str, float]) -> float:
        """Calculate constraint violation."""
        value = self.evaluate(parameters)

        if self.constraint_type == "equality":
            if self.upper_bound is not None:
                return abs(value - self.upper_bound)
            return 0.0

        # Inequality
        violation = 0.0
        if self.lower_bound is not None and value < self.lower_bound:
            violation = max(violation, self.lower_bound - value)
        if self.upper_bound is not None and value > self.upper_bound:
            violation = max(violation, value - self.upper_bound)

        return violation


@dataclass
class NonlinearConstraint:
    """
    Nonlinear constraint: g(x) <= 0 or h(x) == 0

    Attributes:
        name: Constraint name
        function: Constraint function g(parameters) -> float
        constraint_type: "inequality" (g(x) <= 0) or "equality" (h(x) == 0)
        tolerance: Tolerance for equality constraints
    """

    name: str
    function: Callable[[Dict[str, float]], float]
    constraint_type: str = "inequality"
    tolerance: float = 1e-6

    def evaluate(self, parameters: Dict[str, float]) -> float:
        """Evaluate constraint function."""
        return self.function(parameters)

    def is_feasible(self, parameters: Dict[str, float]) -> bool:
        """Check if constraint is satisfied."""
        value = self.evaluate(parameters)

        if self.constraint_type == "equality":
            return abs(value) <= self.tolerance

        # Inequality: g(x) <= 0
        return value <= self.tolerance

    def violation(self, parameters: Dict[str, float]) -> float:
        """Calculate violation amount."""
        value = self.evaluate(parameters)

        if self.constraint_type == "equality":
            return abs(value)

        # Inequality
        return max(0.0, value)


class PenaltyFunction(ABC):
    """
    Abstract base class for penalty functions.

    Penalty functions are used to handle soft constraints by adding
    a penalty term to the objective function.
    """

    @abstractmethod
    def __call__(self, violation: float, weight: float = 1.0) -> float:
        """
        Calculate penalty for constraint violation.

        Args:
            violation: Magnitude of constraint violation
            weight: Penalty weight

        Returns:
            Penalty value
        """
        pass


class QuadraticPenalty(PenaltyFunction):
    """
    Quadratic penalty function: weight * violation²

    Smooth penalty that increases rapidly with violation.
    """

    def __call__(self, violation: float, weight: float = 1.0) -> float:
        """Calculate quadratic penalty."""
        return weight * violation**2


class LinearPenalty(PenaltyFunction):
    """
    Linear penalty function: weight * |violation|

    Simple linear penalty proportional to violation.
    """

    def __call__(self, violation: float, weight: float = 1.0) -> float:
        """Calculate linear penalty."""
        return weight * abs(violation)


class LogarithmicPenalty(PenaltyFunction):
    """
    Logarithmic barrier penalty: -weight * log(distance_to_bound)

    Creates a barrier that prevents parameters from reaching bounds.
    """

    def __call__(self, violation: float, weight: float = 1.0) -> float:
        """Calculate logarithmic penalty."""
        if violation <= 0:
            return 0.0
        return -weight * np.log(max(1e-10, violation))


class ExponentialPenalty(PenaltyFunction):
    """
    Exponential penalty: weight * exp(violation) - 1

    Penalty grows exponentially with violation.
    """

    def __call__(self, violation: float, weight: float = 1.0) -> float:
        """Calculate exponential penalty."""
        if violation <= 0:
            return 0.0
        return weight * (np.exp(violation) - 1.0)


class BarrierPenalty(PenaltyFunction):
    """
    Inverse barrier penalty: weight / distance_to_bound

    Creates a barrier that approaches infinity at the bound.
    """

    def __call__(self, violation: float, weight: float = 1.0) -> float:
        """Calculate barrier penalty."""
        if violation <= 0:
            return 0.0
        return weight / max(1e-10, violation)


class ParameterConstraints:
    """
    Manager for all parameter constraints.

    Handles box constraints, linear constraints, and nonlinear constraints
    with support for penalty functions.

    Example:
        >>> constraints = ParameterConstraints()
        >>> constraints.add_box_constraint("k_dis", 0.3, 0.8)
        >>> constraints.add_linear_inequality({"k_dis": 1, "Y_su": 1}, upper_bound=1.0)
        >>> is_valid = constraints.is_feasible({"k_dis": 0.5, "Y_su": 0.1})
    """

    def __init__(self, penalty_function: Optional[PenaltyFunction] = None):
        """
        Initialize constraint manager.

        Args:
            penalty_function: Penalty function for soft constraints
                            (default: QuadraticPenalty)
        """
        self.box_constraints: Dict[str, BoxConstraint] = {}
        self.linear_constraints: List[LinearConstraint] = []
        self.nonlinear_constraints: List[NonlinearConstraint] = []

        self.penalty_function = penalty_function or QuadraticPenalty()
        self.penalty_weights: Dict[str, float] = {}

    def add_box_constraint(self, parameter_name: str, lower: float, upper: float, hard: bool = True, weight: float = 1.0):
        """
        Add box constraint (bounds) for a parameter.

        Args:
            parameter_name: Parameter name
            lower: Lower bound
            upper: Upper bound
            hard: Hard constraint (infinite penalty if violated)
            weight: Penalty weight for soft constraints
        """
        self.box_constraints[parameter_name] = BoxConstraint(parameter_name, lower, upper, hard)

        if not hard:
            self.penalty_weights[f"box_{parameter_name}"] = weight

    def add_linear_inequality(
        self,
        coefficients: Dict[str, float],
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
        weight: float = 1.0,
    ):
        """
        Add linear inequality constraint.

        Args:
            coefficients: Coefficients for each parameter
            lower_bound: Lower bound (sum >= lower_bound)
            upper_bound: Upper bound (sum <= upper_bound)
            weight: Penalty weight
        """
        constraint = LinearConstraint(coefficients, lower_bound, upper_bound, constraint_type="inequality")

        self.linear_constraints.append(constraint)

        # Store penalty weight
        constraint_id = f"linear_{len(self.linear_constraints)}"
        self.penalty_weights[constraint_id] = weight

    def add_linear_equality(self, coefficients: Dict[str, float], target: float, weight: float = 1.0):
        """
        Add linear equality constraint: sum(coefficients * parameters) == target

        Args:
            coefficients: Coefficients for each parameter
            target: Target value
            weight: Penalty weight
        """
        constraint = LinearConstraint(coefficients, lower_bound=None, upper_bound=target, constraint_type="equality")

        self.linear_constraints.append(constraint)

        constraint_id = f"linear_eq_{len(self.linear_constraints)}"
        self.penalty_weights[constraint_id] = weight

    def add_nonlinear_constraint(
        self,
        name: str,
        function: Callable[[Dict[str, float]], float],
        constraint_type: str = "inequality",
        weight: float = 1.0,
    ):
        """
        Add nonlinear constraint.

        Args:
            name: Constraint name
            function: Constraint function g(params) -> float
                     For inequality: g(x) <= 0
                     For equality: g(x) == 0
            constraint_type: "inequality" or "equality"
            weight: Penalty weight
        """
        constraint = NonlinearConstraint(name, function, constraint_type)

        self.nonlinear_constraints.append(constraint)

        constraint_id = f"nonlinear_{name}"
        self.penalty_weights[constraint_id] = weight

    def is_feasible(self, parameters: Dict[str, float]) -> bool:
        """
        Check if parameters satisfy all hard constraints.

        Args:
            parameters: Parameter values

        Returns:
            True if all hard constraints are satisfied
        """
        # Check box constraints
        for name, constraint in self.box_constraints.items():
            if constraint.hard:
                value = parameters.get(name, 0.0)
                if not constraint.is_feasible(value):
                    return False

        # Check linear constraints (all treated as hard for feasibility)
        for constraint in self.linear_constraints:
            if not constraint.is_feasible(parameters):
                return False

        # Check nonlinear constraints (all treated as hard)
        for constraint in self.nonlinear_constraints:
            if not constraint.is_feasible(parameters):
                return False

        return True

    def calculate_penalty(self, parameters: Dict[str, float]) -> float:
        """
        Calculate total penalty for constraint violations.

        Args:
            parameters: Parameter values

        Returns:
            Total penalty value
        """
        total_penalty = 0.0

        # Box constraint penalties
        for name, constraint in self.box_constraints.items():
            value = parameters.get(name, 0.0)
            violation = constraint.violation(value)

            if violation > 0:
                if constraint.hard:
                    return float("inf")

                weight = self.penalty_weights.get(f"box_{name}", 1.0)
                penalty = self.penalty_function(violation, weight)
                total_penalty += penalty

        # Linear constraint penalties
        for i, constraint in enumerate(self.linear_constraints, 1):
            violation = constraint.violation(parameters)

            if violation > 0:
                weight = self.penalty_weights.get(f"linear_{i}", 1.0)
                penalty = self.penalty_function(violation, weight)
                total_penalty += penalty

        # Nonlinear constraint penalties
        for constraint in self.nonlinear_constraints:
            violation = constraint.violation(parameters)

            if violation > 0:
                weight = self.penalty_weights.get(f"nonlinear_{constraint.name}", 1.0)
                penalty = self.penalty_function(violation, weight)
                total_penalty += penalty

        return total_penalty

    def project_to_feasible(self, parameters: Dict[str, float]) -> Dict[str, float]:
        """
        Project parameters to feasible region (box constraints only).

        Args:
            parameters: Parameter values

        Returns:
            Projected parameters
        """
        projected = parameters.copy()

        for name, constraint in self.box_constraints.items():
            if name in projected:
                projected[name] = constraint.project(projected[name])

        return projected

    def get_bounds_array(self, parameter_names: List[str]) -> np.ndarray:
        """
        Get bounds as array for scipy optimizers.

        Args:
            parameter_names: Ordered list of parameter names

        Returns:
            Array of shape (n_params, 2) with [lower, upper] for each parameter
        """
        bounds = []
        for name in parameter_names:
            if name in self.box_constraints:
                constraint = self.box_constraints[name]
                bounds.append([constraint.lower, constraint.upper])
            else:
                bounds.append([None, None])

        return np.array(bounds)

    def get_scipy_constraints(self, parameter_names: List[str]) -> List[Dict]:
        """
        Convert constraints to scipy format for constrained optimization.

        Args:
            parameter_names: Ordered list of parameter names

        Returns:
            List of constraint dictionaries for scipy.optimize.minimize
        """
        scipy_constraints = []

        # Linear constraints
        for constraint in self.linear_constraints:
            # Build coefficient array
            coef_array = np.array([constraint.coefficients.get(name, 0.0) for name in parameter_names])

            if constraint.constraint_type == "equality":
                # Equality: sum(coef * x) == target
                scipy_constraints.append(
                    {"type": "eq", "fun": lambda x, c=coef_array, b=constraint.upper_bound: np.dot(c, x) - b}
                )
            else:
                # Inequality constraints
                if constraint.lower_bound is not None:
                    # sum(coef * x) >= lower
                    scipy_constraints.append(
                        {"type": "ineq", "fun": lambda x, c=coef_array, b=constraint.lower_bound: np.dot(c, x) - b}
                    )
                if constraint.upper_bound is not None:
                    # sum(coef * x) <= upper  =>  upper - sum(coef * x) >= 0
                    scipy_constraints.append(
                        {"type": "ineq", "fun": lambda x, c=coef_array, b=constraint.upper_bound: b - np.dot(c, x)}
                    )

        # Nonlinear constraints
        for constraint in self.nonlinear_constraints:

            def constraint_func(x, names=parameter_names, func=constraint.function):
                params = {name: val for name, val in zip(names, x)}
                return func(params)

            if constraint.constraint_type == "equality":
                scipy_constraints.append({"type": "eq", "fun": constraint_func})
            else:
                # g(x) <= 0  =>  -g(x) >= 0
                scipy_constraints.append({"type": "ineq", "fun": lambda x, f=constraint_func: -f(x)})

        return scipy_constraints

    def validate_parameters(self, parameters: Dict[str, float]) -> Tuple[bool, List[str]]:
        """
        Validate parameters and return detailed error messages.

        Args:
            parameters: Parameter values

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check box constraints
        for name, constraint in self.box_constraints.items():
            if name in parameters:
                value = parameters[name]
                if not constraint.is_feasible(value):
                    errors.append(
                        f"Parameter '{name}' = {value:.4f} violates bounds "
                        f"[{constraint.lower:.4f}, {constraint.upper:.4f}]"
                    )

        # Check linear constraints
        for i, constraint in enumerate(self.linear_constraints, 1):
            if not constraint.is_feasible(parameters):
                value = constraint.evaluate(parameters)
                if constraint.constraint_type == "equality":
                    errors.append(f"Linear constraint {i}: {value:.4f} != {constraint.upper_bound:.4f}")
                else:
                    if constraint.lower_bound and value < constraint.lower_bound:
                        errors.append(f"Linear constraint {i}: {value:.4f} < {constraint.lower_bound:.4f}")
                    if constraint.upper_bound and value > constraint.upper_bound:
                        errors.append(f"Linear constraint {i}: {value:.4f} > {constraint.upper_bound:.4f}")

        # Check nonlinear constraints
        for constraint in self.nonlinear_constraints:
            if not constraint.is_feasible(parameters):
                value = constraint.evaluate(parameters)
                errors.append(
                    f"Nonlinear constraint '{constraint.name}': g(x) = {value:.4f} violates "
                    f"{constraint.constraint_type} constraint"
                )

        return len(errors) == 0, errors


def create_penalty_function(penalty_type: str) -> PenaltyFunction:
    """
    Factory function for creating penalty functions.

    Args:
        penalty_type: Type of penalty ("quadratic", "linear", "logarithmic",
                     "exponential", "barrier")

    Returns:
        PenaltyFunction instance

    Example:
        >>> penalty = create_penalty_function("quadratic")
        >>> value = penalty(violation=0.5, weight=2.0)
    """
    penalty_type = penalty_type.lower()

    penalty_map = {
        "quadratic": QuadraticPenalty,
        "linear": LinearPenalty,
        "logarithmic": LogarithmicPenalty,
        "log": LogarithmicPenalty,
        "exponential": ExponentialPenalty,
        "exp": ExponentialPenalty,
        "barrier": BarrierPenalty,
    }

    if penalty_type not in penalty_map:
        available = ", ".join(penalty_map.keys())
        raise ValueError(f"Unknown penalty type: {penalty_type}. Available: {available}")

    return penalty_map[penalty_type]()
