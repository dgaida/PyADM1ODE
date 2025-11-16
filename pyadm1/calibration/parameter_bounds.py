# pyadm1/calibration/parameter_bounds.py
"""
Parameter Bounds Management for ADM1 Calibration

This module provides classes and functions for managing parameter bounds during
calibration, including physically plausible ranges, substrate-specific constraints,
and penalty functions for soft constraints.

Features:
- Default bounds based on literature and physical constraints
- Substrate-specific bounds for kinetic parameters
- Temperature-dependent bounds for biochemical rates
- Soft and hard constraints with penalty functions
- Bounds validation and consistency checking

Example:
    >>> from pyadm1.calibration.parameter_bounds import (
    ...     create_default_bounds,
    ...     ParameterBounds,
    ...     BoundType
    ... )
    >>>
    >>> # Get default bounds
    >>> bounds_manager = create_default_bounds()
    >>>
    >>> # Get bounds for a parameter
    >>> k_dis_bounds = bounds_manager.get_bounds("k_dis")
    >>> print(f"k_dis range: [{k_dis_bounds.lower}, {k_dis_bounds.upper}]")
    >>>
    >>> # Check if value is within bounds
    >>> is_valid = bounds_manager.is_within_bounds("k_dis", 0.5)
    >>>
    >>> # Apply penalty for out-of-bounds values
    >>> penalty = bounds_manager.calculate_penalty("k_dis", 1.0)
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
from enum import Enum
import numpy as np


class BoundType(Enum):
    """Type of parameter bound constraint."""

    HARD = "hard"  # Must not be violated
    SOFT = "soft"  # Can be violated with penalty
    FIXED = "fixed"  # Parameter is fixed, not calibrated


@dataclass
class ParameterBound:
    """
    Bounds for a single parameter.

    Attributes:
        name: Parameter name
        lower: Lower bound
        upper: Upper bound
        default: Default value
        bound_type: Type of bound (hard, soft, fixed)
        penalty_weight: Weight for soft constraint penalty
        description: Description of parameter
        unit: Parameter unit
        substrate_dependent: Whether bound varies by substrate
    """

    name: str
    lower: float
    upper: float
    default: float
    bound_type: BoundType = BoundType.HARD
    penalty_weight: float = 1.0
    description: str = ""
    unit: str = ""
    substrate_dependent: bool = False

    def is_within_bounds(self, value: float, tolerance: float = 0.0) -> bool:
        """
        Check if value is within bounds.

        Args:
            value: Value to check
            tolerance: Tolerance for boundary (default: 0.0)

        Returns:
            True if value is within bounds
        """
        return (self.lower - tolerance) <= value <= (self.upper + tolerance)

    def clip_to_bounds(self, value: float) -> float:
        """
        Clip value to bounds.

        Args:
            value: Value to clip

        Returns:
            Clipped value
        """
        return np.clip(value, self.lower, self.upper)

    def calculate_penalty(self, value: float, penalty_type: str = "quadratic") -> float:
        """
        Calculate penalty for out-of-bounds value.

        Args:
            value: Parameter value
            penalty_type: Type of penalty function:
                - "quadratic": (distance)^2
                - "linear": abs(distance)
                - "logarithmic": -log(distance to bound)
                - "barrier": 1/distance (approaches infinity at bound)

        Returns:
            Penalty value (0 if within bounds)
        """
        if self.bound_type == BoundType.HARD:
            # Hard constraints: infinite penalty if violated
            if not self.is_within_bounds(value):
                return np.inf
            return 0.0

        elif self.bound_type == BoundType.SOFT:
            # Soft constraints: calculate penalty
            if self.is_within_bounds(value):
                return 0.0

            # Calculate distance to nearest bound
            if value < self.lower:
                distance = self.lower - value
            else:
                distance = value - self.upper

            # Apply penalty function
            if penalty_type == "quadratic":
                penalty = self.penalty_weight * distance**2
            elif penalty_type == "linear":
                penalty = self.penalty_weight * abs(distance)
            elif penalty_type == "logarithmic":
                penalty = -self.penalty_weight * np.log(max(1e-10, distance))
            elif penalty_type == "barrier":
                penalty = self.penalty_weight / max(1e-10, distance)
            else:
                raise ValueError(f"Unknown penalty type: {penalty_type}")

            return penalty

        else:  # FIXED
            return 0.0

    def get_relative_position(self, value: float) -> float:
        """
        Get relative position of value in bounds [0, 1].

        Args:
            value: Parameter value

        Returns:
            Relative position (0 = lower bound, 1 = upper bound)
        """
        if self.upper == self.lower:
            return 0.5
        return (value - self.lower) / (self.upper - self.lower)


class ParameterBounds:
    """
    Manager for parameter bounds in ADM1 calibration.

    Provides methods for accessing bounds, validating parameters,
    and calculating penalties for constraint violations.

    Attributes:
        bounds: Dictionary mapping parameter names to ParameterBound objects

    Example:
        >>> bounds = ParameterBounds()
        >>> bounds.add_bound("k_dis", lower=0.3, upper=0.8, default=0.5)
        >>> is_valid = bounds.is_within_bounds("k_dis", 0.6)
    """

    def __init__(self):
        """Initialize empty parameter bounds manager."""
        self.bounds: Dict[str, ParameterBound] = {}

    def add_bound(
        self,
        name: str,
        lower: float,
        upper: float,
        default: float,
        bound_type: BoundType = BoundType.HARD,
        penalty_weight: float = 1.0,
        description: str = "",
        unit: str = "",
        substrate_dependent: bool = False,
    ) -> None:
        """
        Add parameter bound.

        Args:
            name: Parameter name
            lower: Lower bound
            upper: Upper bound
            default: Default value
            bound_type: Type of bound
            penalty_weight: Penalty weight for soft constraints
            description: Parameter description
            unit: Parameter unit
            substrate_dependent: Whether bound varies by substrate
        """
        self.bounds[name] = ParameterBound(
            name=name,
            lower=lower,
            upper=upper,
            default=default,
            bound_type=bound_type,
            penalty_weight=penalty_weight,
            description=description,
            unit=unit,
            substrate_dependent=substrate_dependent,
        )

    def get_bounds(self, name: str) -> Optional[ParameterBound]:
        """
        Get bounds for parameter.

        Args:
            name: Parameter name

        Returns:
            ParameterBound object or None if not found
        """
        return self.bounds.get(name)

    def get_bounds_tuple(self, name: str) -> Optional[Tuple[float, float]]:
        """
        Get bounds as tuple (lower, upper).

        Args:
            name: Parameter name

        Returns:
            Tuple of (lower, upper) or None
        """
        bound = self.get_bounds(name)
        if bound is None:
            return None
        return (bound.lower, bound.upper)

    def is_within_bounds(self, name: str, value: float, tolerance: float = 0.0) -> bool:
        """
        Check if parameter value is within bounds.

        Args:
            name: Parameter name
            value: Parameter value
            tolerance: Tolerance for boundary

        Returns:
            True if within bounds, False otherwise
        """
        bound = self.get_bounds(name)
        if bound is None:
            return True  # No bounds defined, assume valid
        return bound.is_within_bounds(value, tolerance)

    def clip_to_bounds(self, name: str, value: float) -> float:
        """
        Clip parameter value to bounds.

        Args:
            name: Parameter name
            value: Parameter value

        Returns:
            Clipped value
        """
        bound = self.get_bounds(name)
        if bound is None:
            return value
        return bound.clip_to_bounds(value)

    def calculate_penalty(self, name: str, value: float, penalty_type: str = "quadratic") -> float:
        """
        Calculate penalty for parameter value.

        Args:
            name: Parameter name
            value: Parameter value
            penalty_type: Type of penalty function

        Returns:
            Penalty value
        """
        bound = self.get_bounds(name)
        if bound is None:
            return 0.0
        return bound.calculate_penalty(value, penalty_type)

    def calculate_total_penalty(self, parameters: Dict[str, float], penalty_type: str = "quadratic") -> float:
        """
        Calculate total penalty for all parameters.

        Args:
            parameters: Dictionary of parameter values
            penalty_type: Type of penalty function

        Returns:
            Total penalty
        """
        total_penalty = 0.0
        for name, value in parameters.items():
            penalty = self.calculate_penalty(name, value, penalty_type)
            if np.isinf(penalty):
                return np.inf
            total_penalty += penalty
        return total_penalty

    def validate_parameters(self, parameters: Dict[str, float], raise_on_invalid: bool = False) -> Tuple[bool, List[str]]:
        """
        Validate all parameters against bounds.

        Args:
            parameters: Dictionary of parameter values
            raise_on_invalid: Raise exception if invalid

        Returns:
            Tuple of (all_valid, list of error messages)
        """
        errors = []

        for name, value in parameters.items():
            bound = self.get_bounds(name)
            if bound is None:
                continue

            if not bound.is_within_bounds(value):
                error = f"Parameter '{name}' = {value:.4f} is outside bounds " f"[{bound.lower:.4f}, {bound.upper:.4f}]"
                errors.append(error)

        if errors and raise_on_invalid:
            raise ValueError("\n".join(errors))

        return (len(errors) == 0, errors)

    def get_default_values(self, parameter_names: List[str]) -> Dict[str, float]:
        """
        Get default values for parameters.

        Args:
            parameter_names: List of parameter names

        Returns:
            Dictionary of default values
        """
        defaults = {}
        for name in parameter_names:
            bound = self.get_bounds(name)
            if bound is not None:
                defaults[name] = bound.default
        return defaults

    def scale_to_unit_interval(self, name: str, value: float) -> float:
        """
        Scale parameter value to unit interval [0, 1].

        Useful for optimization algorithms that work better with
        normalized parameters.

        Args:
            name: Parameter name
            value: Parameter value

        Returns:
            Scaled value in [0, 1]
        """
        bound = self.get_bounds(name)
        if bound is None:
            return value

        if bound.upper == bound.lower:
            return 0.5

        return (value - bound.lower) / (bound.upper - bound.lower)

    def unscale_from_unit_interval(self, name: str, scaled_value: float) -> float:
        """
        Unscale parameter value from unit interval [0, 1].

        Args:
            name: Parameter name
            scaled_value: Scaled value in [0, 1]

        Returns:
            Unscaled parameter value
        """
        bound = self.get_bounds(name)
        if bound is None:
            return scaled_value

        return bound.lower + scaled_value * (bound.upper - bound.lower)


def create_default_bounds() -> ParameterBounds:
    """
    Create parameter bounds with default ADM1 values.

    Bounds are based on:
    - Batstone et al. (2002): ADM1 base values
    - Rosen et al. (2006): BSM2 values
    - Gaida (2014): Agricultural substrate calibrations
    - Literature ranges for agricultural biogas plants

    Returns:
        ParameterBounds with default ADM1 parameter bounds

    Example:
        >>> bounds = create_default_bounds()
        >>> k_dis_bounds = bounds.get_bounds("k_dis")
    """
    bounds = ParameterBounds()

    # Disintegration rate [1/d] - substrate dependent
    bounds.add_bound(
        "k_dis",
        lower=0.1,
        upper=1.0,
        default=0.5,
        bound_type=BoundType.SOFT,
        penalty_weight=2.0,
        description="Disintegration rate constant",
        unit="1/d",
        substrate_dependent=True,
    )

    # Hydrolysis rates [1/d] - substrate dependent
    bounds.add_bound(
        "k_hyd_ch",
        lower=5.0,
        upper=15.0,
        default=10.0,
        bound_type=BoundType.SOFT,
        penalty_weight=1.0,
        description="Hydrolysis rate for carbohydrates",
        unit="1/d",
        substrate_dependent=True,
    )

    bounds.add_bound(
        "k_hyd_pr",
        lower=5.0,
        upper=15.0,
        default=10.0,
        bound_type=BoundType.SOFT,
        penalty_weight=1.0,
        description="Hydrolysis rate for proteins",
        unit="1/d",
        substrate_dependent=True,
    )

    bounds.add_bound(
        "k_hyd_li",
        lower=5.0,
        upper=15.0,
        default=10.0,
        bound_type=BoundType.SOFT,
        penalty_weight=1.0,
        description="Hydrolysis rate for lipids",
        unit="1/d",
        substrate_dependent=True,
    )

    # Yield coefficients [kg COD/kg COD]
    bounds.add_bound(
        "Y_su",
        lower=0.05,
        upper=0.15,
        default=0.10,
        bound_type=BoundType.HARD,
        description="Yield of sugar degraders",
        unit="kg COD/kg COD",
    )

    bounds.add_bound(
        "Y_aa",
        lower=0.04,
        upper=0.12,
        default=0.08,
        bound_type=BoundType.HARD,
        description="Yield of amino acid degraders",
        unit="kg COD/kg COD",
    )

    bounds.add_bound(
        "Y_fa",
        lower=0.03,
        upper=0.10,
        default=0.06,
        bound_type=BoundType.HARD,
        description="Yield of LCFA degraders",
        unit="kg COD/kg COD",
    )

    bounds.add_bound(
        "Y_c4",
        lower=0.03,
        upper=0.10,
        default=0.06,
        bound_type=BoundType.HARD,
        description="Yield of valerate and butyrate degraders",
        unit="kg COD/kg COD",
    )

    bounds.add_bound(
        "Y_pro",
        lower=0.02,
        upper=0.08,
        default=0.04,
        bound_type=BoundType.HARD,
        description="Yield of propionate degraders",
        unit="kg COD/kg COD",
    )

    bounds.add_bound(
        "Y_ac",
        lower=0.03,
        upper=0.08,
        default=0.05,
        bound_type=BoundType.HARD,
        description="Yield of acetate degraders",
        unit="kg COD/kg COD",
    )

    bounds.add_bound(
        "Y_h2",
        lower=0.03,
        upper=0.10,
        default=0.06,
        bound_type=BoundType.HARD,
        description="Yield of hydrogen degraders",
        unit="kg COD/kg COD",
    )

    # Maximum uptake rates [1/d] - substrate dependent
    bounds.add_bound(
        "k_m_su",
        lower=20.0,
        upper=40.0,
        default=30.0,
        bound_type=BoundType.SOFT,
        penalty_weight=1.0,
        description="Maximum uptake rate for sugars",
        unit="1/d",
    )

    bounds.add_bound(
        "k_m_aa",
        lower=35.0,
        upper=65.0,
        default=50.0,
        bound_type=BoundType.SOFT,
        penalty_weight=1.0,
        description="Maximum uptake rate for amino acids",
        unit="1/d",
    )

    bounds.add_bound(
        "k_m_fa",
        lower=3.0,
        upper=10.0,
        default=6.0,
        bound_type=BoundType.SOFT,
        penalty_weight=1.0,
        description="Maximum uptake rate for LCFA",
        unit="1/d",
    )

    bounds.add_bound(
        "k_m_c4",
        lower=15.0,
        upper=30.0,
        default=20.0,
        bound_type=BoundType.SOFT,
        penalty_weight=1.5,
        description="Maximum uptake rate for valerate and butyrate",
        unit="1/d",
        substrate_dependent=True,
    )

    bounds.add_bound(
        "k_m_pro",
        lower=8.0,
        upper=18.0,
        default=13.0,
        bound_type=BoundType.SOFT,
        penalty_weight=1.5,
        description="Maximum uptake rate for propionate",
        unit="1/d",
        substrate_dependent=True,
    )

    bounds.add_bound(
        "k_m_ac",
        lower=4.0,
        upper=12.0,
        default=8.0,
        bound_type=BoundType.SOFT,
        penalty_weight=2.0,
        description="Maximum uptake rate for acetate",
        unit="1/d",
        substrate_dependent=True,
    )

    bounds.add_bound(
        "k_m_h2",
        lower=25.0,
        upper=45.0,
        default=35.0,
        bound_type=BoundType.SOFT,
        penalty_weight=1.5,
        description="Maximum uptake rate for hydrogen",
        unit="1/d",
        substrate_dependent=True,
    )

    # Half-saturation constants [kg COD/m³]
    bounds.add_bound(
        "K_S_su",
        lower=0.3,
        upper=0.7,
        default=0.5,
        bound_type=BoundType.SOFT,
        penalty_weight=1.0,
        description="Half-saturation constant for sugars",
        unit="kg COD/m³",
    )

    bounds.add_bound(
        "K_S_aa",
        lower=0.2,
        upper=0.5,
        default=0.3,
        bound_type=BoundType.SOFT,
        penalty_weight=1.0,
        description="Half-saturation constant for amino acids",
        unit="kg COD/m³",
    )

    bounds.add_bound(
        "K_S_fa",
        lower=0.3,
        upper=0.6,
        default=0.4,
        bound_type=BoundType.SOFT,
        penalty_weight=1.0,
        description="Half-saturation constant for LCFA",
        unit="kg COD/m³",
    )

    bounds.add_bound(
        "K_S_c4",
        lower=0.15,
        upper=0.4,
        default=0.2,
        bound_type=BoundType.SOFT,
        penalty_weight=1.0,
        description="Half-saturation constant for C4",
        unit="kg COD/m³",
    )

    bounds.add_bound(
        "K_S_pro",
        lower=0.05,
        upper=0.15,
        default=0.1,
        bound_type=BoundType.SOFT,
        penalty_weight=1.0,
        description="Half-saturation constant for propionate",
        unit="kg COD/m³",
    )

    bounds.add_bound(
        "K_S_ac",
        lower=0.1,
        upper=0.25,
        default=0.15,
        bound_type=BoundType.SOFT,
        penalty_weight=1.0,
        description="Half-saturation constant for acetate",
        unit="kg COD/m³",
    )

    bounds.add_bound(
        "K_S_h2",
        lower=5e-6,
        upper=1e-5,
        default=7e-6,
        bound_type=BoundType.SOFT,
        penalty_weight=1.0,
        description="Half-saturation constant for hydrogen",
        unit="kg COD/m³",
    )

    # Decay rates [1/d]
    bounds.add_bound(
        "k_dec_X_su",
        lower=0.01,
        upper=0.04,
        default=0.02,
        bound_type=BoundType.SOFT,
        penalty_weight=0.5,
        description="Decay rate for sugar degraders",
        unit="1/d",
    )

    bounds.add_bound(
        "k_dec_X_aa",
        lower=0.01,
        upper=0.04,
        default=0.02,
        bound_type=BoundType.SOFT,
        penalty_weight=0.5,
        description="Decay rate for amino acid degraders",
        unit="1/d",
    )

    bounds.add_bound(
        "k_dec_X_fa",
        lower=0.01,
        upper=0.04,
        default=0.02,
        bound_type=BoundType.SOFT,
        penalty_weight=0.5,
        description="Decay rate for LCFA degraders",
        unit="1/d",
    )

    bounds.add_bound(
        "k_dec_X_c4",
        lower=0.01,
        upper=0.04,
        default=0.02,
        bound_type=BoundType.SOFT,
        penalty_weight=0.5,
        description="Decay rate for C4 degraders",
        unit="1/d",
    )

    bounds.add_bound(
        "k_dec_X_pro",
        lower=0.01,
        upper=0.04,
        default=0.02,
        bound_type=BoundType.SOFT,
        penalty_weight=0.5,
        description="Decay rate for propionate degraders",
        unit="1/d",
    )

    bounds.add_bound(
        "k_dec_X_ac",
        lower=0.01,
        upper=0.04,
        default=0.02,
        bound_type=BoundType.SOFT,
        penalty_weight=0.5,
        description="Decay rate for acetate degraders",
        unit="1/d",
    )

    bounds.add_bound(
        "k_dec_X_h2",
        lower=0.01,
        upper=0.04,
        default=0.02,
        bound_type=BoundType.SOFT,
        penalty_weight=0.5,
        description="Decay rate for hydrogen degraders",
        unit="1/d",
    )

    return bounds
