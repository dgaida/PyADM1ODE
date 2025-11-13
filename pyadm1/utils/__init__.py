"""
Utility Functions and Helper Classes

Common utilities used throughout the PyADM1 framework.

Modules:
    math_helpers: Mathematical helper functions including safe division with zero
                 handling, numerical derivatives, interpolation methods, statistical
                 functions, and matrix operations for process calculations.

    unit_conversion: Unit conversion utilities for common biogas plant units including
                    volume (m³, L), mass (kg, t), energy (kWh, MJ), concentration
                    (g/L, mol/L), and custom biogas-specific units (Nm³, m³/t VS).

    logging: Logging configuration with structured logging, multiple handlers (console,
            file, database), log levels per module, and integration with monitoring
            systems for production environments.

    validators: Validation functions for parameters, configurations, and data including
               type checking, range validation, consistency checks, and custom
               validators with descriptive error messages.

Example:
    >>> from pyadm1.utils import (
    ...     safe_divide,
    ...     convert_units,
    ...     setup_logging,
    ...     validate_parameters
    ... )
    >>>
    >>> # Safe division
    >>> result = safe_divide(numerator=10, denominator=0, default=0.0)
    >>>
    >>> # Unit conversion
    >>> energy_mj = convert_units(100, from_unit="kWh", to_unit="MJ")
    >>>
    >>> # Setup logging
    >>> logger = setup_logging(level="INFO", log_file="simulation.log")
    >>>
    >>> # Validate parameters
    >>> is_valid = validate_parameters(
    ...     params={"k_dis": 0.5, "Y_su": 0.1},
    ...     bounds={"k_dis": (0.3, 0.8), "Y_su": (0.05, 0.15)}
    ... )
"""

from pyadm1.utils.math_helpers import (
    safe_divide,
    safe_log,
    numerical_derivative,
    interpolate_linear,
)
from pyadm1.utils.unit_conversion import (
    UnitConverter,
    convert_units,
    BiogasUnits,
)
from pyadm1.utils.logging import (
    setup_logging,
    get_logger,
    LogConfig,
)
from pyadm1.utils.validators import (
    validate_parameters,
    validate_configuration,
    ParameterValidator,
)

__all__ = [
    "safe_divide",
    "safe_log",
    "numerical_derivative",
    "interpolate_linear",
    "UnitConverter",
    "convert_units",
    "BiogasUnits",
    "setup_logging",
    "get_logger",
    "LogConfig",
    "validate_parameters",
    "validate_configuration",
    "ParameterValidator",
]
