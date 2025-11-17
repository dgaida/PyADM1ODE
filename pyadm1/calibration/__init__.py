"""
Model Calibration Framework

Tools for initial calibration and online re-calibration of biogas plant models
against measurement data.

Modules:
    calibrator: Main Calibrator class providing unified interface for initial and
               online calibration, managing optimization runs, parameter updates,
               and validation against measurement data with logging and reporting.

    initial: Initial calibration from historical measurement data using batch
            optimization, supports multiple objectives (gas production, pH, VFA),
            with sensitivity analysis and parameter identifiability assessment.

    online: Online re-calibration during plant operation triggered by high state
           estimation variance, uses moving window of recent data, enforces parameter
           bounds to prevent drift from physical values.

    parameter_bounds: Parameter bound management defining physically plausible ranges
                     for ADM1 parameters, substrate-specific constraints, and penalty
                     functions for soft constraints during optimization.

    validation: Calibration result validation including goodness-of-fit metrics
               (RMSE, R², Nash-Sutcliffe), residual analysis, parameter correlation
               analysis, and cross-validation on held-out data.

Subpackage:
    optimization: Optimization algorithms and objective functions including
                 gradient-free methods (Nelder-Mead, differential evolution),
                 gradient-based methods (L-BFGS-B), and multi-objective
                 optimization with Pareto front analysis.

Example:
    >>> from pyadm1.calibration import Calibrator
    >>> from pyadm1.io import MeasurementData
    >>>
    >>> # Load measurement data
    >>> measurements = MeasurementData.from_csv("plant_data.csv")
    >>>
    >>> # Initial calibration
    >>> calibrator = Calibrator(plant)
    >>> result = calibrator.calibrate_initial(
    ...     measurements=measurements,
    ...     parameters=["k_dis", "k_hyd_ch", "Y_su"],
    ...     bounds={"k_dis": (0.3, 0.8)},
    ...     method="differential_evolution"
    ... )
    >>>
    >>> # Online re-calibration
    >>> calibrator.calibrate_online(
    ...     measurements=new_measurements,
    ...     variance_threshold=0.1,
    ...     max_parameter_change=0.2
    ... )
"""

from pyadm1.calibration.calibrator import Calibrator, CalibrationResult
from pyadm1.calibration.initial import InitialCalibrator

from pyadm1.calibration.online import OnlineCalibrator
from pyadm1.calibration.parameter_bounds import (
    ParameterBounds,
    BoundType,
    create_default_bounds,
)
from pyadm1.calibration.validation import (
    CalibrationValidator,
    ValidationMetrics,
)

# Import optimization subpackage
from pyadm1.calibration import optimization

__all__ = [
    "Calibrator",
    "CalibrationResult",
    "InitialCalibrator",
    "OnlineCalibrator",
    "ParameterBounds",
    "BoundType",
    "create_default_bounds",
    "CalibrationValidator",
    "ValidationMetrics",
    "optimization",
]
