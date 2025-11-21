"""
Optimization Algorithms and Objective Functions

Optimization methods for parameter calibration with various algorithms and
customizable objective functions.

Modules:
    optimizer: Abstract Optimizer base class and concrete implementations including
              gradient-free methods (Nelder-Mead, Powell, differential evolution,
              particle swarm), gradient-based methods (L-BFGS-B, SLSQP), and
              multi-objective optimization (NSGA-II).

    objective: Objective function classes for single and multi-objective optimization
              including weighted sum of errors, likelihood-based objectives, and
              custom cost functions with support for different error metrics (MSE,
              MAE, log-likelihood).

    constraints: Constraint handling for parameter optimization including box constraints,
                linear constraints, nonlinear constraints, and penalty methods with
                different penalty functions (quadratic, logarithmic, barrier).

Example:
    >>> from pyadm1.calibration.optimization import (
    ...     DifferentialEvolutionOptimizer,
    ...     MultiObjectiveFunction,
    ...     ParameterConstraints
    ... )
    >>>
    >>> # Define objective function
    >>> objective = MultiObjectiveFunction(
    ...     targets=["Q_ch4", "pH", "VFA"],
    ...     weights=[0.6, 0.2, 0.2],
    ...     error_metric="rmse"
    ... )
    >>>
    >>> # Set up optimizer with constraints
    >>> optimizer = DifferentialEvolutionOptimizer(
    ...     objective=objective,
    ...     bounds=parameter_bounds,
    ...     population_size=50,
    ...     max_iterations=100
    ... )
    >>>
    >>> # Run optimization
    >>> result = optimizer.optimize(
    ...     plant=plant,
    ...     measurements=measurements
    ... )
"""

from pyadm1.calibration.optimization.optimizer import (
    Optimizer,
    GradientFreeOptimizer,
    GradientBasedOptimizer,
    DifferentialEvolutionOptimizer,
    ParticleSwarmOptimizer,
    NelderMeadOptimizer,
    LBFGSBOptimizer,
    create_optimizer,
)
from pyadm1.calibration.optimization.objective import (
    ObjectiveFunction,
    SingleObjective,
    MultiObjectiveFunction,
    WeightedSumObjective,
    LikelihoodObjective,
)
from pyadm1.calibration.optimization.constraints import (
    ParameterConstraints,
    BoxConstraint,
    LinearConstraint,
    NonlinearConstraint,
    PenaltyFunction,
)

__all__ = [
    "Optimizer",
    "GradientFreeOptimizer",
    "GradientBasedOptimizer",
    "DifferentialEvolutionOptimizer",
    "ParticleSwarmOptimizer",
    "NelderMeadOptimizer",
    "LBFGSBOptimizer",
    "create_optimizer",
    "ObjectiveFunction",
    "SingleObjective",
    "MultiObjectiveFunction",
    "WeightedSumObjective",
    "LikelihoodObjective",
    "ParameterConstraints",
    "BoxConstraint",
    "LinearConstraint",
    "NonlinearConstraint",
    "PenaltyFunction",
]
