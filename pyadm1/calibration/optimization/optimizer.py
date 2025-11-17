# pyadm1/calibration/optimization/optimizer.py
"""
Optimization Algorithms for Parameter Calibration

This module provides abstract and concrete optimizer classes for ADM1 parameter
calibration. Supports gradient-free and gradient-based methods with proper
abstraction for easy extension and testing.

Available optimizers:
- Gradient-free: Differential Evolution, Particle Swarm, Nelder-Mead, Powell
- Gradient-based: L-BFGS-B, SLSQP
- Multi-objective: NSGA-II (evolutionary multi-objective optimization)

Example:
    >>> from pyadm1.calibration.optimization import DifferentialEvolutionOptimizer
    >>> from pyadm1.calibration.optimization import WeightedSumObjective
    >>>
    >>> # Create objective function
    >>> objective = WeightedSumObjective(
    ...     simulator=simulator,
    ...     measurements=measurements,
    ...     objectives=["Q_ch4", "pH"],
    ...     weights={"Q_ch4": 0.8, "pH": 0.2}
    ... )
    >>>
    >>> # Create optimizer
    >>> optimizer = DifferentialEvolutionOptimizer(
    ...     bounds={"k_dis": (0.3, 0.8), "Y_su": (0.05, 0.15)},
    ...     max_iterations=100,
    ...     population_size=15
    ... )
    >>>
    >>> # Run optimization
    >>> result = optimizer.optimize(objective, initial_guess=None)
    >>> print(f"Best parameters: {result.x}")
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import differential_evolution, minimize, OptimizeResult
import time


@dataclass
class OptimizationResult:
    """
    Result from an optimization run.

    Attributes:
        success: Whether optimization converged
        x: Optimal parameter values
        fun: Objective function value at optimum
        nit: Number of iterations
        nfev: Number of function evaluations
        message: Status message
        parameter_names: Names of optimized parameters
        parameter_dict: Parameters as dictionary
        history: Optimization history (if tracked)
        execution_time: Wall clock time [seconds]
    """

    success: bool
    x: np.ndarray
    fun: float
    nit: int
    nfev: int
    message: str
    parameter_names: List[str]
    parameter_dict: Dict[str, float] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    execution_time: float = 0.0

    @classmethod
    def from_scipy_result(
        cls, result: OptimizeResult, parameter_names: List[str], execution_time: float, history: Optional[List] = None
    ) -> "OptimizationResult":
        """Create from scipy OptimizeResult."""
        param_dict = {name: float(val) for name, val in zip(parameter_names, result.x)}

        return cls(
            success=result.success,
            x=result.x,
            fun=result.fun,
            nit=result.nit,
            nfev=result.nfev,
            message=result.message,
            parameter_names=parameter_names,
            parameter_dict=param_dict,
            history=history or [],
            execution_time=execution_time,
        )


class Optimizer(ABC):
    """
    Abstract base class for optimization algorithms.

    All optimizers must implement the optimize() method and provide
    a consistent interface for parameter calibration.

    Attributes:
        bounds: Parameter bounds as {name: (min, max)}
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance
        verbose: Enable progress output
    """

    def __init__(
        self,
        bounds: Dict[str, Tuple[float, float]],
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        verbose: bool = True,
    ):
        """
        Initialize optimizer.

        Args:
            bounds: Parameter bounds {name: (min, max)}
            max_iterations: Maximum iterations
            tolerance: Convergence tolerance
            verbose: Enable output
        """
        self.bounds = bounds
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.verbose = verbose

        self.parameter_names = list(bounds.keys())
        self.bounds_array = np.array([bounds[name] for name in self.parameter_names])

        # History tracking
        self.history: List[Dict[str, Any]] = []
        self._best_value = float("inf")
        self._n_evaluations = 0

    @abstractmethod
    def optimize(
        self, objective_func: Callable[[np.ndarray], float], initial_guess: Optional[np.ndarray] = None
    ) -> OptimizationResult:
        """
        Run optimization.

        Args:
            objective_func: Function to minimize f(x) -> float
            initial_guess: Optional initial parameter guess

        Returns:
            OptimizationResult object
        """
        pass

    def _wrap_objective(self, objective_func: Callable[[np.ndarray], float]) -> Callable[[np.ndarray], float]:
        """
        Wrap objective function to track evaluations and history.

        Args:
            objective_func: Original objective function

        Returns:
            Wrapped objective function
        """

        def wrapped(x: np.ndarray) -> float:
            # Evaluate objective
            value = objective_func(x)

            # Track evaluation
            self._n_evaluations += 1

            # Update history
            param_dict = {name: float(val) for name, val in zip(self.parameter_names, x)}
            self.history.append({"parameters": param_dict, "objective": float(value), "iteration": self._n_evaluations})

            # Track best
            if value < self._best_value:
                self._best_value = value
                if self.verbose:
                    param_str = ", ".join([f"{name}={val:.4f}" for name, val in param_dict.items()])
                    print(f"  Iteration {self._n_evaluations}: f={value:.6f} | {param_str}")

            return value

        return wrapped

    def _reset_tracking(self):
        """Reset history and counters."""
        self.history = []
        self._best_value = float("inf")
        self._n_evaluations = 0

    def _check_bounds(self, x: np.ndarray) -> bool:
        """Check if parameters are within bounds."""
        return np.all(x >= self.bounds_array[:, 0]) and np.all(x <= self.bounds_array[:, 1])

    def _project_to_bounds(self, x: np.ndarray) -> np.ndarray:
        """Project parameters to bounds."""
        return np.clip(x, self.bounds_array[:, 0], self.bounds_array[:, 1])


class GradientFreeOptimizer(Optimizer):
    """Base class for gradient-free optimization methods."""

    pass


class GradientBasedOptimizer(Optimizer):
    """Base class for gradient-based optimization methods."""

    pass


class DifferentialEvolutionOptimizer(GradientFreeOptimizer):
    """
    Differential Evolution optimizer.

    Global optimization using evolutionary algorithm. Good for multimodal
    problems with many local minima. Recommended for initial calibration.

    Attributes:
        population_size: Population size (default: 15)
        strategy: DE strategy (default: 'best1bin')
        mutation: Mutation constant (default: (0.5, 1.0))
        recombination: Crossover probability (default: 0.7)
        seed: Random seed for reproducibility

    Example:
        >>> optimizer = DifferentialEvolutionOptimizer(
        ...     bounds={"k_dis": (0.3, 0.8), "Y_su": (0.05, 0.15)},
        ...     population_size=20,
        ...     max_iterations=100
        ... )
    """

    def __init__(
        self,
        bounds: Dict[str, Tuple[float, float]],
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        verbose: bool = True,
        population_size: int = 15,
        strategy: str = "best1bin",
        mutation: Tuple[float, float] = (0.5, 1.0),
        recombination: float = 0.7,
        seed: Optional[int] = None,
    ):
        """Initialize Differential Evolution optimizer."""
        super().__init__(bounds, max_iterations, tolerance, verbose)

        self.population_size = population_size
        self.strategy = strategy
        self.mutation = mutation
        self.recombination = recombination
        self.seed = seed

    def optimize(
        self, objective_func: Callable[[np.ndarray], float], initial_guess: Optional[np.ndarray] = None
    ) -> OptimizationResult:
        """
        Run differential evolution optimization.

        Args:
            objective_func: Objective function to minimize
            initial_guess: Not used (DE generates initial population)

        Returns:
            OptimizationResult
        """
        if self.verbose:
            print("Starting Differential Evolution optimization")
            print(f"  Population size: {self.population_size}")
            print(f"  Max iterations: {self.max_iterations}")

        # Reset tracking
        self._reset_tracking()

        # Wrap objective for tracking
        wrapped_objective = self._wrap_objective(objective_func)

        # Run optimization
        start_time = time.time()

        result = differential_evolution(
            func=wrapped_objective,
            bounds=self.bounds_array,
            strategy=self.strategy,
            maxiter=self.max_iterations,
            popsize=self.population_size,
            tol=self.tolerance,
            mutation=self.mutation,
            recombination=self.recombination,
            seed=self.seed,
            disp=False,  # We handle our own display
            polish=True,  # Local refinement at end
        )

        execution_time = time.time() - start_time

        if self.verbose:
            print(f"\nOptimization complete in {execution_time:.1f}s")
            print(f"  Success: {result.success}")
            print(f"  Objective: {result.fun:.6f}")
            print(f"  Iterations: {result.nit}")
            print(f"  Function evaluations: {result.nfev}")

        return OptimizationResult.from_scipy_result(result, self.parameter_names, execution_time, self.history)


class ParticleSwarmOptimizer(GradientFreeOptimizer):
    """
    Particle Swarm Optimization.

    Swarm intelligence algorithm. Alternative to DE, sometimes faster
    convergence but may get stuck in local minima.

    Note: Requires pyswarm package (not in base dependencies).
    """

    def __init__(
        self,
        bounds: Dict[str, Tuple[float, float]],
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        verbose: bool = True,
        swarm_size: int = 20,
        omega: float = 0.5,  # Inertia
        phip: float = 0.5,  # Personal best weight
        phig: float = 0.5,  # Global best weight
    ):
        """Initialize Particle Swarm optimizer."""
        super().__init__(bounds, max_iterations, tolerance, verbose)

        self.swarm_size = swarm_size
        self.omega = omega
        self.phip = phip
        self.phig = phig

    def optimize(
        self, objective_func: Callable[[np.ndarray], float], initial_guess: Optional[np.ndarray] = None
    ) -> OptimizationResult:
        """Run particle swarm optimization."""
        try:
            from pyswarm import pso
        except ImportError:
            raise ImportError("Particle Swarm requires 'pyswarm' package: pip install pyswarm")

        if self.verbose:
            print("Starting Particle Swarm optimization")

        self._reset_tracking()
        wrapped_objective = self._wrap_objective(objective_func)

        start_time = time.time()

        lb = self.bounds_array[:, 0]
        ub = self.bounds_array[:, 1]

        xopt, fopt = pso(
            wrapped_objective,
            lb,
            ub,
            swarmsize=self.swarm_size,
            omega=self.omega,
            phip=self.phip,
            phig=self.phig,
            maxiter=self.max_iterations,
            debug=self.verbose,
        )

        execution_time = time.time() - start_time

        # Create result in OptimizeResult format
        result = OptimizeResult(
            x=xopt,
            fun=fopt,
            success=True,
            nit=self.max_iterations,
            nfev=len(self.history),
            message="Optimization terminated successfully",
        )

        return OptimizationResult.from_scipy_result(result, self.parameter_names, execution_time, self.history)


class NelderMeadOptimizer(GradientFreeOptimizer):
    """
    Nelder-Mead simplex optimizer.

    Local optimization method. Fast but may not find global optimum.
    Good for online calibration and fine-tuning.

    Example:
        >>> optimizer = NelderMeadOptimizer(
        ...     bounds={"k_dis": (0.45, 0.55)},
        ...     max_iterations=50
        ... )
    """

    def __init__(
        self,
        bounds: Dict[str, Tuple[float, float]],
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        verbose: bool = True,
        adaptive: bool = True,
    ):
        """Initialize Nelder-Mead optimizer."""
        super().__init__(bounds, max_iterations, tolerance, verbose)
        self.adaptive = adaptive

    def optimize(
        self, objective_func: Callable[[np.ndarray], float], initial_guess: Optional[np.ndarray] = None
    ) -> OptimizationResult:
        """
        Run Nelder-Mead optimization.

        Args:
            objective_func: Objective function
            initial_guess: Initial parameter guess (required for Nelder-Mead)

        Returns:
            OptimizationResult
        """
        if initial_guess is None:
            # Use midpoint of bounds
            initial_guess = np.mean(self.bounds_array, axis=1)

        if self.verbose:
            print("Starting Nelder-Mead optimization")

        self._reset_tracking()

        # Nelder-Mead doesn't strictly enforce bounds, so we add penalty
        def penalized_objective(x):
            if not self._check_bounds(x):
                # Outside bounds - return large penalty
                return 1e10
            return objective_func(x)

        wrapped_objective = self._wrap_objective(penalized_objective)

        start_time = time.time()

        result = minimize(
            fun=wrapped_objective,
            x0=initial_guess,
            method="Nelder-Mead",
            options={
                "maxiter": self.max_iterations,
                "xatol": self.tolerance,
                "fatol": self.tolerance,
                "adaptive": self.adaptive,
                "disp": False,
            },
        )

        execution_time = time.time() - start_time

        if self.verbose:
            print(f"\nOptimization complete in {execution_time:.1f}s")
            print(f"  Success: {result.success}")
            print(f"  Objective: {result.fun:.6f}")

        return OptimizationResult.from_scipy_result(result, self.parameter_names, execution_time, self.history)


class PowellOptimizer(GradientFreeOptimizer):
    """
    Powell's conjugate direction method.

    Local optimization without gradient. Good alternative to Nelder-Mead.
    """

    def optimize(
        self, objective_func: Callable[[np.ndarray], float], initial_guess: Optional[np.ndarray] = None
    ) -> OptimizationResult:
        """Run Powell optimization."""
        if initial_guess is None:
            initial_guess = np.mean(self.bounds_array, axis=1)

        if self.verbose:
            print("Starting Powell optimization")

        self._reset_tracking()

        def penalized_objective(x):
            if not self._check_bounds(x):
                return 1e10
            return objective_func(x)

        wrapped_objective = self._wrap_objective(penalized_objective)

        start_time = time.time()

        result = minimize(
            fun=wrapped_objective,
            x0=initial_guess,
            method="Powell",
            options={"maxiter": self.max_iterations, "ftol": self.tolerance, "disp": False},
        )

        execution_time = time.time() - start_time

        return OptimizationResult.from_scipy_result(result, self.parameter_names, execution_time, self.history)


class LBFGSBOptimizer(GradientBasedOptimizer):
    """
    L-BFGS-B optimizer (gradient-based with bounds).

    Fast gradient-based method with box constraints. Requires smooth
    objective function. Good for well-behaved problems.

    Example:
        >>> optimizer = LBFGSBOptimizer(
        ...     bounds={"k_dis": (0.3, 0.8)},
        ...     max_iterations=100
        ... )
    """

    def __init__(
        self,
        bounds: Dict[str, Tuple[float, float]],
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        verbose: bool = True,
        gtol: float = 1e-5,
    ):
        """Initialize L-BFGS-B optimizer."""
        super().__init__(bounds, max_iterations, tolerance, verbose)
        self.gtol = gtol

    def optimize(
        self, objective_func: Callable[[np.ndarray], float], initial_guess: Optional[np.ndarray] = None
    ) -> OptimizationResult:
        """Run L-BFGS-B optimization."""
        if initial_guess is None:
            initial_guess = np.mean(self.bounds_array, axis=1)

        if self.verbose:
            print("Starting L-BFGS-B optimization")

        self._reset_tracking()
        wrapped_objective = self._wrap_objective(objective_func)

        start_time = time.time()

        result = minimize(
            fun=wrapped_objective,
            x0=initial_guess,
            method="L-BFGS-B",
            bounds=self.bounds_array,
            options={"maxiter": self.max_iterations, "ftol": self.tolerance, "gtol": self.gtol, "disp": False},
        )

        execution_time = time.time() - start_time

        if self.verbose:
            print(f"\nOptimization complete in {execution_time:.1f}s")

        return OptimizationResult.from_scipy_result(result, self.parameter_names, execution_time, self.history)


class SLSQPOptimizer(GradientBasedOptimizer):
    """
    Sequential Least Squares Programming.

    Gradient-based method supporting equality and inequality constraints.
    More flexible than L-BFGS-B but slower.
    """

    def __init__(
        self,
        bounds: Dict[str, Tuple[float, float]],
        max_iterations: int = 100,
        tolerance: float = 1e-6,
        verbose: bool = True,
        constraints: Optional[List] = None,
    ):
        """Initialize SLSQP optimizer."""
        super().__init__(bounds, max_iterations, tolerance, verbose)
        self.constraints = constraints or []

    def optimize(
        self, objective_func: Callable[[np.ndarray], float], initial_guess: Optional[np.ndarray] = None
    ) -> OptimizationResult:
        """Run SLSQP optimization."""
        if initial_guess is None:
            initial_guess = np.mean(self.bounds_array, axis=1)

        if self.verbose:
            print("Starting SLSQP optimization")

        self._reset_tracking()
        wrapped_objective = self._wrap_objective(objective_func)

        start_time = time.time()

        result = minimize(
            fun=wrapped_objective,
            x0=initial_guess,
            method="SLSQP",
            bounds=self.bounds_array,
            constraints=self.constraints,
            options={"maxiter": self.max_iterations, "ftol": self.tolerance, "disp": False},
        )

        execution_time = time.time() - start_time

        return OptimizationResult.from_scipy_result(result, self.parameter_names, execution_time, self.history)


# Factory function for creating optimizers
def create_optimizer(
    method: str, bounds: Dict[str, Tuple[float, float]], max_iterations: int = 100, verbose: bool = True, **kwargs
) -> Optimizer:
    """
    Factory function to create optimizer instances.

    Args:
        method: Optimization method name
        bounds: Parameter bounds
        max_iterations: Maximum iterations
        verbose: Enable output
        **kwargs: Additional method-specific arguments

    Returns:
        Optimizer instance

    Example:
        >>> optimizer = create_optimizer(
        ...     method="differential_evolution",
        ...     bounds={"k_dis": (0.3, 0.8)},
        ...     population_size=20
        ... )
    """
    method = method.lower().replace("-", "_").replace(" ", "_")

    optimizer_map = {
        "differential_evolution": DifferentialEvolutionOptimizer,
        "de": DifferentialEvolutionOptimizer,
        "particle_swarm": ParticleSwarmOptimizer,
        "pso": ParticleSwarmOptimizer,
        "nelder_mead": NelderMeadOptimizer,
        "nm": NelderMeadOptimizer,
        "powell": PowellOptimizer,
        "lbfgsb": LBFGSBOptimizer,
        "l_bfgs_b": LBFGSBOptimizer,
        "slsqp": SLSQPOptimizer,
    }

    if method not in optimizer_map:
        available = ", ".join(optimizer_map.keys())
        raise ValueError(f"Unknown optimization method: {method}. Available: {available}")

    optimizer_class = optimizer_map[method]
    return optimizer_class(bounds=bounds, max_iterations=max_iterations, verbose=verbose, **kwargs)
