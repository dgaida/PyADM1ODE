# pyadm1/core/solver.py
"""
ODE solver wrapper for ADM1 simulations.

This module provides a clean interface to scipy's ODE solvers with appropriate
settings for stiff systems (BDF method), time step management, and result handling.

Example:
    >>> from pyadm1.core.solver import create_solver
    >>>
    >>> # Create BDF solver for stiff systems
    >>> solver = create_solver(method='BDF', rtol=1e-6)
    >>>
    >>> # Define ODE system
    >>> def ode_func(t, y):
    ...     return [-0.5 * y[0]]
    >>>
    >>> # Solve
    >>> result = solver.solve(ode_func, (0, 10), [1.0])
"""

import numpy as np
import scipy.integrate
from typing import Callable, List, Tuple, Optional, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    from scipy.integrate._ivp.ivp import OdeResult
else:
    OdeResult = object


@dataclass
class SolverConfig:
    """
    Configuration for ODE solver.

    Attributes:
        method: Integration method ('BDF' for stiff ODEs)
        rtol: Relative tolerance for solver
        atol: Absolute tolerance for solver
        min_step: Minimum allowed time step [days]
        max_step: Maximum allowed time step [days]
        first_step: Initial step size [days]
    """

    method: str = "BDF"
    rtol: float = 1e-6
    atol: float = 1e-8
    min_step: float = 1e-6
    max_step: float = 0.1
    first_step: Optional[float] = None


class ODESolver:
    """
    ODE solver wrapper for ADM1 system.

    Provides a clean interface to scipy's solve_ivp with appropriate settings
    for stiff biogas process ODEs. Uses BDF (Backward Differentiation Formula)
    method which is suitable for stiff systems.

    Example:
        >>> def ode_func(t, y):
        ...     return [-0.5 * y[0], 0.5 * y[0] - 0.1 * y[1]]
        >>> solver = ODESolver()
        >>> result = solver.solve(ode_func, [0, 10], [1.0, 0.0])
        >>> print(result.y[:, -1])  # Final state
    """

    def __init__(self, config: Optional[SolverConfig] = None):
        """
        Initialize ODE solver with configuration.

        Args:
            config: Solver configuration. If None, uses default BDF settings.
        """
        self.config = config or SolverConfig()

    def solve(
        self,
        fun: Callable[[float, List[float]], List[float]],
        t_span: Tuple[float, float],
        y0: List[float],
        t_eval: Optional[np.ndarray] = None,
        dense_output: bool = False,
    ) -> "OdeResult":
        """
        Solve ODE system over time span.

        Args:
            fun: Right-hand side of ODE system dy/dt = fun(t, y)
            t_span: Integration time span (t_start, t_end) [days]
            y0: Initial state vector
            t_eval: Times at which to store solution. If None, uses automatic
                time points with 0.05 day resolution
            dense_output: If True, returns continuous solution object

        Returns:
            OdeResult object with solution (from scipy.integrate.solve_ivp)

        Raises:
            RuntimeError: If integration fails
        """
        # Set default evaluation times if not provided
        if t_eval is None:
            t_eval = np.arange(t_span[0], t_span[1], 0.05)

        # Prepare solver arguments
        solver_args = {
            "method": self.config.method,
            "rtol": self.config.rtol,
            "atol": self.config.atol,
            "dense_output": dense_output,
        }

        # Add optional step size constraints
        if self.config.min_step is not None:
            solver_args["min_step"] = self.config.min_step
        if self.config.max_step is not None:
            solver_args["max_step"] = self.config.max_step
        if self.config.first_step is not None:
            solver_args["first_step"] = self.config.first_step

        try:
            result = scipy.integrate.solve_ivp(fun=fun, t_span=t_span, y0=y0, t_eval=t_eval, **solver_args)

            if not result.success:
                raise RuntimeError(f"ODE integration failed: {result.message}")

            return result

        except Exception as e:
            raise RuntimeError(f"Error during ODE integration: {str(e)}") from e

    def solve_to_steady_state(
        self,
        fun: Callable[[float, List[float]], List[float]],
        y0: List[float],
        max_time: float = 1000.0,
        steady_state_tol: float = 1e-6,
        check_interval: float = 10.0,
    ) -> Tuple[List[float], float, bool]:
        """
        Integrate until steady state is reached or max time exceeded.

        Args:
            fun: Right-hand side of ODE system
            y0: Initial state vector
            max_time: Maximum integration time [days]
            steady_state_tol: Tolerance for steady state detection
            check_interval: Interval for checking steady state [days]

        Returns:
            Tuple of (final_state, final_time, converged)
            - final_state: State vector at end of integration
            - final_time: Time at end of integration [days]
            - converged: True if steady state was reached
        """
        current_time = 0.0
        current_state = y0

        while current_time < max_time:
            # Integrate for check_interval
            t_span = (current_time, current_time + check_interval)
            result = self.solve(fun, t_span, current_state)

            # Get final state
            new_state = result.y[:, -1]

            # Check if steady state reached
            state_change = np.linalg.norm(new_state - current_state)
            state_norm = np.linalg.norm(new_state)

            if state_norm > 0:
                relative_change = state_change / state_norm
                if relative_change < steady_state_tol:
                    return list(new_state), current_time + check_interval, True

            # Update for next iteration
            current_state = new_state
            current_time += check_interval

        # Max time exceeded without reaching steady state
        return list(current_state), current_time, False

    def solve_sequential(
        self, fun: Callable[[float, List[float]], List[float]], t_points: List[float], y0: List[float]
    ) -> List[List[float]]:
        """
        Solve ODE system sequentially through multiple time points.

        Useful for simulations where conditions change at specific times
        (e.g., substrate feed changes).

        Args:
            fun: Right-hand side of ODE system
            t_points: List of time points [days]
            y0: Initial state vector

        Returns:
            List of state vectors at each time point
        """
        states = [y0]
        current_state = y0

        for i in range(len(t_points) - 1):
            t_span = (t_points[i], t_points[i + 1])
            result = self.solve(fun, t_span, current_state)
            current_state = result.y[:, -1].tolist()
            states.append(current_state)

        return states


class AdaptiveODESolver(ODESolver):
    """
    Adaptive ODE solver that adjusts tolerances based on solution behavior.

    Monitors the solution and can tighten tolerances if instabilities are
    detected, or relax them for faster computation when solution is smooth.
    """

    def __init__(
        self, config: Optional[SolverConfig] = None, adaptive: bool = True, min_rtol: float = 1e-8, max_rtol: float = 1e-4
    ):
        """
        Initialize adaptive ODE solver.

        Args:
            config: Base solver configuration
            adaptive: Enable adaptive tolerance adjustment
            min_rtol: Minimum relative tolerance
            max_rtol: Maximum relative tolerance
        """
        super().__init__(config)
        self.adaptive = adaptive
        self.min_rtol = min_rtol
        self.max_rtol = max_rtol
        self._solution_history = []

    def solve(
        self,
        fun: Callable[[float, List[float]], List[float]],
        t_span: Tuple[float, float],
        y0: List[float],
        t_eval: Optional[np.ndarray] = None,
        dense_output: bool = False,
    ) -> "OdeResult":
        """
        Solve with adaptive tolerance adjustment.

        Same interface as parent class but monitors solution quality.
        """
        result = super().solve(fun, t_span, y0, t_eval, dense_output)

        if self.adaptive:
            self._update_tolerances(result)

        return result

    def _update_tolerances(self, result: "OdeResult") -> None:
        """
        Update solver tolerances based on solution behavior.

        Args:
            result: ODE solution result to analyze
        """
        # Calculate solution smoothness (using second derivatives)
        if len(result.t) > 2:
            y = result.y
            t = result.t

            # Estimate second derivatives
            dy_dt = np.gradient(y, t, axis=1)
            d2y_dt2 = np.gradient(dy_dt, t, axis=1)

            # Calculate maximum relative second derivative
            max_curvature = np.max(np.abs(d2y_dt2) / (np.abs(y) + 1e-10))

            # Adjust tolerances based on curvature
            if max_curvature > 1.0:
                # Solution has high curvature, tighten tolerances
                self.config.rtol = max(self.min_rtol, self.config.rtol * 0.5)
                self.config.atol = max(self.config.atol * 0.5, 1e-10)
            elif max_curvature < 0.1 and self.config.rtol < self.max_rtol:
                # Solution is smooth, can relax tolerances
                self.config.rtol = min(self.max_rtol, self.config.rtol * 1.5)
                self.config.atol = min(self.config.atol * 1.5, 1e-6)


def create_solver(method: str = "BDF", adaptive: bool = False, **kwargs) -> ODESolver:
    """
    Factory function to create appropriate solver instance.

    Args:
        method: Integration method ('BDF' recommended for ADM1)
        adaptive: Use adaptive tolerance adjustment
        **kwargs: Additional configuration parameters for SolverConfig

    Returns:
        Configured solver instance

    Example:
        >>> solver = create_solver(method='BDF', rtol=1e-7)
        >>> # Use solver for integration...
    """
    config = SolverConfig(method=method, **kwargs)

    if adaptive:
        return AdaptiveODESolver(config)
    else:
        return ODESolver(config)
