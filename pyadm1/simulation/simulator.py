# pyadm1/simulation/simulator.py
"""
Simulator class for running ADM1 simulations with different scenarios.

This class orchestrates ADM1 simulations using the refactored solver interface,
providing methods for single simulations and multi-scenario optimization.

Example:
    >>> from pyadm1.core import ADM1
    >>> from pyadm1.simulation import Simulator
    >>> from pyadm1.substrates import Feedstock
    >>>
    >>> feedstock = Feedstock(feeding_freq=48)
    >>> adm1 = ADM1(feedstock)
    >>> simulator = Simulator(adm1)
    >>>
    >>> # Run simulation
    >>> state = [0.01] * 37
    >>> final_state = simulator.simulate_AD_plant([0, 1], state)
"""

import numpy as np
from typing import List, Tuple

from pyadm1.core.adm1 import ADM1
from pyadm1.core.solver import create_solver, ODESolver


class Simulator:
    """
    Handles ADM1 simulation runs with various configurations.

    This class provides high-level interfaces for running ADM1 simulations,
    including single runs and multi-scenario optimization for substrate feed
    determination.

    Attributes:
        adm1: ADM1 model instance
        solver: ODE solver instance

    Example:
        >>> simulator = Simulator(adm1)
        >>> result = simulator.simulate_AD_plant([0, 10], initial_state)
    """

    def __init__(self, adm1: ADM1, solver: ODESolver = None) -> None:
        """
        Initialize simulator with ADM1 model instance.

        Args:
            adm1: ADM1 model instance
            solver: Optional ODE solver. If None, creates default BDF solver
        """
        self._adm1 = adm1
        self._solver = solver or create_solver(method="BDF", rtol=1e-6, atol=1e-8)

    def determine_best_feed_by_n_sims(
        self,
        state_zero: List[float],
        Q: List[float],
        Qch4sp: float,
        feeding_freq: int,
        n: int = 13,
    ) -> Tuple[float, float, List[float], float, float, List[float], float, float, float, float]:
        """
        Determine optimal substrate feed by running n simulations.

        Runs n simulations with varying substrate feed rates around Q and
        returns the feed rate yielding methane production closest to setpoint.

        The first simulation uses Q, the 2nd and 3rd use Q ± 1.5 m³/d,
        and remaining simulations use random variations.

        Args:
            state_zero: Initial ADM1 state vector (37 elements)
            Q: Initial volumetric flow rates [m³/d], e.g. [15, 10, 0, ...]
            Qch4sp: Methane flow rate setpoint [m³/d]
            feeding_freq: Feeding frequency [hours]
            n: Number of simulations to run (default: 13, minimum: 3)

        Returns:
            Tuple containing:
                - Q_Gas_7d_best: Best biogas production after 7 days [m³/d]
                - Q_CH4_7d_best: Best methane production after 7 days [m³/d]
                - Qbest: Best substrate feed rates [m³/d]
                - Q_Gas_7d_initial: Initial biogas production after 7 days [m³/d]
                - Q_CH4_7d_initial: Initial methane production after 7 days [m³/d]
                - Q_initial: Initial substrate feed rates [m³/d]
                - q_gas_best_2d: Best biogas after feeding_freq/24 days [m³/d]
                - q_ch4_best_2d: Best methane after feeding_freq/24 days [m³/d]
                - q_gas_2d: Initial biogas after feeding_freq/24 days [m³/d]
                - q_ch4_2d: Initial methane after feeding_freq/24 days [m³/d]

        Example:
            >>> result = simulator.determine_best_feed_by_n_sims(
            ...     state, [15, 10, 0, 0, 0, 0, 0, 0, 0, 0], 900, 48, n=13
            ... )
            >>> Q_best = result[2]
        """
        if n < 3:
            raise ValueError("n must be at least 3")

        Q_CH4_7d = [0.0] * n
        Q_Gas_7d = [0.0] * n

        # Generate substrate feed mixtures
        Qnew = self._adm1.feedstock.get_substrate_feed_mixtures(Q, n)

        # Run n simulations
        for i, q in enumerate(Qnew):
            Q_Gas_7d[i], Q_CH4_7d[i] = self._simulate_without_saving_state([0, 7], state_zero, q)

        # Find scenario closest to methane setpoint
        ii = np.argmin([np.sum((q - Qch4sp) ** 2) for q in Q_CH4_7d])
        Qbest = Qnew[ii]

        # Simulate with initial Q for feeding_freq/24 days
        q_gas_2d, q_ch4_2d = self._simulate_without_saving_state([0, feeding_freq / 24], state_zero, Qnew[0])

        # Simulate with best Q for feeding_freq/24 days
        q_gas_best_2d, q_ch4_best_2d = self._simulate_without_saving_state([0, feeding_freq / 24], state_zero, Qbest)

        return (
            Q_Gas_7d[ii][-1] if hasattr(Q_Gas_7d[ii], "__iter__") else Q_Gas_7d[ii],
            Q_CH4_7d[ii][-1] if hasattr(Q_CH4_7d[ii], "__iter__") else Q_CH4_7d[ii],
            Qbest,
            Q_Gas_7d[0][-1] if hasattr(Q_Gas_7d[0], "__iter__") else Q_Gas_7d[0],
            Q_CH4_7d[0][-1] if hasattr(Q_CH4_7d[0], "__iter__") else Q_CH4_7d[0],
            Qnew[0],
            q_gas_best_2d[-1] if hasattr(q_gas_best_2d, "__iter__") else q_gas_best_2d,
            q_ch4_best_2d[-1] if hasattr(q_ch4_best_2d, "__iter__") else q_ch4_best_2d,
            q_gas_2d[-1] if hasattr(q_gas_2d, "__iter__") else q_gas_2d,
            q_ch4_2d[-1] if hasattr(q_ch4_2d, "__iter__") else q_ch4_2d,
        )

    def simulate_AD_plant(self, tstep: List[float], state_zero: List[float]) -> List[float]:
        """
        Simulate ADM1 for specified time span and return final state.

        This is the main simulation method that integrates the ADM1 ODEs
        and tracks process values for operator information.

        Args:
            tstep: Time span [t_start, t_end] in days
            state_zero: Initial ADM1 state vector (37 elements)

        Returns:
            Final ADM1 state vector after simulation (37 elements)

        Example:
            >>> final_state = simulator.simulate_AD_plant([0, 1], initial_state)
            >>> print(f"Final pH: {final_state[...])
        """
        final_state = self._simulate_and_return_final_state(tstep, state_zero)

        # Print process parameters for monitoring
        self._adm1.print_params_at_current_state(final_state)

        return final_state

    def _simulate_without_saving_state(
        self, tstep: List[float], state_zero: List[float], Q: List[float]
    ) -> Tuple[float, float]:
        """
        Run simulation without saving final state.

        Used internally for optimization scenarios where we only need
        gas production values.

        Args:
            tstep: Time span [t_start, t_end] in days
            state_zero: Initial ADM1 state vector
            Q: Volumetric flow rates [m³/d]

        Returns:
            Tuple of (q_gas, q_ch4) - biogas and methane production rates [m³/d]
        """
        # Create ADM1 input stream
        self._adm1.create_influent(Q, 0)

        # Integrate ODEs
        result = self._solver.solve(fun=self._adm1.ADM1_ODE, t_span=tstep, y0=state_zero)

        # Extract final gas phase state
        final_state = result.y[:, -1]
        pi_Sh2, pi_Sch4, pi_Sco2, pTOTAL = final_state[33:37]

        # Calculate gas production rates
        q_gas, q_ch4, q_co2, p_gas = self._adm1.calc_gas(pi_Sh2, pi_Sch4, pi_Sco2, pTOTAL)

        return q_gas, q_ch4

    def _simulate_and_return_final_state(self, tstep: List[float], state_zero: List[float]) -> List[float]:
        """
        Simulate and return final state vector.

        Args:
            tstep: Time span [t_start, t_end] in days
            state_zero: Initial ADM1 state vector

        Returns:
            Final ADM1 state vector (37 elements)
        """
        # Integrate ODEs using solver
        result = self._solver.solve(fun=self._adm1.ADM1_ODE, t_span=tstep, y0=state_zero)

        # Extract final state
        final_state = result.y[:, -1].tolist()

        return final_state

    @property
    def adm1(self) -> ADM1:
        """ADM1 model instance."""
        return self._adm1

    @property
    def solver(self) -> ODESolver:
        """ODE solver instance."""
        return self._solver
