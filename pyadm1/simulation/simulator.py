# pyadm1/simulation/simulator.py
"""
Single-plant simulation driver wrapping the ADM1da model.

The :class:`Simulator` integrates the 41-state ADM1 ODE system over a given
time span using :class:`pyadm1.core.solver.ODESolver`.  It is a thin wrapper
over :func:`scipy.integrate.solve_ivp` for users who want to exercise the ODE
directly without going through the component framework.

For component-based plant simulation (with gas storage, CHP, heating, etc.),
use :class:`pyadm1.BiogasPlant` together with :class:`Digester`.

Example:
    >>> from pyadm1.core import ADM1, STATE_SIZE
    >>> from pyadm1.simulation import Simulator
    >>> from pyadm1 import Feedstock
    >>>
    >>> fs = Feedstock(["maize_silage_milk_ripeness"], feeding_freq=24)
    >>> adm = ADM1(fs, V_liq=1200, V_gas=216, T_ad=315.15)
    >>> adm.set_influent_dataframe(fs.get_influent_dataframe(Q=15.0))
    >>> adm.create_influent(Q=[15.0], i=0)
    >>>
    >>> sim = Simulator(adm)
    >>> state0 = [0.01] * STATE_SIZE
    >>> final_state = sim.simulate_AD_plant([0.0, 1.0], state0)
"""

from typing import List, Tuple

from pyadm1.core.adm1 import ADM1
from pyadm1.core.solver import create_solver, ODESolver


class Simulator:
    """
    Single-plant ADM1 simulation driver.

    Attributes
    ----------
    adm1 : ADM1
        The model instance being driven.
    solver : ODESolver
        ODE solver wrapper (default BDF, rtol=1e-6, atol=1e-8).
    """

    def __init__(self, adm1: ADM1, solver: ODESolver = None) -> None:
        self._adm1 = adm1
        self._solver = solver or create_solver(method="BDF", rtol=1e-6, atol=1e-8)

    def simulate_AD_plant(self, tstep: List[float], state_zero: List[float]) -> List[float]:
        """
        Integrate the ADM1 ODE for the requested time span.

        Parameters
        ----------
        tstep : [t_start, t_end]
            Time span [days].
        state_zero : list of float
            Initial state vector (41 elements).

        Returns
        -------
        list of float
            Final state vector after integration.
        """
        result = self._solver.solve(fun=self._adm1.ADM_ODE, t_span=tstep, y0=state_zero)
        final_state = result.y[:, -1].tolist()

        # Update tracked process indicators (pH, gas) so that downstream code
        # can read them from the ADM1 instance directly.
        self._adm1.print_params_at_current_state(final_state)

        return final_state

    def simulate_gas_production(
        self,
        tstep: List[float],
        state_zero: List[float],
        Q: List[float],
    ) -> Tuple[float, float]:
        """
        Integrate the ODE and return final-state gas production rates.

        Parameters
        ----------
        tstep : [t_start, t_end]
            Time span [days].
        state_zero : list of float
            Initial state vector (41 elements).
        Q : list of float
            Substrate feed rates [m³/d].

        Returns
        -------
        (q_gas, q_ch4) : tuple of float
            Total biogas flow and methane flow [Nm³/d].
        """
        self._adm1.create_influent(Q, 0)
        result = self._solver.solve(fun=self._adm1.ADM_ODE, t_span=tstep, y0=state_zero)
        final_state = result.y[:, -1]
        q_gas, q_ch4, _, _, _ = self._adm1.calc_gas(
            float(final_state[37]),
            float(final_state[38]),
            float(final_state[39]),
            float(final_state[40]),
        )
        return float(q_gas), float(q_ch4)

    @property
    def adm1(self) -> ADM1:
        """ADM1 model instance."""
        return self._adm1

    @property
    def solver(self) -> ODESolver:
        """ODE solver instance."""
        return self._solver
