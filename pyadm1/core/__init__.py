# pyadm1/core/__init__.py
"""
Core ADM1da implementation.

Modules:
    adm1:        ADM1da model — 41-state agricultural extension of ADM1 (Schlattmann 2011).
    adm_params:  ADMParams — stoichiometric, kinetic, gas, and inhibition constants.
    solver:      ODE solver wrappers (BDF, adaptive).

Example:
    >>> from pyadm1.core import ADM1
    >>> from pyadm1 import Feedstock
    >>> fs = Feedstock(["maize_silage_milk_ripeness", "swine_manure"], feeding_freq=24)
    >>> adm = ADM1(fs, V_liq=1200, V_gas=216, T_ad=315.15)
    >>> adm.set_influent_dataframe(fs.get_influent_dataframe(Q=[11.4, 6.1]))
    >>> adm.create_influent(Q=[11.4, 6.1], i=0)
"""

from .adm1 import ADM1, INFLUENT_COLUMNS, STATE_SIZE, get_state_zero_from_csv
from .adm_params import ADMParams
from .solver import (
    AdaptiveODESolver,
    ODESolver,
    SolverConfig,
    create_solver,
)

__all__ = [
    "ADM1",
    "INFLUENT_COLUMNS",
    "STATE_SIZE",
    "ADMParams",
    "AdaptiveODESolver",
    "ODESolver",
    "SolverConfig",
    "create_solver",
    "get_state_zero_from_csv",
]
