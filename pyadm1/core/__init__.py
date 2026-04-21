# pyadm1/core/__init__.py
"""
Core ADM model implementations (ADM1 and ADM1da).

This module contains the core implementations of:

  ADM1   -- Standard 37-state Anaerobic Digestion Model No. 1 (Batstone et al. 2002)
             as a pure ODE system adapted for agricultural biogas plants.

  ADM1da -- SIMBA# biogas 41-state extension (Schlattmann 2011) featuring a
             two-pool sub-fraction disintegration approach (XPS slow, XPF fast),
             temperature-dependent kinetics, and modified inhibition kinetics.

Modules:
    adm_base:      Abstract base class shared by all ADM variants.
    adm1:          ADM1 (37 states) — standard implementation.
    adm1da:        ADM1da (41 states) — SIMBA# biogas extension.
    adm_params:    ADM1 parameter class (stoichiometric, kinetic, physico-chemical).
    adm1da_params: ADM1da parameter class (temperature factors, modified inhibition).
    adm_equations: Reusable process-rate and inhibition functions.
    solver:        ODE solver wrappers (BDF, adaptive).

Example (ADM1):
    >>> from pyadm1.core import ADM1, ADMParams, create_solver
    >>> from pyadm1.substrates import Feedstock
    >>> feedstock = Feedstock(feeding_freq=48)
    >>> adm1 = ADM1(feedstock, V_liq=2000, T_ad=308.15)

Example (ADM1da):
    >>> from pyadm1.core import ADM1da
    >>> da = ADM1da(feedstock, V_liq=2000, T_ad=308.15)
    >>> da.set_influent_dataframe(influent_df)  # 38-column DataFrame
    >>> da.create_influent([15.0, 10.0], i=0)
"""

from .adm_base import ADMBase
from .adm1 import ADM1, get_state_zero_from_initial_state
from .adm1da import ADM1da, get_state_zero_from_csv, INFLUENT_COLUMNS
from .adm1da_params import ADM1daParams
from .adm_params import ADMParams
from .adm_equations import (
    InhibitionFunctions,
    ProcessRates,
    AcidBaseKinetics,
    GasTransfer,
    BiochemicalProcesses,
)
from .solver import (
    ODESolver,
    AdaptiveODESolver,
    SolverConfig,
    create_solver,
)

__all__ = [
    # Base class
    "ADMBase",
    # ADM1 class
    "ADM1",
    "get_state_zero_from_initial_state",
    # ADM1da class
    "ADM1da",
    "get_state_zero_from_csv",
    "INFLUENT_COLUMNS",
    # Parameters
    "ADMParams",
    "ADM1daParams",
    # Process equations
    "InhibitionFunctions",
    "ProcessRates",
    "AcidBaseKinetics",
    "GasTransfer",
    "BiochemicalProcesses",
    # Solvers
    "ODESolver",
    "AdaptiveODESolver",
    "SolverConfig",
    "create_solver",
]
