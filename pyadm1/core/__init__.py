"""
Core ADM1 Implementation

This module contains the core implementation of the Anaerobic Digestion Model No. 1
(ADM1) as a pure ODE system without DAEs, specifically adapted for agricultural
biogas plants.

Modules:
    adm1: Main ADM1 class implementing the complete ODE system with 37 state variables,
          including methods for creating influent streams, calculating gas production,
          and managing simulation state.

    adm_params: Static parameter class providing all stoichiometric, kinetic, and
               physical-chemical parameters for ADM1, including temperature-dependent
               parameters and pH inhibition factors.

    adm_equations: Process rate equations, inhibition functions, and biochemical
                  transformations used in the ADM1 model, separated for clarity
                  and easier modification.

    solver: ODE solver wrapper providing interface to scipy solvers with
           appropriate settings for stiff systems (BDF method), time step
           management, and result handling.

Example:
    >>> from pyadm1.core import ADM1, ADMParams
    >>> from pyadm1.substrates import Feedstock
    >>>
    >>> feedstock = Feedstock(feeding_freq=48)
    >>> adm1 = ADM1(feedstock, V_liq=2000, T_ad=308.15)
    >>> params = ADMParams.get_all_params(R=0.08314, T_base=298.15, T_ad=308.15)
"""

from pyadm1.core.adm1 import ADM1, get_state_zero_from_initial_state
from pyadm1.core.adm_params import ADMParams
from pyadm1.core.adm_equations import (
    InhibitionFunctions,
    ProcessRates,
    AcidBaseKinetics,
    GasTransfer,
)
from pyadm1.core.solver import ODESolver

__all__ = [
    "ADM1",
    "get_state_zero_from_initial_state",
    "ADMParams",
    "InhibitionFunctions",
    "ProcessRates",
    "AcidBaseKinetics",
    "GasTransfer",
    "ODESolver",
]
