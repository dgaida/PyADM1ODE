"""
Energy Conversion and Storage Components

Components for energy generation, conversion, and storage in biogas plants.

Modules:
    chp: Combined Heat and Power (CHP) units including gas engines and micro-turbines,
        with electrical and thermal efficiency curves, part-load behavior, maintenance
        scheduling, and emissions calculation.

    boiler: Auxiliary heating systems (gas, oil, biomass boilers) for peak heat demand
           and backup heating, with efficiency curves, fuel consumption, and emission
           factors.

    gas_storage: Biogas storage systems including low-pressure (membrane, dome) and
                high-pressure (compressed gas) storage, with pressure control, safety
                management, and capacity utilization.

    flare: Safety gas combustion system for excess biogas or emergency situations,
          with destruction efficiency, emissions calculation, and automatic ignition
          control.

Example:
    >>> from pyadm1.components.energy import CHP, Boiler, GasStorage, Flare
    >>>
    >>> # CHP unit with 500 kW electrical output
    >>> chp = CHP("chp1", P_el_nom=500, eta_el=0.40, eta_th=0.45,
    ...          type="gas_engine")
    >>>
    >>> # Low-pressure gas storage (membrane roof)
    >>> storage = GasStorage("storage1", volume=1000,
    ...                     storage_type="membrane", p_max=0.015)
    >>>
    >>> # Emergency flare
    >>> flare = Flare("flare1", capacity=500, destruction_efficiency=0.98)
"""

from pyadm1.components.energy.chp import CHP
from pyadm1.components.energy.boiler import Boiler
from pyadm1.components.energy.gas_storage import GasStorage
from pyadm1.components.energy.flare import Flare

__all__ = [
    "CHP",
    "Boiler",
    "GasStorage",
    "Flare",
]
