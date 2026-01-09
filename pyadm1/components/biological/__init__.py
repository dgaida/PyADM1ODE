"""
Biological Process Components

Components for biological conversion processes in biogas plants.

Modules:

    digester: Main fermenter component implementing ADM1 model for anaerobic digestion,
             supporting single or multiple fermenters in series/parallel, with flexible
             volume, temperature, and retention time configuration.

    hydrolysis: Pre-treatment tank for hydrolysis-dominated processes, useful for
               substrates with high lignocellulosic content, can operate at different
               temperatures and retention times than main digester.

    separator: Solid-liquid separation component for digestate processing, models
              mechanical (screw press, centrifuge) or gravitational separation with
              configurable separation efficiency and dry matter content.

Example:

    >>> from pyadm1.components.biological import Digester, Hydrolysis, Separator
    >>> from pyadm1.substrates import Feedstock
    >>>
    >>> feedstock = Feedstock(feeding_freq=48)
    >>>
    >>> # Two-stage digestion with hydrolysis pre-treatment
    >>> hydrolysis = Hydrolysis("hydro1", feedstock, V_liq=500, T_ad=318.15)
    >>> digester = Digester("dig1", feedstock, V_liq=2000, T_ad=308.15)
    >>> separator = Separator("sep1", separation_efficiency=0.95)
"""

from .digester import Digester
from .hydrolysis import Hydrolysis
from .separator import Separator

__all__ = [
    "Digester",
    "Hydrolysis",
    "Separator",
]
