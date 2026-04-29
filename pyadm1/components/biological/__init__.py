"""
Biological process components.

Modules:
    digester:  ADM1da fermenter component — usable as a primary
               digester, hydrolysis pre-tank, post-fermenter, or digestate
               storage simply by tuning V_liq, T_ad and the substrate feed.
    separator: Solid–liquid separation component for digestate processing.

Example:
    >>> from pyadm1.components.biological import Digester, Separator
    >>> from pyadm1 import Feedstock
    >>>
    >>> fs = Feedstock(["maize_silage_milk_ripeness", "swine_manure"], feeding_freq=24)
    >>> digester = Digester("dig1", fs, V_liq=1200, V_gas=216, T_ad=315.15)
    >>> separator = Separator("sep1", separation_efficiency=0.95)
"""

from .digester import Digester
from .separator import Separator

__all__ = [
    "Digester",
    "Separator",
]
