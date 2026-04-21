"""
Substrate Management and Characterization

Modules:
    feedstock:         ADM1 Feedstock class (substrate mixing via C# DLLs).
    adm1da_feedstock:  Pure-Python substrate characterization for ADM1da.
                       Converts Weender analysis + pH/FFS/NH4 data into the
                       38-column influent DataFrame expected by ADM1da.

Example (ADM1):
    >>> from pyadm1.substrates import Feedstock
    >>> feedstock = Feedstock(feeding_freq=48)
    >>> influent_df = feedstock.get_influent_dataframe([15, 10, 0, 0, 0, 0, 0, 0, 0, 0])

Example (ADM1da):
    >>> from pyadm1.substrates import ADM1daFeedstock, SubstrateRegistry
    >>> reg = SubstrateRegistry()
    >>> fs = ADM1daFeedstock(reg.get("maize_silage_milk_ripeness"), feeding_freq=48)
    >>> df = fs.get_influent_dataframe(Q=15.0)
    >>> da.set_influent_dataframe(df)
"""

from .feedstock import Feedstock
from .adm1da_feedstock import (
    ADM1daSubstrateParams,
    ADM1daFeedstock,
    SubstrateRegistry,
    load_substrate_xml,
)

__all__ = [
    # ADM1 (DLL-based)
    "Feedstock",
    # ADM1da (pure Python)
    "ADM1daSubstrateParams",
    "ADM1daFeedstock",
    "SubstrateRegistry",
    "load_substrate_xml",
]
