"""
Substrate Management and Characterization (ADM1da).

Pure-Python substrate characterization for ADM1da.  Converts Weender analysis +
pH/FFS/NH4 data into the 38-column influent DataFrame expected by ADM1.

Example:
    >>> from pyadm1.substrates import Feedstock, SubstrateRegistry
    >>> reg = SubstrateRegistry()
    >>> fs = Feedstock(reg.get("maize_silage_milk_ripeness"), feeding_freq=48)
    >>> df = fs.get_influent_dataframe(Q=15.0)
"""

from .feedstock import (
    Feedstock,
    SubstrateParams,
    SubstrateRegistry,
    load_substrate_xml,
)

__all__ = [
    "Feedstock",
    "SubstrateParams",
    "SubstrateRegistry",
    "load_substrate_xml",
]
