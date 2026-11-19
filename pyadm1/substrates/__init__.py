"""
Substrate Management and Characterization (ADM1da).

Pure-Python substrate characterization for ADM1da.  Converts Weender analysis +
pH/FFS/NH4 data into the 38-column influent DataFrame expected by ADM1.

Substrates may also be described by the measured parameters of the legacy
Gummersbach library (van Soest analysis, degradation level, single acids,
titrated acid capacity); see :mod:`pyadm1.substrates.gummersbach`.

Example:
    >>> from pyadm1.substrates import Feedstock, SubstrateRegistry
    >>> reg = SubstrateRegistry()
    >>> fs = Feedstock(reg.get("maize_silage_milk_ripeness"), feeding_freq=48)
    >>> df = fs.get_influent_dataframe(Q=15.0)
"""

from .feedstock import (
    DEFAULT_BMP_RECOVERY,
    Feedstock,
    SubstrateParams,
    SubstrateRegistry,
    fit_degradability_to_bmp,
    load_substrate,
    load_substrate_toml,
    load_substrate_xml,
    load_substrate_yaml,
)
from .gummersbach import (
    MEASURED_PARAMETERS,
    gummersbach_substrate,
    load_gummersbach_library,
    params_from_measured,
)

__all__ = [
    "DEFAULT_BMP_RECOVERY",
    "MEASURED_PARAMETERS",
    "Feedstock",
    "SubstrateParams",
    "SubstrateRegistry",
    "fit_degradability_to_bmp",
    "gummersbach_substrate",
    "load_gummersbach_library",
    "load_substrate",
    "load_substrate_toml",
    "load_substrate_xml",
    "load_substrate_yaml",
    "params_from_measured",
]
