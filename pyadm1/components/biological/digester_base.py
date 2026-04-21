# ============================================================================
# pyadm1/components/biological/digester_base.py
# ============================================================================
"""
Abstract base class for digester components.

Defines the interface and shared state that concrete digester wrappers
(:class:`pyadm1.components.biological.Digester` for legacy ADM1,
:class:`pyadm1.components.biological.ADM1daDigester` for the SIMBA# ADM1da
extension) must provide.  Plant-level code can treat either concrete class
uniformly via ``isinstance(obj, DigesterBase)``.

Subclasses are responsible for:

* creating the underlying biological model instance (``ADM1`` or ``ADM1da``)
* implementing :meth:`step`, :meth:`initialize`, and :meth:`from_dict`
"""

from typing import Any, Dict, List, Optional

from ..base import Component, ComponentType
from ..energy import GasStorage


class DigesterBase(Component):
    """
    Abstract base for digester components.

    Encapsulates the portions of construction that are identical across the
    ADM1 and ADM1da back-ends: geometry/temperature storage, substrate feed
    buffer, empty state vector, and the per-digester :class:`GasStorage`.

    Attributes
    ----------
    feedstock : Feedstock or ADM1daFeedstock
        Feedstock object used to derive substrate influent.
    V_liq, V_gas, T_ad : float
        Reactor liquid volume [m³], gas headspace [m³] and temperature [K].
    adm1_state : list of float
        Current biological state vector.  Length depends on the back-end
        (37 for ADM1, 41 for ADM1da).
    Q_substrates : list of float
        Substrate feed rates [m³/d], one entry per substrate slot.
    gas_storage : GasStorage
        Low-pressure membrane storage sized from ``V_gas``.
    """

    def __init__(
        self,
        component_id: str,
        feedstock,
        V_liq: float,
        V_gas: float,
        T_ad: float,
        name: Optional[str] = None,
    ):
        super().__init__(component_id, ComponentType.DIGESTER, name)

        self.feedstock = feedstock
        self.V_liq = V_liq
        self.V_gas = V_gas
        self.T_ad = T_ad

        self.gas_storage: GasStorage = GasStorage(
            component_id=f"{self.component_id}_storage",
            storage_type="membrane",
            capacity_m3=max(50.0, float(self.V_gas)),
            p_min_bar=0.95,
            p_max_bar=1.05,
            initial_fill_fraction=0.1,
            name=f"{self.name} Gas Storage",
        )

        self.adm1_state: List[float] = []
        self.Q_substrates: List[float] = [0.0] * 10

    def to_dict(self) -> Dict[str, Any]:
        return {
            "component_id": self.component_id,
            "component_type": self.component_type.value,
            "name": self.name,
            "V_liq": self.V_liq,
            "V_gas": self.V_gas,
            "T_ad": self.T_ad,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "state": self.state,
        }
