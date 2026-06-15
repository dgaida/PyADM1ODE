# ============================================================================
# pyadm1/components/energy/biogas_upgrading.py
# ============================================================================
"""
Biogas upgrading unit (Biogasaufbereitungsanlage, BGAA).

Upgrades raw biogas to grid-quality biomethane (Biomethan-Netzeinspeisung).
Models gas balance only — no electrical or thermal outputs.

Gas mass balance per timestep:
    CH4_in        = Q_gas_in * ch4_fraction_in
    CH4_recovered = CH4_in * ch4_recovery
    Q_biomethane  = CH4_recovered / ch4_content_out   [m3/d biomethane at grid spec]
    Q_offgas      = max(0, Q_actual - Q_biomethane)   [CO2-rich reject, to atmosphere]
    Q_gas_out     = max(0, Q_gas_in - capacity_m3d)   [capacity overflow to paired flare]

References:
    - DVGW G 262 (2011): Nutzung von Gasen aus regenerativen Quellen in der öffentlichen Gasversorgung
    - KTBL (2018): Faustzahlen Biogas, 4th ed., table "Aufbereitungsverfahren"
    - DIN EN 16723-2 (2017): Natural gas and biomethane for use in transport and biomethane for injection
"""

from typing import Dict, Any, Optional

from ..base import Component, ComponentType


class BiogasUpgrading(Component):
    """
    Biogas upgrading unit for grid-quality biomethane production.

    Receives raw biogas from gas storages, upgrades it to biomethane, and
    passes capacity overflow to a downstream Flare via the ``gas`` connection.

    Parameters
    ----------
    component_id : str
        Unique identifier.
    capacity_m3h : float
        Maximum raw biogas throughput [m³/h]. Default 500.
    ch4_recovery : float
        Fraction of incoming CH4 recovered as biomethane [0..1]. Default 0.98.
    ch4_content_in : float
        Expected CH4 fraction in raw biogas [0..1]. Default 0.55.
        Overridden per-step by the ``CH4_fraction`` input key if provided.
    ch4_content_out : float
        Required CH4 fraction in biomethane product (grid spec) [0..1]. Default 0.97.
    name : str, optional
        Human-readable display name.
    """

    def __init__(
        self,
        component_id: str,
        capacity_m3h: float = 500.0,
        ch4_recovery: float = 0.98,
        ch4_content_in: float = 0.55,
        ch4_content_out: float = 0.97,
        name: Optional[str] = None,
    ):
        super().__init__(component_id, ComponentType.UPGRADING, name)

        self.capacity_m3h = float(capacity_m3h)
        self.ch4_recovery = float(ch4_recovery)
        self.ch4_content_in = float(ch4_content_in)
        self.ch4_content_out = float(ch4_content_out)

        self._cum_gas_in_m3: float = 0.0
        self._cum_biomethane_m3: float = 0.0
        self._cum_offgas_m3: float = 0.0
        self._cum_overflow_m3: float = 0.0

        self.initialize()

    @property
    def capacity_m3_per_day(self) -> float:
        """Nominal throughput in m³/day (raw biogas)."""
        return self.capacity_m3h * 24.0

    # ------------------------------------------------------------------
    # Component interface
    # ------------------------------------------------------------------

    def initialize(self, initial_state: Optional[Dict[str, Any]] = None) -> None:
        """Initialize or restore cumulative state."""
        if initial_state:
            self._cum_gas_in_m3 = float(initial_state.get("cumulative_gas_in_m3", 0.0))
            self._cum_biomethane_m3 = float(initial_state.get("cumulative_biomethane_m3", 0.0))
            self._cum_offgas_m3 = float(initial_state.get("cumulative_offgas_m3", 0.0))
            self._cum_overflow_m3 = float(initial_state.get("cumulative_overflow_m3", 0.0))
        else:
            self._cum_gas_in_m3 = 0.0
            self._cum_biomethane_m3 = 0.0
            self._cum_offgas_m3 = 0.0
            self._cum_overflow_m3 = 0.0

        self.outputs_data = {
            "Q_biomethane_m3_per_day": 0.0,
            "Q_offgas_m3_per_day": 0.0,
            "Q_gas_out_m3_per_day": 0.0,
            "utilization": 0.0,
            "cumulative_gas_in_m3": 0.0,
            "cumulative_biomethane_m3": 0.0,
            "cumulative_offgas_m3": 0.0,
            "cumulative_overflow_m3": 0.0,
        }
        self._initialized = True

    def step(self, t: float, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process one simulation timestep.

        Args:
            t:   current simulation time [days]
            dt:  timestep length [days]
            inputs: may contain:
                ``Q_gas_in_m3_per_day``  — raw biogas inflow [m³/day]
                ``CH4_fraction``         — actual CH4 fraction (overrides ch4_content_in)

        Returns:
            outputs_data with keys:
                ``Q_biomethane_m3_per_day``   — biomethane to grid [m³/day]
                ``Q_offgas_m3_per_day``       — CO2-rich reject gas [m³/day]
                ``Q_gas_out_m3_per_day``      — capacity overflow to paired flare [m³/day]
                ``utilization``               — fraction of capacity used [0..1]
                ``cumulative_gas_in_m3``
                ``cumulative_biomethane_m3``
                ``cumulative_offgas_m3``
                ``cumulative_overflow_m3``
        """
        Q_in = float(inputs.get("Q_gas_in_m3_per_day", 0.0))
        ch4_frac = float(inputs.get("CH4_fraction", self.ch4_content_in))

        # Capacity limit — overflow passes through to the paired flare
        Q_actual = min(Q_in, self.capacity_m3_per_day)
        Q_overflow = max(0.0, Q_in - Q_actual)

        # CH4 mass balance
        ch4_in = Q_actual * ch4_frac
        ch4_recovered = ch4_in * self.ch4_recovery
        Q_biomethane = ch4_recovered / max(self.ch4_content_out, 1e-9)

        # CO2-rich reject gas (vented to atmosphere, not to flare)
        Q_offgas = max(0.0, Q_actual - Q_biomethane)

        # Update cumulative totals [m³]
        self._cum_gas_in_m3 += Q_actual * dt
        self._cum_biomethane_m3 += Q_biomethane * dt
        self._cum_offgas_m3 += Q_offgas * dt
        self._cum_overflow_m3 += Q_overflow * dt

        utilization = Q_actual / max(self.capacity_m3_per_day, 1e-9)

        self.outputs_data = {
            "Q_biomethane_m3_per_day": float(Q_biomethane),
            "Q_offgas_m3_per_day": float(Q_offgas),
            "Q_gas_out_m3_per_day": float(Q_overflow),
            "utilization": float(utilization),
            "cumulative_gas_in_m3": float(self._cum_gas_in_m3),
            "cumulative_biomethane_m3": float(self._cum_biomethane_m3),
            "cumulative_offgas_m3": float(self._cum_offgas_m3),
            "cumulative_overflow_m3": float(self._cum_overflow_m3),
        }
        return self.outputs_data

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration and cumulative state."""
        return {
            "component_id": self.component_id,
            "component_type": self.component_type.value,
            "name": self.name,
            "capacity_m3h": self.capacity_m3h,
            "ch4_recovery": self.ch4_recovery,
            "ch4_content_in": self.ch4_content_in,
            "ch4_content_out": self.ch4_content_out,
            "cumulative_gas_in_m3": self._cum_gas_in_m3,
            "cumulative_biomethane_m3": self._cum_biomethane_m3,
            "cumulative_offgas_m3": self._cum_offgas_m3,
            "cumulative_overflow_m3": self._cum_overflow_m3,
            "outputs_data": self.outputs_data,
            "inputs": self.inputs,
            "outputs": self.outputs,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "BiogasUpgrading":
        """Reconstruct from dict produced by to_dict."""
        obj = cls(
            component_id=config["component_id"],
            capacity_m3h=config.get("capacity_m3h", 500.0),
            ch4_recovery=config.get("ch4_recovery", 0.98),
            ch4_content_in=config.get("ch4_content_in", 0.55),
            ch4_content_out=config.get("ch4_content_out", 0.97),
            name=config.get("name"),
        )
        obj.initialize(
            {
                "cumulative_gas_in_m3": config.get("cumulative_gas_in_m3", 0.0),
                "cumulative_biomethane_m3": config.get("cumulative_biomethane_m3", 0.0),
                "cumulative_offgas_m3": config.get("cumulative_offgas_m3", 0.0),
                "cumulative_overflow_m3": config.get("cumulative_overflow_m3", 0.0),
            }
        )
        return obj
