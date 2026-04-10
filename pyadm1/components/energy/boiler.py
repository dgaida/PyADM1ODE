# ============================================================================
# pyadm1/components/energy/boiler.py
# ============================================================================
"""
Auxiliary gas boiler for backup and peak heating in biogas plants.

The boiler covers the residual heat demand that the CHP waste heat cannot
supply.  It supports three fuel modes:

    biogas       — burns only biogas from the plant's gas storage
    natural_gas  — burns only natural gas (grid supply)
    dual         — burns biogas first; switches to natural gas for the
                   remainder when biogas is insufficient

Heat output model (steady-state):
    P_th_supplied = min(P_th_demand, P_th_nom)
    Q_fuel        = P_th_supplied / (eta_actual * LHV_fuel)   [m3/d]

Part-load efficiency (simplified linear correction, VDI 4631):
    eta_actual = eta_nom * (1 - lambda_part * (1 - load_fraction))
    lambda_part = 0.05   (5% efficiency penalty at zero load)

Lower heating values used (at standard conditions):
    Biogas     (60% CH4): LHV ≈ 6.0 kWh/m3     (KTBL 2013, DIN 51624)
    Natural gas (H-gas):  LHV ≈ 10.0 kWh/m3    (DVGW G 260)

References:
    - KTBL (2013): Faustzahlen Biogas, 3rd ed., KTBL-Heft 469, pp. 160-165
    - VDI 4631 (2012): Part-load efficiency of heating systems
    - EN 303-1 (2018): Gas-fired heating boilers — general requirements
    - DVGW G 260 (2021): Gas quality, LHV of natural gas H-gas

Example:
    >>> from pyadm1.components.energy import Boiler
    >>>
    >>> boiler = Boiler("boiler_1", P_th_nom=200.0, fuel_type="dual")
    >>> boiler.initialize()
    >>> result = boiler.step(t=0, dt=1/24, inputs={"P_th_demand": 120.0})
    >>> print(f"Supplied: {result['P_th_supplied']:.1f} kW  "
    ...       f"Gas: {result['Q_gas_consumed_m3_per_day']:.1f} m3/d")
"""

from typing import Dict, Any, Optional
from ..base import Component, ComponentType

# Lower heating values [kWh/m3] at standard conditions
_LHV_BIOGAS = 6.0  # KTBL 2013, 60% CH4
_LHV_NATURAL_GAS = 10.0  # DVGW G 260, H-gas

# Relative efficiency penalty at part load (VDI 4631 simplified)
_PART_LOAD_PENALTY = 0.05


class Boiler(Component):
    """
    Auxiliary gas boiler for backup and peak heating.

    Covers the residual heat demand (``P_aux_heat`` from :class:`HeatingSystem`)
    that CHP waste heat cannot supply.  Supports biogas-only, natural-gas-only,
    or dual-fuel operation.

    Attributes:
        P_th_nom (float):    Nominal thermal output [kW].
        efficiency (float):  Rated thermal efficiency at full load (0-1).
        fuel_type (str):     Fuel mode: ``"biogas"`` | ``"natural_gas"`` | ``"dual"``.
        lhv_biogas (float):  Lower heating value of biogas [kWh/m3].
        lhv_natural_gas (float): LHV of natural gas [kWh/m3].

    Example:
        >>> boiler = Boiler("boiler_1", P_th_nom=200.0, fuel_type="dual",
        ...                 efficiency=0.92)
        >>> boiler.initialize()
        >>> out = boiler.step(t=0, dt=1/24,
        ...     inputs={"P_th_demand": 80.0,
        ...             "Q_gas_available_m3_per_day": 50.0})
    """

    def __init__(
        self,
        component_id: str,
        P_th_nom: float = 200.0,
        efficiency: float = 0.90,
        fuel_type: str = "dual",
        lhv_biogas: float = _LHV_BIOGAS,
        lhv_natural_gas: float = _LHV_NATURAL_GAS,
        name: Optional[str] = None,
    ):
        """
        Initialize boiler.

        Args:
            component_id:    Unique identifier.
            P_th_nom:        Nominal thermal power [kW].  Default 200.
            efficiency:      Rated full-load thermal efficiency (0-1).  Default 0.90.
            fuel_type:       Fuel mode — ``"biogas"``, ``"natural_gas"``,
                             or ``"dual"`` (biogas first, then natural gas).
                             Default ``"dual"``.
            lhv_biogas:      Lower heating value of biogas [kWh/m3].
                             Default 6.0 (KTBL 2013 at 60% CH4).
            lhv_natural_gas: Lower heating value of natural gas [kWh/m3].
                             Default 10.0 (DVGW G 260 H-gas).
            name:            Human-readable display name.
        """
        super().__init__(component_id, ComponentType.BOILER, name)

        self.P_th_nom = float(P_th_nom)
        self.efficiency = float(efficiency)

        fuel_type = fuel_type.lower()
        if fuel_type not in ("biogas", "natural_gas", "dual"):
            raise ValueError("fuel_type must be 'biogas', 'natural_gas', or 'dual'")
        self.fuel_type = fuel_type

        self.lhv_biogas = float(lhv_biogas)
        self.lhv_natural_gas = float(lhv_natural_gas)

        # Cumulative tracking
        self.energy_supplied = 0.0  # kWh
        self.gas_consumed_total = 0.0  # m3 biogas
        self.ng_consumed_total = 0.0  # m3 natural gas
        self.operating_hours = 0.0  # h

        self.initialize()

    # ------------------------------------------------------------------
    # Component interface
    # ------------------------------------------------------------------

    def initialize(self, initial_state: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize boiler state.

        Args:
            initial_state: Optional dict with keys:
                - ``energy_supplied``:    cumulative heat output [kWh]
                - ``gas_consumed_total``: cumulative biogas consumed [m3]
                - ``ng_consumed_total``:  cumulative natural gas consumed [m3]
                - ``operating_hours``:    cumulative run time [h]
        """
        if initial_state:
            self.energy_supplied = float(initial_state.get("energy_supplied", 0.0))
            self.gas_consumed_total = float(initial_state.get("gas_consumed_total", 0.0))
            self.ng_consumed_total = float(initial_state.get("ng_consumed_total", 0.0))
            self.operating_hours = float(initial_state.get("operating_hours", 0.0))

        self.state = {
            "energy_supplied": self.energy_supplied,
            "gas_consumed_total": self.gas_consumed_total,
            "ng_consumed_total": self.ng_consumed_total,
            "operating_hours": self.operating_hours,
        }

        self.outputs_data = {
            "P_th_supplied": 0.0,
            "P_th_available": 0.0,  # alias — HeatingSystem reads this key
            "Q_gas_consumed_m3_per_day": 0.0,
            "Q_natural_gas_m3_per_day": 0.0,
            "is_running": False,
            "load_fraction": 0.0,
            "efficiency_actual": self.efficiency,
        }

        self._initialized = True

    def step(self, t: float, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform one simulation time step.

        The boiler fires to satisfy ``P_th_demand`` up to its nominal
        capacity.  Gas is drawn first from available biogas (if provided),
        then from the natural gas grid if the fuel mode allows it.

        Args:
            t:  Current simulation time [days].
            dt: Time step [days].
            inputs: Dict with optional keys:
                - ``P_th_demand`` [kW]:
                    Heat demand to be covered.  Accepts ``P_aux_heat``
                    (the key emitted by :class:`HeatingSystem`) as alias.
                - ``Q_gas_available_m3_per_day`` [m3/d]:
                    Biogas available from gas storage.  Only relevant for
                    ``fuel_type`` ``"biogas"`` or ``"dual"``.
                - ``enable`` [bool]:
                    If *False*, boiler is forced off.  Default *True*.

        Returns:
            Dict with keys:
                - ``P_th_supplied`` [kW]:
                    Actual thermal output delivered.
                - ``P_th_available`` [kW]:
                    Alias for ``P_th_supplied`` (for HeatingSystem input).
                - ``Q_gas_consumed_m3_per_day`` [m3/d]:
                    Biogas consumption rate.
                - ``Q_natural_gas_m3_per_day`` [m3/d]:
                    Natural gas consumption rate.
                - ``is_running`` [bool]:
                    Whether the boiler is currently firing.
                - ``load_fraction`` [float]:
                    Current output as fraction of nominal (0-1).
                - ``efficiency_actual`` [float]:
                    Effective efficiency at this load point.
        """
        enable = bool(inputs.get("enable", True))

        # Accept both P_th_demand and P_aux_heat (HeatingSystem output key)
        P_demand = float(inputs.get("P_th_demand", inputs.get("P_aux_heat", 0.0)))
        Q_gas_avail = float(inputs.get("Q_gas_available_m3_per_day", 0.0))

        if not enable or P_demand <= 0.0:
            self._update_outputs(0.0, 0.0, 0.0, dt)
            return self.outputs_data

        # Clamp demand to nominal capacity
        P_th_supplied = min(P_demand, self.P_th_nom)
        load_fraction = P_th_supplied / self.P_th_nom

        # Part-load efficiency (linear correction, VDI 4631)
        eta_actual = self.efficiency * (1.0 - _PART_LOAD_PENALTY * (1.0 - load_fraction))
        eta_actual = max(0.01, eta_actual)

        # --- Fuel allocation ------------------------------------------
        Q_gas_consumed = 0.0  # m3/d biogas
        Q_ng_consumed = 0.0  # m3/d natural gas

        if self.fuel_type == "natural_gas":
            # Burn natural gas only
            Q_ng_consumed = P_th_supplied * 24.0 / (eta_actual * self.lhv_natural_gas)

        elif self.fuel_type == "biogas":
            # Burn biogas only (even if unavailable — caller must manage supply)
            Q_gas_consumed = P_th_supplied * 24.0 / (eta_actual * self.lhv_biogas)

        else:  # dual fuel: biogas first, natural gas for remainder
            # Maximum biogas combustion that would cover the demand
            Q_gas_needed_total = P_th_supplied * 24.0 / (eta_actual * self.lhv_biogas)

            if Q_gas_avail >= Q_gas_needed_total:
                # Sufficient biogas: no natural gas needed
                Q_gas_consumed = Q_gas_needed_total
            else:
                # Burn all available biogas, supplement with natural gas
                Q_gas_consumed = Q_gas_avail
                P_from_biogas = Q_gas_consumed * eta_actual * self.lhv_biogas / 24.0
                P_from_ng = P_th_supplied - P_from_biogas
                Q_ng_consumed = P_from_ng * 24.0 / (eta_actual * self.lhv_natural_gas)

        self._update_outputs(P_th_supplied, Q_gas_consumed, Q_ng_consumed, dt)
        return self.outputs_data

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration and cumulative state to dictionary."""
        return {
            "component_id": self.component_id,
            "component_type": self.component_type.value,
            "name": self.name,
            "P_th_nom": self.P_th_nom,
            "efficiency": self.efficiency,
            "fuel_type": self.fuel_type,
            "lhv_biogas": self.lhv_biogas,
            "lhv_natural_gas": self.lhv_natural_gas,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "Boiler":
        """
        Create Boiler from dictionary (produced by :meth:`to_dict`).

        Args:
            config: Configuration dictionary.

        Returns:
            Initialized Boiler instance.
        """
        boiler = cls(
            component_id=config["component_id"],
            P_th_nom=config.get("P_th_nom", 200.0),
            efficiency=config.get("efficiency", 0.90),
            fuel_type=config.get("fuel_type", "dual"),
            lhv_biogas=config.get("lhv_biogas", _LHV_BIOGAS),
            lhv_natural_gas=config.get("lhv_natural_gas", _LHV_NATURAL_GAS),
            name=config.get("name"),
        )

        boiler.inputs = config.get("inputs", [])
        boiler.outputs = config.get("outputs", [])

        if "state" in config:
            boiler.initialize(config["state"])

        return boiler

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_outputs(
        self,
        P_th_supplied: float,
        Q_gas_consumed: float,
        Q_ng_consumed: float,
        dt: float,
    ) -> None:
        """Update outputs_data and cumulative state variables."""
        is_running = P_th_supplied > 0.0
        load_fraction = P_th_supplied / self.P_th_nom if self.P_th_nom > 0 else 0.0
        eta_actual = self.efficiency * (1.0 - _PART_LOAD_PENALTY * (1.0 - load_fraction)) if is_running else self.efficiency

        dt_hours = dt * 24.0
        self.energy_supplied += P_th_supplied * dt_hours  # kWh
        self.gas_consumed_total += Q_gas_consumed * dt  # m3
        self.ng_consumed_total += Q_ng_consumed * dt  # m3
        if is_running:
            self.operating_hours += dt_hours

        self.state.update(
            {
                "energy_supplied": self.energy_supplied,
                "gas_consumed_total": self.gas_consumed_total,
                "ng_consumed_total": self.ng_consumed_total,
                "operating_hours": self.operating_hours,
            }
        )

        self.outputs_data = {
            "P_th_supplied": float(P_th_supplied),
            "P_th_available": float(P_th_supplied),  # HeatingSystem input key
            "Q_gas_consumed_m3_per_day": float(Q_gas_consumed),
            "Q_natural_gas_m3_per_day": float(Q_ng_consumed),
            "is_running": bool(is_running),
            "load_fraction": float(load_fraction),
            "efficiency_actual": float(eta_actual),
            # Cumulative
            "energy_supplied_kwh": float(self.energy_supplied),
            "gas_consumed_total_m3": float(self.gas_consumed_total),
            "ng_consumed_total_m3": float(self.ng_consumed_total),
            "operating_hours": float(self.operating_hours),
        }
