# ============================================================================
# pyadm1/components/energy/heating.py
# ============================================================================
"""
Heating system component for digester temperature control.

Heat demand is computed with a lumped (0D) heat balance using the
overall heat transfer coefficient method (UAΔT) plus sensible heating
of the influent:

    Q_dot_demand = Q_dot_loss + Q_dot_feed
    Q_dot_loss   = UA * (T_dig - T_amb)                      # heat losses to ambient
    Q_dot_feed   = Σ (m_dot_i * c_p,i * (T_dig - T_in,i))    # sensible heat to warm feed

The sensible-heat term is evaluated in pure Python from the substrate
fresh-matter density (already provided by :class:`Feedstock`) and a
component-weighted specific heat capacity (Choi–Okos 1986 coefficients
for carbohydrate / protein / lipid / ash / water and acetic acid for the
VFA pool).

Available CHP waste heat is used first, remaining demand is covered by
an auxiliary heater:

    P_chp_used = min(P_chp_available, Q_dot_demand)
    P_aux      = max(0, Q_dot_demand - P_chp_used)
    E_aux     += P_aux * Δt_hours

Units: UA in kW/K, heat flows in kW, auxiliary energy in kWh.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..base import Component, ComponentType

# Component specific heat capacities [kJ/(kg·K)] — Choi & Okos 1986
# (standard reference values for biomass / food thermodynamics).
_CP_CH = 1.5488  # carbohydrate
_CP_PR = 2.0082  # protein
_CP_LI = 1.9842  # lipid
_CP_MI = 1.0926  # mineral / ash
_CP_AC = 2.0430  # organic acid (acetic acid)
_CP_H2O = 4.1813  # water

#: Heat loss coefficient [kW/K] used when neither the heating system nor a
#: connected digester provides one.
_DEFAULT_HEAT_LOSS_COEFFICIENT = 0.5

#: Soil temperature [K] (10 °C) used when no ``T_ground`` input is given.
_DEFAULT_GROUND_TEMPERATURE = 283.15

#: Substrate inlet temperature [K] for substrates that carry no ``Temp``.
_DEFAULT_SUBSTRATE_TEMPERATURE = 288.15


def _substrate_cp(s) -> float:
    """
    Effective specific heat capacity of one substrate [kJ/(kg·K)].

    Mass fractions are derived from the same Weender breakdown that
    :meth:`Feedstock._calc_density` uses, so the (ρ, c_p) pair is
    self-consistent.
    """
    fTS = s.TS / 1000.0
    f_fiber = fTS * s.fRF
    f_protein = fTS * s.fRP
    f_lipid = fTS * s.fRFe
    f_ash = fTS * s.fRA
    f_NFE = fTS - f_fiber - f_protein - f_lipid - f_ash
    f_CH = f_fiber + f_NFE
    f_AC = getattr(s, "vfa_mass", s.FFS) / 1000.0
    f_H2O = max(0.0, 1.0 - fTS - f_AC)

    return f_CH * _CP_CH + f_protein * _CP_PR + f_lipid * _CP_LI + f_ash * _CP_MI + f_AC * _CP_AC + f_H2O * _CP_H2O


def _calc_process_heat_kw(
    q_substrates: Sequence[float] | None,
    feedstock,
    target_temperature: float,
    t_inlet: float | None = None,
) -> float:
    """
    Sensible heat required to warm the substrate feed to
    ``target_temperature`` [kW].

    Each substrate starts at its own storage temperature (``Temp`` of its
    substrate definition, °C). ``t_inlet`` [K] overrides that for all
    substrates. The ambient temperature plays no role here: manure from the
    barn or silage from the clamp does not follow the outside air.

    Returns 0 when no feed or no feedstock is provided; substrates already
    at or above the target contribute nothing.
    """
    if not q_substrates or feedstock is None:
        return 0.0

    substrates = getattr(feedstock, "_subs", None)
    if not substrates:
        return 0.0

    n = min(len(q_substrates), len(substrates))
    energy_per_day_kJ = 0.0
    for i in range(n):
        q = float(q_substrates[i])
        if q <= 0.0:
            continue
        if t_inlet is not None:
            t_in = float(t_inlet)
        else:
            temp_c = getattr(substrates[i], "Temp", None)
            t_in = _DEFAULT_SUBSTRATE_TEMPERATURE if temp_c is None else float(temp_c) + 273.15
        delta_t = float(target_temperature) - t_in
        if delta_t <= 0.0:
            continue
        m_dot = q * 1000.0  # feed rates are fresh matter in t/d -> kg/d
        cp = _substrate_cp(substrates[i])  # kJ/(kg·K)
        energy_per_day_kJ += m_dot * cp * delta_t  # kJ/d

    return energy_per_day_kJ / 86400.0  # kJ/s = kW


class HeatingSystem(Component):
    """
    Heating system for maintaining digester temperature.

    Calculates heat demand and energy consumption based on temperature
    difference and available waste heat from CHP.

    Attributes:
        target_temperature: Target digester temperature in K.
        heat_loss_coefficient: Heat loss coefficient (UA) in kW/K, or ``None``
            to take it from a connected digester that defines k-values
            (falling back to 0.5 kW/K).
        feedstock: Feedstock used to derive per-substrate density and c_p
            for the sensible-heat term. Optional — when omitted, only
            the UAΔT loss term contributes to demand.

    Example:
        >>> heating = HeatingSystem("heat1", target_temperature=315.15, heat_loss_coefficient=0.5)
        >>> heating.initialize()
        >>> result = heating.step(t=0, dt=1/24, inputs={"T_digester": 315.15, "P_th_available": 200})
    """

    def __init__(
        self,
        component_id: str,
        target_temperature: float = 315.15,
        heat_loss_coefficient: float | None = None,
        name: str | None = None,
        feedstock=None,
    ):
        """
        Initialize heating system.

        Args:
            component_id: Unique identifier.
            target_temperature: Target digester temperature in K. Defaults to 315.15
                (42 °C), matching the default of :class:`Digester`.
            heat_loss_coefficient: Heat loss coefficient (UA) in kW/K. An explicit
                value always wins. When ``None`` (default), the coefficient a
                connected digester derives from its k-values and geometry is
                used, and 0.5 kW/K when there is none.
            name: Human-readable name. Defaults to component_id.
            feedstock: Optional :class:`Feedstock` for sensible-heat calculation.
        """
        super().__init__(component_id, ComponentType.HEATING, name)

        self.target_temperature = target_temperature
        self.heat_loss_coefficient = heat_loss_coefficient
        self.feedstock = feedstock

        # Auto-initialize with default state
        self.initialize()

    def initialize(self, initial_state: dict[str, Any] | None = None) -> None:
        """
        Initialize heating system state.

        Args:
            initial_state: Optional state to restore; ``energy_consumed``
                (cumulative auxiliary energy in kWh) is carried over.
        """
        initial_state = initial_state or {}
        self.state = {
            "Q_heat_demand": 0.0,
            "Q_heat_supplied": 0.0,
            "energy_consumed": float(initial_state.get("energy_consumed", 0.0)),
            "heat_loss_coefficient_used": (
                _DEFAULT_HEAT_LOSS_COEFFICIENT if self.heat_loss_coefficient is None else self.heat_loss_coefficient
            ),
        }

        # Ensure outputs_data is also initialized
        self.outputs_data = {
            "Q_heat_supplied": 0.0,
            "P_th_used": 0.0,
            "P_aux_heat": 0.0,
        }

        self._initialized = True

    def step(self, t: float, dt: float, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Perform one simulation time step.

        Args:
            t: Current time in days.
            dt: Time step in days.
            inputs: Input data with keys:
                - 'T_digester': Current digester temperature [K]
                - 'T_ambient': Ambient temperature [K] (heat losses only)
                - 'T_ground': Soil temperature [K]; applies to the floor-slab
                  share of a digester-derived heat loss coefficient
                - 'heat_loss_coefficient_ground': that floor-slab share [kW/K]
                - 'T_substrate': Substrate inlet temperature [K]; overrides the
                  per-substrate ``Temp`` of the feedstock
                - 'heat_loss_coefficient': UA of the connected digester [kW/K],
                  used when this heating system has none of its own
                - 'P_th_available': Available thermal power from CHP [kW]
                - 'Q_substrates': Substrate feed rates [m³/d]
                  ('Q_substrates_fed' from a connected digester is accepted too)

        Returns:
            Output data with keys:
                - 'Q_heat_supplied': Heat supplied [kW]
                - 'P_th_used': Thermal power used from CHP [kW]
                - 'P_aux_heat': Auxiliary heating needed [kW]
        """
        T_digester = inputs.get("T_digester", self.target_temperature)
        T_ambient = inputs.get("T_ambient", 288.15)  # 15°C default
        P_th_available = inputs.get("P_th_available", 0.0)
        q_substrates = inputs.get("Q_substrates", inputs.get("Q_substrates_fed"))

        # A flat coefficient (own or default) has no known floor share and
        # acts against the ambient air as a whole. Only a coefficient derived
        # from the digester's k-values is split: the slab loses to the soil.
        UA = self.heat_loss_coefficient
        UA_ground = 0.0
        if UA is None:
            UA = inputs.get("heat_loss_coefficient", _DEFAULT_HEAT_LOSS_COEFFICIENT)
            if "heat_loss_coefficient" in inputs:
                UA_ground = min(float(inputs.get("heat_loss_coefficient_ground", 0.0)), UA)
        T_ground = inputs.get("T_ground", _DEFAULT_GROUND_TEMPERATURE)

        # Heat loss to environment
        Q_loss = (UA - UA_ground) * (T_digester - T_ambient) + UA_ground * (T_digester - T_ground)
        Q_process = _calc_process_heat_kw(q_substrates, self.feedstock, self.target_temperature, inputs.get("T_substrate"))

        # Heat demand to maintain temperature
        Q_demand = Q_loss + Q_process

        # Use CHP heat first
        P_th_used = min(P_th_available, Q_demand)

        # Additional heating needed
        P_aux_heat = max(0.0, Q_demand - P_th_used)

        Q_heat_supplied = P_th_used + P_aux_heat

        # Update state
        self.state["Q_heat_demand"] = Q_demand
        self.state["Q_heat_supplied"] = Q_heat_supplied
        self.state["heat_loss_coefficient_used"] = UA
        self.state["energy_consumed"] += P_aux_heat * dt * 24.0  # kWh

        self.outputs_data = {
            "Q_heat_supplied": Q_heat_supplied,
            "P_th_used": P_th_used,
            "P_aux_heat": P_aux_heat,
        }

        return self.outputs_data

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize to dictionary.

        Returns:
            Component configuration as dictionary.
        """
        return {
            "component_id": self.component_id,
            "component_type": self.component_type.value,
            "name": self.name,
            "target_temperature": self.target_temperature,
            "heat_loss_coefficient": self.heat_loss_coefficient,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> HeatingSystem:
        """
        Create from dictionary.

        Args:
            config: Component configuration.

        Returns:
            Initialized heating system component.
        """
        heating = cls(
            component_id=config["component_id"],
            target_temperature=config.get("target_temperature", 315.15),
            heat_loss_coefficient=config.get("heat_loss_coefficient"),
            name=config.get("name"),
        )

        heating.inputs = config.get("inputs", [])
        heating.outputs = config.get("outputs", [])

        if "state" in config:
            heating.initialize(config["state"])
        else:
            heating.initialize()

        return heating
