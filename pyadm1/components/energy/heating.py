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
                 # (implemented via calcHeatPower(...) / substrate stream properties)

Available CHP waste heat is used first, remaining demand is covered by
an auxiliary heater:

    P_chp_used = min(P_chp_available, Q_dot_demand)
    P_aux      = max(0, Q_dot_demand - P_chp_used)
    E_aux     += P_aux * Δt_hours

Units: UA in kW/K, heat flows in kW, auxiliary energy in kWh.
"""

import os
import platform
from pathlib import Path
from typing import Dict, Any, Optional

from ..base import Component, ComponentType

_PHYSVALUE = None
_BIOGAS = None
_SUBSTRATES_FACTORY = None
_SUBSTRATES_INSTANCE = None
_DLL_INIT_DONE = False
_HEAT_CALC_MODE = None


def _init_heating_dll() -> None:
    """Initialize DLLs for heating calculations."""
    global _PHYSVALUE, _BIOGAS, _SUBSTRATES_FACTORY, _DLL_INIT_DONE
    if _DLL_INIT_DONE:
        return
    _DLL_INIT_DONE = True

    try:
        import biogas as _biogas
        from physchem import physValue as _phys_value
    except Exception:
        try:
            import biogas as _biogas
            from physchem import PhysValue as _phys_value
        except Exception:
            return

    _BIOGAS = _biogas
    _SUBSTRATES_FACTORY = getattr(_biogas, "substrates", None)
    _PHYSVALUE = _phys_value
def _get_substrates_instance():
    """Get the substrate instance from factory."""
    global _SUBSTRATES_INSTANCE
    if _SUBSTRATES_INSTANCE is not None:
        return _SUBSTRATES_INSTANCE

    _init_heating_dll()
    if _SUBSTRATES_FACTORY is None:
        return None

    try:
        xml_path = Path(__file__).resolve().parents[3] / "data" / "substrates" / "substrate_gummersbach.xml"
        _SUBSTRATES_INSTANCE = _SUBSTRATES_FACTORY(str(xml_path))
    except Exception:
        _SUBSTRATES_INSTANCE = None

    return _SUBSTRATES_INSTANCE


def _calc_process_heat_kw(q_substrates, target_temperature: float) -> float:
    """Calculate process heat demand in kW."""
    global _HEAT_CALC_MODE

    if not q_substrates:
        return 0.0

    substrates = _get_substrates_instance()
    if substrates is None or _PHYSVALUE is None:
        return 0.0

    t_target = _PHYSVALUE(float(target_temperature), "K")

    if _HEAT_CALC_MODE is None:
        if hasattr(substrates, "calcHeatPower"):
            _HEAT_CALC_MODE = "substrates_calcHeatPower"
        elif _BIOGAS is not None and hasattr(_BIOGAS, "ADMstate") and hasattr(_BIOGAS.ADMstate, "calcHeatPower"):
            _HEAT_CALC_MODE = "admstate_calcHeatPower"
        elif hasattr(substrates, "calcSumQuantityOfHeatPerDay"):
            _HEAT_CALC_MODE = "substrates_calcSumQuantityOfHeatPerDay"
        else:
            _HEAT_CALC_MODE = "none"

    try:
        if _HEAT_CALC_MODE == "substrates_calcHeatPower":
            res = substrates.calcHeatPower(q_substrates, t_target)
            return float(getattr(res, "Value", res))
        if _HEAT_CALC_MODE == "admstate_calcHeatPower":
            res = _BIOGAS.ADMstate.calcHeatPower(substrates, q_substrates, t_target)
            return float(getattr(res, "Value", res))
        if _HEAT_CALC_MODE == "substrates_calcSumQuantityOfHeatPerDay":
            return float(substrates.calcSumQuantityOfHeatPerDay(q_substrates, t_target).Value) / 24.0
    except Exception:
        return 0.0

    return 0.0


class HeatingSystem(Component):
    """
    Heating system for maintaining digester temperature.

    Calculates heat demand and energy consumption based on temperature
    difference and available waste heat from CHP.

    Attributes:
        target_temperature: Target digester temperature in K.
        heat_loss_coefficient: Heat loss coefficient in kW/K.

    Example:
        >>> heating = HeatingSystem("heat1", target_temperature=308.15, heat_loss_coefficient=0.5)
        >>> heating.initialize()
        >>> result = heating.step(t=0, dt=1/24, inputs={"T_digester": 308.15, "P_th_available": 200})
    """

    def __init__(
        self,
        component_id: str,
        target_temperature: float = 308.15,
        heat_loss_coefficient: float = 0.5,
        name: Optional[str] = None,
    ):
        """
        Initialize heating system.

        Args:
            component_id: Unique identifier.
            target_temperature: Target digester temperature in K. Defaults to 308.15 (35°C).
            heat_loss_coefficient: Heat loss coefficient in kW/K. Defaults to 0.5.
            name: Human-readable name. Defaults to component_id.
        """
        super().__init__(component_id, ComponentType.HEATING, name)

        self.target_temperature = target_temperature
        self.heat_loss_coefficient = heat_loss_coefficient

        # Auto-initialize with default state
        self.initialize()

    def initialize(self, initial_state: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize heating system state.

        Args:
            initial_state: Initial state (not used currently).
        """
        self.state = {
            "Q_heat_demand": 0.0,
            "Q_heat_supplied": 0.0,
            "energy_consumed": 0.0,
        }

        # Ensure outputs_data is also initialized
        self.outputs_data = {
            "Q_heat_supplied": 0.0,
            "P_th_used": 0.0,
            "P_aux_heat": 0.0,
        }

    def step(self, t: float, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform one simulation time step.

        Args:
            t: Current time in days.
            dt: Time step in days.
            inputs: Input data with keys:
                - 'T_digester': Current digester temperature [K]
                - 'T_ambient': Ambient temperature [K]
                - 'V_liq': Liquid volume [m³]
                - 'P_th_available': Available thermal power from CHP [kW]

        Returns:
            Output data with keys:
                - 'Q_heat_supplied': Heat supplied [kW]
                - 'P_th_used': Thermal power used from CHP [kW]
                - 'P_aux_heat': Auxiliary heating needed [kW]
        """
        T_digester = inputs.get("T_digester", self.target_temperature)
        T_ambient = inputs.get("T_ambient", 288.15)  # 15°C default
        P_th_available = inputs.get("P_th_available", 0.0)
        q_substrates = inputs.get("Q_substrates")

        # Heat loss to environment
        T_diff = T_digester - T_ambient
        Q_loss = self.heat_loss_coefficient * T_diff
        Q_process = _calc_process_heat_kw(q_substrates, self.target_temperature)

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
        self.state["energy_consumed"] += P_aux_heat * dt * 24.0  # kWh

        self.outputs_data = {
            "Q_heat_supplied": Q_heat_supplied,
            "P_th_used": P_th_used,
            "P_aux_heat": P_aux_heat,
        }

        return self.outputs_data

    def to_dict(self) -> Dict[str, Any]:
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
    def from_dict(cls, config: Dict[str, Any]) -> "HeatingSystem":
        """
        Create from dictionary.

        Args:
            config: Component configuration.

        Returns:
            Initialized heating system component.
        """
        heating = cls(
            component_id=config["component_id"],
            target_temperature=config.get("target_temperature", 308.15),
            heat_loss_coefficient=config.get("heat_loss_coefficient", 0.5),
            name=config.get("name"),
        )

        heating.inputs = config.get("inputs", [])
        heating.outputs = config.get("outputs", [])

        if "state" in config:
            heating.initialize(config["state"])
        else:
            heating.initialize()

        return heating
