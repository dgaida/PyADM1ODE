# ============================================================================
# pyadm1/components/energy/chp.py
# ============================================================================
"""
Combined Heat and Power (CHP) unit component.

This module provides the CHP class for converting biogas to electricity
and heat in a biogas plant configuration.
"""

from typing import Dict, Any, Optional

from pyadm1.components.base import Component, ComponentType


class CHP(Component):
    """
    Combined Heat and Power unit.

    Converts biogas to electricity and heat with configurable efficiency.

    Attributes:
        P_el_nom (float): Nominal electrical power in kW.
        eta_el (float): Electrical efficiency (0-1).
        eta_th (float): Thermal efficiency (0-1).
        load_factor (float): Current operating point (0-1).

    Example:
        >>> chp = CHP("chp1", P_el_nom=500, eta_el=0.40, eta_th=0.45)
        >>> chp.initialize()
        >>> result = chp.step(t=0, dt=1/24, inputs={"Q_ch4": 1000})
    """

    def __init__(
        self,
        component_id: str,
        P_el_nom: float = 500.0,
        eta_el: float = 0.40,
        eta_th: float = 0.45,
        name: Optional[str] = None,
    ):
        """
        Initialize CHP unit.

        Args:
            component_id (str): Unique identifier.
            P_el_nom (float): Nominal electrical power in kW. Defaults to 500.0.
            eta_el (float): Electrical efficiency (0-1). Defaults to 0.40.
            eta_th (float): Thermal efficiency (0-1). Defaults to 0.45.
            name (Optional[str]): Human-readable name. Defaults to component_id.
        """
        super().__init__(component_id, ComponentType.CHP, name)

        self.P_el_nom = P_el_nom
        self.eta_el = eta_el
        self.eta_th = eta_th

        # Operating point (0-1)
        self.load_factor = 0.0

        # Auto-initialize with default state
        self.initialize()

    def initialize(self, initial_state: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize CHP state.

        Args:
            initial_state (Optional[Dict[str, Any]]): Initial state with keys:
                - 'load_factor': Initial load factor (0-1)
                If None, uses default initialization.
        """
        if initial_state is None:
            initial_state = {}

        self.load_factor = initial_state.get("load_factor", 0.0)

        self.state = {
            "load_factor": self.load_factor,
            "P_el": 0.0,
            "P_th": 0.0,
            "Q_gas_consumed": 0.0,
            "operating_hours": 0.0,
        }

        # Ensure outputs_data is also initialized
        self.outputs_data = {
            "P_el": 0.0,
            "P_th": 0.0,
            "Q_gas_consumed": 0.0,
            "Q_ch4_remaining": 0.0,
        }

    def step(self, t: float, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform one simulation time step.

        Args:
            t (float): Current time in days.
            dt (float): Time step in days.
            inputs (Dict[str, Any]): Input data with keys:
                - 'Q_ch4': Methane flow rate [m³/d] (direct input)
                - 'Q_gas_supplied_m3_per_day': Available biogas from storage [m³/d]
                - 'load_setpoint': Desired load factor [0-1] (optional)

        Returns:
            Dict[str, Any]: Output data with keys:
                - 'P_el': Electrical power [kW]
                - 'P_th': Thermal power [kW]
                - 'Q_gas_consumed': Biogas consumption [m³/d]
                - 'Q_gas_out_m3_per_day': Gas demand for connected storages [m³/d]
                - 'Q_ch4_remaining': Remaining methane [m³/d]
        """
        # Determine gas source priority:
        # 1. Direct gas supply from storage (preferred, plant_builder managed)
        # 2. Direct CH4 from digester (fallback for simple configurations)

        Q_gas_from_storage = inputs.get("Q_gas_supplied_m3_per_day", 0.0)
        Q_ch4_direct = inputs.get("Q_ch4", 0.0)

        # Assume 60% methane content in biogas from storage
        CH4_content = 0.60

        if Q_gas_from_storage > 0:
            # Use gas from storage (normal operation)
            Q_ch4_available = Q_gas_from_storage * CH4_content
        else:
            # Fallback: use direct CH4 from digester
            Q_ch4_available = Q_ch4_direct

        load_setpoint = inputs.get("load_setpoint", 1.0)

        # Methane energy content: ~10 kWh/m³
        E_ch4 = 10.0  # kWh/m³

        # Available power from methane
        P_available = Q_ch4_available / 24.0 * E_ch4  # kW (convert m³/d to m³/h)

        # Determine actual load
        P_el_max = min(self.P_el_nom, P_available / self.eta_el)
        self.load_factor = min(load_setpoint, P_el_max / self.P_el_nom) if self.P_el_nom > 0 else 0.0

        # Calculate outputs
        P_el = self.load_factor * self.P_el_nom
        P_th = P_el * self.eta_th / self.eta_el if self.eta_el > 0 else 0.0
        Q_ch4_consumed = (P_el / self.eta_el) * 24.0 / E_ch4 if self.eta_el > 0 else 0.0  # m³/d
        Q_gas_consumed = Q_ch4_consumed / CH4_content if CH4_content > 0 else 0.0  # m³/d biogas

        # Update state
        self.state["load_factor"] = self.load_factor
        self.state["P_el"] = P_el
        self.state["P_th"] = P_th
        self.state["Q_gas_consumed"] = Q_gas_consumed
        self.state["operating_hours"] += dt * 24.0

        self.outputs_data = {
            "P_el": P_el,
            "P_th": P_th,
            "Q_gas_consumed": Q_gas_consumed,
            "Q_gas_out_m3_per_day": Q_gas_consumed,  # Demand signal for storages
            "Q_ch4_remaining": Q_ch4_available - Q_ch4_consumed,
        }

        return self.outputs_data

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dictionary.

        Returns:
            Dict[str, Any]: Component configuration as dictionary.
        """
        return {
            "component_id": self.component_id,
            "component_type": self.component_type.value,
            "name": self.name,
            "P_el_nom": self.P_el_nom,
            "eta_el": self.eta_el,
            "eta_th": self.eta_th,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "CHP":
        """
        Create from dictionary.

        Args:
            config (Dict[str, Any]): Component configuration.

        Returns:
            CHP: Initialized CHP component.
        """
        chp = cls(
            component_id=config["component_id"],
            P_el_nom=config.get("P_el_nom", 500.0),
            eta_el=config.get("eta_el", 0.40),
            eta_th=config.get("eta_th", 0.45),
            name=config.get("name"),
        )

        chp.inputs = config.get("inputs", [])
        chp.outputs = config.get("outputs", [])

        if "state" in config:
            chp.initialize(config["state"])
        else:
            chp.initialize()

        return chp
