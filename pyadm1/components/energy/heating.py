# ============================================================================
# pyadm1/components/energy/heating.py
# ============================================================================
"""
Heating system component for digester temperature control.

This module provides the HeatingSystem class for maintaining digester
temperature using waste heat from CHP or auxiliary heating.
"""

from typing import Dict, Any, Optional

from ..base import Component, ComponentType


class HeatingSystem(Component):
    """
    Heating system for maintaining digester temperature.

    Calculates heat demand and energy consumption based on temperature
    difference and available waste heat from CHP.

    Attributes:
        target_temperature (float): Target digester temperature in K.
        heat_loss_coefficient (float): Heat loss coefficient in kW/K.

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
            component_id (str): Unique identifier.
            target_temperature (float): Target digester temperature in K. Defaults to 308.15 (35°C).
            heat_loss_coefficient (float): Heat loss coefficient in kW/K. Defaults to 0.5.
            name (Optional[str]): Human-readable name. Defaults to component_id.
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
            initial_state (Optional[Dict[str, Any]]): Initial state (not used currently).
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
            t (float): Current time in days.
            dt (float): Time step in days.
            inputs (Dict[str, Any]): Input data with keys:
                - 'T_digester': Current digester temperature [K]
                - 'T_ambient': Ambient temperature [K]
                - 'V_liq': Liquid volume [m³]
                - 'P_th_available': Available thermal power from CHP [kW]

        Returns:
            Dict[str, Any]: Output data with keys:
                - 'Q_heat_supplied': Heat supplied [kW]
                - 'P_th_used': Thermal power used from CHP [kW]
                - 'P_aux_heat': Auxiliary heating needed [kW]
        """
        T_digester = inputs.get("T_digester", self.target_temperature)
        T_ambient = inputs.get("T_ambient", 288.15)  # 15°C default
        P_th_available = inputs.get("P_th_available", 0.0)

        # TODO: Call C# DLL functions for heating calculations
        # For now, use simplified model

        # Heat loss to environment
        T_diff = T_digester - T_ambient
        Q_loss = self.heat_loss_coefficient * T_diff

        # Heat demand to maintain temperature
        Q_demand = Q_loss

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
            Dict[str, Any]: Component configuration as dictionary.
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
            config (Dict[str, Any]): Component configuration.

        Returns:
            HeatingSystem: Initialized heating system component.
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
