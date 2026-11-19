# ============================================================================
# pyadm1/components/energy/chp.py
# ============================================================================
"""
Combined Heat and Power (CHP) unit component.

This module provides the CHP class for converting biogas to electricity
and heat in a biogas plant configuration.
"""

from __future__ import annotations

from typing import Any

from ..base import Component, ComponentType


class CHP(Component):
    """
    Combined Heat and Power unit.

    Converts biogas to electricity and heat with configurable efficiency.

    Attributes:
        P_el_nom (float): Nominal electrical power in kW.
        eta_el (float): Electrical efficiency (0-1).
        eta_th (float): Thermal efficiency (0-1).
        ch4_content_default (float): Methane fraction assumed only when the
            supplied gas carries no composition (0-1).
        lhv_ch4 (float): Lower heating value of methane in kWh/m³.
        own_consumption_fraction (float): Share of the generated electricity
            the unit's own auxiliaries draw (0-1).
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
        name: str | None = None,
        ch4_content_default: float = 0.60,
        lhv_ch4: float = 10.0,
        own_consumption_fraction: float = 0.03,
    ):
        """
        Initialize CHP unit.

        Args:
            component_id (str): Unique identifier.
            P_el_nom (float): Nominal electrical power in kW. Defaults to 500.0.
            eta_el (float): Electrical efficiency (0-1). Defaults to 0.40.
            eta_th (float): Thermal efficiency (0-1). Defaults to 0.45.
            name (Optional[str]): Human-readable name. Defaults to component_id.
            ch4_content_default (float): Methane fraction (0-1) used only as a
                fallback when neither the storage nor the digester reports the
                composition of the supplied gas. Defaults to 0.60.
            lhv_ch4 (float): Lower heating value of methane in kWh/m³.
                Defaults to 10.0 (9.97 kWh/m³ at standard conditions).
            own_consumption_fraction (float): Share of the generated
                electricity drawn by the unit's own auxiliaries (coolant pumps,
                radiator fans, gas blower, controls), 0-1. Defaults to 0.03,
                a typical value; ``P_el`` stays the gross output.
        """
        super().__init__(component_id, ComponentType.CHP, name)

        if not 0.0 < float(ch4_content_default) <= 1.0:
            raise ValueError(f"CHP '{component_id}': ch4_content_default must be in (0, 1], got {ch4_content_default}")
        self.ch4_content_default = float(ch4_content_default)
        if float(lhv_ch4) <= 0.0:
            raise ValueError(f"CHP '{component_id}': lhv_ch4 must be positive, got {lhv_ch4}")
        self.lhv_ch4 = float(lhv_ch4)
        if not 0.0 <= float(own_consumption_fraction) < 1.0:
            raise ValueError(
                f"CHP '{component_id}': own_consumption_fraction must be in [0, 1), got {own_consumption_fraction}"
            )
        self.own_consumption_fraction = float(own_consumption_fraction)

        self.P_el_nom = P_el_nom
        self.eta_el = eta_el
        self.eta_th = eta_th

        # Operating point (0-1)
        self.load_factor = 0.0

        # Auto-initialize with default state
        self.initialize()

    def initialize(self, initial_state: dict[str, Any] | None = None) -> None:
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
            "operating_hours": float(initial_state.get("operating_hours", 0.0)),
        }

        # Ensure outputs_data is also initialized
        self.outputs_data = {
            "P_el": 0.0,
            "P_th": 0.0,
            "Q_gas_consumed": 0.0,
            "Q_ch4_remaining": 0.0,
        }

        self._initialized = True

    def step(self, t: float, dt: float, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Perform one simulation time step.

        Args:
            t (float): Current time in days.
            dt (float): Time step in days.
            inputs (Dict[str, Any]): Input data with keys:
                - 'Q_ch4': Methane flow rate [m³/d] (direct input)
                - 'Q_gas_supplied_m3_per_day': Available biogas from storage [m³/d]
                - 'Q_ch4_supplied_m3_per_day': Methane within that biogas [m³/d]
                - 'CH4_fraction': Methane fraction of the supplied biogas [0-1],
                  used when 'Q_ch4_supplied_m3_per_day' is absent
                - 'Q_gas': Total biogas of the direct digester stream [m³/d],
                  gives the methane fraction of the direct 'Q_ch4' path
                - 'load_setpoint': Desired load factor [0-1] (optional)

        Returns:
            Dict[str, Any]: Output data with keys:
                - 'P_el': Electrical power [kW]
                - 'P_th': Thermal power [kW]
                - 'Q_gas_consumed': Biogas consumption [m³/d]
                - 'Q_gas_out_m3_per_day': Gas demand for connected storages [m³/d]
                - 'Q_ch4_remaining': Remaining methane [m³/d]
                - 'CH4_fraction': Methane fraction the step was computed with
                - 'P_consumed': Electricity drawn by the unit's auxiliaries [kW]
                - 'P_el_net': 'P_el' minus 'P_consumed' [kW]
        """
        # Determine gas source priority:
        # 1. Direct gas supply from storage (preferred, plant_builder managed)
        # 2. Direct CH4 from digester (fallback for simple configurations)

        Q_gas_from_storage = inputs.get("Q_gas_supplied_m3_per_day", 0.0)
        Q_ch4_direct = inputs.get("Q_ch4", 0.0)

        # Methane content of the gas actually burnt. The storage reports it
        # (as a methane flow or a fraction); the direct digester stream gives
        # it as Q_ch4 / Q_gas. ``ch4_content_default`` is only the fallback
        # for a source that carries no composition at all.
        CH4_content = self.ch4_content_default

        # Use gas from storage during normal operation, otherwise fall back to
        # the direct CH4 stream from the digester.
        if Q_gas_from_storage > 0:
            if "Q_ch4_supplied_m3_per_day" in inputs:
                Q_ch4_available = float(inputs["Q_ch4_supplied_m3_per_day"])
                CH4_content = Q_ch4_available / Q_gas_from_storage
            else:
                CH4_content = float(inputs.get("CH4_fraction", CH4_content))
                Q_ch4_available = Q_gas_from_storage * CH4_content
        else:
            Q_ch4_available = Q_ch4_direct
            Q_gas_direct = float(inputs.get("Q_gas", 0.0))
            if Q_gas_direct > 0.0 and Q_ch4_direct > 0.0:
                CH4_content = Q_ch4_direct / Q_gas_direct

        load_setpoint = inputs.get("load_setpoint", 1.0)

        # Methane energy content [kWh/m³]
        E_ch4 = self.lhv_ch4

        # Available power from methane
        P_available = Q_ch4_available / 24.0 * E_ch4  # kW (convert m³/d to m³/h)

        # Determine actual load
        # P_available is the input thermal power from methane; multiply by eta_el
        # to get the maximum possible electrical output power.
        P_el_max = min(self.P_el_nom, P_available * self.eta_el)
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
        # Hours count only while the engine actually runs.
        if P_el > 0.0:
            self.state["operating_hours"] += dt * 24.0

        self.outputs_data = {
            "P_el": P_el,
            "P_th": P_th,
            "P_th_available": P_th,  # Key expected by HeatingSystem
            "Q_gas_consumed": Q_gas_consumed,
            "Q_gas_out_m3_per_day": Q_gas_consumed,  # Demand signal for storages
            "Q_ch4_remaining": Q_ch4_available - Q_ch4_consumed,
            "CH4_fraction": CH4_content,
            "P_consumed": P_el * self.own_consumption_fraction,
            "P_el_net": P_el * (1.0 - self.own_consumption_fraction),
        }

        return self.outputs_data

    def to_dict(self) -> dict[str, Any]:
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
            "ch4_content_default": self.ch4_content_default,
            "lhv_ch4": self.lhv_ch4,
            "own_consumption_fraction": self.own_consumption_fraction,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> CHP:
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
            ch4_content_default=config.get("ch4_content_default", 0.60),
            lhv_ch4=config.get("lhv_ch4", 10.0),
            own_consumption_fraction=config.get("own_consumption_fraction", 0.03),
        )

        chp.inputs = config.get("inputs", [])
        chp.outputs = config.get("outputs", [])

        if "state" in config:
            chp.initialize(config["state"])
        else:
            chp.initialize()

        return chp
