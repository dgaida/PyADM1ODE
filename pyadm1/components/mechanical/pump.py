# pyadm1/components/mechanical/pump.py
"""
Pump Components for Biogas Plant Material Handling

This module provides pump models for substrate feeding, recirculation, and
digestate handling in biogas plants. Includes models for different pump types
with power consumption calculations based on flow rate and pressure head.

Pump Types:
- Centrifugal pumps (for liquids with low viscosity)
- Progressive cavity pumps (for viscous slurries)
- Piston pumps (for high-pressure applications)

Power consumption is calculated based on:
- Flow rate and pressure head
- Pump efficiency and mechanical losses
- Fluid properties (density, viscosity)
- Operating point on pump curve

References:
- VDI 2067: Economic efficiency of building installations
- Karassik, I.J. et al. (2008): Pump Handbook, 4th Edition
- Gülich, J.F. (2014): Centrifugal Pumps, 3rd Edition

Example:
    >>> from pyadm1.components.mechanical import Pump
    >>>
    >>> # Substrate feeding pump
    >>> pump = Pump(
    ...     component_id="pump1",
    ...     pump_type="progressive_cavity",
    ...     Q_nom=15.0,
    ...     pressure_head=5.0
    ... )
    >>> pump.initialize()
    >>> result = pump.step(t=0, dt=1/24, inputs={'Q_setpoint': 12.0})
    >>> print(f"Power: {result['P_consumed']:.1f} kW")
"""

from typing import Dict, Any, Optional
from enum import Enum

from ..base import Component, ComponentType


class PumpType(str, Enum):
    """Enumeration of pump types."""

    CENTRIFUGAL = "centrifugal"
    PROGRESSIVE_CAVITY = "progressive_cavity"
    PISTON = "piston"


# TODO: pump has to be connected between digesters or between substrate feed and primary digester so that it knows
#  how much fluid it is pumping (m^3/d).
# TODO: it must be a mistake that Q_nom: Nominal flow rate [m³/h], this must be m^3/d.


class Pump(Component):
    """
    Pump component for material handling in biogas plants.

    Models different pump types for substrate feeding, recirculation, and
    digestate transfer. Calculates power consumption based on flow rate,
    pressure head, and pump efficiency.

    Attributes:
        pump_type: Type of pump (centrifugal, progressive_cavity, piston)
        Q_nom: Nominal flow rate [m³/h]
        pressure_head: Pressure head [m] or [bar]
        efficiency: Pump efficiency at nominal point (0-1)
        motor_efficiency: Motor efficiency (0-1)
        fluid_density: Fluid density [kg/m³]
        speed_control: Enable variable speed drive (VSD)
        current_flow: Current flow rate [m³/h]
        is_running: Pump operating state

    Example:
        >>> pump = Pump(
        ...     "feed_pump",
        ...     pump_type="progressive_cavity",
        ...     Q_nom=10.0,
        ...     pressure_head=50.0
        ... )
        >>> pump.initialize()
        >>> result = pump.step(0, 1/24, {'Q_setpoint': 8.0})
    """

    def __init__(
        self,
        component_id: str,
        pump_type: str = "progressive_cavity",
        Q_nom: float = 10.0,
        pressure_head: float = 50.0,
        efficiency: Optional[float] = None,
        motor_efficiency: float = 0.90,
        fluid_density: float = 1020.0,
        speed_control: bool = True,
        name: Optional[str] = None,
    ):
        """
        Initialize pump component.

        Args:
            component_id: Unique identifier
            pump_type: Type of pump ("centrifugal", "progressive_cavity", "piston")
            Q_nom: Nominal flow rate [m³/h]
            pressure_head: Design pressure head [m]
            efficiency: Pump efficiency (0-1), calculated if None
            motor_efficiency: Motor efficiency (0-1)
            fluid_density: Fluid density [kg/m³]
            speed_control: Enable variable speed drive
            name: Human-readable name
        """
        super().__init__(component_id, ComponentType.MIXER, name)  # Use MIXER as closest type

        # Pump configuration
        self.pump_type = PumpType(pump_type.lower())
        self.Q_nom = Q_nom
        self.pressure_head = pressure_head
        self.motor_efficiency = motor_efficiency
        self.fluid_density = fluid_density
        self.speed_control = speed_control

        # Calculate default efficiency based on pump type and size
        self.efficiency = efficiency or self._estimate_pump_efficiency()

        # Operating state
        self.current_flow = 0.0
        self.is_running = False
        self.operating_hours = 0.0
        self.energy_consumed = 0.0
        self.total_volume_pumped = 0.0

        # Performance tracking
        self.actual_efficiency = self.efficiency
        self.speed_fraction = 1.0

        # Initialize state
        self.initialize()

    def initialize(self, initial_state: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize pump state.

        Args:
            initial_state: Optional initial state dictionary with keys:
                - 'is_running': Initial pump state
                - 'current_flow': Initial flow rate [m³/h]
                - 'operating_hours': Cumulative operating hours
                - 'energy_consumed': Cumulative energy [kWh]
                - 'total_volume_pumped': Cumulative volume [m³]
        """
        if initial_state:
            self.is_running = initial_state.get("is_running", False)
            self.current_flow = initial_state.get("current_flow", 0.0)
            self.operating_hours = initial_state.get("operating_hours", 0.0)
            self.energy_consumed = initial_state.get("energy_consumed", 0.0)
            self.total_volume_pumped = initial_state.get("total_volume_pumped", 0.0)

        self.state = {
            "is_running": self.is_running,
            "current_flow": self.current_flow,
            "operating_hours": self.operating_hours,
            "energy_consumed": self.energy_consumed,
            "total_volume_pumped": self.total_volume_pumped,
            "efficiency": self.efficiency,
            "speed_fraction": self.speed_fraction,
        }

        self.outputs_data = {
            "P_consumed": 0.0,
            "Q_actual": 0.0,
            "is_running": self.is_running,
            "efficiency": self.efficiency,
            "pressure_actual": 0.0,
        }

        self._initialized = True

    def step(self, t: float, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform one simulation time step.

        Args:
            t: Current time [days]
            dt: Time step [days]
            inputs: Input data with optional keys:
                - 'Q_setpoint': Desired flow rate [m³/h]
                - 'enable_pump': Enable/disable pump
                - 'fluid_density': Fluid density [kg/m³]
                - 'fluid_viscosity': Fluid viscosity [Pa·s]
                - 'pressure_head': Required pressure head [m]

        Returns:
            Dict with keys:
                - 'P_consumed': Power consumption [kW]
                - 'Q_actual': Actual flow rate [m³/h]
                - 'is_running': Current running state
                - 'efficiency': Current operating efficiency
                - 'pressure_actual': Actual pressure head [m]
                - 'speed_fraction': Speed as fraction of nominal
        """
        # Update fluid properties if provided
        if "fluid_density" in inputs:
            self.fluid_density = inputs["fluid_density"]

        # Update operating state
        enable_pump = inputs.get("enable_pump", True)
        Q_setpoint = inputs.get("Q_setpoint", self.Q_nom)
        pressure_head_req = inputs.get("pressure_head", self.pressure_head)

        # Determine if pump should run
        self.is_running = enable_pump and Q_setpoint > 0

        if not self.is_running:
            # Pump is off
            self.current_flow = 0.0
            self.speed_fraction = 0.0
            P_consumed = 0.0
            pressure_actual = 0.0

        else:
            # Calculate operating point
            if self.speed_control:
                # Variable speed: adjust speed to match flow
                self.speed_fraction = min(1.2, Q_setpoint / self.Q_nom)  # Allow 20% overload
                Q_actual = min(Q_setpoint, self.Q_nom * 1.2)
            else:
                # Fixed speed: flow is nominal
                self.speed_fraction = 1.0
                Q_actual = self.Q_nom

            self.current_flow = Q_actual

            # Calculate efficiency at operating point
            self.actual_efficiency = self._calculate_efficiency_at_operating_point(Q_actual, pressure_head_req)

            # Calculate actual pressure head (may differ from required)
            pressure_actual = self._calculate_pressure_head(Q_actual)

            # Calculate power consumption
            P_consumed = self._calculate_power_consumption(Q_actual, pressure_head_req)

        # Update cumulative values
        dt_hours = dt * 24.0
        if self.is_running:
            self.operating_hours += dt_hours
            self.total_volume_pumped += self.current_flow * dt_hours

        self.energy_consumed += P_consumed * dt_hours

        # Update state
        self.state.update(
            {
                "is_running": self.is_running,
                "current_flow": self.current_flow,
                "operating_hours": self.operating_hours,
                "energy_consumed": self.energy_consumed,
                "total_volume_pumped": self.total_volume_pumped,
                "efficiency": self.actual_efficiency,
                "speed_fraction": self.speed_fraction,
            }
        )

        # Prepare outputs
        self.outputs_data = {
            "P_consumed": float(P_consumed),
            "Q_actual": float(self.current_flow),
            "is_running": bool(self.is_running),
            "efficiency": float(self.actual_efficiency),
            "pressure_actual": float(pressure_actual),
            "speed_fraction": float(self.speed_fraction),
            "specific_energy": float(P_consumed / max(self.current_flow, 1e-6)),  # kWh/m³
        }

        return self.outputs_data

    def _estimate_pump_efficiency(self) -> float:
        """
        Estimate pump efficiency based on type and size.

        Uses empirical correlations from pump handbooks.

        Returns:
            Estimated pump efficiency (0-1)
        """
        # Efficiency increases with pump size
        # Based on Gülich (2014) correlations

        if self.pump_type == PumpType.CENTRIFUGAL:
            # Centrifugal pumps: 65-85% for typical biogas applications
            if self.Q_nom < 10:
                eta = 0.65
            elif self.Q_nom < 50:
                eta = 0.70
            else:
                eta = 0.75

        elif self.pump_type == PumpType.PROGRESSIVE_CAVITY:
            # Progressive cavity: 50-75% (volumetric pumps are less efficient)
            if self.Q_nom < 10:
                eta = 0.50
            elif self.Q_nom < 50:
                eta = 0.60
            else:
                eta = 0.70

        else:  # PISTON
            # Piston pumps: 70-85%
            if self.Q_nom < 10:
                eta = 0.70
            elif self.Q_nom < 50:
                eta = 0.75
            else:
                eta = 0.80

        return eta

    def _calculate_efficiency_at_operating_point(self, Q: float, H: float) -> float:
        """
        Calculate pump efficiency at current operating point.

        Efficiency varies with flow rate and head. Maximum efficiency
        occurs at design point (Q_nom, H_nom).

        Args:
            Q: Flow rate [m³/h]
            H: Pressure head [m]

        Returns:
            Operating efficiency (0-1)
        """
        if Q <= 0:
            return 0.0

        # Calculate relative flow
        Q_rel = Q / self.Q_nom

        # Efficiency curve (parabolic approximation)
        # Maximum at Q_opt ≈ 1.0 × Q_nom
        Q_opt = 1.0

        if self.pump_type == PumpType.CENTRIFUGAL:
            # Centrifugal pumps have broader efficiency curve
            # eta(Q) = eta_max * (1 - a*(Q/Q_opt - 1)²)
            a = 0.3
            eta = self.efficiency * (1 - a * (Q_rel / Q_opt - 1) ** 2)

        else:  # Volumetric pumps (PC, piston)
            # Volumetric pumps maintain efficiency better at part load
            # but drop off at overload
            if Q_rel <= 1.0:
                # Slight increase at part load due to reduced slip
                eta = self.efficiency * (0.95 + 0.05 * Q_rel)
            else:
                # Efficiency drops at overload
                a = 0.5
                eta = self.efficiency * (1 - a * (Q_rel - 1) ** 2)

        # Ensure reasonable bounds
        eta = max(0.1, min(eta, 0.95))

        return eta

    def _calculate_pressure_head(self, Q: float) -> float:
        """
        Calculate actual pressure head at given flow rate.

        Uses pump characteristic curve. For centrifugal pumps, head
        decreases with flow. For volumetric pumps, head is nearly constant.

        Args:
            Q: Flow rate [m³/h]

        Returns:
            Pressure head [m]
        """
        if self.pump_type == PumpType.CENTRIFUGAL:
            # Centrifugal pump curve: H = H0 - k*Q²
            # At Q_nom: H = H_nom
            # Estimate H0 ≈ 1.2 * H_nom
            H0 = 1.2 * self.pressure_head
            k = (H0 - self.pressure_head) / (self.Q_nom**2)
            H = H0 - k * Q**2

        else:  # Volumetric pumps
            # Nearly constant head (slight drop due to slip at high pressure)
            H = self.pressure_head * 0.98

        return max(0.0, H)

    def _calculate_power_consumption(self, Q: float, H: float) -> float:
        """
        Calculate electrical power consumption.

        Uses hydraulic power formula with efficiency corrections.

        Args:
            Q: Flow rate [m³/h]
            H: Pressure head [m]

        Returns:
            Power consumption [kW]
        """
        if Q <= 0:
            return 0.0

        # Convert flow to m³/s
        Q_m3_per_s = Q / 3600.0

        # Hydraulic power: P_hyd = ρ * g * Q * H [W]
        g = 9.81  # m/s²
        P_hydraulic = self.fluid_density * g * Q_m3_per_s * H / 1000.0  # kW

        # Shaft power (accounting for pump efficiency)
        P_shaft = P_hydraulic / max(self.actual_efficiency, 0.01)

        # Electrical power (accounting for motor efficiency)
        P_electrical = P_shaft / max(self.motor_efficiency, 0.01)

        return P_electrical

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize pump to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "component_id": self.component_id,
            "component_type": self.component_type.value,
            "name": self.name,
            "pump_type": self.pump_type.value,
            "Q_nom": self.Q_nom,
            "pressure_head": self.pressure_head,
            "efficiency": self.efficiency,
            "motor_efficiency": self.motor_efficiency,
            "fluid_density": self.fluid_density,
            "speed_control": self.speed_control,
            "state": self.state,
            "inputs": self.inputs,
            "outputs": self.outputs,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "Pump":
        """
        Create pump from dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            Pump instance
        """
        pump = cls(
            component_id=config["component_id"],
            pump_type=config.get("pump_type", "progressive_cavity"),
            Q_nom=config.get("Q_nom", 10.0),
            pressure_head=config.get("pressure_head", 50.0),
            efficiency=config.get("efficiency"),
            motor_efficiency=config.get("motor_efficiency", 0.90),
            fluid_density=config.get("fluid_density", 1020.0),
            speed_control=config.get("speed_control", True),
            name=config.get("name"),
        )

        # Restore state if present
        if "state" in config:
            pump.initialize(config["state"])

        pump.inputs = config.get("inputs", [])
        pump.outputs = config.get("outputs", [])

        return pump
