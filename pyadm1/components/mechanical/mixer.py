# pyadm1/components/mechanical/mixer.py
"""
Mixer and Agitator Components for Biogas Digesters
This module provides mixer/agitator models for maintaining homogeneity in
anaerobic digesters. Includes models for different mixer types with power
consumption calculations based on fluid properties and operating conditions.

Mixer Types:
- Propeller mixers (axial flow)
- Paddle mixers (radial flow)
- Jet mixers (hydraulic mixing)

Power consumption is calculated based on:
- Mixing intensity and speed
- Fluid viscosity and density
- Tank geometry (diameter, height)
- Mixer geometry (impeller diameter, blade configuration)

References:
- VDI 2167 (2006): Gärtechnik in Biogasanlagen (TODO: Quelle stimmt nicht)
- Nienow, A.W. (1997): On impeller circulation and mixing effectiveness
- Paul, E.L. et al. (2004): Handbook of Industrial Mixing: Science and Practice

Example:
    >>> from pyadm1.components.mechanical import Mixer
    >>>
    >>> # Propeller mixer for 2000 m³ digester
    >>> mixer = Mixer(
    ...     component_id="mix1",
    ...     mixer_type="propeller",
    ...     tank_volume=2000,
    ...     tank_diameter=15,
    ...     mixing_intensity="medium",
    ...     power_installed=15.0
    ... )
    >>> mixer.initialize()
    >>> result = mixer.step(t=0, dt=1/24, inputs={})
    >>> print(f"Power consumption: {result['P_consumed']:.1f} kW")
"""

from typing import Dict, Any, Optional
from enum import Enum
import numpy as np
from ..base import Component, ComponentType


class MixerType(str, Enum):
    """Enumeration of mixer types."""

    PROPELLER = "propeller"
    PADDLE = "paddle"
    JET = "jet"


class MixingIntensity(str, Enum):
    """Enumeration of mixing intensity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Mixer(Component):
    """
    Mixer/agitator component for biogas digesters.
    Models mechanical or hydraulic mixing systems that maintain homogeneity
    in anaerobic digesters. Calculates power consumption based on mixer type,
    operating conditions, and fluid properties.

    Attributes:
        mixer_type: Type of mixer (propeller, paddle, jet)
        tank_volume: Tank volume [m³]
        tank_diameter: Tank diameter [m]
        tank_height: Tank height [m]
        mixing_intensity: Mixing intensity level
        power_installed: Installed mixer power [kW]
        impeller_diameter: Impeller diameter [m]
        operating_speed: Mixer rotational speed [rpm]
        intermittent: Intermittent operation mode
        on_time_fraction: Fraction of time mixer is on (0-1)

    Example:
        >>> mixer = Mixer(
        ...     "mix1",
        ...     mixer_type="propeller",
        ...     tank_volume=2000,
        ...     mixing_intensity="medium"
        ... )
        >>> mixer.initialize()
        >>> result = mixer.step(0, 1/24, {})
    """

    def __init__(
        self,
        component_id: str,
        mixer_type: str = "propeller",
        tank_volume: float = 2000.0,
        tank_diameter: Optional[float] = None,
        tank_height: Optional[float] = None,
        mixing_intensity: str = "medium",
        power_installed: Optional[float] = None,
        impeller_diameter: Optional[float] = None,
        operating_speed: Optional[float] = None,
        intermittent: bool = True,
        on_time_fraction: float = 0.25,
        name: Optional[str] = None,
    ):
        """
        Initialize mixer component.

        Args:
            component_id: Unique identifier
            mixer_type: Type of mixer ("propeller", "paddle", "jet")
            tank_volume: Tank liquid volume [m³]
            tank_diameter: Tank diameter [m] (calculated if None)
            tank_height: Tank height [m] (calculated if None)
            mixing_intensity: Intensity level ("low", "medium", "high")
            power_installed: Installed power [kW] (calculated if None)
            impeller_diameter: Impeller diameter [m] (calculated if None)
            operating_speed: Rotational speed [rpm] (calculated if None)
            intermittent: Enable intermittent operation
            on_time_fraction: Fraction of time mixer is on (0-1)
            name: Human-readable name
        """
        super().__init__(component_id, ComponentType.MIXER, name)

        # Mixer configuration
        self.mixer_type = MixerType(mixer_type.lower())
        self.mixing_intensity = MixingIntensity(mixing_intensity.lower())
        self.intermittent = intermittent
        self.on_time_fraction = min(1.0, max(0.0, on_time_fraction))

        # Tank geometry
        self.tank_volume = tank_volume
        self.tank_diameter = tank_diameter or self._estimate_tank_diameter(tank_volume)
        self.tank_height = tank_height or self._estimate_tank_height(tank_volume, self.tank_diameter)

        # Mixer geometry and power
        self.impeller_diameter = impeller_diameter or self._estimate_impeller_diameter()
        self.operating_speed = operating_speed or self._estimate_operating_speed()
        self.power_installed = power_installed or self._estimate_power_requirement()

        # Fluid properties (typical biogas substrate)
        self.fluid_density = 1020.0  # kg/m³
        self.fluid_viscosity = 0.050  # Pa·s (50 mPa·s, typical for biogas substrate)

        # Operating state
        self.is_running = True
        self.current_speed_fraction = 1.0  # Fraction of nominal speed
        self.operating_hours = 0.0
        self.energy_consumed = 0.0

        # Performance tracking
        self.mixing_time = 0.0  # Time to achieve homogeneity [min]
        self.power_number = 0.0  # Dimensionless power number
        self.reynolds_number = 0.0  # Reynolds number for mixing

        # Initialize state
        self.initialize()

    def initialize(self, initial_state: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize mixer state.

        Args:
            initial_state: Optional initial state dictionary with keys:
                - 'is_running': Mixer running state
                - 'current_speed_fraction': Speed fraction (0-1)
                - 'operating_hours': Cumulative operating hours
                - 'energy_consumed': Cumulative energy [kWh]
        """
        if initial_state:
            self.is_running = initial_state.get("is_running", True)
            self.current_speed_fraction = initial_state.get("current_speed_fraction", 1.0)
            self.operating_hours = initial_state.get("operating_hours", 0.0)
            self.energy_consumed = initial_state.get("energy_consumed", 0.0)

        # Calculate initial performance parameters
        self._calculate_mixing_parameters()

        self.state = {
            "is_running": self.is_running,
            "current_speed_fraction": self.current_speed_fraction,
            "operating_hours": self.operating_hours,
            "energy_consumed": self.energy_consumed,
            "power_number": self.power_number,
            "reynolds_number": self.reynolds_number,
            "mixing_time": self.mixing_time,
        }

        self.outputs_data = {"P_consumed": 0.0, "is_running": self.is_running, "mixing_quality": 1.0}

        self._initialized = True

    def step(self, t: float, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform one simulation time step.

        Args:
            t: Current time [days]
            dt: Time step [days]
            inputs: Input data with optional keys:
                - 'speed_setpoint': Desired speed fraction (0-1)
                - 'enable_mixing': Enable/disable mixer
                - 'fluid_viscosity': Fluid viscosity [Pa·s]
                - 'temperature': Fluid temperature [K]

        Returns:
            Dict with keys:
                - 'P_consumed': Power consumption [kW]
                - 'P_average': Time-averaged power [kW]
                - 'is_running': Current running state
                - 'mixing_quality': Mixing quality index (0-1)
                - 'reynolds_number': Reynolds number
                - 'power_number': Power number
                - 'mixing_time': Mixing time [min]
                - 'shear_rate': Average shear rate [1/s]
        """
        # Update fluid properties if provided
        if "fluid_viscosity" in inputs:
            self.fluid_viscosity = inputs["fluid_viscosity"]

        if "temperature" in inputs:
            self._update_fluid_properties(inputs["temperature"])

        # Update operating state
        enable_mixing = inputs.get("enable_mixing", True)
        speed_setpoint = inputs.get("speed_setpoint", 1.0)

        # Intermittent operation logic
        if self.intermittent:
            # Simple on/off cycling based on on_time_fraction
            cycle_time = 1.0  # 1 hour cycle
            time_in_cycle = (t % cycle_time) / cycle_time
            self.is_running = time_in_cycle < self.on_time_fraction and enable_mixing
        else:
            self.is_running = enable_mixing

        # Update speed
        # take max of speed_setpoint, 0.0 - to avoid wrong calculations if accidently a negative setpoint is given
        self.current_speed_fraction = max(speed_setpoint, 0.0) if self.is_running else 0.0

        # Calculate mixing parameters
        self._calculate_mixing_parameters()

        # Calculate power consumption
        P_consumed = self._calculate_power_consumption()

        # Calculate time-averaged power (accounting for intermittent operation)
        if self.intermittent:
            P_average = P_consumed * self.on_time_fraction
        else:
            P_average = P_consumed

        # Calculate mixing quality
        mixing_quality = self._calculate_mixing_quality()

        # Calculate average shear rate
        shear_rate = self._calculate_shear_rate()

        # Update cumulative values
        dt_hours = dt * 24.0
        if self.is_running:
            self.operating_hours += dt_hours
        self.energy_consumed += P_consumed * dt_hours

        # Update state
        self.state.update(
            {
                "is_running": self.is_running,
                "current_speed_fraction": self.current_speed_fraction,
                "operating_hours": self.operating_hours,
                "energy_consumed": self.energy_consumed,
                "power_number": self.power_number,
                "reynolds_number": self.reynolds_number,
                "mixing_time": self.mixing_time,
            }
        )

        # Prepare outputs
        self.outputs_data = {
            "P_consumed": float(P_consumed),
            "P_average": float(P_average),
            "is_running": bool(self.is_running),
            "mixing_quality": float(mixing_quality),
            "reynolds_number": float(self.reynolds_number),
            "power_number": float(self.power_number),
            "mixing_time": float(self.mixing_time),
            "shear_rate": float(shear_rate),
            "specific_power": float(P_consumed / self.tank_volume),  # kW/m³
            "tip_speed": float(self._calculate_tip_speed()),  # m/s
        }

        return self.outputs_data

    def _calculate_mixing_parameters(self) -> None:
        """Calculate mixing performance parameters."""
        # Current rotational speed
        N = self.operating_speed * self.current_speed_fraction / 60.0  # Hz (rev/s)
        D = self.impeller_diameter

        # Reynolds number for mixing
        # Re = ρ * N * D² / μ
        self.reynolds_number = self.fluid_density * N * D**2 / self.fluid_viscosity

        # Power number (depends on mixer type and Reynolds number)
        self.power_number = self._calculate_power_number()

        # Mixing time estimation (Nienow correlation)
        # θ_mix = C * (D_T/D)^α * (H/D_T)^β / N
        # where C, α, β depend on mixer type
        D_T = self.tank_diameter
        H = self.tank_height

        if self.mixer_type == MixerType.PROPELLER:
            C, alpha, beta = 5.3, 2.0, 0.5
        elif self.mixer_type == MixerType.PADDLE:
            C, alpha, beta = 6.5, 2.5, 0.7
        else:  # JET
            C, alpha, beta = 4.0, 1.5, 0.3

        if N > 1e-6:  # Only calculate if mixer is actually running
            self.mixing_time = C * (D_T / D) ** alpha * (H / D_T) ** beta / (N * 60.0)  # minutes
        else:
            self.mixing_time = float("inf")

    def _calculate_power_number(self) -> float:
        """
        Calculate power number based on Reynolds number and mixer type.

        When mixer is not running (Re < 1e-6), returns a safe default value.

        Returns:
            Power number (dimensionless)
        """
        Re = self.reynolds_number

        # Handle zero Reynolds number (mixer not running)
        if Re < 1e-6:
            # Return a reasonable default for the mixer type
            if self.mixer_type == MixerType.PROPELLER:
                return 0.32  # Turbulent regime value
            elif self.mixer_type == MixerType.PADDLE:
                return 5.0
            else:  # JET
                return 0.1

        if self.mixer_type == MixerType.PROPELLER:
            # Propeller: transition from laminar to turbulent
            if Re < 100:
                Np = 14.0 * Re ** (-0.67)  # Laminar regime
            elif Re < 10000:
                Np = 1.2 * Re ** (-0.15)  # Transition regime
            else:
                Np = 0.32  # Turbulent regime (constant)

        elif self.mixer_type == MixerType.PADDLE:
            # Paddle mixer
            if Re < 10:
                Np = 300.0 / Re
            elif Re < 10000:
                Np = 8.0 * Re ** (-0.25)
            else:
                Np = 5.0

        else:  # JET
            # Jet mixer (based on jet momentum)
            # Power number is not directly applicable, use empirical value
            Np = 0.1

        return Np

    def _calculate_power_consumption(self) -> float:
        """
        Calculate actual power consumption.

        Returns:
            Power consumption [kW]
        """
        if not self.is_running:
            return 0.0

        # Mechanical power from power number correlation
        # P = Np * ρ * N³ * D⁵
        N = self.operating_speed * self.current_speed_fraction / 60.0  # Hz
        D = self.impeller_diameter

        P_mech = self.power_number * self.fluid_density * N**3 * D**5 / 1000.0  # Convert W to kW

        # Account for motor efficiency (typical 85-95%)
        motor_efficiency = 0.90
        P_electrical = P_mech / motor_efficiency

        # Limit to installed power
        P_actual = min(P_electrical, self.power_installed)

        return P_actual

    def _calculate_mixing_quality(self) -> float:
        """
        Calculate mixing quality index based on mixing time and intensity.

        Returns:
            Mixing quality (0-1, where 1 is perfect mixing)
        """
        if not self.is_running:
            return 0.0

        # Quality based on mixing time
        # Good mixing: < 5 min, Poor mixing: > 30 min
        if self.mixing_time < 5.0:
            quality = 1.0
        elif self.mixing_time > 30.0:
            quality = 0.3
        else:
            quality = 1.0 - 0.7 * (self.mixing_time - 5.0) / 25.0

        # Adjust for speed fraction
        quality *= self.current_speed_fraction

        # Adjust for Reynolds number (laminar flow reduces quality)
        if self.reynolds_number < 1000:
            quality *= self.reynolds_number / 1000.0

        return min(1.0, quality)

    def _calculate_shear_rate(self) -> float:
        """
        Calculate average shear rate in the tank.

        Returns:
            Average shear rate [1/s]
        """
        # Average shear rate estimation
        # γ̇ ≈ k * N * (D/D_T)
        # where k is a constant (typically 10-15 for propeller mixers)

        N = self.operating_speed * self.current_speed_fraction / 60.0  # Hz
        D = self.impeller_diameter
        D_T = self.tank_diameter

        if self.mixer_type == MixerType.PROPELLER:
            k = 13.0
        elif self.mixer_type == MixerType.PADDLE:
            k = 11.0
        else:  # JET
            k = 8.0

        shear_rate = k * N * (D / D_T)

        return shear_rate

    def _calculate_tip_speed(self) -> float:
        """
        Calculate impeller tip speed.

        Returns:
            Tip speed [m/s]
        """
        N = self.operating_speed * self.current_speed_fraction / 60.0  # Hz
        D = self.impeller_diameter

        tip_speed = np.pi * N * D

        return tip_speed

    def _update_fluid_properties(self, temperature: float) -> None:
        """
        Update fluid properties based on temperature.

        Args:
            temperature: Fluid temperature [K]
        """
        # Viscosity temperature dependence (Arrhenius-type)
        # μ(T) = μ₀ * exp(Ea/R * (1/T - 1/T₀))
        T_0 = 308.15  # Reference temperature (35°C)
        mu_0 = 0.050  # Reference viscosity [Pa·s]
        Ea_R = 2000.0  # Activation energy / gas constant [K]

        self.fluid_viscosity = mu_0 * np.exp(Ea_R * (1 / temperature - 1 / T_0))

    @staticmethod
    def _estimate_tank_diameter(volume: float) -> float:
        """
        Estimate tank diameter from volume assuming cylindrical tank.

        Args:
            volume: Tank volume [m³]

        Returns:
            Estimated diameter [m]
        """
        # Assume H/D = 1.5 (typical for digesters)
        # V = π/4 * D² * H = π/4 * D² * 1.5*D = 1.178 * D³
        diameter = (volume / 1.178) ** (1 / 3)
        return diameter

    @staticmethod
    def _estimate_tank_height(volume: float, diameter: float) -> float:
        """
        Estimate tank height from volume and diameter.

        Args:
            volume: Tank volume [m³]
            diameter: Tank diameter [m]

        Returns:
            Estimated height [m]
        """
        # V = π/4 * D² * H
        height = volume / (np.pi / 4 * diameter**2)
        return height

    def _estimate_impeller_diameter(self) -> float:
        """
        Estimate impeller diameter based on tank size and mixer type.

        Returns:
            Impeller diameter [m]
        """
        D_T = self.tank_diameter

        # D/D_T ratio depends on mixer type
        if self.mixer_type == MixerType.PROPELLER:
            ratio = 0.33  # Typically 1/3 tank diameter
        elif self.mixer_type == MixerType.PADDLE:
            ratio = 0.50  # Larger for paddles
        else:  # JET
            ratio = 0.10  # Jet nozzle diameter

        return ratio * D_T

    def _estimate_operating_speed(self) -> float:
        """
        Estimate operating speed based on mixer type and tank size.

        Returns:
            Operating speed [rpm]
        """
        if self.mixer_type == MixerType.PROPELLER:
            # Propellers: typically 40-100 rpm for large digesters
            speed = 60.0
        elif self.mixer_type == MixerType.PADDLE:
            # Paddles: typically 20-60 rpm
            speed = 40.0
        else:  # JET
            # Jet mixers: recirculation pump speed
            speed = 1450.0  # Typical pump speed

        # Scale with tank size (smaller tanks → higher speed)
        scale_factor = (2000.0 / self.tank_volume) ** (1 / 3)
        speed *= scale_factor

        return speed

    def _estimate_power_requirement(self) -> float:
        """
        Estimate power requirement based on mixing intensity and tank volume.

        Returns:
            Power requirement [kW]
        """
        # Specific power input [W/m³] depends on intensity
        if self.mixing_intensity == MixingIntensity.LOW:
            specific_power = 3.0  # W/m³
        elif self.mixing_intensity == MixingIntensity.MEDIUM:
            specific_power = 5.0  # W/m³
        else:  # HIGH
            specific_power = 8.0  # W/m³

        # Total power
        power = specific_power * self.tank_volume / 1000.0  # kW

        # Adjust for mixer type
        if self.mixer_type == MixerType.PROPELLER:
            power *= 1.0  # Baseline
        elif self.mixer_type == MixerType.PADDLE:
            power *= 1.2  # Paddles typically need more power
        else:  # JET
            power *= 1.5  # Jet mixers include pump power

        return power

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize mixer to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "component_id": self.component_id,
            "component_type": self.component_type.value,
            "name": self.name,
            "mixer_type": self.mixer_type.value,
            "mixing_intensity": self.mixing_intensity.value,
            "tank_volume": self.tank_volume,
            "tank_diameter": self.tank_diameter,
            "tank_height": self.tank_height,
            "impeller_diameter": self.impeller_diameter,
            "operating_speed": self.operating_speed,
            "power_installed": self.power_installed,
            "intermittent": self.intermittent,
            "on_time_fraction": self.on_time_fraction,
            "fluid_density": self.fluid_density,
            "fluid_viscosity": self.fluid_viscosity,
            "state": self.state,
            "inputs": self.inputs,
            "outputs": self.outputs,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "Mixer":
        """
        Create mixer from dictionary.

        Args:
            config: Configuration dictionary

        Returns:
            Mixer instance
        """
        mixer = cls(
            component_id=config["component_id"],
            mixer_type=config.get("mixer_type", "propeller"),
            tank_volume=config.get("tank_volume", 2000.0),
            tank_diameter=config.get("tank_diameter"),
            tank_height=config.get("tank_height"),
            mixing_intensity=config.get("mixing_intensity", "medium"),
            power_installed=config.get("power_installed"),
            impeller_diameter=config.get("impeller_diameter"),
            operating_speed=config.get("operating_speed"),
            intermittent=config.get("intermittent", True),
            on_time_fraction=config.get("on_time_fraction", 0.25),
            name=config.get("name"),
        )

        # Restore state if present
        if "state" in config:
            mixer.initialize(config["state"])

        mixer.inputs = config.get("inputs", [])
        mixer.outputs = config.get("outputs", [])

        return mixer
