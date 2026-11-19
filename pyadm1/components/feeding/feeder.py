# pyadm1/components/feeding/feeder.py
"""
Feeder Component for Substrate Dosing

Models automated dosing systems for feeding substrates into biogas digesters.
Supports different feeder types with realistic operational characteristics
including dosing accuracy, capacity limits, and power consumption.

Feeder Types:
- Screw feeders (single/twin screw) for solid substrates
- Progressive cavity pumps for viscous slurries
- Piston feeders for fibrous materials
- Centrifugal pumps for liquid substrates
- Mixer wagons with integrated feeding

Features:
- Flow rate control with PID-like regulation
- Dosing accuracy modeling with variance
- Capacity and speed limits
- Power consumption calculation
- Blockage detection and handling
- Maintenance tracking

References:
- VDI 2263: Dust fires and dust explosions - Hazards, assessment, measures
- DIN 15101: Belt conveyors for bulk materials
- Schulte et al. (2017): Substrate feeding systems for biogas plants
- Chen et al. (2020): Precision feeding in anaerobic digestion

Example:
    >>> from pyadm1.components.feeding import Feeder
    >>>
    >>> # Screw feeder for corn silage
    >>> feeder = Feeder(
    ...     component_id="feed1",
    ...     feeder_type="screw",
    ...     Q_max=20,
    ...     substrate_type="solid"
    ... )
    >>> feeder.initialize()
    >>> result = feeder.step(t=0, dt=1/24, inputs={'Q_setpoint': 15})
    >>> print(f"Flow: {result['Q_actual']:.2f} m³/d")
"""

from __future__ import annotations

import zlib
from enum import Enum
from typing import Any, ClassVar

import numpy as np

from ..base import Component, ComponentType


class FeederType(str, Enum):
    """Enumeration of feeder types."""

    SCREW = "screw"
    TWIN_SCREW = "twin_screw"
    PROGRESSIVE_CAVITY = "progressive_cavity"
    PISTON = "piston"
    CENTRIFUGAL_PUMP = "centrifugal_pump"
    MIXER_WAGON = "mixer_wagon"


class SubstrateCategory(str, Enum):
    """Substrate physical categories."""

    SOLID = "solid"  # Silages, solid manure
    SLURRY = "slurry"  # Liquid manure, slurries
    LIQUID = "liquid"  # Water-like liquids
    FIBROUS = "fibrous"  # Straw, hay, fibrous waste


class Feeder(Component):
    """
    Feeder component for automated substrate dosing.

    Models feeding systems that transfer substrates from storage to digesters.
    Includes realistic operational characteristics like dosing accuracy,
    capacity limits, and power consumption.

    Attributes:
        feeder_type: Type of feeding system
        Q_max: Maximum flow rate [m³/d or t/d]
        substrate_type: Physical category of substrate
        dosing_accuracy: Accuracy of flow control (std dev as fraction)
        power_installed: Installed motor power [kW]
        current_flow: Current actual flow rate [m³/d or t/d]
        is_running: Operating state

    Example:
        >>> feeder = Feeder(
        ...     "feed1",
        ...     feeder_type="screw",
        ...     Q_max=20,
        ...     substrate_type="solid"
        ... )
        >>> feeder.initialize()
        >>> result = feeder.step(0, 1/24, {'Q_setpoint': 15})
    """

    def __init__(
        self,
        component_id: str,
        feeder_type: str | None = None,
        Q_max: float = 20.0,
        substrate_type: str | None = None,
        dosing_accuracy: float | None = None,
        power_installed: float | None = None,
        enable_dosing_noise: bool = True,
        name: str | None = None,
        Q_setpoint: float | None = None,
        substrate_index: int | None = None,
        flow_to_mass_factor: float = 1.0,
        specific_energy: float | None = None,
        seed: int | None = None,
    ):
        """
        Initialize feeder component.

        Args:
            component_id: Unique identifier
            feeder_type: Type of feeder ("screw", "progressive_cavity", etc.)
            Q_max: Maximum flow rate [m³/d or t/d]
            substrate_type: Substrate category ("solid", "slurry", "liquid", "fibrous")
            dosing_accuracy: Standard deviation of flow as fraction (auto if None)
            power_installed: Installed power [kW] (auto-calculated if None)
            enable_dosing_noise: Add realistic dosing variance
            name: Human-readable name
            Q_setpoint: Standing flow setpoint, in the unit of the digester's
                ``Q_substrates``. Used whenever the step inputs carry no
                ``Q_setpoint``; ``None`` leaves the feeder idle until one arrives.
            substrate_index: Slot of the digester's ``Q_substrates`` this
                feeder delivers into. Required to feed a digester in a plant.
            flow_to_mass_factor: Tonnes of fresh matter per unit of flow
                [t per flow unit]. 1.0 when the flow is in t/d (the default
                ``Feedstock`` convention); ``density / 1000`` when it is in m³/d.
                Converts the flow for the storage balance and for
                ``specific_energy``.
            specific_energy: Electricity demand per tonne fed [kWh/t FM]. When
                set it replaces the load-curve estimate from
                ``power_installed``; a blockage still multiplies it by 1.5.
            seed: Seed of the feeder's own random generator (dosing noise,
                blockages). ``None`` derives it from ``component_id``, so
                every feeder has its own stream and repeated runs give the
                same results.
        """
        super().__init__(component_id, ComponentType.FEEDER, name)

        if substrate_type is None:
            substrate_type = "solid" if feeder_type is None else self._default_substrate_for_feeder_type(feeder_type)

        if feeder_type is None:
            # Default to screw feeder for solid substrates
            feeder_type = "screw" if substrate_type.lower() in ["solid", "fibrous"] else "centrifugal_pump"

        # Configuration
        self.feeder_type = FeederType(feeder_type.lower())
        self.substrate_type = SubstrateCategory(substrate_type.lower())
        self._validate_feeder_substrate_compatibility()
        self.Q_max = float(Q_max)
        self.enable_dosing_noise = enable_dosing_noise
        self.Q_setpoint = None if Q_setpoint is None else float(Q_setpoint)
        self.substrate_index = None if substrate_index is None else int(substrate_index)
        if float(flow_to_mass_factor) <= 0.0:
            raise ValueError(f"Feeder '{component_id}': flow_to_mass_factor must be positive, got {flow_to_mass_factor}")
        self.flow_to_mass_factor = float(flow_to_mass_factor)
        self.specific_energy = None if specific_energy is None else float(specific_energy)

        # Dosing characteristics
        # An explicit 0.0 means exact dosing and must not fall back to the estimate.
        self.dosing_accuracy = self._estimate_dosing_accuracy() if dosing_accuracy is None else float(dosing_accuracy)
        self.power_installed = power_installed or self._estimate_power_requirement()

        self.seed = zlib.crc32(component_id.encode("utf-8")) if seed is None else int(seed)
        self._rng = np.random.default_rng(self.seed)

        # Operating state
        self.current_flow = 0.0
        self.is_running = False
        self.blockage_detected = False

        # Performance tracking
        self.operating_hours = 0.0
        self.energy_consumed = 0.0  # kWh
        self.total_mass_fed = 0.0  # t or m³
        self.n_starts = 0
        self.n_blockages = 0

        # Speed control (for variable speed feeders)
        self.speed_fraction = 1.0  # Fraction of nominal speed (0-1)

        # Initialize
        self.initialize()

    # feeder types that cannot handle the given substrate category
    _INCOMPATIBLE: ClassVar[dict[FeederType, set]] = {
        FeederType.SCREW: {SubstrateCategory.LIQUID},
        FeederType.TWIN_SCREW: {SubstrateCategory.LIQUID},
        FeederType.CENTRIFUGAL_PUMP: {
            SubstrateCategory.SOLID,
            SubstrateCategory.FIBROUS,
        },
        FeederType.PISTON: {SubstrateCategory.LIQUID},
        FeederType.PROGRESSIVE_CAVITY: {
            SubstrateCategory.SOLID,
            SubstrateCategory.FIBROUS,
        },
        FeederType.MIXER_WAGON: set(),  # handles all substrate types
    }

    @staticmethod
    def _default_substrate_for_feeder_type(feeder_type: str) -> str:
        """Infer a compatible default substrate category for a chosen feeder."""
        defaults = {
            FeederType.SCREW.value: SubstrateCategory.SOLID.value,
            FeederType.TWIN_SCREW.value: SubstrateCategory.SOLID.value,
            FeederType.PROGRESSIVE_CAVITY.value: SubstrateCategory.SLURRY.value,
            FeederType.PISTON.value: SubstrateCategory.FIBROUS.value,
            FeederType.CENTRIFUGAL_PUMP.value: SubstrateCategory.LIQUID.value,
            FeederType.MIXER_WAGON.value: SubstrateCategory.SOLID.value,
        }
        return defaults.get(feeder_type.lower(), SubstrateCategory.SOLID.value)

    def _validate_feeder_substrate_compatibility(self) -> None:
        """Raise ValueError if feeder type cannot handle the substrate category."""
        incompatible = self._INCOMPATIBLE.get(self.feeder_type, set())
        if self.substrate_type in incompatible:
            raise ValueError(
                f"Feeder type '{self.feeder_type.value}' is not compatible with "
                f"substrate category '{self.substrate_type.value}'. "
                f"Incompatible combinations for this feeder: "
                f"{[s.value for s in incompatible]}."
            )

    def initialize(self, initial_state: dict[str, Any] | None = None) -> None:
        """
        Initialize feeder state.

        Args:
            initial_state: Optional initial state with keys:
                - 'is_running': Initial operating state
                - 'current_flow': Initial flow rate [m³/d or t/d]
                - 'operating_hours': Cumulative operating hours
                - 'energy_consumed': Cumulative energy [kWh]
                - 'total_mass_fed': Cumulative mass [t or m³]
                - 'n_starts', 'n_blockages': Event counters
        """
        if initial_state:
            self.is_running = initial_state.get("is_running", False)
            self.current_flow = float(initial_state.get("current_flow", 0.0))
            self.operating_hours = float(initial_state.get("operating_hours", 0.0))
            self.energy_consumed = float(initial_state.get("energy_consumed", 0.0))
            self.total_mass_fed = float(initial_state.get("total_mass_fed", 0.0))
            self.n_starts = int(initial_state.get("n_starts", 0))
            self.n_blockages = int(initial_state.get("n_blockages", 0))

        self.state = {
            "is_running": self.is_running,
            "current_flow": self.current_flow,
            "operating_hours": self.operating_hours,
            "energy_consumed": self.energy_consumed,
            "total_mass_fed": self.total_mass_fed,
            "blockage_detected": self.blockage_detected,
            "n_starts": self.n_starts,
            "n_blockages": self.n_blockages,
        }

        self.outputs_data = {
            "Q_actual": 0.0,
            "is_running": False,
            "load_factor": 0.0,
            "P_consumed": 0.0,
            "blockage_detected": False,
        }

        self._initialized = True

    def step(self, t: float, dt: float, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Perform one simulation time step.

        Args:
            t: Current time [days]
            dt: Time step [days]
            inputs: Input data with optional keys:
                - 'Q_setpoint': Desired flow rate [m³/d or t/d]; falls back to
                  the ``Q_setpoint`` given to the constructor
                - 'enable_feeding': Enable/disable feeder
                - 'substrate_available': Amount available in storage [t or m³]
                - 'speed_setpoint': Desired speed fraction (0-1)

        Returns:
            Dict with keys:
                - 'Q_actual': Actual flow rate [m³/d or t/d]
                - 'is_running': Current operating state
                - 'load_factor': Operating load (0-1)
                - 'P_consumed': Power consumption [kW]
                - 'blockage_detected': Blockage alarm
                - 'dosing_error': Deviation from setpoint [%]
                - 'speed_fraction': Current speed fraction
        """
        # Get control inputs
        enable_feeding = inputs.get("enable_feeding", True)
        Q_setpoint = float(inputs.get("Q_setpoint", self.Q_setpoint or 0.0))
        substrate_available = float(inputs.get("substrate_available", float("inf")))
        speed_setpoint = float(inputs.get("speed_setpoint", 1.0))

        # Determine operating state
        should_run = enable_feeding and Q_setpoint > 0.01

        # Track starts
        if should_run and not self.is_running:
            self.n_starts += 1

        self.is_running = should_run

        if not self.is_running:
            # Feeder is off
            self.current_flow = 0.0
            self.speed_fraction = 0.0
            P_consumed = 0.0
            self.blockage_detected = False

        else:
            # Feeder is running

            # Update speed (for variable speed feeders)
            self.speed_fraction = min(1.0, speed_setpoint)

            # Calculate target flow
            Q_target = min(Q_setpoint, self.Q_max * self.speed_fraction)

            # Apply dosing noise (realistic variance)
            if self.enable_dosing_noise and Q_target > 0:
                noise = self._rng.normal(0, self.dosing_accuracy * Q_target)
                Q_actual = max(0.0, Q_target + noise)
            else:
                Q_actual = Q_target

            # Substrate availability limits the dosed flow, noise included:
            # the feeder cannot deliver more than the storage holds.
            max_available = substrate_available / dt  # Convert to daily rate
            Q_actual = min(Q_actual, max_available)

            self.current_flow = Q_actual

            # Random blockage simulation (very low probability)
            if self._rng.random() < 0.0001 * dt:  # ~0.01% per day
                self.blockage_detected = True
                self.n_blockages += 1
                self.current_flow *= 0.1  # Reduced flow during blockage
            else:
                self.blockage_detected = False

            # Calculate power consumption
            P_consumed = self._calculate_power_consumption()

        # Update cumulative values
        dt_hours = dt * 24.0
        if self.is_running:
            self.operating_hours += dt_hours
            self.total_mass_fed += self.current_flow * dt

        self.energy_consumed += P_consumed * dt_hours

        # Calculate load factor
        load_factor = self.current_flow / max(1e-6, self.Q_max) if self.is_running else 0.0

        # Calculate dosing error
        dosing_error = abs(self.current_flow - Q_setpoint) / Q_setpoint * 100 if Q_setpoint > 0 else 0.0

        # Update state
        self.state.update(
            {
                "is_running": self.is_running,
                "current_flow": self.current_flow,
                "operating_hours": self.operating_hours,
                "energy_consumed": self.energy_consumed,
                "total_mass_fed": self.total_mass_fed,
                "blockage_detected": self.blockage_detected,
                "n_starts": self.n_starts,
                "n_blockages": self.n_blockages,
            }
        )

        # Prepare outputs
        self.outputs_data = {
            "Q_actual": float(self.current_flow),
            "is_running": bool(self.is_running),
            "load_factor": float(load_factor),
            "P_consumed": float(P_consumed),
            "blockage_detected": bool(self.blockage_detected),
            "dosing_error": float(dosing_error),
            "speed_fraction": float(self.speed_fraction),
            "dosing_accuracy": float(self.dosing_accuracy),
            "total_mass_fed": float(self.total_mass_fed),
            "mass_flow_t_per_day": float(self.current_flow * self.flow_to_mass_factor),
        }
        if self.substrate_index is not None:
            self.outputs_data["substrate_index"] = self.substrate_index

        return self.outputs_data

    def _calculate_power_consumption(self) -> float:
        """
        Calculate power consumption based on load.

        Returns:
            Power consumption [kW]
        """
        if not self.is_running or self.current_flow < 1e-6:
            return 0.0

        if self.specific_energy is not None:
            # kWh/t * t/d / 24 h/d = kW
            P_specific = self.specific_energy * self.current_flow * self.flow_to_mass_factor / 24.0
            return P_specific * 1.5 if self.blockage_detected else P_specific

        # Base power (idling)
        P_base = self.power_installed * 0.2

        # Load-dependent power
        load_factor = self.current_flow / max(1e-6, self.Q_max)
        P_load = self.power_installed * 0.8 * load_factor

        # Blockage increases power consumption
        if self.blockage_detected:
            P_load *= 1.5

        return P_base + P_load

    def _estimate_dosing_accuracy(self) -> float:
        """
        Estimate dosing accuracy based on feeder type.

        Returns:
            Standard deviation as fraction of flow rate
        """
        # Typical dosing accuracy (std dev as fraction)
        accuracies = {
            FeederType.SCREW: 0.05,  # ±5%
            FeederType.TWIN_SCREW: 0.03,  # ±3% (better control)
            FeederType.PROGRESSIVE_CAVITY: 0.02,  # ±2% (volumetric)
            FeederType.PISTON: 0.01,  # ±1% (most accurate)
            FeederType.CENTRIFUGAL_PUMP: 0.08,  # ±8% (less precise)
            FeederType.MIXER_WAGON: 0.10,  # ±10% (batch feeding)
        }

        return accuracies.get(self.feeder_type, 0.05)

    def _estimate_power_requirement(self) -> float:
        """
        Estimate power requirement based on feeder type and capacity.

        TODO: Check these publications that provide numbers:

        Frey, J., Grüssing, F., Nägele, H. J., & Oechsner, H. (2013). Eigenstromverbrauch an Biogasanlagen senken:
        Der Einfluss neuer Techniken. agricultural engineering. eu, 68(1), 58-63.

         Naegele, H.-J., Lemmer, A., Oechsner, H., & Jungbluth, T. (2012). Electric Energy Consumption of the Full
         Scale Research Biogas Plant “Unterer Lindenhof”: Results of Longterm and Full Detail Measurements. Energies,
         5(12), 5198-5214. https://doi.org/10.3390/en5125198

        Returns:
            Power requirement [kW]
        """
        # Specific power requirements [kW per m³/h]
        specific_powers = {
            FeederType.SCREW: 0.8,
            FeederType.TWIN_SCREW: 1.0,
            FeederType.PROGRESSIVE_CAVITY: 1.2,
            FeederType.PISTON: 1.5,
            FeederType.CENTRIFUGAL_PUMP: 0.5,
            FeederType.MIXER_WAGON: 2.0,
        }

        specific_power = specific_powers.get(self.feeder_type, 0.8)

        Q_max_per_hour = self.Q_max / 24.0
        power = specific_power * Q_max_per_hour

        # Substrate type modifier: fibrous materials need more power, liquids less.
        if self.substrate_type == SubstrateCategory.FIBROUS:
            power *= 1.8
        elif self.substrate_type == SubstrateCategory.SOLID:
            power *= 1.4
        elif self.substrate_type == SubstrateCategory.LIQUID:
            power *= 0.7

        power *= 1.3  # safety margin

        return max(1.0, power)

    def to_dict(self) -> dict[str, Any]:
        """Serialize feeder to dictionary."""
        return {
            "component_id": self.component_id,
            "component_type": self.component_type.value,
            "name": self.name,
            "feeder_type": self.feeder_type.value,
            "substrate_type": self.substrate_type.value,
            "Q_max": self.Q_max,
            "dosing_accuracy": self.dosing_accuracy,
            "power_installed": self.power_installed,
            "enable_dosing_noise": self.enable_dosing_noise,
            "Q_setpoint": self.Q_setpoint,
            "substrate_index": self.substrate_index,
            "flow_to_mass_factor": self.flow_to_mass_factor,
            "specific_energy": self.specific_energy,
            "seed": self.seed,
            "state": self.state,
            "inputs": self.inputs,
            "outputs": self.outputs,
        }

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> Feeder:
        """Create feeder from dictionary."""
        feeder = cls(
            component_id=config["component_id"],
            feeder_type=config.get("feeder_type", "screw"),
            Q_max=config.get("Q_max", 20.0),
            substrate_type=config.get("substrate_type", "solid"),
            dosing_accuracy=config.get("dosing_accuracy"),
            power_installed=config.get("power_installed"),
            enable_dosing_noise=config.get("enable_dosing_noise", True),
            name=config.get("name"),
            Q_setpoint=config.get("Q_setpoint"),
            substrate_index=config.get("substrate_index"),
            flow_to_mass_factor=config.get("flow_to_mass_factor", 1.0),
            specific_energy=config.get("specific_energy"),
            seed=config.get("seed"),
        )

        # Restore state if present
        if "state" in config:
            feeder.initialize(config["state"])

        feeder.inputs = config.get("inputs", [])
        feeder.outputs = config.get("outputs", [])

        return feeder
