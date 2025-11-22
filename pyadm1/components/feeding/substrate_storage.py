# pyadm1/components/feeding/substrate_storage.py
"""
Substrate Storage Component

Models storage facilities for different substrate types including silos for
solid substrates and tanks for liquid substrates. Tracks inventory, quality
degradation over time, and capacity utilization.

Storage Types:
- Silos (horizontal, vertical, bunker) for solid substrates
- Tanks (above-ground, below-ground) for liquid substrates
- Open storage (clamps, piles) for seasonal substrates

Features:
- Inventory management with first-in-first-out (FIFO)
- Quality degradation modeling (dry matter losses, VS reduction)
- Temperature-dependent aerobic degradation
- Capacity monitoring with alerts
- Batch tracking for quality assurance

References:
- DIN 11622: Agricultural storage facilities
- VDI 3475: Emission control in agriculture, storage
- Herrmann et al. (2011): Biomass storage and handling
- Parra-Orobio et al. (2018): Storage effects on methane potential

Example:
    >>> from pyadm1.components.feeding import SubstrateStorage
    >>>
    >>> # Corn silage silo
    >>> silo = SubstrateStorage(
    ...     component_id="silo1",
    ...     storage_type="vertical_silo",
    ...     substrate_type="corn_silage",
    ...     capacity=1000,
    ...     initial_level=800
    ... )
    >>> silo.initialize()
    >>> result = silo.step(t=0, dt=1, inputs={'withdrawal_rate': 15})
    >>> print(f"Level: {result['current_level']:.1f} t")
"""

from typing import Dict, Any, Optional
from enum import Enum
import numpy as np

from pyadm1.components.base import Component, ComponentType


class StorageType(str, Enum):
    """Enumeration of storage types."""

    VERTICAL_SILO = "vertical_silo"
    HORIZONTAL_SILO = "horizontal_silo"
    BUNKER_SILO = "bunker_silo"
    ABOVE_GROUND_TANK = "above_ground_tank"
    BELOW_GROUND_TANK = "below_ground_tank"
    CLAMP = "clamp"
    PILE = "pile"


class SubstrateType(str, Enum):
    """Enumeration of substrate categories."""

    CORN_SILAGE = "corn_silage"
    GRASS_SILAGE = "grass_silage"
    WHOLE_CROP_SILAGE = "whole_crop_silage"
    MANURE_LIQUID = "manure_liquid"
    MANURE_SOLID = "manure_solid"
    BIOWASTE = "biowaste"
    FOOD_WASTE = "food_waste"
    GENERIC_SOLID = "generic_solid"
    GENERIC_LIQUID = "generic_liquid"


class SubstrateStorage(Component):
    """
    Storage facility component for biogas plant substrates.

    Models storage of different substrate types with inventory tracking,
    quality degradation, and capacity management. Supports both solid
    (silage, solid manure) and liquid (liquid manure, slurry) substrates.

    Attributes:
        storage_type: Type of storage facility
        substrate_type: Category of substrate stored
        capacity: Maximum storage capacity [t or m³]
        current_level: Current inventory level [t or m³]
        quality_factor: Current quality relative to fresh (0-1)
        degradation_rate: Quality degradation rate [1/d]
        density: Substrate bulk density [kg/m³]
        dry_matter: Dry matter content [%]
        vs_content: Volatile solids [% of DM]

    Example:
        >>> storage = SubstrateStorage(
        ...     "silo1",
        ...     storage_type="vertical_silo",
        ...     substrate_type="corn_silage",
        ...     capacity=1000,
        ...     initial_level=600
        ... )
        >>> storage.initialize()
        >>> outputs = storage.step(0, 1, {'withdrawal_rate': 15})
    """

    def __init__(
        self,
        component_id: str,
        storage_type: str = "vertical_silo",
        substrate_type: str = "corn_silage",
        capacity: float = 1000.0,
        initial_level: float = 0.0,
        degradation_rate: Optional[float] = None,
        temperature: float = 288.15,  # 15°C
        name: Optional[str] = None,
    ):
        """
        Initialize substrate storage component.

        Args:
            component_id: Unique identifier
            storage_type: Type of storage ("vertical_silo", "tank", etc.)
            substrate_type: Substrate category ("corn_silage", "manure_liquid", etc.)
            capacity: Maximum capacity [t or m³]
            initial_level: Initial inventory [t or m³]
            degradation_rate: Quality degradation rate [1/d] (auto-calculated if None)
            temperature: Storage temperature [K]
            name: Human-readable name
        """
        super().__init__(component_id, ComponentType.STORAGE, name)

        # Configuration
        self.storage_type = StorageType(storage_type.lower())
        self.substrate_type = SubstrateType(substrate_type.lower())
        self.capacity = float(capacity)
        self.temperature = float(temperature)

        # Inventory
        self.current_level = float(min(initial_level, capacity))
        self.quality_factor = 1.0  # 1.0 = fresh, 0.0 = completely degraded

        # Substrate properties (typical values, can be updated)
        self._set_substrate_properties()

        # Degradation parameters
        self.degradation_rate = degradation_rate or self._estimate_degradation_rate()

        # Storage losses
        self.cumulative_losses = 0.0  # Total mass lost [t or m³]
        self.cumulative_withdrawals = 0.0  # Total withdrawn [t or m³]

        # Tracking
        self.storage_time = 0.0  # Time substrate has been stored [days]
        self.n_refills = 0

        # Initialize
        self.initialize()

    def initialize(self, initial_state: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize storage state.

        Args:
            initial_state: Optional initial state with keys:
                - 'current_level': Inventory level [t or m³]
                - 'quality_factor': Quality factor (0-1)
                - 'storage_time': Time stored [days]
                - 'cumulative_losses': Total losses [t or m³]
        """
        if initial_state:
            self.current_level = float(initial_state.get("current_level", self.current_level))
            # current_level cannot be larger than capacity
            self.current_level = min(self.current_level, self.capacity)
            self.quality_factor = float(initial_state.get("quality_factor", 1.0))
            self.storage_time = float(initial_state.get("storage_time", 0.0))
            self.cumulative_losses = float(initial_state.get("cumulative_losses", 0.0))

        self.state = {
            "current_level": self.current_level,
            "quality_factor": self.quality_factor,
            "storage_time": self.storage_time,
            "cumulative_losses": self.cumulative_losses,
            "cumulative_withdrawals": self.cumulative_withdrawals,
            "n_refills": self.n_refills,
        }

        self.outputs_data = {
            "current_level": self.current_level,
            "utilization": self.current_level / max(1e-6, self.capacity),
            "quality_factor": self.quality_factor,
            "available_mass": self.current_level * self.quality_factor,
            "degradation_rate": self.degradation_rate,
            "is_empty": self.current_level < 1e-3,
            "is_full": self.current_level > 0.95 * self.capacity,
        }

        self._initialized = True

    def step(self, t: float, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform one simulation time step.

        Args:
            t: Current time [days]
            dt: Time step [days]
            inputs: Input data with optional keys:
                - 'withdrawal_rate': Withdrawal rate [t/d or m³/d]
                - 'refill_amount': Amount to add [t or m³]
                - 'refill_quality': Quality of refill (0-1)
                - 'temperature': Ambient/storage temperature [K]

        Returns:
            Dict with keys:
                - 'current_level': Current inventory [t or m³]
                - 'utilization': Fill level (0-1)
                - 'quality_factor': Current quality (0-1)
                - 'available_mass': Usable inventory [t or m³]
                - 'degradation_rate': Current degradation rate [1/d]
                - 'losses_this_step': Mass lost this timestep [t or m³]
                - 'withdrawn_this_step': Mass withdrawn [t or m³]
                - 'is_empty': Storage empty flag
                - 'is_full': Storage full flag
        """
        # Update temperature if provided
        if "temperature" in inputs:
            self.temperature = float(inputs["temperature"])
            self.degradation_rate = self._estimate_degradation_rate()

        # Handle refilling
        refill_amount = float(inputs.get("refill_amount", 0.0))
        if refill_amount > 0:
            self._add_substrate(refill_amount, inputs.get("refill_quality", 1.0))

        # Calculate quality degradation
        losses_this_step = self._calculate_degradation_losses(dt)

        # Handle withdrawal
        withdrawal_rate = float(inputs.get("withdrawal_rate", 0.0))
        withdrawn_this_step = self._withdraw_substrate(withdrawal_rate, dt)

        # Update storage time
        self.storage_time += dt

        # Update state
        self.state.update(
            {
                "current_level": self.current_level,
                "quality_factor": self.quality_factor,
                "storage_time": self.storage_time,
                "cumulative_losses": self.cumulative_losses,
                "cumulative_withdrawals": self.cumulative_withdrawals,
                "n_refills": self.n_refills,
            }
        )

        # Prepare outputs
        self.outputs_data = {
            "current_level": float(self.current_level),
            "utilization": float(self.current_level / max(1e-6, self.capacity)),
            "quality_factor": float(self.quality_factor),
            "available_mass": float(self.current_level * self.quality_factor),
            "degradation_rate": float(self.degradation_rate),
            "losses_this_step": float(losses_this_step),
            "withdrawn_this_step": float(withdrawn_this_step),
            "is_empty": bool(self.current_level < 1e-3),
            "is_full": bool(self.current_level > 0.95 * self.capacity),
            "storage_time": float(self.storage_time),
            "dry_matter": float(self.dry_matter),
            "vs_content": float(self.vs_content),
        }

        return self.outputs_data

    def _add_substrate(self, amount: float, quality: float = 1.0) -> None:
        """
        Add substrate to storage.

        Args:
            amount: Amount to add [t or m³]
            quality: Quality of added substrate (0-1)
        """
        # Calculate new weighted quality
        total_mass = self.current_level + amount
        if total_mass > 0:
            new_quality = (self.current_level * self.quality_factor + amount * quality) / total_mass
            self.quality_factor = min(1.0, new_quality)

        # Add to inventory (respect capacity)
        available_space = self.capacity - self.current_level
        added = min(amount, available_space)
        self.current_level += added

        # Reset storage time on refill
        if added > 0:
            self.storage_time = 0.0
            self.n_refills += 1

    def _withdraw_substrate(self, rate: float, dt: float) -> float:
        """
        Withdraw substrate from storage.

        Args:
            rate: Withdrawal rate [t/d or m³/d]
            dt: Time step [days]

        Returns:
            Actual amount withdrawn [t or m³]
        """
        requested = rate * dt
        actual = min(requested, self.current_level)

        self.current_level -= actual
        self.cumulative_withdrawals += actual

        return actual

    def _calculate_degradation_losses(self, dt: float) -> float:
        """
        Calculate substrate degradation losses.

        Models aerobic degradation and dry matter losses during storage.

        Args:
            dt: Time step [days]

        Returns:
            Mass lost [t or m³]
        """
        if self.current_level < 1e-6:
            return 0.0

        # Exponential quality decay
        old_quality = self.quality_factor
        self.quality_factor *= np.exp(-self.degradation_rate * dt)
        self.quality_factor = max(0.0, self.quality_factor)

        # Calculate mass loss
        quality_loss = old_quality - self.quality_factor
        mass_loss = self.current_level * quality_loss

        # Remove lost mass from inventory
        self.current_level -= mass_loss
        self.cumulative_losses += mass_loss

        return mass_loss

    def _estimate_degradation_rate(self) -> float:
        """
        Estimate degradation rate based on storage type and conditions.

        Returns:
            Degradation rate [1/d]
        """
        # TODO: get a source confirming those values
        # Base rates for different storage types [1/d]
        # Reduced to more realistic values
        base_rates = {
            StorageType.VERTICAL_SILO: 0.0005,  # Reduced from 0.001 - Well sealed, very low losses
            StorageType.HORIZONTAL_SILO: 0.0008,  # Reduced from 0.002 - Good sealing
            StorageType.BUNKER_SILO: 0.001,  # Reduced from 0.003 - Moderate losses
            StorageType.CLAMP: 0.0025,  # Reduced from 0.005 - Higher losses
            StorageType.PILE: 0.004,  # Reduced from 0.008 - Highest losses
            StorageType.ABOVE_GROUND_TANK: 0.0002,  # Reduced from 0.0005 - Very low for liquids
            StorageType.BELOW_GROUND_TANK: 0.0001,  # Reduced from 0.0003 - Lowest losses
        }

        base_rate = base_rates.get(self.storage_type, 0.001)

        # Temperature correction (Q10 = 2)
        T_ref = 288.15  # 15°C reference
        T_factor = 2.0 ** ((self.temperature - T_ref) / 10.0)

        # Storage type modifier
        if self.storage_type in [StorageType.ABOVE_GROUND_TANK, StorageType.BELOW_GROUND_TANK]:
            # Liquid storage has minimal degradation if sealed
            # Further limit temperature effect for liquid storage
            T_factor = min(T_factor, 1.2)  # Changed from 1.5

        return base_rate * T_factor

    def _set_substrate_properties(self) -> None:
        """Set typical substrate properties based on substrate type."""
        # Default properties (density, DM, VS)
        properties = {
            SubstrateType.CORN_SILAGE: (650, 35, 95),
            SubstrateType.GRASS_SILAGE: (700, 30, 92),
            SubstrateType.WHOLE_CROP_SILAGE: (680, 32, 94),
            SubstrateType.MANURE_LIQUID: (1020, 8, 80),
            SubstrateType.MANURE_SOLID: (850, 25, 75),
            SubstrateType.BIOWASTE: (900, 35, 85),
            SubstrateType.FOOD_WASTE: (1000, 20, 90),
            SubstrateType.GENERIC_SOLID: (700, 30, 90),
            SubstrateType.GENERIC_LIQUID: (1000, 10, 80),
        }

        self.density, self.dry_matter, self.vs_content = properties.get(self.substrate_type, (800, 25, 85))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize storage to dictionary."""
        return {
            "component_id": self.component_id,
            "component_type": self.component_type.value,
            "name": self.name,
            "storage_type": self.storage_type.value,
            "substrate_type": self.substrate_type.value,
            "capacity": self.capacity,
            "temperature": self.temperature,
            "degradation_rate": self.degradation_rate,
            "density": self.density,
            "dry_matter": self.dry_matter,
            "vs_content": self.vs_content,
            "state": self.state,
            "inputs": self.inputs,
            "outputs": self.outputs,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "SubstrateStorage":
        """Create storage from dictionary."""
        storage = cls(
            component_id=config["component_id"],
            storage_type=config.get("storage_type", "vertical_silo"),
            substrate_type=config.get("substrate_type", "corn_silage"),
            capacity=config.get("capacity", 1000.0),
            initial_level=0.0,
            degradation_rate=config.get("degradation_rate"),
            temperature=config.get("temperature", 288.15),
            name=config.get("name"),
        )

        # Restore state if present
        if "state" in config:
            storage.initialize(config["state"])

        storage.inputs = config.get("inputs", [])
        storage.outputs = config.get("outputs", [])

        return storage
