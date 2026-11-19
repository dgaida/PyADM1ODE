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

from __future__ import annotations

from enum import Enum
from typing import Any

import numpy as np

from ..base import Component, ComponentType


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
        initial_level: float | None = None,
        degradation_rate: float | None = None,
        temperature: float = 288.15,  # 15°C
        name: str | None = None,
        density: float | None = None,
        dry_matter: float | None = None,
        vs_content: float | None = None,
    ):
        """
        Initialize substrate storage component.

        Args:
            component_id: Unique identifier
            storage_type: Type of storage ("vertical_silo", "tank", etc.)
            substrate_type: Substrate category ("corn_silage", "manure_liquid", etc.)
            capacity: Maximum capacity [t or m³]
            initial_level: Initial inventory [t or m³]. ``None`` (default)
                makes the stock unlimited: the storage always delivers what is
                asked, tracks no level, needs no refills and loses nothing to
                degradation. Use it for long runs where the inventory is not
                of interest. Pass a number (``0.0`` for an empty storage) to
                track the stock.
            degradation_rate: Quality degradation rate [1/d] (auto-calculated if None)
            temperature: Storage temperature [K]
            name: Human-readable name
        """
        super().__init__(component_id, ComponentType.SUBSTRATE_STORAGE, name)

        # Configuration
        self.storage_type = StorageType(storage_type.lower())
        self.substrate_type = SubstrateType(substrate_type.lower())
        self.capacity = float(capacity)
        self.temperature = float(temperature)

        # Inventory. An unlimited storage reports itself as full and is
        # never drawn down; only the cumulative withdrawals are counted.
        self.unlimited = initial_level is None
        self.current_level = float(capacity) if initial_level is None else float(min(initial_level, capacity))
        self.quality_factor = 1.0  # 1.0 = fresh, 0.0 = completely degraded

        # Substrate properties: typical values per substrate type, replaced
        # by the measured ones where given (e.g. from the Feedstock).
        self._set_substrate_properties()
        if density is not None:
            self.density = float(density)
        if dry_matter is not None:
            self.dry_matter = float(dry_matter)
        if vs_content is not None:
            self.vs_content = float(vs_content)

        # Degradation parameters. A rate given by the user is kept; only an
        # estimated one follows the storage temperature.
        self._degradation_rate_given = degradation_rate is not None
        # An explicit 0.0 means loss-free storage and must not fall back to the estimate.
        self.degradation_rate = self._estimate_degradation_rate() if degradation_rate is None else float(degradation_rate)

        # Storage losses
        self.cumulative_losses = 0.0  # Total mass lost [t or m³]
        self.cumulative_withdrawals = 0.0  # Total withdrawn [t or m³]

        # Tracking
        self.storage_time = 0.0  # Time substrate has been stored [days]
        self.n_refills = 0

        # Initialize
        self.initialize()

    def initialize(self, initial_state: dict[str, Any] | None = None) -> None:
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

    def step(self, t: float, dt: float, inputs: dict[str, Any]) -> dict[str, Any]:
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
            if not self._degradation_rate_given:
                self.degradation_rate = self._estimate_degradation_rate()

        # Handle refilling
        refill_amount = float(inputs.get("refill_amount", 0.0))
        if refill_amount > 0 and not self.unlimited:
            self._add_substrate(refill_amount, inputs.get("refill_quality", 1.0))

        # Calculate quality degradation (an unlimited stock is always fresh)
        losses_this_step = 0.0 if self.unlimited else self._calculate_degradation_losses(dt)

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
            "unlimited": bool(self.unlimited),
        }

        return self.outputs_data

    @property
    def available_stock(self) -> float:
        """Stock a feeder may draw from [t or m³]; infinite for an unlimited storage."""
        return float("inf") if self.unlimited else float(self.current_level)

    def withdraw(self, amount: float) -> float:
        """
        Take substrate out of the storage between steps.

        A feeder decides how much it draws only after it has seen the stock, so
        the plant steps the storage first (losses, refill) and books the
        feeder's actual withdrawal here, without running the step twice.

        Args:
            amount: Requested amount [same unit as ``capacity``, tonnes FM in a plant].

        Returns:
            Amount actually withdrawn, limited by the stock.
        """
        actual = self._withdraw_substrate(max(0.0, float(amount)), 1.0)

        self.state.update({"current_level": self.current_level, "cumulative_withdrawals": self.cumulative_withdrawals})
        self.outputs_data.update(
            {
                "current_level": float(self.current_level),
                "utilization": float(self.current_level / max(1e-6, self.capacity)),
                "available_mass": float(self.current_level * self.quality_factor),
                "withdrawn_this_step": float(self.outputs_data.get("withdrawn_this_step", 0.0) + actual),
                "is_empty": bool(self.current_level < 1e-3),
                "is_full": bool(self.current_level > 0.95 * self.capacity),
            }
        )
        return actual

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
        if self.unlimited:
            self.cumulative_withdrawals += requested
            return requested

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
        base_rates = {
            StorageType.VERTICAL_SILO: 0.0005,  # well sealed, very low losses
            StorageType.HORIZONTAL_SILO: 0.0008,  # good sealing
            StorageType.BUNKER_SILO: 0.001,  # moderate losses
            StorageType.CLAMP: 0.0025,  # higher losses
            StorageType.PILE: 0.004,  # highest losses
            StorageType.ABOVE_GROUND_TANK: 0.0002,  # very low for liquids
            StorageType.BELOW_GROUND_TANK: 0.0001,  # lowest losses
        }

        base_rate = base_rates.get(self.storage_type, 0.001)

        # Temperature correction (Q10 = 2)
        T_ref = 288.15  # 15°C reference
        T_factor = 2.0 ** ((self.temperature - T_ref) / 10.0)

        # Storage type modifier
        if self.storage_type in [
            StorageType.ABOVE_GROUND_TANK,
            StorageType.BELOW_GROUND_TANK,
        ]:
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

    def to_dict(self) -> dict[str, Any]:
        """Serialize storage to dictionary."""
        return {
            "component_id": self.component_id,
            "component_type": self.component_type.value,
            "name": self.name,
            "storage_type": self.storage_type.value,
            "substrate_type": self.substrate_type.value,
            "capacity": self.capacity,
            "unlimited": self.unlimited,
            "temperature": self.temperature,
            # An estimated rate is stored as None so it keeps following the
            # storage temperature after a reload.
            "degradation_rate": self.degradation_rate if self._degradation_rate_given else None,
            "density": self.density,
            "dry_matter": self.dry_matter,
            "vs_content": self.vs_content,
            "state": self.state,
            "inputs": self.inputs,
            "outputs": self.outputs,
        }

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> SubstrateStorage:
        """Create storage from dictionary."""
        storage = cls(
            component_id=config["component_id"],
            storage_type=config.get("storage_type", "vertical_silo"),
            substrate_type=config.get("substrate_type", "corn_silage"),
            capacity=config.get("capacity", 1000.0),
            # Files written before the flag existed always tracked their stock.
            initial_level=None if config.get("unlimited", False) else 0.0,
            degradation_rate=config.get("degradation_rate"),
            temperature=config.get("temperature", 288.15),
            name=config.get("name"),
            density=config.get("density"),
            dry_matter=config.get("dry_matter"),
            vs_content=config.get("vs_content"),
        )

        # Restore state if present
        if "state" in config:
            storage.initialize(config["state"])

        storage.inputs = config.get("inputs", [])
        storage.outputs = config.get("outputs", [])

        return storage
