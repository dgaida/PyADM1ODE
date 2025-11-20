# tests/unit/test_components/test_feeding/test_substrate_storage.py
# -*- coding: utf-8 -*-
"""
Unit tests for the SubstrateStorage component.

This module tests the SubstrateStorage class which models storage facilities
for different substrate types including silos and tanks.
"""

import numpy as np
from pyadm1.components.feeding.substrate_storage import (
    SubstrateStorage,
    StorageType,
    SubstrateType,
)


class TestSubstrateStorageInitialization:
    """Test suite for SubstrateStorage component initialization."""

    def test_storage_initialization_sets_component_id(self) -> None:
        """Test that storage initialization sets the component_id."""
        storage = SubstrateStorage("storage_1")

        assert storage.component_id == "storage_1", "Component ID should be set correctly"

    def test_storage_initialization_sets_capacity(self) -> None:
        """Test that storage initialization sets capacity."""
        capacity = 1500.0
        storage = SubstrateStorage("storage_1", capacity=capacity)

        assert storage.capacity == capacity, f"Capacity should be {capacity}"

    def test_storage_initialization_sets_storage_type(self) -> None:
        """Test that storage accepts different storage types."""
        storage = SubstrateStorage("storage_1", storage_type="vertical_silo")

        assert storage.storage_type == StorageType.VERTICAL_SILO, "Storage type should be VERTICAL_SILO"

    def test_storage_initialization_sets_substrate_type(self) -> None:
        """Test that storage accepts substrate type."""
        storage = SubstrateStorage("storage_1", substrate_type="corn_silage")

        assert storage.substrate_type == SubstrateType.CORN_SILAGE, "Substrate type should be CORN_SILAGE"

    def test_storage_initialization_default_values(self) -> None:
        """Test that storage has reasonable default values."""
        storage = SubstrateStorage("storage_1")

        assert storage.capacity == 1000.0, "Default capacity should be 1000.0 t"
        assert storage.storage_type == StorageType.VERTICAL_SILO, "Default type should be vertical_silo"
        assert storage.substrate_type == SubstrateType.CORN_SILAGE, "Default substrate should be corn_silage"

    def test_storage_initialization_with_custom_name(self) -> None:
        """Test that storage accepts custom name parameter."""
        custom_name = "Corn Silage Silo 1"
        storage = SubstrateStorage("storage_1", name=custom_name)

        assert storage.name == custom_name, f"Name should be '{custom_name}'"

    def test_storage_initialization_sets_initial_level(self) -> None:
        """Test that storage sets initial inventory level."""
        initial_level = 600.0
        storage = SubstrateStorage("storage_1", capacity=1000.0, initial_level=initial_level)

        assert storage.current_level == initial_level, f"Initial level should be {initial_level}"

    def test_storage_initialization_limits_initial_level_to_capacity(self) -> None:
        """Test that initial level cannot exceed capacity."""
        storage = SubstrateStorage("storage_1", capacity=1000.0, initial_level=1500.0)

        assert storage.current_level <= storage.capacity, "Initial level should be capped at capacity"

    def test_storage_initialization_estimates_degradation_rate(self) -> None:
        """Test that storage estimates degradation rate if not provided."""
        storage = SubstrateStorage("storage_1", storage_type="vertical_silo")

        assert storage.degradation_rate > 0, "Degradation rate should be positive"
        assert storage.degradation_rate < 0.05, "Degradation rate should be reasonable"

    def test_storage_initialization_accepts_custom_degradation_rate(self) -> None:
        """Test that storage accepts custom degradation rate."""
        custom_rate = 0.003
        storage = SubstrateStorage("storage_1", degradation_rate=custom_rate)

        assert storage.degradation_rate == custom_rate, "Custom degradation rate should be set"

    def test_storage_initialization_sets_substrate_properties(self) -> None:
        """Test that storage sets substrate properties."""
        storage = SubstrateStorage("storage_1", substrate_type="corn_silage")

        assert storage.density > 0, "Density should be set"
        assert storage.dry_matter > 0, "Dry matter should be set"
        assert storage.vs_content > 0, "VS content should be set"

    def test_storage_initialization_creates_state_dict(self) -> None:
        """Test that initialization creates state dictionary."""
        storage = SubstrateStorage("storage_1")

        assert hasattr(storage, "state"), "Storage should have state attribute"
        assert isinstance(storage.state, dict), "state should be a dictionary"


class TestSubstrateStorageInitialize:
    """Test suite for SubstrateStorage initialize method."""

    def test_initialize_sets_initial_level(self) -> None:
        """Test that initialize sets inventory level."""
        storage = SubstrateStorage("storage_1", capacity=1000.0, initial_level=600.0)
        storage.initialize()

        assert storage.current_level == 600.0, "Initial level should be set"

    def test_initialize_creates_state_dict(self) -> None:
        """Test that initialize creates proper state dictionary."""
        storage = SubstrateStorage("storage_1")
        storage.initialize()

        required_keys = [
            "current_level",
            "quality_factor",
            "storage_time",
            "cumulative_losses",
            "cumulative_withdrawals",
            "n_refills",
        ]
        for key in required_keys:
            assert key in storage.state, f"State should have '{key}' key"

    def test_initialize_with_custom_state(self) -> None:
        """Test initialize with custom initial state."""
        storage = SubstrateStorage("storage_1")
        storage.initialize(
            {
                "current_level": 750.0,
                "quality_factor": 0.95,
                "storage_time": 10.0,
                "cumulative_losses": 50.0,
            }
        )

        assert storage.current_level == 750.0, "Level should be set"
        assert storage.quality_factor == 0.95, "Quality should be set"
        assert storage.storage_time == 10.0, "Storage time should be set"

    def test_initialize_creates_outputs_data(self) -> None:
        """Test that initialize creates outputs_data dictionary."""
        storage = SubstrateStorage("storage_1")
        storage.initialize()

        assert hasattr(storage, "outputs_data"), "Storage should have outputs_data"
        assert isinstance(storage.outputs_data, dict), "outputs_data should be a dictionary"

    def test_initialize_clamps_level_to_capacity(self) -> None:
        """Test that initialize clamps level to capacity."""
        storage = SubstrateStorage("storage_1", capacity=1000.0)
        storage.initialize({"current_level": 1500.0})

        assert storage.current_level <= storage.capacity, "Level should be clamped"


class TestSubstrateStorageStep:
    """Test suite for SubstrateStorage step method (simulation)."""

    def test_step_returns_dict(self) -> None:
        """Test that step method returns a dictionary."""
        storage = SubstrateStorage("storage_1", capacity=1000.0, initial_level=500.0)
        storage.initialize()

        inputs = {"withdrawal_rate": 10.0}
        result = storage.step(t=0.0, dt=1.0, inputs=inputs)

        assert isinstance(result, dict), "step should return a dictionary"

    def test_step_output_contains_required_fields(self) -> None:
        """Test that step output contains required information."""
        storage = SubstrateStorage("storage_1", initial_level=500.0)
        storage.initialize()

        inputs = {"withdrawal_rate": 10.0}
        result = storage.step(t=0.0, dt=1.0, inputs=inputs)

        assert "current_level" in result, "Result should contain current_level"
        assert "utilization" in result, "Result should contain utilization"
        assert "quality_factor" in result, "Result should contain quality_factor"
        assert "available_mass" in result, "Result should contain available_mass"
        assert "degradation_rate" in result, "Result should contain degradation_rate"
        assert "losses_this_step" in result, "Result should contain losses_this_step"
        assert "withdrawn_this_step" in result, "Result should contain withdrawn_this_step"
        assert "is_empty" in result, "Result should contain is_empty"
        assert "is_full" in result, "Result should contain is_full"

    def test_step_withdraws_material(self) -> None:
        """Test that step withdraws material from storage."""
        storage = SubstrateStorage("storage_1", capacity=1000.0, initial_level=500.0)
        storage.initialize()

        initial_level = storage.current_level
        withdrawal_rate = 10.0  # t/d
        dt = 1.0  # 1 day

        inputs = {"withdrawal_rate": withdrawal_rate}
        result = storage.step(t=0.0, dt=dt, inputs=inputs)

        expected_level = initial_level - withdrawal_rate * dt
        assert abs(storage.current_level - expected_level) < 1.0, "Level should decrease by withdrawal"
        assert result["withdrawn_this_step"] > 0, "Should report withdrawn amount"

    def test_step_limits_withdrawal_to_available(self) -> None:
        """Test that withdrawal is limited to available inventory."""
        storage = SubstrateStorage("storage_1", capacity=1000.0, initial_level=5.0)
        storage.initialize()

        inputs = {"withdrawal_rate": 100.0}  # Request more than available
        result = storage.step(t=0.0, dt=1.0, inputs=inputs)

        assert result["withdrawn_this_step"] <= 5.0, "Cannot withdraw more than available"
        assert storage.current_level >= 0, "Level should not go negative"

    def test_step_adds_refill(self) -> None:
        """Test that step adds refill material."""
        storage = SubstrateStorage("storage_1", capacity=1000.0, initial_level=300.0)
        storage.initialize()

        initial_level = storage.current_level
        refill_amount = 200.0  # t

        inputs = {"refill_amount": refill_amount}
        storage.step(t=0.0, dt=1.0, inputs=inputs)

        expected_level = initial_level + refill_amount
        assert abs(storage.current_level - expected_level) < 1.0, "Level should increase by refill"

    def test_step_limits_refill_to_capacity(self) -> None:
        """Test that refill is limited to capacity."""
        storage = SubstrateStorage("storage_1", capacity=1000.0, initial_level=900.0)
        storage.initialize()

        inputs = {"refill_amount": 500.0}  # Would exceed capacity
        storage.step(t=0.0, dt=1.0, inputs=inputs)

        assert storage.current_level <= storage.capacity, "Level should not exceed capacity"

    def test_step_calculates_degradation(self) -> None:
        """Test that step calculates quality degradation."""
        storage = SubstrateStorage("storage_1", initial_level=500.0)
        storage.initialize()

        initial_quality = storage.quality_factor

        # Simulate several days
        for day in range(30):
            storage.step(t=day, dt=1.0, inputs={})

        # Quality should have degraded
        assert storage.quality_factor < initial_quality, "Quality should degrade over time"
        assert storage.quality_factor > 0, "Quality should remain positive"

    def test_step_tracks_cumulative_losses(self) -> None:
        """Test that step tracks cumulative losses."""
        storage = SubstrateStorage("storage_1", initial_level=500.0)
        storage.initialize()

        initial_losses = storage.cumulative_losses

        # Simulate time passage with degradation
        for day in range(10):
            storage.step(t=day, dt=1.0, inputs={})

        # Losses should accumulate
        assert storage.cumulative_losses > initial_losses, "Losses should accumulate"

    def test_step_tracks_cumulative_withdrawals(self) -> None:
        """Test that step tracks cumulative withdrawals."""
        storage = SubstrateStorage("storage_1", initial_level=500.0)
        storage.initialize()

        total_withdrawn = 0.0

        # Withdraw over multiple days
        for day in range(10):
            result = storage.step(t=day, dt=1.0, inputs={"withdrawal_rate": 10.0})
            total_withdrawn += result["withdrawn_this_step"]

        assert abs(storage.cumulative_withdrawals - total_withdrawn) < 1.0, "Cumulative withdrawals should match sum"

    def test_step_resets_storage_time_on_refill(self) -> None:
        """Test that storage time resets when refilled."""
        storage = SubstrateStorage("storage_1", initial_level=500.0)
        storage.initialize()

        # Age the storage
        for day in range(20):
            storage.step(t=day, dt=1.0, inputs={})

        assert storage.storage_time > 15, "Storage time should have increased"

        # Refill
        storage.step(t=20.0, dt=1.0, inputs={"refill_amount": 100.0})

        # set to 1.0, because we are doing a simulation over one day
        assert storage.storage_time == 1.0, "Storage time should reset after refill"

    def test_step_counts_refills(self) -> None:
        """Test that step counts number of refills."""
        storage = SubstrateStorage("storage_1", initial_level=500.0)
        storage.initialize()

        initial_refills = storage.n_refills

        # Multiple refills
        for i in range(5):
            storage.step(t=i, dt=1.0, inputs={"refill_amount": 50.0})

        assert storage.n_refills == initial_refills + 5, "Should count refills"

    def test_step_calculates_utilization(self) -> None:
        """Test that step calculates storage utilization."""
        storage = SubstrateStorage("storage_1", capacity=1000.0, initial_level=600.0)
        storage.initialize()

        result = storage.step(t=0.0, dt=1.0, inputs={})

        expected_utilization = 600.0 / 1000.0
        assert abs(result["utilization"] - expected_utilization) < 0.01, "Utilization should be correct"

    def test_step_reports_empty_status(self) -> None:
        """Test that step reports when storage is empty."""
        storage = SubstrateStorage("storage_1", capacity=1000.0, initial_level=0.5)
        storage.initialize()

        result = storage.step(t=0.0, dt=6.0, inputs={"withdrawal_rate": 100.0})

        assert result["is_empty"] is True, "Should report empty when level < 1"

    def test_step_reports_full_status(self) -> None:
        """Test that step reports when storage is full."""
        storage = SubstrateStorage("storage_1", capacity=1000.0, initial_level=970.0)
        storage.initialize()

        result = storage.step(t=0.0, dt=1.0, inputs={})

        assert result["is_full"] is True, "Should report full when level > 95% capacity"

    def test_step_updates_temperature(self) -> None:
        """Test that step accepts temperature updates."""
        storage = SubstrateStorage("storage_1", initial_level=500.0, temperature=288.15)
        storage.initialize()

        initial_rate = storage.degradation_rate

        # Update temperature
        inputs = {"temperature": 298.15}  # Higher temperature
        storage.step(t=0.0, dt=1.0, inputs=inputs)

        # Degradation rate should increase with temperature
        assert storage.degradation_rate > initial_rate, "Degradation should increase with temperature"


class TestStorageTypes:
    """Test suite for different storage types."""

    def test_vertical_silo_initialization(self) -> None:
        """Test vertical silo initialization."""
        storage = SubstrateStorage("storage_1", storage_type="vertical_silo")

        assert storage.storage_type == StorageType.VERTICAL_SILO
        # Vertical silos have low degradation rates
        assert storage.degradation_rate < 0.002, "Vertical silo should have low degradation"

    def test_horizontal_silo_initialization(self) -> None:
        """Test horizontal silo initialization."""
        storage = SubstrateStorage("storage_1", storage_type="horizontal_silo")

        assert storage.storage_type == StorageType.HORIZONTAL_SILO
        assert storage.degradation_rate > 0, "Should have degradation rate"

    def test_bunker_silo_initialization(self) -> None:
        """Test bunker silo initialization."""
        storage = SubstrateStorage("storage_1", storage_type="bunker_silo")

        assert storage.storage_type == StorageType.BUNKER_SILO
        # Bunker silos have moderate degradation
        assert 0.001 < storage.degradation_rate < 0.01

    def test_clamp_storage_initialization(self) -> None:
        """Test clamp storage initialization."""
        storage = SubstrateStorage("storage_1", storage_type="clamp")

        assert storage.storage_type == StorageType.CLAMP
        # Clamps have higher degradation - in class set to 0.0025
        assert storage.degradation_rate > 0.002

    def test_tank_storage_initialization(self) -> None:
        """Test tank storage initialization."""
        storage = SubstrateStorage("storage_1", storage_type="above_ground_tank")

        assert storage.storage_type == StorageType.ABOVE_GROUND_TANK
        # Tanks for liquids have very low degradation
        assert storage.degradation_rate < 0.001

    def test_different_storages_different_degradation(self) -> None:
        """Test that different storage types have appropriate degradation rates."""
        vertical = SubstrateStorage("s1", storage_type="vertical_silo")
        clamp = SubstrateStorage("s2", storage_type="clamp")
        tank = SubstrateStorage("s3", storage_type="above_ground_tank")

        # Tank < Vertical < Clamp
        assert tank.degradation_rate < vertical.degradation_rate
        assert vertical.degradation_rate < clamp.degradation_rate


class TestSubstrateTypes:
    """Test suite for different substrate types."""

    def test_corn_silage_properties(self) -> None:
        """Test corn silage substrate properties."""
        storage = SubstrateStorage("storage_1", substrate_type="corn_silage")

        assert storage.substrate_type == SubstrateType.CORN_SILAGE
        assert 600 <= storage.density <= 700, "Corn silage density should be typical"
        assert 30 <= storage.dry_matter <= 40, "Corn silage DM should be typical"
        assert storage.vs_content > 90, "Corn silage VS should be high"

    def test_grass_silage_properties(self) -> None:
        """Test grass silage substrate properties."""
        storage = SubstrateStorage("storage_1", substrate_type="grass_silage")

        assert storage.substrate_type == SubstrateType.GRASS_SILAGE
        assert storage.dry_matter < storage.density / 10.0, "Properties should be set"

    def test_manure_liquid_properties(self) -> None:
        """Test liquid manure substrate properties."""
        storage = SubstrateStorage("storage_1", substrate_type="manure_liquid")

        assert storage.substrate_type == SubstrateType.MANURE_LIQUID
        assert storage.density > 1000, "Liquid manure should have high density"
        assert storage.dry_matter < 15, "Liquid manure should have low DM"

    def test_manure_solid_properties(self) -> None:
        """Test solid manure substrate properties."""
        storage = SubstrateStorage("storage_1", substrate_type="manure_solid")

        assert storage.substrate_type == SubstrateType.MANURE_SOLID
        assert storage.dry_matter > storage.density / 100.0, "Solid manure should have properties"


class TestDegradation:
    """Test suite for quality degradation calculations."""

    def test_degradation_reduces_quality(self) -> None:
        """Test that degradation reduces quality factor over time."""
        storage = SubstrateStorage("storage_1", initial_level=500.0)
        storage.initialize()

        initial_quality = storage.quality_factor

        # Simulate 60 days
        for day in range(60):
            storage.step(t=day, dt=1.0, inputs={})

        assert storage.quality_factor < initial_quality, "Quality should degrade"

    def test_degradation_is_exponential(self) -> None:
        """Test that degradation follows exponential decay."""
        storage = SubstrateStorage("storage_1", initial_level=500.0, degradation_rate=0.01)
        storage.initialize()

        # Simulate and track quality
        qualities = [storage.quality_factor]
        for day in range(30):
            storage.step(t=day, dt=1.0, inputs={})
            qualities.append(storage.quality_factor)

        # Check exponential pattern (rate of decay proportional to current quality)
        # Q(t) = Q(0) * exp(-k*t)
        expected_final = 1.0 * np.exp(-0.01 * 30)
        assert abs(qualities[-1] - expected_final) < 0.1, "Should follow exponential decay"

    def test_degradation_causes_mass_loss(self) -> None:
        """Test that degradation causes mass loss."""
        storage = SubstrateStorage("storage_1", initial_level=500.0)
        storage.initialize()

        initial_level = storage.current_level

        # Simulate time with no withdrawal
        for day in range(30):
            storage.step(t=day, dt=1.0, inputs={})

        # Level should decrease due to degradation
        assert storage.current_level < initial_level, "Mass should be lost to degradation"
        assert storage.cumulative_losses > 0, "Losses should be tracked"

    def test_temperature_affects_degradation(self) -> None:
        """Test that temperature affects degradation rate."""
        storage1 = SubstrateStorage("s1", initial_level=500.0, temperature=278.15)  # 5°C
        storage2 = SubstrateStorage("s2", initial_level=500.0, temperature=298.15)  # 25°C

        storage1.initialize()
        storage2.initialize()

        # Simulate same duration
        for day in range(30):
            storage1.step(t=day, dt=1.0, inputs={})
            storage2.step(t=day, dt=1.0, inputs={})

        # Warmer storage should have more degradation
        assert storage2.cumulative_losses > storage1.cumulative_losses, "Warmer storage should have more degradation"

    def test_no_degradation_when_empty(self) -> None:
        """Test that empty storage has no degradation."""
        storage = SubstrateStorage("storage_1", capacity=1000.0, initial_level=0.0)
        storage.initialize()

        # Simulate time
        for day in range(10):
            result = storage.step(t=day, dt=1.0, inputs={})
            assert result["losses_this_step"] == 0.0, "No losses when empty"


class TestRefillQuality:
    """Test suite for refill quality mixing."""

    def test_refill_mixes_quality(self) -> None:
        """Test that refill mixes with existing material quality."""
        storage = SubstrateStorage("storage_1", capacity=1000.0, initial_level=400.0)
        storage.initialize()

        # Degrade existing material
        for day in range(30):
            storage.step(t=day, dt=1.0, inputs={})

        degraded_quality = storage.quality_factor
        existing_mass = storage.current_level

        # Add fresh material
        refill_amount = 300.0
        refill_quality = 1.0
        storage.step(t=30.0, dt=1.0, inputs={"refill_amount": refill_amount, "refill_quality": refill_quality})

        # Quality should be weighted average
        expected_quality = (existing_mass * degraded_quality + refill_amount * refill_quality) / (
            existing_mass + refill_amount
        )

        assert abs(storage.quality_factor - expected_quality) < 0.05, "Quality should be weighted average"

    def test_refill_with_degraded_material(self) -> None:
        """Test refill with partially degraded material."""
        storage = SubstrateStorage("storage_1", capacity=1000.0, initial_level=500.0)
        storage.initialize()

        # Add degraded material
        storage.step(t=0.0, dt=1.0, inputs={"refill_amount": 200.0, "refill_quality": 0.85})

        # Quality should decrease
        assert storage.quality_factor < 1.0, "Quality should reflect degraded refill"


class TestAvailableMass:
    """Test suite for available mass calculations."""

    def test_available_mass_considers_quality(self) -> None:
        """Test that available mass accounts for quality factor."""
        storage = SubstrateStorage("storage_1", initial_level=500.0)
        storage.initialize()

        # Degrade quality
        for day in range(30):
            storage.step(t=day, dt=1.0, inputs={})

        result = storage.step(t=30.0, dt=1.0, inputs={})

        expected_available = storage.current_level * storage.quality_factor
        assert abs(result["available_mass"] - expected_available) < 1.0, "Available mass should be level * quality"

    def test_fresh_material_fully_available(self) -> None:
        """Test that fresh material is fully available."""
        storage = SubstrateStorage("storage_1", initial_level=500.0)
        storage.initialize()

        result = storage.step(t=0.0, dt=1.0, inputs={})

        assert abs(result["available_mass"] - 500.0) < 1.0, "Fresh material should be fully available"


class TestSubstrateStorageSerialization:
    """Test suite for SubstrateStorage serialization methods."""

    def test_to_dict_returns_dict(self) -> None:
        """Test that to_dict method returns a dictionary."""
        storage = SubstrateStorage(
            "storage_1",
            storage_type="vertical_silo",
            substrate_type="corn_silage",
            capacity=1500.0,
        )
        storage.initialize()

        config = storage.to_dict()

        assert isinstance(config, dict), "to_dict should return a dictionary"

    def test_to_dict_contains_required_fields(self) -> None:
        """Test that to_dict includes all required fields."""
        storage = SubstrateStorage("storage_1")
        storage.initialize()

        config = storage.to_dict()

        required_fields = [
            "component_id",
            "component_type",
            "storage_type",
            "substrate_type",
            "capacity",
            "temperature",
            "degradation_rate",
            "density",
            "dry_matter",
            "vs_content",
        ]
        for field in required_fields:
            assert field in config, f"to_dict should include '{field}'"

    def test_from_dict_recreates_storage(self) -> None:
        """Test that from_dict can recreate storage from configuration."""
        original = SubstrateStorage(
            "storage_1",
            storage_type="bunker_silo",
            substrate_type="grass_silage",
            capacity=2000.0,
            degradation_rate=0.004,
            temperature=285.15,
            name="Main Silo",
        )
        original.initialize()

        config = original.to_dict()
        recreated = SubstrateStorage.from_dict(config)

        assert recreated.component_id == original.component_id
        assert recreated.storage_type == original.storage_type
        assert recreated.substrate_type == original.substrate_type
        assert recreated.capacity == original.capacity
        assert recreated.name == original.name

    def test_roundtrip_preserves_configuration(self) -> None:
        """Test that serialization roundtrip preserves configuration."""
        original = SubstrateStorage(
            "storage_1",
            storage_type="horizontal_silo",
            capacity=1800.0,
            initial_level=1200.0,
        )
        original.initialize()

        config = original.to_dict()
        recreated = SubstrateStorage.from_dict(config)

        # Restore state
        if "state" in config:
            recreated.initialize(config["state"])

        assert recreated.capacity == original.capacity
        assert recreated.storage_type == original.storage_type


class TestSubstrateStorageConnections:
    """Test suite for SubstrateStorage component connections."""

    def test_add_input_connection(self) -> None:
        """Test adding input connections to storage."""
        storage = SubstrateStorage("storage_1")

        storage.add_input("delivery_truck")

        assert "delivery_truck" in storage.inputs, "Input should be added"

    def test_add_output_connection(self) -> None:
        """Test adding output connections from storage."""
        storage = SubstrateStorage("storage_1")

        storage.add_output("feeder_1")

        assert "feeder_1" in storage.outputs, "Output should be added"

    def test_multiple_connections(self) -> None:
        """Test adding multiple connections."""
        storage = SubstrateStorage("storage_1")

        storage.add_input("truck")
        storage.add_output("feeder_1")
        storage.add_output("feeder_2")

        assert len(storage.inputs) == 1
        assert len(storage.outputs) == 2


class TestSubstrateStorageProperties:
    """Test suite for SubstrateStorage properties and attributes."""

    def test_current_level_non_negative(self) -> None:
        """Test that current level is always non-negative."""
        storage = SubstrateStorage("storage_1", initial_level=10.0)
        storage.initialize()

        # Withdraw more than available
        inputs = {"withdrawal_rate": 50.0}
        storage.step(t=0.0, dt=1.0, inputs=inputs)

        assert storage.current_level >= 0, "Level should not go negative"

    def test_quality_factor_bounds(self) -> None:
        """Test that quality factor stays within valid bounds."""
        storage = SubstrateStorage("storage_1", initial_level=500.0)
        storage.initialize()

        # Simulate long time
        for day in range(365):
            storage.step(t=day, dt=1.0, inputs={})

        assert 0.0 <= storage.quality_factor <= 1.0, "Quality should be between 0 and 1"

    def test_utilization_bounds(self) -> None:
        """Test that utilization stays within valid bounds."""
        storage = SubstrateStorage("storage_1", capacity=1000.0, initial_level=600.0)
        storage.initialize()

        result = storage.step(t=0.0, dt=1.0, inputs={})

        assert 0.0 <= result["utilization"] <= 1.0, "Utilization should be between 0 and 1"

    def test_storage_time_increments(self) -> None:
        """Test that storage time increments with each step."""
        storage = SubstrateStorage("storage_1", initial_level=500.0)
        storage.initialize()

        initial_time = storage.storage_time

        storage.step(t=0.0, dt=5.0, inputs={})

        assert storage.storage_time == initial_time + 5.0, "Storage time should increment"

    def test_get_state_returns_dict(self) -> None:
        """Test that get_state returns the state dictionary."""
        storage = SubstrateStorage("storage_1")
        storage.initialize()

        state = storage.get_state()

        assert isinstance(state, dict), "get_state should return a dictionary"
        assert "current_level" in state
        assert "quality_factor" in state

    def test_set_state_updates_properties(self) -> None:
        """Test that set_state updates component state."""
        storage = SubstrateStorage("storage_1")
        storage.initialize()

        new_state = {
            "current_level": 750.0,
            "quality_factor": 0.90,
            "storage_time": 15.0,
            "cumulative_losses": 25.0,
        }
        storage.set_state(new_state)

        assert storage.state.get("current_level") == 750.0
        assert storage.state.get("quality_factor") == 0.90


class TestSubstrateStorageEdgeCases:
    """Test suite for storage edge cases and error handling."""

    def test_storage_with_zero_capacity(self) -> None:
        """Test storage behavior with very small capacity."""
        storage = SubstrateStorage("storage_1", capacity=1.0, initial_level=0.5)
        storage.initialize()

        result = storage.step(t=0.0, dt=1.0, inputs={"withdrawal_rate": 0.2})

        assert result["current_level"] >= 0, "Should handle small capacities"

    def test_storage_with_very_large_capacity(self) -> None:
        """Test storage with very large capacity."""
        storage = SubstrateStorage("storage_1", capacity=50000.0, initial_level=25000.0)
        storage.initialize()

        result = storage.step(t=0.0, dt=1.0, inputs={"withdrawal_rate": 100.0})

        assert result["current_level"] > 0, "Should handle large capacities"

    def test_storage_empty_then_refill(self) -> None:
        """Test emptying storage then refilling."""
        storage = SubstrateStorage("storage_1", capacity=1000.0, initial_level=50.0)
        storage.initialize()

        # Empty the storage
        storage.step(t=0.0, dt=1.0, inputs={"withdrawal_rate": 100.0})
        assert storage.current_level < 1.0, "Should be empty"

        # Refill
        result = storage.step(t=1.0, dt=1.0, inputs={"refill_amount": 500.0})
        assert result["current_level"] > 400.0, "Should be refilled"

    def test_storage_multiple_consecutive_steps(self) -> None:
        """Test storage over multiple consecutive time steps."""
        storage = SubstrateStorage("storage_1", initial_level=500.0)
        storage.initialize()

        # Simulate 30 days with regular withdrawal
        for day in range(30):
            result = storage.step(t=day, dt=1.0, inputs={"withdrawal_rate": 10.0})
            assert result["current_level"] >= 0.0

        # Should have withdrawn and degraded
        assert storage.current_level < 500.0

    def test_storage_alternating_refill_withdrawal(self) -> None:
        """Test alternating refill and withdrawal cycles."""
        storage = SubstrateStorage("storage_1", capacity=1000.0, initial_level=500.0)
        storage.initialize()

        for cycle in range(10):
            # Withdraw
            storage.step(t=cycle * 2, dt=1.0, inputs={"withdrawal_rate": 50.0})
            # Refill
            storage.step(t=cycle * 2 + 1, dt=1.0, inputs={"refill_amount": 40.0})

        # Storage should still be operating
        assert storage.current_level > 0
        assert storage.n_refills == 10


class TestSubstrateStorageIntegration:
    """Test suite for storage integration scenarios."""

    def test_storage_monthly_operation_cycle(self) -> None:
        """Test storage over a complete monthly cycle."""
        storage = SubstrateStorage("storage_1", capacity=1000.0, initial_level=800.0)
        storage.initialize()

        total_withdrawn = 0.0

        # Simulate 30 days with daily withdrawal
        for day in range(30):
            result = storage.step(t=day, dt=1.0, inputs={"withdrawal_rate": 15.0})
            total_withdrawn += result["withdrawn_this_step"]

        # Check mass balance
        expected_final = 800.0 - total_withdrawn - storage.cumulative_losses
        assert abs(storage.current_level - expected_final) < 5.0, "Mass balance should be correct"

    def test_storage_with_varying_withdrawal_profile(self) -> None:
        """Test storage with time-varying withdrawal."""
        storage = SubstrateStorage("storage_1", capacity=1000.0, initial_level=700.0)
        storage.initialize()

        # Varying withdrawal rates
        rates = [10.0, 15.0, 20.0, 15.0, 12.0, 8.0]

        for i, rate in enumerate(rates):
            storage.step(t=i, dt=1.0, inputs={"withdrawal_rate": rate})

        assert storage.cumulative_withdrawals > 0

    def test_storage_seasonal_refill_pattern(self) -> None:
        """Test storage with seasonal refill pattern."""
        storage = SubstrateStorage("storage_1", capacity=2000.0, initial_level=500.0)
        storage.initialize()

        # Simulate harvest season (large refills)
        for day in range(10):
            if day % 3 == 0:  # Harvest days
                storage.step(t=day, dt=1.0, inputs={"refill_amount": 200.0, "withdrawal_rate": 20.0})
            else:
                storage.step(t=day, dt=1.0, inputs={"withdrawal_rate": 20.0})

        assert storage.n_refills > 0
        assert storage.current_level > 500.0, "Should have net increase during harvest"

    def test_storage_long_term_operation(self) -> None:
        """Test storage over long-term operation (1 year)."""
        storage = SubstrateStorage("storage_1", capacity=2000.0, initial_level=1500.0)
        storage.initialize()

        # Simulate 365 days
        for day in range(365):
            # Weekly refills
            if day % 7 == 0:
                storage.step(t=day, dt=1.0, inputs={"refill_amount": 150.0, "withdrawal_rate": 15.0})
            else:
                storage.step(t=day, dt=1.0, inputs={"withdrawal_rate": 15.0})

        # Should still be operational
        assert storage.current_level > 0
        assert storage.quality_factor > 0.5, "Quality should still be reasonable with regular refills"


class TestSubstrateStoragePhysics:
    """Test suite for storage physics and calculations."""

    def test_degradation_temperature_dependency(self) -> None:
        """Test Q10 temperature dependency of degradation."""
        storage = SubstrateStorage("storage_1", initial_level=500.0, temperature=288.15)
        storage.initialize()

        rate_15C = storage.degradation_rate

        # Increase temperature by 10K
        storage.temperature = 298.15
        storage.degradation_rate = storage._estimate_degradation_rate()
        rate_25C = storage.degradation_rate

        # Q10 = 2, so rate should approximately double
        ratio = rate_25C / rate_15C
        assert 1.5 <= ratio <= 2.5, "Q10 relationship should hold approximately"

    def test_storage_type_affects_degradation(self) -> None:
        """Test that storage type significantly affects degradation."""
        storage1 = SubstrateStorage("s1", storage_type="vertical_silo", initial_level=500.0)
        storage2 = SubstrateStorage("s2", storage_type="clamp", initial_level=500.0)

        storage1.initialize()
        storage2.initialize()

        # Same conditions, different storage types
        for day in range(30):
            storage1.step(t=day, dt=1.0, inputs={})
            storage2.step(t=day, dt=1.0, inputs={})

        # Clamp should have more losses
        assert (
            storage2.cumulative_losses > storage1.cumulative_losses * 2
        ), "Open storage should have significantly more losses"

    def test_liquid_storage_low_degradation(self) -> None:
        """Test that liquid storage has minimal degradation."""
        storage = SubstrateStorage(
            "storage_1",
            storage_type="above_ground_tank",
            substrate_type="manure_liquid",
            initial_level=500.0,
        )
        storage.initialize()

        # Simulate 60 days
        for day in range(60):
            storage.step(t=day, dt=1.0, inputs={})

        # Liquid storage should have very low losses
        assert storage.cumulative_losses < 10.0, "Liquid storage should have minimal losses"
        assert storage.quality_factor > 0.95, "Quality should be well preserved"


class TestSubstrateStorageValidation:
    """Test suite for storage validation and constraints."""

    def test_storage_enforces_capacity_constraint(self) -> None:
        """Test that storage never exceeds capacity."""
        storage = SubstrateStorage("storage_1", capacity=1000.0, initial_level=950.0)
        storage.initialize()

        # Try to overfill multiple times
        for i in range(10):
            storage.step(t=i, dt=1.0, inputs={"refill_amount": 200.0})
            assert (
                storage.current_level <= storage.capacity
            ), f"Level {storage.current_level} should not exceed capacity {storage.capacity}"

    def test_storage_enforces_non_negative_level(self) -> None:
        """Test that storage level never goes negative."""
        storage = SubstrateStorage("storage_1", capacity=1000.0, initial_level=10.0)
        storage.initialize()

        # Try to over-withdraw multiple times
        for i in range(10):
            storage.step(t=i, dt=1.0, inputs={"withdrawal_rate": 50.0})
            assert storage.current_level >= 0, "Level should never be negative"

    def test_quality_factor_never_exceeds_one(self) -> None:
        """Test that quality factor never exceeds 1.0."""
        storage = SubstrateStorage("storage_1", capacity=1000.0, initial_level=500.0)
        storage.initialize()

        # Try adding fresh material multiple times
        for i in range(20):
            storage.step(t=i, dt=1.0, inputs={"refill_amount": 50.0, "refill_quality": 1.0})
            assert storage.quality_factor <= 1.0, "Quality should not exceed 1.0"

    def test_utilization_never_exceeds_one(self) -> None:
        """Test that utilization never exceeds 1.0."""
        storage = SubstrateStorage("storage_1", capacity=1000.0, initial_level=500.0)
        storage.initialize()

        # Fill to capacity
        for i in range(10):
            result = storage.step(t=i, dt=1.0, inputs={"refill_amount": 100.0})
            assert result["utilization"] <= 1.0, "Utilization should not exceed 1.0"


class TestSubstrateStorageDocumentation:
    """Test suite to verify storage documentation and examples work."""

    def test_basic_example_from_docstring(self) -> None:
        """Test basic example from component docstring."""
        storage = SubstrateStorage(
            "silo1", storage_type="vertical_silo", substrate_type="corn_silage", capacity=1000, initial_level=600
        )
        storage.initialize()
        result = storage.step(t=0, dt=1, inputs={"withdrawal_rate": 15})

        assert "current_level" in result
        assert result["current_level"] < 600

    def test_module_example_from_docstring(self) -> None:
        """Test example from module-level docstring."""
        storage = SubstrateStorage(
            "silo1", storage_type="vertical_silo", substrate_type="corn_silage", capacity=1000, initial_level=800
        )
        storage.initialize()
        outputs = storage.step(0, 1, {"withdrawal_rate": 15})

        assert isinstance(outputs, dict)
        assert "current_level" in outputs


class TestSubstrateStorageRealism:
    """Test suite for realistic storage behavior."""

    def test_storage_dry_matter_losses_realistic(self) -> None:
        """Test that DM losses are in realistic range."""
        storage = SubstrateStorage(
            "storage_1",
            storage_type="bunker_silo",
            substrate_type="corn_silage",
            initial_level=1000.0,
        )
        storage.initialize()

        # Simulate 6 months
        for day in range(180):
            storage.step(t=day, dt=1.0, inputs={})

        # Typical DM losses in bunker silos: 5-15% over 6 months
        loss_percentage = (storage.cumulative_losses / 1000.0) * 100
        assert 2.0 <= loss_percentage <= 20.0, f"DM losses {loss_percentage:.1f}% should be in realistic range"

    def test_storage_quality_degradation_realistic(self) -> None:
        """Test that quality degradation is realistic."""
        storage = SubstrateStorage(
            "storage_1",
            storage_type="vertical_silo",
            substrate_type="corn_silage",
            initial_level=1000.0,
        )
        storage.initialize()

        # Simulate 90 days
        for day in range(90):
            storage.step(t=day, dt=1.0, inputs={})

        # Well-managed silage should maintain >90% quality after 3 months
        assert storage.quality_factor > 0.85, "Quality should be well preserved in vertical silo"

    def test_storage_temperature_effect_realistic(self) -> None:
        """Test that temperature effect is realistic."""
        # Cold storage
        storage_cold = SubstrateStorage(
            "s1",
            storage_type="vertical_silo",
            initial_level=500.0,
            temperature=278.15,  # 5°C
        )

        # Warm storage
        storage_warm = SubstrateStorage(
            "s2",
            storage_type="vertical_silo",
            initial_level=500.0,
            temperature=303.15,  # 30°C
        )

        storage_cold.initialize()
        storage_warm.initialize()

        # Simulate 30 days
        for day in range(30):
            storage_cold.step(t=day, dt=1.0, inputs={})
            storage_warm.step(t=day, dt=1.0, inputs={})

        # Warm storage should have significantly more degradation
        assert (
            storage_warm.cumulative_losses > storage_cold.cumulative_losses * 2
        ), "Warm storage should have much higher losses"


class TestSubstrateStoragePerformance:
    """Test suite for storage performance metrics."""

    def test_storage_calculates_all_metrics(self) -> None:
        """Test that storage calculates all expected metrics."""
        storage = SubstrateStorage("storage_1", initial_level=500.0)
        storage.initialize()

        result = storage.step(t=0.0, dt=1.0, inputs={"withdrawal_rate": 10.0})

        # All key metrics should be present
        assert "current_level" in result
        assert "utilization" in result
        assert "quality_factor" in result
        assert "available_mass" in result
        assert "degradation_rate" in result
        assert "losses_this_step" in result
        assert "withdrawn_this_step" in result
        assert "is_empty" in result
        assert "is_full" in result
        assert "storage_time" in result
        assert "dry_matter" in result
        assert "vs_content" in result

    def test_storage_mass_balance(self) -> None:
        """Test that mass balance is maintained."""
        storage = SubstrateStorage("storage_1", capacity=1000.0, initial_level=600.0)
        storage.initialize()

        initial_level = storage.current_level

        # Track all changes
        total_refills = 0.0
        total_withdrawals = 0.0
        total_losses = 0.0

        for day in range(30):
            if day % 5 == 0:
                refill = 50.0
                total_refills += refill
                result = storage.step(t=day, dt=1.0, inputs={"refill_amount": refill, "withdrawal_rate": 15.0})
            else:
                result = storage.step(t=day, dt=1.0, inputs={"withdrawal_rate": 15.0})

            total_withdrawals += result["withdrawn_this_step"]
            total_losses += result["losses_this_step"]

        # Mass balance
        expected_final = initial_level + total_refills - total_withdrawals - total_losses
        actual_final = storage.current_level

        assert (
            abs(actual_final - expected_final) < 1.0
        ), f"Mass balance error: expected {expected_final:.1f}, got {actual_final:.1f}"
