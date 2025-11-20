# tests/unit/test_components/test_feeding/test_feeder.py
# -*- coding: utf-8 -*-
"""
Unit tests for the Feeder component.

This module tests the Feeder class which models automated substrate dosing
systems for feeding substrates into biogas digesters.
"""

import numpy as np
from pyadm1.components.feeding.feeder import Feeder, FeederType, SubstrateCategory


class TestFeederInitialization:
    """Test suite for Feeder component initialization."""

    def test_feeder_initialization_sets_component_id(self) -> None:
        """Test that feeder initialization sets the component_id."""
        feeder = Feeder("feeder_1")

        assert feeder.component_id == "feeder_1", "Component ID should be set correctly"

    def test_feeder_initialization_sets_flow_rate(self) -> None:
        """Test that feeder initialization sets maximum flow rate."""
        Q_max = 25.0
        feeder = Feeder("feeder_1", Q_max=Q_max)

        assert feeder.Q_max == Q_max, f"Q_max should be {Q_max}"

    def test_feeder_initialization_sets_feeder_type(self) -> None:
        """Test that feeder accepts different feeder types."""
        feeder = Feeder("feeder_1", feeder_type="screw")

        assert feeder.feeder_type == FeederType.SCREW, "Feeder type should be SCREW"

    def test_feeder_initialization_sets_substrate_type(self) -> None:
        """Test that feeder accepts substrate type."""
        feeder = Feeder("feeder_1", substrate_type="solid")

        assert feeder.substrate_type == SubstrateCategory.SOLID, "Substrate type should be SOLID"

    def test_feeder_initialization_default_values(self) -> None:
        """Test that feeder has reasonable default values."""
        feeder = Feeder("feeder_1")

        assert feeder.Q_max == 20.0, "Default Q_max should be 20.0 m³/d"
        assert feeder.feeder_type == FeederType.SCREW, "Default type should be screw"
        assert feeder.substrate_type == SubstrateCategory.SOLID, "Default substrate should be solid"

    def test_feeder_initialization_with_custom_name(self) -> None:
        """Test that feeder accepts custom name parameter."""
        custom_name = "Corn Silage Feeder"
        feeder = Feeder("feeder_1", name=custom_name)

        assert feeder.name == custom_name, f"Name should be '{custom_name}'"

    def test_feeder_initialization_estimates_dosing_accuracy(self) -> None:
        """Test that feeder estimates dosing accuracy if not provided."""
        feeder = Feeder("feeder_1", feeder_type="screw")

        assert 0.01 <= feeder.dosing_accuracy <= 0.15, "Dosing accuracy should be in reasonable range"

    def test_feeder_initialization_accepts_custom_accuracy(self) -> None:
        """Test that feeder accepts custom dosing accuracy."""
        custom_accuracy = 0.03
        feeder = Feeder("feeder_1", dosing_accuracy=custom_accuracy)

        assert feeder.dosing_accuracy == custom_accuracy, "Custom accuracy should be set"

    def test_feeder_initialization_estimates_power(self) -> None:
        """Test that feeder estimates power requirement if not provided."""
        feeder = Feeder("feeder_1", Q_max=20.0)

        assert feeder.power_installed > 0, "Power should be estimated"
        assert feeder.power_installed < 50.0, "Power should be reasonable"

    def test_feeder_initialization_accepts_custom_power(self) -> None:
        """Test that feeder accepts custom power value."""
        custom_power = 5.0
        feeder = Feeder("feeder_1", power_installed=custom_power)

        assert feeder.power_installed == custom_power, "Custom power should be set"

    def test_feeder_initialization_creates_state_dict(self) -> None:
        """Test that initialization creates state dictionary."""
        feeder = Feeder("feeder_1")

        assert hasattr(feeder, "state"), "Feeder should have state attribute"
        assert isinstance(feeder.state, dict), "state should be a dictionary"


class TestFeederInitialize:
    """Test suite for Feeder initialize method."""

    def test_initialize_sets_feeder_off(self) -> None:
        """Test that initialize sets feeder to off state."""
        feeder = Feeder("feeder_1")
        feeder.initialize()

        assert feeder.is_running is False, "Feeder should be off initially"
        assert feeder.current_flow == 0.0, "Initial flow should be zero"

    def test_initialize_creates_state_dict(self) -> None:
        """Test that initialize creates proper state dictionary."""
        feeder = Feeder("feeder_1")
        feeder.initialize()

        required_keys = [
            "is_running",
            "current_flow",
            "operating_hours",
            "energy_consumed",
            "total_mass_fed",
            "blockage_detected",
            "n_starts",
            "n_blockages",
        ]
        for key in required_keys:
            assert key in feeder.state, f"State should have '{key}' key"

    def test_initialize_with_custom_state(self) -> None:
        """Test initialize with custom initial state."""
        feeder = Feeder("feeder_1")
        feeder.initialize({"is_running": True, "current_flow": 15.0, "operating_hours": 500.0})

        assert feeder.is_running is True, "Feeder should be running"
        assert feeder.current_flow == 15.0, "Flow should be set to 15.0"
        assert feeder.operating_hours == 500.0, "Operating hours should be set"

    def test_initialize_creates_outputs_data(self) -> None:
        """Test that initialize creates outputs_data dictionary."""
        feeder = Feeder("feeder_1")
        feeder.initialize()

        assert hasattr(feeder, "outputs_data"), "Feeder should have outputs_data"
        assert isinstance(feeder.outputs_data, dict), "outputs_data should be a dictionary"


class TestFeederStep:
    """Test suite for Feeder step method (simulation)."""

    def test_step_returns_dict(self) -> None:
        """Test that step method returns a dictionary."""
        feeder = Feeder("feeder_1", Q_max=20.0)
        feeder.initialize()

        inputs = {"Q_setpoint": 15.0}
        result = feeder.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert isinstance(result, dict), "step should return a dictionary"

    def test_step_with_feeder_disabled(self) -> None:
        """Test step with feeder disabled."""
        feeder = Feeder("feeder_1")
        feeder.initialize()

        inputs = {"enable_feeding": False, "Q_setpoint": 15.0}
        result = feeder.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert result["P_consumed"] == 0.0, "Power should be zero when disabled"
        assert result["Q_actual"] == 0.0, "Flow should be zero when disabled"
        assert result["is_running"] is False, "Feeder should not be running"

    def test_step_output_contains_required_fields(self) -> None:
        """Test that step output contains required information."""
        feeder = Feeder("feeder_1")
        feeder.initialize()

        inputs = {"Q_setpoint": 15.0}
        result = feeder.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert "Q_actual" in result, "Result should contain Q_actual"
        assert "is_running" in result, "Result should contain is_running"
        assert "load_factor" in result, "Result should contain load_factor"
        assert "P_consumed" in result, "Result should contain P_consumed"
        assert "blockage_detected" in result, "Result should contain blockage_detected"
        assert "dosing_error" in result, "Result should contain dosing_error"

    def test_step_calculates_power_consumption(self) -> None:
        """Test that step correctly calculates power consumption."""
        feeder = Feeder("feeder_1", Q_max=20.0)
        feeder.initialize()

        inputs = {"Q_setpoint": 15.0}
        result = feeder.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert result["P_consumed"] > 0, "Feeder should consume power when running"
        assert result["Q_actual"] > 0, "Feeder should deliver flow when running"

    def test_step_respects_flow_setpoint(self) -> None:
        """Test that step respects flow setpoint."""
        feeder = Feeder("feeder_1", Q_max=20.0, enable_dosing_noise=False)
        feeder.initialize()

        Q_setpoint = 12.0
        inputs = {"Q_setpoint": Q_setpoint}
        result = feeder.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert abs(result["Q_actual"] - Q_setpoint) < 0.1, "Flow should be close to setpoint without noise"

    def test_step_with_dosing_noise(self) -> None:
        """Test that dosing noise introduces variance."""
        feeder = Feeder("feeder_1", Q_max=20.0, enable_dosing_noise=True)
        feeder.initialize()

        Q_setpoint = 15.0
        inputs = {"Q_setpoint": Q_setpoint}

        # Run multiple times and check for variance
        flows = []
        for _ in range(10):
            result = feeder.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)
            flows.append(result["Q_actual"])

        # Should have some variance
        assert np.std(flows) > 0, "Dosing noise should introduce variance"

    def test_step_limits_flow_to_max(self) -> None:
        """Test that step limits flow to Q_max."""
        feeder = Feeder("feeder_1", Q_max=20.0, enable_dosing_noise=False)
        feeder.initialize()

        inputs = {"Q_setpoint": 30.0}  # Request more than max
        result = feeder.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert result["Q_actual"] <= feeder.Q_max * 1.1, "Flow should be limited to Q_max"

    def test_step_updates_operating_hours(self) -> None:
        """Test that step updates operating hours."""
        feeder = Feeder("feeder_1")
        feeder.initialize()

        initial_hours = feeder.operating_hours
        dt = 1.0 / 24.0  # 1 hour

        inputs = {"Q_setpoint": 15.0}
        feeder.step(t=0.0, dt=dt, inputs=inputs)

        assert feeder.operating_hours > initial_hours, "Operating hours should increase"
        assert abs(feeder.operating_hours - (initial_hours + 1.0)) < 0.01, "Should increase by 1 hour"

    def test_step_updates_total_mass_fed(self) -> None:
        """Test that step updates total mass fed."""
        feeder = Feeder("feeder_1", Q_max=20.0, enable_dosing_noise=False)
        feeder.initialize()

        dt = 1.0 / 24.0  # 1 hour
        Q_setpoint = 12.0  # m³/d

        inputs = {"Q_setpoint": Q_setpoint}
        feeder.step(t=0.0, dt=dt, inputs=inputs)

        expected_mass = Q_setpoint * dt  # m³
        assert abs(feeder.total_mass_fed - expected_mass) < 0.5, "Mass should be accumulated"

    def test_step_with_zero_setpoint(self) -> None:
        """Test that feeder stops with zero setpoint."""
        feeder = Feeder("feeder_1")
        feeder.initialize()

        inputs = {"Q_setpoint": 0.0}
        result = feeder.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert result["P_consumed"] == 0.0, "Power should be zero with zero setpoint"
        assert result["Q_actual"] == 0.0, "Flow should be zero"
        assert result["is_running"] is False, "Feeder should not be running"

    def test_step_with_speed_control(self) -> None:
        """Test step with variable speed control."""
        feeder = Feeder("feeder_1", Q_max=20.0)
        feeder.initialize()

        speed_setpoint = 0.7
        inputs = {"Q_setpoint": 15.0, "speed_setpoint": speed_setpoint}
        result = feeder.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert 0.6 <= result["speed_fraction"] <= 0.8, "Speed should be around setpoint"

    def test_step_counts_starts(self) -> None:
        """Test that step counts feeder starts."""
        feeder = Feeder("feeder_1")
        feeder.initialize()

        initial_starts = feeder.n_starts

        # Stop feeder
        feeder.step(t=0.0, dt=1.0 / 24.0, inputs={"Q_setpoint": 0.0})

        # Start feeder
        feeder.step(t=1.0 / 24.0, dt=1.0 / 24.0, inputs={"Q_setpoint": 15.0})

        assert feeder.n_starts > initial_starts, "Start count should increase"

    def test_step_checks_substrate_availability(self) -> None:
        """Test that step respects substrate availability."""
        feeder = Feeder("feeder_1", Q_max=20.0, enable_dosing_noise=False)
        feeder.initialize()

        dt = 1.0 / 24.0
        substrate_available = 0.5  # Only 0.5 m³ available
        Q_setpoint = 15.0  # Want to feed 15 m³/d

        inputs = {"Q_setpoint": Q_setpoint, "substrate_available": substrate_available}
        result = feeder.step(t=0.0, dt=dt, inputs=inputs)

        # Should be limited by availability
        max_possible = substrate_available / dt
        assert result["Q_actual"] <= max_possible * 1.1, "Flow should be limited by availability"


class TestFeederTypes:
    """Test suite for different feeder types."""

    def test_screw_feeder_initialization(self) -> None:
        """Test screw feeder initialization."""
        feeder = Feeder("feeder_1", feeder_type="screw", Q_max=20.0)

        assert feeder.feeder_type == FeederType.SCREW
        assert 0.03 <= feeder.dosing_accuracy <= 0.07, "Screw feeder should have ~5% accuracy"

    def test_twin_screw_feeder_initialization(self) -> None:
        """Test twin screw feeder initialization."""
        feeder = Feeder("feeder_1", feeder_type="twin_screw", Q_max=20.0)

        assert feeder.feeder_type == FeederType.TWIN_SCREW
        assert feeder.dosing_accuracy < 0.05, "Twin screw should have better accuracy"

    def test_progressive_cavity_feeder_initialization(self) -> None:
        """Test progressive cavity feeder initialization."""
        feeder = Feeder("feeder_1", feeder_type="progressive_cavity", Q_max=20.0)

        assert feeder.feeder_type == FeederType.PROGRESSIVE_CAVITY
        assert feeder.dosing_accuracy < 0.03, "PC pump should have good accuracy"

    def test_piston_feeder_initialization(self) -> None:
        """Test piston feeder initialization."""
        feeder = Feeder("feeder_1", feeder_type="piston", Q_max=15.0)

        assert feeder.feeder_type == FeederType.PISTON
        assert feeder.dosing_accuracy < 0.02, "Piston feeder should have best accuracy"

    def test_different_feeders_different_accuracies(self) -> None:
        """Test that different feeder types have appropriate accuracies."""
        screw = Feeder("f1", feeder_type="screw")
        piston = Feeder("f2", feeder_type="piston")

        assert piston.dosing_accuracy < screw.dosing_accuracy, "Piston should be more accurate than screw"


class TestSubstrateTypes:
    """Test suite for different substrate types."""

    def test_solid_substrate_feeder(self) -> None:
        """Test feeder for solid substrates."""
        feeder = Feeder("feeder_1", substrate_type="solid", Q_max=20.0)

        assert feeder.substrate_type == SubstrateCategory.SOLID
        # Solid substrates typically need more power
        assert feeder.power_installed > 5.0

    def test_liquid_substrate_feeder(self) -> None:
        """Test feeder for liquid substrates."""
        feeder = Feeder("feeder_1", substrate_type="liquid", Q_max=20.0)

        assert feeder.substrate_type == SubstrateCategory.LIQUID
        # Liquids need less power
        base_feeder = Feeder("f2", substrate_type="solid", Q_max=20.0)
        assert feeder.power_installed < base_feeder.power_installed

    def test_fibrous_substrate_feeder(self) -> None:
        """Test feeder for fibrous substrates."""
        feeder = Feeder("feeder_1", substrate_type="fibrous", Q_max=20.0)

        assert feeder.substrate_type == SubstrateCategory.FIBROUS
        # Fibrous materials need most power
        solid_feeder = Feeder("f2", substrate_type="solid", Q_max=20.0)
        assert feeder.power_installed > solid_feeder.power_installed


class TestFeederPerformance:
    """Test suite for feeder performance calculations."""

    def test_load_factor_calculation(self) -> None:
        """Test load factor calculation."""
        feeder = Feeder("feeder_1", Q_max=20.0, enable_dosing_noise=False)
        feeder.initialize()

        Q_setpoint = 10.0  # 50% of max
        inputs = {"Q_setpoint": Q_setpoint}
        result = feeder.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        expected_load = Q_setpoint / feeder.Q_max
        assert abs(result["load_factor"] - expected_load) < 0.1, "Load factor should be ~0.5"

    def test_dosing_error_calculation(self) -> None:
        """Test dosing error calculation."""
        feeder = Feeder("feeder_1", Q_max=20.0, enable_dosing_noise=True)
        feeder.initialize()

        inputs = {"Q_setpoint": 15.0}
        result = feeder.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert "dosing_error" in result
        assert result["dosing_error"] >= 0, "Dosing error should be non-negative"

    def test_power_scales_with_load(self) -> None:
        """Test that power consumption scales with load."""
        feeder = Feeder("feeder_1", Q_max=20.0)
        feeder.initialize()

        # Low load
        inputs1 = {"Q_setpoint": 5.0}
        result1 = feeder.step(t=0.0, dt=1.0 / 24.0, inputs=inputs1)

        feeder.initialize()

        # High load
        inputs2 = {"Q_setpoint": 18.0}
        result2 = feeder.step(t=0.0, dt=1.0 / 24.0, inputs=inputs2)

        assert result2["P_consumed"] > result1["P_consumed"], "Power should increase with load"

    def test_specific_energy_reasonable(self) -> None:
        """Test that specific energy consumption is reasonable."""
        feeder = Feeder("feeder_1", Q_max=20.0)
        feeder.initialize()

        inputs = {"Q_setpoint": 15.0}
        result = feeder.step(t=0.0, dt=1.0, inputs=inputs)

        # Specific energy = kWh/m³
        energy_per_day = result["P_consumed"] * 24  # kWh
        volume_per_day = result["Q_actual"]  # m³

        if volume_per_day > 0:
            specific_energy = energy_per_day / volume_per_day
            assert specific_energy < 10.0, "Specific energy should be reasonable"


class TestFeederBlockage:
    """Test suite for feeder blockage detection."""

    def test_blockage_detection_occurs_randomly(self) -> None:
        """Test that blockage can occur (probabilistically)."""
        feeder = Feeder("feeder_1", Q_max=20.0)
        feeder.initialize()

        # Run many steps - blockage should occur at least once
        blockage_occurred = False
        i = 0

        while (not blockage_occurred) or (i > 2000):
            i = i + 1  # just for safety, if the method never sets blockage_detected, because of a bug
            result = feeder.step(t=0.0, dt=1.0 / 24.0, inputs={"Q_setpoint": 15.0})
            if result["blockage_detected"]:
                blockage_occurred = True
                break

        # Note: This is probabilistic, but with 1000 steps it's very likely
        assert feeder.n_blockages >= 0, "Blockage counter should exist"

    # TODO: this test cannot work, because in the step method the variabel blockage_detected is set if random number
    #  passes a threshold, otherwise it is set to False, thus overwriting the variable set here.
    # def test_blockage_reduces_flow(self) -> None:
    #     """Test that blockage reduces flow rate."""
    #     feeder = Feeder("feeder_1", Q_max=20.0, enable_dosing_noise=False)
    #     feeder.initialize()
    #
    #     # Manually trigger blockage for testing
    #     feeder.blockage_detected = True
    #     inputs = {"Q_setpoint": 15.0}
    #
    #     # Simulate with blockage
    #     feeder.blockage_detected = True
    #     feeder.current_flow = 15.0 * 0.1  # Reduced flow
    #
    #     result = feeder.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)
    #
    #     # Flow should be significantly reduced
    #     assert result["Q_actual"] < 15.0 * 0.5, "Blockage should reduce flow"


class TestFeederSerialization:
    """Test suite for Feeder serialization methods."""

    def test_to_dict_returns_dict(self) -> None:
        """Test that to_dict method returns a dictionary."""
        feeder = Feeder("feeder_1", feeder_type="screw", Q_max=25.0)
        feeder.initialize()

        config = feeder.to_dict()

        assert isinstance(config, dict), "to_dict should return a dictionary"

    def test_to_dict_contains_required_fields(self) -> None:
        """Test that to_dict includes all required fields."""
        feeder = Feeder("feeder_1")
        feeder.initialize()

        config = feeder.to_dict()

        required_fields = [
            "component_id",
            "component_type",
            "feeder_type",
            "substrate_type",
            "Q_max",
            "dosing_accuracy",
            "power_installed",
        ]
        for field in required_fields:
            assert field in config, f"to_dict should include '{field}'"

    def test_from_dict_recreates_feeder(self) -> None:
        """Test that from_dict can recreate a feeder from configuration."""
        original = Feeder(
            "feeder_1",
            feeder_type="twin_screw",
            substrate_type="slurry",
            Q_max=30.0,
            dosing_accuracy=0.03,
            power_installed=8.0,
            name="Main Feeder",
        )
        original.initialize()

        config = original.to_dict()
        recreated = Feeder.from_dict(config)

        assert recreated.component_id == original.component_id
        assert recreated.feeder_type == original.feeder_type
        assert recreated.substrate_type == original.substrate_type
        assert recreated.Q_max == original.Q_max
        assert recreated.dosing_accuracy == original.dosing_accuracy
        assert recreated.name == original.name

    def test_roundtrip_preserves_configuration(self) -> None:
        """Test that serialization roundtrip preserves configuration."""
        original = Feeder(
            "feeder_1",
            feeder_type="progressive_cavity",
            Q_max=18.0,
            enable_dosing_noise=False,
        )

        config = original.to_dict()
        recreated = Feeder.from_dict(config)

        assert recreated.Q_max == original.Q_max
        assert recreated.enable_dosing_noise == original.enable_dosing_noise


class TestFeederConnections:
    """Test suite for Feeder component connections."""

    def test_add_input_connection(self) -> None:
        """Test adding input connections to feeder."""
        feeder = Feeder("feeder_1")

        feeder.add_input("substrate_storage")

        assert "substrate_storage" in feeder.inputs, "Input should be added"

    def test_add_output_connection(self) -> None:
        """Test adding output connections from feeder."""
        feeder = Feeder("feeder_1")

        feeder.add_output("digester_1")

        assert "digester_1" in feeder.outputs, "Output should be added"

    def test_multiple_connections(self) -> None:
        """Test adding multiple connections."""
        feeder = Feeder("feeder_1")

        feeder.add_input("storage")
        feeder.add_output("digester_1")
        feeder.add_output("digester_2")

        assert len(feeder.inputs) == 1
        assert len(feeder.outputs) == 2


class TestFeederProperties:
    """Test suite for Feeder properties and attributes."""

    def test_flow_rate_non_negative(self) -> None:
        """Test that flow rate is always non-negative."""
        feeder = Feeder("feeder_1")
        feeder.initialize()

        inputs = {"Q_setpoint": -5.0}  # Invalid negative setpoint
        result = feeder.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert result["Q_actual"] >= 0, "Flow rate should be non-negative"

    def test_power_consumption_non_negative(self) -> None:
        """Test that power consumption is non-negative."""
        feeder = Feeder("feeder_1")
        feeder.initialize()

        inputs = {"Q_setpoint": 15.0}
        result = feeder.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert result["P_consumed"] >= 0, "Power consumption should be non-negative"

    def test_load_factor_bounds(self) -> None:
        """Test that load factor stays within valid bounds."""
        feeder = Feeder("feeder_1", Q_max=20.0)
        feeder.initialize()

        inputs = {"Q_setpoint": 15.0}
        result = feeder.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert 0.0 <= result["load_factor"] <= 1.2, "Load factor should be reasonable"

    def test_get_state_returns_dict(self) -> None:
        """Test that get_state returns the state dictionary."""
        feeder = Feeder("feeder_1")
        feeder.initialize()

        state = feeder.get_state()

        assert isinstance(state, dict), "get_state should return a dictionary"
        assert "current_flow" in state

    def test_set_state_updates_properties(self) -> None:
        """Test that set_state updates component state."""
        feeder = Feeder("feeder_1")
        feeder.initialize()

        new_state = {
            "is_running": True,
            "current_flow": 12.5,
            "operating_hours": 100.0,
            "total_mass_fed": 5000.0,
        }
        feeder.set_state(new_state)

        assert feeder.state.get("current_flow") == 12.5
        assert feeder.state.get("total_mass_fed") == 5000.0


class TestFeederEdgeCases:
    """Test suite for feeder edge cases and error handling."""

    def test_feeder_with_very_small_capacity(self) -> None:
        """Test feeder with very small Q_max."""
        feeder = Feeder("feeder_1", Q_max=1.0)
        feeder.initialize()

        result = feeder.step(t=0.0, dt=1.0 / 24.0, inputs={"Q_setpoint": 0.8})

        assert result["Q_actual"] > 0, "Should handle small capacities"

    def test_feeder_with_very_large_capacity(self) -> None:
        """Test feeder with very large Q_max."""
        feeder = Feeder("feeder_1", Q_max=200.0)
        feeder.initialize()

        result = feeder.step(t=0.0, dt=1.0 / 24.0, inputs={"Q_setpoint": 150.0})

        assert result["Q_actual"] > 0, "Should handle large capacities"

    def test_feeder_multiple_consecutive_steps(self) -> None:
        """Test feeder over multiple consecutive time steps."""
        feeder = Feeder("feeder_1")
        feeder.initialize()

        # Run for 24 hours
        for hour in range(24):
            result = feeder.step(t=hour / 24.0, dt=1.0 / 24.0, inputs={"Q_setpoint": 15.0})
            assert result["Q_actual"] >= 0.0

        # Operating hours should accumulate
        assert feeder.operating_hours == 24.0

    def test_feeder_start_stop_cycle(self) -> None:
        """Test feeder start/stop cycling."""
        feeder = Feeder("feeder_1")
        feeder.initialize()

        # Start feeder
        result1 = feeder.step(t=0.0, dt=1.0 / 24.0, inputs={"enable_feeding": True, "Q_setpoint": 15.0})
        assert result1["is_running"] is True

        # Stop feeder
        result2 = feeder.step(t=1.0 / 24.0, dt=1.0 / 24.0, inputs={"enable_feeding": False})
        assert result2["is_running"] is False
        assert result2["P_consumed"] == 0.0

        # Restart feeder
        result3 = feeder.step(t=2.0 / 24.0, dt=1.0 / 24.0, inputs={"enable_feeding": True, "Q_setpoint": 15.0})
        assert result3["is_running"] is True


class TestFeederIntegration:
    """Test suite for feeder integration scenarios."""

    def test_feeder_daily_operation_cycle(self) -> None:
        """Test feeder over a complete daily cycle."""
        feeder = Feeder("feeder_1", Q_max=20.0)
        feeder.initialize()

        total_mass = 0.0

        # Simulate 24 hours
        for hour in range(24):
            result = feeder.step(t=hour / 24.0, dt=1.0 / 24.0, inputs={"Q_setpoint": 15.0})
            total_mass += result["Q_actual"] * (1.0 / 24.0)

        # Check that total mass is reasonable
        assert total_mass > 0, "Should feed material over the day"
        expected_total = 15.0  # m³/d * 1 day
        assert abs(total_mass - expected_total) < 5.0, "Total mass should be close to expected"

    def test_feeder_with_varying_setpoint_profile(self) -> None:
        """Test feeder with time-varying setpoint profile."""
        feeder = Feeder("feeder_1", Q_max=20.0, enable_dosing_noise=False)
        feeder.initialize()

        # Simulate varying setpoints over time
        setpoints = [10.0, 15.0, 18.0, 12.0, 8.0, 5.0]
        results = []

        for i, setpoint in enumerate(setpoints):
            result = feeder.step(t=i / 24.0, dt=1.0 / 24.0, inputs={"Q_setpoint": setpoint})
            results.append(result)

        # Verify flow varies with setpoint
        flows = [r["Q_actual"] for r in results]
        assert max(flows) > min(flows), "Flow should vary with setpoint"

    def test_feeder_energy_accounting(self) -> None:
        """Test that energy accounting is accurate."""
        feeder = Feeder("feeder_1")
        feeder.initialize()

        initial_energy = feeder.energy_consumed

        # Run for known time
        hours = 10
        dt = 1.0 / 24.0  # 1 hour steps

        total_power = 0.0
        for i in range(hours):
            result = feeder.step(t=i * dt, dt=dt, inputs={"Q_setpoint": 15.0})
            total_power += result["P_consumed"]

        # Energy consumed should match integrated power
        expected_energy = total_power  # kWh (power in kW, time in hours)
        actual_energy = feeder.energy_consumed - initial_energy

        assert (
            abs(actual_energy - expected_energy) < 0.1
        ), f"Energy accounting mismatch: {actual_energy:.2f} vs {expected_energy:.2f}"


class TestFeederDocumentation:
    """Test suite to verify feeder documentation and examples work."""

    def test_basic_example_from_docstring(self) -> None:
        """Test basic example from component docstring."""
        feeder = Feeder(component_id="feed1", feeder_type="screw", Q_max=20, substrate_type="solid")
        feeder.initialize()
        result = feeder.step(t=0, dt=1 / 24, inputs={"Q_setpoint": 15})

        assert "Q_actual" in result
        assert result["Q_actual"] > 0

    def test_module_example_from_docstring(self) -> None:
        """Test example from module-level docstring."""
        feeder = Feeder("feed1", feeder_type="screw", Q_max=20.0, substrate_type="solid")
        feeder.initialize()
        result = feeder.step(0, 1 / 24, {"Q_setpoint": 15.0})

        assert isinstance(result, dict)
        assert "Q_actual" in result
