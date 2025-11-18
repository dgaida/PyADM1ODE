# tests/unit/test_components/test_pump.py
# -*- coding: utf-8 -*-
"""
Unit tests for the Pump component.

This module tests the Pump class which models different pump types
for substrate feeding, recirculation, and digestate handling in biogas plants.
"""

from pyadm1.components.mechanical.pump import Pump, PumpType


class TestPumpInitialization:
    """Test suite for Pump component initialization."""

    def test_pump_initialization_sets_component_id(self) -> None:
        """Test that pump initialization sets the component_id."""
        pump = Pump("pump_1")

        assert pump.component_id == "pump_1", "Component ID should be set correctly"

    def test_pump_initialization_sets_flow_rate(self) -> None:
        """Test that pump initialization sets nominal flow rate."""
        Q_nom = 15.0
        pump = Pump("pump_1", Q_nom=Q_nom)

        assert pump.Q_nom == Q_nom, f"Q_nom should be {Q_nom}"

    def test_pump_initialization_sets_pressure_head(self) -> None:
        """Test that pump initialization sets pressure head."""
        pressure_head = 50.0
        pump = Pump("pump_1", pressure_head=pressure_head)

        assert pump.pressure_head == pressure_head, f"pressure_head should be {pressure_head}"

    def test_pump_initialization_with_pump_type(self) -> None:
        """Test that pump accepts different pump types."""
        pump = Pump("pump_1", pump_type="centrifugal")

        assert pump.pump_type == PumpType.CENTRIFUGAL, "Pump type should be CENTRIFUGAL"

    def test_pump_initialization_default_values(self) -> None:
        """Test that pump has reasonable default values."""
        pump = Pump("pump_1")

        assert pump.Q_nom == 10.0, "Default Q_nom should be 10.0 m³/h"
        assert pump.pressure_head == 50.0, "Default pressure_head should be 50.0 m"
        assert pump.pump_type == PumpType.PROGRESSIVE_CAVITY, "Default type should be progressive_cavity"

    def test_pump_initialization_with_custom_name(self) -> None:
        """Test that pump accepts custom name parameter."""
        custom_name = "Feed Pump 1"
        pump = Pump("pump_1", name=custom_name)

        assert pump.name == custom_name, f"Name should be '{custom_name}'"

    def test_pump_initialization_estimates_efficiency(self) -> None:
        """Test that pump estimates efficiency if not provided."""
        pump = Pump("pump_1")

        assert 0.3 <= pump.efficiency <= 0.9, "Efficiency should be in reasonable range"

    def test_pump_initialization_accepts_custom_efficiency(self) -> None:
        """Test that pump accepts custom efficiency value."""
        custom_efficiency = 0.75
        pump = Pump("pump_1", efficiency=custom_efficiency)

        assert pump.efficiency == custom_efficiency, "Custom efficiency should be set"

    def test_pump_initialization_creates_state_dict(self) -> None:
        """Test that initialization creates state dictionary."""
        pump = Pump("pump_1")

        assert hasattr(pump, "state"), "Pump should have state attribute"
        assert isinstance(pump.state, dict), "state should be a dictionary"


class TestPumpInitialize:
    """Test suite for Pump initialize method."""

    def test_initialize_sets_pump_off(self) -> None:
        """Test that initialize sets pump to off state."""
        pump = Pump("pump_1")
        pump.initialize()

        assert pump.is_running is False, "Pump should be off initially"
        assert pump.current_flow == 0.0, "Initial flow should be zero"

    def test_initialize_creates_state_dict(self) -> None:
        """Test that initialize creates proper state dictionary."""
        pump = Pump("pump_1")
        pump.initialize()

        required_keys = [
            "is_running",
            "current_flow",
            "operating_hours",
            "energy_consumed",
            "total_volume_pumped",
            "efficiency",
        ]
        for key in required_keys:
            assert key in pump.state, f"State should have '{key}' key"

    def test_initialize_with_custom_state(self) -> None:
        """Test initialize with custom initial state."""
        pump = Pump("pump_1")
        pump.initialize({"is_running": True, "current_flow": 8.0, "operating_hours": 100.0})

        assert pump.is_running is True, "Pump should be running"
        assert pump.current_flow == 8.0, "Flow should be set to 8.0"
        assert pump.operating_hours == 100.0, "Operating hours should be set"

    def test_initialize_creates_outputs_data(self) -> None:
        """Test that initialize creates outputs_data dictionary."""
        pump = Pump("pump_1")
        pump.initialize()

        assert hasattr(pump, "outputs_data"), "Pump should have outputs_data"
        assert isinstance(pump.outputs_data, dict), "outputs_data should be a dictionary"


class TestPumpStep:
    """Test suite for Pump step method (simulation)."""

    def test_step_returns_dict(self) -> None:
        """Test that step method returns a dictionary."""
        pump = Pump("pump_1", Q_nom=10.0)
        pump.initialize()

        inputs = {"Q_setpoint": 8.0}
        result = pump.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert isinstance(result, dict), "step should return a dictionary"

    def test_step_with_pump_disabled(self) -> None:
        """Test step with pump disabled."""
        pump = Pump("pump_1")
        pump.initialize()

        inputs = {"enable_pump": False, "Q_setpoint": 10.0}
        result = pump.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert result["P_consumed"] == 0.0, "Power should be zero when disabled"
        assert result["Q_actual"] == 0.0, "Flow should be zero when disabled"
        assert result["is_running"] is False, "Pump should not be running"

    def test_step_output_contains_required_fields(self) -> None:
        """Test that step output contains required information."""
        pump = Pump("pump_1")
        pump.initialize()

        inputs = {"Q_setpoint": 8.0}
        result = pump.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert "P_consumed" in result, "Result should contain P_consumed"
        assert "Q_actual" in result, "Result should contain Q_actual"
        assert "is_running" in result, "Result should contain is_running"
        assert "efficiency" in result, "Result should contain efficiency"
        assert "pressure_actual" in result, "Result should contain pressure_actual"

    def test_step_calculates_power_consumption(self) -> None:
        """Test that step correctly calculates power consumption."""
        pump = Pump("pump_1", Q_nom=10.0, pressure_head=50.0)
        pump.initialize()

        inputs = {"Q_setpoint": 8.0}
        result = pump.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert result["P_consumed"] > 0, "Pump should consume power when running"
        assert result["Q_actual"] > 0, "Pump should deliver flow when running"

    def test_step_respects_flow_setpoint(self) -> None:
        """Test that step respects flow setpoint with VSD."""
        pump = Pump("pump_1", Q_nom=10.0, speed_control=True)
        pump.initialize()

        Q_setpoint = 6.0
        inputs = {"Q_setpoint": Q_setpoint}
        result = pump.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert abs(result["Q_actual"] - Q_setpoint) < 0.1, "Flow should match setpoint"

    def test_step_fixed_speed_pump(self) -> None:
        """Test that fixed speed pump runs at nominal flow."""
        pump = Pump("pump_1", Q_nom=10.0, speed_control=False)
        pump.initialize()

        inputs = {"Q_setpoint": 6.0}
        result = pump.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert abs(result["Q_actual"] - pump.Q_nom) < 0.1, "Fixed speed pump should run at Q_nom"

    def test_step_updates_operating_hours(self) -> None:
        """Test that step updates operating hours."""
        pump = Pump("pump_1")
        pump.initialize()

        initial_hours = pump.operating_hours
        dt = 1.0 / 24.0  # 1 hour

        inputs = {"Q_setpoint": 8.0}
        pump.step(t=0.0, dt=dt, inputs=inputs)

        assert pump.operating_hours > initial_hours, "Operating hours should increase"
        assert abs(pump.operating_hours - (initial_hours + 1.0)) < 0.01, "Should increase by 1 hour"

    def test_step_updates_total_volume_pumped(self) -> None:
        """Test that step updates total volume pumped."""
        pump = Pump("pump_1", Q_nom=10.0)
        pump.initialize()

        # initial_volume = pump.total_volume_pumped
        dt = 1.0 / 24.0  # 1 hour
        Q_setpoint = 8.0  # m³/h

        inputs = {"Q_setpoint": Q_setpoint}
        pump.step(t=0.0, dt=dt, inputs=inputs)

        expected_volume = Q_setpoint * 1.0  # m³
        assert abs(pump.total_volume_pumped - expected_volume) < 0.1, "Volume should be accumulated"

    def test_step_with_zero_setpoint(self) -> None:
        """Test that pump stops with zero setpoint."""
        pump = Pump("pump_1")
        pump.initialize()

        inputs = {"Q_setpoint": 0.0}
        result = pump.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert result["P_consumed"] == 0.0, "Power should be zero with zero setpoint"
        assert result["Q_actual"] == 0.0, "Flow should be zero"


class TestPumpTypes:
    """Test suite for different pump types."""

    def test_centrifugal_pump_initialization(self) -> None:
        """Test centrifugal pump initialization."""
        pump = Pump("pump_1", pump_type="centrifugal", Q_nom=20.0)

        assert pump.pump_type == PumpType.CENTRIFUGAL
        assert pump.efficiency > 0.5, "Centrifugal pump should have reasonable efficiency"

    def test_progressive_cavity_pump_initialization(self) -> None:
        """Test progressive cavity pump initialization."""
        pump = Pump("pump_1", pump_type="progressive_cavity", Q_nom=10.0)

        assert pump.pump_type == PumpType.PROGRESSIVE_CAVITY
        assert pump.efficiency > 0.4, "PC pump should have reasonable efficiency"

    def test_piston_pump_initialization(self) -> None:
        """Test piston pump initialization."""
        pump = Pump("pump_1", pump_type="piston", Q_nom=5.0)

        assert pump.pump_type == PumpType.PISTON
        assert pump.efficiency > 0.6, "Piston pump should have good efficiency"

    def test_different_pumps_different_efficiencies(self) -> None:
        """Test that different pump types have appropriate efficiencies."""
        centrifugal = Pump("p1", pump_type="centrifugal", Q_nom=50.0)
        progressive = Pump("p2", pump_type="progressive_cavity", Q_nom=50.0)
        piston = Pump("p3", pump_type="piston", Q_nom=50.0)

        # Generally: piston > centrifugal > progressive_cavity for efficiency
        assert piston.efficiency >= progressive.efficiency
        assert centrifugal.efficiency >= progressive.efficiency


class TestPumpEfficiency:
    """Test suite for pump efficiency calculations."""

    def test_efficiency_at_nominal_point(self) -> None:
        """Test efficiency at nominal operating point."""
        pump = Pump("pump_1", Q_nom=10.0, efficiency=0.70)
        pump.initialize()

        inputs = {"Q_setpoint": 10.0}
        result = pump.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert abs(result["efficiency"] - 0.70) < 0.1, "Efficiency at nominal should be close to design"

    def test_efficiency_at_part_load(self) -> None:
        """Test that efficiency changes at part load."""
        pump = Pump("pump_1", Q_nom=10.0, efficiency=0.70, pump_type="centrifugal")
        pump.initialize()

        inputs = {"Q_setpoint": 5.0}  # 50% load
        result = pump.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        # Efficiency should decrease at part load for centrifugal pumps
        assert result["efficiency"] > 0.3, "Efficiency should still be reasonable at part load"

    def test_power_increases_with_flow(self) -> None:
        """Test that power consumption increases with flow rate."""
        pump = Pump("pump_1", Q_nom=10.0)
        pump.initialize()

        # Test at 50% flow
        inputs1 = {"Q_setpoint": 5.0}
        result1 = pump.step(t=0.0, dt=1.0 / 24.0, inputs=inputs1)

        pump.initialize()  # Reset

        # Test at 100% flow
        inputs2 = {"Q_setpoint": 10.0}
        result2 = pump.step(t=0.0, dt=1.0 / 24.0, inputs=inputs2)

        assert result2["P_consumed"] > result1["P_consumed"], "Power should increase with flow"


class TestPumpPressure:
    """Test suite for pump pressure calculations."""

    def test_pressure_head_calculation(self) -> None:
        """Test pressure head calculation."""
        pump = Pump("pump_1", Q_nom=10.0, pressure_head=50.0)
        pump.initialize()

        inputs = {"Q_setpoint": 8.0}
        result = pump.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert result["pressure_actual"] > 0, "Pressure should be positive"

    def test_centrifugal_pressure_decreases_with_flow(self) -> None:
        """Test that centrifugal pump head decreases with flow."""
        pump = Pump("pump_1", pump_type="centrifugal", Q_nom=10.0, pressure_head=50.0)
        pump.initialize()

        # Low flow
        inputs1 = {"Q_setpoint": 3.0}
        result1 = pump.step(t=0.0, dt=1.0 / 24.0, inputs=inputs1)

        pump.initialize()

        # High flow
        inputs2 = {"Q_setpoint": 10.0}
        result2 = pump.step(t=0.0, dt=1.0 / 24.0, inputs=inputs2)

        # For centrifugal pumps, head decreases with flow
        assert result1["pressure_actual"] >= result2["pressure_actual"], "Centrifugal pump head should decrease with flow"

    def test_volumetric_pump_constant_pressure(self) -> None:
        """Test that volumetric pumps maintain nearly constant pressure."""
        pump = Pump("pump_1", pump_type="progressive_cavity", Q_nom=10.0, pressure_head=50.0)
        pump.initialize()

        inputs = {"Q_setpoint": 8.0}
        result = pump.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        # Pressure should be close to design value
        assert abs(result["pressure_actual"] - 50.0) < 5.0, "Volumetric pump should maintain nearly constant pressure"


class TestPumpSerialization:
    """Test suite for Pump serialization methods."""

    def test_to_dict_returns_dict(self) -> None:
        """Test that to_dict method returns a dictionary."""
        pump = Pump("pump_1", Q_nom=15.0, pressure_head=60.0)
        pump.initialize()

        config = pump.to_dict()

        assert isinstance(config, dict), "to_dict should return a dictionary"

    def test_to_dict_contains_required_fields(self) -> None:
        """Test that to_dict includes all required fields."""
        pump = Pump("pump_1", Q_nom=15.0)
        pump.initialize()

        config = pump.to_dict()

        required_fields = ["component_id", "component_type", "pump_type", "Q_nom", "pressure_head", "efficiency"]
        for field in required_fields:
            assert field in config, f"to_dict should include '{field}'"

    def test_from_dict_recreates_pump(self) -> None:
        """Test that from_dict can recreate a pump from configuration."""
        original = Pump("pump_1", pump_type="centrifugal", Q_nom=20.0, pressure_head=60.0, efficiency=0.75, name="Main Pump")
        original.initialize()

        config = original.to_dict()
        recreated = Pump.from_dict(config)

        assert recreated.component_id == original.component_id
        assert recreated.pump_type == original.pump_type
        assert recreated.Q_nom == original.Q_nom
        assert recreated.pressure_head == original.pressure_head
        assert recreated.efficiency == original.efficiency
        assert recreated.name == original.name

    def test_roundtrip_preserves_configuration(self) -> None:
        """Test that serialization roundtrip preserves configuration."""
        original = Pump("pump_1", pump_type="piston", Q_nom=8.0, pressure_head=100.0, speed_control=False)

        config = original.to_dict()
        recreated = Pump.from_dict(config)

        assert recreated.Q_nom == original.Q_nom
        assert recreated.pressure_head == original.pressure_head
        assert recreated.speed_control == original.speed_control


class TestPumpConnections:
    """Test suite for Pump component connections."""

    def test_add_input_connection(self) -> None:
        """Test adding input connections to pump."""
        pump = Pump("pump_1")

        pump.add_input("storage_tank")

        assert "storage_tank" in pump.inputs, "Input should be added"

    def test_add_output_connection(self) -> None:
        """Test adding output connections from pump."""
        pump = Pump("pump_1")

        pump.add_output("digester_1")

        assert "digester_1" in pump.outputs, "Output should be added"

    def test_multiple_connections(self) -> None:
        """Test adding multiple connections."""
        pump = Pump("pump_1")

        pump.add_input("tank")
        pump.add_output("digester_1")
        pump.add_output("digester_2")

        assert len(pump.inputs) == 1
        assert len(pump.outputs) == 2


class TestPumpProperties:
    """Test suite for Pump properties and attributes."""

    def test_flow_rate_non_negative(self) -> None:
        """Test that flow rate is always non-negative."""
        pump = Pump("pump_1")
        pump.initialize()

        inputs = {"Q_setpoint": -5.0}  # Invalid negative setpoint
        result = pump.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert result["Q_actual"] >= 0, "Flow rate should be non-negative"

    def test_power_consumption_non_negative(self) -> None:
        """Test that power consumption is non-negative."""
        pump = Pump("pump_1")
        pump.initialize()

        inputs = {"Q_setpoint": 8.0}
        result = pump.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert result["P_consumed"] >= 0, "Power consumption should be non-negative"

    def test_efficiency_bounds(self) -> None:
        """Test that efficiency stays within valid bounds."""
        pump = Pump("pump_1")
        pump.initialize()

        inputs = {"Q_setpoint": 5.0}
        result = pump.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert 0.0 <= result["efficiency"] <= 1.0, "Efficiency should be between 0 and 1"

    def test_speed_fraction_bounds(self) -> None:
        """Test that speed fraction stays within reasonable bounds."""
        pump = Pump("pump_1", speed_control=True)
        pump.initialize()

        inputs = {"Q_setpoint": 12.0}  # Overload
        result = pump.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert 0.0 <= result["speed_fraction"] <= 1.3, "Speed fraction should be reasonable"

    def test_get_state_returns_dict(self) -> None:
        """Test that get_state returns the state dictionary."""
        pump = Pump("pump_1")
        pump.initialize()

        state = pump.get_state()

        assert isinstance(state, dict), "get_state should return a dictionary"
        assert "current_flow" in state

    def test_set_state_updates_properties(self) -> None:
        """Test that set_state updates component state."""
        pump = Pump("pump_1")
        pump.initialize()

        new_state = {"is_running": True, "current_flow": 7.5, "operating_hours": 50.0}
        pump.set_state(new_state)

        assert pump.state.get("current_flow") == 7.5


class TestPumpFluidProperties:
    """Test suite for handling different fluid properties."""

    def test_pump_with_different_density(self) -> None:
        """Test pump power calculation with different fluid density."""
        pump1 = Pump("pump_1", fluid_density=1000.0)  # Water
        pump2 = Pump("pump_2", fluid_density=1100.0)  # Denser fluid

        pump1.initialize()
        pump2.initialize()

        inputs = {"Q_setpoint": 10.0}
        result1 = pump1.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)
        result2 = pump2.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        # Denser fluid should require more power
        assert result2["P_consumed"] > result1["P_consumed"], "Higher density fluid should require more power"

    def test_pump_accepts_runtime_density_update(self) -> None:
        """Test that pump accepts fluid density updates during simulation."""
        pump = Pump("pump_1", fluid_density=1000.0)
        pump.initialize()

        inputs = {"Q_setpoint": 10.0, "fluid_density": 1200.0}
        pump.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert pump.fluid_density == 1200.0, "Fluid density should be updated"
