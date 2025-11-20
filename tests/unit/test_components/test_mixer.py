# tests/unit/test_components/test_mixer.py
# -*- coding: utf-8 -*-
"""
Unit tests for the Mixer component.

This module tests the Mixer class which models agitators and stirrers
for maintaining homogeneity in anaerobic digesters.
"""

import numpy as np
from pyadm1.components.mechanical.mixer import Mixer, MixerType, MixingIntensity


class TestMixerInitialization:
    """Test suite for Mixer component initialization."""

    def test_mixer_initialization_sets_component_id(self) -> None:
        """Test that mixer initialization sets the component_id."""
        mixer = Mixer("mixer_1")

        assert mixer.component_id == "mixer_1", "Component ID should be set correctly"

    def test_mixer_initialization_sets_mixer_type(self) -> None:
        """Test that mixer initialization sets mixer type."""
        mixer = Mixer("mixer_1", mixer_type="propeller")

        assert mixer.mixer_type == MixerType.PROPELLER, "Mixer type should be PROPELLER"

    def test_mixer_initialization_sets_tank_volume(self) -> None:
        """Test that mixer initialization sets tank volume."""
        tank_volume = 2500.0
        mixer = Mixer("mixer_1", tank_volume=tank_volume)

        assert mixer.tank_volume == tank_volume, f"Tank volume should be {tank_volume}"

    def test_mixer_initialization_sets_mixing_intensity(self) -> None:
        """Test that mixer initialization sets mixing intensity."""
        mixer = Mixer("mixer_1", mixing_intensity="high")

        assert mixer.mixing_intensity == MixingIntensity.HIGH, "Intensity should be HIGH"

    def test_mixer_initialization_default_values(self) -> None:
        """Test that mixer has reasonable default values."""
        mixer = Mixer("mixer_1")

        assert mixer.tank_volume == 2000.0, "Default tank volume should be 2000.0 m³"
        assert mixer.mixer_type == MixerType.PROPELLER, "Default type should be propeller"
        assert mixer.mixing_intensity == MixingIntensity.MEDIUM, "Default intensity should be medium"

    def test_mixer_initialization_with_custom_name(self) -> None:
        """Test that mixer accepts custom name parameter."""
        custom_name = "Main Digester Mixer"
        mixer = Mixer("mixer_1", name=custom_name)

        assert mixer.name == custom_name, f"Name should be '{custom_name}'"

    def test_mixer_initialization_estimates_tank_geometry(self) -> None:
        """Test that mixer estimates tank dimensions if not provided."""
        mixer = Mixer("mixer_1", tank_volume=2000.0)

        assert mixer.tank_diameter > 0, "Tank diameter should be estimated"
        assert mixer.tank_height > 0, "Tank height should be estimated"

    def test_mixer_initialization_accepts_custom_geometry(self) -> None:
        """Test that mixer accepts custom tank geometry."""
        mixer = Mixer("mixer_1", tank_volume=2000.0, tank_diameter=15.0, tank_height=12.0)

        assert mixer.tank_diameter == 15.0, "Tank diameter should be set"
        assert mixer.tank_height == 12.0, "Tank height should be set"

    def test_mixer_initialization_estimates_impeller_diameter(self) -> None:
        """Test that mixer estimates impeller diameter."""
        mixer = Mixer("mixer_1", tank_diameter=15.0)

        assert mixer.impeller_diameter > 0, "Impeller diameter should be estimated"
        assert mixer.impeller_diameter < mixer.tank_diameter, "Impeller should be smaller than tank"

    def test_mixer_initialization_estimates_power(self) -> None:
        """Test that mixer estimates power requirement if not provided."""
        mixer = Mixer("mixer_1", tank_volume=2000.0)

        assert mixer.power_installed > 0, "Power should be estimated"
        assert mixer.power_installed < 50.0, "Power should be reasonable for tank size"

    def test_mixer_initialization_accepts_custom_power(self) -> None:
        """Test that mixer accepts custom power value."""
        custom_power = 18.0
        mixer = Mixer("mixer_1", power_installed=custom_power)

        assert mixer.power_installed == custom_power, "Custom power should be set"

    def test_mixer_initialization_creates_state_dict(self) -> None:
        """Test that initialization creates state dictionary."""
        mixer = Mixer("mixer_1")

        assert hasattr(mixer, "state"), "Mixer should have state attribute"
        assert isinstance(mixer.state, dict), "state should be a dictionary"


class TestMixerInitialize:
    """Test suite for Mixer initialize method."""

    def test_initialize_sets_mixer_running(self) -> None:
        """Test that initialize sets mixer to running state by default."""
        mixer = Mixer("mixer_1")
        mixer.initialize()

        assert mixer.is_running is True, "Mixer should be running initially"

    def test_initialize_creates_state_dict(self) -> None:
        """Test that initialize creates proper state dictionary."""
        mixer = Mixer("mixer_1")
        mixer.initialize()

        required_keys = [
            "is_running",
            "current_speed_fraction",
            "operating_hours",
            "energy_consumed",
            "power_number",
            "reynolds_number",
            "mixing_time",
        ]
        for key in required_keys:
            assert key in mixer.state, f"State should have '{key}' key"

    def test_initialize_with_custom_state(self) -> None:
        """Test initialize with custom initial state."""
        mixer = Mixer("mixer_1")
        mixer.initialize({"is_running": False, "current_speed_fraction": 0.5, "operating_hours": 500.0})

        assert mixer.is_running is False, "Mixer should be stopped"
        assert mixer.current_speed_fraction == 0.5, "Speed should be set to 0.5"
        assert mixer.operating_hours == 500.0, "Operating hours should be set"

    def test_initialize_calculates_mixing_parameters(self) -> None:
        """Test that initialize calculates mixing parameters."""
        mixer = Mixer("mixer_1")
        mixer.initialize()

        assert mixer.power_number > 0, "Power number should be calculated"
        assert mixer.reynolds_number > 0, "Reynolds number should be calculated"
        assert mixer.mixing_time > 0, "Mixing time should be calculated"

    def test_initialize_creates_outputs_data(self) -> None:
        """Test that initialize creates outputs_data dictionary."""
        mixer = Mixer("mixer_1")
        mixer.initialize()

        assert hasattr(mixer, "outputs_data"), "Mixer should have outputs_data"
        assert isinstance(mixer.outputs_data, dict), "outputs_data should be a dictionary"


class TestMixerStep:
    """Test suite for Mixer step method (simulation)."""

    def test_step_returns_dict(self) -> None:
        """Test that step method returns a dictionary."""
        mixer = Mixer("mixer_1")
        mixer.initialize()

        inputs = {}
        result = mixer.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert isinstance(result, dict), "step should return a dictionary"

    def test_step_with_mixer_disabled(self) -> None:
        """Test step with mixer disabled."""
        mixer = Mixer("mixer_1")
        mixer.initialize()

        inputs = {"enable_mixing": False}
        result = mixer.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert result["P_consumed"] == 0.0, "Power should be zero when disabled"
        assert result["is_running"] is False, "Mixer should not be running"

    def test_step_output_contains_required_fields(self) -> None:
        """Test that step output contains required information."""
        mixer = Mixer("mixer_1")
        mixer.initialize()

        inputs = {}
        result = mixer.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert "P_consumed" in result, "Result should contain P_consumed"
        assert "P_average" in result, "Result should contain P_average"
        assert "is_running" in result, "Result should contain is_running"
        assert "mixing_quality" in result, "Result should contain mixing_quality"
        assert "reynolds_number" in result, "Result should contain reynolds_number"
        assert "power_number" in result, "Result should contain power_number"

    def test_step_calculates_power_consumption(self) -> None:
        """Test that step correctly calculates power consumption."""
        mixer = Mixer("mixer_1", tank_volume=2000.0, power_installed=15.0)
        mixer.initialize()

        inputs = {}
        result = mixer.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert result["P_consumed"] > 0, "Mixer should consume power when running"
        assert result["P_consumed"] <= mixer.power_installed * 1.1, "Power should not significantly exceed installed power"

    def test_step_respects_speed_setpoint(self) -> None:
        """Test that step respects speed setpoint."""
        mixer = Mixer("mixer_1")
        mixer.initialize()

        speed_setpoint = 0.7
        inputs = {"speed_setpoint": speed_setpoint}
        mixer.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert abs(mixer.current_speed_fraction - speed_setpoint) < 0.01, "Speed should match setpoint"

    def test_step_updates_operating_hours(self) -> None:
        """Test that step updates operating hours."""
        mixer = Mixer("mixer_1")
        mixer.initialize()

        initial_hours = mixer.operating_hours
        dt = 1.0 / 24.0  # 1 hour

        inputs = {}
        mixer.step(t=0.0, dt=dt, inputs=inputs)

        assert mixer.operating_hours > initial_hours, "Operating hours should increase"
        assert abs(mixer.operating_hours - (initial_hours + 1.0)) < 0.01, "Should increase by 1 hour"

    def test_step_updates_energy_consumed(self) -> None:
        """Test that step updates energy consumption."""
        mixer = Mixer("mixer_1")
        mixer.initialize()

        mixer.energy_consumed
        dt = 1.0 / 24.0  # 1 hour

        inputs = {}
        result = mixer.step(t=0.0, dt=dt, inputs=inputs)

        expected_energy = result["P_consumed"] * 1.0  # kWh
        assert abs(mixer.energy_consumed - expected_energy) < 0.1, "Energy should be accumulated correctly"

    def test_step_intermittent_operation(self) -> None:
        """Test intermittent mixer operation."""
        mixer = Mixer("mixer_1", intermittent=True, on_time_fraction=0.25)
        mixer.initialize()

        inputs = {}

        # Run multiple steps and check that mixer cycles on/off
        running_count = 0
        for i in range(10):
            result = mixer.step(t=i / 24.0, dt=1.0 / 24.0, inputs=inputs)
            if result["is_running"]:
                running_count += 1

        # Should be running approximately 25% of the time
        assert running_count > 0, "Mixer should run at least once"
        assert running_count < 10, "Mixer should not run all the time"

    def test_step_calculates_mixing_quality(self) -> None:
        """Test that step calculates mixing quality."""
        mixer = Mixer("mixer_1")
        mixer.initialize()

        inputs = {}
        result = mixer.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert 0.0 <= result["mixing_quality"] <= 1.0, "Mixing quality should be between 0 and 1"

    def test_step_calculates_shear_rate(self) -> None:
        """Test that step calculates shear rate."""
        mixer = Mixer("mixer_1")
        mixer.initialize()

        inputs = {}
        result = mixer.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert "shear_rate" in result, "Result should contain shear_rate"
        assert result["shear_rate"] > 0, "Shear rate should be positive when running"


class TestMixerTypes:
    """Test suite for different mixer types."""

    def test_propeller_mixer_initialization(self) -> None:
        """Test propeller mixer initialization."""
        mixer = Mixer("mixer_1", mixer_type="propeller", tank_volume=2000.0)

        assert mixer.mixer_type == MixerType.PROPELLER
        # Propeller should have D/DT ≈ 0.33
        expected_ratio = 0.33
        actual_ratio = mixer.impeller_diameter / mixer.tank_diameter
        assert abs(actual_ratio - expected_ratio) < 0.05, "Propeller D/DT ratio should be around 0.33"

    def test_paddle_mixer_initialization(self) -> None:
        """Test paddle mixer initialization."""
        mixer = Mixer("mixer_1", mixer_type="paddle", tank_volume=2000.0)

        assert mixer.mixer_type == MixerType.PADDLE
        # Paddle should have larger D/DT ≈ 0.50
        expected_ratio = 0.50
        actual_ratio = mixer.impeller_diameter / mixer.tank_diameter
        assert abs(actual_ratio - expected_ratio) < 0.05, "Paddle D/DT ratio should be around 0.50"

    def test_jet_mixer_initialization(self) -> None:
        """Test jet mixer initialization."""
        mixer = Mixer("mixer_1", mixer_type="jet", tank_volume=2000.0)

        assert mixer.mixer_type == MixerType.JET
        # Jet should have smaller D/DT (nozzle diameter)
        actual_ratio = mixer.impeller_diameter / mixer.tank_diameter
        assert actual_ratio < 0.15, "Jet D/DT ratio should be small"

    def test_different_mixers_different_speeds(self) -> None:
        """Test that different mixer types have appropriate speeds."""
        propeller = Mixer("m1", mixer_type="propeller", tank_volume=2000.0)
        paddle = Mixer("m2", mixer_type="paddle", tank_volume=2000.0)

        # Propellers typically run faster than paddles
        assert propeller.operating_speed > paddle.operating_speed, "Propeller should run faster than paddle"

    def test_different_mixers_different_power(self) -> None:
        """Test that mixer power scales with type."""
        propeller = Mixer("m1", mixer_type="propeller", tank_volume=2000.0, mixing_intensity="medium")
        Mixer("m2", mixer_type="paddle", tank_volume=2000.0, mixing_intensity="medium")
        jet = Mixer("m3", mixer_type="jet", tank_volume=2000.0, mixing_intensity="medium")

        # Jet mixers typically need more power (include pump)
        assert jet.power_installed >= propeller.power_installed, "Jet mixer should need more power"


class TestMixingIntensity:
    """Test suite for mixing intensity levels."""

    def test_low_intensity_lower_power(self) -> None:
        """Test that low intensity requires less power."""
        low = Mixer("m1", mixing_intensity="low", tank_volume=2000.0)
        medium = Mixer("m2", mixing_intensity="medium", tank_volume=2000.0)

        assert low.power_installed < medium.power_installed, "Low intensity should require less power"

    def test_high_intensity_higher_power(self) -> None:
        """Test that high intensity requires more power."""
        medium = Mixer("m1", mixing_intensity="medium", tank_volume=2000.0)
        high = Mixer("m2", mixing_intensity="high", tank_volume=2000.0)

        assert high.power_installed > medium.power_installed, "High intensity should require more power"

    def test_intensity_affects_mixing_quality(self) -> None:
        """Test that intensity affects mixing quality."""
        low = Mixer("m1", mixing_intensity="low", tank_volume=2000.0)
        high = Mixer("m2", mixing_intensity="high", tank_volume=2000.0)

        low.initialize()
        high.initialize()

        low.step(t=0.0, dt=1.0 / 24.0, inputs={})
        high.step(t=0.0, dt=1.0 / 24.0, inputs={})

        # Higher intensity should give better mixing (lower mixing time)
        assert low.mixing_time >= high.mixing_time, "High intensity should have shorter mixing time"


class TestMixerPerformance:
    """Test suite for mixer performance calculations."""

    def test_reynolds_number_calculation(self) -> None:
        """Test Reynolds number calculation."""
        mixer = Mixer("mixer_1", tank_volume=2000.0)
        mixer.initialize()

        assert mixer.reynolds_number > 0, "Reynolds number should be positive"
        # For typical biogas conditions, should be turbulent (Re > 10000)
        assert mixer.reynolds_number > 1000, "Reynolds number should indicate turbulent flow"

    def test_power_number_calculation(self) -> None:
        """Test power number calculation."""
        mixer = Mixer("mixer_1", mixer_type="propeller")
        mixer.initialize()

        assert mixer.power_number > 0, "Power number should be positive"
        # For propeller in turbulent regime, Np ≈ 0.3-0.5
        assert 0.1 <= mixer.power_number <= 10.0, "Power number should be in reasonable range"

    def test_mixing_time_calculation(self) -> None:
        """Test mixing time calculation."""
        mixer = Mixer("mixer_1", tank_volume=2000.0)
        mixer.initialize()

        assert mixer.mixing_time > 0, "Mixing time should be positive"
        # For good mixing in biogas digesters: < 30 minutes
        assert mixer.mixing_time < 60.0, "Mixing time should be reasonable"

    def test_tip_speed_calculation(self) -> None:
        """Test impeller tip speed calculation."""
        mixer = Mixer("mixer_1")
        mixer.initialize()

        inputs = {}
        result = mixer.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert "tip_speed" in result, "Result should contain tip_speed"
        assert result["tip_speed"] > 0, "Tip speed should be positive"
        # Typical tip speeds: 3-10 m/s
        assert result["tip_speed"] < 20.0, "Tip speed should be reasonable"

    def test_specific_power_calculation(self) -> None:
        """Test specific power calculation."""
        mixer = Mixer("mixer_1", tank_volume=2000.0)
        mixer.initialize()

        inputs = {}
        result = mixer.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert "specific_power" in result, "Result should contain specific_power"
        specific_power = result["specific_power"]  # kW/m³
        # Typical values for biogas: 3-8 W/m³
        assert 0.002 <= specific_power <= 0.020, "Specific power should be in typical range (3-8 W/m³)"


class TestMixerEfficiency:
    """Test suite for mixer efficiency and power consumption."""

    def test_power_scales_with_speed_cubed(self) -> None:
        """Test that power scales approximately with N³."""
        mixer = Mixer("mixer_1", tank_volume=2000.0)
        mixer.initialize()

        # Test at 50% speed
        result1 = mixer.step(t=0.0, dt=1.0 / 24.0, inputs={"speed_setpoint": 0.5})
        P1 = result1["P_consumed"]

        mixer.initialize()  # Reset

        # Test at 100% speed
        result2 = mixer.step(t=0.0, dt=1.0 / 24.0, inputs={"speed_setpoint": 1.0})
        P2 = result2["P_consumed"]

        # P ∝ N³, so P2/P1 should be close to (1.0/0.5)³ = 8
        if P1 > 0 and P2 > 0:
            ratio = P2 / P1
            expected_ratio = (1.0 / 0.5) ** 3
            # Allow significant tolerance due to efficiency variations
            assert 4.0 <= ratio <= 12.0, f"Power ratio {ratio:.2f} should be close to speed³ ratio {expected_ratio:.2f}"

    def test_intermittent_reduces_average_power(self) -> None:
        """Test that intermittent operation reduces average power."""
        continuous = Mixer("m1", intermittent=False)
        intermittent = Mixer("m2", intermittent=True, on_time_fraction=0.25)

        continuous.initialize()
        intermittent.initialize()

        result_cont = continuous.step(t=0.0, dt=1.0 / 24.0, inputs={})
        result_int = intermittent.step(t=0.0, dt=1.0 / 24.0, inputs={})

        # Intermittent average power should be lower
        assert result_int["P_average"] < result_cont["P_average"], "Intermittent operation should reduce average power"

    def test_part_load_efficiency(self) -> None:
        """Test that efficiency is reported at part load."""
        mixer = Mixer("mixer_1")
        mixer.initialize()

        # Run at part load
        result = mixer.step(t=0.0, dt=1.0 / 24.0, inputs={"speed_setpoint": 0.6})

        # Power should be reduced but not linearly (due to N³ relationship)
        assert result["P_consumed"] < mixer.power_installed, "Part load power should be less than installed"


class TestMixerFluidProperties:
    """Test suite for handling different fluid properties."""

    def test_mixer_with_different_viscosity(self) -> None:
        """Test mixer behavior with different fluid viscosity."""
        mixer = Mixer("mixer_1", tank_volume=2000.0)
        mixer.initialize()

        # Test with low viscosity
        result1 = mixer.step(t=0.0, dt=1.0 / 24.0, inputs={"fluid_viscosity": 0.01})  # Low viscosity

        mixer.initialize()

        # Test with high viscosity
        result2 = mixer.step(t=0.0, dt=1.0 / 24.0, inputs={"fluid_viscosity": 0.10})  # High viscosity

        # Reynolds number should decrease with increasing viscosity
        assert result2["reynolds_number"] < result1["reynolds_number"], "Higher viscosity should give lower Reynolds number"

    def test_mixer_with_temperature_update(self) -> None:
        """Test that mixer accepts temperature updates."""
        mixer = Mixer("mixer_1")
        mixer.initialize()

        # Test with different temperature (affects viscosity)
        result = mixer.step(t=0.0, dt=1.0 / 24.0, inputs={"temperature": 318.15})

        # Should complete without error
        assert "P_consumed" in result


class TestMixerSerialization:
    """Test suite for Mixer serialization methods."""

    def test_to_dict_returns_dict(self) -> None:
        """Test that to_dict method returns a dictionary."""
        mixer = Mixer("mixer_1", tank_volume=2000.0)
        mixer.initialize()

        config = mixer.to_dict()

        assert isinstance(config, dict), "to_dict should return a dictionary"

    def test_to_dict_contains_required_fields(self) -> None:
        """Test that to_dict includes all required fields."""
        mixer = Mixer("mixer_1", tank_volume=2000.0)
        mixer.initialize()

        config = mixer.to_dict()

        required_fields = [
            "component_id",
            "component_type",
            "mixer_type",
            "tank_volume",
            "mixing_intensity",
            "power_installed",
        ]
        for field in required_fields:
            assert field in config, f"to_dict should include '{field}'"

    def test_from_dict_recreates_mixer(self) -> None:
        """Test that from_dict can recreate a mixer from configuration."""
        original = Mixer(
            "mixer_1",
            mixer_type="paddle",
            tank_volume=2500.0,
            mixing_intensity="high",
            power_installed=20.0,
            name="Main Mixer",
        )
        original.initialize()

        config = original.to_dict()
        recreated = Mixer.from_dict(config)

        assert recreated.component_id == original.component_id
        assert recreated.mixer_type == original.mixer_type
        assert recreated.tank_volume == original.tank_volume
        assert recreated.mixing_intensity == original.mixing_intensity
        assert recreated.power_installed == original.power_installed
        assert recreated.name == original.name

    def test_roundtrip_preserves_configuration(self) -> None:
        """Test that serialization roundtrip preserves configuration."""
        original = Mixer(
            "mixer_1", mixer_type="propeller", tank_volume=1800.0, tank_diameter=14.0, intermittent=True, on_time_fraction=0.30
        )

        config = original.to_dict()
        recreated = Mixer.from_dict(config)

        assert recreated.tank_volume == original.tank_volume
        assert recreated.tank_diameter == original.tank_diameter
        assert recreated.intermittent == original.intermittent
        assert recreated.on_time_fraction == original.on_time_fraction


class TestMixerConnections:
    """Test suite for Mixer component connections."""

    def test_add_input_connection(self) -> None:
        """Test adding input connections to mixer."""
        mixer = Mixer("mixer_1")

        mixer.add_input("temperature_sensor")

        assert "temperature_sensor" in mixer.inputs, "Input should be added"

    def test_add_output_connection(self) -> None:
        """Test adding output connections from mixer."""
        mixer = Mixer("mixer_1")

        mixer.add_output("quality_monitor")

        assert "quality_monitor" in mixer.outputs, "Output should be added"

    def test_multiple_connections(self) -> None:
        """Test adding multiple connections."""
        mixer = Mixer("mixer_1")

        mixer.add_input("sensor1")
        mixer.add_input("sensor2")
        mixer.add_output("monitor")

        assert len(mixer.inputs) == 2
        assert len(mixer.outputs) == 1


class TestMixerProperties:
    """Test suite for Mixer properties and attributes."""

    def test_component_type_is_mixer(self) -> None:
        """Test that component type is correctly identified."""
        mixer = Mixer("mixer_1")

        assert mixer.component_type.value == "mixer", "Component type should be 'mixer'"

    def test_power_consumption_non_negative(self) -> None:
        """Test that power consumption is non-negative."""
        mixer = Mixer("mixer_1")
        mixer.initialize()

        result = mixer.step(t=0.0, dt=1.0 / 24.0, inputs={})

        assert result["P_consumed"] >= 0, "Power should be non-negative"

    def test_mixing_quality_bounds(self) -> None:
        """Test that mixing quality is bounded."""
        mixer = Mixer("mixer_1")
        mixer.initialize()

        result = mixer.step(t=0.0, dt=1.0 / 24.0, inputs={})

        assert 0.0 <= result["mixing_quality"] <= 1.0, "Mixing quality should be between 0 and 1"

    def test_reynolds_number_positive(self) -> None:
        """Test that Reynolds number is positive when running."""
        mixer = Mixer("mixer_1")
        mixer.initialize()

        result = mixer.step(t=0.0, dt=1.0 / 24.0, inputs={})

        assert result["reynolds_number"] > 0, "Reynolds number should be positive"

    def test_power_number_positive(self) -> None:
        """Test that power number is positive."""
        mixer = Mixer("mixer_1")
        mixer.initialize()

        result = mixer.step(t=0.0, dt=1.0 / 24.0, inputs={})

        assert result["power_number"] > 0, "Power number should be positive"

    def test_get_state_returns_dict(self) -> None:
        """Test that get_state returns the state dictionary."""
        mixer = Mixer("mixer_1")
        mixer.initialize()

        state = mixer.get_state()

        assert isinstance(state, dict), "get_state should return a dictionary"
        assert "is_running" in state
        assert "current_speed_fraction" in state

    def test_set_state_updates_properties(self) -> None:
        """Test that set_state updates component state."""
        mixer = Mixer("mixer_1")
        mixer.initialize()

        new_state = {"is_running": False, "current_speed_fraction": 0.0, "operating_hours": 1000.0}
        mixer.set_state(new_state)

        assert mixer.state.get("operating_hours") == 1000.0

    def test_mixing_time_reasonable(self) -> None:
        """Test that mixing time is in reasonable range."""
        mixer = Mixer("mixer_1", tank_volume=2000.0, mixing_intensity="medium")
        mixer.initialize()

        # For biogas digesters, mixing time should typically be < 30 minutes
        assert mixer.mixing_time < 60.0, "Mixing time should be reasonable (< 60 minutes)"
        assert mixer.mixing_time > 0.5, "Mixing time should be at least 0.5 minutes"


class TestMixerTankGeometry:
    """Test suite for mixer tank geometry calculations."""

    def test_tank_diameter_estimation(self) -> None:
        """Test tank diameter estimation from volume."""
        mixer = Mixer("mixer_1", tank_volume=2000.0)

        # V = π/4 * D² * H, with H/D = 1.5
        # D ≈ (V/1.178)^(1/3)
        expected_diameter = (2000.0 / 1.178) ** (1 / 3)

        assert abs(mixer.tank_diameter - expected_diameter) < 1.0, "Tank diameter should be estimated correctly"

    def test_tank_height_estimation(self) -> None:
        """Test tank height estimation from volume and diameter."""
        volume = 2000.0
        diameter = 15.0
        mixer = Mixer("mixer_1", tank_volume=volume, tank_diameter=diameter)

        # V = π/4 * D² * H
        expected_height = volume / (np.pi / 4 * diameter**2)

        assert abs(mixer.tank_height - expected_height) < 1.0, "Tank height should be estimated correctly"

    def test_impeller_to_tank_diameter_ratio(self) -> None:
        """Test that impeller/tank diameter ratio is reasonable."""
        mixer = Mixer("mixer_1", mixer_type="propeller", tank_diameter=15.0)

        ratio = mixer.impeller_diameter / mixer.tank_diameter

        # For propeller mixers, D/DT ≈ 0.33
        assert 0.25 <= ratio <= 0.60, "Impeller to tank diameter ratio should be reasonable"

    def test_larger_tank_larger_impeller(self) -> None:
        """Test that larger tanks get larger impellers."""
        small = Mixer("m1", tank_volume=1000.0)
        large = Mixer("m2", tank_volume=3000.0)

        assert large.impeller_diameter > small.impeller_diameter, "Larger tank should have larger impeller"


class TestMixerScaling:
    """Test suite for mixer scaling with tank size."""

    def test_power_scales_with_tank_size(self) -> None:
        """Test that power requirement scales with tank size."""
        small = Mixer("m1", tank_volume=1000.0, mixing_intensity="medium")
        large = Mixer("m2", tank_volume=3000.0, mixing_intensity="medium")

        assert large.power_installed > small.power_installed, "Larger tank should require more power"

    def test_specific_power_consistent(self) -> None:
        """Test that specific power (W/m³) is similar for different sizes."""
        small = Mixer("m1", tank_volume=1000.0, mixing_intensity="medium")
        large = Mixer("m2", tank_volume=3000.0, mixing_intensity="medium")

        small.initialize()
        large.initialize()

        result_small = small.step(t=0.0, dt=1.0 / 24.0, inputs={})
        result_large = large.step(t=0.0, dt=1.0 / 24.0, inputs={})

        sp_small = result_small["specific_power"]
        sp_large = result_large["specific_power"]

        # Specific power should be in similar range (within factor of 2)
        ratio = max(sp_small, sp_large) / min(sp_small, sp_large)
        assert ratio < 2.5, "Specific power should be consistent across scales"

    def test_operating_speed_scales_with_size(self) -> None:
        """Test that operating speed decreases with tank size."""
        small = Mixer("m1", tank_volume=500.0)
        large = Mixer("m2", tank_volume=4000.0)

        # Larger tanks typically run slower
        assert small.operating_speed >= large.operating_speed, "Smaller tanks should run at higher speed"


class TestMixerEdgeCases:
    """Test suite for mixer edge cases and error handling."""

    def test_mixer_with_very_small_tank(self) -> None:
        """Test mixer with very small tank volume."""
        mixer = Mixer("mixer_1", tank_volume=10.0)
        mixer.initialize()

        result = mixer.step(t=0.0, dt=1.0 / 24.0, inputs={})

        assert result["P_consumed"] > 0, "Should handle small tanks"
        assert result["P_consumed"] < 5.0, "Power should be reasonable for small tank"

    def test_mixer_with_very_large_tank(self) -> None:
        """Test mixer with very large tank volume."""
        mixer = Mixer("mixer_1", tank_volume=10000.0)
        mixer.initialize()

        result = mixer.step(t=0.0, dt=1.0 / 24.0, inputs={})

        assert result["P_consumed"] > 0, "Should handle large tanks"
        assert result["P_consumed"] < 200.0, "Power should be reasonable"

    def test_mixer_with_zero_speed_setpoint(self) -> None:
        """Test mixer with zero speed setpoint."""
        mixer = Mixer("mixer_1")
        mixer.initialize()

        result = mixer.step(t=0.0, dt=1.0 / 24.0, inputs={"speed_setpoint": 0.0})

        assert result["P_consumed"] == 0.0, "Zero speed should give zero power"
        assert result["mixing_quality"] == 0.0, "Zero speed should give zero quality"

    def test_mixer_with_negative_speed_setpoint(self) -> None:
        """Test that negative speed setpoint is handled."""
        mixer = Mixer("mixer_1")
        mixer.initialize()

        # Should not crash with negative setpoint
        result = mixer.step(t=0.0, dt=1.0 / 24.0, inputs={"speed_setpoint": -0.5})

        # Should clamp to zero or positive
        assert result["P_consumed"] >= 0.0

    def test_mixer_with_excessive_speed(self) -> None:
        """Test mixer with speed > 100%."""
        mixer = Mixer("mixer_1")
        mixer.initialize()

        result = mixer.step(t=0.0, dt=1.0 / 24.0, inputs={"speed_setpoint": 1.5})

        # Should handle overload condition
        assert result["P_consumed"] >= 0.0
        assert mixer.state["current_speed_fraction"] > 1.0 or result["P_consumed"] <= mixer.power_installed * 1.2

    def test_mixer_multiple_consecutive_steps(self) -> None:
        """Test mixer over multiple consecutive time steps."""
        mixer = Mixer("mixer_1", intermittent=False)
        mixer.initialize()

        # Run for 24 hours
        for hour in range(24):
            result = mixer.step(t=hour / 24.0, dt=1.0 / 24.0, inputs={})
            assert result["P_consumed"] >= 0.0

        # Operating hours should accumulate
        assert mixer.operating_hours == 24.0

    def test_mixer_multiple_consecutive_steps_intermittent(self) -> None:
        """Test mixer over multiple consecutive time steps in intermittent operation (default: 25%)."""
        mixer = Mixer("mixer_1")
        mixer.initialize()

        # Run for 24 hours
        for hour in range(24):
            result = mixer.step(t=hour / 24.0, dt=1.0 / 24.0, inputs={})
            assert result["P_consumed"] >= 0.0

        # Operating hours should accumulate: 25% of 24 h = 6 h
        assert mixer.operating_hours == 6.0

    def test_mixer_start_stop_cycle(self) -> None:
        """Test mixer start/stop cycling."""
        mixer = Mixer("mixer_1")
        mixer.initialize()

        # Start mixer
        result1 = mixer.step(t=0.0, dt=1.0 / 24.0, inputs={"enable_mixing": True})
        assert result1["is_running"] is True

        # Stop mixer
        result2 = mixer.step(t=1.0 / 24.0, dt=1.0 / 24.0, inputs={"enable_mixing": False})
        assert result2["is_running"] is False
        assert result2["P_consumed"] == 0.0

        # Restart mixer
        result3 = mixer.step(t=2.0 / 24.0, dt=1.0 / 24.0, inputs={"enable_mixing": True})
        assert result3["is_running"] is True


class TestMixerPhysics:
    """Test suite for mixer physics and calculations."""

    def test_power_number_varies_with_reynolds(self) -> None:
        """Test that power number varies with Reynolds number."""
        mixer = Mixer("mixer_1", mixer_type="propeller")

        # Calculate at different viscosities (affects Reynolds number)
        mixer.initialize()
        mixer.fluid_viscosity = 0.001  # Low viscosity, high Re
        mixer._calculate_mixing_parameters()
        Np_high_Re = mixer.power_number
        Re_high = mixer.reynolds_number

        mixer.fluid_viscosity = 0.100  # High viscosity, low Re
        mixer._calculate_mixing_parameters()
        Np_low_Re = mixer.power_number
        Re_low = mixer.reynolds_number

        # At low Reynolds numbers, Np should be higher
        assert Re_low < Re_high, "Higher viscosity should give lower Re"
        if Re_low < 100:  # In laminar regime
            assert Np_low_Re > Np_high_Re, "Power number should increase at low Reynolds numbers"

    def test_mixing_time_decreases_with_speed(self) -> None:
        """Test that mixing time decreases with higher speed."""
        mixer = Mixer("mixer_1", operating_speed=40.0)
        mixer.initialize()
        mixing_time_slow = mixer.mixing_time

        mixer2 = Mixer(
            "mixer_2",
            operating_speed=80.0,
            tank_volume=mixer.tank_volume,
            tank_diameter=mixer.tank_diameter,
            tank_height=mixer.tank_height,
            impeller_diameter=mixer.impeller_diameter,
        )
        mixer2.initialize()
        mixing_time_fast = mixer2.mixing_time

        assert mixing_time_fast < mixing_time_slow, "Higher speed should give shorter mixing time"

    def test_shear_rate_proportional_to_speed(self) -> None:
        """Test that shear rate is proportional to rotational speed."""
        mixer = Mixer("mixer_1")
        mixer.initialize()

        # Test at different speeds
        result1 = mixer.step(t=0.0, dt=1.0 / 24.0, inputs={"speed_setpoint": 0.5})
        shear1 = result1["shear_rate"]

        mixer.initialize()

        result2 = mixer.step(t=0.0, dt=1.0 / 24.0, inputs={"speed_setpoint": 1.0})
        shear2 = result2["shear_rate"]

        # Shear rate should roughly double
        if shear1 > 0 and shear2 > 0:
            ratio = shear2 / shear1
            assert 1.8 <= ratio <= 2.2, "Shear rate should be approximately proportional to speed"

    def test_tip_speed_proportional_to_rotational_speed(self) -> None:
        """Test that tip speed is proportional to rotational speed."""
        mixer = Mixer("mixer_1")
        mixer.initialize()

        result1 = mixer.step(t=0.0, dt=1.0 / 24.0, inputs={"speed_setpoint": 0.5})
        tip_speed1 = result1["tip_speed"]

        mixer.initialize()

        result2 = mixer.step(t=0.0, dt=1.0 / 24.0, inputs={"speed_setpoint": 1.0})
        tip_speed2 = result2["tip_speed"]

        # Tip speed = π * N * D, should be proportional to N
        if tip_speed1 > 0:
            ratio = tip_speed2 / tip_speed1
            assert 1.8 <= ratio <= 2.2, "Tip speed should be proportional to rotational speed"


class TestMixerIntegration:
    """Test suite for mixer integration scenarios."""

    def test_mixer_daily_operation_cycle(self) -> None:
        """Test mixer over a complete daily cycle."""
        mixer = Mixer("mixer_1", intermittent=True, on_time_fraction=0.25)
        mixer.initialize()

        total_energy = 0.0

        # Simulate 24 hours
        for hour in range(24):
            result = mixer.step(t=hour / 24.0, dt=1.0 / 24.0, inputs={})
            total_energy += result["P_consumed"] * 1.0  # kWh

        # Check that total energy is reasonable
        assert total_energy > 0, "Should consume energy over the day"
        # Average power should be close to 25% of continuous operation
        avg_power = total_energy / 24.0
        expected_avg = mixer.power_installed * 0.25
        # avg_power should be 0.25 of power_installed
        assert 0.9 * expected_avg <= avg_power <= 1.1 * expected_avg, "Average power should reflect intermittent operation"

    def test_mixer_with_varying_speed_profile(self) -> None:
        """Test mixer with time-varying speed profile."""
        mixer = Mixer("mixer_1", intermittent=False)
        mixer.initialize()

        # Simulate varying speed over time
        speeds = [0.5, 0.7, 1.0, 0.8, 0.6, 0.4]
        results = []

        for i, speed in enumerate(speeds):
            result = mixer.step(t=i / 24.0, dt=1.0 / 24.0, inputs={"speed_setpoint": speed})
            results.append(result)

        # Verify power varies with speed
        powers = [r["P_consumed"] for r in results]
        assert max(powers) > min(powers), "Power should vary with speed"

    def test_mixer_energy_accounting(self) -> None:
        """Test that energy accounting is accurate."""
        mixer = Mixer("mixer_1")
        mixer.initialize()

        initial_energy = mixer.energy_consumed

        # Run for known time
        hours = 10
        dt = 1.0 / 24.0  # 1 hour steps

        total_power = 0.0
        for i in range(hours):
            result = mixer.step(t=i * dt, dt=dt, inputs={})
            total_power += result["P_consumed"]

        # Energy consumed should match integrated power
        expected_energy = total_power  # kWh (power in kW, time in hours)
        actual_energy = mixer.energy_consumed - initial_energy

        assert (
            abs(actual_energy - expected_energy) < 0.1
        ), f"Energy accounting mismatch: {actual_energy:.2f} vs {expected_energy:.2f}"


class TestMixerDocumentation:
    """Test suite to verify mixer documentation and examples work."""

    def test_basic_example_from_docstring(self) -> None:
        """Test basic example from component docstring."""
        # Example from Mixer class docstring
        mixer = Mixer("mix1", mixer_type="propeller", tank_volume=2000, mixing_intensity="medium", power_installed=15.0)
        mixer.initialize()
        result = mixer.step(t=0, dt=1 / 24, inputs={})

        assert "P_consumed" in result
        assert result["P_consumed"] > 0

    def test_module_example_from_docstring(self) -> None:
        """Test example from module docstring."""
        # Example from module-level docstring
        mixer = Mixer(component_id="pump1", mixer_type="propeller", tank_volume=2000.0, mixing_intensity="medium")
        mixer.initialize()
        result = mixer.step(t=0, dt=1 / 24, inputs={})

        assert isinstance(result, dict)
        assert "P_consumed" in result
