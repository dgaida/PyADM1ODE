# -*- coding: utf-8 -*-
"""
Unit tests for the CHP (Combined Heat and Power) component.

This module tests the CHP class which converts biogas to electricity
and heat in a biogas plant configuration.
"""

from pyadm1.components.energy.chp import CHP


class TestCHPInitialization:
    """Test suite for CHP component initialization."""

    def test_chp_initialization_sets_component_id(self) -> None:
        """Test that CHP initialization sets the component_id."""
        chp = CHP("chp_1")

        assert chp.component_id == "chp_1", "Component ID should be set correctly"

    def test_chp_initialization_sets_electrical_power(self) -> None:
        """Test that CHP initialization sets electrical power."""
        P_el_nom = 500.0
        chp = CHP("chp_1", P_el_nom=P_el_nom)

        assert chp.P_el_nom == P_el_nom, f"P_el_nom should be {P_el_nom}"

    def test_chp_initialization_sets_efficiencies(self) -> None:
        """Test that CHP initialization sets efficiency values."""
        eta_el = 0.40
        eta_th = 0.45
        chp = CHP("chp_1", eta_el=eta_el, eta_th=eta_th)

        assert chp.eta_el == eta_el, f"eta_el should be {eta_el}"
        assert chp.eta_th == eta_th, f"eta_th should be {eta_th}"

    def test_chp_initialization_with_custom_name(self) -> None:
        """Test that CHP accepts custom name parameter."""
        custom_name = "Main CHP Unit"
        chp = CHP("chp_1", name=custom_name)

        assert chp.name == custom_name, f"Name should be '{custom_name}'"

    def test_chp_initialization_default_values(self) -> None:
        """Test that CHP has reasonable default values."""
        chp = CHP("chp_1")

        assert chp.P_el_nom == 500.0, "Default P_el_nom should be 500.0 kW"
        assert chp.eta_el == 0.40, "Default eta_el should be 0.40"
        assert chp.eta_th == 0.45, "Default eta_th should be 0.45"

    def test_chp_initialization_creates_state_dict(self) -> None:
        """Test that initialization creates state dictionary."""
        chp = CHP("chp_1")

        assert hasattr(chp, "state"), "CHP should have state attribute"
        assert isinstance(chp.state, dict), "state should be a dictionary"


class TestCHPInitialize:
    """Test suite for CHP initialize method."""

    def test_initialize_sets_load_factor_to_zero(self) -> None:
        """Test that initialize sets load_factor to zero."""
        chp = CHP("chp_1")
        chp.initialize()

        assert chp.load_factor == 0.0, "Initial load_factor should be 0.0"

    def test_initialize_creates_state_dict(self) -> None:
        """Test that initialize creates proper state dictionary."""
        chp = CHP("chp_1")
        chp.initialize()

        required_keys = ["load_factor", "P_el", "P_th", "Q_gas_consumed", "operating_hours"]
        for key in required_keys:
            assert key in chp.state, f"State should have '{key}' key"

    def test_initialize_with_custom_load_factor(self) -> None:
        """Test initialize with custom load factor."""
        chp = CHP("chp_1")
        chp.initialize({"load_factor": 0.75})

        assert chp.load_factor == 0.75, "Load factor should be set to 0.75"

    def test_initialize_creates_outputs_data(self) -> None:
        """Test that initialize creates outputs_data dictionary."""
        chp = CHP("chp_1")
        chp.initialize()

        assert hasattr(chp, "outputs_data"), "CHP should have outputs_data"
        assert isinstance(chp.outputs_data, dict), "outputs_data should be a dictionary"


class TestCHPStep:
    """Test suite for CHP step method (simulation)."""

    def test_step_returns_dict(self) -> None:
        """Test that step method returns a dictionary."""
        chp = CHP("chp_1", P_el_nom=500.0)
        chp.initialize()

        inputs = {"Q_ch4": 1000.0}
        result = chp.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert isinstance(result, dict), "step should return a dictionary"

    def test_step_with_zero_gas_input(self) -> None:
        """Test step with no biogas input."""
        chp = CHP("chp_1", P_el_nom=500.0)
        chp.initialize()

        inputs = {"Q_ch4": 0.0}
        result = chp.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert result["P_el"] == 0.0, "P_el should be 0 with no gas input"
        assert result["P_th"] == 0.0, "P_th should be 0 with no gas input"

    def test_step_output_contains_power_values(self) -> None:
        """Test that step output contains power information."""
        chp = CHP("chp_1", P_el_nom=500.0, eta_el=0.40, eta_th=0.45)
        chp.initialize()

        inputs = {"Q_ch4": 1000.0}
        result = chp.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert "P_el" in result, "Result should contain P_el"
        assert "P_th" in result, "Result should contain P_th"
        assert "Q_gas_consumed" in result, "Result should contain Q_gas_consumed"

    def test_step_calculates_power_output(self) -> None:
        """Test that step correctly calculates power output."""
        chp = CHP("chp_1", P_el_nom=500.0, eta_el=0.40, eta_th=0.45)
        chp.initialize()

        # 10 kWh/m³ CH4, so 1000 m³/d should give energy
        inputs = {"Q_ch4": 1000.0}
        result = chp.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        # Should produce some electrical power
        assert result["P_el"] > 0, "CHP should produce electrical power with CH4 input"
        assert result["P_th"] > 0, "CHP should produce thermal power with CH4 input"

    def test_step_respects_load_setpoint(self) -> None:
        """Test that step respects load_setpoint input."""
        chp = CHP("chp_1", P_el_nom=500.0)
        chp.initialize()

        # Provide abundant gas but limit load
        inputs = {"Q_ch4": 5000.0, "load_setpoint": 0.5}
        result = chp.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        # Electrical power should be limited to 50% of nominal
        assert result["P_el"] <= chp.P_el_nom * 0.55, "Load should be limited by setpoint"

    def test_step_updates_operating_hours(self) -> None:
        """Test that step updates operating hours."""
        chp = CHP("chp_1")
        chp.initialize()

        initial_hours = chp.state["operating_hours"]
        dt = 1.0 / 24.0  # 1 hour

        inputs = {"Q_ch4": 1000.0}
        chp.step(t=0.0, dt=dt, inputs=inputs)

        # Operating hours should increase by 1
        assert chp.state["operating_hours"] > initial_hours, "Operating hours should increase"

    def test_step_with_gas_storage_input(self) -> None:
        """Test step with gas storage supply input."""
        chp = CHP("chp_1", P_el_nom=500.0)
        chp.initialize()

        inputs = {"Q_gas_supplied_m3_per_day": 1000.0, "Q_gas_out_m3_per_day": 500.0}
        result = chp.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert isinstance(result, dict), "Should handle gas storage inputs"


class TestCHPEfficiency:
    """Test suite for CHP efficiency calculations."""

    def test_electrical_efficiency_bounds(self) -> None:
        """Test that electrical efficiency is in reasonable bounds."""
        # Valid CHP efficiency
        chp = CHP("chp_1", eta_el=0.40)
        chp.initialize()

        assert 0.30 <= chp.eta_el <= 0.50, "Electrical efficiency should be realistic"

    def test_total_efficiency_calculation(self) -> None:
        """Test that total efficiency (electrical + thermal) is reasonable."""
        chp = CHP("chp_1", eta_el=0.40, eta_th=0.45)

        total_efficiency = chp.eta_el + chp.eta_th
        assert 0.80 <= total_efficiency <= 0.95, "Total efficiency should be between 80-95%"

    def test_thermal_to_electrical_ratio(self) -> None:
        """Test thermal power output is proportional to electrical output."""
        chp = CHP("chp_1", P_el_nom=500.0, eta_el=0.40, eta_th=0.45)
        chp.initialize()

        inputs = {"Q_ch4": 1000.0}
        result = chp.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        # Thermal power should be roughly proportional to electrical power
        if result["P_el"] > 0:
            ratio = result["P_th"] / result["P_el"]
            expected_ratio = chp.eta_th / chp.eta_el
            # Allow some tolerance due to rounding
            assert abs(ratio - expected_ratio) < 0.1, "P_th/P_el ratio should match efficiency ratio"


class TestCHPSerialization:
    """Test suite for CHP serialization methods."""

    def test_to_dict_returns_dict(self) -> None:
        """Test that to_dict method returns a dictionary."""
        chp = CHP("chp_1", P_el_nom=500.0, eta_el=0.40, eta_th=0.45)
        chp.initialize()

        config = chp.to_dict()

        assert isinstance(config, dict), "to_dict should return a dictionary"

    def test_to_dict_contains_required_fields(self) -> None:
        """Test that to_dict includes all required fields."""
        chp = CHP("chp_1", P_el_nom=500.0)
        chp.initialize()

        config = chp.to_dict()

        required_fields = ["component_id", "component_type", "P_el_nom", "eta_el", "eta_th"]
        for field in required_fields:
            assert field in config, f"to_dict should include '{field}'"

    def test_from_dict_recreates_chp(self) -> None:
        """Test that from_dict can recreate a CHP from configuration."""
        original = CHP("chp_1", P_el_nom=500.0, eta_el=0.40, eta_th=0.45, name="Main CHP")
        original.initialize()

        config = original.to_dict()
        recreated = CHP.from_dict(config)

        assert recreated.component_id == original.component_id, "Component ID should match"
        assert recreated.P_el_nom == original.P_el_nom, "P_el_nom should match"
        assert recreated.eta_el == original.eta_el, "eta_el should match"
        assert recreated.eta_th == original.eta_th, "eta_th should match"
        assert recreated.name == original.name, "Name should match"

    def test_roundtrip_preserves_efficiency(self) -> None:
        """Test that serialization preserves efficiency values."""
        original = CHP("chp_1", P_el_nom=750.0, eta_el=0.42, eta_th=0.48)

        config = original.to_dict()
        recreated = CHP.from_dict(config)

        assert recreated.eta_el == original.eta_el, "eta_el should be preserved"
        assert recreated.eta_th == original.eta_th, "eta_th should be preserved"


class TestCHPConnections:
    """Test suite for CHP component connections."""

    def test_add_input_connection(self) -> None:
        """Test adding input connections to CHP."""
        chp = CHP("chp_1")

        chp.add_input("digester_1")

        assert "digester_1" in chp.inputs, "Input should be added"

    def test_add_multiple_inputs(self) -> None:
        """Test adding multiple input connections."""
        chp = CHP("chp_1")

        chp.add_input("digester_1")
        chp.add_input("digester_2")

        assert len(chp.inputs) == 2, "Should have two inputs"
        assert "digester_1" in chp.inputs, "digester_1 should be in inputs"
        assert "digester_2" in chp.inputs, "digester_2 should be in inputs"

    def test_add_output_connection(self) -> None:
        """Test adding output connections from CHP."""
        chp = CHP("chp_1")

        chp.add_output("heating_1")

        assert "heating_1" in chp.outputs, "Output should be added"

    def test_no_duplicate_connections(self) -> None:
        """Test that duplicate connections are not added."""
        chp = CHP("chp_1")

        chp.add_input("source")
        chp.add_input("source")

        assert chp.inputs.count("source") == 1, "Duplicate inputs should not be added"


class TestCHPProperties:
    """Test suite for CHP properties and attributes."""

    def test_component_type_is_chp(self) -> None:
        """Test that component type is correctly identified as CHP."""
        chp = CHP("chp_1")

        assert chp.component_type.value == "chp", "Component type should be 'chp'"

    def test_load_factor_range(self) -> None:
        """Test that load factor stays within valid range."""
        chp = CHP("chp_1")
        chp.initialize()

        # After simulation with reasonable input
        inputs = {"Q_ch4": 500.0}
        chp.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert 0.0 <= chp.load_factor <= 1.0, "Load factor should be between 0 and 1"

    def test_gas_consumption_is_positive(self) -> None:
        """Test that gas consumption is non-negative."""
        chp = CHP("chp_1", P_el_nom=500.0)
        chp.initialize()

        inputs = {"Q_ch4": 1000.0}
        result = chp.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert result["Q_gas_consumed"] >= 0, "Gas consumption should be non-negative"

    def test_get_state_returns_dict(self) -> None:
        """Test that get_state returns the state dictionary."""
        chp = CHP("chp_1")
        chp.initialize()

        state = chp.get_state()

        assert isinstance(state, dict), "get_state should return a dictionary"
        assert "P_el" in state, "State should contain P_el"

    def test_set_state_updates_properties(self) -> None:
        """Test that set_state updates component state."""
        chp = CHP("chp_1")
        chp.initialize()

        new_state = {"load_factor": 0.8, "P_el": 400.0}
        chp.set_state(new_state)

        assert chp.state.get("load_factor") == 0.8, "State should be updated"
