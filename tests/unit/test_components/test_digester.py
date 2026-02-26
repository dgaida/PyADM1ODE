# -*- coding: utf-8 -*-
"""
Unit tests for the Digester component.

This module tests the Digester class which wraps the ADM1 model
and provides a component-based interface for biogas plant simulations.
"""

import pytest
from unittest.mock import Mock, patch

import pyadm1.components.biological.digester as digester_module
from pyadm1.components.biological.digester import Digester
from pyadm1.substrates.feedstock import Feedstock
from pyadm1.components.energy.gas_storage import GasStorage


class TestDigesterInitialization:
    """Test suite for Digester component initialization."""

    @pytest.fixture
    def mock_feedstock(self) -> Mock:
        """
        Create a mock Feedstock object.

        Returns:
            Mock Feedstock object with necessary attributes.
        """
        feedstock = Mock(spec=Feedstock)
        feedstock.mySubstrates = Mock()
        feedstock.get_influent_dataframe = Mock(return_value=Mock())
        return feedstock

    def test_digester_initialization_sets_component_id(self, mock_feedstock: Mock) -> None:
        """
        Test that digester initialization sets the component_id.

        Args:
            mock_feedstock: Mock Feedstock fixture.
        """
        digester = Digester("dig_1", mock_feedstock)

        assert digester.component_id == "dig_1", "Component ID should be set correctly"

    def test_digester_initialization_sets_volume(self, mock_feedstock: Mock) -> None:
        """
        Test that digester initialization sets liquid and gas volumes.

        Args:
            mock_feedstock: Mock Feedstock fixture.
        """
        V_liq = 2000.0
        V_gas = 300.0

        digester = Digester("dig_1", mock_feedstock, V_liq=V_liq, V_gas=V_gas)

        assert digester.V_liq == V_liq, f"V_liq should be {V_liq}"
        assert digester.V_gas == V_gas, f"V_gas should be {V_gas}"

    def test_digester_initialization_sets_temperature(self, mock_feedstock: Mock) -> None:
        """
        Test that digester initialization sets operating temperature.

        Args:
            mock_feedstock: Mock Feedstock fixture.
        """
        T_ad = 313.15  # 40°C

        digester = Digester("dig_1", mock_feedstock, T_ad=T_ad)

        assert digester.T_ad == T_ad, f"T_ad should be {T_ad}"

    def test_digester_creates_gas_storage(self, mock_feedstock: Mock) -> None:
        """
        Test that digester creates an associated gas storage component.

        Args:
            mock_feedstock: Mock Feedstock fixture.
        """
        digester = Digester("dig_1", mock_feedstock)

        assert hasattr(digester, "gas_storage"), "Digester should have gas_storage attribute"
        assert isinstance(digester.gas_storage, GasStorage), "gas_storage should be a GasStorage instance"
        assert digester.gas_storage.component_id == "dig_1_storage", "Storage ID should follow naming convention"

    def test_digester_initialization_creates_state_dict(self, mock_feedstock: Mock) -> None:
        """
        Test that initialization creates state dictionary.

        Args:
            mock_feedstock: Mock Feedstock fixture.
        """
        digester = Digester("dig_1", mock_feedstock)

        assert hasattr(digester, "state"), "Digester should have state attribute"
        assert isinstance(digester.state, dict), "state should be a dictionary"

    def test_digester_initialization_with_custom_name(self, mock_feedstock: Mock) -> None:
        """
        Test that digester accepts custom name parameter.

        Args:
            mock_feedstock: Mock Feedstock fixture.
        """
        custom_name = "Main Digester"
        digester = Digester("dig_1", mock_feedstock, name=custom_name)

        assert digester.name == custom_name, f"Name should be '{custom_name}'"


class TestDigesterInitialize:
    """Test suite for Digester initialize method."""

    @pytest.fixture
    def mock_feedstock(self) -> Mock:
        """
        Create a mock Feedstock object.

        Returns:
            Mock Feedstock object.
        """
        feedstock = Mock(spec=Feedstock)
        feedstock.mySubstrates = Mock()
        return feedstock

    def test_initialize_with_default_state(self, mock_feedstock: Mock) -> None:
        """
        Test initialization with default state.

        Args:
            mock_feedstock: Mock Feedstock fixture.
        """
        digester = Digester("dig_1", mock_feedstock)
        digester.initialize()

        assert digester._initialized, "Digester should be marked as initialized"
        assert len(digester.adm1_state) == 37, "ADM1 state should have 37 elements"

    def test_initialize_with_custom_state(self, mock_feedstock: Mock) -> None:
        """
        Test initialization with custom initial state.

        Args:
            mock_feedstock: Mock Feedstock fixture.
        """
        custom_state = [0.02] * 37
        custom_Q = [20, 15, 0, 0, 0, 0, 0, 0, 0, 0]

        digester = Digester("dig_1", mock_feedstock)
        digester.initialize({"adm1_state": custom_state, "Q_substrates": custom_Q})

        assert digester.adm1_state == custom_state, "Custom state should be set"
        assert digester.Q_substrates == custom_Q, "Custom Q should be set"

    def test_initialize_creates_state_dict(self, mock_feedstock: Mock) -> None:
        """
        Test that initialize creates proper state dictionary.

        Args:
            mock_feedstock: Mock Feedstock fixture.
        """
        digester = Digester("dig_1", mock_feedstock)
        digester.initialize()

        required_keys = ["adm1_state", "Q_substrates", "Q_gas", "Q_ch4", "Q_co2", "pH", "VFA", "TAC"]
        for key in required_keys:
            assert key in digester.state, f"State should have '{key}' key"

    def test_initialize_initializes_gas_storage(self, mock_feedstock: Mock) -> None:
        """
        Test that initialize also initializes the gas storage.

        Args:
            mock_feedstock: Mock Feedstock fixture.
        """
        digester = Digester("dig_1", mock_feedstock)
        digester.initialize()

        assert digester.gas_storage._initialized, "Gas storage should be initialized"


class TestDigesterStep:
    """Test suite for Digester step method (simulation)."""

    @pytest.fixture
    def mock_feedstock(self) -> Mock:
        """
        Create a mock Feedstock object.

        Returns:
            Mock Feedstock object.
        """
        feedstock = Mock(spec=Feedstock)
        feedstock.mySubstrates = Mock()
        return feedstock

    @pytest.fixture
    def initialized_digester(self, mock_feedstock: Mock) -> Digester:
        """
        Create an initialized digester for testing.

        Args:
            mock_feedstock: Mock Feedstock fixture.

        Returns:
            Initialized Digester instance.
        """
        digester = Digester("dig_1", mock_feedstock)
        digester.initialize()
        return digester

    def test_step_returns_dict(self, initialized_digester: Digester) -> None:
        """
        Test that step method returns a dictionary.

        Args:
            initialized_digester: Initialized digester fixture.
        """
        inputs = {"Q_substrates": [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]}

        with patch.object(initialized_digester, "adm1") as mock_adm1:
            mock_adm1.create_influent = Mock()
            mock_adm1.calc_gas = Mock(return_value=(1500, 900, 600, 0.95))

            with patch.object(initialized_digester.simulator, "simulate_AD_plant") as mock_sim:
                mock_sim.return_value = [0.02] * 37

                result = initialized_digester.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert isinstance(result, dict), "step should return a dictionary"

    def test_step_updates_state(self, initialized_digester: Digester) -> None:
        """
        Test that step updates the internal state.

        Args:
            initialized_digester: Initialized digester fixture.
        """
        inputs = {"Q_substrates": [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]}
        initial_state = initialized_digester.adm1_state.copy()

        with patch.object(initialized_digester, "adm1") as mock_adm1:
            mock_adm1.create_influent = Mock()
            mock_adm1.calc_gas = Mock(return_value=(1500, 900, 600, 0.95))

            with patch.object(initialized_digester.simulator, "simulate_AD_plant") as mock_sim:
                new_state = [0.02] * 37
                mock_sim.return_value = new_state

                initialized_digester.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        # State should be updated
        assert initialized_digester.adm1_state != initial_state, "State should be updated"

    def test_step_outputs_data_contains_gas_flows(self, initialized_digester: Digester) -> None:
        """
        Test that step output contains gas flow information.

        Args:
            initialized_digester: Initialized digester fixture.
        """
        inputs = {"Q_substrates": [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]}

        with patch.object(initialized_digester, "adm1") as mock_adm1:
            mock_adm1.create_influent = Mock()
            q_gas_expected = 1500.0
            q_ch4_expected = 900.0
            mock_adm1.calc_gas = Mock(return_value=(q_gas_expected, q_ch4_expected, 600, 0.95))

            with patch.object(initialized_digester.simulator, "simulate_AD_plant") as mock_sim:
                mock_sim.return_value = [0.02] * 37

                result = initialized_digester.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert "Q_gas" in result, "Result should contain Q_gas"
        assert "Q_ch4" in result, "Result should contain Q_ch4"
        assert result["Q_gas"] == q_gas_expected, "Q_gas should match calculation"
        assert result["Q_ch4"] == q_ch4_expected, "Q_ch4 should match calculation"

    def test_step_with_no_substrates(self, initialized_digester: Digester) -> None:
        """
        Test step with zero substrate feed.

        Args:
            initialized_digester: Initialized digester fixture.
        """
        inputs = {"Q_substrates": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}

        with patch.object(initialized_digester, "adm1") as mock_adm1:
            mock_adm1.create_influent = Mock()
            mock_adm1.calc_gas = Mock(return_value=(0, 0, 0, 1.0))

            with patch.object(initialized_digester.simulator, "simulate_AD_plant") as mock_sim:
                mock_sim.return_value = initialized_digester.adm1_state

                result = initialized_digester.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert "Q_gas" in result, "Should still return output structure"


class TestDigesterSerialization:
    """Test suite for Digester serialization methods."""

    @pytest.fixture
    def mock_feedstock(self) -> Mock:
        """
        Create a mock Feedstock object.

        Returns:
            Mock Feedstock object.
        """
        feedstock = Mock(spec=Feedstock)
        feedstock.mySubstrates = Mock()
        return feedstock

    def test_to_dict_returns_dict(self, mock_feedstock: Mock) -> None:
        """
        Test that to_dict method returns a dictionary.

        Args:
            mock_feedstock: Mock Feedstock fixture.
        """
        digester = Digester("dig_1", mock_feedstock, V_liq=2000, V_gas=300)
        digester.initialize()

        config = digester.to_dict()

        assert isinstance(config, dict), "to_dict should return a dictionary"

    def test_to_dict_contains_required_fields(self, mock_feedstock: Mock) -> None:
        """
        Test that to_dict includes all required fields.

        Args:
            mock_feedstock: Mock Feedstock fixture.
        """
        digester = Digester("dig_1", mock_feedstock, V_liq=2000, V_gas=300, T_ad=308.15)
        digester.initialize()

        config = digester.to_dict()

        required_fields = ["component_id", "component_type", "name", "V_liq", "V_gas", "T_ad"]
        for field in required_fields:
            assert field in config, f"to_dict should include '{field}'"

    def test_from_dict_recreates_digester(self, mock_feedstock: Mock) -> None:
        """
        Test that from_dict can recreate a digester from configuration.

        Args:
            mock_feedstock: Mock Feedstock fixture.
        """
        original = Digester("dig_1", mock_feedstock, V_liq=2000, V_gas=300, T_ad=308.15, name="Test Digester")
        original.initialize()

        config = original.to_dict()
        recreated = Digester.from_dict(config, mock_feedstock)

        assert recreated.component_id == original.component_id, "Component ID should match"
        assert recreated.V_liq == original.V_liq, "V_liq should match"
        assert recreated.V_gas == original.V_gas, "V_gas should match"
        assert recreated.T_ad == original.T_ad, "T_ad should match"
        assert recreated.name == original.name, "Name should match"

    def test_to_dict_roundtrip_preserves_state(self, mock_feedstock: Mock) -> None:
        """
        Test that serialization and deserialization preserves state.

        Args:
            mock_feedstock: Mock Feedstock fixture.
        """
        custom_state = [0.015] * 37
        custom_Q = [18, 12, 0, 0, 0, 0, 0, 0, 0, 0]

        original = Digester("dig_1", mock_feedstock)
        original.initialize({"adm1_state": custom_state, "Q_substrates": custom_Q})

        config = original.to_dict()
        recreated = Digester.from_dict(config, mock_feedstock)

        assert recreated.adm1_state == original.adm1_state, "ADM1 state should be preserved"


class TestDigesterConnections:
    """Test suite for Digester component connections."""

    @pytest.fixture
    def mock_feedstock(self) -> Mock:
        """
        Create a mock Feedstock object.

        Returns:
            Mock Feedstock object.
        """
        feedstock = Mock(spec=Feedstock)
        feedstock.mySubstrates = Mock()
        return feedstock

    def test_add_input_connection(self, mock_feedstock: Mock) -> None:
        """
        Test adding input connections to digester.

        Args:
            mock_feedstock: Mock Feedstock fixture.
        """
        digester = Digester("dig_1", mock_feedstock)

        digester.add_input("previous_stage")

        assert "previous_stage" in digester.inputs, "Input should be added"

    def test_add_output_connection(self, mock_feedstock: Mock) -> None:
        """
        Test adding output connections to digester.

        Args:
            mock_feedstock: Mock Feedstock fixture.
        """
        digester = Digester("dig_1", mock_feedstock)

        digester.add_output("chp_unit")

        assert "chp_unit" in digester.outputs, "Output should be added"

    def test_no_duplicate_inputs(self, mock_feedstock: Mock) -> None:
        """
        Test that duplicate inputs are not added.

        Args:
            mock_feedstock: Mock Feedstock fixture.
        """
        digester = Digester("dig_1", mock_feedstock)

        digester.add_input("source")
        digester.add_input("source")

        assert digester.inputs.count("source") == 1, "Duplicate inputs should not be added"


class TestDigesterProperties:
    """Test suite for Digester properties."""

    @pytest.fixture
    def mock_feedstock(self) -> Mock:
        """
        Create a mock Feedstock object.

        Returns:
            Mock Feedstock object.
        """
        feedstock = Mock(spec=Feedstock)
        feedstock.mySubstrates = Mock()
        return feedstock

    def test_component_type_is_digester(self, mock_feedstock: Mock) -> None:
        """
        Test that component type is correctly identified as digester.

        Args:
            mock_feedstock: Mock Feedstock fixture.
        """
        digester = Digester("dig_1", mock_feedstock)

        assert digester.component_type.value == "digester", "Component type should be 'digester'"

    def test_get_state_returns_copy(self, mock_feedstock: Mock) -> None:
        """
        Test that get_state returns a copy of state.

        Args:
            mock_feedstock: Mock Feedstock fixture.
        """
        digester = Digester("dig_1", mock_feedstock)
        digester.initialize()

        state_copy = digester.get_state()
        state_copy["Q_gas"] = 9999

        assert digester.state["Q_gas"] != 9999, "get_state should return a copy, not reference"

    def test_set_state_updates_state(self, mock_feedstock: Mock) -> None:
        """
        Test that set_state updates the state.

        Args:
            mock_feedstock: Mock Feedstock fixture.
        """
        digester = Digester("dig_1", mock_feedstock)
        digester.initialize()

        new_state = digester.state.copy()
        new_state["Q_gas"] = 2000

        digester.set_state(new_state)

        assert digester.state["Q_gas"] == 2000, "set_state should update state"


class TestDigesterClrLoading:
    """Tests for CLR loading helper branches and module import guards."""

    def test_try_load_clr_returns_none_on_darwin(self) -> None:
        """Darwin should short-circuit and return None."""
        with patch("platform.system", return_value="Darwin"):
            assert digester_module.try_load_clr() is None

    def test_try_load_clr_prints_and_returns_none_on_import_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Import failures for clr should be handled gracefully."""
        import builtins

        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "clr":
                raise RuntimeError("clr import failed")
            return original_import(name, *args, **kwargs)

        with patch("platform.system", return_value="Windows"):
            with patch("builtins.__import__", side_effect=fake_import):
                assert digester_module.try_load_clr() is None

        captured = capsys.readouterr()
        assert "clr import failed" in captured.out

    def test_module_import_raises_when_clr_unavailable(self) -> None:
        """Importing the module on Darwin should raise the runtime guard error."""
        import importlib.util
        import sys
        from pathlib import Path

        module_path = Path(digester_module.__file__)
        module_name = "pyadm1.components.biological._digester_runtimeerror_test"
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        assert spec is not None and spec.loader is not None

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            with patch("platform.system", return_value="Darwin"):
                with pytest.raises(RuntimeError, match="CLR features unavailable on this platform"):
                    spec.loader.exec_module(module)
        finally:
            sys.modules.pop(module_name, None)


class TestDigesterUncoveredBranches:
    """Tests for fallback branches not covered by the default happy path."""

    @pytest.fixture
    def mock_feedstock(self) -> Mock:
        feedstock = Mock(spec=Feedstock)
        feedstock.mySubstrates = Mock()
        return feedstock

    def test_initialize_uses_provided_gas_storage_state_and_falls_back_on_error(
        self, mock_feedstock: Mock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Gas storage init should retry without state if custom init fails."""
        digester = Digester("dig_1", mock_feedstock)
        custom_gs_state = {"stored_volume_m3": 12.0}
        failing_init = Mock(side_effect=[RuntimeError("storage init failed"), None])

        with patch.object(digester, "gas_storage") as mock_storage:
            mock_storage.initialize = failing_init
            digester.initialize({"adm1_state": [0.02] * 37, "gas_storage": custom_gs_state})

        assert failing_init.call_count == 2
        assert failing_init.call_args_list[0].args == (custom_gs_state,)
        assert failing_init.call_args_list[1].args == ()
        assert "storage init failed" in capsys.readouterr().out

    def test_step_handles_upstream_input_branch_and_indicator_failure(
        self, mock_feedstock: Mock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Step should enter the upstream-input branch and keep running if DLL indicators fail."""
        digester = Digester("dig_1", mock_feedstock)
        digester.initialize()
        simulated_state = [0.02] * 37
        simulated_state[33:37] = [0.1, 0.2, 0.3, 1.0]

        with patch.object(digester, "adm1") as mock_adm1:
            mock_adm1.create_influent = Mock()
            mock_adm1.calc_gas = Mock(return_value=(12.0, 7.0, 5.0, 1.01))
            with patch.object(digester.simulator, "simulate_AD_plant", return_value=simulated_state):
                digester.gas_storage.outputs_data = {}
                with patch.object(
                    digester.gas_storage,
                    "step",
                    return_value={
                        "stored_volume_m3": 5.0,
                        "pressure_bar": 1.0,
                        "vented_volume_m3": 0.0,
                        "Q_gas_supplied_m3_per_day": 0.0,
                    },
                ) as mock_storage_step:
                    admstate_mock = Mock()
                    admstate_mock.calcPHOfADMstate.side_effect = RuntimeError("indicator failure")
                    with patch.object(digester_module, "ADMstate", admstate_mock):
                        result = digester.step(
                            t=0.0,
                            dt=1.0 / 24.0,
                            inputs={
                                "Q_substrates": [1.0] * 10,
                                "Q_in": 2.5,
                                "state_in": [0.01] * 37,
                            },
                        )

        assert mock_storage_step.called
        assert result["Q_out"] == pytest.approx(10.0)
        assert result["pH"] == 7.0
        assert "Warning: Could not calculate process indicators" in capsys.readouterr().out


class TestDigesterCalibrationParameters:
    """Tests for calibration parameter helper methods."""

    @pytest.fixture
    def mock_feedstock(self) -> Mock:
        feedstock = Mock(spec=Feedstock)
        feedstock.mySubstrates = Mock()
        return feedstock

    def test_get_calibration_parameters_returns_empty_when_unset(self, mock_feedstock: Mock) -> None:
        digester = Digester("dig_1", mock_feedstock)
        assert digester.get_calibration_parameters() == {}

    def test_apply_get_and_clear_calibration_parameters_with_verbose_logging(
        self, mock_feedstock: Mock, capsys: pytest.CaptureFixture[str]
    ) -> None:
        digester = Digester("dig_1", mock_feedstock)
        digester._verbose = True

        digester.apply_calibration_parameters({"k_dis": 0.55, "Y_su": 0.105})

        params = digester.get_calibration_parameters()
        assert params == {"k_dis": 0.55, "Y_su": 0.105}
        assert digester.adm1._calibration_params == params
        assert digester.adm1._calibration_params is not params

        digester.clear_calibration_parameters()

        assert digester.get_calibration_parameters() == {}
        assert not hasattr(digester.adm1, "_calibration_params")
        output = capsys.readouterr().out
        assert "Applied 2 calibration parameters" in output
        assert "Cleared calibration parameters" in output
