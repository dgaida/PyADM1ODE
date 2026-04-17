"""
Unit tests for ADM1 core functionality.

This module tests the main ADM1 class which implements the
ADM1 ODE system for anaerobic digestion simulation.
"""

import pytest
import numpy as np
import pandas as pd
from types import SimpleNamespace
from unittest.mock import Mock, patch
from pathlib import Path
import builtins

import pyadm1.core.adm1 as adm1_module
from pyadm1.core.adm1 import ADM1, get_state_zero_from_initial_state
from pyadm1.substrates.feedstock import Feedstock


class TestGetStateZeroFromInitialState:
    """Test suite for loading initial states from CSV files."""

    @pytest.fixture
    def sample_csv_data(self, tmp_path: Path) -> Path:
        """
        Create a temporary CSV file with sample initial state data.

        Args:
            tmp_path: pytest fixture providing temporary directory path.

        Returns:
            Path to the created CSV file.
        """
        csv_file = tmp_path / "test_initial.csv"

        # Create sample data matching ADM1 state structure
        data = {
            "S_su": [0.01],
            "S_aa": [0.001],
            "S_fa": [0.04],
            "S_va": [0.005],
            "S_bu": [0.01],
            "S_pro": [0.028],
            "S_ac": [0.97],
            "S_h2": [1e-7],
            "S_ch4": [0.056],
            "S_co2": [0.012],
            "S_nh4": [0.24],
            "S_I": [10.7],
            "X_xc": [18.0],
            "X_ch": [0.25],
            "X_pr": [0.055],
            "X_li": [0.018],
            "X_su": [7.8],
            "X_aa": [1.36],
            "X_fa": [0.33],
            "X_c4": [1.02],
            "X_pro": [0.88],
            "X_ac": [3.18],
            "X_h2": [1.67],
            "X_I": [38.3],
            "X_p": [2.04],
            "S_cation": [0.0],
            "S_anion": [0.0],
            "S_va_ion": [0.005],
            "S_bu_ion": [0.01],
            "S_pro_ion": [0.028],
            "S_ac_ion": [0.97],
            "S_hco3_ion": [0.23],
            "S_nh3": [0.013],
            "pi_Sh2": [6e-6],
            "pi_Sch4": [0.56],
            "pi_Sco2": [0.43],
            "pTOTAL": [0.98],
        }

        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)

        return csv_file

    def test_get_state_zero_returns_list(self, sample_csv_data: Path) -> None:
        """
        Test that the function returns a list.

        Args:
            sample_csv_data: Fixture providing path to sample CSV file.
        """
        state = get_state_zero_from_initial_state(str(sample_csv_data))

        assert isinstance(state, list), "Should return a list"

    def test_get_state_zero_correct_length(self, sample_csv_data: Path) -> None:
        """
        Test that the returned state has 37 elements.

        Args:
            sample_csv_data: Fixture providing path to sample CSV file.
        """
        state = get_state_zero_from_initial_state(str(sample_csv_data))

        assert len(state) == 37, f"State should have 37 elements, got {len(state)}"

    def test_get_state_zero_all_numeric(self, sample_csv_data: Path) -> None:
        """
        Test that all state values are numeric.

        Args:
            sample_csv_data: Fixture providing path to sample CSV file.
        """
        state = get_state_zero_from_initial_state(str(sample_csv_data))

        for i, val in enumerate(state):
            assert isinstance(val, (int, float)), f"Element {i} should be numeric"
            assert np.isfinite(val), f"Element {i} should be finite"

    def test_get_state_zero_file_not_found(self) -> None:
        """Test that appropriate error is raised for non-existent file."""
        with pytest.raises(FileNotFoundError):
            get_state_zero_from_initial_state("nonexistent_file.csv")


class TestADM1Initialization:
    """Test suite for ADM1 initialization."""

    @pytest.fixture
    def mock_feedstock(self) -> Mock:
        """
        Create a mock Feedstock object.

        Returns:
            Mock Feedstock object with necessary attributes.
        """
        feedstock = Mock(spec=Feedstock)
        feedstock.mySubstrates = Mock()
        return feedstock

    def test_initialization_sets_liquid_volume(self, mock_feedstock: Mock) -> None:
        """
        Test that initialization sets the liquid volume correctly.

        Args:
            mock_feedstock: Mock Feedstock fixture.
        """
        adm1 = ADM1(mock_feedstock)

        assert hasattr(adm1, "V_liq"), "Should have V_liq attribute"
        assert adm1.V_liq == 1977, f"V_liq should be 1977, got {adm1.V_liq}"

    def test_initialization_stores_feedstock(self, mock_feedstock: Mock) -> None:
        """
        Test that feedstock is stored correctly.

        Args:
            mock_feedstock: Mock Feedstock fixture.
        """
        adm1 = ADM1(mock_feedstock)

        assert adm1.feedstock == mock_feedstock, "Should store feedstock reference"

    def test_initialization_creates_empty_lists(self, mock_feedstock: Mock) -> None:
        """
        Test that initialization creates empty lists for tracking results.

        Args:
            mock_feedstock: Mock Feedstock fixture.
        """
        adm1 = ADM1(mock_feedstock)

        assert isinstance(adm1._Q_GAS, list), "_Q_GAS should be a list"
        assert isinstance(adm1._Q_CH4, list), "_Q_CH4 should be a list"
        assert isinstance(adm1._pH_l, list), "_pH_l should be a list"
        assert len(adm1._Q_GAS) == 0, "_Q_GAS should be empty initially"


class TestADM1CalcGas:
    """Test suite for biogas calculation methods."""

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
    def adm1_instance(self, mock_feedstock: Mock) -> ADM1:
        """
        Create a ADM1 instance for testing.

        Args:
            mock_feedstock: Mock Feedstock fixture.

        Returns:
            ADM1 instance.
        """
        return ADM1(mock_feedstock)

    def test_calc_gas_returns_four_values(self, adm1_instance: ADM1) -> None:
        """
        Test that calc_gas returns 4 values.

        Args:
            adm1_instance: ADM1 fixture.
        """
        result = adm1_instance.calc_gas(pi_Sh2=5e-6, pi_Sch4=0.55, pi_Sco2=0.42, pTOTAL=0.98)

        assert len(result) == 4, "calc_gas should return 4 values"

    def test_calc_gas_all_positive(self, adm1_instance: ADM1) -> None:
        """
        Test that all gas flow rates are positive.

        Args:
            adm1_instance: ADM1 fixture.
        """
        q_gas, q_ch4, q_co2, p_gas = adm1_instance.calc_gas(pi_Sh2=5e-6, pi_Sch4=0.55, pi_Sco2=0.42, pTOTAL=0.98)

        assert q_gas >= 0, "Total gas flow should be non-negative"
        assert q_ch4 >= 0, "Methane flow should be non-negative"
        assert q_co2 >= 0, "CO2 flow should be non-negative"
        assert p_gas >= 0, "Gas pressure should be non-negative"

    def test_calc_gas_methane_less_than_total(self, adm1_instance: ADM1) -> None:
        """
        Test that methane flow is less than or equal to total gas flow.

        Args:
            adm1_instance: ADM1 fixture.
        """
        q_gas, q_ch4, q_co2, p_gas = adm1_instance.calc_gas(pi_Sh2=5e-6, pi_Sch4=0.55, pi_Sco2=0.42, pTOTAL=0.98)

        assert q_ch4 <= q_gas, "Methane flow should be <= total gas flow"
        assert q_co2 <= q_gas, "CO2 flow should be <= total gas flow"

    def test_calc_gas_with_zero_pressure(self, adm1_instance: ADM1) -> None:
        """
        Test calc_gas with zero partial pressures.

        Args:
            adm1_instance: ADM1 fixture.
        """
        q_gas, q_ch4, q_co2, p_gas = adm1_instance.calc_gas(pi_Sh2=0.0, pi_Sch4=0.0, pi_Sco2=0.0, pTOTAL=1.0)

        # Should handle zero pressures gracefully
        assert np.isfinite(q_gas), "q_gas should be finite"
        assert np.isfinite(q_ch4), "q_ch4 should be finite"
        assert np.isfinite(q_co2), "q_co2 should be finite"

    def test_calc_gas_supports_numpy_scalar_arrays(self, adm1_instance: ADM1) -> None:
        """Cover ndarray non-negativity branches in calc_gas."""
        with patch.object(
            adm1_module.ADMParams,
            "getADMgasparams",
            return_value=(None, np.array([1.0]), None, None, None, None),
        ):
            q_gas, q_ch4, q_co2, _ = adm1_instance.calc_gas(
                pi_Sh2=1e-6,
                pi_Sch4=0.55,
                pi_Sco2=0.42,
                pTOTAL=1.5,
            )

        assert isinstance(q_gas, np.ndarray)
        assert isinstance(q_ch4, np.ndarray)
        assert isinstance(q_co2, np.ndarray)
        assert float(np.asarray(q_gas).reshape(-1)[0]) >= 0.0


class TestADM1CreateInfluent:
    """Test suite for influent stream creation."""

    @pytest.fixture
    def mock_feedstock(self) -> Mock:
        """
        Create a mock Feedstock with get_influent_dataframe method.

        Returns:
            Mock Feedstock object.
        """
        feedstock = Mock(spec=Feedstock)
        feedstock.mySubstrates = Mock()

        # Mock get_influent_dataframe to return a DataFrame
        mock_df = pd.DataFrame(
            {
                "S_su": [0.01],
                "S_aa": [0.001],
                "S_fa": [0.04],
                "S_va": [0.005],
                "S_bu": [0.01],
                "S_pro": [0.028],
                "S_ac": [0.97],
                "S_h2": [1e-7],
                "S_ch4": [0.056],
                "S_co2": [0.012],
                "S_nh4": [0.24],
                "S_I": [10.7],
                "X_xc": [18.0],
                "X_ch": [0.25],
                "X_pr": [0.055],
                "X_li": [0.018],
                "X_su": [7.8],
                "X_aa": [1.36],
                "X_fa": [0.33],
                "X_c4": [1.02],
                "X_pro": [0.88],
                "X_ac": [3.18],
                "X_h2": [1.67],
                "X_I": [38.3],
                "X_p": [2.04],
                "S_cation": [0.0],
                "S_anion": [0.0],
                "S_va_ion": [0.005],
                "S_bu_ion": [0.01],
                "S_pro_ion": [0.028],
                "S_ac_ion": [0.97],
                "S_hco3_ion": [0.23],
                "S_nh3": [0.013],
                "Q": [25.0],
            }
        )
        feedstock.get_influent_dataframe.return_value = mock_df

        return feedstock

    def test_create_influent_stores_flow_rates(self, mock_feedstock: Mock) -> None:
        """
        Test that createInfluent stores the flow rate vector.

        Args:
            mock_feedstock: Mock Feedstock fixture.
        """
        adm1 = ADM1(mock_feedstock)
        Q = [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]

        adm1.create_influent(Q, 0)

        assert adm1._Q == Q, "Should store flow rate vector"

    def test_create_influent_calls_get_influent_dataframe(self, mock_feedstock: Mock) -> None:
        """
        Test that createInfluent calls feedstock method.

        Args:
            mock_feedstock: Mock Feedstock fixture.
        """
        adm1 = ADM1(mock_feedstock)
        Q = [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]

        adm1.create_influent(Q, 0)

        mock_feedstock.get_influent_dataframe.assert_called_once_with(Q)

    def test_set_influent_uses_last_row_when_index_out_of_bounds(self, mock_feedstock: Mock) -> None:
        """Cover out-of-bounds index fallback in _set_influent."""
        adm1 = ADM1(mock_feedstock)
        influent_df = pd.DataFrame(
            {
                "S_su": [0.01, 0.99],
                "S_aa": [0.001, 0.002],
                "S_fa": [0.04, 0.05],
                "S_va": [0.005, 0.006],
                "S_bu": [0.01, 0.02],
                "S_pro": [0.028, 0.029],
                "S_ac": [0.97, 0.98],
                "S_h2": [1e-7, 2e-7],
                "S_ch4": [0.056, 0.057],
                "S_co2": [0.012, 0.013],
                "S_nh4": [0.24, 0.25],
                "S_I": [10.7, 10.8],
                "X_xc": [18.0, 18.1],
                "X_ch": [0.25, 0.26],
                "X_pr": [0.055, 0.056],
                "X_li": [0.018, 0.019],
                "X_su": [7.8, 7.9],
                "X_aa": [1.36, 1.37],
                "X_fa": [0.33, 0.34],
                "X_c4": [1.02, 1.03],
                "X_pro": [0.88, 0.89],
                "X_ac": [3.18, 3.19],
                "X_h2": [1.67, 1.68],
                "X_I": [38.3, 38.4],
                "X_p": [2.04, 2.05],
                "S_cation": [0.0, 0.1],
                "S_anion": [0.0, 0.1],
                "S_va_ion": [0.005, 0.015],
                "S_bu_ion": [0.01, 0.02],
                "S_pro_ion": [0.028, 0.038],
                "S_ac_ion": [0.97, 1.07],
                "S_hco3_ion": [0.23, 0.24],
                "S_nh3": [0.013, 0.014],
                "Q": [25.0, 30.0],
            }
        )

        adm1._set_influent(influent_df, i=999)

        assert adm1._state_input is not None
        assert adm1._state_input[0] == 0.99
        assert adm1._state_input[-1] == 30.0


class TestADM1SaveFinalState:
    """Test suite for saving final state to CSV."""

    @pytest.fixture
    def mock_feedstock(self) -> Mock:
        """
        Create a mock Feedstock object.

        Returns:
            Mock Feedstock object.
        """
        feedstock = Mock(spec=Feedstock)
        feedstock.mySubstrates = Mock()

        # Mock header method
        feedstock.header.return_value = [
            "S_su",
            "S_aa",
            "S_fa",
            "S_va",
            "S_bu",
            "S_pro",
            "S_ac",
            "S_h2",
            "S_ch4",
            "S_co2",
            "S_nh4",
            "S_I",
            "X_xc",
            "X_ch",
            "X_pr",
            "X_li",
            "X_su",
            "X_aa",
            "X_fa",
            "X_c4",
            "X_pro",
            "X_ac",
            "X_h2",
            "X_I",
            "X_p",
            "S_cation",
            "S_anion",
            "S_va_ion",
            "S_bu_ion",
            "S_pro_ion",
            "S_ac_ion",
            "S_hco3_ion",
            "S_nh3",
            "Q",
        ]

        return feedstock

    def test_save_final_state_creates_csv(self, mock_feedstock: Mock, tmp_path: Path) -> None:
        """
        Test that save_final_state_in_csv creates a CSV file.

        Args:
            mock_feedstock: Mock Feedstock fixture.
            tmp_path: pytest fixture providing temporary directory.
        """
        adm1 = ADM1(mock_feedstock)

        # Create sample simulation results
        state = [0.01] * 37  # 37 state variables
        simulate_results = [state, state, state]

        output_file = tmp_path / "test_output.csv"
        adm1.save_final_state_in_csv(simulate_results, str(output_file))

        assert output_file.exists(), "CSV file should be created"

    def test_save_final_state_contains_only_last_state(self, mock_feedstock: Mock, tmp_path: Path) -> None:
        """
        Test that saved CSV contains only the last state.

        Args:
            mock_feedstock: Mock Feedstock fixture.
            tmp_path: pytest fixture providing temporary directory.
        """
        adm1 = ADM1(mock_feedstock)

        # Create sample simulation results with different values
        state1 = [0.01] * 37
        state2 = [0.02] * 37
        state3 = [0.03] * 37
        simulate_results = [state1, state2, state3]

        output_file = tmp_path / "test_output.csv"
        adm1.save_final_state_in_csv(simulate_results, str(output_file))

        # Read back and check
        df = pd.read_csv(output_file)
        assert len(df) == 1, "Should contain only one row (final state)"


class TestADM1Properties:
    """Test suite for ADM1 properties."""

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

    def test_property_T_ad(self, mock_feedstock: Mock) -> None:
        """
        Test T_ad property returns digester temperature.

        Args:
            mock_feedstock: Mock Feedstock fixture.
        """
        adm1 = ADM1(mock_feedstock)

        T_ad = adm1.T_ad
        assert T_ad == 308.15, f"T_ad should be 308.15 K, got {T_ad}"

    def test_property_feedstock(self, mock_feedstock: Mock) -> None:
        """
        Test feedstock property returns stored feedstock object.

        Args:
            mock_feedstock: Mock Feedstock fixture.
        """
        adm1 = ADM1(mock_feedstock)

        assert adm1.feedstock == mock_feedstock

    def test_properties_return_lists(self, mock_feedstock: Mock) -> None:
        """
        Test that result properties return lists.

        Args:
            mock_feedstock: Mock Feedstock fixture.
        """
        adm1 = ADM1(mock_feedstock)

        assert isinstance(adm1.Q_GAS, list), "Q_GAS should return a list"
        assert isinstance(adm1.Q_CH4, list), "Q_CH4 should return a list"
        assert isinstance(adm1.pH_l, list), "pH_l should return a list"
        assert isinstance(adm1.VFA, list), "VFA should return a list"
        assert isinstance(adm1.TAC, list), "TAC should return a list"

    def test_remaining_result_properties_return_backing_lists(self, mock_feedstock: Mock) -> None:
        adm1 = ADM1(mock_feedstock)
        adm1._Q_CO2 = [1.1]
        adm1._P_GAS = [0.98]
        adm1._FOSTAC = [0.25]
        adm1._AcvsPro = [1.5]

        assert adm1.Q_CO2 == [1.1]
        assert adm1.P_GAS == [0.98]
        assert adm1.VFA_TA == [0.25]
        assert adm1.AcvsPro == [1.5]


class TestADM1AdditionalBranches:
    """Targeted tests for remaining uncovered helper branches in adm1.py."""

    @pytest.fixture
    def mock_feedstock(self) -> Mock:
        feedstock = Mock(spec=Feedstock)
        feedstock.mySubstrates = Mock()
        return feedstock

    def test_resume_from_broken_simulation_appends_ch4_values(self, mock_feedstock: Mock) -> None:
        adm1 = ADM1(mock_feedstock)
        adm1.resume_from_broken_simulation([1.0, 2.0, 3.0])
        assert adm1._Q_CH4 == [1.0, 2.0, 3.0]

    def test_print_params_at_current_state_initializes_tracking_lists(self, mock_feedstock: Mock) -> None:
        adm1 = ADM1(mock_feedstock)
        state = [0.01] * 37
        state[33:37] = [0.1, 0.2, 0.3, 1.0]

        admstate_mock = SimpleNamespace(
            calcPHOfADMstate=lambda s: 7.23,  # noqa: ARG005
            calcFOSTACOfADMstate=lambda s: SimpleNamespace(Value=0.25),  # noqa: ARG005
            calcAcetic_vs_PropionicOfADMstate=lambda s: SimpleNamespace(Value=1.4),  # noqa: ARG005
            calcVFAOfADMstate=lambda s, u: SimpleNamespace(Value=0.55),  # noqa: ARG005
            calcTACOfADMstate=lambda s, u: SimpleNamespace(Value=2.5),  # noqa: ARG005
        )

        with patch.object(adm1_module, "ADMstate", admstate_mock):
            with patch.object(adm1, "calc_gas", return_value=(100.0, 60.0, 40.0, 0.6)) as calc_gas:
                adm1.print_params_at_current_state(state)

        calc_gas.assert_called_once_with(0.1, 0.2, 0.3, 1.0)
        assert adm1._pH_l == [7.2, 7.2]
        assert adm1._FOSTAC == [0.25, 0.25]
        assert adm1._AcvsPro == [1.4, 1.4]
        assert adm1._VFA == [0.55, 0.55]
        assert adm1._TAC == [2.5, 2.5]
        assert adm1._Q_GAS == [100.0, 100.0, 100.0]
        assert adm1._Q_CH4 == [60.0, 60.0, 60.0]
        assert adm1._Q_CO2 == [40.0, 40.0, 40.0]
        assert adm1._P_GAS == [0.6, 0.6, 0.6]

    def test_get_substrate_dependent_params_returns_defaults_when_q_not_set(self, mock_feedstock: Mock) -> None:
        adm1 = ADM1(mock_feedstock)
        params = adm1._get_substrate_dependent_params()
        assert params["k_dis"] == 0.5
        assert params["k_m_h2"] == 35.0

    def test_get_substrate_dependent_params_raises_after_all_fallback_paths_fail(self, mock_feedstock: Mock) -> None:
        adm1 = ADM1(mock_feedstock)
        adm1._Q = [1.0]

        failing_subs = Mock()
        failing_subs.calcfFactors.side_effect = RuntimeError("cf fail")
        failing_subs.calcDisintegrationParam.side_effect = RuntimeError("dis fail")
        mock_feedstock.mySubstrates.return_value = failing_subs

        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: ANN001, A002
            if name == "System":
                raise ImportError("no System")
            return real_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=fake_import):
            with pytest.raises(TypeError, match="Failed to evaluate substrate-dependent parameters"):
                adm1._get_substrate_dependent_params()

    def test_adm1_calibration_parameter_helpers(self, mock_feedstock: Mock) -> None:
        adm1 = ADM1(mock_feedstock)

        assert adm1.get_calibration_parameters() == {}

        adm1.set_calibration_parameters({"k_dis": 0.55})
        params = adm1.get_calibration_parameters()
        assert params == {"k_dis": 0.55}

        adm1.clear_calibration_parameters()
        assert adm1.get_calibration_parameters() == {}


class TestADM1ODEIntegration:
    """Integration tests for ADM1 ODE system."""

    @pytest.fixture
    def mock_feedstock_full(self) -> Mock:
        """
        Create a fully configured mock Feedstock.

        Returns:
            Mock Feedstock with all necessary methods.
        """
        feedstock = Mock(spec=Feedstock)

        # Mock mySubstrates with required methods
        mock_substrates = Mock()
        mock_substrates.calcfFactors.return_value = (0.2, 0.2, 0.3, 0.2, 0.1, 0.0)
        mock_substrates.calcDisintegrationParam.return_value = 0.5
        mock_substrates.calcHydrolysisParams.return_value = (10, 10, 10)
        mock_substrates.calcMaxUptakeRateParams.return_value = (20, 13, 8, 35)

        feedstock.mySubstrates = Mock(return_value=mock_substrates)

        return feedstock

    def test_ADM1_ODE_returns_correct_length(self, mock_feedstock_full: Mock) -> None:
        """
        Test that ADM1_ODE returns 37 derivatives.

        Args:
            mock_feedstock_full: Mock Feedstock fixture.
        """
        adm1 = ADM1(mock_feedstock_full)

        # Set up influent
        Q = [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]
        adm1._Q = Q
        adm1._state_input = [0.01] * 34  # Mock input state

        # Create initial state
        state_zero = [0.01] * 37

        # Calculate derivatives
        derivatives = adm1.ADM1_ODE(0, state_zero)

        assert len(derivatives) == 37, f"Should return 37 derivatives, got {len(derivatives)}"

    def test_ADM1_ODE_all_finite(self, mock_feedstock_full: Mock) -> None:
        """
        Test that all derivatives are finite numbers.

        Args:
            mock_feedstock_full: Mock Feedstock fixture.
        """
        adm1 = ADM1(mock_feedstock_full)

        Q = [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]
        adm1._Q = Q
        adm1._state_input = [0.01] * 34

        state_zero = [0.01] * 37

        derivatives = adm1.ADM1_ODE(0, state_zero)

        for i, deriv in enumerate(derivatives):
            assert np.isfinite(deriv), f"Derivative {i} is not finite: {deriv}"
