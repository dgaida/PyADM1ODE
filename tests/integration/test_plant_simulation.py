"""
Unit tests for the Simulator class.

This module tests the Simulator class which orchestrates ADM1 simulations
with different substrate feed scenarios.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from pyadm1.simulation.simulator import Simulator
from pyadm1.core.adm1 import ADM1


class TestSimulatorInitialization:
    """Test suite for Simulator initialization."""

    @pytest.fixture
    def mock_adm1(self) -> Mock:
        """
        Create a mock ADM1 object.

        Returns:
            Mock ADM1 object with necessary attributes.
        """
        adm1 = Mock(spec=ADM1)
        adm1.V_liq = 1977
        adm1.create_influent = Mock()
        adm1.ADM1_ODE = Mock(return_value=[0.01] * 37)
        return adm1

    def test_initialization_stores_adm1(self, mock_adm1: Mock) -> None:
        """
        Test that Simulator stores the ADM1 instance.

        Args:
            mock_adm1: Mock ADM1 fixture.
        """
        simulator = Simulator(mock_adm1)

        assert simulator._adm1 == mock_adm1, "Should store ADM1 reference"

    def test_initialization_sets_solver_method(self, mock_adm1: Mock) -> None:
        """
        Test that Simulator sets the solver method.

        Args:
            mock_adm1: Mock ADM1 fixture.
        """
        simulator = Simulator(mock_adm1)

        assert hasattr(simulator, "_solvermethod"), "Should have solver method attribute"
        assert simulator._solvermethod == "BDF", "Should use BDF solver for stiff ODEs"


class TestSimulatorSimulateADPlant:
    """Test suite for simulateADplant method."""

    @pytest.fixture
    def mock_adm1_with_params(self) -> Mock:
        """
        Create a mock ADM1 with print_params_at_current_state method.

        Returns:
            Mock ADM1 object.
        """
        adm1 = Mock(spec=ADM1)
        adm1.V_liq = 1977
        adm1.ADM1_ODE = Mock(return_value=[0.01] * 37)
        adm1.print_params_at_current_state = Mock()
        return adm1

    def test_simulate_returns_list(self, mock_adm1_with_params: Mock) -> None:
        """
        Test that simulateADplant returns a list.

        Args:
            mock_adm1_with_params: Mock ADM1 fixture.
        """
        simulator = Simulator(mock_adm1_with_params)

        state_zero = [0.01] * 37
        tstep = [0, 1]

        with patch("scipy.integrate.solve_ivp") as mock_solve:
            # Mock the solver result
            mock_result = Mock()
            mock_result.y = np.array([[val] * 20 for val in state_zero])
            mock_solve.return_value = mock_result

            result = simulator.simulate_AD_plant(tstep, state_zero)

        assert isinstance(result, list), "Should return a list"

    def test_simulate_returns_correct_length(self, mock_adm1_with_params: Mock) -> None:
        """
        Test that simulateADplant returns state with 37 elements.

        Args:
            mock_adm1_with_params: Mock ADM1 fixture.
        """
        simulator = Simulator(mock_adm1_with_params)

        state_zero = [0.01] * 37
        tstep = [0, 1]

        with patch("scipy.integrate.solve_ivp") as mock_solve:
            mock_result = Mock()
            mock_result.y = np.array([[val] * 20 for val in state_zero])
            mock_solve.return_value = mock_result

            result = simulator.simulate_AD_plant(tstep, state_zero)

        assert len(result) == 37, f"Should return 37 elements, got {len(result)}"

    def test_simulate_calls_print_params(self, mock_adm1_with_params: Mock) -> None:
        """
        Test that simulateADplant calls print_params_at_current_state.

        Args:
            mock_adm1_with_params: Mock ADM1 fixture.
        """
        simulator = Simulator(mock_adm1_with_params)

        state_zero = [0.01] * 37
        tstep = [0, 1]

        with patch("scipy.integrate.solve_ivp") as mock_solve:
            mock_result = Mock()
            mock_result.y = np.array([[val] * 20 for val in state_zero])
            mock_solve.return_value = mock_result

            simulator.simulate_AD_plant(tstep, state_zero)

        mock_adm1_with_params.print_params_at_current_state.assert_called_once()


class TestSimulatorDetermineBestFeed:
    """Test suite for determineBestFeedbyNSims method."""

    @pytest.fixture
    def mock_adm1_for_optimization(self) -> Mock:
        """
        Create a mock ADM1 for feed optimization tests.

        Returns:
            Mock ADM1 object with necessary methods.
        """
        adm1 = Mock(spec=ADM1)
        adm1.V_liq = 1977
        adm1.createInfluent = Mock()
        adm1.ADM1_ODE = Mock(return_value=[0.01] * 37)
        adm1.calc_gas = Mock(return_value=(1500, 900, 600, 0.95))
        adm1.feedstock = Mock()
        adm1.feedstock.return_value.get_substrate_feed_mixtures = Mock(
            return_value=[[15, 10, 0, 0, 0, 0, 0, 0, 0, 0] for _ in range(13)]
        )
        return adm1

    def test_determine_best_feed_returns_ten_values(self, mock_adm1_for_optimization: Mock) -> None:
        """
        Test that determineBestFeedbyNSims returns 10 values.

        Args:
            mock_adm1_for_optimization: Mock ADM1 fixture.
        """
        # Fix feedstock mock to match how Simulator expects to call it
        from pyadm1.substrates.feedstock import Feedstock

        mock_feedstock = Mock(spec=Feedstock)
        mock_feedstock.get_substrate_feed_mixtures = Mock(
            return_value=[[15 + i, 10 + i, 0, 0, 0, 0, 0, 0, 0, 0] for i in range(13)]
        )

        mock_adm1_for_optimization.feedstock = mock_feedstock

        simulator = Simulator(mock_adm1_for_optimization)

        state_zero = [0.01] * 37
        Q = [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]
        Qch4sp = 900
        feeding_freq = 48

        with patch("scipy.integrate.solve_ivp") as mock_solve:
            mock_result = Mock()
            mock_result.y = np.array([[val] * 140 for val in state_zero])
            mock_solve.return_value = mock_result

            result = simulator.determine_best_feed_by_n_sims(state_zero, Q, Qch4sp, feeding_freq, n=13)

        assert len(result) == 10, f"Should return 10 values, got {len(result)}"

    def test_determine_best_feed_best_Q_is_list(self, mock_adm1_for_optimization: Mock) -> None:
        """
        Test that the best Q returned is a list.

        Args:
            mock_adm1_for_optimization: Mock ADM1 fixture.
        """
        # Fix feedstock mock to match how Simulator expects to call it
        from pyadm1.substrates.feedstock import Feedstock

        mock_feedstock = Mock(spec=Feedstock)
        mock_feedstock.get_substrate_feed_mixtures = Mock(
            return_value=[[15 + i, 10 + i, 0, 0, 0, 0, 0, 0, 0, 0] for i in range(13)]
        )

        mock_adm1_for_optimization.feedstock = mock_feedstock

        simulator = Simulator(mock_adm1_for_optimization)

        state_zero = [0.01] * 37
        Q = [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]
        Qch4sp = 900
        feeding_freq = 48

        with patch("scipy.integrate.solve_ivp") as mock_solve:
            mock_result = Mock()
            mock_result.y = np.array([[val] * 140 for val in state_zero])
            mock_solve.return_value = mock_result

            result = simulator.determine_best_feed_by_n_sims(state_zero, Q, Qch4sp, feeding_freq, n=13)

        # Best Q is at index 2
        best_Q = result[2]
        assert isinstance(best_Q, list), "Best Q should be a list"
        assert len(best_Q) == len(Q), "Best Q should have same length as input Q"

    def test_determine_best_feed_with_minimum_n(self, mock_adm1_for_optimization: Mock) -> None:
        """
        Test determineBestFeedbyNSims with minimum n=3.

        Args:
            mock_adm1_for_optimization: Mock ADM1 fixture.
        """
        # Fix feedstock mock to match how Simulator expects to call it
        from pyadm1.substrates.feedstock import Feedstock

        mock_feedstock = Mock(spec=Feedstock)
        mock_feedstock.get_substrate_feed_mixtures = Mock(
            return_value=[[15 + i, 10 + i, 0, 0, 0, 0, 0, 0, 0, 0] for i in range(3)]
        )

        mock_adm1_for_optimization.feedstock = mock_feedstock

        simulator = Simulator(mock_adm1_for_optimization)

        state_zero = [0.01] * 37
        Q = [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]
        Qch4sp = 900
        feeding_freq = 48

        with patch("scipy.integrate.solve_ivp") as mock_solve:
            mock_result = Mock()
            mock_result.y = np.array([[val] * 140 for val in state_zero])
            mock_solve.return_value = mock_result

            # Should work with n=3 (minimum)
            result = simulator.determine_best_feed_by_n_sims(state_zero, Q, Qch4sp, feeding_freq, n=3)

        assert result is not None, "Should work with n=3"

    def test_determine_best_feed_gas_flows_positive(self, mock_adm1_for_optimization: Mock) -> None:
        """
        Test that returned gas flows are positive.

        Args:
            mock_adm1_for_optimization: Mock ADM1 fixture.
        """
        # Fix feedstock mock to match how Simulator expects to call it
        from pyadm1.substrates.feedstock import Feedstock

        mock_feedstock = Mock(spec=Feedstock)
        mock_feedstock.get_substrate_feed_mixtures = Mock(
            return_value=[[15 + i, 10 + i, 0, 0, 0, 0, 0, 0, 0, 0] for i in range(5)]
        )

        mock_adm1_for_optimization.feedstock = mock_feedstock

        simulator = Simulator(mock_adm1_for_optimization)

        state_zero = [0.01] * 37
        Q = [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]
        Qch4sp = 900
        feeding_freq = 48

        with patch("scipy.integrate.solve_ivp") as mock_solve:
            mock_result = Mock()
            mock_result.y = np.array([[val] * 140 for val in state_zero])
            mock_solve.return_value = mock_result

            result = simulator.determine_best_feed_by_n_sims(state_zero, Q, Qch4sp, feeding_freq, n=5)

        # Check gas flows (indices 0, 1, 3, 4, 6, 7, 8, 9)
        q_gas_best = result[0]
        q_ch4_best = result[1]
        q_gas_initial = result[3]
        q_ch4_initial = result[4]

        assert q_gas_best >= 0, "Best biogas flow should be non-negative"
        assert q_ch4_best >= 0, "Best methane flow should be non-negative"
        assert q_gas_initial >= 0, "Initial biogas flow should be non-negative"
        assert q_ch4_initial >= 0, "Initial methane flow should be non-negative"


class TestSimulatorPrivateMethods:
    """Test suite for Simulator private methods."""

    @pytest.fixture
    def mock_adm1_basic(self) -> Mock:
        """
        Create a basic mock ADM1 object.

        Returns:
            Mock ADM1 object.
        """
        adm1 = Mock(spec=ADM1)
        adm1.V_liq = 1977
        adm1.createInfluent = Mock()
        adm1.ADM1_ODE = Mock(return_value=[0.01] * 37)
        adm1.calc_gas = Mock(return_value=(1500, 900, 600, 0.95))
        return adm1

    def test_simulate_uses_bdf_solver(self, mock_adm1_basic: Mock) -> None:
        """
        Test that _simulate uses BDF solver method.

        Args:
            mock_adm1_basic: Mock ADM1 fixture.
        """
        simulator = Simulator(mock_adm1_basic)

        state_zero = [0.01] * 37
        tstep = [0, 1]

        with patch("scipy.integrate.solve_ivp") as mock_solve:
            mock_result = Mock()
            mock_result.y = np.array([[val] * 20 for val in state_zero])
            mock_solve.return_value = mock_result

            simulator._simulate(tstep, state_zero)

            # Check that solve_ivp was called with method='BDF'
            call_kwargs = mock_solve.call_args[1]
            assert call_kwargs["method"] == "BDF", "Should use BDF solver"

    def test_simulate_uses_correct_time_eval(self, mock_adm1_basic: Mock) -> None:
        """
        Test that _simulate uses time evaluation with 0.05 step.

        Args:
            mock_adm1_basic: Mock ADM1 fixture.
        """
        simulator = Simulator(mock_adm1_basic)

        state_zero = [0.01] * 37
        tstep = [0, 1]

        with patch("scipy.integrate.solve_ivp") as mock_solve:
            mock_result = Mock()
            mock_result.y = np.array([[val] * 20 for val in state_zero])
            mock_solve.return_value = mock_result

            simulator._simulate(tstep, state_zero)

            call_kwargs = mock_solve.call_args[1]
            t_eval = call_kwargs.get("t_eval")

            assert t_eval is not None, "Should provide t_eval"
            # Check that step size is approximately 0.05
            if len(t_eval) > 1:
                step = t_eval[1] - t_eval[0]
                assert np.isclose(step, 0.05, rtol=1e-6), "Step size should be 0.05"

    def test_simulate_wosavinglaststate_returns_two_values(self, mock_adm1_basic: Mock) -> None:
        """
        Test that _simulate_wosavinglaststate returns biogas and methane flows.

        Args:
            mock_adm1_basic: Mock ADM1 fixture.
        """
        simulator = Simulator(mock_adm1_basic)

        state_zero = [0.01] * 37
        Q = [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]
        tstep = [0, 7]

        with patch("scipy.integrate.solve_ivp") as mock_solve:
            mock_result = Mock()
            mock_result.y = np.array([[val] * 140 for val in state_zero])
            mock_solve.return_value = mock_result

            q_gas, q_ch4 = simulator._simulate_without_saving_state(tstep, state_zero, Q)

        assert isinstance(q_gas, (int, float, np.ndarray)), "q_gas should be numeric"
        assert isinstance(q_ch4, (int, float, np.ndarray)), "q_ch4 should be numeric"

    def test_simulate_returnlaststate_returns_correct_length(self, mock_adm1_basic: Mock) -> None:
        """
        Test that _simulate_returnlaststate returns state with 37 elements.

        Args:
            mock_adm1_basic: Mock ADM1 fixture.
        """
        simulator = Simulator(mock_adm1_basic)

        state_zero = [0.01] * 37
        tstep = [0, 1]

        with patch("scipy.integrate.solve_ivp") as mock_solve:
            mock_result = Mock()
            mock_result.y = np.array([[val] * 20 for val in state_zero])
            mock_solve.return_value = mock_result

            result = simulator._simulate_and_return_final_state(tstep, state_zero)

        assert len(result) == 37, f"Should return 37 elements, got {len(result)}"


class TestSimulatorIntegration:
    """Integration tests for Simulator class."""

    @pytest.fixture
    def mock_adm1_full(self) -> Mock:
        """
        Create a fully configured mock ADM1 for integration testing.

        Returns:
            Mock ADM1 object with realistic behavior.
        """
        adm1 = Mock(spec=ADM1)
        adm1.V_liq = 1977
        adm1._V_gas = 304
        adm1._R = 0.08314
        adm1._T_ad = 308.15
        adm1._pext = 1.04

        # Mock methods with realistic returns
        adm1.createInfluent = Mock()
        adm1.print_params_at_current_state = Mock()

        # Simple ODE that returns small changes
        def mock_ode(t, state):
            return [0.001 * val for val in state]

        adm1.ADM1_ODE = mock_ode

        # Realistic gas calculation
        adm1.calc_gas = Mock(return_value=(1500.0, 900.0, 600.0, 0.95))

        return adm1

    def test_full_simulation_workflow(self, mock_adm1_full: Mock) -> None:
        """
        Test a complete simulation workflow.

        Args:
            mock_adm1_full: Mock ADM1 fixture.
        """
        simulator = Simulator(mock_adm1_full)

        # Set up simulation
        state_zero = [0.01] * 37

        # Run multiple time steps
        current_state = state_zero
        time_points = [0, 1, 2, 3]

        with patch("scipy.integrate.solve_ivp") as mock_solve:
            for i in range(len(time_points) - 1):
                # Create realistic solver result
                mock_result = Mock()
                n_points = 20
                mock_result.y = np.array([[current_state[j] * (1 + 0.001 * k) for k in range(n_points)] for j in range(37)])
                mock_solve.return_value = mock_result

                tstep = [time_points[i], time_points[i + 1]]
                current_state = simulator.simulate_AD_plant(tstep, current_state)

                assert len(current_state) == 37, "State should maintain 37 elements"
                assert all(np.isfinite(current_state)), "All state values should be finite"

    def test_optimization_finds_reasonable_feed(self, mock_adm1_full: Mock) -> None:
        """
        Test that optimization finds a feed rate.

        Args:
            mock_adm1_full: Mock ADM1 fixture.
        """
        # instead of making feedstock callable, create a mock that exposes the method directly
        from pyadm1.substrates.feedstock import Feedstock

        mock_feedstock = Mock(spec=Feedstock)
        mock_feedstock.get_substrate_feed_mixtures = Mock(
            return_value=[[15 + i, 10 + i, 0, 0, 0, 0, 0, 0, 0, 0] for i in range(5)]
        )
        mock_adm1_full.feedstock = mock_feedstock

        simulator = Simulator(mock_adm1_full)

        state_zero = [0.01] * 37
        Q = [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]
        Qch4sp = 900
        feeding_freq = 48

        with patch("scipy.integrate.solve_ivp") as mock_solve:
            # Mock solver with varying results
            def create_mock_result(state, n_points=140):
                return Mock(y=np.array([[state[j] * (1 + 0.001 * k) for k in range(n_points)] for j in range(37)]))

            mock_solve.return_value = create_mock_result(state_zero)

            result = simulator.determine_best_feed_by_n_sims(state_zero, Q, Qch4sp, feeding_freq, n=5)

            best_Q = result[2]

            # Check that best Q is reasonable (within ±5 of original)
            for i in range(len(Q)):
                if Q[i] > 0:
                    assert abs(best_Q[i] - Q[i]) <= 5, f"Best Q[{i}] should be within ±5 of original"
