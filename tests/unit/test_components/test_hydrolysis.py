# -*- coding: utf-8 -*-
"""Unit tests for the Hydrolysis component."""

from unittest.mock import Mock, patch

import pytest

from pyadm1.components.biological.hydrolysis import Hydrolysis
from pyadm1.substrates.feedstock import Feedstock


@pytest.fixture
def mock_feedstock() -> Mock:
    """Create a lightweight feedstock mock for Hydrolysis tests."""
    feedstock = Mock(spec=Feedstock)
    feedstock.mySubstrates = Mock()
    return feedstock


class TestHydrolysisInitialization:
    """Constructor and basic properties."""

    def test_init_sets_attributes_and_defaults(self, mock_feedstock: Mock) -> None:
        hydrolysis = Hydrolysis("hyd_1", mock_feedstock)

        assert hydrolysis.component_id == "hyd_1"
        assert hydrolysis.feedstock is mock_feedstock
        assert hydrolysis.V_liq == 500.0
        assert hydrolysis.T_ad == 328.15
        assert hydrolysis.component_type.value == "digester"

    def test_init_accepts_custom_values_and_name(self, mock_feedstock: Mock) -> None:
        hydrolysis = Hydrolysis("hyd_2", mock_feedstock, V_liq=750.0, T_ad=320.15, name="Hyd Stage")

        assert hydrolysis.V_liq == 750.0
        assert hydrolysis.T_ad == 320.15
        assert hydrolysis.name == "Hyd Stage"


class TestHydrolysisBehavior:
    """Hydrolysis state and step behaviour."""

    def test_initialize_sets_process_state_even_with_initial_state(self, mock_feedstock: Mock) -> None:
        hydrolysis = Hydrolysis("hyd_1", mock_feedstock)

        hydrolysis.initialize({"ignored": "value"})

        assert hydrolysis.state["adm1_state"] == []
        assert hydrolysis.state["Q_gas"] == 0.0
        assert hydrolysis.state["pH"] == 7.0

    def test_step_returns_process_outputs(self, mock_feedstock: Mock) -> None:
        hydrolysis = Hydrolysis("hyd_1", mock_feedstock)
        hydrolysis.initialize({"Q_substrates": [1.0] * 10})

        simulated_state = [0.02] * 37
        simulated_state[33:37] = [0.1, 0.2, 0.3, 1.0]

        with patch.object(
            hydrolysis.adm1,
            "create_influent",
            side_effect=lambda q_substrates, _: setattr(
                hydrolysis.adm1,
                "_state_input",
                [0.01] * 33 + [float(sum(q_substrates))],
            ),
        ):
            with patch.object(hydrolysis.adm1, "calc_gas", return_value=(5.0, 3.0, 2.0, 0.0, 1.01)):
                with patch.object(hydrolysis.simulator, "simulate_AD_plant", return_value=simulated_state):
                    with patch.object(
                        hydrolysis.gas_storage,
                        "step",
                        return_value={
                            "stored_volume_m3": 2.0,
                            "pressure_bar": 1.0,
                            "vented_volume_m3": 0.0,
                            "Q_gas_supplied_m3_per_day": 0.0,
                        },
                    ):
                        with patch("pyadm1.components.biological.hydrolysis.ADMstate") as admstate_mock:
                            admstate_mock.calcPHOfADMstate.return_value = 7.2
                            admstate_mock.calcVFAOfADMstate.return_value = Mock(Value=0.8)
                            admstate_mock.calcTACOfADMstate.return_value = Mock(Value=3.5)
                            result = hydrolysis.step(
                                t=0.0,
                                dt=1.0 / 24.0,
                                inputs={"Q_substrates": [1.0] * 10},
                            )

        assert result["Q_out"] == pytest.approx(10.0)
        assert result["Q_gas"] == 5.0
        assert result["pH"] == 7.2
        assert result["gas_storage"]["stored_volume_m3"] == 2.0


class TestHydrolysisSerialization:
    """Serialization helpers."""

    def test_to_dict_returns_component_config(self, mock_feedstock: Mock) -> None:
        hydrolysis = Hydrolysis("hyd_1", mock_feedstock)

        result = hydrolysis.to_dict()

        assert result["component_id"] == "hyd_1"
        assert result["component_type"] == "digester"
        assert result["V_liq"] == 500.0
        assert result["T_ad"] == 328.15
        assert "state" in result

    def test_from_dict_recreates_instance_with_defaults(self, mock_feedstock: Mock) -> None:
        hydrolysis = Hydrolysis.from_dict({"component_id": "hyd_from_cfg"}, mock_feedstock)

        assert isinstance(hydrolysis, Hydrolysis)
        assert hydrolysis.component_id == "hyd_from_cfg"
        assert hydrolysis.feedstock is mock_feedstock
        assert hydrolysis.V_liq == 500.0
        assert hydrolysis.T_ad == 328.15
