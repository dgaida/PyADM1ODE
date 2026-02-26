# -*- coding: utf-8 -*-
"""Unit tests for the Hydrolysis component stub."""

from unittest.mock import Mock

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
        assert hydrolysis.T_ad == 318.15
        assert hydrolysis.component_type.value == "digester"

    def test_init_accepts_custom_values_and_name(self, mock_feedstock: Mock) -> None:
        hydrolysis = Hydrolysis("hyd_2", mock_feedstock, V_liq=750.0, T_ad=320.15, name="Hyd Stage")

        assert hydrolysis.V_liq == 750.0
        assert hydrolysis.T_ad == 320.15
        assert hydrolysis.name == "Hyd Stage"


class TestHydrolysisBehavior:
    """Stub behavior methods should return simple defaults."""

    def test_initialize_sets_empty_state_even_with_initial_state(self, mock_feedstock: Mock) -> None:
        hydrolysis = Hydrolysis("hyd_1", mock_feedstock)

        hydrolysis.initialize({"ignored": "value"})

        assert hydrolysis.state == {}

    def test_step_returns_empty_dict(self, mock_feedstock: Mock) -> None:
        hydrolysis = Hydrolysis("hyd_1", mock_feedstock)

        result = hydrolysis.step(t=0.0, dt=1.0, inputs={"anything": 1})

        assert result == {}


class TestHydrolysisSerialization:
    """Serialization helpers for the stub component."""

    def test_to_dict_returns_minimal_component_config(self, mock_feedstock: Mock) -> None:
        hydrolysis = Hydrolysis("hyd_1", mock_feedstock)

        result = hydrolysis.to_dict()

        assert result == {"component_id": "hyd_1", "component_type": "digester"}

    def test_from_dict_recreates_instance_with_defaults(self, mock_feedstock: Mock) -> None:
        hydrolysis = Hydrolysis.from_dict({"component_id": "hyd_from_cfg"}, mock_feedstock)

        assert isinstance(hydrolysis, Hydrolysis)
        assert hydrolysis.component_id == "hyd_from_cfg"
        assert hydrolysis.feedstock is mock_feedstock
        assert hydrolysis.V_liq == 500.0
        assert hydrolysis.T_ad == 318.15
