# -*- coding: utf-8 -*-
"""Unit tests for the Separator component."""

from pyadm1.components.biological.separator import Separator


class TestSeparatorInitialization:
    """Constructor and attribute behavior."""

    def test_init_sets_defaults(self) -> None:
        separator = Separator("sep_1")

        assert separator.component_id == "sep_1"
        assert separator.separation_efficiency == 0.6
        assert separator.component_type.value == "separator"

    def test_init_accepts_custom_efficiency_and_name(self) -> None:
        separator = Separator("sep_2", separation_efficiency=0.8, name="Digestate Separator")

        assert separator.separation_efficiency == 0.8
        assert separator.name == "Digestate Separator"


class TestSeparatorBehavior:
    """Separator state and step behaviour."""

    def test_initialize_sets_tracking_state(self) -> None:
        separator = Separator("sep_1")

        separator.initialize({"ignored": True})

        assert separator.state == {
            "total_solid_mass": 0.0,
            "total_liquid_vol": 0.0,
            "energy_consumed": 0.0,
        }

    def test_step_returns_separation_outputs(self) -> None:
        separator = Separator("sep_1")

        result = separator.step(t=0.0, dt=1.0, inputs={"Q_in": 5.0, "TS_in": 40.0})

        assert result["Q_liquid"] > 0.0
        assert result["Q_solid"] > 0.0
        assert result["P_consumed"] > 0.0
        assert result["separation_efficiency"] == 0.6


class TestSeparatorSerialization:
    """Serialization helpers."""

    def test_to_dict_returns_config(self) -> None:
        separator = Separator("sep_1")

        result = separator.to_dict()

        assert result["component_id"] == "sep_1"
        assert result["component_type"] == "separator"
        assert result["separator_type"] == "screw_press"
        assert result["separation_efficiency"] == 0.6
        assert "state" in result

    def test_from_dict_recreates_instance_with_defaults(self) -> None:
        separator = Separator.from_dict({"component_id": "sep_cfg"})

        assert isinstance(separator, Separator)
        assert separator.component_id == "sep_cfg"
        assert separator.separation_efficiency == 0.6
