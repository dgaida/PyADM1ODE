# -*- coding: utf-8 -*-
"""Unit tests for the Separator component stub."""

from pyadm1.components.biological.separator import Separator


class TestSeparatorInitialization:
    """Constructor and attribute behavior."""

    def test_init_sets_defaults(self) -> None:
        separator = Separator("sep_1")

        assert separator.component_id == "sep_1"
        assert separator.separation_efficiency == 0.95
        assert separator.component_type.value == "separator"

    def test_init_accepts_custom_efficiency_and_name(self) -> None:
        separator = Separator("sep_2", separation_efficiency=0.8, name="Digestate Separator")

        assert separator.separation_efficiency == 0.8
        assert separator.name == "Digestate Separator"


class TestSeparatorBehavior:
    """Stub methods should return simple defaults."""

    def test_initialize_sets_empty_state(self) -> None:
        separator = Separator("sep_1")

        separator.initialize({"ignored": True})

        assert separator.state == {}

    def test_step_returns_empty_dict(self) -> None:
        separator = Separator("sep_1")

        result = separator.step(t=0.0, dt=1.0, inputs={"digestate_in": 5.0})

        assert result == {}


class TestSeparatorSerialization:
    """Serialization helpers."""

    def test_to_dict_returns_minimal_config(self) -> None:
        separator = Separator("sep_1")

        assert separator.to_dict() == {
            "component_id": "sep_1",
            "component_type": "separator",
        }

    def test_from_dict_recreates_instance_with_defaults(self) -> None:
        separator = Separator.from_dict({"component_id": "sep_cfg"})

        assert isinstance(separator, Separator)
        assert separator.component_id == "sep_cfg"
        assert separator.separation_efficiency == 0.95
