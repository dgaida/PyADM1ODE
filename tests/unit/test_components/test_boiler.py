# -*- coding: utf-8 -*-
"""Unit tests for the Boiler component stub."""

from pyadm1.components.energy.boiler import Boiler


class TestBoilerInitialization:
    """Constructor and attribute behavior."""

    def test_init_sets_defaults(self) -> None:
        boiler = Boiler("boiler_1")

        assert boiler.component_id == "boiler_1"
        assert boiler.P_th_nom == 500.0
        assert boiler.efficiency == 0.9
        assert boiler.component_type.value == "boiler"

    def test_init_accepts_custom_values_and_name(self) -> None:
        boiler = Boiler("boiler_2", P_th_nom=750.0, efficiency=0.85, name="Backup Boiler")

        assert boiler.P_th_nom == 750.0
        assert boiler.efficiency == 0.85
        assert boiler.name == "Backup Boiler"


class TestBoilerBehavior:
    """Stub methods should return simple defaults."""

    def test_initialize_sets_empty_state(self) -> None:
        boiler = Boiler("boiler_1")

        boiler.initialize({"ignored": True})

        assert boiler.state == {}

    def test_step_returns_empty_dict(self) -> None:
        boiler = Boiler("boiler_1")

        result = boiler.step(t=0.0, dt=1.0, inputs={"heat_demand_kw": 100.0})

        assert result == {}


class TestBoilerSerialization:
    """Serialization helpers."""

    def test_to_dict_returns_minimal_config(self) -> None:
        boiler = Boiler("boiler_1")

        assert boiler.to_dict() == {"component_id": "boiler_1", "component_type": "boiler"}

    def test_from_dict_recreates_instance_with_defaults(self) -> None:
        boiler = Boiler.from_dict({"component_id": "boiler_cfg"})

        assert isinstance(boiler, Boiler)
        assert boiler.component_id == "boiler_cfg"
        assert boiler.P_th_nom == 500.0
        assert boiler.efficiency == 0.9
