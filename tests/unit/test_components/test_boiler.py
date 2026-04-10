# -*- coding: utf-8 -*-
"""Unit tests for the Boiler component."""

from pyadm1.components.energy.boiler import Boiler


class TestBoilerInitialization:
    """Constructor and attribute behavior."""

    def test_init_sets_defaults(self) -> None:
        boiler = Boiler("boiler_1")

        assert boiler.component_id == "boiler_1"
        assert boiler.P_th_nom == 200.0
        assert boiler.efficiency == 0.9
        assert boiler.component_type.value == "boiler"

    def test_init_accepts_custom_values_and_name(self) -> None:
        boiler = Boiler("boiler_2", P_th_nom=750.0, efficiency=0.85, name="Backup Boiler")

        assert boiler.P_th_nom == 750.0
        assert boiler.efficiency == 0.85
        assert boiler.name == "Backup Boiler"


class TestBoilerBehavior:
    """Boiler state and step behaviour."""

    def test_initialize_sets_tracking_state(self) -> None:
        boiler = Boiler("boiler_1")

        boiler.initialize({"ignored": True})

        assert boiler.state == {
            "energy_supplied": 0.0,
            "gas_consumed_total": 0.0,
            "ng_consumed_total": 0.0,
            "operating_hours": 0.0,
        }

    def test_step_returns_operating_outputs(self) -> None:
        boiler = Boiler("boiler_1")

        result = boiler.step(
            t=0.0,
            dt=1.0,
            inputs={"P_th_demand": 100.0, "Q_gas_available_m3_per_day": 500.0},
        )

        assert result["P_th_supplied"] == 100.0
        assert result["P_th_available"] == 100.0
        assert result["Q_gas_consumed_m3_per_day"] > 0.0
        assert result["Q_natural_gas_m3_per_day"] == 0.0
        assert result["is_running"] is True


class TestBoilerSerialization:
    """Serialization helpers."""

    def test_to_dict_returns_full_config(self) -> None:
        boiler = Boiler("boiler_1")

        result = boiler.to_dict()

        assert result["component_id"] == "boiler_1"
        assert result["component_type"] == "boiler"
        assert result["P_th_nom"] == 200.0
        assert result["efficiency"] == 0.9
        assert "state" in result

    def test_from_dict_recreates_instance_with_defaults(self) -> None:
        boiler = Boiler.from_dict({"component_id": "boiler_cfg"})

        assert isinstance(boiler, Boiler)
        assert boiler.component_id == "boiler_cfg"
        assert boiler.P_th_nom == 200.0
        assert boiler.efficiency == 0.9
