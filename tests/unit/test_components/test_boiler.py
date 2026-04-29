# -*- coding: utf-8 -*-
"""Unit tests for the Boiler component."""

import pytest

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

    def test_from_dict_restores_cumulative_state(self) -> None:
        boiler = Boiler.from_dict(
            {
                "component_id": "boiler_cfg",
                "P_th_nom": 150.0,
                "fuel_type": "biogas",
                "state": {
                    "energy_supplied": 1234.5,
                    "gas_consumed_total": 678.9,
                    "ng_consumed_total": 0.0,
                    "operating_hours": 12.3,
                },
            }
        )

        assert boiler.energy_supplied == 1234.5
        assert boiler.gas_consumed_total == 678.9
        assert boiler.operating_hours == 12.3


class TestBoilerInvalidConfiguration:
    """Constructor validation."""

    def test_invalid_fuel_type_raises(self) -> None:
        with pytest.raises(ValueError, match="fuel_type must be"):
            Boiler("boiler_bad", fuel_type="coal")


class TestBoilerOffBranch:
    """Off-state behaviour (no demand, or explicitly disabled)."""

    def test_zero_demand_returns_off_outputs(self) -> None:
        boiler = Boiler("boiler_1")

        result = boiler.step(t=0.0, dt=1.0, inputs={"P_th_demand": 0.0})

        assert result["is_running"] is False
        assert result["P_th_supplied"] == 0.0
        assert result["Q_gas_consumed_m3_per_day"] == 0.0
        assert result["Q_natural_gas_m3_per_day"] == 0.0

    def test_disabled_boiler_returns_off_outputs_even_with_demand(self) -> None:
        boiler = Boiler("boiler_1")

        result = boiler.step(
            t=0.0,
            dt=1.0,
            inputs={"P_th_demand": 100.0, "enable": False},
        )

        assert result["is_running"] is False
        assert result["P_th_supplied"] == 0.0


class TestBoilerFuelModes:
    """Fuel allocation by ``fuel_type``."""

    def test_natural_gas_only_does_not_consume_biogas(self) -> None:
        boiler = Boiler("boiler_ng", fuel_type="natural_gas")

        result = boiler.step(
            t=0.0,
            dt=1.0,
            inputs={"P_th_demand": 100.0, "Q_gas_available_m3_per_day": 500.0},
        )

        assert result["Q_gas_consumed_m3_per_day"] == 0.0
        assert result["Q_natural_gas_m3_per_day"] > 0.0

    def test_biogas_only_does_not_consume_ng(self) -> None:
        boiler = Boiler("boiler_bg", fuel_type="biogas")

        result = boiler.step(
            t=0.0,
            dt=1.0,
            inputs={"P_th_demand": 100.0, "Q_gas_available_m3_per_day": 500.0},
        )

        assert result["Q_gas_consumed_m3_per_day"] > 0.0
        assert result["Q_natural_gas_m3_per_day"] == 0.0

    def test_dual_fuel_supplements_with_ng_when_biogas_runs_short(self) -> None:
        # Demand high, only a tiny amount of biogas available → most heat must
        # come from natural gas, but some biogas should be burned first.
        boiler = Boiler("boiler_dual", P_th_nom=200.0, fuel_type="dual")

        result = boiler.step(
            t=0.0,
            dt=1.0,
            inputs={"P_th_demand": 200.0, "Q_gas_available_m3_per_day": 50.0},
        )

        assert result["Q_gas_consumed_m3_per_day"] == pytest.approx(50.0)
        assert result["Q_natural_gas_m3_per_day"] > 0.0
        assert result["P_th_supplied"] == pytest.approx(200.0)

    def test_dual_fuel_uses_only_biogas_when_supply_is_sufficient(self) -> None:
        boiler = Boiler("boiler_dual", P_th_nom=100.0, fuel_type="dual")

        result = boiler.step(
            t=0.0,
            dt=1.0,
            inputs={"P_th_demand": 100.0, "Q_gas_available_m3_per_day": 1.0e6},
        )

        assert result["Q_natural_gas_m3_per_day"] == 0.0
        assert result["Q_gas_consumed_m3_per_day"] > 0.0
