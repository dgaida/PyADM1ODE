# -*- coding: utf-8 -*-
"""Unit tests for the Separator component."""

import pytest

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

    def test_from_dict_restores_cumulative_state(self) -> None:
        separator = Separator.from_dict(
            {
                "component_id": "sep_cfg",
                "separation_efficiency": 0.7,
                "state": {
                    "total_solid_mass": 123.4,
                    "total_liquid_vol": 45.6,
                    "energy_consumed": 7.8,
                },
            }
        )

        assert separator.total_solid_mass == 123.4
        assert separator.total_liquid_vol == 45.6
        assert separator.energy_consumed == 7.8


class TestSeparatorEdgeCases:
    """Edge branches: zero flow, missing TS, ADM1-state estimation."""

    def test_zero_flow_returns_unchanged_outputs(self) -> None:
        separator = Separator("sep_1")
        baseline = dict(separator.outputs_data)

        result = separator.step(t=0.0, dt=1.0, inputs={"Q_in": 0.0})

        assert result == baseline

    def test_missing_ts_with_adm1_state_uses_estimate(self) -> None:
        # Provide a 41-element ADM1 state with non-zero particulate pools so
        # _estimate_ts_from_adm1 produces a positive TS estimate.  The
        # particulate indices the helper reads are 12..24 inclusive.
        adm1_state = [0.01] * 41
        adm1_state[12:25] = [10.0] * 13  # all particulate COD pools the helper sums

        separator = Separator("sep_1")
        result = separator.step(
            t=0.0,
            dt=1.0,
            inputs={"Q_in": 5.0, "state_out": adm1_state},
        )

        assert result["Q_solid"] > 0.0
        # Default ``screw_press`` ts_solid_target = 230 kg/m³ (KTBL 2013).
        assert result["TS_solid"] == 230.0

    def test_missing_ts_without_adm1_state_uses_default(self) -> None:
        separator = Separator("sep_1")
        # No TS_in, no state_out → fallback to 40 kg/m3
        result = separator.step(t=0.0, dt=1.0, inputs={"Q_in": 5.0})

        # TS_in defaulted to 40 kg/m3 → mass split is consistent with that.
        m_TS_total = 5.0 * 40.0  # 200 kg/d
        m_TS_solid_expected = m_TS_total * separator.separation_efficiency
        assert result["Q_solid"] * result["TS_solid"] == pytest.approx(m_TS_solid_expected, rel=1e-6)


class TestEstimateTSFromADM1:
    """Direct unit tests for _estimate_ts_from_adm1 helper."""

    def test_estimate_returns_positive_value_for_realistic_state(self) -> None:
        from pyadm1.components.biological.separator import Separator

        state = [0.01] * 41
        state[12:21] = [5.0] * 9  # particulate COD pools

        ts = Separator._estimate_ts_from_adm1(state)
        assert ts > 0.0

    def test_estimate_falls_back_to_default_on_invalid_state(self) -> None:
        from pyadm1.components.biological.separator import Separator

        # The helper reads particulate indices 12..24, so the state must be at
        # least 25 elements long for any of those reads to actually execute.
        # ``float(None)`` raises TypeError, triggering the except branch.
        invalid_state = [None] * 25
        ts = Separator._estimate_ts_from_adm1(invalid_state)
        assert ts == 40.0  # fallback default
