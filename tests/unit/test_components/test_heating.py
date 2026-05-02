# -*- coding: utf-8 -*-
"""
Unit tests for the HeatingSystem component and the pure-Python sensible-heat
helper :func:`_calc_process_heat_kw`.
"""

import math
from types import SimpleNamespace

import pyadm1.components.energy.heating as heating_module
from pyadm1.components.energy.heating import HeatingSystem


class TestHeatingInitialization:
    """Test suite for HeatingSystem initialization."""

    def test_heating_initialization_sets_defaults(self) -> None:
        heating = HeatingSystem("heat_1")
        assert heating.target_temperature == 308.15
        assert heating.heat_loss_coefficient == 0.5

    def test_initialize_sets_state_and_outputs(self) -> None:
        heating = HeatingSystem("heat_1")
        heating.initialize()

        assert "Q_heat_demand" in heating.state
        assert "Q_heat_supplied" in heating.state
        assert "energy_consumed" in heating.state
        assert "Q_heat_supplied" in heating.outputs_data
        assert "P_th_used" in heating.outputs_data
        assert "P_aux_heat" in heating.outputs_data


class TestHeatingStep:
    """Test suite for HeatingSystem step method."""

    def test_step_basic_heat_balance(self, monkeypatch) -> None:
        monkeypatch.setattr(heating_module, "_calc_process_heat_kw", lambda *_: 0.0)
        heating = HeatingSystem("heat_1", heat_loss_coefficient=0.5)
        heating.initialize()

        inputs = {
            "T_digester": 308.15,
            "T_ambient": 288.15,
            "P_th_available": 6.0,
        }

        result = heating.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert math.isclose(result["Q_heat_supplied"], 10.0, rel_tol=1e-6)
        assert math.isclose(result["P_th_used"], 6.0, rel_tol=1e-6)
        assert math.isclose(result["P_aux_heat"], 4.0, rel_tol=1e-6)
        assert math.isclose(heating.state["energy_consumed"], 4.0, rel_tol=1e-6)

    def test_step_zero_temperature_difference(self, monkeypatch) -> None:
        monkeypatch.setattr(heating_module, "_calc_process_heat_kw", lambda *_: 0.0)
        heating = HeatingSystem("heat_1", heat_loss_coefficient=0.5)
        heating.initialize()

        inputs = {
            "T_digester": 300.0,
            "T_ambient": 300.0,
            "P_th_available": 50.0,
        }

        result = heating.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert math.isclose(result["Q_heat_supplied"], 0.0, rel_tol=1e-6)
        assert math.isclose(result["P_th_used"], 0.0, rel_tol=1e-6)
        assert math.isclose(result["P_aux_heat"], 0.0, rel_tol=1e-6)

    def test_step_chp_covers_demand(self, monkeypatch) -> None:
        monkeypatch.setattr(heating_module, "_calc_process_heat_kw", lambda *_: 0.0)
        heating = HeatingSystem("heat_1", heat_loss_coefficient=0.5)
        heating.initialize()

        inputs = {
            "T_digester": 308.15,
            "T_ambient": 288.15,
            "P_th_available": 20.0,
        }

        result = heating.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert math.isclose(result["P_th_used"], 10.0, rel_tol=1e-6)
        assert math.isclose(result["P_aux_heat"], 0.0, rel_tol=1e-6)

    def test_step_includes_process_heat(self, monkeypatch) -> None:
        monkeypatch.setattr(heating_module, "_calc_process_heat_kw", lambda *_: 3.5)
        heating = HeatingSystem("heat_1", target_temperature=308.15, heat_loss_coefficient=0.5)
        heating.initialize()

        inputs = {
            "T_digester": 308.15,
            "T_ambient": 288.15,
            "P_th_available": 0.0,
            "Q_substrates": [10.0] + [0.0] * 9,
        }

        result = heating.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)
        # Q_loss=10.0 and Q_process=3.5
        assert math.isclose(heating.state["Q_heat_demand"], 13.5, rel_tol=1e-6)
        assert math.isclose(result["P_aux_heat"], 13.5, rel_tol=1e-6)

    def test_step_defaults_used_when_inputs_missing(self, monkeypatch) -> None:
        monkeypatch.setattr(heating_module, "_calc_process_heat_kw", lambda *_: 0.0)
        heating = HeatingSystem("heat_1", target_temperature=300.0, heat_loss_coefficient=0.5)
        heating.initialize()

        # T_digester defaults to target_temperature, T_ambient defaults to 288.15
        result = heating.step(t=0.0, dt=1.0 / 24.0, inputs={})
        expected_q = 0.5 * (300.0 - 288.15)

        assert math.isclose(heating.state["Q_heat_demand"], expected_q, rel_tol=1e-6)
        assert math.isclose(result["P_th_used"], 0.0, rel_tol=1e-6)
        assert math.isclose(result["P_aux_heat"], expected_q, rel_tol=1e-6)


class TestHeatingSerialization:
    """Test serialization and deserialization paths."""

    def test_to_dict_contains_expected_fields(self) -> None:
        heating = HeatingSystem("heat_1", target_temperature=310.0, heat_loss_coefficient=0.7, name="H")
        d = heating.to_dict()

        assert d["component_id"] == "heat_1"
        assert d["name"] == "H"
        assert d["target_temperature"] == 310.0
        assert d["heat_loss_coefficient"] == 0.7
        assert "inputs" in d
        assert "outputs" in d
        assert "state" in d

    def test_from_dict_with_state_calls_initialize_with_state(self) -> None:
        cfg = {
            "component_id": "heat_1",
            "target_temperature": 311.0,
            "heat_loss_coefficient": 0.9,
            "name": "Heat",
            "inputs": ["a"],
            "outputs": ["b"],
            "state": {"Q_heat_demand": 1.0},
        }

        called_with = []
        original_initialize = HeatingSystem.initialize

        def fake_initialize(self, initial_state=None):  # noqa: ANN001
            called_with.append(initial_state)
            original_initialize(self, initial_state)

        try:
            HeatingSystem.initialize = fake_initialize
            heating = HeatingSystem.from_dict(cfg)
        finally:
            HeatingSystem.initialize = original_initialize

        assert heating.target_temperature == 311.0
        assert heating.heat_loss_coefficient == 0.9
        assert heating.inputs == ["a"]
        assert heating.outputs == ["b"]
        assert {"Q_heat_demand": 1.0} in called_with

    def test_from_dict_without_state_calls_default_initialize(self) -> None:
        cfg = {
            "component_id": "heat_1",
            "target_temperature": 311.0,
            "heat_loss_coefficient": 0.9,
        }
        heating = HeatingSystem.from_dict(cfg)
        assert "Q_heat_demand" in heating.state
        assert heating.state["Q_heat_demand"] == 0.0


def _stub_substrate(
    *,
    TS: float = 100.0,
    fRF: float = 0.20,
    fRP: float = 0.15,
    fRFe: float = 0.05,
    fRA: float = 0.10,
    FFS: float = 10.0,
):
    """SubstrateParams-shaped stub with just the fields _substrate_cp reads."""
    return SimpleNamespace(TS=TS, fRF=fRF, fRP=fRP, fRFe=fRFe, fRA=fRA, FFS=FFS)


def _stub_feedstock(densities, substrates):
    """Feedstock-shaped stub exposing _densities and _subs."""
    return SimpleNamespace(_densities=list(densities), _subs=list(substrates))


class TestSubstrateCp:
    """Component-weighted specific heat capacity helper."""

    def test_pure_water_substrate_returns_water_cp(self) -> None:
        s = _stub_substrate(TS=0.0, fRF=0.0, fRP=0.0, fRFe=0.0, fRA=0.0, FFS=0.0)
        assert math.isclose(heating_module._substrate_cp(s), heating_module._CP_H2O, rel_tol=1e-9)

    def test_known_composition_matches_choi_okos_sum(self) -> None:
        s = _stub_substrate()
        # f_TS=0.10, f_fiber=0.02, f_protein=0.015, f_lipid=0.005, f_ash=0.01,
        # f_NFE=0.05, f_CH=0.07, f_AC=0.01, f_H2O=0.89
        expected = (
            0.07 * heating_module._CP_CH
            + 0.015 * heating_module._CP_PR
            + 0.005 * heating_module._CP_LI
            + 0.01 * heating_module._CP_MI
            + 0.01 * heating_module._CP_AC
            + 0.89 * heating_module._CP_H2O
        )
        assert math.isclose(heating_module._substrate_cp(s), expected, rel_tol=1e-9)


class TestCalcProcessHeatKw:
    """Pure-Python sensible-heat calculation."""

    def test_returns_zero_for_empty_q(self) -> None:
        fs = _stub_feedstock([1000.0], [_stub_substrate()])
        assert heating_module._calc_process_heat_kw([], fs, 308.15, 288.15) == 0.0

    def test_returns_zero_when_feedstock_missing(self) -> None:
        assert heating_module._calc_process_heat_kw([10.0], None, 308.15, 288.15) == 0.0

    def test_returns_zero_when_delta_t_non_positive(self) -> None:
        fs = _stub_feedstock([1000.0], [_stub_substrate()])
        assert heating_module._calc_process_heat_kw([10.0], fs, 288.15, 308.15) == 0.0
        assert heating_module._calc_process_heat_kw([10.0], fs, 288.15, 288.15) == 0.0

    def test_returns_zero_when_feedstock_has_no_densities(self) -> None:
        fs = SimpleNamespace(_densities=[], _subs=[])
        assert heating_module._calc_process_heat_kw([10.0], fs, 308.15, 288.15) == 0.0

    def test_skips_substrates_with_zero_flow(self) -> None:
        fs = _stub_feedstock([1000.0, 1000.0], [_stub_substrate(), _stub_substrate()])
        only_first = heating_module._calc_process_heat_kw([5.0, 0.0], fs, 308.15, 288.15)
        both = heating_module._calc_process_heat_kw([5.0, 5.0], fs, 308.15, 288.15)
        assert only_first > 0.0
        assert math.isclose(both, 2.0 * only_first, rel_tol=1e-9)

    def test_computes_expected_sensible_heat_for_known_input(self) -> None:
        # Single substrate, ρ=1000 kg/m³, Q=10 m³/d, ΔT=20 K, c_p from stub composition.
        s = _stub_substrate()
        cp = heating_module._substrate_cp(s)  # ≈ 3.901 kJ/(kg·K)
        fs = _stub_feedstock([1000.0], [s])

        out = heating_module._calc_process_heat_kw([10.0], fs, 308.15, 288.15)
        expected = 10.0 * 1000.0 * cp * 20.0 / 86400.0
        assert math.isclose(out, expected, rel_tol=1e-9)

    def test_truncates_to_min_length_of_inputs(self) -> None:
        # Q has 3 entries, feedstock has 2 — only the first two should contribute.
        fs = _stub_feedstock([1000.0, 1000.0], [_stub_substrate(), _stub_substrate()])
        truncated = heating_module._calc_process_heat_kw([5.0, 5.0, 5.0], fs, 308.15, 288.15)
        full = heating_module._calc_process_heat_kw([5.0, 5.0], fs, 308.15, 288.15)
        assert math.isclose(truncated, full, rel_tol=1e-9)

    def test_step_invokes_calc_with_feedstock_and_t_ambient(self) -> None:
        s = _stub_substrate()
        fs = _stub_feedstock([1000.0], [s])
        heating = HeatingSystem(
            "heat_1",
            target_temperature=308.15,
            heat_loss_coefficient=0.0,  # isolate the process-heat term
            feedstock=fs,
        )
        result = heating.step(
            t=0.0,
            dt=1.0 / 24.0,
            inputs={
                "T_digester": 308.15,
                "T_ambient": 288.15,
                "P_th_available": 0.0,
                "Q_substrates": [10.0],
            },
        )
        cp = heating_module._substrate_cp(s)
        expected = 10.0 * 1000.0 * cp * 20.0 / 86400.0
        assert math.isclose(result["P_aux_heat"], expected, rel_tol=1e-9)
