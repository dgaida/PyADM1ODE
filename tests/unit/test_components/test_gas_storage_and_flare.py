# tests/unit/test_gas_storage_and_flare.py
import math

import pytest

from pyadm1.components.energy.gas_storage import GasStorage
from pyadm1.components.energy.flare import Flare


def test_gas_storage_basic_charge_and_vent():
    """GasStorage stores inflow and vents overflow above capacity."""
    gs = GasStorage(
        "test_stor",
        storage_type="membrane",
        capacity_m3=100.0,
        p_min_bar=0.95,
        p_max_bar=1.05,
        initial_fill_fraction=0.0,
    )
    gs.initialize()

    # send more gas than capacity in one day -> some must be vented
    out = gs.step(t=0.0, dt=1.0, inputs={"Q_gas_in_m3_per_day": 200.0})
    assert out["stored_volume_m3"] <= 100.0 + 1e-6
    assert out["vented_volume_m3"] > 0.0
    assert math.isclose(out["utilization"], out["stored_volume_m3"] / 100.0, rel_tol=1e-6)


def test_gas_storage_supply_limits_and_setpoint():
    """GasStorage restricts outflow when setpoint requests higher pressure."""
    gs = GasStorage(
        "s2",
        storage_type="membrane",
        capacity_m3=50.0,
        initial_fill_fraction=0.5,
        p_min_bar=0.95,
        p_max_bar=1.05,
    )
    gs.initialize()
    # request large outflow with a high setpoint that requests charging => outflow restricted
    gs.outputs_data.copy()
    res = gs.step(t=0.0, dt=1.0, inputs={"Q_gas_out_m3_per_day": 1000.0, "set_pressure": 1.05})
    assert res["Q_gas_supplied_m3_per_day"] >= 0.0
    # after request with setpoint higher than current pressure, supply should be less than requested
    assert res["Q_gas_supplied_m3_per_day"] < 1000.0


def test_gas_storage_rejects_invalid_storage_type():
    with pytest.raises(ValueError, match="storage_type must be"):
        GasStorage("bad", storage_type="invalid")


def test_gas_storage_initialize_restores_state_and_clamps_volume():
    gs = GasStorage("restore", storage_type="dome", capacity_m3=20.0, initial_fill_fraction=0.0)

    gs.initialize({"stored_volume_m3": 50.0, "pressure_setpoint_bar": 1.02})

    assert gs.stored_volume_m3 == 20.0
    assert gs.pressure_setpoint_bar == 1.02
    assert gs.outputs_data["stored_volume_m3"] == 20.0


def test_gas_storage_compressed_pressure_estimation_uses_nonlinear_branch():
    gs = GasStorage(
        "c1",
        storage_type="compressed",
        capacity_m3=100.0,
        p_min_bar=5.0,
        p_max_bar=200.0,
        initial_fill_fraction=0.5,
    )

    pressure = gs._estimate_pressure_bar()

    expected = 5.0 + (0.5**2) * (200.0 - 5.0)
    assert math.isclose(pressure, expected, rel_tol=1e-9)


def test_gas_storage_invalid_set_pressure_is_ignored_and_printed(
    capsys: pytest.CaptureFixture[str],
):
    gs = GasStorage("sp_invalid", capacity_m3=10.0, initial_fill_fraction=0.2)
    gs.pressure_setpoint_bar = 1.0

    out = gs.step(t=0.0, dt=1.0, inputs={"set_pressure": object()})

    assert out["pressure_setpoint_bar"] == 1.0
    captured = capsys.readouterr().out
    assert "float() argument" in captured or "must be" in captured


def test_gas_storage_overflow_counts_as_vented_even_when_vent_flag_false():
    gs = GasStorage("no_vent_flag", capacity_m3=10.0, initial_fill_fraction=0.9)

    out = gs.step(t=0.0, dt=1.0, inputs={"Q_gas_in_m3_per_day": 5.0, "vent_to_flare": False})

    assert out["vented_volume_m3"] == pytest.approx(4.0)
    assert out["cumulative_vented_m3"] == pytest.approx(4.0)
    assert out["stored_volume_m3"] == pytest.approx(10.0)


def test_gas_storage_step_handles_invalid_low_pressure_span_for_reserve_logic():
    gs = GasStorage(
        "flat_lp",
        storage_type="membrane",
        capacity_m3=10.0,
        p_min_bar=1.01325,
        p_max_bar=1.01325,
        initial_fill_fraction=1.0,
    )

    out = gs.step(t=0.0, dt=1.0, inputs={"Q_gas_out_m3_per_day": 100.0})

    assert out["Q_gas_supplied_m3_per_day"] == pytest.approx(10.0)
    assert out["stored_volume_m3"] == pytest.approx(0.0)


def test_gas_storage_step_handles_compressed_reserve_logic_branches():
    gs = GasStorage(
        "comp_reserve",
        storage_type="compressed",
        capacity_m3=100.0,
        p_min_bar=5.0,
        p_max_bar=50.0,
        initial_fill_fraction=1.0,
    )

    out = gs.step(t=0.0, dt=1.0, inputs={"Q_gas_out_m3_per_day": 200.0})
    # compressed branch keeps a small reserve fraction (1%)
    assert out["stored_volume_m3"] == pytest.approx(1.0)
    assert out["Q_gas_supplied_m3_per_day"] == pytest.approx(99.0)

    gs_flat = GasStorage(
        "comp_flat",
        storage_type="compressed",
        capacity_m3=10.0,
        p_min_bar=5.0,
        p_max_bar=5.0,
        initial_fill_fraction=1.0,
    )
    out_flat = gs_flat.step(t=0.0, dt=1.0, inputs={"Q_gas_out_m3_per_day": 100.0})
    assert out_flat["stored_volume_m3"] == pytest.approx(0.0)


def test_gas_storage_membrane_safety_overpressure_branch_can_vent(
    monkeypatch: pytest.MonkeyPatch,
):
    gs = GasStorage(
        "safety_mem",
        storage_type="membrane",
        capacity_m3=100.0,
        initial_fill_fraction=0.0,
    )
    gs.stored_volume_m3 = 120.0  # force impossible state to exercise safety branch

    pressures = iter([1.0, gs.p_max_bar + 0.1, gs.p_max_bar])
    monkeypatch.setattr(gs, "_estimate_pressure_bar", lambda: next(pressures))

    out = gs.step(t=0.0, dt=1.0, inputs={"Q_gas_in_m3_per_day": 0.0})

    assert out["vented_volume_m3"] == pytest.approx(20.0)
    assert out["stored_volume_m3"] == pytest.approx(100.0)


def test_gas_storage_membrane_safety_overpressure_branch_can_compute_zero_vent(
    monkeypatch: pytest.MonkeyPatch,
):
    gs = GasStorage(
        "safety_mem_zero",
        storage_type="membrane",
        capacity_m3=100.0,
        initial_fill_fraction=1.0,
    )
    gs.stored_volume_m3 = 100.0  # already at target volume

    pressures = iter([1.0, gs.p_max_bar + 0.1, gs.p_max_bar])
    monkeypatch.setattr(gs, "_estimate_pressure_bar", lambda: next(pressures))

    out = gs.step(t=0.0, dt=1.0, inputs={"Q_gas_in_m3_per_day": 0.0})

    assert out["vented_volume_m3"] == 0.0
    assert out["stored_volume_m3"] == pytest.approx(100.0)


def test_gas_storage_compressed_safety_overpressure_branch_can_vent(
    monkeypatch: pytest.MonkeyPatch,
):
    gs = GasStorage(
        "safety_comp",
        storage_type="compressed",
        capacity_m3=100.0,
        p_min_bar=5.0,
        p_max_bar=200.0,
        initial_fill_fraction=0.0,
    )
    gs.stored_volume_m3 = 100.0

    pressures = iter([10.0, gs.p_max_bar + 1.0, gs.p_max_bar])
    monkeypatch.setattr(gs, "_estimate_pressure_bar", lambda: next(pressures))

    out = gs.step(t=0.0, dt=1.0, inputs={"Q_gas_in_m3_per_day": 0.0})

    assert out["vented_volume_m3"] == pytest.approx(0.1)
    assert out["stored_volume_m3"] == pytest.approx(99.9)


def test_gas_storage_to_dict_includes_state_and_connections():
    gs = GasStorage(
        "serde",
        storage_type="dome",
        capacity_m3=25.0,
        initial_fill_fraction=0.2,
        name="Tank",
    )
    gs.add_input("digester_1")
    gs.add_output("chp_1")
    gs.step(t=0.0, dt=1.0, inputs={"Q_gas_in_m3_per_day": 1.0, "set_pressure": 1.01})

    data = gs.to_dict()

    assert data["component_id"] == "serde"
    assert data["component_type"] == "storage"
    assert data["storage_type"] == "dome"
    assert data["inputs"] == ["digester_1"]
    assert data["outputs"] == ["chp_1"]
    assert "outputs_data" in data


def test_gas_storage_from_dict_restores_values_and_handles_invalid_stored_volume(
    capsys: pytest.CaptureFixture[str],
):
    restored = GasStorage.from_dict(
        {
            "component_id": "gs_restored",
            "storage_type": "compressed",
            "capacity_m3": 50.0,
            "p_min_bar": 5.0,
            "p_max_bar": 100.0,
            "name": "Restored",
            "stored_volume_m3": 12.5,
            "pressure_setpoint_bar": 20.0,
        }
    )
    invalid = GasStorage.from_dict(
        {
            "component_id": "gs_invalid",
            "stored_volume_m3": object(),
            "pressure_setpoint_bar": 1.02,
        }
    )

    assert restored.component_id == "gs_restored"
    assert restored.storage_type == "compressed"
    assert restored.stored_volume_m3 == pytest.approx(12.5)
    assert restored.pressure_setpoint_bar == 20.0
    assert restored.outputs_data["stored_volume_m3"] == pytest.approx(12.5)

    # invalid stored_volume prints an error and falls back to constructor/initialize state
    assert invalid.component_id == "gs_invalid"
    assert invalid.pressure_setpoint_bar == 1.02
    captured = capsys.readouterr().out
    assert "float() argument" in captured or "must be" in captured


def test_flare_combusts_and_tracks_cumulative():
    f = Flare("flare_1", destruction_efficiency=0.9)
    f.initialize()
    out1 = f.step(t=0.0, dt=1.0, inputs={"Q_gas_in_m3_per_day": 10.0, "CH4_fraction": 0.6})
    assert out1["vented_volume_m3"] == 10.0
    assert out1["CH4_destroyed_m3"] == 10.0 * 0.6 * 0.9
    # second step increases cumulative
    out2 = f.step(t=1.0, dt=1.0, inputs={"Q_gas_in_m3_per_day": 5.0})
    assert out2["cumulative_vented_m3"] == out1["vented_volume_m3"] + out2["vented_volume_m3"]


def test_flare_initialize_invalid_cumulative_value_falls_back_to_zero():
    f = Flare("flare_invalid_restore")

    f.initialize({"cumulative_vented_m3": object()})

    assert f.outputs_data["cumulative_vented_m3"] == 0.0
    assert f.outputs_data["vented_volume_m3"] == 0.0
    assert f.outputs_data["CH4_destroyed_m3"] == 0.0


def test_flare_to_dict_serializes_configuration_and_state():
    f = Flare("flare_cfg", destruction_efficiency=0.85, name="Emergency Flare")
    f.step(t=0.0, dt=0.5, inputs={"Q_gas_in_m3_per_day": 8.0, "CH4_fraction": 0.5})

    result = f.to_dict()

    assert result["component_id"] == "flare_cfg"
    assert result["component_type"] == "storage"
    assert result["name"] == "Emergency Flare"
    assert result["destruction_efficiency"] == 0.85
    assert result["cumulative_vented_m3"] == 4.0


def test_flare_from_dict_restores_state_and_defaults():
    restored = Flare.from_dict(
        {
            "component_id": "flare_restored",
            "destruction_efficiency": 0.91,
            "name": "Restored Flare",
            "cumulative_vented_m3": 12.5,
        }
    )
    defaulted = Flare.from_dict({"component_id": "flare_default"})

    assert restored.component_id == "flare_restored"
    assert restored.name == "Restored Flare"
    assert restored.destruction_efficiency == 0.91
    assert restored.outputs_data["cumulative_vented_m3"] == 12.5

    assert defaulted.component_id == "flare_default"
    assert defaulted.destruction_efficiency == 0.98
    assert defaulted.outputs_data["cumulative_vented_m3"] == 0.0
