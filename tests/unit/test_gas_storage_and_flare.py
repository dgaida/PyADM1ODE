# tests/unit/test_gas_storage_and_flare.py
import math

from pyadm1.components.energy.gas_storage import GasStorage
from pyadm1.components.energy.flare import Flare


def test_gas_storage_basic_charge_and_vent():
    """GasStorage stores inflow and vents overflow above capacity."""
    gs = GasStorage(
        "test_stor", storage_type="membrane", capacity_m3=100.0, p_min_bar=0.95, p_max_bar=1.05, initial_fill_fraction=0.0
    )
    gs.initialize()

    # send more gas than capacity in one day -> some must be vented
    out = gs.step(t=0.0, dt=1.0, inputs={"Q_gas_in_m3_per_day": 200.0})
    assert out["stored_volume_m3"] <= 100.0 + 1e-6
    assert out["vented_volume_m3"] > 0.0
    assert math.isclose(out["utilization"], out["stored_volume_m3"] / 100.0, rel_tol=1e-6)


def test_gas_storage_supply_limits_and_setpoint():
    """GasStorage restricts outflow when setpoint requests higher pressure."""
    gs = GasStorage("s2", storage_type="membrane", capacity_m3=50.0, initial_fill_fraction=0.5, p_min_bar=0.95, p_max_bar=1.05)
    gs.initialize()
    # request large outflow with a high setpoint that requests charging => outflow restricted
    gs.outputs_data.copy()
    res = gs.step(t=0.0, dt=1.0, inputs={"Q_gas_out_m3_per_day": 1000.0, "set_pressure": 1.05})
    assert res["Q_gas_supplied_m3_per_day"] >= 0.0
    # after request with setpoint higher than current pressure, supply should be less than requested
    assert res["Q_gas_supplied_m3_per_day"] < 1000.0


def test_flare_combusts_and_tracks_cumulative():
    f = Flare("flare_1", destruction_efficiency=0.9)
    f.initialize()
    out1 = f.step(t=0.0, dt=1.0, inputs={"Q_gas_in_m3_per_day": 10.0, "CH4_fraction": 0.6})
    assert out1["vented_volume_m3"] == 10.0
    assert out1["CH4_destroyed_m3"] == 10.0 * 0.6 * 0.9
    # second step increases cumulative
    out2 = f.step(t=1.0, dt=1.0, inputs={"Q_gas_in_m3_per_day": 5.0})
    assert out2["cumulative_vented_m3"] == out1["vented_volume_m3"] + out2["vented_volume_m3"]
