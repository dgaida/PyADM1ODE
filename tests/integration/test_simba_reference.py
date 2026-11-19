"""Validation against SIMBA# biogas: a 600 d run with a substrate switch.

The reference data (``data/simba_adm1da_cattle_600d.json``) holds what a
SIMBA# 5.0 run with the ADM1da model produced: start state, influent of both
feeding phases, the reactor state at 300 d and 600 d and a coarse trajectory,
all in PyADM1ODE's state convention. The influent is the one SIMBA# computed
from its own substrate blocks, so the test does not depend on the substrate
library; it compares the digestion model alone.

SIMBA#'s headspace is not saturated with water vapour (about 80 % relative
humidity at 1.017 bar). The main run therefore fixes the dry gas pressure of
the SIMBA# run; a second run shows what the default headspace changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyadm1.components.biological import Digester
from pyadm1.core.adm1 import INFLUENT_COLUMNS

DATA = json.loads((Path(__file__).parent / "data" / "simba_adm1da_cattle_600d.json").read_text(encoding="utf-8"))
NAMES = INFLUENT_COLUMNS[:-1]

# Relative tolerance per state at the two reference days. Acetate sits on the
# flat part of its inhibition curve at this operating point and reacts ten
# times stronger than everything else, hence the wider band.
STATE_TOLERANCE = dict.fromkeys(NAMES, 2.0e-3)
STATE_TOLERANCE.update({"S_ac": 2.0e-2, "S_ac_ion": 2.0e-2, "S_co2": 1.0e-2, "S_hco3_ion": 1.0e-2, "S_ch4": 5.0e-3})


def _simulate(headspace: str, dt: float) -> pd.DataFrame:
    """Run both feeding phases through a :class:`Digester`.

    Args:
        headspace: ``"simba"`` fixes the dry gas pressure of the SIMBA# run,
            ``"default"`` keeps the saturated headspace at atmospheric pressure.
        dt: Output step [d].

    Returns:
        One row per step with the indicators, the gas flows and the state.
    """
    reactor = DATA["reactor"]
    digester = Digester("fermenter", None, V_liq=reactor["V_liq"], V_gas=reactor["V_gas"], T_ad=reactor["T_ad"])
    adm1 = digester.adm1
    if headspace == "simba":
        adm1.p_ext = reactor["p_dry"] + adm1._p_gas_h2o
    p_dry = adm1.p_ext - adm1._p_gas_h2o
    digester.initialize({"adm1_state": [*DATA["start_state"], 1.0e-6, 0.5 * p_dry, 0.5 * p_dry - 1.0e-6, p_dry]})

    rows, t = [], 0.0
    for phase in DATA["phases"]:
        adm1.set_influent_dataframe(pd.DataFrame([[*phase["influent"], phase["q_in"]]], columns=INFLUENT_COLUMNS))
        while t < phase["t_end"] - 1.0e-9:
            out = digester.step(t, dt, {"Q_substrates": [phase["q_in"]]})
            t += dt
            state = np.asarray(out["state_out"], dtype=float)
            rows.append(
                {
                    "t": t,
                    "pH": float(out["pH"]),
                    "VFA": float(out["VFA"]),
                    "TAC": float(out["TAC"]),
                    "S_ac": state[6],
                    "S_nh4": state[10],
                    "Q_ch4": float(out["Q_ch4"]),
                    "Q_co2": float(out["Q_co2"]),
                    "Q_gas": float(out["Q_gas"]),
                    "state": state,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def run() -> pd.DataFrame:
    return _simulate("simba", dt=1.0)


def _at(frame: pd.DataFrame, day: float) -> pd.Series:
    return frame.iloc[(frame["t"] - day).abs().argmin()]


@pytest.mark.parametrize("day", ["300", "600"])
def test_every_state_matches_simba(run: pd.DataFrame, day: str) -> None:
    """All 37 liquid states agree with SIMBA# at the end of each feeding phase."""
    state = _at(run, float(day))["state"]
    reference = np.asarray(DATA["reference"][day]["state"])

    for i, name in enumerate(NAMES):
        if abs(reference[i]) < 1.0e-12:  # cations are zero in the reference run
            assert abs(state[i]) < 1.0e-9, name
            continue
        assert state[i] == pytest.approx(reference[i], rel=STATE_TOLERANCE[name]), name


@pytest.mark.parametrize("day", ["300", "600"])
def test_indicators_and_gas_match_simba(run: pd.DataFrame, day: str) -> None:
    """pH, FOS, TAC, the gas flows and the methane content agree with SIMBA#."""
    row = _at(run, float(day))
    reference = DATA["reference"][day]

    assert row["pH"] == pytest.approx(reference["pH"], abs=0.005)
    assert row["TAC"] == pytest.approx(reference["TAC"], rel=5.0e-3)
    assert row["VFA"] == pytest.approx(reference["VFA"], rel=2.0e-2)
    assert row["Q_ch4"] == pytest.approx(reference["Q_ch4"], rel=2.0e-3)
    assert row["Q_co2"] == pytest.approx(reference["Q_co2"], rel=2.0e-3)
    # SIMBA# reports wet gas; its dry part is what PyADM1ODE reports.
    assert row["Q_gas"] == pytest.approx(reference["Q_gas_dry"], rel=2.0e-3)
    assert 100.0 * row["Q_ch4"] / row["Q_gas"] == pytest.approx(reference["CH4_dry_percent"], abs=0.1)


def test_ammonia_in_the_gas_matches_simba(run: pd.DataFrame) -> None:
    """Free ammonia leaves with the gas at the level SIMBA# reports (about 300 ppm)."""
    row = _at(run, 600.0)
    digester_state = row["state"]
    reactor = DATA["reactor"]
    probe = Digester("probe", None, V_liq=reactor["V_liq"], V_gas=reactor["V_gas"], T_ad=reactor["T_ad"]).adm1
    ppm = probe._K_H_nh3 * digester_state[36] * probe._RT / digester_state[40] * 1.0e6

    assert ppm == pytest.approx(DATA["reference"]["600"]["NH3_ppm"], rel=0.05)


def test_trajectory_follows_simba_through_the_substrate_switch(run: pd.DataFrame) -> None:
    """From day 10 on the run stays on the SIMBA# trajectory, including the switch at day 300."""
    trajectory = DATA["trajectory"]
    grid = np.asarray(trajectory["t"], dtype=float)
    late = grid >= 10.0
    tolerance = {"VFA": 3.0e-2, "S_ac": 3.0e-2, "TAC": 5.0e-3, "S_nh4": 1.0e-3, "Q_ch4": 3.0e-3, "Q_co2": 2.0e-3}

    ph = np.interp(grid, run["t"], run["pH"])
    assert np.max(np.abs(ph - np.asarray(trajectory["pH"]))[late]) < 0.005

    for key, rel in tolerance.items():
        mine = np.interp(grid, run["t"], run[key])
        reference = np.asarray(trajectory[key])
        worst = np.max(np.abs(mine / reference - 1.0)[late])
        assert worst < rel, f"{key}: {100 * worst:.2f} % off"


def test_default_headspace_holds_slightly_less_dissolved_co2() -> None:
    """A saturated headspace at 1.013 bar has 2 % less dry gas pressure than the SIMBA# run.

    Dissolved CO2 and bicarbonate follow by the same 2 %, and acetate rises a
    few percent to keep the pH; nitrogen and the gas flows are unaffected.
    """
    row = _at(_simulate("default", dt=10.0), 600.0)
    reference = DATA["reference"]["600"]
    state, ref_state = row["state"], np.asarray(reference["state"])

    assert state[40] == pytest.approx(0.98 * ref_state[40], rel=5.0e-3)
    assert state[9] / ref_state[9] == pytest.approx(0.984, abs=0.006)
    assert 1.0 < state[6] / ref_state[6] < 1.06
    assert state[10] == pytest.approx(ref_state[10], rel=1.0e-3)
    assert row["pH"] == pytest.approx(reference["pH"], abs=0.005)
    assert row["Q_ch4"] == pytest.approx(reference["Q_ch4"], rel=3.0e-3)
