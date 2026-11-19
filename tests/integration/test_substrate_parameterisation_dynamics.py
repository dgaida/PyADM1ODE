"""Dynamics of a digester under the SIMBA and the measured substrate parameters.

The same two substrates of the legacy Gummersbach library are fed once through
the superseded YAML files in ``data/substrates/legacy`` (Weender analysis plus
the SIMBA fractionation) and once through their measured parameters (cell wall, degradation level, single
acids, titrated acid capacity). The measured parameters must give a healthy
digester with the same methane level, and a step response that starts as fast
but levels off later, because more of the COD sits in the slow pool.

The feed is diluted with water. Undiluted, the swine manure of the library
(4 g/l ammonium) drives the model into ammonia inhibition under either
parameterisation, which would mask the disintegration dynamics tested here.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyadm1 import Feedstock
from pyadm1.components.biological import Digester
from pyadm1.substrates import params_from_measured

VOLUME = 3000.0  # m³
TEMPERATURE = 313.15  # K
BASE = [30.0, 15.0, 45.0]  # maize, swine manure, water [t/d]
STEP = [45.0, 15.0, 45.0]
SPIN_UP_DAYS = 200
STEP_DAYS = 60
METHANE_PER_COD = 0.35  # Nm³ CH4 per kg COD

SUBSTRATES = {
    "simba": ["legacy:maize_silage_gummersbach", "legacy:swine_manure_gummersbach"],
    "measured": ["gummersbach:maize", "gummersbach:swinemanure"],
}
DEGRADABLE = (
    "X_PS_ch",
    "X_PS_pr",
    "X_PS_li",
    "X_PF_ch",
    "X_PF_pr",
    "X_PF_li",
    "X_S_ch",
    "X_S_pr",
    "X_S_li",
    "S_ac",
    "S_pro",
    "S_bu",
    "S_va",
)
SLOW = ("X_PS_ch", "X_PS_pr", "X_PS_li")


def _feedstock(variant: str) -> Feedstock:
    water = params_from_measured("water", {"TS": 0.0, "VS": 0.0, "RF": 0.0, "RP": 0.0, "RL": 0.0, "pH": 7.0, "aXI": 0.0})
    return Feedstock([*SUBSTRATES[variant], water], feeding_freq=24, total_simtime=10)


def _time_to_reach(progress: np.ndarray, level: float) -> float:
    """Days until the normalised step response first reaches *level* (linear interpolation)."""
    index = int(np.nonzero(progress >= level)[0][0])
    if index == 0:
        return level / progress[0]
    return index + (level - progress[index - 1]) / (progress[index] - progress[index - 1])


def _simulate(variant: str) -> dict:
    """Steady state at the base feed, then 60 days at 50 % more maize."""
    feedstock = _feedstock(variant)
    digester = Digester("d", feedstock, V_liq=VOLUME, V_gas=450.0, T_ad=TEMPERATURE)
    digester.initialize({"Q_substrates": BASE})

    day = 0.0
    for _ in range(SPIN_UP_DAYS):
        steady = digester.step(day, 1.0, {"Q_substrates": BASE})
        day += 1.0
    methane = []
    for _ in range(STEP_DAYS):
        stepped = digester.step(day, 1.0, {"Q_substrates": STEP})
        methane.append(stepped["Q_ch4"])
        day += 1.0

    series = np.asarray(methane)
    blend = feedstock.blended_concentrations(BASE)
    degradable = sum(blend[key] for key in DEGRADABLE)
    return {
        "steady": steady,
        "stepped": stepped,
        "progress": (series - steady["Q_ch4"]) / (series[-1] - steady["Q_ch4"]),
        "cod_load": degradable * sum(feedstock.actual_Q(BASE)),
        "slow_share": sum(blend[key] for key in SLOW) / degradable,
    }


@pytest.fixture(scope="module")
def both() -> dict[str, dict]:
    """Both parameterisations, simulated once for the whole module."""
    return {variant: _simulate(variant) for variant in SUBSTRATES}


@pytest.fixture(params=list(SUBSTRATES))
def run(request: pytest.FixtureRequest, both: dict[str, dict]) -> dict:
    """One parameterisation at a time."""
    return both[request.param]


class TestEachParameterisation:
    def test_digester_is_healthy(self, run: dict) -> None:
        for state in (run["steady"], run["stepped"]):
            assert 7.0 < state["pH"] < 7.6
            assert state["VFA"] < 1.0

    def test_methane_matches_the_degradable_cod(self, run: dict) -> None:
        """Less than the potential: part of the slow pool is washed out, part becomes biomass."""
        conversion = run["steady"]["Q_ch4"] / (run["cod_load"] * METHANE_PER_COD)

        assert 0.65 < conversion < 0.95

    def test_more_maize_gives_proportionally_more_methane(self, run: dict) -> None:
        gain = run["stepped"]["Q_ch4"] / run["steady"]["Q_ch4"] - 1.0

        assert 0.35 < gain < 0.50

    def test_step_response_rises_without_overshoot(self, run: dict) -> None:
        progress = run["progress"]

        assert np.all(np.diff(progress) > -1e-3)
        assert progress.max() < 1.02

    def test_response_times_are_those_of_the_two_pools(self, run: dict) -> None:
        """Half of the change within days (fast pool), the rest within weeks (slow pool)."""
        assert 1.0 < _time_to_reach(run["progress"], 0.5) < 5.0
        assert 5.0 < _time_to_reach(run["progress"], 0.9) < 25.0


class TestMeasuredAgainstSimba:
    def test_same_methane_level(self, both: dict[str, dict]) -> None:
        assert both["measured"]["steady"]["Q_ch4"] == pytest.approx(both["simba"]["steady"]["Q_ch4"], rel=0.10)

    def test_cell_wall_puts_more_cod_into_the_slow_pool(self, both: dict[str, dict]) -> None:
        assert both["measured"]["slow_share"] > 1.2 * both["simba"]["slow_share"]

    def test_first_half_of_the_response_is_equally_fast(self, both: dict[str, dict]) -> None:
        simba = _time_to_reach(both["simba"]["progress"], 0.5)
        measured = _time_to_reach(both["measured"]["progress"], 0.5)

        assert measured == pytest.approx(simba, abs=1.0)

    def test_response_levels_off_later(self, both: dict[str, dict]) -> None:
        simba = _time_to_reach(both["simba"]["progress"], 0.9)
        measured = _time_to_reach(both["measured"]["progress"], 0.9)

        assert simba + 1.0 < measured < 2.0 * simba
