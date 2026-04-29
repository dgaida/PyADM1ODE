# -*- coding: utf-8 -*-
"""Unit tests for the Simulator class."""

import pytest

from pyadm1 import Feedstock
from pyadm1.core.adm1 import ADM1, STATE_SIZE
from pyadm1.simulation.simulator import Simulator


@pytest.fixture
def adm1_model() -> ADM1:
    fs = Feedstock(
        ["maize_silage_milk_ripeness", "swine_manure"],
        feeding_freq=24,
        total_simtime=5,
    )
    adm = ADM1(fs, V_liq=1200.0, V_gas=216.0, T_ad=315.15)
    adm.set_influent_dataframe(fs.get_influent_dataframe(Q=[11.4, 6.1]))
    adm.create_influent([11.4, 6.1], 0)
    return adm


def _initial_state() -> list:
    state = [0.01] * STATE_SIZE
    state[37:41] = [1.0e-5, 0.65, 0.33, 0.65 + 0.33 + 1.0e-5]
    return state


def test_simulator_constructs_default_solver(adm1_model: ADM1) -> None:
    sim = Simulator(adm1_model)
    assert sim.adm1 is adm1_model
    assert sim.solver is not None


def test_simulate_AD_plant_returns_state_size_vector(adm1_model: ADM1) -> None:
    sim = Simulator(adm1_model)
    final = sim.simulate_AD_plant([0.0, 1.0], _initial_state())

    assert len(final) == STATE_SIZE
    assert all(isinstance(v, float) for v in final)


def test_simulate_gas_production_returns_two_floats(adm1_model: ADM1) -> None:
    sim = Simulator(adm1_model)
    q_gas, q_ch4 = sim.simulate_gas_production([0.0, 1.0], _initial_state(), [11.4, 6.1])

    assert q_gas >= 0.0
    assert q_ch4 >= 0.0
    assert q_ch4 <= q_gas
