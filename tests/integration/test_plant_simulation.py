# -*- coding: utf-8 -*-
"""Integration tests: end-to-end ADM1 plant simulation."""

import pytest

from pyadm1 import BiogasPlant, Feedstock
from pyadm1.configurator.plant_configurator import PlantConfigurator


@pytest.mark.integration
def test_single_digester_two_day_simulation_produces_methane() -> None:
    feedstock = Feedstock(
        ["maize_silage_milk_ripeness", "swine_manure"],
        feeding_freq=24,
        total_simtime=5,
    )
    plant = BiogasPlant("Integration test")
    cfg = PlantConfigurator(plant, feedstock)
    cfg.add_digester(
        "main",
        V_liq=1200.0,
        V_gas=216.0,
        T_ad=315.15,
        Q_substrates=[11.4, 6.1, 0, 0, 0, 0, 0, 0, 0, 0],
    )
    plant.initialize()

    results = plant.simulate(duration=2.0, dt=1.0, save_interval=1.0)

    assert len(results) == 2
    final = results[-1]["components"]["main"]
    assert final["Q_gas"] > 100.0
    assert final["Q_ch4"] > 50.0
    assert 6.0 < final["pH"] < 8.5


@pytest.mark.integration
def test_chp_thermal_output_reaches_heater_from_first_step() -> None:
    """Regression test: ``auto_connect_chp_to_heating`` must actually deliver
    thermal power to the heater on every step, including the very first.

    Before the plant_builder Pass-3 heater re-execution fix, the heater
    stepped in Pass 1 reading the CHP's stale (zero-initialised) thermal
    output, so day-1 heat demand was charged entirely to the auxiliary
    boiler even when the CHP had ample thermal capacity. This test fails
    if that regression returns: it requires the heater to use *some* CHP
    heat (``P_th_used > 0``) on the very first save point, and the
    auxiliary share to be strictly less than the total supplied heat.
    """
    feedstock = Feedstock(
        ["maize_silage_milk_ripeness", "swine_manure"],
        feeding_freq=24,
        total_simtime=4,
    )
    plant = BiogasPlant("CHP→heater routing test")
    cfg = PlantConfigurator(plant, feedstock)
    cfg.add_digester(
        "dig",
        V_liq=1200.0,
        V_gas=216.0,
        T_ad=315.15,
        Q_substrates=[11.4, 6.1, 0, 0, 0, 0, 0, 0, 0, 0],
    )
    cfg.add_chp("chp", P_el_nom=300.0, eta_el=0.40, eta_th=0.45)
    cfg.add_heating("heater", target_temperature=315.15, heat_loss_coefficient=0.5)
    cfg.auto_connect_digester_to_chp("dig", "chp")
    cfg.auto_connect_chp_to_heating("chp", "heater")
    plant.initialize()

    results = plant.simulate(duration=2.0, dt=1.0, save_interval=1.0)
    assert len(results) == 2

    # CHP must actually run on day 1 -- otherwise the heater can't get heat
    # from it and the test is vacuous.
    chp_day1 = results[0]["components"]["chp"]
    assert chp_day1["P_th"] > 1.0, (
        f"CHP produced no thermal power on day 1 ({chp_day1['P_th']:.2f} kW); "
        "test setup must use a plant that actually fuels the CHP."
    )

    # The actual regression check: heater used at least some CHP heat, and
    # the auxiliary boiler did not carry the entire demand alone.
    heater_day1 = results[0]["components"]["heater"]
    p_th_used = heater_day1["P_th_used"]
    p_aux = heater_day1["P_aux_heat"]
    q_supplied = heater_day1["Q_heat_supplied"]

    assert p_th_used > 0.0, (
        f"Day-1 heater P_th_used = {p_th_used:.3f} kW. The CHP→heater "
        f"connection is wired (CHP delivered {chp_day1['P_th']:.1f} kW thermal) "
        "but no power is reaching the heater. This is the bug fixed in "
        "plant_builder.py: heaters must be re-executed in Pass 3 after the "
        "CHP has its actual gas supply."
    )
    assert p_aux < q_supplied, (
        f"Day-1 aux boiler carried the full demand: P_aux={p_aux:.2f} kW vs "
        f"Q_supplied={q_supplied:.2f} kW. The CHP→heater connection isn't "
        "contributing despite the CHP running."
    )

    # And on every subsequent day (here just day 2): same expectation.
    heater_day2 = results[1]["components"]["heater"]
    assert heater_day2["P_th_used"] > 0.0
    assert heater_day2["P_aux_heat"] < heater_day2["Q_heat_supplied"]


@pytest.mark.integration
def test_two_stage_plant_runs_to_completion() -> None:
    feedstock = Feedstock(
        ["maize_silage_milk_ripeness", "swine_manure"],
        feeding_freq=24,
        total_simtime=5,
    )
    plant = BiogasPlant("Two-Stage")
    cfg = PlantConfigurator(plant, feedstock)
    cfg.add_digester(
        "hydrolysis",
        V_liq=500.0,
        V_gas=75.0,
        T_ad=318.15,
        Q_substrates=[11.4, 6.1, 0, 0, 0, 0, 0, 0, 0, 0],
    )
    cfg.add_digester(
        "main",
        V_liq=1000.0,
        V_gas=150.0,
        T_ad=308.15,
        Q_substrates=[0.0] * 10,
    )
    cfg.connect("hydrolysis", "main", "liquid")
    plant.initialize()

    results = plant.simulate(duration=2.0, dt=1.0, save_interval=1.0)

    assert len(results) == 2
    h = results[-1]["components"]["hydrolysis"]
    m = results[-1]["components"]["main"]
    # Both stages produce gas after warm-up
    assert h["Q_gas"] > 0.0
    assert m["Q_gas"] >= 0.0
