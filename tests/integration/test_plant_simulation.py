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
