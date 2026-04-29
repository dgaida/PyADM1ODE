# -*- coding: utf-8 -*-
"""
Minimal example script demonstrating PyADM1 usage.

Run a 30-day single-digester simulation with a maize-silage / swine-manure
feed and print daily gas production.
"""

from pyadm1 import BiogasPlant, Feedstock
from pyadm1.configurator.plant_configurator import PlantConfigurator


def main() -> None:
    """Run a 30-day ADM1 simulation."""
    feedstock = Feedstock(
        ["maize_silage_milk_ripeness", "swine_manure"],
        feeding_freq=24,
        total_simtime=30,
    )

    Q_substrates = [11.4, 6.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    plant = BiogasPlant("ADM1 Demo Plant")
    cfg = PlantConfigurator(plant, feedstock)
    cfg.add_digester(
        digester_id="main_digester",
        V_liq=1200.0,
        V_gas=216.0,
        T_ad=315.15,
        Q_substrates=Q_substrates,
    )
    plant.initialize()

    results = plant.simulate(duration=30.0, dt=1.0, save_interval=1.0)

    print(f"\n{'Day':>4s}  {'Biogas':>10s}  {'Methane':>10s}  {'pH':>6s}")
    print("-" * 36)
    for r in results:
        c = r["components"]["main_digester"]
        print(f"{r['time']:>4.0f}  {c['Q_gas']:>10.1f}  {c['Q_ch4']:>10.1f}  {c['pH']:>6.2f}")


if __name__ == "__main__":
    main()
