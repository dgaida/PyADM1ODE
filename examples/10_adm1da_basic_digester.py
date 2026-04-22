#!/usr/bin/env python3
# examples/10_adm1da_basic_digester.py
"""
ADM1da basic single-digester simulation — minimal example.

Mirrors the style of example 01 for the legacy ADM1 model:
    1. Declare the substrates as a list of XML file stems (up to 10 slots).
    2. Declare the flow rates as a matching list [m³/d].
    3. Hand both to the configurator; everything else is automatic.

Digester configuration
----------------------
  Temperature       42 °C  (315.15 K)
  Liquid volume     1050 m³
  Gas headspace     150 m³    (1200 − 1050)
  Gas transfer      200 1/d   (k_L_a override)

Usage
-----
    python examples/10_adm1da_basic_digester.py
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np


def main():
    from pyadm1.configurator.plant_builder import BiogasPlant
    from pyadm1.configurator.plant_configurator import PlantConfigurator
    from pyadm1.substrates.adm1da_feedstock import ADM1daFeedstock

    # --- 1. Substrates (up to 10 slots, list of XML file stems) ---
    substrate_ids = [
        "maize_silage_milk_ripeness",
        "swine_manure",
    ]

    # --- 2. Flow rates [m³/d], one entry per slot ---
    Q_substrates = [11.4, 6.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # --- 3. Build plant ---
    feedstock = ADM1daFeedstock(substrate_ids, feeding_freq=24, total_simtime=160)
    plant = BiogasPlant("ADM1da Basic Plant")
    cfg = PlantConfigurator(plant, feedstock)

    cfg.add_adm1da_digester(
        "main_digester",
        V_liq=1050.0,
        V_gas=150.0,
        T_ad=315.15,
        Q_substrates=Q_substrates,
        k_L_a=200.0,
    )
    plant.initialize()

    # --- 4. Simulate ---
    results = plant.simulate(duration=150.0, dt=1.0, save_interval=1.0)

    # --- 5. Report ---
    times = np.array([r["time"] for r in results])
    dig = [r["components"]["main_digester"] for r in results]
    q_gas = np.array([c["Q_gas"] for c in dig])
    q_ch4 = np.array([c["Q_ch4"] for c in dig])
    pH = np.array([c["pH"] for c in dig])

    print(f"{'Day':>5s}  {'Biogas':>10s}  {'Methane':>10s}  {'CH4%':>6s}  {'pH':>6s}")
    print("-" * 50)
    for day in sorted(set(list(range(0, int(times[-1]) + 1, 10)) + [int(times[-1])])):
        i = int(np.argmin(np.abs(times - day)))
        ch4p = q_ch4[i] / q_gas[i] * 100.0 if q_gas[i] > 0.0 else 0.0
        print(f"{day:>5d}  {q_gas[i]:>10.1f}  {q_ch4[i]:>10.1f}  {ch4p:>6.1f}  {pH[i]:>6.2f}")

    return plant, results


if __name__ == "__main__":
    main()
