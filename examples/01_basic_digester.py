#!/usr/bin/env python3
# examples/01_basic_digester.py
"""
PyADM1 basic single-digester simulation — minimal example.

    1. Declare the flow rates per substrate slot (up to 10).
    2. Hand them to the PlantConfigurator.
    3. Simulate.

Digester configuration
----------------------
  Temperature       35 °C  (308.15 K)
  Liquid volume     2000 m³
  Gas headspace     300 m³

Usage
-----
    python examples/01_basic_digester.py
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main():
    from pyadm1.configurator.plant_builder import BiogasPlant
    from pyadm1.configurator.plant_configurator import PlantConfigurator
    from pyadm1.substrates.feedstock import Feedstock

    # ------------------------------------------------------------------
    # 1. Define the plant
    # ------------------------------------------------------------------
    # Substrate flow rates [m³/d], one entry per slot (up to 10).
    # Slot 0: corn silage, slot 1: swine manure — defined in substrate_gummersbach.xml.
    Q_substrates = [15.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    feedstock = Feedstock(feeding_freq=48)
    plant = BiogasPlant("Quickstart Plant")
    cfg = PlantConfigurator(plant, feedstock)

    cfg.add_digester(
        digester_id="main_digester",
        V_liq=2000.0,
        V_gas=300.0,
        T_ad=308.15,
        name="Main Digester",
        Q_substrates=Q_substrates,
    )
    plant.initialize()

    # ------------------------------------------------------------------
    # 2. Simulate
    # ------------------------------------------------------------------
    results = plant.simulate(duration=30.0, dt=1.0 / 24.0, save_interval=1.0)

    # ------------------------------------------------------------------
    # 3. Print results
    # ------------------------------------------------------------------
    # Pick the snapshot closest to each whole day (float-drift safe).
    daily = {}
    for r in results:
        day = int(round(r["time"]))
        if day < 1:
            continue
        prev = daily.get(day)
        if prev is None or abs(r["time"] - day) < abs(prev["time"] - day):
            daily[day] = r

    print(f"\n{'Day':>4s}  {'Biogas':>10s}  {'Methane':>10s}  {'CH4%':>6s}  {'pH':>6s}")
    print("-" * 44)
    for day in sorted(daily):
        c = daily[day]["components"]["main_digester"]
        q_gas = float(c.get("Q_gas", 0.0))
        q_ch4 = float(c.get("Q_ch4", 0.0))
        ch4p = q_ch4 / q_gas * 100.0 if q_gas > 0.0 else 0.0
        print(f"{day:>4d}  {q_gas:>10.1f}  {q_ch4:>10.1f}  {ch4p:>6.1f}  {c.get('pH', 0.0):>6.2f}")

    final = results[-1]["components"]["main_digester"]
    q_gas, q_ch4 = float(final["Q_gas"]), float(final["Q_ch4"])
    print("\nFinal state:")
    print(f"  Biogas  : {q_gas:7.1f} m³/d")
    print(f"  Methane : {q_ch4:7.1f} m³/d  ({q_ch4 / q_gas * 100.0:.1f} %)")
    print(f"  pH      : {final['pH']:7.2f}")

    # ------------------------------------------------------------------
    # 4. Save results
    # ------------------------------------------------------------------
    output_path = REPO_ROOT / "output"
    output_path.mkdir(exist_ok=True)
    plant.to_json(str(output_path / "quickstart_config.json"))

    return plant, results


if __name__ == "__main__":
    main()
