#!/usr/bin/env python3
# examples/02_two_stage_plant.py
"""
Two-stage biogas plant — hydrolysis pre-tank + main fermenter, both ADM1.

Demonstrates:
    - Two digesters in series (hydrolysis pre-tank → main methanogenesis)
    - Auto gas storage attached to each fermenter
    - CHP unit consuming biogas, with safety flare
    - Heating systems for both stages
    - Plant JSON serialisation

Usage:
    python examples/02_two_stage_plant.py
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main():
    from pyadm1 import BiogasPlant, Feedstock
    from pyadm1.configurator.plant_configurator import PlantConfigurator

    print("=" * 70)
    print("PyADM1 Two-Stage Plant Example")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Feedstock
    # ------------------------------------------------------------------
    feedstock = Feedstock(
        ["maize_silage_milk_ripeness", "swine_manure"],
        feeding_freq=24,
        total_simtime=30,
    )

    # Substrate feed: hydrolysis stage gets fresh substrate, main stage inherits.
    hydro_feed = [11.4, 6.1, 0, 0, 0, 0, 0, 0, 0, 0]
    main_feed = [0.0] * 10

    # ------------------------------------------------------------------
    # Step 2: Build plant
    # ------------------------------------------------------------------
    print("\nBuilding plant components...")
    plant = BiogasPlant("Two-Stage Plant (ADM1 / SIMBA# biogas)")
    cfg = PlantConfigurator(plant, feedstock)

    # Hydrolysis pre-tank — short HRT, slightly thermophilic
    print("  - Hydrolysis pre-tank: 500 m³ liquid @ 45 °C (HRT ≈ 28 d)")
    cfg.add_digester(
        digester_id="hydrolysis",
        V_liq=500.0,
        V_gas=75.0,
        T_ad=318.15,
        name="Hydrolysis Pre-Tank",
        Q_substrates=hydro_feed,
    )

    # Main methanogenesis fermenter — larger volume, mesophilic
    print("  - Main fermenter:   1000 m³ liquid @ 35 °C")
    cfg.add_digester(
        digester_id="main_digester",
        V_liq=1000.0,
        V_gas=150.0,
        T_ad=308.15,
        name="Main Fermenter",
        Q_substrates=main_feed,
    )

    # CHP + heating
    print("  - CHP: 500 kW_el, η_el=40%, η_th=45%")
    cfg.add_chp(chp_id="chp_1", P_el_nom=500.0, eta_el=0.40, eta_th=0.45)
    cfg.add_heating(heating_id="heating_hydro", target_temperature=318.15, name="Hydrolysis Heating")
    cfg.add_heating(heating_id="heating_main", target_temperature=308.15, name="Main Stage Heating")

    # ------------------------------------------------------------------
    # Step 3: Connect components
    # ------------------------------------------------------------------
    print("\nConnecting components...")
    cfg.connect("hydrolysis", "main_digester", "liquid")
    cfg.auto_connect_digester_to_chp("hydrolysis", "chp_1")
    cfg.auto_connect_digester_to_chp("main_digester", "chp_1")
    cfg.auto_connect_chp_to_heating("chp_1", "heating_hydro")
    cfg.auto_connect_chp_to_heating("chp_1", "heating_main")

    plant.initialize()
    print(plant.get_summary())

    # ------------------------------------------------------------------
    # Step 4: Simulate
    # ------------------------------------------------------------------
    duration = 10.0
    dt = 1.0 / 24.0
    print(f"\nSimulating {duration:.0f} days at dt={dt*24:.1f} h...")
    results = plant.simulate(duration=duration, dt=dt, save_interval=1.0)

    # ------------------------------------------------------------------
    # Step 5: Report final state
    # ------------------------------------------------------------------
    final = results[-1]
    h = final["components"]["hydrolysis"]
    m = final["components"]["main_digester"]

    print("\n" + "=" * 70)
    print(f"Final state (day {final['time']:.1f}):")
    print("=" * 70)
    print("\nHydrolysis pre-tank:")
    print(f"  Biogas:    {h['Q_gas']:>8.1f} m³/d")
    print(f"  Methane:   {h['Q_ch4']:>8.1f} m³/d")
    print(f"  pH:        {h['pH']:>8.2f}")
    print(f"  VFA:       {h['VFA']:>8.2f} g HAc-eq/L")
    print(f"  TAC:       {h['TAC']:>8.2f} g CaCO3/L")

    print("\nMain fermenter:")
    print(f"  Biogas:    {m['Q_gas']:>8.1f} m³/d")
    print(f"  Methane:   {m['Q_ch4']:>8.1f} m³/d")
    print(f"  pH:        {m['pH']:>8.2f}")
    print(f"  VFA:       {m['VFA']:>8.2f} g HAc-eq/L")
    print(f"  TAC:       {m['TAC']:>8.2f} g CaCO3/L")

    total_gas = h["Q_gas"] + m["Q_gas"]
    total_ch4 = h["Q_ch4"] + m["Q_ch4"]
    print("\nPlant total:")
    print(f"  Biogas:    {total_gas:>8.1f} m³/d")
    print(f"  Methane:   {total_ch4:>8.1f} m³/d")
    if total_gas > 0:
        print(f"  CH4 frac.: {100 * total_ch4 / total_gas:>8.1f} %")

    # ------------------------------------------------------------------
    # Step 6: Save plant configuration
    # ------------------------------------------------------------------
    output_path = REPO_ROOT / "output"
    output_path.mkdir(exist_ok=True)
    plant.to_json(str(output_path / "two_stage_plant_config.json"))

    return plant, results


if __name__ == "__main__":
    main()
