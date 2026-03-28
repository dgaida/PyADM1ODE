#!/usr/bin/env python3
# ============================================================================
# examples/quickstart.py
# ============================================================================
"""
Quickstart example for PyADM1.

Demonstrates:
- Single digester with substrate feed
- Configuration using PlantConfigurator
- Basic simulation
- Result output

Usage:
    python examples/quickstart.py
"""

from pathlib import Path
import csv
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main():
    """Run a simple single-digester simulation using PlantConfigurator."""
    # Import required modules
    from pyadm1.configurator.plant_builder import BiogasPlant
    from pyadm1.substrates.feedstock import Feedstock
    from pyadm1.core.adm1 import get_state_zero_from_initial_state
    from pyadm1.configurator.plant_configurator import PlantConfigurator

    print("=" * 70)
    print("PyADM1 Quickstart Example (PlantConfigurator version)")
    print("=" * 70)

    # Step 1: Create feedstock
    print("\n1. Creating feedstock...")
    feeding_freq = 48  # Controller can change substrate every 48 hours
    feedstock = Feedstock(feeding_freq=feeding_freq)

    # Step 2: Prepare initial state and substrate feed
    print("2. Loading initial state from CSV...")
    data_path = Path(__file__).parent.parent / "data" / "initial_states"
    initial_state_file = data_path / "digester_initial8.csv"

    if not initial_state_file.exists():
        print(f"   Warning: Initial state file not found at {initial_state_file}")
        print("   Using default initialization instead.")
        adm1_state = None
    else:
        print(f"   Loading from: {initial_state_file}")
        adm1_state = get_state_zero_from_initial_state(str(initial_state_file))
    Q_substrates = [
        15,
        10,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]  # Corn silage and swine manure feed m^3/d

    df = feedstock.get_influent_dataframe(Q=Q_substrates)
    for col in df.columns:
        print(f"Column '{col}': {df.iloc[0][col]}")

    # Step 3: Create biogas plant
    print("3. Creating biogas plant...")
    plant = BiogasPlant("Quickstart Plant")

    # Step 4: Use PlantConfigurator to add digester
    print("4. Adding digester with PlantConfigurator...")
    configurator = PlantConfigurator(plant, feedstock)
    digester, state_info = configurator.add_digester(
        digester_id="main_digester",
        V_liq=2000.0,
        V_gas=300.0,
        T_ad=308.15,  # 35°C
        name="Main Digester",
        load_initial_state=True,
        initial_state_file=str(initial_state_file) if adm1_state else None,
        Q_substrates=Q_substrates,
    )

    # Step 5: Initialize plant
    print("5. Initializing plant...")
    plant.initialize()

    calibration_params = digester.get_calibration_parameters()
    if not calibration_params:
        digester.adm1.create_influent(digester.Q_substrates, 0)
        calibration_params = digester.adm1._get_substrate_dependent_params()
    print(calibration_params)

    # Step 6: Run simulation
    print("6. Running simulation...")
    print("   Duration: 5 days")
    print("   Time step: 1 hour")
    print("   Save interval: 1 hour")

    results = plant.simulate(
        duration=30.0,
        dt=1.0 / 24.0,
        save_interval=1.0,  # 30 days  # 1 hour time step  # Save results daily
    )

    # Step 9: Display results
    print("\n" + "=" * 70)
    print("SIMULATION RESULTS")
    print("=" * 70)

    if len(results) > 1:
        dt_days = abs(float(results[1]["time"]) - float(results[0]["time"]))
    else:
        dt_days = 1.0
    day_tolerance = max(1e-9, dt_days * 0.51)

    day_to_result = {}
    for result in results:
        time = float(result["time"])
        day = int(round(time))
        if day >= 1 and abs(time - day) <= day_tolerance:
            day_to_result[day] = result

    daily_results = [day_to_result[day] for day in sorted(day_to_result)]

    print(f"\nGenerated {len(results)} time-step snapshots ({len(daily_results)} daily summaries shown)\n")

    # Show results once per day
    for result in daily_results:
        time = result["time"]
        comp_results = result["components"]["main_digester"]
        q_gas_day = float(comp_results.get("Q_gas", 0.0))
        q_ch4_day = float(comp_results.get("Q_ch4", 0.0))
        ch4_share_day = (q_ch4_day / q_gas_day * 100.0) if q_gas_day > 0 else 0.0

        print(f"Day {int(round(time)):>2d}:")
        print(f"  Biogas:  {q_gas_day:>8.1f} m³/d")
        print(f"  Methane: {q_ch4_day:>8.1f} m³/d")
        print(f"  CH4:     {ch4_share_day:>8.1f} %")
        print()

    # Final summary
    final = results[-1]["components"]["main_digester"]
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Total biogas production:  {final.get('Q_gas', 0):.1f} m³/d")
    print(f"Total methane production: {final.get('Q_ch4', 0):.1f} m³/d")
    print(f"Methane content:          {final.get('Q_ch4', 0) / final.get('Q_gas', 1) * 100:.1f}%")
    print(f"Process stability (pH):   {final.get('pH', 0):.2f}")
    print("=" * 70)

    print("\nSimulation completed successfully!")

    # Optional: Save configuration
    output_path = Path(__file__).parent.parent / "output"
    output_path.mkdir(exist_ok=True)
    config_file = output_path / "quickstart_config.json"
    plant.to_json(str(config_file))
    print(f"\nConfiguration saved to: {config_file}")

    return plant, results


if __name__ == "__main__":
    plant, results = main()
