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

    Q_substrates = [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]  # Corn silage and manure

    # Step 3: Create biogas plant
    print("3. Creating biogas plant...")
    plant = BiogasPlant("Quickstart Plant")

    # Step 4: Use PlantConfigurator to add digester
    print("4. Adding digester with PlantConfigurator...")
    configurator = PlantConfigurator(plant, feedstock)
    configurator.add_digester(
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

    # Step 6: Run simulation
    print("6. Running simulation...")
    print("   Duration: 5 days")
    print("   Time step: 1 hour")
    print("   Save interval: 1 day")

    results = plant.simulate(
        duration=5.0, dt=1.0 / 24.0, save_interval=1.0  # 5 days  # 1 hour time step  # Save results daily
    )

    # Step 7: Display results
    print("\n" + "=" * 70)
    print("SIMULATION RESULTS")
    print("=" * 70)

    print(f"\nGenerated {len(results)} daily result snapshots\n")

    # Show results for each day
    for result in results:
        time = result["time"]
        comp_results = result["components"]["main_digester"]

        print(f"Day {time:.1f}:")
        print(f"  Biogas:  {comp_results.get('Q_gas', 0):>8.1f} m³/d")
        print(f"  Methane: {comp_results.get('Q_ch4', 0):>8.1f} m³/d")
        print(f"  pH:      {comp_results.get('pH', 0):>8.2f}")
        print(f"  VFA:     {comp_results.get('VFA', 0):>8.2f} g/L")
        print(f"  TAC:     {comp_results.get('TAC', 0):>8.2f} g/L")
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

    print("\n✅ Simulation completed successfully!")

    # Optional: Save configuration
    output_path = Path(__file__).parent.parent / "output"
    output_path.mkdir(exist_ok=True)
    config_file = output_path / "quickstart_config.json"
    plant.to_json(str(config_file))
    print(f"\n💾 Configuration saved to: {config_file}")

    return plant, results


if __name__ == "__main__":
    plant, results = main()
