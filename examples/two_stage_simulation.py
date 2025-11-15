#!/usr/bin/env python3
# ============================================================================
# examples/two_stage_simulation.py
# ============================================================================
"""
Example: Two-stage digester in series with CHP and heating,
configured using the PlantConfigurator.

Demonstrates:
- Two digesters in series
- Automatic gas storage per digester
- CHP unit consuming biogas
- Heating systems
- Loading initial state from CSV file
- High-level configuration via PlantConfigurator

Usage:
    python examples/two_stage_simulation.py
"""

from pathlib import Path


def main():
    """Run a two-stage biogas plant simulation using PlantConfigurator."""
    # Import required modules
    from pyadm1.configurator.plant_builder import BiogasPlant
    from pyadm1.substrates.feedstock import Feedstock
    from pyadm1.core.adm1 import get_state_zero_from_initial_state
    from pyadm1.configurator.plant_configurator import PlantConfigurator

    print("=" * 70)
    print("PyADM1 Two-Stage Simulation Example (PlantConfigurator version)")
    print("=" * 70)

    # Step 1: Create feedstock
    feeding_freq = 48  # Can change substrate every 48 hours
    feedstock = Feedstock(feeding_freq=feeding_freq)

    # Step 2: Load initial state from CSV
    data_path = Path(__file__).parent.parent / "data" / "initial_states"
    initial_state_file = data_path / "digester_initial8.csv"
    if initial_state_file.exists():
        print(f"Loading initial state from: {initial_state_file}")
        adm1_state = get_state_zero_from_initial_state(str(initial_state_file))
    else:
        print(f"   Warning: Initial state file not found at {initial_state_file}")
        adm1_state = None

    # Substrate feed rates
    digester1_feed = [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]
    digester2_feed = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Post-digester receives only effluent

    # Step 3: Create biogas plant
    plant = BiogasPlant("Two-Stage Digester Plant")
    configurator = PlantConfigurator(plant, feedstock)

    # Step 4: Add components using configurator
    print("Adding two digesters...")
    configurator.add_digester(
        digester_id="digester_1",
        V_liq=1977.0,
        V_gas=304.0,
        T_ad=308.15,
        name="Main Digester",
        load_initial_state=True,
        initial_state_file=str(initial_state_file) if adm1_state else None,
        Q_substrates=digester1_feed,
    )
    configurator.add_digester(
        digester_id="digester_2",
        V_liq=1000.0,
        V_gas=150.0,
        T_ad=308.15,
        name="Post Digester",
        load_initial_state=True,
        initial_state_file=str(initial_state_file) if adm1_state else None,
        Q_substrates=digester2_feed,
    )

    print("Adding CHP unit...")
    configurator.add_chp(
        chp_id="chp_1",
        P_el_nom=500.0,
        eta_el=0.40,
        eta_th=0.45,
        name="CHP Unit",
    )

    print("Adding heating systems...")
    configurator.add_heating(
        heating_id="heating_1",
        target_temperature=308.15,
        heat_loss_coefficient=0.5,
        name="Main Digester Heating",
    )
    configurator.add_heating(
        heating_id="heating_2",
        target_temperature=308.15,
        heat_loss_coefficient=0.3,
        name="Post Digester Heating",
    )

    # Step 5: Connect digesters in series and set up energy flows
    print("Connecting liquid flow between digesters...")
    configurator.connect("digester_1", "digester_2", "liquid")

    print("Connecting thermal flows...")
    configurator.auto_connect_digester_to_chp("digester_1", "chp_1")
    configurator.auto_connect_digester_to_chp("digester_2", "chp_1")
    configurator.auto_connect_chp_to_heating("chp_1", "heating_1")
    configurator.auto_connect_chp_to_heating("chp_1", "heating_2")

    # Step 6: Initialize plant
    print("Initializing plant...")
    plant.initialize()

    # Print plant summary
    print("\n" + "=" * 70)
    print(plant.get_summary())
    print("=" * 70)

    # Step 7: Save initial configuration to JSON
    output_path = Path(__file__).parent.parent / "output"
    output_path.mkdir(exist_ok=True)
    config_file = output_path / "plant_config_initial.json"
    plant.to_json(str(config_file))
    print(f"\nInitial configuration saved to: {config_file}")

    # Step 8: Run simulation
    print("\n" + "=" * 70)
    print("Starting simulation...")
    print("=" * 70)

    duration = 10.0  # 10 days
    dt = 1.0 / 24.0  # 1 hour time step
    save_interval = 1.0  # Save daily

    results = plant.simulate(duration=duration, dt=dt, save_interval=save_interval)

    print(f"\nSimulation complete. Generated {len(results)} result snapshots.")

    # Print final results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    final_result = results[-1]
    print(f"\nTime: {final_result['time']:.1f} days")

    for comp_id, comp_results in final_result["components"].items():
        comp = plant.components[comp_id]
        print(f"\n{comp.name} ({comp.component_type.value}):")
        print("-" * 50)

        for key, value in comp_results.items():
            if isinstance(value, (int, float)):
                print(f"  {key:20s}: {value:12.2f}")
            elif isinstance(value, list) and len(value) <= 3:
                print(f"  {key:20s}: {value}")

    # Save final plant state
    final_config_file = output_path / "plant_config_final.json"
    plant.to_json(str(final_config_file))
    print(f"\nFinal configuration saved to: {final_config_file}")

    # Print key performance indicators
    print("\n" + "=" * 70)
    print("KEY PERFORMANCE INDICATORS")
    print("=" * 70)

    # Digester 1 metrics
    if "digester_1" in results[-1]["components"]:
        d1_results = results[-1]["components"]["digester_1"]
        print("\nMain Digester (Final Values):")
        print(f"  Biogas production:  {d1_results.get('Q_gas', 0):.1f} m³/d")
        print(f"  Methane production: {d1_results.get('Q_ch4', 0):.1f} m³/d")
        print(f"  pH value:           {d1_results.get('pH', 0):.2f}")
        print(f"  VFA concentration:  {d1_results.get('VFA', 0):.2f} g/L")
        print(f"  TAC concentration:  {d1_results.get('TAC', 0):.2f} g/L")

    # Digester 2 metrics
    if "digester_2" in results[-1]["components"]:
        d2_results = results[-1]["components"]["digester_2"]
        print("\nPost Digester (Final Values):")
        print(f"  Biogas production:  {d2_results.get('Q_gas', 0):.1f} m³/d")
        print(f"  Methane production: {d2_results.get('Q_ch4', 0):.1f} m³/d")
        print(f"  pH value:           {d2_results.get('pH', 0):.2f}")

    # CHP metrics
    if "chp_1" in results[-1]["components"]:
        chp_results = results[-1]["components"]["chp_1"]
        print("\nCHP Unit (Final Values):")
        print(f"  Electrical power:   {chp_results.get('P_el', 0):.1f} kW")
        print(f"  Thermal power:      {chp_results.get('P_th', 0):.1f} kW")
        print(f"  Gas consumption:    {chp_results.get('Q_gas_consumed', 0):.1f} m³/d")

    print("\n" + "=" * 70)
    print("Simulation completed successfully!")
    print("=" * 70 + "\n")

    return plant, results


if __name__ == "__main__":
    plant, results = main()
