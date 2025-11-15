#!/usr/bin/env python3
# ============================================================================
# examples/two_stage_simulation.py
# ============================================================================
"""
Example: Two-stage digester in series with CHP and heating.

This example demonstrates:
- Two digesters in series
- CHP unit consuming biogas
- Heating system using CHP waste heat
- Loading initial state from CSV file

Usage:
    python examples/two_stage_simulation.py
"""

from pathlib import Path


def create_two_stage_plant():
    """
    Create example two-stage biogas plant configuration.

    Returns:
        BiogasPlant: Configured plant with two digesters, CHP, and heating systems.
    """
    from pyadm1.configurator.plant_builder import BiogasPlant
    from pyadm1.components.biological.digester import Digester
    from pyadm1.components.energy.chp import CHP
    from pyadm1.components.energy.heating import HeatingSystem
    from pyadm1.configurator.connection_manager import Connection
    from pyadm1.substrates.feedstock import Feedstock
    from pyadm1.core.adm1 import get_state_zero_from_initial_state

    # Initialize feedstock
    feeding_freq = 48  # hours
    feedstock = Feedstock(feeding_freq)

    # Load initial state from CSV
    data_path = Path(__file__).parent.parent / "data" / "initial_states"
    initial_state_file = data_path / "digester_initial8.csv"

    print(f"Loading initial state from: {initial_state_file}")
    adm1_state = get_state_zero_from_initial_state(str(initial_state_file))

    # Create initial state dictionaries for digesters
    digester1_initial = {
        "adm1_state": adm1_state,
        "Q_substrates": [15, 10, 0, 0, 0, 0, 0, 0, 0, 0],
    }

    # For second digester, use same initial state but no direct substrate feed
    digester2_initial = {
        "adm1_state": adm1_state,
        "Q_substrates": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    }

    # Create plant
    plant = BiogasPlant("Two-Stage Digester Plant")

    # Create first digester (main fermenter)
    digester1 = Digester(
        component_id="digester_1", feedstock=feedstock, V_liq=1977.0, V_gas=304.0, T_ad=308.15, name="Main Digester"
    )
    digester1.initialize(digester1_initial)

    # Create second digester (post-digester)
    digester2 = Digester(
        component_id="digester_2", feedstock=feedstock, V_liq=1000.0, V_gas=150.0, T_ad=308.15, name="Post Digester"
    )
    digester2.initialize(digester2_initial)

    # Create CHP unit
    chp = CHP(component_id="chp_1", P_el_nom=500.0, eta_el=0.40, eta_th=0.45, name="CHP Unit")

    # Create heating system for digester 1
    heating1 = HeatingSystem(
        component_id="heating_1", target_temperature=308.15, heat_loss_coefficient=0.5, name="Main Digester Heating"
    )

    # Create heating system for digester 2
    heating2 = HeatingSystem(
        component_id="heating_2", target_temperature=308.15, heat_loss_coefficient=0.3, name="Post Digester Heating"
    )

    # Add components to plant
    plant.add_component(digester1)
    plant.add_component(digester2)
    plant.add_component(chp)
    plant.add_component(heating1)
    plant.add_component(heating2)

    # Create connections
    # Digester 1 -> Digester 2 (liquid flow)
    plant.add_connection(Connection(from_component="digester_1", to_component="digester_2", connection_type="liquid"))

    # in initialize_plant all gas_storages are automatically connected to all chps
    # Digester 1 -> CHP (biogas)
    # plant.add_connection(Connection(from_component="digester_1", to_component="chp_1", connection_type="gas"))

    # Digester 2 -> CHP (biogas)
    # plant.add_connection(Connection(from_component="digester_2", to_component="chp_1", connection_type="gas"))

    # CHP -> Heating 1 (waste heat)
    plant.add_connection(Connection(from_component="chp_1", to_component="heating_1", connection_type="heat"))

    # CHP -> Heating 2 (waste heat)
    plant.add_connection(Connection(from_component="chp_1", to_component="heating_2", connection_type="heat"))

    return plant


def main():
    """Run example simulation."""
    # Create plant
    print("=" * 70)
    print("Creating two-stage biogas plant...")
    print("=" * 70)
    plant = create_two_stage_plant()

    # Print plant summary
    print("\n" + "=" * 70)
    print(plant.get_summary())
    print("=" * 70)

    # Save initial configuration to JSON
    output_path = Path(__file__).parent.parent / "output"
    output_path.mkdir(exist_ok=True)

    config_file = output_path / "plant_config_initial.json"
    plant.to_json(str(config_file))
    print(f"\nInitial configuration saved to: {config_file}")

    # Run simulation
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
