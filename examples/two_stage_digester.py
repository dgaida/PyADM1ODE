# ============================================================================
# Example usage script: examples/two_stage_digester.py
# ============================================================================
"""
Example: Two-stage digester in series with CHP and heating.

This example demonstrates:
- Two digesters in series
- CHP unit consuming biogas
- Heating system using CHP waste heat
"""


def create_two_stage_plant():
    """Create example two-stage biogas plant configuration."""
    from pyadm1.plant.plant_model import BiogasPlant
    from pyadm1.plant.digester import Digester
    from pyadm1.plant.chp import CHP
    from pyadm1.plant.heating import HeatingSystem
    from pyadm1.plant.connection import Connection
    from pyadm1.substrates.feedstock import Feedstock

    # Initialize feedstock
    feeding_freq = 48  # hours
    feedstock = Feedstock(feeding_freq)

    # Create plant
    plant = BiogasPlant("Two-Stage Digester Plant")

    # Create first digester (main fermenter)
    digester1 = Digester(
        component_id="digester_1", feedstock=feedstock, V_liq=1977.0, V_gas=304.0, T_ad=308.15, name="Main Digester"
    )

    # Create second digester (post-digester)
    digester2 = Digester(
        component_id="digester_2", feedstock=feedstock, V_liq=1000.0, V_gas=150.0, T_ad=308.15, name="Post Digester"
    )

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

    # Digester 1 -> CHP (biogas)
    plant.add_connection(Connection(from_component="digester_1", to_component="chp_1", connection_type="gas"))

    # Digester 2 -> CHP (biogas)
    plant.add_connection(Connection(from_component="digester_2", to_component="chp_1", connection_type="gas"))

    # CHP -> Heating 1 (waste heat)
    plant.add_connection(Connection(from_component="chp_1", to_component="heating_1", connection_type="heat"))

    # CHP -> Heating 2 (waste heat)
    plant.add_connection(Connection(from_component="chp_1", to_component="heating_2", connection_type="heat"))

    return plant


def main():
    """Run example simulation."""
    # Create plant
    plant = create_two_stage_plant()

    # Initialize all components with default states
    plant.initialize()

    # Set initial substrate feed for first digester
    digester1 = plant.components["digester_1"]
    digester1.Q_substrates = [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]

    # Print plant summary
    print(plant.get_summary())

    # Save configuration to JSON
    plant.to_json("plant_config.json")

    # Run simulation
    print("\nStarting simulation...")
    results = plant.simulate(duration=10.0, dt=1.0 / 24.0, save_interval=1.0)  # 10 days  # 1 hour time step  # Save daily

    print(f"\nSimulation complete. Generated {len(results)} result snapshots.")

    # Print final results
    final_result = results[-1]
    print(f"\n=== Final Results (t={final_result['time']:.1f} days) ===")

    for comp_id, comp_results in final_result["components"].items():
        comp = plant.components[comp_id]
        print(f"\n{comp.name}:")
        for key, value in comp_results.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.2f}")

    # Save plant state
    plant.to_json("plant_config_final.json")

    return plant, results


if __name__ == "__main__":
    plant, results = main()
