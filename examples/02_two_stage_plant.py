#!/usr/bin/env python3
# ============================================================================
# examples/two_stage_simulation.py
# ============================================================================
"""
Example: Two-stage digester in series with CHP and heating, configured using the PlantConfigurator.

Demonstrates:
- Two digesters in series (hydrolysis + methanogenesis)
- Pump for substrate feeding and transfer
- Mixers for both digesters
- Automatic gas storage per digester
- CHP unit consuming biogas
- Heating systems for both digesters
- Complete energy integration

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
    from pyadm1.components.mechanical.mixer import Mixer
    from pyadm1.components.mechanical.pump import Pump

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
        print(f"\nLoading initial state from: {initial_state_file}")
        adm1_state = get_state_zero_from_initial_state(str(initial_state_file))
    else:
        print(f"   Warning: Initial state file not found at {initial_state_file}")
        adm1_state = None

    # Substrate feed rates
    digester1_feed = [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]  # Fresh substrate
    digester2_feed = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Only effluent from digester 1

    # Step 3: Create biogas plant
    print("\n" + "=" * 70)
    print("PLANT CONFIGURATION")
    print("=" * 70)

    plant = BiogasPlant("Two-Stage Plant with Mechanical Components")
    configurator = PlantConfigurator(plant, feedstock)

    # Step 4: Add biological components (digesters)
    print("\n1. Adding digesters...")
    print("   - Digester 1 (Hydrolysis): 1977 m³ @ 45°C (thermophilic)")
    print("   - Digester 2 (Methanogenesis): 1000 m³ @ 35°C (mesophilic)")

    configurator.add_digester(
        digester_id="digester_1",
        V_liq=1977.0,
        V_gas=304.0,
        T_ad=318.15,  # 45°C thermophilic for better hydrolysis
        name="Hydrolysis Digester",
        load_initial_state=True,
        initial_state_file=str(initial_state_file) if adm1_state else None,
        Q_substrates=digester1_feed,
    )
    configurator.add_digester(
        digester_id="digester_2",
        V_liq=1000.0,
        V_gas=150.0,
        T_ad=308.15,  # 35°C mesophilic for methanogenesis
        name="Methanogenesis Digester",
        load_initial_state=True,
        initial_state_file=str(initial_state_file) if adm1_state else None,
        Q_substrates=digester2_feed,
    )

    # Step 5: Add mechanical components
    print("\n2. Adding mechanical components...")

    # Feed pump (substrate feeding into digester 1)
    print("   - Feed pump: 30 m³/h capacity")
    feed_pump = Pump(
        component_id="feed_pump",
        pump_type="progressive_cavity",  # Good for viscous substrates
        Q_nom=30.0,  # m³/h nominal flow
        pressure_head=5.0,  # m (low pressure, gravity assist)
        name="Substrate Feed Pump",
    )
    plant.add_component(feed_pump)

    # Transfer pump (digester 1 → digester 2)
    print("   - Transfer pump: 25 m³/h capacity")
    transfer_pump = Pump(
        component_id="transfer_pump",
        pump_type="progressive_cavity",
        Q_nom=25.0,  # m³/h
        pressure_head=8.0,  # m (higher head for transfer)
        name="Digester Transfer Pump",
    )
    plant.add_component(transfer_pump)

    # Mixers for both digesters
    print("   - Mixer 1: Propeller type, 15 kW")
    mixer_1 = Mixer(
        component_id="mixer_1",
        mixer_type="propeller",
        tank_volume=1977.0,
        mixing_intensity="high",  # High for thermophilic digester
        power_installed=15.0,  # kW
        intermittent=True,
        on_time_fraction=0.25,  # Run 25% of time
        name="Hydrolysis Mixer",
    )
    plant.add_component(mixer_1)

    print("   - Mixer 2: Propeller type, 10 kW")
    mixer_2 = Mixer(
        component_id="mixer_2",
        mixer_type="propeller",
        tank_volume=1000.0,
        mixing_intensity="medium",  # Lower for methanogenesis
        power_installed=10.0,  # kW
        intermittent=True,
        on_time_fraction=0.25,
        name="Methanogenesis Mixer",
    )
    plant.add_component(mixer_2)

    # Step 6: Add energy components
    print("\n3. Adding energy components...")
    print("   - CHP: 500 kW electrical, 40% efficiency")
    configurator.add_chp(
        chp_id="chp_1",
        P_el_nom=500.0,
        eta_el=0.40,
        eta_th=0.45,
        name="Main CHP Unit",
    )

    print("   - Heating system 1: For hydrolysis digester (45°C)")
    configurator.add_heating(
        heating_id="heating_1",
        target_temperature=318.15,  # 45°C
        heat_loss_coefficient=0.5,  # Higher loss due to higher temp
        name="Hydrolysis Heating",
    )

    print("   - Heating system 2: For methanogenesis digester (35°C)")
    configurator.add_heating(
        heating_id="heating_2",
        target_temperature=308.15,  # 35°C
        heat_loss_coefficient=0.3,  # Lower loss
        name="Methanogenesis Heating",
    )

    # Step 7: Connect components
    print("\n4. Connecting components...")
    print("   - Liquid flow: Digester 1 → Digester 2")
    configurator.connect("digester_1", "digester_2", "liquid")

    print("   - Gas flows: Both digesters → CHP → Flare")
    configurator.auto_connect_digester_to_chp("digester_1", "chp_1")
    configurator.auto_connect_digester_to_chp("digester_2", "chp_1")

    print("   - Heat flows: CHP → Both heating systems")
    configurator.auto_connect_chp_to_heating("chp_1", "heating_1")
    configurator.auto_connect_chp_to_heating("chp_1", "heating_2")

    # Step 8: Initialize plant
    print("\n5. Initializing plant...")
    plant.initialize()

    # Print plant summary
    print("\n" + "=" * 70)
    print(plant.get_summary())
    print("=" * 70)

    # Step 9: Save initial configuration
    output_path = Path(__file__).parent.parent / "output"
    output_path.mkdir(exist_ok=True)
    config_file = output_path / "two_stage_plant_config.json"
    plant.to_json(str(config_file))
    print(f"\nInitial configuration saved to: {config_file}")

    # Step 10: Run simulation
    print("\n" + "=" * 70)
    print("SIMULATION")
    print("=" * 70)
    print("\nStarting simulation...")
    print("  Duration: 10 days")
    print("  Time step: 1 hour")
    print("  Save interval: 1 day")

    duration = 10.0  # 10 days
    dt = 1.0 / 24.0  # 1 hour time step
    save_interval = 1.0  # Save daily

    results = plant.simulate(duration=duration, dt=dt, save_interval=save_interval)

    print(f"\nSimulation complete. Generated {len(results)} result snapshots.")

    # Step 11: Analyze and display results
    print("\n" + "=" * 70)
    print("RESULTS ANALYSIS")
    print("=" * 70)

    final_result = results[-1]
    print(f"\nFinal State (Day {final_result['time']:.1f}):")
    print("-" * 70)

    # Digester 1 (Hydrolysis)
    if "digester_1" in final_result["components"]:
        d1 = final_result["components"]["digester_1"]
        print("\nHydrolysis Digester:")
        print(f"  Biogas production:    {d1.get('Q_gas', 0):>10.1f} m³/d")
        print(f"  Methane production:   {d1.get('Q_ch4', 0):>10.1f} m³/d")
        print(f"  pH:                   {d1.get('pH', 0):>10.2f}")
        print(f"  VFA:                  {d1.get('VFA', 0):>10.2f} g/L")
        print(f"  Temperature:          {318.15-273.15:>10.1f} °C")

    # Digester 2 (Methanogenesis)
    if "digester_2" in final_result["components"]:
        d2 = final_result["components"]["digester_2"]
        print("\nMethanogenesis Digester:")
        print(f"  Biogas production:    {d2.get('Q_gas', 0):>10.1f} m³/d")
        print(f"  Methane production:   {d2.get('Q_ch4', 0):>10.1f} m³/d")
        print(f"  pH:                   {d2.get('pH', 0):>10.2f}")
        print(f"  VFA:                  {d2.get('VFA', 0):>10.2f} g/L")
        print(f"  Temperature:          {308.15-273.15:>10.1f} °C")

    # Combined biogas production
    total_gas = final_result["components"]["digester_1"].get("Q_gas", 0) + final_result["components"]["digester_2"].get(
        "Q_gas", 0
    )
    total_ch4 = final_result["components"]["digester_1"].get("Q_ch4", 0) + final_result["components"]["digester_2"].get(
        "Q_ch4", 0
    )

    print("\nTotal Plant Production:")
    print(f"  Total biogas:         {total_gas:>10.1f} m³/d")
    print(f"  Total methane:        {total_ch4:>10.1f} m³/d")
    print(f"  Methane content:      {100*total_ch4/total_gas if total_gas > 0 else 0:>10.1f} %")

    # CHP performance
    if "chp_1" in final_result["components"]:
        chp = final_result["components"]["chp_1"]
        print("\nCHP Performance:")
        print(f"  Electrical output:    {chp.get('P_el', 0):>10.1f} kW")
        print(f"  Thermal output:       {chp.get('P_th', 0):>10.1f} kW")
        print(f"  Gas consumption:      {chp.get('Q_gas_consumed', 0):>10.1f} m³/d")
        print(f"  Operating hours:      {duration*24:>10.1f} h")

    # Mechanical components
    if "mixer_1" in final_result["components"]:
        m1 = final_result["components"]["mixer_1"]
        print("\nHydrolysis Mixer:")
        print(f"  Power consumption:    {m1.get('P_average', 0):>10.2f} kW")
        print(f"  Mixing quality:       {m1.get('mixing_quality', 0):>10.2f}")
        print(f"  Reynolds number:      {m1.get('reynolds_number', 0):>10.0f}")

    if "mixer_2" in final_result["components"]:
        m2 = final_result["components"]["mixer_2"]
        print("\nMethanogenesis Mixer:")
        print(f"  Power consumption:    {m2.get('P_average', 0):>10.2f} kW")
        print(f"  Mixing quality:       {m2.get('mixing_quality', 0):>10.2f}")
        print(f"  Reynolds number:      {m2.get('reynolds_number', 0):>10.0f}")

    # Energy balance
    print("\n" + "=" * 70)
    print("ENERGY BALANCE")
    print("=" * 70)

    if "chp_1" in final_result["components"]:
        chp = final_result["components"]["chp_1"]
        P_el = chp.get("P_el", 0)
        P_th = chp.get("P_th", 0)

        # Mixer consumption
        P_mix_1 = final_result["components"]["mixer_1"].get("P_average", 0) if "mixer_1" in final_result["components"] else 0
        P_mix_2 = final_result["components"]["mixer_2"].get("P_average", 0) if "mixer_2" in final_result["components"] else 0
        P_mix_total = P_mix_1 + P_mix_2

        # Heating demand
        Q_heat_1 = (
            final_result["components"]["heating_1"].get("Q_heat_supplied", 0)
            if "heating_1" in final_result["components"]
            else 0
        )
        Q_heat_2 = (
            final_result["components"]["heating_2"].get("Q_heat_supplied", 0)
            if "heating_2" in final_result["components"]
            else 0
        )
        Q_heat_total = Q_heat_1 + Q_heat_2

        print("\nEnergy Production:")
        print(f"  Electrical (gross):   {P_el:>10.1f} kW")
        print(f"  Thermal:              {P_th:>10.1f} kW")

        print("\nParasitic Load:")
        print(f"  Mixer 1:              {P_mix_1:>10.2f} kW")
        print(f"  Mixer 2:              {P_mix_2:>10.2f} kW")
        print(f"  Pumps (estimated):    {2.0:>10.2f} kW")
        print(f"  Total parasitic:      {P_mix_total + 2.0:>10.2f} kW")

        print(f"\nNet Electrical Output:  {P_el - P_mix_total - 2.0:>10.1f} kW")

        print("\nHeat Utilization:")
        print(f"  Heating demand:       {Q_heat_total:>10.1f} kW")
        print(f"  CHP thermal supply:   {P_th:>10.1f} kW")
        print(f"  Heat coverage:        {100*P_th/Q_heat_total if Q_heat_total > 0 else 0:>10.1f} %")

    # Process stability
    print("\n" + "=" * 70)
    print("PROCESS STABILITY ASSESSMENT")
    print("=" * 70)

    d1_ph = final_result["components"]["digester_1"].get("pH", 0) if "digester_1" in final_result["components"] else 0
    d1_vfa = final_result["components"]["digester_1"].get("VFA", 0) if "digester_1" in final_result["components"] else 0
    d1_tac = final_result["components"]["digester_1"].get("TAC", 0) if "digester_1" in final_result["components"] else 0

    d2_ph = final_result["components"]["digester_2"].get("pH", 0) if "digester_2" in final_result["components"] else 0
    d2_vfa = final_result["components"]["digester_2"].get("VFA", 0) if "digester_2" in final_result["components"] else 0
    d2_tac = final_result["components"]["digester_2"].get("TAC", 0) if "digester_2" in final_result["components"] else 0

    print("\nDigester 1 (Hydrolysis):")
    print(f"  pH stability:         {'GOOD' if 6.8 <= d1_ph <= 7.5 else 'CHECK'}")
    print(f"  VFA level:            {'GOOD' if d1_vfa < 3.0 else 'HIGH'}")
    print(
        f"  FOS/TAC ratio:        {d1_vfa/d1_tac if d1_tac > 0 else 0:.3f} {'(Stable)' if d1_vfa/d1_tac < 0.3 else '(Monitor)'}"
    )

    print("\nDigester 2 (Methanogenesis):")
    print(f"  pH stability:         {'GOOD' if 6.8 <= d2_ph <= 7.5 else 'CHECK'}")
    print(f"  VFA level:            {'GOOD' if d2_vfa < 3.0 else 'HIGH'}")
    print(
        f"  FOS/TAC ratio:        {d2_vfa/d2_tac if d2_tac > 0 else 0:.3f} {'(Stable)' if d2_vfa/d2_tac < 0.3 else '(Monitor)'}"
    )

    # Save final configuration
    final_config_file = output_path / "two_stage_plant_final.json"
    plant.to_json(str(final_config_file))
    print(f"\n💾 Final configuration saved to: {final_config_file}")

    print("\n" + "=" * 70)
    print("✅ Simulation completed successfully!")
    print("=" * 70 + "\n")

    return plant, results


if __name__ == "__main__":
    plant, results = main()
