#!/usr/bin/env python3
# ============================================================================
# examples/parallel_two_stage_simulation.py
# ============================================================================
"""
Example: Parallel simulation of two-stage biogas plant with parameter sweeps.

Demonstrates:
- Building a two-stage plant with PlantConfigurator
- Running multiple scenarios in parallel with ParallelSimulator
- Parameter sweeps for optimization
- Monte Carlo analysis for uncertainty quantification
- Result visualization and comparison

Usage:
    python examples/parallel_two_stage_simulation.py
"""

from pathlib import Path
import numpy as np
import time


def main():
    """Run parallel simulations of two-stage biogas plant."""
    # Import required modules
    from pyadm1.configurator.plant_builder import BiogasPlant
    from pyadm1.substrates.feedstock import Feedstock
    from pyadm1.core.adm1 import get_state_zero_from_initial_state
    from pyadm1.configurator.plant_configurator import PlantConfigurator
    from pyadm1.simulation.parallel import ParallelSimulator, ParameterSweepConfig, MonteCarloConfig

    print("=" * 70)
    print("Parallel Two-Stage Biogas Plant Simulation")
    print("=" * 70)

    # ========================================================================
    # Step 1: Create Base Plant Model
    # ========================================================================
    print("\n1. Creating base two-stage plant model...")

    feeding_freq = 48
    feedstock = Feedstock(feeding_freq=feeding_freq)

    # Load initial state
    data_path = Path(__file__).parent.parent / "data" / "initial_states"
    initial_state_file = data_path / "digester_initial8.csv"

    if initial_state_file.exists():
        print(f"   Loading initial state from: {initial_state_file}")
        adm1_state = get_state_zero_from_initial_state(str(initial_state_file))
    else:
        print("   Warning: Initial state file not found, using defaults")
        adm1_state = [0.01] * 37

    # Base substrate feed rates
    base_feed_hydro = [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]
    base_feed_main = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # Create plant
    plant = BiogasPlant("Parallel Test Plant")
    configurator = PlantConfigurator(plant, feedstock)

    # Add components
    print("   Adding digesters...")
    configurator.add_digester(
        digester_id="digester_1", V_liq=1977.0, V_gas=304.0, T_ad=308.15, name="Main Digester", Q_substrates=base_feed_hydro
    )
    configurator.add_digester(
        digester_id="digester_2", V_liq=1000.0, V_gas=150.0, T_ad=308.15, name="Post Digester", Q_substrates=base_feed_main
    )

    print("   Adding CHP and heating...")
    configurator.add_chp(chp_id="chp_1", P_el_nom=500.0, name="CHP Unit")
    configurator.add_heating(heating_id="heating_1", target_temperature=308.15, name="Main Digester Heating")
    configurator.add_heating(heating_id="heating_2", target_temperature=308.15, name="Post Digester Heating")

    # Connect components
    print("   Connecting components...")
    configurator.connect("digester_1", "digester_2", "liquid")
    configurator.auto_connect_digester_to_chp("digester_1", "chp_1")
    configurator.auto_connect_digester_to_chp("digester_2", "chp_1")
    configurator.auto_connect_chp_to_heating("chp_1", "heating_1")
    configurator.auto_connect_chp_to_heating("chp_1", "heating_2")

    # Initialize
    print("   Initializing plant...")
    plant.initialize()

    print("\n   ✓ Plant model created successfully")

    # ========================================================================
    # Step 2: Define Scenarios for Parallel Execution
    # ========================================================================
    print("\n2. Defining simulation scenarios...")

    # Scenario 1: Different substrate feed rates
    feed_scenarios = [
        {"name": "Low Feed", "Q_digester_1": [10, 8, 0, 0, 0, 0, 0, 0, 0, 0]},
        {"name": "Base Feed", "Q_digester_1": [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]},
        {"name": "High Feed", "Q_digester_1": [20, 12, 0, 0, 0, 0, 0, 0, 0, 0]},
        {"name": "Very High Feed", "Q_digester_1": [25, 15, 0, 0, 0, 0, 0, 0, 0, 0]},
    ]

    print(f"   Defined {len(feed_scenarios)} feed rate scenarios")

    # ========================================================================
    # Step 3: Run Basic Parallel Scenarios
    # ========================================================================
    print("\n3. Running basic parallel scenarios...")
    print("   (This demonstrates running different feed rates in parallel)")

    # Note: For this example, we'll simulate individual digesters
    # In a full implementation, you'd simulate the entire plant

    # Get the first digester's ADM1 model for parallel simulation
    digester_1 = plant.components["digester_1"]
    adm1_model = digester_1.adm1

    # Create parallel simulator
    parallel = ParallelSimulator(adm1_model, n_workers=4, verbose=True)

    # Prepare scenarios for ParallelSimulator
    scenarios = [{"Q": scenario["Q_digester_1"]} for scenario in feed_scenarios]

    # Run parallel simulations
    print("\n   Starting parallel execution...")
    start_time = time.time()

    results = parallel.run_scenarios(
        scenarios=scenarios,
        duration=10.0,  # 10 days
        initial_state=adm1_state,
        dt=1.0 / 24.0,  # 1 hour time step
        compute_metrics=True,
        save_time_series=False,
    )

    elapsed = time.time() - start_time
    print(f"\n   ✓ Parallel execution completed in {elapsed:.1f} seconds")

    # Display results
    print("\n" + "=" * 70)
    print("SCENARIO COMPARISON RESULTS")
    print("=" * 70)

    for i, result in enumerate(results):
        if result.success:
            scenario_name = feed_scenarios[i]["name"]
            Q_total = sum(scenarios[i]["Q"])

            print(f"\n{scenario_name}:")
            print(f"  Feed Rate: {Q_total:.1f} m³/d")
            print(f"  Biogas:    {result.metrics.get('Q_gas', 0):.1f} m³/d")
            print(f"  Methane:   {result.metrics.get('Q_ch4', 0):.1f} m³/d")
            print(f"  CH4 %:     {result.metrics.get('CH4_content', 0)*100:.1f}%")
            print(f"  pH:        {result.metrics.get('pH', 0):.2f}")
            print(f"  VFA:       {result.metrics.get('VFA', 0):.2f} g/L")
            print(f"  FOS/TAC:   {result.metrics.get('FOS_TAC', 0):.3f}")
            print(f"  HRT:       {result.metrics.get('HRT', 0):.1f} days")
            print(f"  Exec Time: {result.execution_time:.2f} seconds")
        else:
            print(f"\n{feed_scenarios[i]['name']}: FAILED")
            print(f"  Error: {result.error[:100]}")

    # ========================================================================
    # Step 4: Parameter Sweep Example
    # ========================================================================
    print("\n" + "=" * 70)
    print("4. Running parameter sweep (Disintegration Rate k_dis)")
    print("=" * 70)

    # Sweep disintegration rate parameter
    sweep_config = ParameterSweepConfig(
        parameter_name="k_dis", values=[0.10, 0.14, 0.18, 0.22, 0.26, 0.30], other_params={"Q": base_feed_hydro}
    )

    print("\n   Testing 6 different k_dis values...")
    sweep_results = parallel.parameter_sweep(
        config=sweep_config, duration=10.0, initial_state=adm1_state, compute_metrics=True
    )

    print("\n   Parameter Sweep Results:")
    print("   " + "-" * 60)
    print(f"   {'k_dis':>8} | {'Q_gas':>10} | {'Q_ch4':>10} | {'pH':>6} | {'VFA':>8}")
    print("   " + "-" * 60)

    for i, result in enumerate(sweep_results):
        if result.success:
            k_dis = sweep_config.values[i]
            print(
                f"   {k_dis:>8.2f} | "
                f"{result.metrics.get('Q_gas', 0):>10.1f} | "
                f"{result.metrics.get('Q_ch4', 0):>10.1f} | "
                f"{result.metrics.get('pH', 0):>6.2f} | "
                f"{result.metrics.get('VFA', 0):>8.2f}"
            )

    # Find optimal k_dis for maximum methane production
    ch4_productions = [r.metrics.get("Q_ch4", 0) for r in sweep_results if r.success]
    if ch4_productions:
        best_idx = np.argmax(ch4_productions)
        best_k_dis = sweep_config.values[best_idx]
        best_ch4 = ch4_productions[best_idx]
        print(f"\n   ✓ Optimal k_dis = {best_k_dis:.2f} (CH4 = {best_ch4:.1f} m³/d)")

    # ========================================================================
    # Step 5: Multi-Parameter Sweep Example
    # ========================================================================
    print("\n" + "=" * 70)
    print("5. Running multi-parameter sweep (k_dis vs Feed Rate)")
    print("=" * 70)

    parameter_configs = {
        "k_dis": [0.14, 0.18, 0.22],
        # Vary total feed by adjusting first substrate
        "Q_substrate_0": [12, 15, 18],  # Corn silage feed rate
    }

    # Build scenarios manually since Q is a list
    multi_scenarios = []
    for k_dis in parameter_configs["k_dis"]:
        for q0 in parameter_configs["Q_substrate_0"]:
            Q = [q0, 10, 0, 0, 0, 0, 0, 0, 0, 0]
            multi_scenarios.append({"k_dis": k_dis, "Q": Q})

    print(f"\n   Testing {len(multi_scenarios)} parameter combinations...")
    multi_results = parallel.run_scenarios(
        scenarios=multi_scenarios, duration=10.0, initial_state=adm1_state, compute_metrics=True
    )

    print("\n   Multi-Parameter Sweep Results:")
    print("   " + "-" * 70)
    print(f"   {'k_dis':>8} | {'Feed':>8} | {'Q_gas':>10} | {'Q_ch4':>10} | {'Yield':>8}")
    print("   " + "-" * 70)

    scenario_idx = 0
    for k_dis in parameter_configs["k_dis"]:
        for q0 in parameter_configs["Q_substrate_0"]:
            result = multi_results[scenario_idx]
            if result.success:
                Q_total = q0 + 10
                q_ch4 = result.metrics.get("Q_ch4", 0)
                specific_yield = q_ch4 / Q_total if Q_total > 0 else 0
                print(
                    f"   {k_dis:>8.2f} | "
                    f"{Q_total:>8.1f} | "
                    f"{result.metrics.get('Q_gas', 0):>10.1f} | "
                    f"{q_ch4:>10.1f} | "
                    f"{specific_yield:>8.3f}"
                )
            scenario_idx += 1

    # ========================================================================
    # Step 6: Monte Carlo Analysis
    # ========================================================================
    print("\n" + "=" * 70)
    print("6. Running Monte Carlo uncertainty analysis")
    print("=" * 70)

    mc_config = MonteCarloConfig(
        n_samples=50,
        parameter_distributions={
            "k_dis": (0.18, 0.03),  # mean ± std
            # Note: Y_su would need to be applied differently in practice
        },
        fixed_params={"Q": base_feed_hydro},
        seed=42,
    )

    print("\n   Running 50 Monte Carlo samples...")
    print("   (Testing parameter uncertainty)")

    mc_results = parallel.monte_carlo(config=mc_config, duration=10.0, initial_state=adm1_state, compute_metrics=True)

    # Summarize Monte Carlo results
    summary = parallel.summarize_results(mc_results)

    print("\n   Monte Carlo Summary Statistics:")
    print("   " + "-" * 60)

    if "metrics" in summary:
        for metric_name, stats in summary["metrics"].items():
            print(f"\n   {metric_name}:")
            print(f"      Mean:   {stats['mean']:>10.2f}")
            print(f"      Std:    {stats['std']:>10.2f}")
            print(f"      Min:    {stats['min']:>10.2f}")
            print(f"      Max:    {stats['max']:>10.2f}")
            print(f"      Median: {stats['median']:>10.2f}")
            print(f"      Q25:    {stats['q25']:>10.2f}")
            print(f"      Q75:    {stats['q75']:>10.2f}")

    print(f"\n   Success Rate: {summary['success_rate']*100:.1f}%")

    # ========================================================================
    # Step 7: Statistical Analysis
    # ========================================================================
    print("\n" + "=" * 70)
    print("7. Statistical Analysis of All Results")
    print("=" * 70)

    # Combine all successful results
    all_results = results + sweep_results + multi_results + mc_results
    successful_results = [r for r in all_results if r.success]

    print(f"\n   Total simulations run: {len(all_results)}")
    print(f"   Successful: {len(successful_results)}")
    print(f"   Failed: {len(all_results) - len(successful_results)}")
    print(f"   Success rate: {len(successful_results)/len(all_results)*100:.1f}%")

    # Compute overall statistics
    if successful_results:
        ch4_values = [r.metrics.get("Q_ch4", 0) for r in successful_results]
        gas_values = [r.metrics.get("Q_gas", 0) for r in successful_results]

        print("\n   Methane Production Range:")
        print(f"      Minimum: {min(ch4_values):.1f} m³/d")
        print(f"      Maximum: {max(ch4_values):.1f} m³/d")
        print(f"      Mean:    {np.mean(ch4_values):.1f} m³/d")
        print(f"      Std Dev: {np.std(ch4_values):.1f} m³/d")

        print("\n   Total Biogas Production Range:")
        print(f"      Minimum: {min(gas_values):.1f} m³/d")
        print(f"      Maximum: {max(gas_values):.1f} m³/d")
        print(f"      Mean:    {np.mean(gas_values):.1f} m³/d")
        print(f"      Std Dev: {np.std(gas_values):.1f} m³/d")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("SIMULATION SUMMARY")
    print("=" * 70)

    total_time = time.time() - start_time

    print(f"\nTotal execution time: {total_time:.1f} seconds")
    print(f"Average time per simulation: {total_time/len(all_results):.2f} seconds")
    print("\nParallel efficiency:")
    print(f"  Workers used: {parallel.n_workers}")
    sequential_time = sum(r.execution_time for r in all_results)
    speedup = sequential_time / total_time if total_time > 0 else 0
    efficiency = speedup / parallel.n_workers if parallel.n_workers > 0 else 0
    print(f"  Theoretical sequential time: {sequential_time:.1f} seconds")
    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Parallel efficiency: {efficiency*100:.1f}%")

    print("\n" + "=" * 70)
    print("✓ Parallel simulation demonstration completed successfully!")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    results = main()
