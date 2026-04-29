#!/usr/bin/env python3
# examples/05_parallel_two_stage_simulation.py
"""
Parallel parameter sweeps and Monte Carlo analysis for ADM1 biogas plants.

Demonstrates:
    - Building a base ADM1 model
    - Running multiple feed-rate scenarios in parallel
    - Single-parameter sweep on the maximum acetate-uptake rate (k_m_ac)
    - Monte Carlo uncertainty quantification on k_m_ac

Usage:
    python examples/05_parallel_two_stage_simulation.py
"""

from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np


def main():
    from pyadm1 import Feedstock
    from pyadm1.core.adm1 import ADM1, STATE_SIZE
    from pyadm1.simulation.parallel import (
        MonteCarloConfig,
        ParallelSimulator,
        ParameterSweepConfig,
    )

    print("=" * 70)
    print("PyADM1 Parallel Simulation Example")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Base ADM1 model
    # ------------------------------------------------------------------
    print("\n1. Building base ADM1 model...")

    feedstock = Feedstock(
        ["maize_silage_milk_ripeness", "swine_manure"],
        feeding_freq=24,
        total_simtime=15,
    )
    base_feed = [11.4, 6.1, 0, 0, 0, 0, 0, 0, 0, 0]

    adm1 = ADM1(feedstock, V_liq=1200.0, V_gas=216.0, T_ad=315.15)
    adm1.set_influent_dataframe(feedstock.get_influent_dataframe(Q=base_feed))
    adm1.create_influent(base_feed, 0)

    initial_state = [0.01] * STATE_SIZE
    initial_state[37:41] = [1.02e-5, 0.65, 0.33, 0.65 + 0.33 + 1.02e-5]

    parallel = ParallelSimulator(adm1, n_workers=4, verbose=True)

    # ------------------------------------------------------------------
    # Step 2: Compare four feed rates in parallel
    # ------------------------------------------------------------------
    print("\n2. Running feed-rate comparison...")

    feed_scenarios = [
        ("Low Feed", [8.0, 4.0, 0, 0, 0, 0, 0, 0, 0, 0]),
        ("Base Feed", [11.4, 6.1, 0, 0, 0, 0, 0, 0, 0, 0]),
        ("High Feed", [15.0, 8.0, 0, 0, 0, 0, 0, 0, 0, 0]),
        ("Very High Feed", [20.0, 10.0, 0, 0, 0, 0, 0, 0, 0, 0]),
    ]
    scenarios = [{"Q": Q} for _, Q in feed_scenarios]

    start = time.time()
    results = parallel.run_scenarios(
        scenarios=scenarios,
        duration=10.0,
        initial_state=initial_state,
        dt=1.0,
        compute_metrics=True,
    )
    print(f"   Done in {time.time() - start:.1f} s")

    print("\n   Feed-rate results:")
    print("   " + "-" * 60)
    print(f"   {'Scenario':<18s} {'Q_total':>8s} {'Q_gas':>10s} {'Q_ch4':>10s} {'pH':>6s}")
    for (label, Q), result in zip(feed_scenarios, results):
        if not result.success:
            print(f"   {label:<18s} FAILED: {result.error[:60]}")
            continue
        Q_total = sum(Q)
        m = result.metrics
        print(
            f"   {label:<18s} {Q_total:>8.1f} {m.get('Q_gas', 0):>10.1f} " f"{m.get('Q_ch4', 0):>10.1f} {m.get('pH', 0):>6.2f}"
        )

    # ------------------------------------------------------------------
    # Step 3: Parameter sweep — acetate uptake rate (k_m_ac)
    # ------------------------------------------------------------------
    print("\n3. Parameter sweep over k_m_ac...")

    sweep_config = ParameterSweepConfig(
        parameter_name="k_m_ac",
        values=[5.0, 7.0, 8.0, 10.0, 13.0],
        other_params={"Q": base_feed},
    )

    sweep_results = parallel.parameter_sweep(
        config=sweep_config,
        duration=10.0,
        initial_state=initial_state,
    )

    print("\n   Sweep results:")
    print("   " + "-" * 60)
    print(f"   {'k_m_ac':>8s} {'Q_gas':>10s} {'Q_ch4':>10s} {'pH':>6s}")
    for k_val, r in zip(sweep_config.values, sweep_results):
        if r.success:
            m = r.metrics
            print(f"   {k_val:>8.2f} {m.get('Q_gas', 0):>10.1f} {m.get('Q_ch4', 0):>10.1f} {m.get('pH', 0):>6.2f}")

    if sweep_results:
        ch4_values = [r.metrics.get("Q_ch4", 0) for r in sweep_results if r.success]
        if ch4_values:
            best_idx = int(np.argmax(ch4_values))
            print(f"\n   Best k_m_ac = {sweep_config.values[best_idx]:.2f} (CH4 = {ch4_values[best_idx]:.1f} m³/d)")

    # ------------------------------------------------------------------
    # Step 4: Monte Carlo uncertainty
    # ------------------------------------------------------------------
    print("\n4. Monte Carlo uncertainty (50 samples)...")

    mc_config = MonteCarloConfig(
        n_samples=50,
        parameter_distributions={"k_m_ac": (8.0, 1.0)},
        fixed_params={"Q": base_feed},
        seed=42,
    )

    mc_results = parallel.monte_carlo(
        config=mc_config,
        duration=10.0,
        initial_state=initial_state,
    )
    summary = parallel.summarize_results(mc_results)

    print(f"\n   Success rate: {summary['success_rate'] * 100:.1f}%")
    if "metrics" in summary and "Q_ch4" in summary["metrics"]:
        s = summary["metrics"]["Q_ch4"]
        print("\n   Methane production [m³/d]:")
        print(f"     mean ± std : {s['mean']:.1f} ± {s['std']:.1f}")
        print(f"     min  / max : {s['min']:.1f} / {s['max']:.1f}")
        print(f"     median     : {s['median']:.1f}")

    print("\n" + "=" * 70)
    print("Parallel demonstration completed successfully!")
    print("=" * 70)

    return mc_results


if __name__ == "__main__":
    main()
