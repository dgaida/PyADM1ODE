#!/usr/bin/env python3
# examples/05_parallel_two_stage_simulation.py
"""
Parallel feed-rate comparison, parameter sweep and Monte Carlo analysis for one digester.

The script simulates a single digester (one ADM1 model) many times side by
side with :class:`pyadm1.simulation.parallel.ParallelSimulator`. Despite the
file name it does not build a two-stage plant.

Demonstrates:
    - Building a base ADM1 model that serves as the template for every scenario
    - Running four feed-rate scenarios in parallel
    - Single-parameter sweep over the maximum acetate-uptake rate (k_m_ac)
    - Monte Carlo analysis of an uncertain k_m_ac

Units:
    - Feed rates are tonnes of fresh matter per day [t FM/d] under the default
      Feedstock convention (``simba_q_convention=True``).
    - Gas flows are dry gas at standard conditions (0 °C, 1.01325 bar) [Nm³/d].

Usage:
    python examples/05_parallel_two_stage_simulation.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Sweep values of k_m_ac as factors of the rate the model uses. The low end
# lies far below that rate, where the digester acidifies.
K_M_AC_SWEEP_FACTORS = [0.15, 0.3, 0.5, 1.0, 1.5]

# Relative standard deviation of k_m_ac in the Monte Carlo analysis.
K_M_AC_REL_STD = 0.10


def nominal_rate(adm1, name):
    """Return the kinetic rate the model uses at its operating temperature.

    The model keeps its rates at the reference temperature of 35 °C and
    corrects them to the operating temperature. A calibration value set on the
    model replaces the corrected rate.

    Args:
        adm1 (ADM1): Model whose temperature and calibration apply.
        name (str): Name of the rate, for example ``"k_m_ac"``.

    Returns:
        float: Rate at the operating temperature of the model [1/d].
    """
    from pyadm1.core import ADMParams

    calibration = adm1.get_calibration_parameters()
    if name in calibration:
        return float(calibration[name])
    rates = ADMParams.apply_temperature_corrections(
        ADMParams.get_kinetic_params(),
        ADMParams.get_temperature_factors(),
        adm1.T_ad,
    )
    return float(rates[name])


def main():
    from pyadm1 import Feedstock
    from pyadm1.core.adm1 import ADM1, get_state_zero_from_csv
    from pyadm1.simulation.parallel import (
        MonteCarloConfig,
        ParallelSimulator,
        ParameterSweepConfig,
    )

    print("=" * 70)
    print("PyADM1 Parallel Simulation Example")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Base ADM1 model (one digester)
    # ------------------------------------------------------------------
    print("\n1. Building base ADM1 model...")

    duration = 30.0  # days

    feedstock = Feedstock(["maize_silage_milk_ripeness", "swine_manure"])
    base_feed = [11.4, 6.1]  # t FM/d: maize silage, swine manure

    adm1 = ADM1(feedstock, V_liq=1200.0, V_gas=216.0, T_ad=315.15)

    initial_state = get_state_zero_from_csv(str(REPO_ROOT / "data" / "initial_states" / "digester_initial8.csv"))

    # The workers rebuild the model from the configuration of this base model.
    # The feed is not part of it: every scenario brings its own "Q".
    parallel = ParallelSimulator(adm1, n_workers=4, verbose=True)

    # ------------------------------------------------------------------
    # Step 2: Compare four feed rates in parallel
    # ------------------------------------------------------------------
    print("\n2. Running feed-rate comparison...")

    feed_scenarios = [
        ("Low Feed", [8.0, 4.0]),
        ("Base Feed", base_feed),
        ("High Feed", [15.0, 8.0]),
        ("Very High Feed", [20.0, 10.0]),
    ]
    scenarios = [{"Q": Q} for _, Q in feed_scenarios]

    results = parallel.run_scenarios(
        scenarios=scenarios,
        duration=duration,
        initial_state=initial_state,
    )

    print("\n   Feed-rate results (feed in t FM/d, gas flows in Nm³/d, HRT in d):")
    print("   " + "-" * 66)
    print(f"   {'Scenario':<16s} {'Feed':>6s} {'Q_gas':>9s} {'Q_ch4':>9s} {'CH4 %':>7s} {'pH':>6s} {'HRT':>7s}")
    for (label, _), result in zip(feed_scenarios, results):
        if not result.success:
            print(f"   {label:<16s} FAILED: {result.error[:60]}")
            continue
        m = result.metrics
        print(
            f"   {label:<16s} {m['Q_total']:>6.1f} {m['Q_gas']:>9.1f} {m['Q_ch4']:>9.1f} "
            f"{100 * m['CH4_content']:>7.1f} {m['pH']:>6.2f} {m['HRT']:>7.1f}"
        )

    # ------------------------------------------------------------------
    # Step 3: Parameter sweep — acetate uptake rate (k_m_ac)
    # ------------------------------------------------------------------
    print("\n3. Parameter sweep over k_m_ac...")

    # A rate passed in a scenario is used as given, at the operating
    # temperature: the simulator applies no temperature correction to it.
    # The sweep therefore starts from the rate the model itself uses there.
    k_m_ac_nominal = nominal_rate(adm1, "k_m_ac")
    print(f"   k_m_ac of the model at {adm1.T_ad - 273.15:.0f} °C: {k_m_ac_nominal:.2f} 1/d")

    sweep_config = ParameterSweepConfig(
        parameter_name="k_m_ac",
        values=[factor * k_m_ac_nominal for factor in K_M_AC_SWEEP_FACTORS],
        other_params={"Q": base_feed},
    )

    sweep_results = parallel.parameter_sweep(
        config=sweep_config,
        duration=duration,
        initial_state=initial_state,
    )

    print("\n   Sweep results (k_m_ac in 1/d, gas flows in Nm³/d):")
    print("   " + "-" * 66)
    print(f"   {'Factor':>6s} {'k_m_ac':>8s} {'Q_gas':>9s} {'Q_ch4':>9s} {'CH4 %':>7s} {'pH':>6s}")
    successful = [(factor, r) for factor, r in zip(K_M_AC_SWEEP_FACTORS, sweep_results) if r.success]
    for factor, r in successful:
        m = r.metrics
        print(
            f"   {factor:>6.2f} {r.parameters['k_m_ac']:>8.2f} {m['Q_gas']:>9.1f} {m['Q_ch4']:>9.1f} "
            f"{100 * m['CH4_content']:>7.1f} {m['pH']:>6.2f}"
        )

    if successful:
        best_factor, best = max(successful, key=lambda item: item[1].metrics["Q_ch4"])
        print(
            f"\n   Highest methane flow: {best.metrics['Q_ch4']:.1f} Nm³/d at "
            f"k_m_ac = {best.parameters['k_m_ac']:.2f} 1/d (factor {best_factor:.2f})"
        )

    # ------------------------------------------------------------------
    # Step 4: Monte Carlo analysis of an uncertain k_m_ac
    # ------------------------------------------------------------------
    print("\n4. Monte Carlo uncertainty (50 samples)...")

    mc_config = MonteCarloConfig(
        n_samples=50,
        parameter_distributions={"k_m_ac": (k_m_ac_nominal, K_M_AC_REL_STD * k_m_ac_nominal)},
        fixed_params={"Q": base_feed},
        seed=42,
    )

    mc_results = parallel.monte_carlo(
        config=mc_config,
        duration=duration,
        initial_state=initial_state,
    )
    summary = parallel.summarize_results(mc_results)

    draws = [r.parameters["k_m_ac"] for r in mc_results]
    print(f"\n   Success rate: {summary['success_rate'] * 100:.1f}%")
    print(f"   k_m_ac draws [1/d]: {min(draws):.2f} to {max(draws):.2f}")
    if "Q_ch4" in summary["metrics"]:
        s = summary["metrics"]["Q_ch4"]
        print("\n   Methane production [Nm³/d]:")
        print(f"     mean ± std : {s['mean']:.1f} ± {s['std']:.1f}")
        print(f"     min  / max : {s['min']:.1f} / {s['max']:.1f}")
        print(f"     median     : {s['median']:.1f}")

    print("\n" + "=" * 70)
    print("Parallel demonstration completed successfully!")
    print("=" * 70)

    return mc_results


if __name__ == "__main__":
    main()
