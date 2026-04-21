#!/usr/bin/env python3
# examples/10_adm1da_basic_digester.py
"""
ADM1da basic single-digester simulation.

Digester configuration
----------------------
  Temperature            42 °C   (315.15 K)
  Max. tank volume       1200 m³
  Liquid volume          1050 m³
  Gas headspace          150 m³   (1200 - 1050)
  Gas transfer coeff.    200 1/d  (k_L_a, ADM1da default)

Substrate feed
--------------
  Maize silage (milk ripeness)   11.4 m³/d
  Swine manure                    6.1 m³/d
  Total feed                     17.5 m³/d
  HRT                            60 d   (1050 / 17.5)

Substrates are loaded from XML files in data/substrates/adm1da/.

Usage
-----
    python examples/10_adm1da_basic_digester.py

Output
------
  - Console: daily biogas/methane production, pH
  - output/adm1da_basic_state.csv: full state history
  - output/adm1da_basic_summary.csv: daily summary
"""

from pathlib import Path
import sys

# Ensure repo root is on the Python path when run directly
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Helper: build a reasonable steady-state initial condition for ADM1da
# ---------------------------------------------------------------------------


def _build_initial_state(
    conc_maize: dict, conc_swine: dict, Q_maize: float, Q_swine: float, V_liq: float, T_ad: float
) -> list:
    """
    Build a pre-inoculated initial condition for ADM1da.

    Particulate organics start from the blended influent (retention factor).
    Dissolved components represent a healthy running digester at pH 7.
    S_cation is computed from the charge balance to enforce pH = 7.0 exactly,
    so that methanogenesis is active from the first integration step.
    """
    q_tot = Q_maize + Q_swine
    f_m = Q_maize / q_tot
    f_s = Q_swine / q_tot

    def blend(key):
        return f_m * conc_maize.get(key, 0.0) + f_s * conc_swine.get(key, 0.0)

    # --- Acid-base constants (temperature-corrected to T_ad) ---
    R_gas = 0.08314  # bar·m³·kmol⁻¹·K⁻¹
    T_base = 298.15  # 25 °C reference temperature [K]
    K_a_va = 10.0**-4.86
    K_a_bu = 10.0**-4.82
    K_a_pro = 10.0**-4.88
    K_a_ac = 10.0**-4.76
    K_a_co2 = 10.0**-6.35
    K_a_IN = 10.0**-9.25
    K_w = 1.0e-14 * np.exp((55900.0 / (100.0 * R_gas)) * (1.0 / T_base - 1.0 / T_ad))

    # --- Target pH 7 for initial state ---
    S_H_0 = 10.0**-7.0

    # --- Typical healthy-digester dissolved concentrations ---
    S_ac_0 = 0.10  # [kg COD/m³] — low acetate
    S_pro_0 = 0.02
    S_bu_0 = 0.01
    S_va_0 = 0.01
    S_co2_0 = 0.18  # total inorganic carbon (S_IC) [kmol C/m³]

    # S_nh4 elevated above influent (protein degradation already releases NH4)
    S_nh4_0 = blend("S_nh4") * 1.5

    # --- Ion concentrations at pH 7 ---
    S_ac_ion_0 = K_a_ac / (K_a_ac + S_H_0) * S_ac_0
    S_pro_ion_0 = K_a_pro / (K_a_pro + S_H_0) * S_pro_0
    S_bu_ion_0 = K_a_bu / (K_a_bu + S_H_0) * S_bu_0
    S_va_ion_0 = K_a_va / (K_a_va + S_H_0) * S_va_0
    S_nh3_0 = K_a_IN / (K_a_IN + S_H_0) * S_nh4_0

    # Bicarbonate consistent with S_IC at pH 7
    S_hco3_0 = K_a_co2 * S_co2_0 / (S_H_0 + K_a_co2)

    # VFA kmol for charge balance
    vfa_kmol_0 = S_ac_ion_0 / 64.0 + S_pro_ion_0 / 112.0 + S_bu_ion_0 / 160.0 + S_va_ion_0 / 208.0

    # S_anion from influent (conservative tracer — no internal sources)
    S_anion_0 = blend("S_anion")

    # S_cation computed from charge balance to enforce pH = 7 at t = 0
    # Balance: S_cat + S_nh4 - S_nh3 + S_H = S_an + S_hco3 + vfa_kmol + Kw/S_H
    S_cation_0 = S_anion_0 + S_hco3_0 + vfa_kmol_0 + K_w / S_H_0 - S_nh4_0 + S_nh3_0 - S_H_0

    # --- Particulate pools at true steady-state concentrations ---
    # Retention factor: X_ss = D / (D + k_dis) * X_in
    # SIMBA# kinetics at 35 °C: k_dis_PF=0.4, k_dis_PS=0.04, k_hyd=4.0 d⁻¹
    # theta_dis=1.035, theta_hyd=1.07
    D = (Q_maize + Q_swine) / V_liq  # dilution rate [d⁻¹]
    dT = T_ad - 308.15  # temperature offset from 35 °C [K]
    k_dis_PF = 0.4 * (1.035**dT)
    k_dis_PS = 0.04 * (1.035**dT)
    k_hyd = 4.0 * (1.07**dT)  # same for ch/pr/li

    ret_PF = D / (D + k_dis_PF)
    ret_PS = D / (D + k_dis_PS)

    # X_S pools are not in the influent; initialise at their SS production value
    # SIMBA#: fXI_PS = fXI_PF = 0.0 (no inert fraction from disintegration)
    fXI_PF = 0.0
    fXI_PS = 0.0
    X_PF_ch_ss = blend("X_PF_ch") * ret_PF
    X_PF_pr_ss = blend("X_PF_pr") * ret_PF
    X_PF_li_ss = blend("X_PF_li") * ret_PF
    X_PS_ch_ss = blend("X_PS_ch") * ret_PS
    X_PS_pr_ss = blend("X_PS_pr") * ret_PS
    X_PS_li_ss = blend("X_PS_li") * ret_PS

    # X_S_ch at SS: produced by disintegration, consumed by hydrolysis
    X_S_ch_0 = ((1.0 - fXI_PF) * k_dis_PF * X_PF_ch_ss + (1.0 - fXI_PS) * k_dis_PS * X_PS_ch_ss) / (D + k_hyd)
    X_S_pr_0 = ((1.0 - fXI_PF) * k_dis_PF * X_PF_pr_ss + (1.0 - fXI_PS) * k_dis_PS * X_PS_pr_ss) / (D + k_hyd)
    X_S_li_0 = ((1.0 - fXI_PF) * k_dis_PF * X_PF_li_ss + (1.0 - fXI_PS) * k_dis_PS * X_PS_li_ss) / (D + k_hyd)

    # X_I accumulates from influent and disintegration inerts
    X_I_in = blend("X_I")
    X_I_0 = X_I_in  # conservative tracer — SS ≈ influent value (plus small inert term)

    # Gas phase (realistic partial pressures for a running digester)
    p_h2_0 = 1.02e-5
    p_ch4_0 = 0.65
    p_co2_0 = 0.33
    p_tot_0 = p_h2_0 + p_ch4_0 + p_co2_0

    return [
        0.01,  #  0  S_su
        0.001,  #  1  S_aa
        0.05,  #  2  S_fa
        S_va_0,  #  3  S_va
        S_bu_0,  #  4  S_bu
        S_pro_0,  #  5  S_pro
        S_ac_0,  #  6  S_ac
        1.0e-7,  #  7  S_h2
        1.0e-4,  #  8  S_ch4
        S_co2_0,  #  9  S_co2 (S_IC)  [kmol C/m³]
        S_nh4_0,  # 10  S_nh4  [kmol N/m³]
        0.0,  # 11  S_I
        X_PS_ch_ss,  # 12  X_PS_ch
        X_PS_pr_ss,  # 13  X_PS_pr
        X_PS_li_ss,  # 14  X_PS_li
        X_PF_ch_ss,  # 15  X_PF_ch
        X_PF_pr_ss,  # 16  X_PF_pr
        X_PF_li_ss,  # 17  X_PF_li
        X_S_ch_0,  # 18  X_S_ch
        X_S_pr_0,  # 19  X_S_pr
        X_S_li_0,  # 20  X_S_li
        X_I_0,  # 21  X_I
        0.50,  # 22  X_su
        0.50,  # 23  X_aa
        0.30,  # 24  X_fa
        0.40,  # 25  X_c4
        0.30,  # 26  X_pro
        1.20,  # 27  X_ac
        0.30,  # 28  X_h2
        S_cation_0,  # 29  S_cation  [kmol/m³]
        S_anion_0,  # 30  S_anion   [kmol/m³]
        S_va_ion_0,  # 31  S_va_ion
        S_bu_ion_0,  # 32  S_bu_ion
        S_pro_ion_0,  # 33  S_pro_ion
        S_ac_ion_0,  # 34  S_ac_ion
        S_hco3_0,  # 35  S_hco3    [kmol C/m³]
        S_nh3_0,  # 36  S_nh3     [kmol N/m³]
        p_h2_0,  # 37  p_gas_h2  [bar]
        p_ch4_0,  # 38  p_gas_ch4 [bar]
        p_co2_0,  # 39  p_gas_co2 [bar]
        p_tot_0,  # 40  pTOTAL = sum of partial pressures [bar]
    ]


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------


def main():
    from pyadm1.configurator.plant_builder import BiogasPlant
    from pyadm1.configurator.plant_configurator import PlantConfigurator
    from pyadm1.components.biological import ADM1daDigester
    from pyadm1.components.energy.gas_storage import GasStorage
    from pyadm1.substrates.feedstock import Feedstock
    from pyadm1.substrates.adm1da_feedstock import ADM1daFeedstock, SubstrateRegistry
    from pyadm1.core.adm1da import STATE_SIZE, INFLUENT_COLUMNS

    print("=" * 70)
    print("ADM1da Basic Digester Simulation (component-based)")
    print("=" * 70)

    # --- Digester configuration ---
    T_ad = 315.15  # 42 °C in Kelvin
    V_liq = 1050.0  # m³
    V_gas = 150.0  # m³  (1200 - 1050)

    # --- Substrate feed rates ---
    Q_maize = 11.4  # m³/d
    Q_swine = 6.1  # m³/d
    Q_total = Q_maize + Q_swine

    # --- Simulation parameters ---
    sim_duration = 150.0  # days — long enough to approach steady state at HRT=60 d
    dt_save = 1.0  # save every day

    print("\nDigester setup:")
    print(f"  Temperature       : {T_ad - 273.15:.0f} °C  ({T_ad} K)")
    print(f"  Liquid volume     : {V_liq} m³")
    print(f"  Gas headspace     : {V_gas} m³")
    print(f"  Total tank volume : {V_liq + V_gas} m³")
    print(f"  HRT               : {V_liq / Q_total:.1f} d")

    print("\nFeed:")
    print(f"  Maize silage (milk ripeness) : {Q_maize} m³/d")
    print(f"  Swine manure                 : {Q_swine} m³/d")
    print(f"  Total feed                   : {Q_total} m³/d")

    # --- Load substrates from XML ---
    print("\nLoading substrates from XML...")
    registry = SubstrateRegistry()
    available = registry.available()
    print(f"  Available substrates: {available}")

    total_simtime = int(sim_duration) + 10
    fs_maize = ADM1daFeedstock(
        registry.get("maize_silage_milk_ripeness"),
        feeding_freq=24,
        total_simtime=total_simtime,
    )
    fs_swine = ADM1daFeedstock(
        registry.get("swine_manure"),
        feeding_freq=24,
        total_simtime=total_simtime,
    )

    print(f"\n  Maize silage density  : {fs_maize.density:.1f} kg/m³")
    print(f"  Swine manure density  : {fs_swine.density:.1f} kg/m³")

    # --- Blend the two substrates into one ADM1da influent DataFrame ---
    c_m = fs_maize.concentrations
    c_s = fs_swine.concentrations
    mixed_row = {col: (c_m.get(col, 0.0) * Q_maize + c_s.get(col, 0.0) * Q_swine) / Q_total for col in INFLUENT_COLUMNS[:-1]}
    mixed_row["Q"] = Q_total
    influent_df = pd.DataFrame([mixed_row] * total_simtime, columns=INFLUENT_COLUMNS)
    rho_mix = (Q_maize * fs_maize.density + Q_swine * fs_swine.density) / Q_total

    # OLR information
    VS_maize = fs_maize.vs_content()
    VS_swine = fs_swine.vs_content()
    OLR = (VS_maize * Q_maize + VS_swine * Q_swine) / V_liq
    print(f"\n  Organic loading rate (VS-based) : {OLR:.2f} kg VS/(m³·d)")
    print(f"  Maize silage BMP (reference)    : {fs_maize.substrate.BMP:.1f} Nm³ CH4/t VS")
    print(f"  Swine manure  BMP (reference)   : {fs_swine.substrate.BMP:.1f} Nm³ CH4/t VS")

    # --- Plant & component assembly (mirrors example 01 / example 13 pattern) ---
    print("\nBuilding plant...")
    feedstock = Feedstock(feeding_freq=24)
    plant = BiogasPlant("ADM1da Basic Plant")
    cfg = PlantConfigurator(plant, feedstock)

    digester = ADM1daDigester(
        component_id="main_digester",
        feedstock=feedstock,
        V_liq=V_liq,
        V_gas=V_gas,
        T_ad=T_ad,
        name="ADM1da Main Digester",
    )

    # Supply the pre-blended ADM1da influent DataFrame and calibration overrides
    # directly to the underlying ADM1da solver (bypasses the ADM1→ADM1da remap).
    digester.adm1._calibration_params["k_L_a"] = 200.0
    digester.adm1.set_influent_dataframe(influent_df)
    digester.adm1.set_influent_density(rho_mix)

    # Build realistic initial state and initialise the digester
    state0 = _build_initial_state(c_m, c_s, Q_maize, Q_swine, V_liq, T_ad)
    assert len(state0) == STATE_SIZE, f"Expected {STATE_SIZE}, got {len(state0)}"
    digester.initialize({"adm1_state": state0, "Q_substrates": [Q_total]})
    plant.add_component(digester)

    # Gas storage attached to the digester (same topology as example 13)
    storage = GasStorage(
        component_id="main_digester_storage",
        storage_type="membrane",
        capacity_m3=max(50.0, V_gas),
        name="Gas Storage",
    )
    plant.add_component(storage)
    cfg.connect("main_digester", "main_digester_storage", "gas")

    plant.initialize()

    print(f"  Model             : {digester.adm1.model_name}")
    print(f"  State vector size : {digester.adm1.get_state_size()}")

    print(f"\nRunning simulation for {sim_duration:.0f} days...")
    print("  (This may take a few seconds for a stiff 41-state ODE system.)\n")

    results = plant.simulate(
        duration=sim_duration,
        dt=dt_save,
        save_interval=dt_save,
    )
    print(f"Integration complete: {len(results)} saved time points.")

    # --- Post-processing: collect time series from plant results ---
    times = np.array([r["time"] for r in results])
    comp_series = [r["components"]["main_digester"] for r in results]
    states = np.array([c["state_out"] for c in comp_series]).T  # shape: (41, n_t)
    q_gas_arr = np.array([c["Q_gas"] for c in comp_series])
    q_ch4_arr = np.array([c["Q_ch4"] for c in comp_series])
    q_co2_arr = np.array([c["Q_co2"] for c in comp_series])
    pH_arr = np.array([c["pH"] for c in comp_series])

    # --- Console output: daily summary ---
    print()
    print(f"{'Day':>5s}  {'Biogas':>10s}  {'Methane':>10s}  {'CH4%':>6s}  {'pH':>6s}")
    print("-" * 50)

    # Print every 10 days + final day
    report_days = list(range(0, int(sim_duration) + 1, 10))
    if int(sim_duration) not in report_days:
        report_days.append(int(sim_duration))

    for day in report_days:
        idx = np.argmin(np.abs(times - float(day)))
        q_g = q_gas_arr[idx]
        q_ch = q_ch4_arr[idx]
        ch4p = (q_ch / q_g * 100.0) if q_g > 0.0 else 0.0
        pH = pH_arr[idx]
        print(f"{day:>5d}  {q_g:>10.1f}  {q_ch:>10.1f}  {ch4p:>6.1f}  {pH:>6.2f}")

    # --- Final summary ---
    print()
    print("=" * 70)
    print("FINAL STATE  (day {:.0f})".format(times[-1]))
    print("=" * 70)
    final_gas = q_gas_arr[-1]
    final_ch4 = q_ch4_arr[-1]
    final_ch4p = (final_ch4 / final_gas * 100.0) if final_gas > 0.0 else 0.0
    final_pH = pH_arr[-1]

    # Expected methane yield from BMP reference
    bmp_expected = (  # Nm³ CH4/d
        fs_maize.substrate.BMP / 1000.0 * VS_maize * Q_maize + fs_swine.substrate.BMP / 1000.0 * VS_swine * Q_swine
    )
    print(f"  Biogas production  : {final_gas:.1f}  m³/d")
    print(f"  Methane production : {final_ch4:.1f}  m³/d")
    print(f"  CH4 content        : {final_ch4p:.1f}  %")
    print(f"  pH                 : {final_pH:.2f}")
    print(f"  Expected CH4 (BMP) : {bmp_expected:.1f}  m³/d  (reference, 100% biodegradability)")
    print("=" * 70)

    # --- Save results to CSV ---
    output_dir = REPO_ROOT / "output"
    output_dir.mkdir(exist_ok=True)

    # Full state history
    state_labels = [
        "S_su",
        "S_aa",
        "S_fa",
        "S_va",
        "S_bu",
        "S_pro",
        "S_ac",
        "S_h2",
        "S_ch4",
        "S_co2",
        "S_nh4",
        "S_I",
        "X_PS_ch",
        "X_PS_pr",
        "X_PS_li",
        "X_PF_ch",
        "X_PF_pr",
        "X_PF_li",
        "X_S_ch",
        "X_S_pr",
        "X_S_li",
        "X_I",
        "X_su",
        "X_aa",
        "X_fa",
        "X_c4",
        "X_pro",
        "X_ac",
        "X_h2",
        "S_cation",
        "S_anion",
        "S_va_ion",
        "S_bu_ion",
        "S_pro_ion",
        "S_ac_ion",
        "S_hco3_ion",
        "S_nh3",
        "p_gas_h2",
        "p_gas_ch4",
        "p_gas_co2",
        "pTOTAL",
    ]
    state_df = pd.DataFrame(states.T, columns=state_labels)
    state_df.insert(0, "time_d", times)
    state_df["Q_gas_m3d"] = q_gas_arr
    state_df["Q_ch4_m3d"] = q_ch4_arr
    state_df["Q_co2_m3d"] = q_co2_arr
    state_df["pH"] = pH_arr

    state_file = output_dir / "adm1da_basic_state.csv"
    state_df.to_csv(state_file, index=False, float_format="%.6g")
    print(f"\nFull state history saved to: {state_file}")

    # Daily summary
    summary_df = state_df[["time_d", "Q_gas_m3d", "Q_ch4_m3d", "Q_co2_m3d", "pH"]]
    summary_file = output_dir / "adm1da_basic_summary.csv"
    summary_df.to_csv(summary_file, index=False, float_format="%.4f")
    print(f"Daily summary saved to     : {summary_file}")

    return plant, results, state_df


if __name__ == "__main__":
    plant, results, df = main()
