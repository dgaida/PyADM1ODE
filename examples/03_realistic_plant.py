#!/usr/bin/env python3
# examples/03_realistic_plant.py
"""
Three-stage biogas plant — 365 days of maize + cattle co-digestion.

    Primary Digester -> Secondary Digester -> Digestate Storage
                            gas -> CHP 250 kW -> heating circuits

Usage:
    python examples/03_realistic_plant.py
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import matplotlib.pyplot as plt

Q_PRIMARY = [25.0, 15.0, 0, 0, 0, 0, 0, 0, 0, 0]  # m^3/d: maize, cattle
Q_PASSTHROUGH = [0.0] * 10

STAGES = [
    ("primary", "Primary Digester", 1200.0, 216.0, 315.15),
    ("secondary", "Secondary Digester", 1200.0, 216.0, 315.15),
    ("storage", "Digestate Storage", 3500.0, 476.0, 308.15),
]

CHP_P_EL = 250.0
CHP_ETA_EL = 0.40
CHP_ETA_TH = 0.45
DURATION = 365.0


def main():
    from pyadm1 import BiogasPlant, Feedstock
    from pyadm1.components.biological.digester import Digester
    from pyadm1.configurator.plant_configurator import PlantConfigurator

    # ===================================================================
    # BUILD
    # ===================================================================
    feedstock = Feedstock(
        ["maize_silage_milk_ripeness", "cattle_manure"],
        feeding_freq=24,
        total_simtime=int(DURATION),
    )

    plant = BiogasPlant("Three-stage co-digestion plant")
    cfg = PlantConfigurator(plant, feedstock)

    # Shared inoculum so every stage starts above the X_ac washout threshold.
    _sid0, _, _Vl0, _Vg0, _T0 = STAGES[0]
    _inoc = Digester(component_id="_inoc", feedstock=feedstock, V_liq=_Vl0, V_gas=_Vg0, T_ad=_T0)
    inoculum_state = _inoc._build_pre_inoculated_state(Q_PRIMARY)

    for sid, name, V_liq, V_gas, T_ad in STAGES:
        cfg.add_digester(
            digester_id=sid,
            V_liq=V_liq,
            V_gas=V_gas,
            T_ad=T_ad,
            name=name,
            Q_substrates=Q_PRIMARY if sid == "primary" else Q_PASSTHROUGH,
            adm1_state=list(inoculum_state),
        )

    cfg.add_chp("chp", P_el_nom=CHP_P_EL, eta_el=CHP_ETA_EL, eta_th=CHP_ETA_TH, name="CHP 250 kW")
    for sid, _name, _Vl, _Vg, T_ad in STAGES:
        cfg.add_heating(f"heating_{sid}", target_temperature=T_ad, name=f"Heating {sid}")

    cfg.connect("primary", "secondary", "liquid")
    cfg.connect("secondary", "storage", "liquid")
    for sid, *_ in STAGES:
        cfg.auto_connect_digester_to_chp(sid, "chp")
        cfg.auto_connect_chp_to_heating("chp", f"heating_{sid}")

    plant.initialize()
    print(plant.get_summary())

    # ===================================================================
    # SIMULATE
    # ===================================================================
    print(f"\nSimulating {DURATION:.0f} days ...")
    results = plant.simulate(duration=DURATION, dt=1.0, save_interval=1.0)
    print(f"Done - {len(results)} samples.")

    t = np.array([r["time"] for r in results])

    def arr(cid, key, default=0.0):
        return np.array([r["components"].get(cid, {}).get(key, default) for r in results])

    stage_keys = ("Q_gas", "Q_ch4", "pH", "VFA", "TAC", "HRT", "V_liq")
    stage = {sid: {k: arr(sid, k) for k in stage_keys} for sid, *_ in STAGES}

    P_el = arr("chp", "P_el")
    P_th = arr("chp", "P_th")
    Q_gas_chp = arr("chp", "Q_gas_consumed")
    P_aux = sum(arr(f"heating_{sid}", "P_aux_heat") for sid, *_ in STAGES)
    flare_cum = arr("chp_flare", "cumulative_vented_m3")

    # ===================================================================
    # PLOTS
    # ===================================================================
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    ax_gas, ax_ch4, ax_power, ax_pH, ax_VFA, ax_heat = axes.flat

    for sid, label, *_ in STAGES:
        ax_gas.plot(t, stage[sid]["Q_gas"], label=label)
        ax_ch4.plot(t, stage[sid]["Q_ch4"], label=label)
        ax_pH.plot(t, stage[sid]["pH"], label=label)
        ax_VFA.plot(t, stage[sid]["VFA"], label=label)

    ax_gas.set_title("Biogas production")
    ax_gas.set_ylabel("Q_gas [m³/d]")
    ax_ch4.set_title("Methane production")
    ax_ch4.set_ylabel("Q_CH4 [m³/d]")
    ax_pH.set_title("pH")
    ax_pH.set_ylabel("pH [-]")
    ax_VFA.set_title("Volatile fatty acids")
    ax_VFA.set_ylabel("VFA [g HAc-eq / L]")

    ax_power.plot(t, P_el, label="P_el (CHP)")
    ax_power.plot(t, P_th, label="P_th (CHP)")
    ax_power.set_title("CHP power output")
    ax_power.set_ylabel("Power [kW]")
    ax_power.legend(loc="best", fontsize=9)

    ax_heat.plot(t, P_aux, label="Aux. boiler", color="tab:red")
    ax_heat.set_title("Auxiliary heat demand")
    ax_heat.set_ylabel("P_aux [kW]")
    ax_heat.legend(loc="best", fontsize=9)

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    for ax in (ax_pH, ax_VFA, ax_heat):
        ax.set_xlabel("Time [d]")
    ax_gas.legend(loc="best", fontsize=9)

    fig.suptitle(f"{plant.plant_name} — {DURATION:.0f}-day maize + cattle co-digestion")
    fig.tight_layout()

    output_path = REPO_ROOT / "output"
    output_path.mkdir(exist_ok=True)
    plot_file = output_path / "03_realistic_plant_states.png"
    fig.savefig(plot_file, dpi=120)
    print(f"\nSaved plot to {plot_file}")

    # ===================================================================
    # OUTPUT
    # ===================================================================
    ss = t >= (t[-1] - 90.0)

    def avg(x):
        return float(np.mean(x[ss]))

    print("\n" + "=" * 72)
    print("Substrate feed (constant)")
    print("-" * 72)
    print(f"  Maize silage       {Q_PRIMARY[0]:6.2f} m^3/d")
    print(f"  Cattle manure      {Q_PRIMARY[1]:6.2f} m^3/d")
    print(f"  Total              {sum(Q_PRIMARY):6.2f} m^3/d")

    print(f"\nFinal state at t = {t[-1]:.0f} d (per stage)")
    print("-" * 72)
    print(f"{'Stage':<22s} {'Q_gas':>8s} {'Q_CH4':>8s} {'pH':>5s} " f"{'VFA':>6s} {'TAC':>6s} {'HRT':>6s} {'V_liq':>7s}")
    print(f"{'':>22s} {'m^3/d':>8s} {'m^3/d':>8s} {'':>5s} " f"{'g/L':>6s} {'g/L':>6s} {'d':>6s} {'m^3':>7s}")
    tot_gas = tot_ch4 = 0.0
    for sid, label, *_ in STAGES:
        s = stage[sid]
        print(
            f"{label:<22s} "
            f"{s['Q_gas'][-1]:>8.1f} {s['Q_ch4'][-1]:>8.1f} "
            f"{s['pH'][-1]:>5.2f} {s['VFA'][-1]:>6.2f} {s['TAC'][-1]:>6.2f} "
            f"{s['HRT'][-1]:>6.1f} {s['V_liq'][-1]:>7.0f}"
        )
        tot_gas += s["Q_gas"][-1]
        tot_ch4 += s["Q_ch4"][-1]
    print("-" * 72)
    ch4_frac = 100.0 * tot_ch4 / tot_gas if tot_gas > 0.0 else 0.0
    print(f"{'TOTAL':<22s} {tot_gas:>8.1f} {tot_ch4:>8.1f}    " f"(CH4 fraction: {ch4_frac:.1f} %)")

    avg_P_el = avg(P_el)
    avg_P_th = avg(P_th)
    avg_P_aux = avg(P_aux)
    avg_Q_gas_chp = avg(Q_gas_chp)
    load_factor = 100.0 * avg_P_el / CHP_P_EL if CHP_P_EL > 0 else 0.0

    yr_el = avg_P_el * 24.0 * 365.0 / 1000.0
    yr_th = avg_P_th * 24.0 * 365.0 / 1000.0
    yr_aux = avg_P_aux * 24.0 * 365.0 / 1000.0

    biogas_total_t = sum(stage[sid]["Q_gas"] for sid, *_ in STAGES)
    methane_total_t = sum(stage[sid]["Q_ch4"] for sid, *_ in STAGES)
    spec_ch4_fm = avg(methane_total_t) / sum(Q_PRIMARY) if sum(Q_PRIMARY) else 0.0

    print("\nEnergy production (averaged over last 90 d of the run)")
    print("-" * 72)
    print(f"  Biogas total           {avg(biogas_total_t):8.1f} m^3/d")
    print(f"  Methane total          {avg(methane_total_t):8.1f} m^3/d")
    print(f"  CHP biogas consumption {avg_Q_gas_chp:8.1f} m^3/d")
    print(f"  CHP P_el (avg)         {avg_P_el:8.1f} kW    " f"({load_factor:.0f} % of rated {CHP_P_EL:.0f} kW)")
    print(f"  CHP P_th (avg)         {avg_P_th:8.1f} kW")
    print(
        f"  Aux. heat (avg)        {avg_P_aux:8.1f} kW   "
        f"({100*avg_P_aux/max(avg_P_th+avg_P_aux, 1e-6):.0f} % of heat demand)"
    )
    print(f"  Flare losses (total)   {flare_cum[-1]:8.1f} m^3 over {DURATION:.0f} d")

    print("\nAnnualised yields (constant load)")
    print("-" * 72)
    print(f"  Electricity            {yr_el:8.0f} MWh_el / yr")
    print(f"  Useful heat (CHP)      {yr_th:8.0f} MWh_th / yr")
    print(f"  Aux. heat (boiler)     {yr_aux:8.0f} MWh_th / yr")
    print(f"  Spec. methane yield    {spec_ch4_fm:8.1f} m^3 CH4 / m^3 fresh feed")
    print("=" * 72)

    plt.show()
    return plant, results


if __name__ == "__main__":
    main()
