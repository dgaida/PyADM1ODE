#!/usr/bin/env python3
# examples/06_parallel_operation_studies.py
"""
Parallel design studies for a 3-substrate ADM1 biogas plant.

Substrates: maize silage (energy crop), cattle manure (buffering /
alkalinity), and grass silage (fibre diversity).

Three plant-operator questions, each examined with a parallel sweep
on top of :class:`pyadm1.simulation.parallel.ParallelSimulator`:

  1. HRT × OLR envelope   — "Which (HRT, OLR) combinations stay stable?"
  2. Substrate mix        — "Best maize / cattle / grass mix at constant feed?"
  3. Temperature regime   — "How does the methane flow change from 35 to 58 °C?"

Each section runs an ensemble of simulations, prints its result and saves
one summary figure to ``output/``. "Stable" combines the pH of the model
with an assumed ceiling on the organic loading rate (``OLR_MAX``); see the
constants below. Section 2 runs at a high total feed (40 t FM/d), and
Section 3 uses the best mix of Section 2. The results depend on the
substrates: swap the substrate list at the top of the file to see how
they shift.

Units:
    - Feed rates are tonnes of fresh matter per day [t FM/d] under the default
      Feedstock convention (``simba_q_convention=True``).
    - Gas flows are dry gas at standard conditions (0 °C, 1.01325 bar) [Nm³/d].

Usage:
    python examples/06_parallel_operation_studies.py
"""

import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Shared configuration
# ---------------------------------------------------------------------------
V_LIQ = 1200.0  # m³  fermenter liquid volume
V_GAS = 216.0  # m³  gas headspace
T_AD_MESO = 315.15  # K   42 °C — reference temperature

# Three-substrate co-digestion: a high-yield maize silage, a buffering
# cattle slurry, and a fibre-rich grass silage. The Q[i] axis is fixed
# by this order across all sections.
SUBSTRATES = ["maize_silage_milk_ripeness", "cattle_manure", "grass_silage"]
I_MAIZE, I_CATTLE, I_GRASS = 0, 1, 2

# Stability envelope used by the studies. The model alone is forgiving: in
# Section 1 it still reports a pH above 7 at the highest loading rates. The
# studies therefore add a ceiling on the organic loading rate. Both limits
# are assumptions of this example, not model results; adjust them to the
# plant at hand.
OLR_MAX = 6.0  # kg VS / m³ d — assumed upper limit for wet digestion
PH_STABLE_MIN = 7.0  # lowest pH counted as stable


def _make_base_adm1(T_ad=T_AD_MESO, duration_days=30):
    """Build a fresh ADM1 base model for a ParallelSimulator."""
    from pyadm1 import Feedstock
    from pyadm1.core.adm1 import ADM1

    fs = Feedstock(SUBSTRATES, feeding_freq=24, total_simtime=int(duration_days + 5))
    adm1 = ADM1(fs, V_liq=V_LIQ, V_gas=V_GAS, T_ad=T_ad)
    return adm1, fs


def _olr(fs, Q):
    """Return the organic loading rate of a feed.

    Args:
        fs (Feedstock): Feedstock that defines the substrates.
        Q (Sequence[float]): Feed rates per substrate [t FM/d].

    Returns:
        float: Volatile solids fed per day and m³ of liquid volume
            [kg VS / m³ d].
    """
    # blended_vs_content is given per m³ of feed, so it is multiplied by the
    # volume flow the model uses, not by the feed in t FM/d.
    return float(np.sum(fs.actual_Q(Q))) * fs.blended_vs_content(Q) / V_LIQ


# Reference feed [t FM/d] the start state of Sections 1 and 2 is built for:
# a roughly even three-way split at 20 t FM/d. Every scenario then moves
# from this state towards its own feed and mix.
_Q_INOCULUM = [7.0, 7.0, 6.0, 0, 0, 0, 0, 0, 0, 0]


def _inoculum_state(fs, T_ad=T_AD_MESO, Q_ref=None):
    """Build a start state with active biomass from the feedstock blend.

    The state comes from the pre-inoculation of the ``Digester`` component:
    particulate pools sized for the reference feed, dissolved species of a
    digester at pH 7 and active biomass, so that methane forms from the
    first day.

    Args:
        fs (Feedstock): Feedstock that defines the substrates.
        T_ad (float): Temperature the state is built for [K].
        Q_ref (Sequence[float] | None): Reference feed [t FM/d]; defaults to
            ``_Q_INOCULUM``.

    Returns:
        list[float]: ADM1 state vector (41 values).
    """
    from pyadm1.components.biological.digester import Digester

    proxy = Digester(component_id="_inoc_proxy", feedstock=fs, V_liq=V_LIQ, V_gas=V_GAS, T_ad=T_ad)
    return proxy._build_pre_inoculated_state(Q_ref or _Q_INOCULUM)


# ===========================================================================
# Section 1 — HRT × OLR envelope
# ===========================================================================


def study_hrt_olr_envelope(n_workers, duration=60.0):
    """Run a 2-D sweep over the total feed and the energy-crop fraction.

    The retention time follows mainly from the total feed, because the
    liquid volume is fixed. The loading rate also follows from the mix: per
    tonne, the energy crops (maize + grass) carry several times the volatile
    solids of cattle manure, so a high ``ec_frac`` raises the OLR at the
    same total feed. Within the energy-crop share the maize : grass split is
    held 50:50; Section 2 explores that axis.

    A scenario counts as stable when its pH is at least ``PH_STABLE_MIN``,
    its gas holds more than 30 % methane and its OLR is at most ``OLR_MAX``.

    Args:
        n_workers (int): Number of worker processes.
        duration (float): Simulated time per scenario [d].
    """
    from pyadm1.simulation.parallel import ParallelSimulator

    print("\n" + "=" * 72)
    print("Section 1 — HRT × OLR operating envelope")
    print("=" * 72)

    Q_total_list = [10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 80.0]
    ec_frac_list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  # energy-crop fraction

    def _mix_Q(Q_tot, ec_frac):
        # Within the energy-crop slice: maize and grass split 50:50.
        Q = [0.0] * 10
        Q[I_MAIZE] = Q_tot * ec_frac * 0.5
        Q[I_GRASS] = Q_tot * ec_frac * 0.5
        Q[I_CATTLE] = Q_tot * (1.0 - ec_frac)
        return Q

    scenarios = [{"Q": _mix_Q(Q_tot, ec)} for Q_tot in Q_total_list for ec in ec_frac_list]

    adm1, fs = _make_base_adm1(T_ad=T_AD_MESO, duration_days=duration)
    parallel = ParallelSimulator(adm1, n_workers=n_workers, verbose=False)

    t0 = time.time()
    print(f"Running {len(scenarios)} scenarios on {parallel.n_workers} workers ...")
    results = parallel.run_scenarios(
        scenarios,
        duration=duration,
        initial_state=_inoculum_state(fs),
        dt=1.0,
    )
    print(f"  done in {time.time() - t0:.1f} s")

    n_Q, n_ec = len(Q_total_list), len(ec_frac_list)
    hrt = np.full((n_Q, n_ec), np.nan)
    olr = np.full((n_Q, n_ec), np.nan)
    q_ch4 = np.full((n_Q, n_ec), np.nan)
    spec_ch4 = np.full((n_Q, n_ec), np.nan)
    ph = np.full((n_Q, n_ec), np.nan)
    stable = np.zeros((n_Q, n_ec), dtype=bool)

    for k, r in enumerate(results):
        i, j = divmod(k, n_ec)
        if not r.success:
            continue
        m = r.metrics
        hrt[i, j] = m.get("HRT", np.nan)
        q_ch4[i, j] = m.get("Q_ch4", np.nan)
        spec_ch4[i, j] = m.get("specific_ch4_production", np.nan)
        ph[i, j] = m.get("pH", np.nan)
        olr[i, j] = _olr(fs, _mix_Q(Q_total_list[i], ec_frac_list[j]))
        # Stability = (model check) AND (assumed OLR ceiling).
        stable[i, j] = (
            ph[i, j] >= PH_STABLE_MIN
            and m.get("CH4_content", 0.0) > 0.30
            and m.get("Q_gas", 0.0) > 1.0
            and olr[i, j] <= OLR_MAX
        )

    print(f"  stable scenarios: {int(stable.sum())} / {stable.size}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    sc = ax.scatter(
        hrt.flatten(), olr.flatten(), c=spec_ch4.flatten(), s=140, cmap="viridis", edgecolors="black", linewidths=0.4
    )
    plt.colorbar(sc, ax=ax, label="Spec. CH₄ yield [Nm³ CH₄ / t FM]")
    ax.set_xlabel("HRT [d]")
    ax.set_ylabel("OLR [kg VS / m³ d]")
    ax.set_title("Methane yield over the HRT × OLR plane")
    ax.set_xscale("log")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    colors = np.where(stable.flatten(), "tab:green", "tab:red")
    ax.scatter(hrt.flatten(), olr.flatten(), c=colors, s=140, edgecolors="black", linewidths=0.4)
    ax.axhline(OLR_MAX, color="black", linestyle="--", alpha=0.6, label=f"engineering OLR ceiling ({OLR_MAX:.1f})")
    ax.set_xlabel("HRT [d]")
    ax.set_ylabel("OLR [kg VS / m³ d]")
    ax.set_title(f"Stability map (green = pH ≥ {PH_STABLE_MIN:.1f} " f"∧ OLR ≤ {OLR_MAX:.0f})")
    ax.set_xscale("log")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Operating envelope — V_liq = 1200 m³ @ 42 °C " "(maize : grass : cattle, energy-crop slice 50:50)")
    fig.tight_layout()
    _savefig(fig, "06_section1_hrt_olr.png")

    print("\n  Q_tot in t FM/d, HRT in d, OLR in kg VS / m³ d, Q_CH4 in Nm³/d, spec in Nm³ CH4 / t FM")
    print(f"  {'Q_tot':>6s} {'ec_frac':>7s} {'HRT':>6s} {'OLR':>6s} " f"{'Q_CH4':>8s} {'spec':>6s} {'pH':>5s} {'stable':>7s}")
    for i, Q_tot in enumerate(Q_total_list):
        for j, ec in enumerate(ec_frac_list):
            print(
                f"  {Q_tot:>6.1f} {ec:>7.1f} {hrt[i,j]:>6.1f} "
                f"{olr[i,j]:>6.2f} {q_ch4[i,j]:>8.1f} "
                f"{spec_ch4[i,j]:>6.2f} {ph[i,j]:>5.2f} "
                f"{'yes' if stable[i,j] else 'no':>7s}"
            )


# ===========================================================================
# Section 2 — Substrate-mix optimisation at constant Q_in
# ===========================================================================


def study_substrate_mix(n_workers, duration=90.0):
    """Sweep the substrate mix at a constant total feed.

    The maize and grass fractions vary in steps of 0.1; cattle manure takes
    the remainder. The total feed is held at a high 40 t FM/d. The best mix
    is the one with the highest methane flow among the mixes that meet both
    limits: pH of the model at least ``PH_STABLE_MIN`` and OLR at most
    ``OLR_MAX``. Which limit decides and which mix wins depends on the
    substrates; the function prints both.

    Args:
        n_workers (int): Number of worker processes.
        duration (float): Simulated time per mix [d].

    Returns:
        dict | None: Fractions, total feed, methane flow, methane content,
            pH and OLR of the best mix, or ``None`` if no mix meets both
            limits.
    """
    from pyadm1.simulation.parallel import ParallelSimulator

    print("\n" + "=" * 72)
    print("Section 2 — Optimal mix (maize × grass × cattle) at high load")
    print("=" * 72)

    Q_total = 40.0  # t FM/d, held constant (high load)
    fracs = np.linspace(0.0, 1.0, 11)  # 11 steps per axis

    grid_combos = []  # (i, j, mf, gf, cf)
    scenarios = []
    for i, mf in enumerate(fracs):
        for j, gf in enumerate(fracs):
            cf = 1.0 - mf - gf
            if cf < -1e-9:
                continue  # infeasible
            cf = max(cf, 0.0)
            Q = [0.0] * 10
            Q[I_MAIZE] = Q_total * mf
            Q[I_CATTLE] = Q_total * cf
            Q[I_GRASS] = Q_total * gf
            scenarios.append({"Q": Q})
            grid_combos.append((i, j, mf, gf, cf))

    adm1, fs = _make_base_adm1(T_ad=T_AD_MESO, duration_days=duration)
    parallel = ParallelSimulator(adm1, n_workers=n_workers, verbose=False)

    t0 = time.time()
    print(f"Running {len(scenarios)} feasible mixes " f"on {parallel.n_workers} workers ...")
    results = parallel.run_scenarios(
        scenarios,
        duration=duration,
        initial_state=_inoculum_state(fs),
        dt=1.0,
    )
    print(f"  done in {time.time() - t0:.1f} s")

    n = len(fracs)
    q_ch4 = np.full((n, n), np.nan)
    ch4_pct = np.full((n, n), np.nan)
    ph = np.full((n, n), np.nan)
    spec_ch4 = np.full((n, n), np.nan)
    olr = np.full((n, n), np.nan)

    for combo, r in zip(grid_combos, results):
        i, j, mf, gf, cf = combo
        # OLR of this mix; the order is maize, cattle manure, grass.
        olr[i, j] = _olr(fs, [Q_total * mf, Q_total * cf, Q_total * gf])
        if not r.success:
            continue
        m = r.metrics
        q_ch4[i, j] = m.get("Q_ch4", np.nan)
        spec_ch4[i, j] = m.get("specific_ch4_production", np.nan)
        ch4_pct[i, j] = 100.0 * m.get("CH4_content", np.nan)
        ph[i, j] = m.get("pH", np.nan)

    stable = (ph >= PH_STABLE_MIN) & (olr <= OLR_MAX)
    print(
        f"\n  Range over all mixes: pH {np.nanmin(ph):.2f} to {np.nanmax(ph):.2f}, "
        f"OLR {np.nanmin(olr):.2f} to {np.nanmax(olr):.2f} kg VS / m³ d"
    )
    best_mix = None
    if stable.any():
        q_stable = np.where(stable, q_ch4, -np.inf)
        bi, bj = np.unravel_index(np.argmax(q_stable), q_stable.shape)
        best_mix = {
            "maize_frac": float(fracs[bi]),
            "grass_frac": float(fracs[bj]),
            "cattle_frac": float(1.0 - fracs[bi] - fracs[bj]),
            "Q_total": float(Q_total),
            "Q_ch4": float(q_ch4[bi, bj]),
            "spec_ch4": float(spec_ch4[bi, bj]),
            "ch4_pct": float(ch4_pct[bi, bj]),
            "pH": float(ph[bi, bj]),
            "OLR": float(olr[bi, bj]),
        }
        print(f"\n  Best mix @ Q_in = {Q_total:.0f} t FM/d " f"(within pH ≥ {PH_STABLE_MIN:.1f} and OLR ≤ {OLR_MAX:.0f}):")
        print(f"    maize  = {best_mix['maize_frac']:.2f}")
        print(f"    cattle = {best_mix['cattle_frac']:.2f}")
        print(f"    grass  = {best_mix['grass_frac']:.2f}")
        print(
            f"    -> Q_CH4 = {best_mix['Q_ch4']:.1f} Nm³/d, "
            f"CH4 = {best_mix['ch4_pct']:.1f} %, "
            f"pH = {best_mix['pH']:.2f}, "
            f"OLR = {best_mix['OLR']:.2f}"
        )
        print(
            f"  {int(stable.sum())} of {int(np.isfinite(q_ch4).sum())} "
            f"simulated mixes stayed inside the safe operating envelope "
            f"(pH ≥ {PH_STABLE_MIN:.1f} and OLR ≤ {OLR_MAX:.0f} kg VS / m³ d)"
        )
    else:
        print(
            "\n  WARNING: no mix sits inside the engineering envelope "
            f"(pH ≥ {PH_STABLE_MIN:.1f} ∧ OLR ≤ {OLR_MAX:.0f}). "
            "Try a lower Q_in or include more buffering substrate."
        )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    ax = axes[0]
    im = ax.imshow(q_ch4.T, origin="lower", extent=(fracs[0], fracs[-1], fracs[0], fracs[-1]), cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax, label="Q_CH₄ [Nm³/d]")
    ax.plot([0, 1], [1, 0], color="white", linestyle="--", alpha=0.6, label="feasibility edge (cattle = 0)")
    if best_mix is not None:
        ax.scatter(
            [best_mix["maize_frac"]],
            [best_mix["grass_frac"]],
            s=200,
            marker="*",
            c="tab:red",
            edgecolors="white",
            linewidths=1.2,
            label="best stable mix",
        )
    ax.set_xlabel("maize fraction [-]")
    ax.set_ylabel("grass fraction [-]")
    ax.set_title("Daily methane (cattle = 1 − maize − grass)")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    # pH map; non-feasible cells stay white.
    im2 = ax.imshow(
        ph.T,
        origin="lower",
        extent=(fracs[0], fracs[-1], fracs[0], fracs[-1]),
        cmap="RdYlGn",
        vmin=5.5,
        vmax=8.0,
        aspect="auto",
    )
    plt.colorbar(im2, ax=ax, label="pH [-]")
    ax.plot([0, 1], [1, 0], color="black", linestyle="--", alpha=0.6)
    ax.set_xlabel("maize fraction [-]")
    ax.set_ylabel("grass fraction [-]")
    ax.set_title("Reactor pH")

    fig.suptitle(f"Substrate mix at Q_in = {Q_total:.0f} t FM/d, " f"V_liq = {V_LIQ:.0f} m³ @ 42 °C")
    fig.tight_layout()
    _savefig(fig, "06_section2_substrate_mix.png")

    return best_mix


# ===========================================================================
# Section 3 — Temperature sweep mesophilic ↔ thermophilic
# ===========================================================================


def study_temperature_sweep(n_workers, duration=30.0, Q=None):
    """Compare digester temperatures at a fixed substrate feed.

    The feed ``Q`` is normally the best mix of Section 2 (passed in by
    :func:`main`). If ``Q`` is ``None``, a 40/40/20 maize / cattle / grass
    mix at 20 t FM/d is used.

    All temperatures go to a single :class:`ParallelSimulator` through the
    per-scenario ``T_ad`` key. Every scenario starts from the same state,
    built for 42 °C, and runs ``duration`` days at its own temperature. The
    values are the state after that time, not steady states.

    The model corrects its rates and equilibrium constants with the
    temperature. It does not describe how the microbial community adapts to
    a temperature change, and its rate corrections rise without an upper
    limit. The sweep shows this model response; it is no recommendation to
    switch a plant to another temperature.

    Args:
        n_workers (int): Number of worker processes.
        duration (float): Simulated time per temperature [d].
        Q (Sequence[float] | None): Feed rates per substrate [t FM/d].
    """
    from pyadm1.simulation.parallel import ParallelSimulator

    print("\n" + "=" * 72)
    print("Section 3 — Temperature sweep (mesophilic ↔ thermophilic)")
    print("=" * 72)

    if Q is None:
        # Fallback: 3-substrate mix at 20 t FM/d.
        Q = [0.0] * 10
        Q[I_MAIZE] = 8.0
        Q[I_CATTLE] = 8.0
        Q[I_GRASS] = 4.0
    T_celsius = [35.0, 37.0, 40.0, 42.0, 45.0, 48.0, 50.0, 52.0, 55.0, 58.0]
    T_K = [T + 273.15 for T in T_celsius]

    print(
        f"Feed = {Q[I_MAIZE]:.2f} maize + {Q[I_CATTLE]:.2f} cattle " f"+ {Q[I_GRASS]:.2f} grass  =  {sum(Q):.2f} t FM/d total"
    )

    # One base ADM1 + one ParallelSimulator handles every T_ad now that
    # the parallel API takes T_ad as a per-scenario parameter.
    adm1, fs = _make_base_adm1(T_ad=T_AD_MESO, duration_days=duration)
    parallel = ParallelSimulator(adm1, n_workers=n_workers, verbose=False)
    state = _inoculum_state(fs, T_ad=T_AD_MESO, Q_ref=Q)

    scenarios = [{"Q": Q, "T_ad": T} for T in T_K]

    t0 = time.time()
    print(f"Running {len(scenarios)} temperatures on {parallel.n_workers} workers ...")
    rows = parallel.run_scenarios(
        scenarios,
        duration=duration,
        initial_state=state,
        dt=1.0,
    )
    print(f"  done in {time.time() - t0:.1f} s")

    q_ch4 = np.array([r.metrics.get("Q_ch4", np.nan) if r.success else np.nan for r in rows])
    ch4_pct = np.array([100.0 * r.metrics.get("CH4_content", np.nan) if r.success else np.nan for r in rows])
    ph = np.array([r.metrics.get("pH", np.nan) if r.success else np.nan for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
    for ax, y, ylabel, title, colour in zip(
        axes,
        [q_ch4, ch4_pct, ph],
        ["Q_CH₄ [Nm³/d]", "CH₄ content [%]", "pH [-]"],
        ["Daily methane production", "Methane content in biogas", "Reactor pH"],
        ["tab:blue", "tab:green", "tab:red"],
    ):
        ax.plot(T_celsius, y, "o-", color=colour)
        ax.axvspan(35.0, 42.0, alpha=0.10, color="tab:blue", label="mesophilic")
        ax.axvspan(48.0, 60.0, alpha=0.10, color="tab:red", label="thermophilic")
        ax.set_xlabel("T_ad [°C]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="best", fontsize=9)
    fig.suptitle(f"Temperature sweep at Q_in = {sum(Q):.0f} t FM/d, V_liq = {V_LIQ:.0f} m³")
    fig.tight_layout()
    _savefig(fig, "06_section3_temperature_sweep.png")

    print("\n  Q_CH4 in Nm³/d")
    print(f"  {'T [°C]':>8s} {'Q_CH4':>8s} {'CH4 %':>7s} {'pH':>5s}")
    for T, Q_ch4, pct, ph_v in zip(T_celsius, q_ch4, ch4_pct, ph):
        print(f"  {T:>8.1f} {Q_ch4:>8.1f} {pct:>7.1f} {ph_v:>5.2f}")


# ---------------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------------
def _savefig(fig, filename):
    output_path = REPO_ROOT / "output"
    output_path.mkdir(exist_ok=True)
    path = output_path / filename
    fig.savefig(path, dpi=120)
    print(f"  saved plot to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(n_workers=4):
    study_hrt_olr_envelope(n_workers=n_workers)
    best_mix = study_substrate_mix(n_workers=n_workers)

    # Section 3 runs the temperature sweep at Section 2's best operating
    # point — same Q_total and same maize / cattle / grass split — so all
    # three studies share a consistent feed.
    Q_section3 = None
    if best_mix is not None:
        Q_section3 = [0.0] * 10
        Q_section3[I_MAIZE] = best_mix["Q_total"] * best_mix["maize_frac"]
        Q_section3[I_CATTLE] = best_mix["Q_total"] * best_mix["cattle_frac"]
        Q_section3[I_GRASS] = best_mix["Q_total"] * best_mix["grass_frac"]

    study_temperature_sweep(n_workers=n_workers, Q=Q_section3)

    # Plots are saved to output/ as PNGs — open them from there. No
    # plt.show() here on purpose: blocking on three GUI windows is more
    # annoying than just viewing the saved files.
    plt.close("all")
    print(f"\nAll plots saved to {(REPO_ROOT / 'output').resolve()}")


if __name__ == "__main__":
    # The report uses symbols (CH₄, ≥, ↔) that a Windows console code page may
    # not have; write UTF-8 so they survive a redirect, and a replacement
    # character where the console cannot show them.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
