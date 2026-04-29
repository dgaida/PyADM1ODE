#!/usr/bin/env python3
# examples/01_basic_digester.py
"""
PyADM1 basic single-digester simulation — minimal example.

The substrate mix is supplied as a dict ``{substrate_id: Q [m³/d]}`` with no
upper bound on the substrate count.  The default mix is a co-digestion of
maize silage, swine manure, and cattle manure.

Digester configuration
----------------------
    Temperature       42 °C  (315.15 K)
    Liquid volume     1050 m³
    Gas headspace     150 m³

Usage
-----
    python examples/01_basic_digester.py
"""

from pathlib import Path
from typing import Dict, Mapping, Optional
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

# Default substrate mix [m³/d] — maize silage + swine + cattle manure.
DEFAULT_FEED: Dict[str, float] = {
    "maize_silage_milk_ripeness": 11.4,
    "swine_manure": 6.1,
    "cattle_manure": 5.0,
}


def main(feed: Optional[Mapping[str, float]] = None):
    """
    Run the basic single-digester example.

    Parameters
    ----------
    feed : mapping of {substrate_id: Q_m3_per_day}, optional
        Substrate mix to feed.  Each key is the XML file stem of a substrate
        in ``data/substrates/adm1da/`` and the value is its volumetric flow
        rate in m³/d.  Any number of substrates may be supplied.  Defaults
        to :data:`DEFAULT_FEED` (maize silage + swine + cattle manure).
    """
    from pyadm1 import BiogasPlant, Feedstock
    from pyadm1.configurator.plant_configurator import PlantConfigurator

    feed = dict(feed) if feed is not None else dict(DEFAULT_FEED)
    if not feed:
        raise ValueError("'feed' must contain at least one substrate.")

    substrate_ids = list(feed.keys())
    Q_substrates = list(feed.values())

    print("Substrate mix:")
    for sid, q in feed.items():
        print(f"  {sid:<35s} {q:>6.2f} m³/d")
    print(f"  {'TOTAL':<35s} {sum(Q_substrates):>6.2f} m³/d\n")

    # --- Build plant ---
    feedstock = Feedstock(substrate_ids, feeding_freq=24, total_simtime=160)
    plant = BiogasPlant("ADM1 Basic Plant")
    cfg = PlantConfigurator(plant, feedstock)

    cfg.add_digester(
        digester_id="main_digester",
        V_liq=1050.0,
        V_gas=150.0,
        T_ad=315.15,
        Q_substrates=Q_substrates,
        k_L_a=200.0,
    )
    plant.initialize()

    # --- Simulate ---
    results = plant.simulate(duration=150.0, dt=1.0, save_interval=1.0)

    # --- Report ---
    times = np.array([r["time"] for r in results])
    dig = [r["components"]["main_digester"] for r in results]
    q_gas = np.array([c["Q_gas"] for c in dig])
    q_ch4 = np.array([c["Q_ch4"] for c in dig])
    pH = np.array([c["pH"] for c in dig])

    print(f"{'Day':>5s}  {'Biogas':>10s}  {'Methane':>10s}  {'CH4%':>6s}  {'pH':>6s}")
    print("-" * 50)
    for day in sorted(set(list(range(0, int(times[-1]) + 1, 10)) + [int(times[-1])])):
        i = int(np.argmin(np.abs(times - day)))
        ch4p = q_ch4[i] / q_gas[i] * 100.0 if q_gas[i] > 0.0 else 0.0
        print(f"{day:>5d}  {q_gas[i]:>10.1f}  {q_ch4[i]:>10.1f}  {ch4p:>6.1f}  {pH[i]:>6.2f}")

    # --- Save final configuration ---
    output_path = REPO_ROOT / "output"
    output_path.mkdir(exist_ok=True)
    plant.to_json(str(output_path / "quickstart_config.json"))

    return plant, results


if __name__ == "__main__":
    main()
