# benchmark/eval/selftest.py
"""
Self-test of the graph matcher without running ADM1.

Builds two candidate plants as dicts (exactly the keys PyADM1ODE's ``to_dict``
produces) and checks them against benchmark/dataset/BGA3/BGA3_text_de.json:

    A) perfect candidate  -> deliberately DIFFERENT ids (proves type matching)
    B) broken candidate   -> wrong V_liq, missing edge, invented digester

Expected: A ~ 100 %, B clearly lower with concrete violations.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from matcher import evaluate  # noqa: E402

DP_PATH = os.path.join(HERE, "..", "dataset", "BGA3", "BGA3_text_de.json")


def dig(cid, V_liq, V_gas, T_ad):
    return {"component_id": cid, "component_type": "digester", "name": cid, "V_liq": V_liq, "V_gas": V_gas, "T_ad": T_ad}


def storage(cid, cap):
    return {"component_id": cid, "component_type": "storage", "name": cid, "storage_type": "membrane", "capacity_m3": cap}


def build_perfect():
    """The gold build, but with completely different identifiers."""
    comps = [
        dig("ferm_a", 2867, 719, 313.15),
        dig("ferm_b", 2867, 719, 313.15),
        dig("nachg", 2867, 719, 313.15),
        dig("lager", 3325, 799, 293.15),
        storage("ferm_a_storage", 719),
        storage("ferm_b_storage", 719),
        storage("nachg_storage", 719),
        storage("lager_storage", 799),
        {"component_id": "bhkw", "component_type": "chp", "name": "BHKW", "P_el_nom": 500.0, "eta_el": 0.40, "eta_th": 0.45},
        {"component_id": "bhkw_flare", "component_type": "flare", "name": "Fackel", "destruction_efficiency": 0.98},
    ]
    conns = [
        {"from": "ferm_a", "to": "nachg", "type": "liquid"},
        {"from": "ferm_b", "to": "nachg", "type": "liquid"},
        {"from": "nachg", "to": "lager", "type": "liquid"},
        {"from": "ferm_a", "to": "ferm_a_storage", "type": "gas"},
        {"from": "ferm_b", "to": "ferm_b_storage", "type": "gas"},
        {"from": "nachg", "to": "nachg_storage", "type": "gas"},
        {"from": "lager", "to": "lager_storage", "type": "gas"},
        {"from": "ferm_a_storage", "to": "bhkw", "type": "gas"},
        {"from": "ferm_b_storage", "to": "bhkw", "type": "gas"},
        {"from": "nachg_storage", "to": "bhkw", "type": "gas"},
        {"from": "lager_storage", "to": "bhkw", "type": "gas"},
        {"from": "bhkw", "to": "bhkw_flare", "type": "gas"},
    ]
    return {"plant_name": "Perfekt", "components": comps, "connections": conns}


def build_broken():
    """Several faults: wrong V_liq, a missing cascade edge, an invented digester,
    an implausibly large CHP."""
    p = build_perfect()
    comps = {c["component_id"]: dict(c) for c in p["components"]}
    comps["ferm_a"]["V_liq"] = 3186  # filled to 6 m -> outside [2548,3028]
    comps["bhkw"]["P_el_nom"] = 2000.0  # outside [450,550]
    comps["ferm_x"] = dig("ferm_x", 2867, 719, 313.15)  # invented 5th digester
    conns = [c for c in p["connections"] if not (c["from"] == "nachg" and c["to"] == "lager")]  # cascade edge missing
    return {"plant_name": "Kaputt", "components": list(comps.values()), "connections": conns}


def main():
    with open(DP_PATH, encoding="utf-8") as f:
        dp = json.load(f)

    print("\n### A) PERFEKTER KANDIDAT (andere IDs) ###")
    print(evaluate(dp, build_perfect()).pretty())

    print("\n### B) KAPUTTER KANDIDAT ###")
    print(evaluate(dp, build_broken()).pretty())

    print("\n### C) LEERER KANDIDAT (build_success=False) ###")
    print(evaluate(dp, {}).pretty())

    print("\n### D) TOTER GASPFAD (Lager-Storage ohne Abnehmer) ###")
    dead = build_perfect()
    dead["connections"] = [c for c in dead["connections"] if not (c["from"] == "lager_storage" and c["to"] == "bhkw")]
    print(evaluate(dp, dead).pretty())


if __name__ == "__main__":
    main()
