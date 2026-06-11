# benchmark/eval/selftest.py
"""
Selbsttest des Graph-Matchers ohne ADM1-Ausfuehrung.

Baut zwei Kandidaten-Anlagen als Dicts (genau die Schluessel, die PyADM1ODE
``to_dict`` erzeugt) und prueft sie gegen benchmark/examples/BGA1.json:

    A) perfekter Kandidat  -> bewusst ANDERE IDs (beweist Typ-Matching)
    B) kaputter Kandidat   -> falsche V_liq, fehlende Kante, erfundener Digester

Erwartung: A ~ 100 %, B deutlich niedriger mit konkreten Verstoessen.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from matcher import evaluate  # noqa: E402

DP_PATH = os.path.join(HERE, "..", "dataset", "BGA1", "BGA1.json")


def dig(cid, V_liq, V_gas, T_ad):
    return {"component_id": cid, "component_type": "digester", "name": cid, "V_liq": V_liq, "V_gas": V_gas, "T_ad": T_ad}


def storage(cid, cap):
    return {"component_id": cid, "component_type": "storage", "name": cid, "storage_type": "membrane", "capacity_m3": cap}


def heating(cid, T):
    return {
        "component_id": cid,
        "component_type": "heating",
        "name": cid,
        "target_temperature": T,
        "heat_loss_coefficient": 0.5,
    }


def build_perfect():
    """Gold-Build, aber mit voellig anderen Bezeichnern."""
    comps = [
        dig("ferm_a", 3325, 500, 313.15),
        dig("ferm_b", 3325, 500, 313.15),
        dig("nachg", 3325, 500, 313.15),
        dig("lager", 3817, 570, 293.15),
        storage("ferm_a_storage", 500),
        storage("ferm_b_storage", 500),
        storage("nachg_storage", 500),
        storage("lager_storage", 570),
        {"component_id": "bhkw", "component_type": "chp", "name": "BHKW", "P_el_nom": 500.0, "eta_el": 0.40, "eta_th": 0.45},
        {"component_id": "bhkw_flare", "component_type": "flare", "name": "Fackel", "destruction_efficiency": 0.98},
        {
            "component_id": "trenner",
            "component_type": "separator",
            "name": "Separator",
            "separator_type": "screw_press",
            "separation_efficiency": 0.60,
        },
        heating("heiz_a", 313.15),
        heating("heiz_b", 313.15),
        heating("heiz_c", 313.15),
    ]
    conns = [
        {"from": "ferm_a", "to": "nachg", "type": "liquid"},
        {"from": "ferm_b", "to": "nachg", "type": "liquid"},
        {"from": "nachg", "to": "lager", "type": "liquid"},
        {"from": "nachg", "to": "trenner", "type": "liquid"},
        {"from": "ferm_a", "to": "ferm_a_storage", "type": "gas"},
        {"from": "ferm_b", "to": "ferm_b_storage", "type": "gas"},
        {"from": "nachg", "to": "nachg_storage", "type": "gas"},
        {"from": "lager", "to": "lager_storage", "type": "gas"},
        {"from": "ferm_a_storage", "to": "bhkw", "type": "gas"},
        {"from": "ferm_b_storage", "to": "bhkw", "type": "gas"},
        {"from": "nachg_storage", "to": "bhkw", "type": "gas"},
        {"from": "lager_storage", "to": "bhkw", "type": "gas"},
        {"from": "bhkw", "to": "bhkw_flare", "type": "gas"},
        {"from": "bhkw", "to": "heiz_a", "type": "heat"},
        {"from": "bhkw", "to": "heiz_b", "type": "heat"},
        {"from": "bhkw", "to": "heiz_c", "type": "heat"},
    ]
    return {"plant_name": "Perfekt", "components": comps, "connections": conns}


def build_broken():
    """Mehrere Fehler: falsche V_liq, fehlende Kaskaden-Kante, erfundener Digester,
    BHKW unplausibel gross."""
    p = build_perfect()
    comps = {c["component_id"]: dict(c) for c in p["components"]}
    comps["ferm_a"]["V_liq"] = 3695  # auf 6 m gefuellt -> ausserhalb [2956,3510]
    comps["bhkw"]["P_el_nom"] = 2000.0  # missing_ask: ausserhalb [250,1000]
    comps["ferm_x"] = dig("ferm_x", 3325, 500, 313.15)  # erfundener 5. Digester
    conns = [c for c in p["connections"] if not (c["from"] == "nachg" and c["to"] == "lager")]  # Kaskaden-Kante fehlt
    return {"plant_name": "Kaputt", "components": list(comps.values()), "connections": conns}


def main():
    dp = json.load(open(DP_PATH, encoding="utf-8"))

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
