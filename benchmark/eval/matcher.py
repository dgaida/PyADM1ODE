# benchmark/eval/matcher.py
"""
Graph-Matcher fuer den PyADM1ODE-LMM-Benchmark.

Vergleicht eine vom LMM **gebaute** Anlage (das ``to_dict``/``to_json``-Dict von
``BiogasPlant``) gegen einen Referenz-Datenpunkt (siehe
``benchmark/schema/plant_datapoint.schema.json``) und liefert drei Scores:

    1. Vollstaendigkeit - sind alle Pflicht-Bauteile und Pflicht-Kanten da? (Recall)
    2. Masse            - simulierte Parameter im Akzeptanzband (Toleranz)
    3. Keine Erfindungen- enthaelt der Kandidat NUR Bauteile/Kanten, die die
                          Referenz kennt? (Precision)

Die drei Achsen sind bewusst disjunkt, damit jeder Fehler genau einmal zaehlt:

    weggelassen  -> Score 1     hinzuerfunden -> Score 3     falscher Wert -> Score 2

Kernideen:
    * Bauteile werden **nach Typ** zugeordnet (bipartites Matching), nicht nach ID
      -- das LMM benennt Komponenten anders.
    * Auto-Knoten (GasStorage je Digester, Flare je CHP) werden ueber die
      **Topologie** ausgerichtet, nicht ueber Namen.
    * Parameter werden im **Akzeptanzband** geprueft (absolut / relativ / kategorial),
      nie als Punktwert. Ein still erfundener, unplausibler Wert senkt damit Score 2.
    * **Pflicht** (Score 1) und **erlaubt** (Score 3) sind zwei verschiedene Mengen:
      eine Kante mit ``obligation: missing_ask`` muss nicht gebaut werden, gilt aber
      auch nicht als Erfindung, wenn sie gebaut wird.
    * Ein Bauteil eines Typs, den die Referenz ueberhaupt nicht kennt (Separator in
      einer Anlage ohne Separator), deckelt Score 3 hart -- das ist die schwerste
      Form der Halluzination.

Reines stdlib, keine externen Abhaengigkeiten. Die Funktionen sind ohne
Code-Ausfuehrung testbar (Kandidat = Dict). Das Ausfuehren von LMM-Code
uebernimmt ``runner.py``.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, field
from typing import Any

from pyadm1.configurator.graph import AUTO_TYPES, Edge, Graph, Node, normalize_candidate

# --------------------------------------------------------------------------
# Typ- und Feld-Mappings (Referenz-Typname  <->  serialisierter component_type)
# --------------------------------------------------------------------------
TYPE_MAP: dict[str, str] = {
    "Digester": "digester",
    "GasStorage": "storage",
    "Separator": "separator",
    "CHP": "chp",
    "Flare": "flare",
    "HeatingSystem": "heating",
    "Boiler": "boiler",
    "BiogasUpgrading": "upgrading",
    "Mixer": "mixer",
}

# Parameter, die PyADM1ODE pro Typ tatsaechlich serialisiert (to_dict).
SERIALIZED_PARAMS: dict[str, set] = {
    "digester": {"V_liq", "V_gas", "T_ad"},
    "chp": {"P_el_nom", "eta_el", "eta_th"},
    "heating": {"target_temperature", "heat_loss_coefficient"},
    "storage": {"capacity_m3", "storage_type"},
    "separator": {"separator_type", "separation_efficiency", "ts_solid_target", "n_to_solid", "p_to_solid"},
    "flare": {"destruction_efficiency"},
    "upgrading": {"capacity_m3h", "ch4_recovery", "ch4_content_in", "ch4_content_out"},
    "boiler": set(),
    "mixer": set(),
}

# Typen, die Gas abnehmen/versenken (gueltige Endpunkte eines Gaspfads).
GAS_CONSUMER_TYPES = {"chp", "flare", "boiler", "upgrading"}

# Obligationen, die eine Existenz/Verbindung strukturell ERFORDERN (Score 1).
# Nicht enthalten -- und damit optional -- ist ausschliesslich "missing_ask":
# das Element gehoert zur Anlage, muss aber nicht gebaut werden.
REQUIRED_NODE_OBLIGATIONS = {"given", "derivable", "inferred", "auto"}
REQUIRED_EDGE_OBLIGATIONS = {"given", "inferred", "auto"}
# Deckel fuer Score 3, wenn ein Bauteil eines in der Referenz unbekannten Typs auftaucht.
INVENTED_TYPE_CAP = 0.5


# ==========================================================================
# Laden / Normalisieren
# ==========================================================================
def expand_reference(dp: dict[str, Any]) -> Graph:
    """Referenz-Datenpunkt -> internen Graph (mit _same_as-Expansion)."""
    raw = {c["id"]: c for c in dp["reference"]["components"]}

    def resolve_params(comp: dict[str, Any]) -> dict[str, Any]:
        params = dict(comp.get("params", {}) or {})
        ref_id = params.pop("_same_as", None)
        if ref_id is not None:
            base = resolve_params(raw[ref_id])
            base.update(params)
            params = base
        return params

    nodes: dict[str, Node] = {}
    for cid, comp in raw.items():
        ctype = TYPE_MAP.get(comp["type"], comp["type"].lower())
        nodes[cid] = Node(
            id=cid,
            ctype=ctype,
            obligation=comp.get("obligation", "given"),
            auto=bool(comp.get("auto_created", False)) or ctype in AUTO_TYPES,
            params=resolve_params(comp),
        )

    edges = [Edge(e["from"], e["to"], e["type"], e.get("obligation", "given")) for e in dp["reference"].get("connections", [])]
    return Graph(nodes, edges)


def lint_gas_paths(g: Graph) -> list[str]:
    """Strukturwarnungen fuer 'tote' Gaspfade.

    PyADM1ODE leitet Biogas nur bedarfsgesteuert weiter, wenn jede GasStorage
    einen Abnehmer (CHP/Flare/Boiler/BGAA) hat. Knoten ohne Abnahme bedeuten, dass
    erzeugtes Gas im Modell nicht genutzt wird.
    """
    warns: list[str] = []
    for nd in g.nodes.values():
        if nd.ctype == "storage":
            outs = g.out_edges(nd.id, "gas")
            if not any(g.nodes.get(e.dst) and g.nodes[e.dst].ctype in GAS_CONSUMER_TYPES for e in outs):
                warns.append(
                    f"GasStorage '{nd.id}' hat keinen Gas-Abnehmer "
                    f"(CHP/Flare/Boiler/BGAA) -> Gas wird nicht weitergeleitet."
                )
        elif nd.ctype == "digester":
            if not g.out_edges(nd.id, "gas"):
                warns.append(f"Digester '{nd.id}' hat keine Gas-Kante -> erzeugtes " f"Biogas wird nicht erfasst.")
    return warns


# ==========================================================================
# Akzeptanz-Pruefung (ein Parameter)
# ==========================================================================
def within_accept(pdef: dict[str, Any], value: Any) -> bool:
    """Liegt ``value`` im Akzeptanzband von ``pdef``?

    Die Bandbreite steckt im Datenpunkt, nicht hier: sie richtet sich danach, ob der
    Wert uebernommen (+-0.1 %) oder gerechnet (+-1 %) wird. Fehlt ``accept``, wird
    exakt verglichen -- so bei kategorialen Werten wie ``separator_type``.
    """
    accept = pdef.get("accept")
    if isinstance(accept, dict) and ("min" in accept or "max" in accept):
        lo = accept.get("min", -math.inf)
        hi = accept.get("max", math.inf)
        return lo - 1e-9 <= float(value) <= hi + 1e-9
    return value == pdef.get("value")


def _param_distance(ref: Node, cand: Node) -> float:
    """Kosten fuer das Matching: kleiner = bessere Uebereinstimmung."""
    serial = SERIALIZED_PARAMS.get(ref.ctype, set())
    cost = 0.0
    for pname, pdef in ref.params.items():
        if pname not in serial or not isinstance(pdef, dict) or "value" not in pdef:
            continue
        cval = cand.params.get(pname)
        if cval is None:
            cost += 1.0
        elif within_accept(pdef, cval):
            cost += 0.0
        else:
            tv = pdef["value"]
            if isinstance(tv, (int, float)) and isinstance(cval, (int, float)):
                cost += min(1.0, abs(float(cval) - float(tv)) / max(1.0, abs(float(tv))))
            else:
                cost += 1.0
    return cost


# ==========================================================================
# Zuordnung der Knoten (Typ-Gruppen)
# ==========================================================================
def _signature(g: Graph, nid: str) -> tuple[int, ...]:
    """Topologische Signatur eines Knotens: Kantengrade je Typ/Richtung.

    Unterscheidet sonst parameter-gleiche Knoten (z. B. Nachgaerer vs. Fermenter:
    der Nachgaerer hat zwei eingehende liquid-Kanten)."""
    return (
        len(g.out_edges(nid, "liquid")),
        len(g.in_edges(nid, "liquid")),
        len(g.out_edges(nid, "gas")),
        len(g.in_edges(nid, "gas")),
        len(g.out_edges(nid, "heat")),
        len(g.in_edges(nid, "heat")),
    )


def _optimal_assign(ref_list: list[Node], cand_list: list[Node], cost_fn) -> dict[str, str]:
    """Minimiert die Summe von ``cost_fn``; brute-force fuer kleine Gruppen."""
    if not ref_list or not cand_list:
        return {}
    n, m = len(ref_list), len(cand_list)
    if n <= 7 and m <= 8:
        best_cost, best = math.inf, None
        for combo in itertools.permutations(range(m), min(n, m)):
            cost = sum(cost_fn(ref_list[i], cand_list[combo[i]]) for i in range(len(combo)))
            if cost < best_cost:
                best_cost, best = cost, combo
        return {ref_list[i].id: cand_list[best[i]].id for i in range(len(best))} if best else {}
    # Greedy-Fallback fuer grosse Gruppen
    assign: dict[str, str] = {}
    used = set()
    for r in ref_list:
        cands = [(c, cost_fn(r, c)) for c in cand_list if c.id not in used]
        if not cands:
            break
        c = min(cands, key=lambda t: t[1])[0]
        assign[r.id] = c.id
        used.add(c.id)
    return assign


def assign_nodes(ref: Graph, cand: Graph) -> dict[str, str]:
    """ref-ID -> cand-ID. Primaertypen ueber Parameter + Topologie, Auto-Knoten ueber Topologie."""
    assign: dict[str, str] = {}

    ref_sig = {nid: _signature(ref, nid) for nid in ref.nodes}
    cand_sig = {nid: _signature(cand, nid) for nid in cand.nodes}

    def cost_fn(r: Node, c: Node) -> float:
        # Parameter-Distanz + (kleiner gewichtete) Signatur-Distanz
        sig_d = sum(abs(a - b) for a, b in zip(ref_sig[r.id], cand_sig[c.id]))
        return _param_distance(r, c) + 0.5 * sig_d

    # 1) Primaertypen (alles ausser storage/flare) gruppenweise zuordnen
    by_type_ref: dict[str, list[Node]] = {}
    by_type_cand: dict[str, list[Node]] = {}
    for nd in ref.nodes.values():
        if nd.ctype not in AUTO_TYPES:
            by_type_ref.setdefault(nd.ctype, []).append(nd)
    for nd in cand.nodes.values():
        if nd.ctype not in AUTO_TYPES:
            by_type_cand.setdefault(nd.ctype, []).append(nd)
    for ctype, refs in by_type_ref.items():
        assign.update(_optimal_assign(refs, by_type_cand.get(ctype, []), cost_fn))

    # 2) GasStorage ueber den speisenden Digester ausrichten
    def storage_of(g: Graph, dig_id: str) -> str | None:
        for e in g.out_edges(dig_id, "gas"):
            if e.dst in g.nodes and g.nodes[e.dst].ctype == "storage":
                return e.dst
        return None

    used_storages = set()
    for nd in ref.nodes.values():
        if nd.ctype != "storage":
            continue
        # speisender Digester in der Referenz
        src_dig = next(
            (e.src for e in ref.in_edges(nd.id, "gas") if ref.nodes.get(e.src) and ref.nodes[e.src].ctype == "digester"), None
        )
        cand_storage = None
        if src_dig and src_dig in assign:
            cand_storage = storage_of(cand, assign[src_dig])
        if cand_storage and cand_storage not in used_storages:
            assign[nd.id] = cand_storage
            used_storages.add(cand_storage)

    # 3) Flare ueber den speisenden CHP ausrichten (sonst per Rest/Anzahl)
    used_flares = set()
    for nd in ref.nodes.values():
        if nd.ctype != "flare":
            continue
        src_chp = next(
            (e.src for e in ref.in_edges(nd.id, "gas") if ref.nodes.get(e.src) and ref.nodes[e.src].ctype == "chp"), None
        )
        cand_flare = None
        if src_chp and src_chp in assign:
            for e in cand.out_edges(assign[src_chp], "gas"):
                if cand.nodes.get(e.dst) and cand.nodes[e.dst].ctype == "flare":
                    cand_flare = e.dst
                    break
        if cand_flare is None:
            cand_flare = next((c.id for c in cand.nodes.values() if c.ctype == "flare" and c.id not in used_flares), None)
        if cand_flare and cand_flare not in used_flares:
            assign[nd.id] = cand_flare
            used_flares.add(cand_flare)

    return assign


# ==========================================================================
# Scoring
# ==========================================================================
def _edge_required(e: Edge) -> bool:
    return e.obligation in REQUIRED_EDGE_OBLIGATIONS


def _node_required(nd: Node) -> bool:
    return nd.obligation in REQUIRED_NODE_OBLIGATIONS or nd.auto


@dataclass
class Report:
    build_success: bool = True
    completeness: float = 0.0
    measures: float = 0.0
    inventions: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def overall(self) -> float:
        return round((self.completeness + self.measures + self.inventions) / 3.0, 3)

    def pretty(self) -> str:
        d = self.details
        lines = [
            "=" * 60,
            (
                f"  Vollstaendigkeit  {self.completeness:6.1%}   "
                f"(Knoten {d.get('node_found',0)}/{d.get('node_req',0)}, "
                f"Kanten {d.get('edge_found',0)}/{d.get('edge_req',0)})"
            ),
            (
                f"  Masse             {self.measures:6.1%}   "
                f"({d.get('meas_pass',0)}/{d.get('meas_total',0)} Parameter im Band)"
            ),
            (
                f"  Keine Erfindungen {self.inventions:6.1%}   "
                f"({d.get('n_extra_nodes',0)} Bauteile, {d.get('n_extra_edges',0)} Kanten erfunden)"
            ),
            "-" * 60,
            f"  GESAMT            {self.overall():6.1%}",
            "=" * 60,
        ]
        if d.get("violations"):
            lines.append("  Verstoesse:")
            lines += [f"    - {v}" for v in d["violations"]]
        if d.get("warnings"):
            lines.append("  Warnungen (Gaspfad-Lint):")
            lines += [f"    ! {w}" for w in d["warnings"]]
        return "\n".join(lines)


def evaluate(datapoint: dict[str, Any], candidate: dict[str, Any]) -> Report:
    """
    Bewertet eine Kandidaten-Anlage gegen einen Referenz-Datenpunkt.

    Parameters
    ----------
    datapoint : dict   Referenz (Schema-konform).
    candidate : dict   ``BiogasPlant``-Serialisierung mit "components" & "connections".
                       Leeres/None-Dict => build_success=False.
    """
    rep = Report()
    if not candidate or not candidate.get("components"):
        rep.build_success = False
        rep.details["violations"] = ["Kandidat leer oder nicht ausfuehrbar (build_success=False)."]
        return rep

    ref = expand_reference(datapoint)
    cand = normalize_candidate(candidate)
    assign = assign_nodes(ref, cand)
    violations: list[str] = []
    cand_edge_set = {(e.src, e.dst, e.etype) for e in cand.edges}

    # ======================================================================
    # 1) VOLLSTAENDIGKEIT -- ist alles Noetige da? (Recall)
    # ======================================================================
    req_nodes = [nd for nd in ref.nodes.values() if _node_required(nd)]
    node_found = 0
    for nd in req_nodes:
        if nd.id in assign:
            node_found += 1
        else:
            violations.append(f"Fehlend: Bauteil '{nd.id}' ({nd.ctype}) wurde nicht gebaut.")
    node_recall = node_found / len(req_nodes) if req_nodes else 1.0

    # Der Nenner umfasst ALLE Pflicht-Kanten. Faellt ein Knoten weg, verschwinden
    # seine Kanten nicht aus der Rechnung -- Weglassen darf sich nicht lohnen.
    req_edges = [e for e in ref.edges if _edge_required(e)]
    edge_found = 0
    for e in req_edges:
        translated = (assign.get(e.src), assign.get(e.dst), e.etype)
        if None not in translated[:2] and translated in cand_edge_set:
            edge_found += 1
        else:
            violations.append(f"Fehlend: Verbindung {e.src} -> {e.dst} ({e.etype}).")
    edge_recall = edge_found / len(req_edges) if req_edges else 1.0
    rep.completeness = round((node_recall + edge_recall) / 2.0, 3)

    # ======================================================================
    # 2) MASSE -- stimmen die Werte? (auch still erfundene Werte landen hier)
    # ======================================================================
    meas_pass = meas_total = 0
    for ref_id, cand_id in assign.items():
        rnode, cnode = ref.nodes[ref_id], cand.nodes[cand_id]
        serial = SERIALIZED_PARAMS.get(rnode.ctype, set())
        for pname, pdef in rnode.params.items():
            if pname not in serial or not isinstance(pdef, dict) or "value" not in pdef or pdef.get("value") is None:
                continue
            meas_total += 1
            cval = cnode.params.get(pname)
            if cval is not None and within_accept(pdef, cval):
                meas_pass += 1
            else:
                violations.append(f"Masse {rnode.id}.{pname}: {cval} ausserhalb Band {pdef.get('accept')}")
    rep.measures = round(meas_pass / meas_total, 3) if meas_total else 1.0

    # ======================================================================
    # 3) KEINE ERFINDUNGEN -- enthaelt der Kandidat NUR Bekanntes? (Precision)
    # ======================================================================
    # "erlaubt" ist weiter gefasst als "Pflicht": auch optionale Referenz-Elemente
    # (missing_ask, inferred mit niedriger Konfidenz) sind keine Erfindung.
    ref_types = {nd.ctype for nd in ref.nodes.values()}
    matched_cand = set(assign.values())
    extra_nodes = [c for c in cand.nodes.values() if c.id not in matched_cand]
    node_prec = len(matched_cand) / len(cand.nodes) if cand.nodes else 1.0

    allowed_edges = {(assign[e.src], assign[e.dst], e.etype) for e in ref.edges if e.src in assign and e.dst in assign}
    extra_edges = [e for e in cand.edges if (e.src, e.dst, e.etype) not in allowed_edges]
    edge_prec = (len(cand.edges) - len(extra_edges)) / len(cand.edges) if cand.edges else 1.0

    rep.inventions = round((node_prec + edge_prec) / 2.0, 3)

    unknown_type = False
    for c in extra_nodes:
        if c.ctype in ref_types:
            violations.append(f"Erfunden: zusaetzliches Bauteil '{c.id}' ({c.ctype}) ohne Entsprechung in der Referenz.")
        else:
            unknown_type = True
            violations.append(f"Erfunden: Bauteil '{c.id}' vom Typ '{c.ctype}' -- die Referenzanlage hat keins.")
    for e in extra_edges:
        violations.append(f"Erfunden: Verbindung {e.src} -> {e.dst} ({e.etype}) gibt es in der Referenz nicht.")
    if unknown_type:
        rep.inventions = round(min(rep.inventions, INVENTED_TYPE_CAP), 3)

    rep.details = {
        "node_found": node_found,
        "node_req": len(req_nodes),
        "edge_found": edge_found,
        "edge_req": len(req_edges),
        "meas_pass": meas_pass,
        "meas_total": meas_total,
        "node_prec": round(node_prec, 3),
        "edge_prec": round(edge_prec, 3),
        "n_extra_nodes": len(extra_nodes),
        "n_extra_edges": len(extra_edges),
        "assignment": assign,
        "extra_candidate_nodes": [c.id for c in extra_nodes],
        "violations": violations,
        "warnings": lint_gas_paths(cand),  # Gaspfad-Lint auf der gebauten Anlage
    }
    return rep


# --------------------------------------------------------------------------
# CLI: matcher.py <datapoint.json> <candidate.json>
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("usage: python matcher.py <datapoint.json> <candidate.json>")
        raise SystemExit(2)
    with open(sys.argv[1], encoding="utf-8") as f:
        dp = json.load(f)
    with open(sys.argv[2], encoding="utf-8") as f:
        ca = json.load(f)
    print(evaluate(dp, ca).pretty())
