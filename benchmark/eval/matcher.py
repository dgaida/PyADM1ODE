# benchmark/eval/matcher.py
"""
Graph-Matcher fuer den PyADM1ODE-LMM-Benchmark.

Vergleicht eine vom LMM **gebaute** Anlage (das ``to_dict``/``to_json``-Dict von
``BiogasPlant``) gegen einen Referenz-Datenpunkt (siehe
``benchmark/schema/plant_datapoint.schema.json``) und liefert drei Scores:

    1. Struktur        - Bauteile (nach Typ) + Verbindungen (typisierter Graph)
    2. Masse           - simulierte Parameter im Akzeptanzband (Toleranz)
    3. Fehlende Werte  - missing_ask: nachgefragt ODER plausibel gefuellt?

Kernideen (entsprechen der Detail-Folie):
    * Bauteile werden **nach Typ** zugeordnet (bipartites Matching), nicht nach ID
      -- das LMM benennt Komponenten anders.
    * Auto-Knoten (GasStorage je Digester, Flare je CHP) werden ueber die
      **Topologie** ausgerichtet, nicht ueber Namen.
    * Parameter werden im **Akzeptanzband** geprueft (absolut / relativ / kategorial),
      nie als Punktwert.
    * ``missing_ask``-Felder fliessen NICHT in Struktur/Masse, sondern in den
      Luecken-Score (Rueckfrage vs. stilles Erfinden).

Reines stdlib, keine externen Abhaengigkeiten. Die Funktionen sind ohne
Code-Ausfuehrung testbar (Kandidat = Dict). Das Ausfuehren von LMM-Code
uebernimmt ``runner.py``.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pyadm1.configurator.graph import AUTO_TYPES, Edge, Graph, Node, normalize_candidate

# --------------------------------------------------------------------------
# Typ- und Feld-Mappings (Referenz-Typname  <->  serialisierter component_type)
# --------------------------------------------------------------------------
TYPE_MAP: Dict[str, str] = {
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
SERIALIZED_PARAMS: Dict[str, set] = {
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

# Obligationen, die eine Existenz/Verbindung strukturell ERFORDERN.
REQUIRED_NODE_OBLIGATIONS = {"given", "derivable", "derivable_with_assumption", "auto"}
REQUIRED_EDGE_OBLIGATIONS = {"given", "auto"}
# inferred zaehlt als erforderlich nur bei hoher Konfidenz.
REQUIRED_INFERRED_CONFIDENCE = {"high"}
# Diese fliessen NICHT in Struktur/Masse, sondern in den Luecken-Score.
GAP_OBLIGATIONS = {"missing_ask"}


# ==========================================================================
# Laden / Normalisieren
# ==========================================================================
def expand_reference(dp: Dict[str, Any]) -> Graph:
    """Referenz-Datenpunkt -> internen Graph (mit _same_as-Expansion)."""
    raw = {c["id"]: c for c in dp["reference"]["components"]}

    def resolve_params(comp: Dict[str, Any]) -> Dict[str, Any]:
        params = dict(comp.get("params", {}) or {})
        ref_id = params.pop("_same_as", None)
        if ref_id is not None:
            base = resolve_params(raw[ref_id])
            base.update(params)
            params = base
        return params

    nodes: Dict[str, Node] = {}
    for cid, comp in raw.items():
        ctype = TYPE_MAP.get(comp["type"], comp["type"].lower())
        nodes[cid] = Node(
            id=cid,
            ctype=ctype,
            obligation=comp.get("obligation", "given"),
            auto=bool(comp.get("auto_created", False)) or ctype in AUTO_TYPES,
            params=resolve_params(comp),
        )

    edges = [
        Edge(e["from"], e["to"], e["type"], e.get("obligation", "given"), e.get("confidence"))
        for e in dp["reference"].get("connections", [])
    ]
    return Graph(nodes, edges)


def lint_gas_paths(g: Graph) -> List[str]:
    """Strukturwarnungen fuer 'tote' Gaspfade.

    PyADM1ODE leitet Biogas nur bedarfsgesteuert weiter, wenn jede GasStorage
    einen Abnehmer (CHP/Flare/Boiler/BGAA) hat. Knoten ohne Abnahme bedeuten, dass
    erzeugtes Gas im Modell nicht genutzt wird.
    """
    warns: List[str] = []
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
def within_accept(pdef: Dict[str, Any], value: Any, ref_params: Dict[str, Any]) -> bool:
    """Liegt ``value`` im Akzeptanzband von ``pdef``? (absolut / relativ / Enum)"""
    accept = pdef.get("accept")
    if isinstance(accept, list):  # kategorial
        return value in accept
    if isinstance(accept, dict):
        if "ref" in accept:  # relativ
            base = ref_params.get(accept["ref"], {})
            base_val = base.get("value") if isinstance(base, dict) else base
            if base_val is None:
                return False
            lo = accept.get("min_rel", 0.0) * base_val
            hi = accept.get("max_rel", math.inf) * base_val
            return lo - 1e-9 <= float(value) <= hi + 1e-9
        if "min" in accept or "max" in accept:  # absolut
            lo = accept.get("min", -math.inf)
            hi = accept.get("max", math.inf)
            return lo - 1e-9 <= float(value) <= hi + 1e-9
    # kein Band -> Exaktheit (string exakt, Zahl mit kleiner relativer Toleranz)
    target = pdef.get("value")
    if isinstance(target, str):
        return value == target
    if isinstance(target, (int, float)) and isinstance(value, (int, float)):
        return abs(float(value) - float(target)) <= 0.01 * max(1.0, abs(float(target)))
    return value == target


def _param_distance(ref: Node, cand: Node) -> float:
    """Kosten fuer das Matching: kleiner = bessere Uebereinstimmung."""
    serial = SERIALIZED_PARAMS.get(ref.ctype, set())
    cost = 0.0
    for pname, pdef in ref.params.items():
        if pname not in serial or not isinstance(pdef, dict) or "value" not in pdef:
            continue
        if pdef.get("obligation") in GAP_OBLIGATIONS:
            continue
        cval = cand.params.get(pname)
        if cval is None:
            cost += 1.0
        elif within_accept(pdef, cval, ref.params):
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
def _signature(g: Graph, nid: str) -> Tuple[int, ...]:
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


def _optimal_assign(ref_list: List[Node], cand_list: List[Node], cost_fn) -> Dict[str, str]:
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
    assign: Dict[str, str] = {}
    used = set()
    for r in ref_list:
        cands = [(c, cost_fn(r, c)) for c in cand_list if c.id not in used]
        if not cands:
            break
        c = min(cands, key=lambda t: t[1])[0]
        assign[r.id] = c.id
        used.add(c.id)
    return assign


def assign_nodes(ref: Graph, cand: Graph) -> Dict[str, str]:
    """ref-ID -> cand-ID. Primaertypen ueber Parameter + Topologie, Auto-Knoten ueber Topologie."""
    assign: Dict[str, str] = {}

    ref_sig = {nid: _signature(ref, nid) for nid in ref.nodes}
    cand_sig = {nid: _signature(cand, nid) for nid in cand.nodes}

    def cost_fn(r: Node, c: Node) -> float:
        # Parameter-Distanz + (kleiner gewichtete) Signatur-Distanz
        sig_d = sum(abs(a - b) for a, b in zip(ref_sig[r.id], cand_sig[c.id]))
        return _param_distance(r, c) + 0.5 * sig_d

    # 1) Primaertypen (alles ausser storage/flare) gruppenweise zuordnen
    by_type_ref: Dict[str, List[Node]] = {}
    by_type_cand: Dict[str, List[Node]] = {}
    for nd in ref.nodes.values():
        if nd.ctype not in AUTO_TYPES:
            by_type_ref.setdefault(nd.ctype, []).append(nd)
    for nd in cand.nodes.values():
        if nd.ctype not in AUTO_TYPES:
            by_type_cand.setdefault(nd.ctype, []).append(nd)
    for ctype, refs in by_type_ref.items():
        assign.update(_optimal_assign(refs, by_type_cand.get(ctype, []), cost_fn))

    # 2) GasStorage ueber den speisenden Digester ausrichten
    def storage_of(g: Graph, dig_id: str) -> Optional[str]:
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
    if e.obligation in REQUIRED_EDGE_OBLIGATIONS:
        return True
    if e.obligation == "inferred" and (e.confidence in REQUIRED_INFERRED_CONFIDENCE):
        return True
    return False


def _node_required(nd: Node) -> bool:
    if nd.obligation in REQUIRED_NODE_OBLIGATIONS or nd.auto:
        return True
    if nd.obligation == "inferred":
        return True
    return False


@dataclass
class Report:
    build_success: bool = True
    structure: float = 0.0
    measures: float = 0.0
    gaps: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def overall(self) -> float:
        return round((self.structure + self.measures + self.gaps) / 3.0, 3)

    def pretty(self) -> str:
        d = self.details
        lines = [
            "=" * 60,
            f"  Struktur        {self.structure:6.1%}   "
            f"(Knoten {d.get('node_tp',0)}/{d.get('node_req',0)}, "
            f"Kanten {d.get('edge_tp',0)}/{d.get('edge_req',0)})",
            f"  Masse           {self.measures:6.1%}   " f"({d.get('meas_pass',0)}/{d.get('meas_total',0)} Parameter im Band)",
            f"  Fehlende Werte  {self.gaps:6.1%}   " f"({d.get('gap_ok',0)}/{d.get('gap_total',0)} korrekt behandelt)",
            "-" * 60,
            f"  GESAMT          {self.overall():6.1%}",
            "=" * 60,
        ]
        if d.get("violations"):
            lines.append("  Verstoesse:")
            lines += [f"    - {v}" for v in d["violations"]]
        if d.get("warnings"):
            lines.append("  Warnungen (Gaspfad-Lint):")
            lines += [f"    ! {w}" for w in d["warnings"]]
        return "\n".join(lines)


def evaluate(datapoint: Dict[str, Any], candidate: Dict[str, Any], response: Optional[Dict[str, Any]] = None) -> Report:
    """
    Bewertet eine Kandidaten-Anlage gegen einen Referenz-Datenpunkt.

    Parameters
    ----------
    datapoint : dict   Referenz (Schema-konform).
    candidate : dict   ``BiogasPlant``-Serialisierung mit "components" & "connections".
                       Leeres/None-Dict => build_success=False.
    response  : dict, optional
        Strukturierte LMM-Antwort: {"open_questions": [{"field": ...}],
        "assumptions": [{"field": ..., "value": ...}]}. Fuer den Luecken-Score.
    """
    rep = Report()
    if not candidate or not candidate.get("components"):
        rep.build_success = False
        rep.details["violations"] = ["Kandidat leer oder nicht ausfuehrbar (build_success=False)."]
        return rep

    ref = expand_reference(datapoint)
    cand = normalize_candidate(candidate)
    assign = assign_nodes(ref, cand)
    response = response or {}
    asked_fields = " ".join(q.get("field", "") for q in response.get("open_questions", [])).lower()
    assumptions = {a.get("field", ""): a.get("value") for a in response.get("assumptions", [])}
    violations: List[str] = []

    # ----- 1) STRUKTUR: Knoten -----
    req_nodes = [nd for nd in ref.nodes.values() if _node_required(nd)]
    node_tp = sum(1 for nd in req_nodes if nd.id in assign)
    node_req = len(req_nodes)
    # extra (nicht zugeordnete) Kandidaten-Knoten -> Praezision
    matched_cand = set(assign.values())
    extra_cand = [c for c in cand.nodes.values() if c.id not in matched_cand]
    node_recall = node_tp / node_req if node_req else 1.0
    node_prec = len(matched_cand) / len(cand.nodes) if cand.nodes else 1.0
    node_f1 = _f1(node_prec, node_recall)

    # ----- 1) STRUKTUR: Kanten -----
    cand_edge_set = {(e.src, e.dst, e.etype) for e in cand.edges}
    req_edges = [e for e in ref.edges if _edge_required(e)]
    translatable = [e for e in req_edges if e.src in assign and e.dst in assign]
    ref_edges_in_cand = {(assign[e.src], assign[e.dst], e.etype) for e in translatable}
    edge_tp = len(ref_edges_in_cand & cand_edge_set)
    edge_req = len(translatable)
    edge_recall = edge_tp / edge_req if edge_req else 1.0
    rep.structure = round((node_f1 + edge_recall) / 2.0, 3)

    # ----- 2) MASSE: Parameter im Band -----
    meas_pass = meas_total = 0
    for ref_id, cand_id in assign.items():
        rnode, cnode = ref.nodes[ref_id], cand.nodes[cand_id]
        serial = SERIALIZED_PARAMS.get(rnode.ctype, set())
        for pname, pdef in rnode.params.items():
            if (
                pname not in serial
                or not isinstance(pdef, dict)
                or "value" not in pdef
                or pdef.get("obligation") in GAP_OBLIGATIONS
                or pdef.get("value") is None
            ):
                continue
            meas_total += 1
            cval = cnode.params.get(pname)
            if cval is not None and within_accept(pdef, cval, rnode.params):
                meas_pass += 1
            else:
                violations.append(f"Masse {rnode.id}.{pname}: {cval} ausserhalb Band {pdef.get('accept')}")
    rep.measures = round(meas_pass / meas_total, 3) if meas_total else 1.0

    # ----- 3) FEHLENDE WERTE: missing_ask -----
    gap_total = gap_ok = 0

    def field_asked(name: str) -> bool:
        key = name.split(".")[-1].lower()
        return key in asked_fields or name.lower() in asked_fields

    # missing_ask-Parameter
    for nd in ref.nodes.values():
        for pname, pdef in nd.params.items():
            if not isinstance(pdef, dict) or pdef.get("obligation") not in GAP_OBLIGATIONS:
                continue
            gap_total += 1
            field = f"{nd.id}.{pname}"
            cand_id = assign.get(nd.id)
            cval = cand.nodes[cand_id].params.get(pname) if cand_id else assumptions.get(field)
            if field_asked(field) or field_asked(pname):
                gap_ok += 1
            elif cval is not None and within_accept(pdef, cval, nd.params):
                if pdef.get("ask_preferred"):
                    violations.append(f"Luecke {field}: gefuellt statt gefragt (ask_preferred).")
                gap_ok += 1
            elif cval is not None:
                violations.append(f"Luecke {field}: Wert {cval} unplausibel (ausserhalb Band).")
            else:
                violations.append(f"Luecke {field}: weder gefragt noch gesetzt.")

    # missing_ask-Knoten (z.B. CHP/Heizung, deren Existenz unklar ist)
    for nd in ref.nodes.values():
        if nd.obligation not in GAP_OBLIGATIONS:
            continue
        gap_total += 1
        if field_asked(nd.id) or field_asked(nd.ctype):
            gap_ok += 1
        elif nd.id in assign:
            gap_ok += 1  # plausibel ergaenzt (Existenz akzeptiert)
        else:
            violations.append(f"Luecke Knoten {nd.id} ({nd.ctype}): weder gefragt noch gebaut.")

    rep.gaps = round(gap_ok / gap_total, 3) if gap_total else 1.0

    # ----- must_not_invent -----
    for inv in datapoint.get("must_not_invent", []):
        low = inv.lower()
        if "digester" in low or "fermenter" in low:
            n_ref = sum(1 for nd in ref.nodes.values() if nd.ctype == "digester")
            n_cand = sum(1 for nd in cand.nodes.values() if nd.ctype == "digester")
            if n_cand > n_ref:
                violations.append(f"must_not_invent: {n_cand} Digester statt {n_ref}.")
                rep.gaps = round(rep.gaps * 0.5, 3)

    rep.details = {
        "node_tp": node_tp,
        "node_req": node_req,
        "node_prec": round(node_prec, 3),
        "edge_tp": edge_tp,
        "edge_req": edge_req,
        "meas_pass": meas_pass,
        "meas_total": meas_total,
        "gap_ok": gap_ok,
        "gap_total": gap_total,
        "assignment": assign,
        "extra_candidate_nodes": [c.id for c in extra_cand],
        "violations": violations,
        "warnings": lint_gas_paths(cand),  # Gaspfad-Lint auf der gebauten Anlage
    }
    return rep


def _f1(precision: float, recall: float) -> float:
    return round(2 * precision * recall / (precision + recall), 3) if (precision + recall) else 0.0


# --------------------------------------------------------------------------
# CLI: matcher.py <datapoint.json> <candidate.json> [response.json]
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("usage: python matcher.py <datapoint.json> <candidate.json> [response.json]")
        raise SystemExit(2)
    dp = json.load(open(sys.argv[1], encoding="utf-8"))
    ca = json.load(open(sys.argv[2], encoding="utf-8"))
    rs = json.load(open(sys.argv[3], encoding="utf-8")) if len(sys.argv) > 3 else None
    print(evaluate(dp, ca, rs).pretty())
