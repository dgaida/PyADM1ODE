# ============================================================================
# pyadm1/configurator/graph.py
# ============================================================================
"""
Typed component/connection graph for a biogas plant.

This is the shared data model used both by :meth:`BiogasPlant.to_graph` and by
the LLM benchmark's graph matcher. A :class:`Graph` holds one :class:`Node` per
component (keyed by component id, carrying its serialized ``component_type`` and
scalar parameters) and one :class:`Edge` per connection (``liquid`` / ``gas`` /
``heat``).

Pure stdlib, no external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Component types that PyADM1ODE creates automatically (one GasStorage per
# digester, one Flare per CHP). The benchmark aligns these via topology, so the
# graph flags them for downstream consumers.
AUTO_TYPES = {"storage", "flare"}


@dataclass
class Node:
    id: str
    ctype: str  # serialized component_type, e.g. "digester"
    obligation: str = "given"
    auto: bool = False
    # name -> {value, obligation, accept} (reference) OR name -> value (candidate)
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    src: str
    dst: str
    etype: str  # "liquid" | "gas" | "heat"
    obligation: str = "given"
    confidence: Optional[str] = None


@dataclass
class Graph:
    nodes: Dict[str, Node]
    edges: List[Edge]

    def out_edges(self, nid: str, etype: Optional[str] = None) -> List[Edge]:
        return [e for e in self.edges if e.src == nid and (etype is None or e.etype == etype)]

    def in_edges(self, nid: str, etype: Optional[str] = None) -> List[Edge]:
        return [e for e in self.edges if e.dst == nid and (etype is None or e.etype == etype)]


def normalize_candidate(cand: Dict[str, Any]) -> Graph:
    """``BiogasPlant`` serialization (to_json/to_dict structure) -> Graph."""
    nodes: Dict[str, Node] = {}
    for c in cand.get("components", []):
        cid = c["component_id"]
        ctype = c["component_type"]
        # Take all serialized scalar fields as parameters.
        params = {
            k: v
            for k, v in c.items()
            if k not in ("component_id", "component_type", "name", "inputs", "outputs", "state", "outputs_data")
            and isinstance(v, (int, float, str, bool))
        }
        nodes[cid] = Node(id=cid, ctype=ctype, auto=ctype in AUTO_TYPES, params=params)

    edges = [Edge(e["from"], e["to"], e["type"]) for e in cand.get("connections", [])]
    return Graph(nodes, edges)
