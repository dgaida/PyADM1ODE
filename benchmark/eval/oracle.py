# benchmark/eval/oracle.py
"""
Oracle: answers LLM questions from a datapoint's ``oracle`` dict.

It stands in for a human expert who supplies missing information (operating
temperature, gas storage volume, ...) when asked. The source of truth is the
"oracle" field of the datapoint JSON.
"""

from __future__ import annotations

import re
from typing import Any

# Canonical field names an LLM typically uses, mapped to oracle keys
_KEYWORD_MAP = {
    # operating temperature (T_ad)
    "t_ad": "T_ad",
    "betriebstemperatur": "T_ad",
    "prozesstemperatur": "T_ad",
    "temperatur": "T_ad",
    "temperature": "T_ad",
    "warm": "T_ad",
    "beheiz": "T_ad",
    "mesophil": "T_ad",
    "thermophil": "T_ad",
    # gas storage volume (V_gas)
    "v_gas": "V_gas",
    "gasspeicher": "V_gas",
    "gasraum": "V_gas",
    "gas_storage": "V_gas",
    "gas storage": "V_gas",
    "headspace": "V_gas",
    # fill level (fill_fraction)
    "fill_fraction": "fill_fraction",
    "füllgrad": "fill_fraction",
    "fuellgrad": "fill_fraction",
    "füllstand": "fill_fraction",
    "befüll": "fill_fraction",
    "fill level": "fill_fraction",
    "fillgrade": "fill_fraction",
    # rated electrical power (P_el_nom)
    "p_el_nom": "P_el_nom",
    "elektrische leistung": "P_el_nom",
    "leistung": "P_el_nom",
    "nennleistung": "P_el_nom",
    "rated power": "P_el_nom",
    # efficiencies (eta_el / eta_th) -- "wirkungsgrad" returns both
    "eta_el": "eta_el",
    "eta_th": "eta_th",
    "wirkungsgrad": "eta",
    "efficiency": "eta",
    # biogas upgrading unit (BGAA)
    "capacity_m3h": "capacity_m3h",
    "kapazität": "capacity_m3h",
    "ch4_recovery": "ch4_recovery",
    "methanausbeute": "ch4_recovery",
    "methane recovery": "ch4_recovery",
    "ch4_content_out": "ch4_content_out",
    "methangehalt": "ch4_content_out",
    # digestate cascade (digestate_cascade)
    "digestate_cascade": "digestate_cascade",
    "kaskade": "digestate_cascade",
    "cascade": "digestate_cascade",
    "gärrestfluss": "digestate_cascade",
    "gärrest": "digestate_cascade",
    "gaerrest": "digestate_cascade",
    "digestat": "digestate_cascade",
    "reihenfolge": "digestate_cascade",
    # gas path to the consumers (gas_routing)
    "gas_routing": "gas_routing",
    "gasweg": "gas_routing",
    "gasführung": "gas_routing",
    "gasfuehrung": "gas_routing",
    "gaspfad": "gas_routing",
    "gasverwertung": "gas_routing",
    "gassammelleitung": "gas_routing",
    "gasabnehmer": "gas_routing",
    "gas routing": "gas_routing",
    "gas path": "gas_routing",
    # substrate feed (substrate_feed)
    "substrate_feed": "substrate_feed",
    "substrat": "substrate_feed",
    "substrate": "substrate_feed",
    "feedstock": "substrate_feed",
    "zufuhr": "substrate_feed",
    "fütter": "substrate_feed",
    "fuetter": "substrate_feed",
    "futter": "substrate_feed",
    "feeding": "substrate_feed",
    "beschickung": "substrate_feed",
    "rohstoff": "substrate_feed",
    "einsatzstoff": "substrate_feed",
    "einsatzmaterial": "substrate_feed",
    "inputmaterial": "substrate_feed",
    "input material": "substrate_feed",
    # separator (type + source) -- "separator"/"sep" returns both sep.* fields
    "separator": "sep",
    "separator_type": "sep",
    "sep.source": "sep.source",
    # component existence
    "bgaa": "bgaa.exists",
    "biogasaufbereitung": "bgaa.exists",
    "chp": "chp.exists",
    "bhkw": "chp.exists",
    "blockheizkraftwerk": "chp.exists",
}


class Oracle:
    """
    Answers LLM questions from a datapoint's oracle dict.

    Usage:
        oracle = Oracle(datapoint)
        answer_text = oracle.answer(questions)  # questions = list[dict] or list[str]
    """

    def __init__(self, datapoint: dict[str, Any]) -> None:
        self.facts: dict[str, Any] = datapoint.get("oracle", {})
        self.regime: str = datapoint.get("regime", "underspecified")

    @property
    def is_underspecified(self) -> bool:
        return self.regime == "underspecified"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer(self, questions: list[Any]) -> str:
        """
        Answer a list of questions.

        ``questions`` may be:
          - list[dict]  with "field" and optionally "question" (structured JSON)
          - list[str]   free-form questions

        Returns formatted text, sent on as a user message unchanged.
        """
        if not questions:
            return self._all_facts_text()

        answered: dict[str, Any] = {}

        for q in questions:
            field = q.get("field") or q.get("question") or "" if isinstance(q, dict) else str(q)

            matches = self._match(field)
            answered.update(matches)

        if not answered:
            return self._all_facts_text()

        lines = ["Antworten auf deine Fragen:\n"]
        for k, v in sorted(answered.items()):
            lines.append(f"- {k}: {self._fmt(v)}")

        # top up with unasked oracle keys when the LLM asked for little
        missing_keys = set(self.facts) - set(answered)
        if 0 < len(missing_keys) <= 5:
            lines.append("\nWeitere relevante Informationen:")
            for k in sorted(missing_keys):
                lines.append(f"- {k}: {self._fmt(self.facts[k])}")

        lines.append("\nBitte schreibe nun den vollständigen Python-Code.")
        return "\n".join(lines)

    def answer_all(self) -> str:
        """Return every oracle fact (used by the --no-oracle mode)."""
        return self._all_facts_text()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _match(self, query: str) -> dict[str, Any]:
        """Find the oracle keys matching a field name or a question."""
        result: dict[str, Any] = {}
        q = query.strip()

        # 1) exact match
        if q in self.facts:
            return {q: self.facts[q]}

        q_lower = q.lower()

        # 2) keyword mapping -> oracle field name -> every matching key
        for kw, canon in _KEYWORD_MAP.items():
            if kw in q_lower:
                for key, val in self.facts.items():
                    if canon.lower() in key.lower():
                        result[key] = val

        # 3) component ids named explicitly (e.g. "F1", "N1", "G1", "bhkw")
        for comp_id in re.findall(r"\b([A-Z]\d+|bhkw|bgaa|sep)\b", q, re.IGNORECASE):
            prefix = comp_id.lower() + "."
            for key, val in self.facts.items():
                if key.lower().startswith(prefix):
                    result[key] = val

        # 4) every oracle key appearing verbatim in the query (case-insensitive)
        for key, val in self.facts.items():
            if key.lower() in q_lower:
                result[key] = val

        return result

    def _all_facts_text(self) -> str:
        lines = ["Alle verfügbaren Informationen zur Anlage:\n"]
        for k, v in sorted(self.facts.items()):
            lines.append(f"- {k}: {self._fmt(v)}")
        lines.append("\nBitte schreibe nun den vollständigen Python-Code.")
        return "\n".join(lines)

    @staticmethod
    def _fmt(v: Any) -> str:
        if isinstance(v, float) and v == int(v):
            return str(int(v)) if v > 1000 else f"{v:.2f}"
        return str(v)
