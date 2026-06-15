# benchmark/eval/oracle.py
"""
Oracle: beantwortet LLM-Fragen aus dem oracle-Dict eines Datenpunkts.

Der Oracle simuliert einen menschlichen Experten, der fehlende Informationen
(z.B. Betriebstemperatur, Gasspeichervolumen) auf Nachfrage liefert.
Grundlage ist das "oracle"-Feld im Datenpunkt-JSON.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

# Kanonische Feldnamen, die das LLM typischerweise nennt, gemappt auf oracle-Keys
_KEYWORD_MAP = {
    "t_ad": "T_ad",
    "betriebstemperatur": "T_ad",
    "temperatur": "T_ad",
    "temperature": "T_ad",
    "v_gas": "V_gas",
    "gasspeicher": "V_gas",
    "gas_storage": "V_gas",
    "gas storage": "V_gas",
    "fill_fraction": "fill_fraction",
    "füllgrad": "fill_fraction",
    "fill level": "fill_fraction",
    "fillgrade": "fill_fraction",
    "p_el_nom": "P_el_nom",
    "elektrische leistung": "P_el_nom",
    "nennleistung": "P_el_nom",
    "rated power": "P_el_nom",
    "eta_el": "eta_el",
    "elektrischer wirkungsgrad": "eta_el",
    "electrical efficiency": "eta_el",
    "eta_th": "eta_th",
    "thermischer wirkungsgrad": "eta_th",
    "thermal efficiency": "eta_th",
    "capacity_m3h": "capacity_m3h",
    "kapazität": "capacity_m3h",
    "ch4_recovery": "ch4_recovery",
    "methanausbeute": "ch4_recovery",
    "methane recovery": "ch4_recovery",
    "ch4_content_out": "ch4_content_out",
    "methangehalt": "ch4_content_out",
    "digestate_cascade": "digestate_cascade",
    "kaskade": "digestate_cascade",
    "cascade": "digestate_cascade",
    "gärrestfluss": "digestate_cascade",
    "separator_type": "separator_type",
    "separator": "separator_type",
    "sep.source": "sep.source",
    "bgaa": "bgaa.exists",
    "biogasaufbereitung": "bgaa.exists",
    "chp": "chp.exists",
    "bhkw": "chp.exists",
}


class Oracle:
    """
    Beantwortet LLM-Fragen anhand des oracle-Dicts eines Datenpunkts.

    Verwendung:
        oracle = Oracle(datapoint)
        answer_text = oracle.answer(questions)  # questions = list[dict] oder list[str]
    """

    def __init__(self, datapoint: Dict[str, Any]) -> None:
        self.facts: Dict[str, Any] = datapoint.get("oracle", {})
        self.regime: str = datapoint.get("regime", "underspecified")

    @property
    def is_underspecified(self) -> bool:
        return self.regime == "underspecified"

    # ------------------------------------------------------------------
    # Oeffentliche API
    # ------------------------------------------------------------------

    def answer(self, questions: List[Any]) -> str:
        """
        Beantwortet eine Liste von Fragen.

        questions kann sein:
          - list[dict]  mit "field" und optional "question" (strukturiertes JSON)
          - list[str]   freie Fragen

        Gibt formatierten Text zurück, der direkt als User-Nachricht gesendet wird.
        """
        if not questions:
            return self._all_facts_text()

        answered: Dict[str, Any] = {}

        for q in questions:
            if isinstance(q, dict):
                field = q.get("field") or q.get("question") or ""
            else:
                field = str(q)

            matches = self._match(field)
            answered.update(matches)

        if not answered:
            return self._all_facts_text()

        lines = ["Antworten auf deine Fragen:\n"]
        for k, v in sorted(answered.items()):
            lines.append(f"- {k}: {self._fmt(v)}")

        # Fehlende oracle-Keys ergänzen, wenn der LLM wenig gefragt hat
        missing_keys = set(self.facts) - set(answered)
        if 0 < len(missing_keys) <= 5:
            lines.append("\nWeitere relevante Informationen:")
            for k in sorted(missing_keys):
                lines.append(f"- {k}: {self._fmt(self.facts[k])}")

        lines.append("\nBitte schreibe nun den vollständigen Python-Code.")
        return "\n".join(lines)

    def answer_all(self) -> str:
        """Gibt alle oracle-Fakten zurück (für --no-oracle-Modus)."""
        return self._all_facts_text()

    # ------------------------------------------------------------------
    # Intern
    # ------------------------------------------------------------------

    def _match(self, query: str) -> Dict[str, Any]:
        """Sucht passende oracle-Keys für eine Feldbezeichnung oder Frage."""
        result: Dict[str, Any] = {}
        q = query.strip()

        # 1) Exakter Treffer
        if q in self.facts:
            return {q: self.facts[q]}

        q_lower = q.lower()

        # 2) Keyword-Mapping -> oracle-Feldname -> alle passenden Keys
        for kw, canon in _KEYWORD_MAP.items():
            if kw in q_lower:
                for key, val in self.facts.items():
                    if canon.lower() in key.lower():
                        result[key] = val

        # 3) Explizit genannte Komponenten-IDs (z.B. "F1", "N1", "G1", "bhkw")
        for comp_id in re.findall(r"\b([A-Z]\d+|bhkw|bgaa|sep)\b", q, re.IGNORECASE):
            prefix = comp_id.lower() + "."
            for key, val in self.facts.items():
                if key.lower().startswith(prefix):
                    result[key] = val

        # 4) Alle oracle-Keys, die direkt im Query auftauchen (case-insensitive)
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
