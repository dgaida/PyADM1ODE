# benchmark/eval/prompt.py
"""
Prompt-Builder für den PyADM1ODE-Benchmark.

Baut die Anthropic-Messages-Liste auf, die an die API gesendet wird.
Unterstützt Text-, Bild- und Hybrid-Datenpunkte.
"""

from __future__ import annotations

import base64
import os
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# System-Prompt (statisch, enthält vollständige PyADM1ODE API-Referenz)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
Du bist ein Experte für Biogasanlagen-Modellierung mit dem Python-Paket PyADM1ODE.
Deine Aufgabe: Schreibe lauffähigen Python-Code, der die beschriebene Biogasanlage aufbaut.

## PyADM1ODE API-Kurzreferenz

```python
from pyadm1 import BiogasPlant, Feedstock
from pyadm1.configurator.plant_configurator import PlantConfigurator

# Substrate werden NICHT bewertet — uebernimm diese Zeile UNVERAENDERT,
# egal welche Substrate in der Beschreibung genannt werden:
feedstock = Feedstock(["cattle_manure"], feeding_freq=24, total_simtime=30)
plant = BiogasPlant("ANLAGE")          # Variable MUSS "plant" heissen
cfg = PlantConfigurator(plant, feedstock)

# Fermenter / Nachgärer / Gärrestlager
cfg.add_digester("F1", V_liq=3325, V_gas=500, T_ad=313.15, name="Fermenter 1")

# BHKW (CHP) — erstellt automatisch <id>_flare (Notfackel)
cfg.add_chp("bhkw", P_el_nom=500.0, eta_el=0.40, eta_th=0.45, name="BHKW 500 kW")

# Biogasaufbereitung (BGAA) — erstellt automatisch <id>_flare
cfg.add_bgaa("bgaa", capacity_m3h=500.0, ch4_recovery=0.98, ch4_content_out=0.97,
             name="Biogasaufbereitung")

# Flüssigverbindungen (Gärrest-Kaskade)
cfg.connect("F1", "N1", "liquid")
cfg.connect("N1", "G1", "liquid")

# Gasverbindungen: Digester → GasStorage → Abnehmer (BHKW oder BGAA)
cfg.auto_connect_digester_to_chp("F1", "bhkw")   # BHKW-Anlage
cfg.auto_connect_digester_to_bgaa("F1", "bgaa")  # BGAA-Anlage

# Separator (optional — nur wenn explizit vorhanden)
from pyadm1.components.biological.separator import Separator
plant.add_component(Separator("sep", separator_type="screw_press", name="Separator"))
cfg.connect("N1", "sep", "liquid")

plant.initialize()
```

## Typische Standardwerte (wenn nicht angegeben)
- Füllgrad: 0.90  →  V_liq = π/4 × D² × 0.90 × H_wall
- T_ad mesophil: 313.15 K (40 °C)
- T_ad unbeheizt (Gärrestlager): 293.15 K (20 °C)
- BHKW-Wirkungsgrade: eta_el = 0.40, eta_th = 0.45

## Ausgaberegeln
- Code ausschließlich in einem ```python … ``` Block
- Variable muss `plant` heißen
- Bewertet wird ausschließlich die **Anlagenstruktur** (Bauteile, Verbindungen,
  Maße). Substrate/Feedstock zählen nicht — nutze die Feedstock-Zeile oben
  unverändert und ignoriere in der Beschreibung genannte Substrate.
- Maßangaben wie „6 × 23 m" meinen Höhe × Durchmesser (H × D), sofern nicht
  ausdrücklich anders bezeichnet.
- Keine Erklärungen außerhalb des Code-Blocks
"""

# ---------------------------------------------------------------------------
# Anweisungen für Datenpunkt-Varianten
# ---------------------------------------------------------------------------

_QUESTION_INSTR = """
---
**Aufgabe:**
Prüfe, ob alle für die Modellierung nötigen Informationen vorhanden sind
(Betriebstemperatur, Gasspeichervolumen, Wirkungsgrade usw.).

Falls Informationen fehlen, liste deine Fragen ZUERST in einem JSON-Block:
```json
{
  "open_questions": [
    {"field": "F1.T_ad",  "question": "Wie hoch ist die Betriebstemperatur?"},
    {"field": "F1.V_gas", "question": "Wie groß ist das Gasspeichervolumen?"}
  ]
}
```
Danach — oder wenn keine Fragen nötig sind — schreibe direkt den Python-Code.
"""

_CODE_ONLY_INSTR = """
---
**Aufgabe:** Alle Informationen sind vollständig angegeben.
Schreibe den Python-Code für diese Anlage.
"""

# ---------------------------------------------------------------------------
# Oeffentliche Funktionen
# ---------------------------------------------------------------------------


def build_messages(
    datapoint: Dict[str, Any],
    dataset_dir: str,
    *,
    allow_questions: bool = True,
) -> List[Dict[str, Any]]:
    """
    Baut die initiale Anthropic-Nachrichten-Liste (User-Turn) auf.

    Parameters
    ----------
    datapoint    : Datenpunkt-Dict (aus JSON)
    dataset_dir  : Verzeichnis, in dem die Datenpunkt-Datei liegt
                   (wird für Bildpfade benötigt)
    allow_questions : True  -> LLM darf Fragen stellen (underspecified)
                      False -> LLM schreibt sofort Code
    """
    inp = datapoint.get("input", {})
    modality = inp.get("modality", "text")
    content_parts: List[Any] = []

    # ---- Bild laden (image / hybrid) ----
    if modality in ("image", "hybrid"):
        image_path = os.path.join(dataset_dir, inp.get("image_path", ""))
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Bild nicht gefunden: {image_path}")
        with open(image_path, "rb") as fh:
            raw = fh.read()
        ext = os.path.splitext(image_path)[1].lower().lstrip(".")
        media_type = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "gif": "image/gif",
            "webp": "image/webp",
        }.get(ext, "image/png")
        content_parts.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64.standard_b64encode(raw).decode(),
                },
            }
        )

    # ---- Textinhalt ----
    text_blocks: List[str] = []
    if modality == "image":
        text_blocks.append("Analysiere die obige Anlagenskizze.")
    elif modality == "hybrid":
        supp = inp.get("content", "")
        if supp:
            text_blocks.append(f"Anlagenskizze (siehe Bild oben).\n\nErgänzende Informationen:\n{supp}")
    else:  # text
        desc = inp.get("content", "")
        if desc:
            text_blocks.append(f"**Anlagenbeschreibung:**\n\n{desc}")

    # ---- Aufgaben-Anweisung anhängen ----
    text_blocks.append(_QUESTION_INSTR if allow_questions else _CODE_ONLY_INSTR)
    content_parts.append({"type": "text", "text": "\n".join(text_blocks)})

    return [{"role": "user", "content": content_parts}]


def add_oracle_answers(
    messages: List[Dict[str, Any]],
    answer_text: str,
) -> List[Dict[str, Any]]:
    """
    Fügt Oracle-Antworten als User-Turn zur Nachrichten-Liste hinzu.
    Die vorherige Assistenten-Antwort muss bereits enthalten sein.
    """
    messages.append(
        {
            "role": "user",
            "content": [{"type": "text", "text": answer_text}],
        }
    )
    return messages


def append_assistant(
    messages: List[Dict[str, Any]],
    text: str,
) -> List[Dict[str, Any]]:
    """Fügt eine Assistenten-Nachricht zur History hinzu."""
    messages.append({"role": "assistant", "content": text})
    return messages
