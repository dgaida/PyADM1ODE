# benchmark/eval/prompt.py
"""
Prompt builder for the PyADM1ODE benchmark.
"""

from __future__ import annotations

import base64
import os
from typing import Any


def extract_pdf_text(pdf_path: str) -> str:
    """
    Read a PDF's text layer in full, page by page.

    Raises
    ------
    ImportError
        If ``pypdf`` is missing.
    ValueError
        If the PDF has no text layer (a pure scan, say) -- the datapoint is then
        unsolvable and the failure should surface early.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError("PDF-Datenpunkte brauchen 'pypdf'. Installieren mit: pip install pypdf") from exc

    parts: list[str] = []
    for i, page in enumerate(PdfReader(pdf_path).pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            parts.append(f"--- Seite {i} ---\n{text}")
    if not parts:
        raise ValueError(
            f"{os.path.basename(pdf_path)} hat keine Textebene (vermutlich ein Scan). "
            "Gescannte PDFs werden derzeit nicht unterstützt."
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# System-Prompt
# ---------------------------------------------------------------------------
#
# MINIMAL System Prompt as example.

SYSTEM_PROMPT = """Schreibe Python-Code, der die beschriebene Biogasanlage mit dem Paket
PyADM1ODE aufbaut.

## API

```python
from pyadm1 import BiogasPlant, Feedstock
from pyadm1.components.biological.separator import Separator
from pyadm1.configurator.plant_configurator import PlantConfigurator

# <substrate_name>: maize_silage_milk_ripeness, cattle_manure, swine_manure,
# corn_cob_mix, grass_silage, green_rye_silage, cereal_gps_silage, onion_waste,
# cattle_manure_solid, chicken_manure_dry, wheat_whole_plant_silage,
# maize_silage_gummersbach, swine_manure_gummersbach
feedstock = Feedstock(["<substrate_name>", ...], feeding_freq=24, total_simtime=30)
plant = BiogasPlant("ANLAGE")
cfg = PlantConfigurator(plant, feedstock)

# Tanks (primary digester, post-digester, digestate store)
cfg.add_digester("<id>", V_liq=..., V_gas=..., T_ad=..., name="...")

# Gas utilisation -- both create a flare "<id>_flare" automatically
cfg.add_chp("<id>", P_el_nom=..., eta_el=..., eta_th=..., name="...")
cfg.add_bgaa("<id>", capacity_m3h=..., ch4_recovery=..., ch4_content_out=..., name="...")

# Separator (separator_type: "screw_press", "decanter", "belt_press",
# "vibrating_screen")
plant.add_component(Separator("<id>", separator_type="screw_press", name="..."))

# Connections
cfg.connect("<from>", "<to>", "liquid")            # digestate, also tank -> separator
cfg.connect("<from>", "<to>", "liquid", split_fraction=...)   # partial stream
cfg.auto_connect_digester_to_chp("<tank>", "<chp>")            # gas path to the CHP
cfg.auto_connect_digester_to_bgaa("<tank>", "<bgaa>")          # gas path to the upgrading unit

plant.initialize()
```

## Regeln
- Antworte mit genau einem ```python-Block und keinem Text daneben.
- Die fertige Anlage muss in der Variablen `plant` stehen.
"""

# ---------------------------------------------------------------------------
# Instructions per datapoint variant
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
# Public functions
# ---------------------------------------------------------------------------


def build_messages(
    datapoint: dict[str, Any],
    dataset_dir: str,
    *,
    allow_questions: bool = True,
) -> list[dict[str, Any]]:
    """
    Build the initial message list (user turn) in the OpenAI/Groq format.

    Parameters
    ----------
    datapoint    : datapoint dict (from JSON)
    dataset_dir  : directory holding the datapoint file
                   (needed to resolve image and document paths)
    allow_questions : True  -> the LLM may ask questions (underspecified)
                      False -> the LLM writes code straight away
    """
    inp = datapoint.get("input", {})
    modality = inp.get("modality", "text")
    content_parts: list[Any] = []

    # ---- load the image (image / hybrid) ----
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
        b64 = base64.standard_b64encode(raw).decode()
        content_parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{b64}"},
            }
        )

    # ---- load the PDF (pdf) ----
    pdf_text = ""
    if modality == "pdf":
        doc_path = os.path.join(dataset_dir, inp.get("document_path", ""))
        if not inp.get("document_path") or not os.path.exists(doc_path):
            raise FileNotFoundError(f"PDF nicht gefunden: {doc_path}")
        pdf_text = extract_pdf_text(doc_path)

    # ---- text content ----
    text_blocks: list[str] = []
    if modality == "image":
        text_blocks.append("Analysiere die obige Anlagenskizze.")
    elif modality == "hybrid":
        supp = inp.get("content", "")
        if supp:
            text_blocks.append(f"Anlagenskizze (siehe Bild oben).\n\nErgänzende Informationen:\n{supp}")
    elif modality == "pdf":
        name = os.path.basename(inp.get("document_path", "Dokument"))
        text_blocks.append(
            f"**Beigefügtes Dokument ({name}):**\n\n{pdf_text}\n\n"
            "Das Dokument ist ein reales Anlagendokument — es enthält neben den "
            "technischen Angaben auch Text, der für die Modellierung irrelevant ist."
        )
        supp = inp.get("content", "")
        if supp:
            text_blocks.append(f"**Ergänzende Informationen:**\n\n{supp}")
    else:  # text
        desc = inp.get("content", "")
        if desc:
            text_blocks.append(f"**Anlagenbeschreibung:**\n\n{desc}")

    # ---- append the task instruction ----
    text_blocks.append(_QUESTION_INSTR if allow_questions else _CODE_ONLY_INSTR)
    content_parts.append({"type": "text", "text": "\n".join(text_blocks)})

    return [{"role": "user", "content": content_parts}]


def add_oracle_answers(
    messages: list[dict[str, Any]],
    answer_text: str,
) -> list[dict[str, Any]]:
    """
    Append the oracle answers to the message list as a user turn.
    The preceding assistant response must already be in the list.
    """
    messages.append(
        {
            "role": "user",
            "content": [{"type": "text", "text": answer_text}],
        }
    )
    return messages


def append_assistant(
    messages: list[dict[str, Any]],
    text: str,
) -> list[dict[str, Any]]:
    """Append an assistant message to the history."""
    messages.append({"role": "assistant", "content": text})
    return messages
