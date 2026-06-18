# Datensatz nutzen

Diese Seite richtet sich an **Nutzer des Datensatzes**, etwa um ein eigenes
Modell zu evaluieren oder zu trainieren. Sie erklärt, **was im Datensatz steckt**,
**was man ausführt** und **wie man ein eigenes Modell anbindet**.

## Datensatz beschaffen

Der Datensatz ist Teil des Repositorys unter `benchmark/`. Es ist kein separater
Download nötig:

```bash
git clone https://github.com/dgaida/PyADM1ODE.git
cd PyADM1ODE
```

Die Umgebung mit PyADM1ODE wird für das Ausführen von Code benötigt (Conda-Umgebung
`biogas` empfohlen). Reine Bewertung von Graphen (`matcher.py`) und der Viewer
kommen ohne PyADM1ODE aus.

## Was steckt drin?

```text
benchmark/
  schema/    plant_datapoint.schema.json   JSON-Schema (Draft 2020-12) eines Datenpunkts
  dataset/   index.json                    Manifest aller Datenpunkte
             BGA1/ BGA2/ ...               je eine Anlage mit Input-Varianten + gold.py
  eval/      solve.py runner.py matcher.py batch.py make_index.py …
  viewer/    index.html                    interaktiver Datenpunkt-Viewer
```

**Ein Ordner pro Anlage.** Darin liegen alle Input-Varianten und eine gemeinsame
Gold-Lösung `gold.py` (korrekter PyADM1ODE-Code).

| Datei | Rolle |
| ----- | ----- |
| `BGAx_<variante>.json` | **Aufgabe**: Input (Text/Bild) für das Modell **und** die Referenz-Anlage (typisierter Graph) zum Abgleich |
| `gold.py` | **Gold-Lösung**: bekannt korrekte Umsetzung, validiert den Harness und dient als Referenzcode |

### Varianten und Regime

Jede Anlage existiert in mehreren Varianten. Das Suffix `_full`
markiert die Vollständigkeit:

- **`fully_specified`** (`_full`): alle Angaben im Input – kein Oracle nötig.  
- **`underspecified`** (ohne `_full`): Werte fehlen – das Modell muss nachfragen  
  oder plausibel ergänzen.

| Achse | Werte | Verteilung (24) |
| ----- | ----- | --------------- |
| Anlage | BGA1, BGA2, BGA3 | je 8 |
| Vollständigkeit | fully_specified / underspecified | 12 / 12 |
| Modalität | text / image / hybrid | 18 / 3 / 3 |
| Sprache | de / en | 18 / 6 |

Den genauen Aufbau eines einzelnen Datenpunkts beschreibt
[Ein Datenpunkt im Detail](datenpunkt.md). Das maßgebliche Format steht im
JSON-Schema unter `benchmark/schema/plant_datapoint.schema.json`.

## Was man ausführt

| Ziel | Befehl |
| ---- | ------ |
| Matcher-Selbsttest (ohne PyADM1ODE) | `python benchmark/eval/selftest.py` |
| Baseline – alle Datenpunkte mit `gold.py` | `python benchmark/eval/batch.py` |
| Ein Datenpunkt: Code ausführen + bewerten | `python benchmark/eval/runner.py <datapoint.json> <code.py>` |
| Nur Graph bewerten (ohne Code-Lauf) | `python benchmark/eval/matcher.py <datapoint.json> <candidate.json>` |
| LLM-Evaluation | `python benchmark/eval/solve.py [Filter]` |
| Viewer/Index aktualisieren | `python benchmark/eval/make_index.py` |

Beispiele:

```bash
# 1) Schneller Funktionscheck des Matchers (kein PyADM1ODE nötig)
python benchmark/eval/selftest.py

# 2) Baseline: alle 24 Datenpunkte gegen ihre gold.py bewerten
python benchmark/eval/batch.py

# 3) Einen einzelnen Datenpunkt mit der Gold-Lösung ausführen und bewerten
python benchmark/eval/runner.py \
    benchmark/dataset/BGA1/BGA1_text_de.json benchmark/dataset/BGA1/gold.py
```

`batch.py` schreibt `benchmark/results.csv`. `solve.py` legt CSV + den generierten
Code je Datenpunkt unter `benchmark/results/` ab.

## Eigenes Modell anbinden oder trainieren

Die **Eingabe** liegt im Datenpunkt unter `input` (Felder `modality`, `language`,
`content`, ggf. `image_path`). Das **Ziel** ist lauffähiger PyADM1ODE-Code, der die
Anlage baut. Die **Referenz** zum Abgleich steht unter `reference` (Bauteile +
Verbindungen als typisierter Graph). `gold.py` zeigt je Anlage eine korrekte
Umsetzung.

Es gibt zwei Wege, ein eigenes Modell zu bewerten:

### A) Offline Code erzeugen, dann bewerten

Lasse dein Modell pro Datenpunkt einen Python-Code erzeugen und speichere ihn
als `<datapoint-id>.py` in einem Ordner. Danach:

```bash
python benchmark/eval/batch.py --candidates pfad/zu/modell_ausgaben
```

`batch.py` führt jeden Kandidaten isoliert aus und bewertet ihn. Liegt neben
`<id>.py` eine `<id>.response.json`, fließt sie in den Lücken-Score ein.

### B) Direkt über solve.py (API)

`solve.py` ist ein vollständiger Lauf inkl. Oracle-Runden. Standardmäßig nutzt
es die Groq-API. Für ein eigenes Modell muss **nur die Client-Sektion** in `solve.py`
angepasst werden, der restliche Ablauf (Prompt, Oracle, Bewertung) bleibt gleich.

 ```bash
pip install groq          # bzw. eigene Client-Bibliothek
export GROQ_API_KEY=...    # bzw. eigener API-Key
python benchmark/eval/solve.py --regime fully_specified   # einfachster Einstieg
```

### `response.json` (für den Lücken-Score)

Für unterspezifizierte Datenpunkte zählt, ob ein fehlendes Feld **erfragt** oder
plausibel **ergänzt** wurde. Eine strukturierte Antwort neben dem Code macht das
explizit:

```json
{
  "open_questions": [{"field": "chp.P_el_nom"}, {"field": "sep.source"}],
  "assumptions":    [{"field": "F1.T_ad", "value": 313.15}]
}
```

Fragt das Modell nach einem `missing_ask`-Feld oder füllt es plausibel im Band,
zählt das als korrekt. Stilles Erfinden eines unplausiblen Werts ist der schwerste
Fehler.

## Programmatischer Zugriff

Das Manifest `dataset/index.json` listet alle Datenpunkte mit `id`, `path`,
`language`, `modality`, `regime`:

```python
import json
from pathlib import Path

root = Path("benchmark/dataset")
index = json.loads((root / "index.json").read_text(encoding="utf-8"))

for entry in index["datapoints"]:
    dp = json.loads((root / entry["path"]).read_text(encoding="utf-8"))
    task = dp["input"]["content"]        # Beschreibung für das Modell
    reference = dp["reference"]          # Soll-Anlage (Bauteile + Verbindungen)
    # ... eigenes Modell aufrufen, Code erzeugen, bewerten ...
```

## Bewertung

Bewertet werden drei Scores:

1. **Struktur** – Bauteile (nach Typ zugeordnet, nicht nach Namen) und Verbindungen.  
2. **Maße** – simulierte Parameter (`V_liq`, `V_gas`, `T_ad`, `P_el_nom`, …) im Akzeptanzband.  
3. **Fehlende Werte** – nachgefragt oder plausibel gefüllt statt still erfunden.  

Wie daraus die Endbewertung entsteht, erklärt [Bewertung & Ablauf](bewertung.md).
Den Datensatz visuell erkunden kannst du im [Viewer](viewer.md).
