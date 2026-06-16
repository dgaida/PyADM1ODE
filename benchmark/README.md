# PyADM1ODE LMM-Benchmark

Bewertet, ob ein LLM aus einer **Beschreibung oder einem Bild** einer Biogasanlage
korrekten PyADM1ODE-Code erzeugt, der die **richtige Anlagenstruktur** baut.

## Aufbau

```text
benchmark/
  schema/    plant_datapoint.schema.json    JSON-Schema (Draft 2020-12) eines Datenpunkts
  dataset/   index.json                     Manifest aller Datenpunkte (von make_index.py erzeugt)
             BGA1/  BGA1_text_de.json       Datenpunkt: Input (Beschreibung/Bild) + Referenz-Anlage
                    BGA1_text_en.json
                    BGA1_terse_de.json
                    BGA1_sketch.json
                    BGA1_text_de_full.json
                    BGA1_text_en_full.json
                    BGA1_terse_de_full.json
                    BGA1_sketch_full.json
                    BGA1_sketch.png
                    gold.py                 Gold-Lösung: ausführbarer PyADM1ODE-Code
             BGA2/  (gleiche Struktur wie BGA1)
             BGA3/  (gleiche Struktur wie BGA1)
  eval/      solve.py                       LLM-Evaluation (Einstiegspunkt, ruft LLM auf)
             oracle.py                      beantwortet LLM-Fragen aus dem oracle-Dict
             prompt.py                      baut System-Prompt + Nachrichten-Liste auf
             runner.py                      führt LLM-Code isoliert aus + bewertet
             matcher.py                     Graph-Matcher: Scoring, stdlib-only
             batch.py                       wertet alle Datenpunkte mit gold.py aus
             make_index.py                  erzeugt dataset/index.json + Viewer-Block
             _harness.py                    Subprozess-Harness (Code -> Anlagen-Dict)
             selftest.py                    Matcher-Selbsttest ohne ADM1
  docs/      dataset_structure_overview.svg Präsentationsgrafik (Übersicht)
             dataset_structure_detail.svg   Präsentationsgrafik (Detail)
  viewer/    index.html                     interaktiver Datenpunkt-Viewer (offline, kein Server)
```

---

## Datensatz

**Ein Unterordner pro Anlage** — darin liegen alle Input-Varianten und die gemeinsame Gold-Lösung.

### Zwei Dateien, zwei Rollen

| Datei                 | Wofür                                                                                                                              | Genutzt von                 |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| `BGA1_text_de.json` | **Aufgabe** — Input (Text/Bild), den das LLM bekommt; enthält zugleich die Referenz-Anlage (typisierter Graph) zum Abgleich | LLM-Prompt +`matcher.py`  |
| `gold.py`           | **Gold-Lösung** — eine bekannt korrekte Umsetzung; validiert den Harness und dient als Referenzcode                         | `runner.py`, `batch.py` |

Geprüft wird der LLM-Code **gegen die Referenz im JSON**, nicht gegen `gold.py`
direkt — `gold.py` ist die Soll-Umsetzung zum Vergleich und zur Harness-Validierung.

### Namenskonvention

Ordner = Anlage. Mehrere **Input-Varianten** derselben Anlage liegen im selben
Ordner und teilen sich `gold.py` (gleiche Anlage, andere Beschreibung — prüft
die Robustheit des LLM):

```text
BGA1/
  BGA1_text_de.json         ausführliche Textbeschreibung (de) mit fehlenden Informationen
  BGA1_text_de_full.json    ausführliche Textbeschreibung (de) ohne fehlende Informationen
  BGA1_text_en.json         englische Variante mit fehlenden Informationen
  BGA1_text_en_full.json    englische Variante ohne fehlende Informationen
  BGA1_terse_de.json        knappe Beschreibung (de) mit fehlenden Informationen
  BGA1_terse_de_full.json   knappe Beschreibung (de) ohne fehlende Informationen
  BGA1_sketch.png           Skizze der Anlage
  BGA1_sketch.json          nur Skizze, mit fehlenden Informationen
  BGA1_sketch_full.json     Skizze + ergänzender Text, ohne fehlende Informationen
  gold.py                   gemeinsame Gold-Lösung (alle Varianten teilen sie)
```

Varianten mit dem Suffix `_full` haben `"regime": "fully_specified"` — alle Informationen
sind im Input enthalten, kein Oracle nötig. Varianten ohne `_full` haben
`"regime": "underspecified"` — das LLM muss fehlende Werte erfragen oder plausibel ergänzen.

---

## Architektur: Evaluation-Pipeline

```text
benchmark/eval/
│
├── solve.py          ← Einstiegspunkt (CLI)
│   Koordiniert alles: lädt Datenpunkte, ruft LLM auf,
│   steuert Oracle-Runden, speichert Ergebnisse als CSV
│
├── oracle.py         ← Oracle-Beantworter
│   Kennt die "echten" Werte aus dem Datenpunkt-JSON.
│   Beantwortet LLM-Fragen durch Keyword-Matching
│   gegen das oracle-Dict (T_ad, V_gas, cascade, …)
│
├── prompt.py         ← Prompt-Builder
│   SYSTEM_PROMPT: statische API-Doku für PyADM1ODE
│   build_messages(): Text / Bild (base64) / Hybrid → Messages-Liste
│
├── runner.py         ← Code-Ausführer + Bewerter
│   evaluate_code(): führt LLM-Code im Subprocess aus → ruft matcher.py
│
├── matcher.py        ← Graph-Matcher + Scorer
│   Vergleicht gebaute Anlage mit Referenz (Struktur / Masse / Lücken)
│
└── _harness.py       ← Subprocess-Isolator
    Führt Kandidaten-Code aus, serialisiert plant → JSON
```

## Ablauf je Datenpunkt

```text
solve.py
  │
  ├─[1]─ prompt.py → build_messages()
  │        Beschreibung + Bild (wenn vorhanden) + Aufgabe
  │
  ├─[2]─ LLM Turn 1 (Groq API)
  │
  ├─[3a] Code in Turn 1? ──────────────────────────────────────────┐
  │                                                                │
  └─[3b] Fragen in Turn 1? (nur underspecified + Oracle aktiv)     │
           │                                                       │
           ├─ oracle.py → answer(questions)                        │
           │   Keyword-Match gegen oracle-Dict                     │
           │   z.B. "T_ad" → F1.T_ad: 313.15 K                     │
           │                                                       │
           └─ LLM Turn 2 → Code extrahieren ───────────────────────┤
                                                                   │
  ├─[4]─ runner.evaluate_code(datapoint, code)  ←──────────────────┘
  │        ↓ subprocess (_harness.py)
  │        plant.to_dict() → matcher.evaluate()
  │        → Scores: Struktur / Masse / Lücken
  │
  └─[5]─ Tabelle + CSV + Code-Dateien speichern
```

---

## CLI-Verwendung

### LLM-Evaluation (solve.py)

Benötigt: `pip install groq` und `GROQ_API_KEY` als Umgebungsvariable (Für eine andere API muss nur die Client-Sektion in solve.py angepasst werden).

```bash
# Nur fully_specified — kein Oracle nötig, einfachster Einstieg:
python benchmark/eval/solve.py --regime fully_specified

# Alle 24 Datenpunkte mit Oracle-Unterstützung:
python benchmark/eval/solve.py

# Einzelnen Datenpunkt testen:
python benchmark/eval/solve.py --id BGA2_text_de_full

# Nur BGA2, Deutsch, mit LLM-Antworten im Terminal:
python benchmark/eval/solve.py --id BGA2 --language de --verbose

# Anderes Modell, ohne Oracle:
python benchmark/eval/solve.py --model openai/gpt-oss-120b --no-oracle

# Nur Bild-Datenpunkte (vision-fähiges Groq-Modell nötig):
python benchmark/eval/solve.py --modality image --model meta-llama/llama-4-scout-17b-16e-instruct
```

Ergebnisse landen in `benchmark/results/` als CSV + je Datenpunkt eine `.py`-Datei
mit dem generierten Code.

### Matcher direkt (ohne Code-Lauf)

```bash
python benchmark/eval/matcher.py benchmark/dataset/BGA1/BGA1_text_de.json candidate.json
```

### Einzelnen Datenpunkt mit Code ausführen und bewerten

```bash
conda run -n biogas --no-capture-output python benchmark/eval/runner.py \
    benchmark/dataset/BGA1/BGA1_text_de.json benchmark/dataset/BGA1/gold.py
```

### Alle Datenpunkte mit gold.py (Baseline-Check)

```bash
conda run -n biogas --no-capture-output python benchmark/eval/batch.py

# LLM-Ausgaben statt gold.py bewerten (sucht <id>.py im Ordner):
conda run -n biogas --no-capture-output python benchmark/eval/batch.py --candidates path/to/llm_outputs
```

`batch.py` führt den Kandidaten-Code isoliert aus, bewertet und schreibt
`benchmark/results.csv`. Varianten, die sich `gold.py` teilen, werden gecacht
(Anlage nur einmal gebaut).

### Viewer aktualisieren

```bash
python benchmark/eval/make_index.py
```

Nach dem Hinzufügen oder Ändern von Datenpunkten ausführen: aktualisiert
`dataset/index.json` und den eingebetteten Block im Viewer.

### Matcher-Selbsttest (ohne PyADM1ODE)

```bash
python benchmark/eval/selftest.py
```

---

## Drei Scores

1. **Struktur** — Bauteile (nach Typ zugeordnet, nicht nach Namen) und Verbindungen als typisierter Graph.  
2. **Maße** — simulierte Parameter (V_liq, V_gas, T_ad, P_el_nom, …) im Akzeptanzband.  
3. **Fehlende Werte** — `missing_ask`-Felder: nachgefragt ODER plausibel gefüllt statt still erfunden.  

Nur Größen, die PyADM1ODE **wirklich simuliert**, fliessen ein. Auto-Knoten
(GasStorage je Digester, Flare je CHP/BGAA) werden über die Topologie ausgerichtet.

---

## Viewer

`benchmark/viewer/index.html` zeigt Datenpunkte als interaktiven Anlagengraphen.
Knoten nach Typ (Farbe), Kanten nach Verbindungstyp (Flüssig / Gas / Wärme),
Knotenrand nach `obligation`. Mit **◀ / ▶**, Dropdown oder Pfeiltasten durch die
Datenpunkte skippen.

Live-Modus (empfohlen) — Änderungen nach Reload sichtbar:

```bash
python benchmark/eval/make_index.py
python -m http.server 8000
# Browser: http://localhost:8000/benchmark/viewer/
```

Beim Doppelklick auf die HTML-Datei (`file://`) zeigt der Viewer eine eingebettete
Kopie. Über **Dateien laden…** lassen sich beliebige Datenpunkt-JSONs manuell öffnen.

---

## `response.json` (optional, für den Lücken-Score)

Strukturierte LLM-Antwort neben dem Code (wird von `batch.py` automatisch gesucht):

```json
{
  "open_questions": [{"field": "chp.P_el_nom"}, {"field": "sep.source"}],
  "assumptions":    [{"field": "F1.T_ad", "value": 313.15}]
}
```

Fragt das Modell nach einem `missing_ask`-Feld oder füllt es plausibel im Band,
zählt das als korrekt. Stilles Erfinden eines unplausiblen Werts ist der schwerste Fehler.

---
