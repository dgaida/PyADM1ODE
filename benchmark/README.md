# PyADM1ODE LMM-Benchmark

Bewertet, ob ein LLM aus einer **Beschreibung oder Skizze** einer Biogasanlage
korrekten PyADM1ODE-Code erzeugt, der die **richtige Anlagenstruktur** baut.

## Aufbau

```text
benchmark/
  schema/    plant_datapoint.schema.json    JSON-Schema (Draft 2020-12) eines Datenpunkts
  dataset/   index.json                     Manifest aller Datenpunkte (von make_index.py erzeugt)
             BGA1/  BGA1_text_de.json       Datenpunkt: Input (Text/Bild) + Referenz-Anlage
                    BGA1_text_en.json
                    BGA1_terse_de.json
                    BGA1_sketch.json
                    BGA1_text_de_full.json
                    BGA1_text_en_full.json
                    BGA1_terse_de_full.json
                    BGA1_sketch_full.json
                    BGA1_sketch.png
                    gold.py                 Gold-Lösung: ausführbarer PyADM1ODE-Code
             BGA2/ 
             BGA3/  
             BGA4/  
             BGA5/  
             BGA6/  
             BGA7/  
             BGA8/  
             BGA9/  
             BGA10/ 
             BGA11/ 
  eval/      solve.py                       LLM-Evaluation (Einstiegspunkt, ruft LLM auf)
             oracle.py                      beantwortet LLM-Fragen aus dem oracle-Dict
             prompt.py                      baut System-Prompt + Nachrichten-Liste auf
             runner.py                      führt LLM-Code isoliert aus + bewertet
             matcher.py                     Graph-Matcher: Scoring, stdlib-only
             batch.py                       wertet alle Datenpunkte mit gold.py aus
             make_index.py                  erzeugt dataset/index.json + Viewer-Block
             validate.py                    prüft Datenpunkte gegen das Schema, stdlib-only
             _harness.py                    Subprozess-Harness (Code -> Anlagen-Dict)
             selftest.py                    Matcher-Selbsttest ohne ADM1
  viewer/    index.html                     interaktiver Datenpunkt-Viewer (offline, kein Server)
```

---

## Datensatz

**Ein Unterordner pro Anlage** — darin liegen alle Input-Varianten und die gemeinsame Gold-Lösung.

### Zwei Dateien, zwei Rollen

| Datei                 | Wofür                                                                                                                              | Genutzt von                 |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| `BGA1_text_de.json` | **Aufgabe** — Input (Text oder Bild), den das LLM bekommt; enthält zugleich die Referenz-Anlage (typisierter Graph) zum Abgleich | LLM-Prompt +`matcher.py`  |
| `gold.py`           | **Gold-Lösung** — eine bekannt korrekte Umsetzung; validiert den Harness und dient als Referenzcode                         | `runner.py`, `batch.py` |

Geprüft wird der LLM-Code **gegen die Referenz im JSON**, nicht gegen `gold.py`
direkt — `gold.py` ist die Soll-Umsetzung zum Vergleich und zur Harness-Validierung.

### Welches Feld liest wer?

| Feld                                                | Gelesen von                                   |
| --------------------------------------------------- | --------------------------------------------- |
| `id`                                                | `solve.py`, `batch.py`, Viewer              |
| `input.{modality,language,content,image_path}`      | `prompt.py` (LLM-Prompt), Viewer      |
| `regime`                                            | `solve.py` (Filter), `oracle.py`, Viewer  |
| `reference.components[].{id,type,obligation,params}`  | `matcher.py` (Struktur + Maße)              |
| `reference.components[].auto_created`               | `matcher.py` (Auto-Knoten via Topologie)    |
| `reference.components[].label`                      | Viewer                                        |
| `reference.connections[]`                           | `matcher.py` (Vollständigkeit + Erfindungen) |
| `params[].{value,obligation,accept}`                | `matcher.py` (Akzeptanzband)                |
| `oracle`                                            | `oracle.py` (Multi-Turn-Antworten) — **nur bei `underspecified`** |
| `metadata`                                          | Viewer (Kopfzeile) + Authoring                |
| `params[].{formula,rationale,note}`, `*.note`       | nur Dokumentation (Herleitung nachvollziehbar) |


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
│   SYSTEM_PROMPT: minimale API-Signaturen (überschreibbar)
│   build_messages(): Text / Bild (base64) / Hybrid → Messages-Liste
│
├── runner.py         ← Code-Ausführer + Bewerter
│   evaluate_code(): führt LLM-Code im Subprocess aus → ruft matcher.py
│
├── matcher.py        ← Graph-Matcher + Scorer
│   Vergleicht gebaute Anlage mit Referenz (Vollständigkeit / Maße / Erfindungen)
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
  │        → Scores: Vollständigkeit / Maße / Keine Erfindungen
  │
  └─[5]─ Tabelle + CSV + Code-Dateien speichern
```

---

## CLI-Verwendung

### LLM-Evaluation (solve.py)

Benötigt: `pip install -e ".[benchmark]"` (groq) und `GROQ_API_KEY` als Umgebungsvariable (Für eine andere API muss nur die Client-Sektion in solve.py angepasst werden).

```bash
# Nur fully_specified — kein Oracle nötig, einfachster Einstieg:
python benchmark/eval/solve.py --regime fully_specified

# Alle 88 Datenpunkte mit Oracle-Unterstützung:
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

Die Spalte **`questions`** (in der Tabelle `Fragen`) zählt, **wie viele Felder das
Modell beim Oracle erfragt hat** — gezählt werden verschiedene `field`-Einträge,
zweimal dasselbe Feld zählt einmal. Das ist eine reine Information neben den drei
Scores.

---

## Drei Scores

| Score | Frage | Kennzahl |
| --- | --- | --- |
| **1. Vollständigkeit** | Ist alles Nötige da? | Recall über Pflicht-Bauteile und Pflicht-Kanten |
| **2. Maße** | Stimmen die Werte? | Anteil Parameter im Akzeptanzband |
| **3. Keine Erfindungen** | Ist *nur* Bekanntes da? | Precision über Bauteile und Kanten |

```text
weggelassen -> Score 1        falscher Wert -> Score 2        hinzuerfunden -> Score 3
```

Details:

- **Pflicht** (Score 1) und **erlaubt** (Score 3) sind zwei verschiedene Mengen. Eine  
  Kante mit `obligation: missing_ask` *muss* nicht gebaut werden, gilt aber auch nicht
  als Erfindung, wenn sie gebaut wird.  
- Ein Bauteil eines Typs, den die Referenz gar nicht kennt (Separator in einer Anlage  
  ohne Separator), **deckelt Score 3 auf 50 %** — das ist die schwerste Halluzination.  
- Der Nenner von Score 1 umfasst *alle* Pflicht-Kanten. Wer einen Knoten weglässt,  
  wird seine Kanten nicht mit los.  
- Ein still erfundener, unplausibler **Wert** senkt Score 2, weil jeder Referenz-Parameter  
  gegen sein Akzeptanzband geprüft wird — unabhängig davon, ob er im Input stand.

### Wie breit ist das Akzeptanzband?

Die Breite hängt nur noch daran, **ob der Wert übernommen oder gerechnet wird** —
erkennbar am Feld `formula`:

| | Band | Warum |
| --- | --- | --- |
| ohne `formula` | **±0,1 %** | Der Wert wird übernommen. Nur Rundung darf abweichen — `40 °C` als `313.0` statt `313.15 K` geht durch, `39 °C` nicht. |
| mit `formula` | **±1 %** | Der Wert wird gerechnet, die Rundungsstelle variiert (`pi = 3.14` statt `3.14159`). |

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
