# PyADM1ODE LMM-Benchmark

Bewertet, ob ein LLM aus einer **Beschreibung, Skizze oder einem Dokument** einer Biogasanlage
korrekten PyADM1ODE-Code erzeugt, der die **richtige Anlagenstruktur** baut.

## Aufbau

```text
benchmark/
  schema/    plant_datapoint.schema.json    JSON-Schema (Draft 2020-12) eines Datenpunkts
  dataset/   index.json                     Manifest aller Datenpunkte (von make_index.py erzeugt)
             BGA1/  BGA1_text_de.json       Datenpunkt: Input (Text/Bild/PDF) + Referenz-Anlage
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
             validate.py                    prüft Datenpunkte gegen das Schema, stdlib-only
             _harness.py                    Subprozess-Harness (Code -> Anlagen-Dict)
             selftest.py                    Matcher-Selbsttest ohne ADM1
  viewer/    index.html                     interaktiver Datenpunkt-Viewer (offline, kein Server)
```

Die Präsentationsgrafiken liegen unter `docs/assets/` (`dataset_structure_detail.svg`
und `_en.svg`, `dataset_structure_overview.svg`) — dort, wo MkDocs sie ausliefern
kann; die Benchmark-Übersichtsseite bindet sie ein.

---

## Datensatz

**Ein Unterordner pro Anlage** — darin liegen alle Input-Varianten und die gemeinsame Gold-Lösung.

### Zwei Dateien, zwei Rollen

| Datei                 | Wofür                                                                                                                              | Genutzt von                 |
| --------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | --------------------------- |
| `BGA1_text_de.json` | **Aufgabe** — Input (Text, Bild oder PDF), den das LLM bekommt; enthält zugleich die Referenz-Anlage (typisierter Graph) zum Abgleich | LLM-Prompt +`matcher.py`  |
| `gold.py`           | **Gold-Lösung** — eine bekannt korrekte Umsetzung; validiert den Harness und dient als Referenzcode                         | `runner.py`, `batch.py` |

Geprüft wird der LLM-Code **gegen die Referenz im JSON**, nicht gegen `gold.py`
direkt — `gold.py` ist die Soll-Umsetzung zum Vergleich und zur Harness-Validierung.

### Welches Feld liest wer?

Der Datenpunkt enthält **nur Felder, die ein Konsument auch wirklich liest**. Wer
ein Feld ergänzt, ohne es auszuwerten, macht die Referenz nur schwerer wartbar —
das Schema erzwingt das über `additionalProperties: false`.

| Feld                                                | Gelesen von                                   |
| --------------------------------------------------- | --------------------------------------------- |
| `id`                                                | `solve.py`, `batch.py`, Viewer              |
| `input.{modality,language,content,image_path,document_path}` | `prompt.py` (LLM-Prompt), Viewer      |
| `regime`                                            | `solve.py` (Filter), `oracle.py`, Viewer  |
| `reference.components[].{id,type,obligation,params}`  | `matcher.py` (Struktur + Maße)              |
| `reference.components[].auto_created`               | `matcher.py` (Auto-Knoten via Topologie)    |
| `reference.components[].label`                      | Viewer                                        |
| `reference.connections[]`                           | `matcher.py` (Vollständigkeit + Erfindungen) |
| `params[].{value,obligation,accept}`                | `matcher.py` (Akzeptanzband)                |
| `oracle`                                            | `oracle.py` (Multi-Turn-Antworten), Viewer  |
| `metadata`                                          | Viewer (Kopfzeile) + Authoring                |
| `params[].{formula,rationale,note}`, `*.note`       | nur Dokumentation (Herleitung nachvollziehbar) |

Bewusst **nicht** im Datenpunkt, weil aus `reference` ableitbar oder unbewertet:

- Zwischenwerte der Herleitung (`D`, `H_wall`, `fill_fraction`) — sie stehen im
  `formula`-String des abgeleiteten `V_liq`, z. B. `pi/4*28^2*6 * 0.90 (Fuellgrad)`.
  In `params` gehört nur, was PyADM1ODE auch serialisiert.
- Anzahl Digester, Topologie, „hat CHP" — steht im Graphen selbst.
- `must_not_invent` — der Erfindungs-Score prüft direkt gegen die Typen in
  `reference`; eine Prosaliste „erfinde keinen Separator" wäre nur eine zweite,
  driftende Quelle derselben Information.
- Die Liste der verworfenen Skizzen-Elemente — das ist keine Eigenschaft eines
  Datenpunkts, sondern die Modellierungsregel unten, einmal formuliert.

### Was nicht in die Referenz kommt — und warum

Beschreibungen und Skizzen enthalten regelmäßig mehr, als PyADM1ODE simuliert. In
die Referenz kommt **nur, was das Paket wirklich rechnet**. Nach dieser Regel
entfallen:

| Was | Warum |
| --- | --- |
| Feststoffdosierer, Vorgrube, Vorlagebehälter, Pumpen | Substratseite — der Zulauf läuft über `Feedstock`/`Q_substrates`, nicht über eine simulierte Kante |
| Kondensat- und Sickerwasserschacht | Nebeneinrichtungen ohne ADM1-Reaktion |
| Rührwerke (Paddel-, Tauchmotor-, Langachs-) | nicht simuliert; nur ihr Behälter zählt |
| Kuppeldach | geht in `V_gas` des Behälters auf, ist kein eigener Knoten |

Der System-Prompt sagt dem Modell dasselbe (siehe `prompt.py`), damit es die
Substratseite gar nicht erst zu modellieren versucht. Wer eine Skizze mit mehr
Objekten als Referenzknoten vor sich hat, findet hier die Erklärung — sie steht
einmal zentral statt in jedem der 24 Datenpunkte.

### PDF-Datenpunkte

Neben Text und Skizze kann ein Datenpunkt ein **echtes Anlagendokument** tragen —
typischerweise ein Angebotsschreiben. Das ist der realistischste Fall: die
Anlagenstruktur steckt zwischen Positionsnummern, Preisen und Zahlungsbedingungen
und muss erst herausgelesen werden.

```json
{
  "id": "BGA4_pdf_de",
  "input": {
    "modality": "pdf",
    "language": "de",
    "document_path": "angebot_2026-0473.pdf",
    "content": "optionale Ergänzung, falls das Dokument etwas offen lässt"
  }
}
```

Das PDF liegt im Datenpunkt-Ordner neben der JSON-Datei. `prompt.py` extrahiert die
**Textebene** und schickt sie als Text mit — das funktioniert bei jedem Anbieter,
auch ohne Vision-Modell. Dafür wird `pypdf` gebraucht:

```bash
pip install -e ".[benchmark]"     # groq + pypdf
```

Das Dokument geht **vollständig** in den Prompt — es gibt bewusst keinen Seiten-
oder Zeichendeckel. Kürzen würde die Aufgabe stillschweigend verändern: steht der
technische Anhang auf Seite 22, wäre der Datenpunkt danach unlösbar, und der
Benchmark würde die Kürzung messen statt das Modell. Was ins Dokument gehört,
entscheidet der Autor beim Anlegen, nicht der Prompt-Builder.

Eine Grenze gibt es doch, und sie meldet sich laut: ein **gescanntes PDF ohne
Textebene** wird abgelehnt, statt eine leere Aufgabe zu erzeugen. Für solche Scans
bräuchte es zusätzlich eine Rasterung nach Bild.

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
`"regime": "underspecified"` — die fehlenden Werte muss das LLM beim Oracle **erfragen**.
Raten hilft nicht: die Akzeptanzbänder liegen bei ±0,1 % (übernommene Werte) bzw. ±1 %
(gerechnete Werte).

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

Benötigt: `pip install -e ".[benchmark]"` (groq + pypdf) und `GROQ_API_KEY` als Umgebungsvariable (Für eine andere API muss nur die Client-Sektion in solve.py angepasst werden).

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

### Datenpunkte gegen das Schema prüfen

```bash
python benchmark/eval/validate.py
```

Läuft automatisch als Teil von `make_index.py` und meldet unbekannte Felder,
fehlende Pflichtfelder und unzulässige Enum-Werte — ohne externe Abhängigkeit.
Das Schema steht überall auf `additionalProperties: false`; erst diese Prüfung
macht daraus einen Vertrag statt einer Absichtserklärung.

---

## Drei Scores

Die drei Achsen sind bewusst **disjunkt**, damit jeder Fehler genau einmal zählt:

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

### Kein Raten: gegeben oder erfragbar

Jede simulierte Größe steht **entweder im Input, oder das Oracle nennt sie auf
Rückfrage**. Einen dritten Fall — „das Modell muss eine ungenannte Annahme treffen" —
gibt es nicht. Deshalb kennt `paramObligation` nur drei Werte:

| `obligation` | Bedeutung |
| --- | --- |
| `given` | Der Wert steht im Input. |
| `derivable` | Der Wert ist aus Angaben im Input berechenbar (`formula` zeigt wie). |
| `missing_ask` | Der Wert steht nicht im Input; das Oracle nennt ihn exakt. |

Ein Datenpunkt, der eine ungenannte Annahme verlangt, ist nicht fair bewertbar: Man
müsste das Band so weit aufziehen, dass es nichts mehr aussagt.

### Wie breit ist das Akzeptanzband?

Die Breite hängt nur noch daran, **ob der Wert übernommen oder gerechnet wird** —
erkennbar am Feld `formula`:

| | Band | Warum |
| --- | --- | --- |
| ohne `formula` | **±0,1 %** | Der Wert wird übernommen. Nur Rundung darf abweichen — `40 °C` als `313.0` statt `313.15 K` geht durch, `39 °C` nicht. |
| mit `formula` | **±1 %** | Der Wert wird gerechnet, die Rundungsstelle variiert (`pi = 3.14` statt `3.14159`). |

Fehlt `accept` ganz, wird exakt verglichen — so bei kategorialen Werten wie
`separator_type`: wer fragt, bekommt den exakten Typ, eine Auswahlliste braucht es
nicht. Ein breiteres Band ist ein **Autorenfehler**: es macht den Datenpunkt
unbewertbar.

Nur Größen, die PyADM1ODE **wirklich simuliert**, fliessen ein. Auto-Knoten
(GasStorage je Digester, Flare je CHP/BGAA) werden über die Topologie ausgerichtet,
Bauteile grundsätzlich **nach Typ** zugeordnet, nie nach Namen.

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

## Rückfragen des Modells

Bei `underspecified`-Datenpunkten darf das Modell zuerst Fragen stellen — als
JSON-Block mit `open_questions` (Format siehe `prompt.py`). `solve.py` reicht sie
an das [Oracle](eval/oracle.py) weiter und schickt die Antworten in einen zweiten
Turn. Die Rückfragen selbst werden **nicht** benotet; bewertet wird ausschließlich
die Anlage, die am Ende dabei herauskommt. Wer nicht fragt und trotzdem plausible
Werte trifft, verliert nichts — wer nicht fragt und daneben liegt, verliert bei
**Maße**.

---
