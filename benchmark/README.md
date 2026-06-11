# PyADM1ODE LMM-Benchmark

Bewertet, ob ein LMM aus einer **Beschreibung oder einem Bild** einer Biogasanlage
korrekten PyADM1ODE-Code erzeugt, der die **richtige Anlagenstruktur** baut.

## Aufbau

```
benchmark/
  schema/   plant_datapoint.schema.json   JSON-Schema (Draft 2020-12) eines Datenpunkts
  dataset/  README.md                       Datensatz-Doku + Namenskonvention
            BGA1/  BGA1.json                 Datenpunkt: Aufgabe (Input + Referenz)
                   gold.py                    Gold-Loesung als ausfuehrbarer Code (~100 %)
  eval/     matcher.py                      Graph-Matcher (Scoring, stdlib-only)
            runner.py                       fuehrt LMM-Code isoliert aus + bewertet
            batch.py                        wertet alle Datenpunkte aus -> Tabelle + CSV
            make_index.py                   erzeugt dataset/index.json (Viewer-Manifest)
            _harness.py                     Subprozess-Harness (Code -> Anlagen-Dict)
            selftest.py                     Matcher-Selbsttest ohne ADM1
  docs/     dataset_structure_overview.svg  Praesentationsgrafik (Uebersicht)
            dataset_structure_detail.svg    Praesentationsgrafik (Detail)
  viewer/   index.html                      interaktiver Datenpunkt-Viewer (offline, kein Server)
```

## Drei Scores (siehe docs/dataset_structure_detail.svg)

1. **Struktur** — Bauteile (nach Typ zugeordnet, nicht nach Namen) + Verbindungen als typisierter Graph.
2. **Masse** — simulierte Parameter (V_liq, V_gas, T_ad, …) im Akzeptanzband.
3. **Fehlende Werte** — `missing_ask`: nachgefragt ODER plausibel gefuellt — statt still erfunden.

Nur Groessen, die PyADM1ODE **wirklich simuliert**, fliessen ein. Auto-Knoten
(GasStorage je Digester, Flare je CHP) werden ueber die Topologie ausgerichtet.

## Nutzung

Matcher gegen eine fertige Anlagen-Serialisierung (kein Code-Lauf noetig):

```bash
python benchmark/eval/matcher.py benchmark/dataset/BGA1/BGA1.json candidate.json [response.json]
```

LMM-Code ausfuehren **und** bewerten (braucht eine Umgebung mit den
PyADM1ODE-Abhaengigkeiten — numpy/pandas/scipy). In der Conda-Umgebung `biogas`:

```bash
conda run -n biogas --no-capture-output python benchmark/eval/runner.py \
    benchmark/dataset/BGA1/BGA1.json benchmark/dataset/BGA1/gold.py [response.json]
```

Kandidaten-Code muss am Ende die Variable `plant` (eine `BiogasPlant`) bereitstellen.
Der Gold-Kandidat erreicht 100 % / 100 % / 100 %.

**Alle** Datenpunkte auf einmal (Score-Tabelle + CSV):

```bash
conda run -n biogas --no-capture-output python benchmark/eval/batch.py
# optional: LLM-Ausgaben statt gold.py bewerten (sucht <id>.py im Ordner)
conda run -n biogas --no-capture-output python benchmark/eval/batch.py --candidates path/to/llm_outputs
```

`batch.py` findet je Datenpunkt den Kandidaten (Default `gold.py` im selben Ordner;
mit `--candidates DIR` ein `<id>.py`), fuehrt ihn isoliert aus, bewertet und schreibt
`benchmark/results.csv`. Varianten, die sich `gold.py` teilen, werden gecacht
(Anlage nur einmal gebaut). Optionale `<id>_response.json`/`response.json` neben dem
Datenpunkt fliessen in den Luecken-Score ein.

## Viewer

`benchmark/viewer/index.html` zeigt Datenpunkte als interaktiven Anlagengraphen:
Knoten nach Typ (Farbe), Kanten nach Verbindungstyp (Fluessig/Gas/Waerme),
Knotenrand nach `obligation`, `missing_ask` mit `?`-Badge. Klick auf einen Knoten
zeigt die Parameter mit Akzeptanzbaendern. Mit **◀ / ▶**, Dropdown oder den
**Pfeiltasten** durch die Datenpunkte skippen; bei gleicher Struktur bleibt der
Graph stehen und nur der Input wechselt.

Zwei Modi (Badge oben rechts):

- **● Live aus dataset/** — empfohlen. Ueber localhost servieren, dann laedt der
  Viewer die Datenpunkte direkt aus den Dateien (Edits nach Reload sichtbar):

  ```bash
  python benchmark/eval/make_index.py          # Manifest erzeugen/aktualisieren
  python -m http.server 8000                    # aus dem Repo-Root
  # Browser: http://localhost:8000/benchmark/viewer/
  ```

- **● eingebettet (Fallback)** — beim Doppelklick (`file://`) kann der Browser keine
  lokalen Dateien lesen; dann zeigt der Viewer eine eingebettete BGA1-Kopie. Per
  **Dateien laden…** lassen sich beliebige Datenpunkt-JSONs manuell oeffnen.

Nach dem Hinzufuegen/Aendern von Datenpunkten `make_index.py` erneut ausfuehren.

Selbsttest des Matchers (ohne ADM1, laeuft ueberall):

```bash
python benchmark/eval/selftest.py
```

## `response.json` (optional, fuer den Luecken-Score)

Strukturierte LMM-Antwort neben dem Code:

```json
{
  "open_questions": [{"field": "chp.P_el_nom"}, {"field": "sep.source"}],
  "assumptions":    [{"field": "F1.T_ad", "value": 313.15}]
}
```

Fragt das Modell nach einem `missing_ask`-Feld (open_questions) oder fuellt es
plausibel im Band, zaehlt das als korrekt; stilles Erfinden eines unplausiblen
Werts ist der schwerste Fehler.

## Sicherheit

`runner.py` nutzt Prozess-Isolation + Timeout — **kein** vollwertiges Sandboxing.
Fuer nicht vertrauenswuerdigen Modell-Code zusaetzlich Container/seccomp einsetzen.
