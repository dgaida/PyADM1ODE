# Datensatz

Die Benchmark-Datenpunkte. **Ein Unterordner pro Datenpunkt**; darin liegen
die Aufgabe und die Gold-Loesung zusammen.

```
dataset/
  BGA1/
    BGA1.json   Datenpunkt: Input (Beschreibung/Bild) + Referenz-Anlage (Graph)
    gold.py     Gold-Loesung: ausfuehrbarer PyADM1ODE-Code, erreicht 100 %
```

Jeder Datenpunkt entspricht dem Schema `../schema/plant_datapoint.schema.json`.

## Zwei Dateien, zwei Rollen

| Datei | Wofuer | Genutzt von |
|-------|--------|-------------|
| `BGA1.json` | **Aufgabe** — Input, mit dem das LLM Code erzeugt; enthaelt zugleich die Referenz-Anlage (typisierter Graph) zum Abgleich | LLM-Prompt + `matcher.py` |
| `gold.py`   | **Gold-Loesung** — eine bekannt korrekte Umsetzung; validiert den Harness und dient als Referenzcode | `runner.py` |

Geprueft wird der LLM-Code **gegen die Referenz im JSON**, nicht gegen `gold.py`
direkt — `gold.py` ist die Soll-Umsetzung zum Vergleich/zur Harness-Validierung.

## Inhalt

Anlage **BGA1** (4 Digester, Separator, BHKW + Heizung) mit drei Input-Varianten,
alle `underspecified`, Split `test`, gemeinsame `gold.py`:

| Datenpunkt | Sprache | Beschreibung |
|------------|---------|--------------|
| `BGA1/BGA1.json`       | de | ausfuehrlicher Herstellertext (inkl. nicht relevanter Infos) |
| `BGA1/BGA1_terse.json` | de | knappe Beschreibung |
| `BGA1/BGA1_en.json`    | en | ausfuehrliche englische Beschreibung |

## Namenskonvention

Ordner = Anlage. Mehrere **Input-Varianten** derselben Anlage liegen im selben
Ordner und teilen sich `gold.py` (gleiche Anlage, andere Beschreibung -> prueft
die Robustheit des LLM, Achse B):

```
BGA1/
  BGA1.json        ausfuehrliche Textbeschreibung (de)
  BGA1_en.json     englische Variante
  BGA1_pid.json    als P&ID/Skizze (image)  + BGA1_pid.png
  BGA1_terse.json  knappe Beschreibung
  gold.py          gemeinsame Gold-Loesung
```

## Validieren / Bewerten

```bash
# Schema-Validierung
python -c "import json,jsonschema; \
  jsonschema.validate(json.load(open('benchmark/dataset/BGA1/BGA1.json',encoding='utf-8')), \
  json.load(open('benchmark/schema/plant_datapoint.schema.json',encoding='utf-8'))); print('ok')"

# Gold-Loesung ausfuehren + bewerten (Conda-Umgebung mit PyADM1ODE-Deps)
conda run -n biogas --no-capture-output python benchmark/eval/runner.py \
    benchmark/dataset/BGA1/BGA1.json benchmark/dataset/BGA1/gold.py

# Anzeigen im Viewer: benchmark/viewer/index.html -> 'Datei laden...' -> BGA1/BGA1.json
```
