# Using the Dataset

This page is for **users of the dataset**, for example to evaluate or train your
own model. It explains **what the dataset contains**, **what you run**, and **how to
plug in your own model**.

The notebook benchmark/llm_benchmark_en.ipynb provides a simple introduction to using the dataset:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dgaida/PyADM1ODE/blob/master/benchmark/llm_benchmark_en.ipynb)

## Getting the dataset

The dataset is part of the repository under `benchmark/`. No separate download is
required:

```bash
git clone https://github.com/dgaida/PyADM1ODE.git
cd PyADM1ODE
```

The PyADM1ODE environment is needed to run code (the `biogas` conda environment is
recommended). Pure graph scoring (`matcher.py`) and the viewer work without
PyADM1ODE.

## What it contains

```text
benchmark/
  schema/    plant_datapoint.schema.json   JSON schema (Draft 2020-12) of a data point
  dataset/   index.json                    manifest of all data points
             BGA1/ BGA2/ ...               one plant each, input variants + gold.py
  eval/      solve.py runner.py matcher.py batch.py make_index.py …
  viewer/    index.html                    interactive data point viewer
```

**One folder per plant.** It holds all input variants plus a shared gold solution
`gold.py` (correct PyADM1ODE code).

| File | Role |
| ---- | ---- |
| `BGAx_<variant>.json` | **Task**: input (text/image) for the model **and** the reference plant (typed graph) for comparison |
| `gold.py` | **Gold solution**: known-correct implementation, validates the harness and serves as reference code |

### Variants and regime

Each plant exists in several variants. The `_full` suffix marks completeness:

- **`fully_specified`** (`_full`): all data is in the input – no oracle needed.  
- **`underspecified`** (no `_full`): values are missing – the model must ask or fill  
  them in plausibly.

| Axis | Values | Distribution (24) |
| ---- | ------ | ----------------- |
| Plant | BGA1, BGA2, BGA3 | 8 each |
| Completeness | fully_specified / underspecified | 12 / 12 |
| Modality | text / image / hybrid | 18 / 3 / 3 |
| Language | de / en | 18 / 6 |

The exact makeup of a single data point is described in
[A Data Point in Detail](datenpunkt.md). The authoritative format lives in the JSON
schema at `benchmark/schema/plant_datapoint.schema.json`.

## What you run

| Goal | Command |
| ---- | ------- |
| Matcher self-test (without PyADM1ODE) | `python benchmark/eval/selftest.py` |
| Baseline – all data points with `gold.py` | `python benchmark/eval/batch.py` |
| One data point: run code + score | `python benchmark/eval/runner.py <datapoint.json> <code.py>` |
| Score the graph only (no code run) | `python benchmark/eval/matcher.py <datapoint.json> <candidate.json>` |
| LLM evaluation | `python benchmark/eval/solve.py [filters]` |
| Refresh viewer/index | `python benchmark/eval/make_index.py` |

Examples:

```bash
# 1) Quick functional check of the matcher (no PyADM1ODE needed)
python benchmark/eval/selftest.py

# 2) Baseline: score all 24 data points against their gold.py
python benchmark/eval/batch.py

# 3) Run and score a single data point with the gold solution
python benchmark/eval/runner.py \
    benchmark/dataset/BGA1/BGA1_text_de.json benchmark/dataset/BGA1/gold.py
```

`batch.py` writes `benchmark/results.csv`. `solve.py` stores a CSV plus the
generated code per data point under `benchmark/results/`.

## Plugging in or training your own model

The **input** is in the data point under `input` (fields `modality`, `language`,
`content`, optionally `image_path`). The **target** is runnable PyADM1ODE code that
builds the plant. The **reference** for comparison is under `reference` (components +
connections as a typed graph). `gold.py` shows a correct implementation per plant.

The quickest path is **interactively in the notebook**. For automated runs there are
two scripted ways (A and B).

### A) Generate code offline, then score

Have your model produce Python code per data point and save it as
`<datapoint-id>.py` in a folder. Then:

```bash
python benchmark/eval/batch.py --candidates path/to/model_outputs
```

`batch.py` runs each candidate in isolation and scores it. If a `<id>.response.json`
sits next to `<id>.py`, it feeds into the missing-values score.

### B) Directly via solve.py (API)

`solve.py` is a full run including oracle rounds. By default it uses the Groq API.
For your own model, **only the client section** in `solve.py` needs adapting; the
rest of the workflow (prompt, oracle, scoring) stays the same.

```bash
pip install groq          # or your own client library
export GROQ_API_KEY=...    # or your own API key
python benchmark/eval/solve.py --regime fully_specified   # simplest entry point
```

### `response.json` (for the missing-values score)

For underspecified data points, what counts is whether a missing field was **asked**
or plausibly **filled in**. A structured response next to the code makes this
explicit:

```json
{
  "open_questions": [{"field": "chp.P_el_nom"}, {"field": "sep.source"}],
  "assumptions":    [{"field": "F1.T_ad", "value": 313.15}]
}
```

If the model asks about a `missing_ask` field or fills it plausibly within the band,
that counts as correct. Silently inventing an implausible value is the worst error.

## Programmatic access

The manifest `dataset/index.json` lists all data points with `id`, `path`,
`language`, `modality`, `regime`:

```python
import json
from pathlib import Path

root = Path("benchmark/dataset")
index = json.loads((root / "index.json").read_text(encoding="utf-8"))

for entry in index["datapoints"]:
    dp = json.loads((root / entry["path"]).read_text(encoding="utf-8"))
    task = dp["input"]["content"]        # description for the model
    reference = dp["reference"]          # target plant (components + connections)
    # ... call your own model, generate code, score ...
```

## Scoring

Three scores are computed:

1. **Structure** – components (matched by type, not by name) and connections.  
2. **Measures** – simulated parameters (`V_liq`, `V_gas`, `T_ad`, `P_el_nom`, …) within the acceptance band.  
3. **Missing values** – asked or plausibly filled instead of silently invented.  

How the final score is produced is explained in [Scoring & Workflow](bewertung.md).
You can explore the dataset visually in the [Viewer](viewer.md).
