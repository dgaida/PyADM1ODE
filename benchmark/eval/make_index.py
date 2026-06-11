# benchmark/eval/make_index.py
"""
Erzeugt ``dataset/index.json`` — das Manifest aller Datenpunkte fuer den
Live-Viewer. Nach dem Hinzufuegen/Aendern von Datenpunkten erneut ausfuehren:

    python benchmark/eval/make_index.py

Reines stdlib (laeuft ueberall). Listet jede *.json unter dataset/, die ein
Datenpunkt ist (hat ``reference`` + ``input``), mit Pfad relativ zu dataset/.
"""

import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.abspath(os.path.join(HERE, "..", "dataset"))


def main() -> None:
    entries = []
    for path in sorted(glob.glob(os.path.join(DATASET, "**", "*.json"), recursive=True)):
        if os.path.basename(path) == "index.json":
            continue
        try:
            d = json.load(open(path, encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not (isinstance(d, dict) and "reference" in d and "input" in d):
            continue
        rel = os.path.relpath(path, DATASET).replace(os.sep, "/")
        inp = d.get("input", {})
        entries.append(
            {
                "id": d.get("id", rel),
                "path": rel,
                "language": inp.get("language"),
                "modality": inp.get("modality"),
                "regime": d.get("regime"),
            }
        )

    out = os.path.join(DATASET, "index.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"datapoints": entries}, fh, ensure_ascii=False, indent=2)
    print(f"{len(entries)} Datenpunkte -> {os.path.relpath(out, os.path.join(HERE, '..', '..'))}")


if __name__ == "__main__":
    main()
