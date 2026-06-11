# benchmark/eval/batch.py
"""
Batch-Runner: wertet alle Datenpunkte unter ``dataset/`` aus.

Fuer jeden Datenpunkt (JSON mit ``reference`` + ``input``):
  1. Kandidaten-Code finden (Default: ``gold.py`` im selben Ordner; optional ein
     ``--candidates DIR`` mit ``<id>.py``),
  2. Code isoliert ausfuehren (runner) -> Anlagen-Dict,
  3. mit dem Graph-Matcher bewerten,
  4. Zeile fuer die Score-Tabelle + CSV sammeln.

Mehrere Input-Varianten im selben Ordner teilen sich ``gold.py``; das Bauen der
Anlage wird pro Kandidaten-Datei gecacht (nur einmal ausgefuehrt).

Ausfuehren in der Umgebung mit PyADM1ODE-Deps:
    conda run -n biogas --no-capture-output python benchmark/eval/batch.py
    conda run -n biogas --no-capture-output python benchmark/eval/batch.py --candidates path/to/llm_outputs
"""

import argparse
import csv
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
from matcher import evaluate  # noqa: E402
from runner import run_candidate_code  # noqa: E402


def is_datapoint(d) -> bool:
    return isinstance(d, dict) and "reference" in d and "input" in d


def discover(dataset_dir: str):
    """Alle *.json, die wie Datenpunkte aussehen (mit reference + input)."""
    items = []
    for path in sorted(glob.glob(os.path.join(dataset_dir, "**", "*.json"), recursive=True)):
        try:
            d = json.load(open(path, encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if is_datapoint(d):
            items.append((path, d))
    return items


def find_candidate(dp_path: str, dp: dict, candidates_dir):
    dpid = dp.get("id") or os.path.splitext(os.path.basename(dp_path))[0]
    if candidates_dir:
        c = os.path.join(candidates_dir, dpid + ".py")
        if os.path.exists(c):
            return c
    gold = os.path.join(os.path.dirname(dp_path), "gold.py")
    return gold if os.path.exists(gold) else None


def find_response(dp_path: str, dp: dict):
    dpid = dp.get("id") or os.path.splitext(os.path.basename(dp_path))[0]
    folder = os.path.dirname(dp_path)
    for cand in (os.path.join(folder, dpid + "_response.json"), os.path.join(folder, "response.json")):
        if os.path.exists(cand):
            try:
                return json.load(open(cand, encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return None
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch-Bewertung aller Datenpunkte.")
    ap.add_argument(
        "--dataset", default=os.path.join(HERE, "..", "dataset"), help="Datensatz-Verzeichnis (Default: benchmark/dataset)."
    )
    ap.add_argument(
        "--candidates",
        default=None,
        help="Optionaler Ordner mit <id>.py (LLM-Ausgaben). " "Default: gold.py im jeweiligen Datenpunkt-Ordner.",
    )
    ap.add_argument("--csv", default=os.path.join(REPO_ROOT, "benchmark", "results.csv"), help="CSV-Ausgabepfad.")
    ap.add_argument("--timeout", type=float, default=120.0, help="Timeout pro Kandidat [s].")
    args = ap.parse_args()

    items = discover(args.dataset)
    if not items:
        print(f"Keine Datenpunkte in {args.dataset} gefunden.")
        return 1

    cand_cache = {}  # code_path -> (cand_dict, err)
    rows = []

    print(f"\nBewerte {len(items)} Datenpunkte aus {os.path.relpath(args.dataset, REPO_ROOT)} ...\n")
    header = f"{'#':>2}  {'id':<26} {'cand':<10} {'build':<6} {'Struk':>6} {'Masse':>6} {'Lueck':>6} {'Gesamt':>7}  W  V"
    print(header)
    print("-" * len(header))

    for i, (dp_path, dp) in enumerate(items, 1):
        dpid = dp.get("id") or os.path.splitext(os.path.basename(dp_path))[0]
        code_path = find_candidate(dp_path, dp, args.candidates)
        response = find_response(dp_path, dp)

        if code_path is None:
            rep = evaluate(dp, {}, response)
            rep.details.setdefault("violations", []).insert(0, "Kein Kandidat (gold.py) gefunden.")
            cand_label = "—"
        else:
            cand_label = os.path.basename(code_path)
            if code_path not in cand_cache:
                cand_cache[code_path] = run_candidate_code(open(code_path, encoding="utf-8").read(), timeout=args.timeout)
            cand, err = cand_cache[code_path]
            if cand is None:
                rep = evaluate(dp, {}, response)
                rep.details.setdefault("violations", []).insert(0, f"Ausfuehrung fehlgeschlagen: {err}")
            else:
                rep = evaluate(dp, cand, response)

        d = rep.details
        nW, nV = len(d.get("warnings", [])), len(d.get("violations", []))
        print(
            f"{i:>2}  {dpid[:26]:<26} {cand_label:<10} "
            f"{'OK' if rep.build_success else 'FAIL':<6} "
            f"{rep.structure:>6.1%} {rep.measures:>6.1%} {rep.gaps:>6.1%} {rep.overall():>7.1%}  "
            f"{nW}  {nV}"
        )

        rows.append(
            {
                "id": dpid,
                "datapoint": os.path.relpath(dp_path, REPO_ROOT),
                "candidate": os.path.relpath(code_path, REPO_ROOT) if code_path else "",
                "build_success": rep.build_success,
                "structure": rep.structure,
                "measures": rep.measures,
                "gaps": rep.gaps,
                "overall": rep.overall(),
                "n_warnings": nW,
                "n_violations": nV,
            }
        )

    # Zusammenfassung
    ok = [r for r in rows if r["build_success"]]
    print("-" * len(header))
    if ok:

        def mean(k):
            return sum(r[k] for r in ok) / len(ok)

        print(
            f"    {'MITTEL (build OK)':<26} {'':<10} {len(ok)}/{len(rows):<4} "
            f"{mean('structure'):>6.1%} {mean('measures'):>6.1%} "
            f"{mean('gaps'):>6.1%} {mean('overall'):>7.1%}"
        )
    n_warn = sum(r["n_warnings"] for r in rows)
    if n_warn:
        print(f"\n  {n_warn} Gaspfad-Warnung(en) insgesamt — Details via runner.py je Datenpunkt.")

    # CSV
    os.makedirs(os.path.dirname(args.csv), exist_ok=True)
    with open(args.csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nCSV geschrieben: {os.path.relpath(args.csv, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
