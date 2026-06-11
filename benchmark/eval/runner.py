# benchmark/eval/runner.py
"""
Runner: fuehrt LMM-generierten PyADM1ODE-Code isoliert aus und bewertet die
gebaute Anlage gegen einen Referenz-Datenpunkt.

Pipeline (entspricht der Detail-Folie):
    Code ausfuehren (Sandbox + Timeout)  ->  Anlage serialisieren (to_dict)
    ->  Graph-Matcher (matcher.evaluate)  ->  Report

CLI:
    python runner.py <datapoint.json> <candidate_code.py> [response.json]

Programmatic:
    from runner import run_candidate_code, evaluate_code
"""

import json
import os
import subprocess
import sys
import tempfile
from typing import Any, Dict, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
HARNESS = os.path.join(HERE, "_harness.py")

sys.path.insert(0, HERE)
from matcher import evaluate, Report  # noqa: E402


def run_candidate_code(code: str, timeout: float = 90.0) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Fuehrt ``code`` in einem isolierten Subprozess aus und gibt die
    Anlagen-Serialisierung zurueck.

    Returns
    -------
    (candidate_dict, error)
        candidate_dict ist None bei Fehler/Timeout; error enthaelt dann die
        Begruendung (sonst "").
    """
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tf:
        tf.write(code)
        code_path = tf.name
    try:
        proc = subprocess.run(
            [sys.executable, HARNESS, REPO_ROOT, code_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None, f"Timeout nach {timeout:.0f}s"
    finally:
        try:
            os.unlink(code_path)
        except OSError:
            pass

    out = (proc.stdout or "").strip()
    if not out:
        return None, f"Keine Ausgabe. stderr:\n{proc.stderr.strip()[:2000]}"
    # Letzte JSON-Zeile parsen (vorherige prints des Kandidaten ignorieren)
    last = out.splitlines()[-1]
    try:
        data = json.loads(last)
    except json.JSONDecodeError:
        return None, f"Ausgabe nicht parsebar:\n{out[:2000]}"
    if "__error__" in data:
        return None, data["__error__"]
    return data, ""


def evaluate_code(
    datapoint: Dict[str, Any], code: str, response: Optional[Dict[str, Any]] = None, timeout: float = 90.0
) -> Report:
    """Code ausfuehren + bewerten. Bei Ausfuehrungsfehler: build_success=False."""
    candidate, err = run_candidate_code(code, timeout=timeout)
    if candidate is None:
        rep = evaluate(datapoint, {}, response)  # liefert build_success=False
        rep.details.setdefault("violations", []).insert(0, f"Ausfuehrung fehlgeschlagen: {err}")
        return rep
    return evaluate(datapoint, candidate, response)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python runner.py <datapoint.json> <candidate_code.py> [response.json]")
        raise SystemExit(2)
    dp = json.load(open(sys.argv[1], encoding="utf-8"))
    code_src = open(sys.argv[2], encoding="utf-8").read()
    resp = json.load(open(sys.argv[3], encoding="utf-8")) if len(sys.argv) > 3 else None
    print(evaluate_code(dp, code_src, resp).pretty())
