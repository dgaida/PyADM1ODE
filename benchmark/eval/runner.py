# benchmark/eval/runner.py
"""
Runner: executes LMM-generated PyADM1ODE code in isolation and scores the
resulting plant against a reference datapoint.

Pipeline (matching the detail slide):
    run the code (sandbox + timeout)  ->  serialise the plant (to_dict)
    ->  graph matcher (matcher.evaluate)  ->  report

CLI:
    python runner.py <datapoint.json> <candidate_code.py>

Programmatic:
    from runner import run_candidate_code, evaluate_code
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import sys
import tempfile
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
HARNESS = os.path.join(HERE, "_harness.py")

sys.path.insert(0, HERE)

from matcher import Report, evaluate  # noqa: E402


def run_candidate_code(code: str, timeout: float = 90.0) -> tuple[dict[str, Any] | None, str]:
    """
    Run ``code`` in an isolated subprocess and return the plant serialisation.

    Returns
    -------
    (candidate_dict, error)
        candidate_dict is None on failure or timeout; ``error`` then carries the
        reason (otherwise "").
    """
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as tf:
        tf.write(code)
        code_path = tf.name
    try:
        proc = subprocess.run(  # noqa: S603 - fixed command with controlled args, no shell
            [sys.executable, HARNESS, REPO_ROOT, code_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, f"Timeout nach {timeout:.0f}s"
    finally:
        with contextlib.suppress(OSError):
            os.unlink(code_path)

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


def evaluate_code(datapoint: dict[str, Any], code: str, timeout: float = 90.0) -> Report:
    """Code ausfuehren + bewerten. Bei Ausfuehrungsfehler: build_success=False."""
    candidate, err = run_candidate_code(code, timeout=timeout)
    if candidate is None:
        rep = evaluate(datapoint, {})  # liefert build_success=False
        rep.details.setdefault("violations", []).insert(0, f"Ausfuehrung fehlgeschlagen: {err}")
        return rep
    return evaluate(datapoint, candidate)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python runner.py <datapoint.json> <candidate_code.py>")
        raise SystemExit(2)
    with open(sys.argv[1], encoding="utf-8") as f:
        dp = json.load(f)
    with open(sys.argv[2], encoding="utf-8") as f:
        code_src = f.read()
    print(evaluate_code(dp, code_src).pretty())
