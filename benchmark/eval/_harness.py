# benchmark/eval/_harness.py
"""
Isolierter Ausfuehrungs-Harness fuer LMM-generierten Code.

Wird von ``runner.py`` als eigener Subprozess gestartet:
    python _harness.py <repo_root> <candidate_code.py>

Fuehrt den Kandidaten-Code aus, sucht das ``BiogasPlant``-Objekt (Variable
``plant`` oder ein beliebiges Objekt mit ``components``/``connections``) und gibt
dessen Serialisierung als JSON auf stdout aus. Fehler werden als
{"__error__": "..."} zurueckgegeben (statt zu crashen), damit der Runner sie als
build_success=False werten kann.

Hinweis: Das ist Prozess-Isolation + Timeout, KEIN vollwertiges Sandboxing.
Fuer nicht vertrauenswuerdigen Code zusaetzlich Container/seccomp verwenden.
"""

import json
import sys
import traceback


def main() -> None:
    repo_root, code_path = sys.argv[1], sys.argv[2]
    sys.path.insert(0, repo_root)

    try:
        with open(code_path, encoding="utf-8") as fh:
            src = fh.read()
        ns = {"__name__": "__candidate__"}
        exec(compile(src, code_path, "exec"), ns)
    except Exception:
        print(json.dumps({"__error__": traceback.format_exc()}))
        return

    plant = ns.get("plant")
    if plant is None:
        for val in ns.values():
            if hasattr(val, "components") and hasattr(val, "connections"):
                plant = val
                break
    if plant is None:
        print(json.dumps({"__error__": "Kein BiogasPlant gefunden (Variable `plant`?)."}))
        return

    try:
        components = [c.to_dict() for c in plant.components.values()]
        connections = [cn.to_dict() for cn in plant.connections]
    except Exception:
        print(json.dumps({"__error__": traceback.format_exc()}))
        return

    print(
        json.dumps(
            {
                "plant_name": getattr(plant, "plant_name", "?"),
                "components": components,
                "connections": connections,
            }
        )
    )


if __name__ == "__main__":
    main()
