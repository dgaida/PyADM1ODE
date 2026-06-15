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


def _install_substrate_fallback() -> None:
    """Make substrate inputs non-fatal for the structure benchmark.

    The benchmark scores plant STRUCTURE only — substrates are deliberately not
    part of the task (``substrates_scored: false`` in every datapoint). Any
    substrate ID the candidate writes (valid, misspelled, or invented) must
    therefore never crash the build. We replace ``Feedstock``'s per-item
    resolver so unknown IDs fall back to a guaranteed-valid default substrate
    instead of raising ``FileNotFoundError``. This patch lives only inside the
    benchmark harness and does not affect production behaviour.
    """
    try:
        from pyadm1.substrates import feedstock as fs
    except Exception:
        return  # PyADM1ODE not importable here — nothing to patch

    # Guaranteed-valid fallback: prefer cattle_manure, else first available file.
    fallback_path = fs._DEFAULT_DATA_DIR / "cattle_manure.yaml"
    if not fallback_path.exists():
        fallback_path = None
        for ext in fs._SUBSTRATE_EXTENSIONS:
            hits = sorted(fs._DEFAULT_DATA_DIR.glob(f"*{ext}"))
            if hits:
                fallback_path = hits[0]
                break
    if fallback_path is None:
        return  # no substrate files to fall back to — leave original behaviour

    original = fs.Feedstock._resolve_substrate  # already a plain function via the class

    def safe_resolve(item):
        try:
            return original(item)
        except Exception:
            return fs.load_substrate(fallback_path)

    fs.Feedstock._resolve_substrate = staticmethod(safe_resolve)


def main() -> None:
    repo_root, code_path = sys.argv[1], sys.argv[2]
    sys.path.insert(0, repo_root)
    _install_substrate_fallback()

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
