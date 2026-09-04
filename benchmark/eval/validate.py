# benchmark/eval/validate.py
"""
Validates datapoints against ``benchmark/schema/plant_datapoint.schema.json``.

The schema sets ``additionalProperties: false`` throughout, which only helps if
someone actually validates -- otherwise fields nobody reads creep back in.
``make_index.py`` therefore calls this check on every run, and it works without
an external dependency (pure stdlib, like the rest of ``eval/``).

Supports the subset of JSON Schema Draft 2020-12 this schema uses: ``$ref``,
``type``, ``enum``, ``required``, ``properties``, ``additionalProperties`` (bool
or subschema), ``items`` and ``oneOf``. For a complete check (when changing the
schema itself, say) ``pip install jsonschema`` stays the reference; both report
the same errors on this schema.

CLI:
    python benchmark/eval/validate.py [dataset_dir]
"""

from __future__ import annotations

import glob
import json
import os
import sys
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA_PATH = os.path.abspath(os.path.join(HERE, "..", "schema", "plant_datapoint.schema.json"))
DATASET_DIR = os.path.abspath(os.path.join(HERE, "..", "dataset"))

_TYPES: dict[str, Any] = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "null": type(None),
}


def _type_ok(value: Any, name: str) -> bool:
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    expected = _TYPES.get(name)
    if expected is None:
        return True  # unbekannter Typname -> nicht pruefen
    if expected is not bool and isinstance(value, bool):
        return False  # True ist in Python ein int, in JSON Schema aber kein number
    return isinstance(value, expected)


def validate(instance: Any, schema: dict[str, Any], root: dict[str, Any], path: str = "") -> list[str]:
    """Return a list of error messages (empty means valid)."""
    if "$ref" in schema:
        ref = schema["$ref"]
        if not ref.startswith("#/"):
            return []  # externe Refs kommen in diesem Schema nicht vor
        target: Any = root
        for part in ref[2:].split("/"):
            target = target[part]
        return validate(instance, target, root, path)

    errors: list[str] = []
    loc = path or "<root>"

    if "oneOf" in schema:
        matches = [b for b in schema["oneOf"] if not validate(instance, b, root, path)]
        if len(matches) != 1:
            titles = ", ".join(b.get("title", "?") for b in schema["oneOf"])
            errors.append(f"{loc}: passt auf {len(matches)} der Alternativen ({titles}), erwartet genau 1")
        return errors

    if "type" in schema:
        names = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_type_ok(instance, n) for n in names):
            return [f"{loc}: Typ {type(instance).__name__}, erwartet {'/'.join(names)}"]

    if "enum" in schema and instance not in schema["enum"]:
        return [f"{loc}: Wert {instance!r} nicht erlaubt, zulaessig sind {schema['enum']}"]

    if isinstance(instance, dict):
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"{loc}: Pflichtfeld '{key}' fehlt")
        props = schema.get("properties", {})
        extra = schema.get("additionalProperties", True)
        for key, value in instance.items():
            sub = f"{path}.{key}" if path else key
            if key in props:
                errors += validate(value, props[key], root, sub)
            elif extra is False:
                errors.append(f"{loc}: unbekanntes Feld '{key}' (Schema erlaubt hier keine weiteren)")
            elif isinstance(extra, dict):
                errors += validate(value, extra, root, sub)

    if isinstance(instance, list) and isinstance(schema.get("items"), dict):
        for i, item in enumerate(instance):
            errors += validate(item, schema["items"], root, f"{path}[{i}]")

    return errors


def load_schema(path: str = SCHEMA_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def validate_datapoint(dp: dict[str, Any], schema: dict[str, Any] | None = None) -> list[str]:
    schema = schema or load_schema()
    return validate(dp, schema, schema)


def validate_dataset(dataset_dir: str = DATASET_DIR) -> dict[str, list[str]]:
    """Path -> error list, only for files with at least one error."""
    schema = load_schema()
    bad: dict[str, list[str]] = {}
    for path in sorted(glob.glob(os.path.join(dataset_dir, "**", "*.json"), recursive=True)):
        if os.path.basename(path) == "index.json":
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                dp = json.load(fh)
        except (json.JSONDecodeError, OSError) as e:
            bad[path] = [f"nicht lesbar: {e}"]
            continue
        if not (isinstance(dp, dict) and "reference" in dp and "input" in dp):
            continue
        errs = validate_datapoint(dp, schema)
        if errs:
            bad[path] = errs
    return bad


def report(dataset_dir: str = DATASET_DIR) -> int:
    """Validate and print; returns the number of invalid files."""
    bad = validate_dataset(dataset_dir)
    if not bad:
        return 0
    print(f"\n{len(bad)} Datenpunkt(e) verletzen das Schema:")
    for path, errs in bad.items():
        print(f"  {os.path.relpath(path, dataset_dir)}")
        for e in errs:
            print(f"      {e}")
    return len(bad)


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else DATASET_DIR
    n_bad = report(target)
    if n_bad == 0:
        print("Alle Datenpunkte schema-konform.")
    raise SystemExit(1 if n_bad else 0)
