# benchmark/eval/solve.py
"""
LLM evaluation for the PyADM1ODE benchmark.

Measures how well a model turns a plant description into runnable PyADM1ODE
code. Supports oracle follow-up questions for datapoints with missing values.

Per datapoint:
  1. build the prompt (text / image / hybrid / pdf)
  2. call the LLM (turn 1)
  3a. code extractable  -> score it directly
  3b. questions found   -> oracle answers -> LLM (turn 2) -> score it
  4. save the results and print the table

Requires: pip install groq
API key:  GROQ_API_KEY environment variable or --api-key

CLI examples:

  # Fully specified datapoints only (no oracle needed):
  python benchmark/eval/solve.py --regime fully_specified

  # All datapoints, oracle enabled:
  python benchmark/eval/solve.py

  # A single datapoint:
  python benchmark/eval/solve.py --id BGA2_text_de_full

  # Another model, custom output directory:
  python benchmark/eval/solve.py --model openai/gpt-oss-120b --output results/gptoss

  # Without the oracle (the LLM has to guess the missing values):
  python benchmark/eval/solve.py --no-oracle

Note: image and hybrid datapoints (the sketches) need a vision-capable model;
the API reports that per model as ``input_modalities``. Text-only runs via
--modality text.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATASET_DIR = os.path.join(HERE, "..", "dataset")
sys.path.insert(0, HERE)

from oracle import Oracle  # noqa: E402
from prompt import (  # noqa: E402
    SYSTEM_PROMPT,
    add_oracle_answers,
    append_assistant,
    build_messages,
)
from runner import evaluate_code  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers: parse the LLM response
# ---------------------------------------------------------------------------


def extract_code(text: str) -> str | None:
    """Extract the first ```python … ``` block from an LLM response."""
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # fallback: no code fence, but an import statement is present
    if "from pyadm1" in text or "import pyadm1" in text:
        return text.strip()
    return None


def extract_questions(text: str) -> list[dict[str, str]] | None:
    """
    Extract ``open_questions`` from a ```json … ``` block of the LLM response.
    Returns None if no JSON block is found or the list is empty.
    """
    m = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None
    questions = obj.get("open_questions", [])
    return questions if questions else None


def count_asked_fields(questions: list[Any]) -> int:
    """How many distinct fields the model asked the oracle for.

    The prompt asks for entries shaped ``{"field": ..., "question": ...}``, so
    distinct ``field`` values are counted; asking twice for ``F1.T_ad`` is one
    field. Entries without a ``field`` key fall back to being counted one by
    one, which is all a free-form question allows.
    """
    fields = {str(q.get("field")).strip() for q in questions if isinstance(q, dict) and str(q.get("field") or "").strip()}
    without_field = [q for q in questions if not (isinstance(q, dict) and str(q.get("field") or "").strip())]
    return len(fields) + len(without_field)


# ---------------------------------------------------------------------------
# Groq API client (OpenAI-compatible; for another API only this section changes)
# ---------------------------------------------------------------------------


def _get_client(api_key: str | None):
    try:
        from groq import Groq
    except ImportError:
        print("Fehler: 'groq' nicht installiert. -> pip install groq")
        sys.exit(1)
    key = api_key or os.environ.get("GROQ_API_KEY")
    if not key:
        print("Fehler: API-Key fehlt. Setze GROQ_API_KEY oder nutze --api-key.")
        sys.exit(1)
    return Groq(api_key=key)


#: Groq states the wait in the 429 message ("Please try again in 2.565s").
_RETRY_AFTER = re.compile(r"try again in ([\d.]+)s")


def call_llm(
    client,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int = 4096,
    retries: int = 3,
    system_prompt: str | None = None,
) -> str:
    """Send the messages to the Groq API (OpenAI-compatible) and return the text.

    A 429 (tokens per minute exhausted) is waited out and retried -- otherwise
    the benchmark measures the rate limit instead of the model. A 413 ("request
    too large") is *not* retried: the request exceeds the tier and still will
    on a second attempt.
    """
    full_messages = [
        {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
        *messages,
    ]
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=full_messages,
            )
            return resp.choices[0].message.content
        except Exception as exc:
            if getattr(exc, "status_code", None) != 429 or attempt == retries:
                raise
            hit = _RETRY_AFTER.search(str(exc))
            wait = float(hit.group(1)) + 0.5 if hit else 5.0 * (attempt + 1)
            print(f"     Rate-Limit: warte {wait:.1f}s (Versuch {attempt + 2}/{retries + 1})")
            time.sleep(min(wait, 60.0))
    raise RuntimeError("unerreichbar")  # pragma: no cover


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    dp_id: str
    regime: str
    modality: str
    language: str
    build_success: bool = False
    completeness: float = 0.0
    measures: float = 0.0
    inventions: float = 0.0
    overall: float = 0.0
    n_oracle_turns: int = 0
    #: how many fields the model asked the oracle for (0 = it did not ask)
    n_questions: int = 0
    error: str = ""
    generated_code: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.dp_id,
            "regime": self.regime,
            "modality": self.modality,
            "language": self.language,
            "build_success": self.build_success,
            "completeness": self.completeness,
            "measures": self.measures,
            "inventions": self.inventions,
            "overall": self.overall,
            "oracle_turns": self.n_oracle_turns,
            "questions": self.n_questions,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Core: evaluate one datapoint
# ---------------------------------------------------------------------------


def _condense(message: str, limit: int = 200) -> str:
    """Shorten a failure message to its most telling line.

    A traceback truncated at the front says "Traceback (most recent call last)"
    and nothing else; its *last* line names the exception that actually
    stopped the build.
    """
    lines = [ln.strip() for ln in message.splitlines() if ln.strip()]
    if not lines:
        return ""
    text = lines[-1] if len(lines) > 1 else lines[0]
    if len(lines) > 1 and not text.endswith("."):
        text = f"{lines[0].rstrip(':')}: {text}" if len(text) < 60 else text
    return text[:limit]


def evaluate_datapoint(
    dp: dict[str, Any],
    dp_dir: str,
    client,
    model: str,
    *,
    use_oracle: bool = True,
    verbose: bool = False,
    max_tokens: int = 4096,
    system_prompt: str | None = None,
) -> EvalResult:
    """
    Run the full evaluation for one datapoint:
      1. build the prompt
      2. LLM turn 1
      3. on questions: oracle -> LLM turn 2
      4. extract the code and score it
    """
    dp_id = dp.get("id", "?")
    regime = dp.get("regime", "underspecified")
    inp = dp.get("input", {})
    modality = inp.get("modality", "text")
    language = inp.get("language", "?")

    result = EvalResult(dp_id=dp_id, regime=regime, modality=modality, language=language)

    # -- build the prompt --
    allow_questions = use_oracle and regime == "underspecified"
    try:
        messages = build_messages(dp, dp_dir, allow_questions=allow_questions)
    except FileNotFoundError as e:
        result.error = str(e)
        return result

    # -- Turn 1 --
    try:
        resp1 = call_llm(client, model, messages, max_tokens=max_tokens, system_prompt=system_prompt)
    except Exception as e:  # noqa: BLE001 - record any LLM API failure and stop this datapoint
        result.error = f"API-Fehler Turn 1: {e}"
        return result

    if verbose:
        print(f"\n  [Turn 1 Antwort]\n{resp1[:400]}{'...' if len(resp1) > 400 else ''}")

    # -- code already in turn 1? --
    code = extract_code(resp1)
    questions = extract_questions(resp1)

    # -- oracle round (turn 2) --
    if code is None and questions and use_oracle and regime == "underspecified":
        result.n_oracle_turns = 1
        result.n_questions = count_asked_fields(questions)
        oracle = Oracle(dp)
        answer_text = oracle.answer(questions)

        if verbose:
            print(f"\n  [Oracle antwortet]\n{answer_text}")

        append_assistant(messages, resp1)
        add_oracle_answers(messages, answer_text)

        try:
            resp2 = call_llm(client, model, messages, max_tokens=max_tokens, system_prompt=system_prompt)
        except Exception as e:  # noqa: BLE001 - record any LLM API failure and stop this datapoint
            result.error = f"API-Fehler Turn 2: {e}"
            return result

        if verbose:
            print(f"\n  [Turn 2 Antwort]\n{resp2[:400]}{'...' if len(resp2) > 400 else ''}")

        code = extract_code(resp2)

    # -- no code extractable --
    if code is None:
        result.error = "Kein Python-Code in LLM-Antwort gefunden."
        return result

    result.generated_code = code

    # -- score it --
    try:
        report = evaluate_code(dp, code)
    except Exception as e:  # noqa: BLE001 - record any evaluation failure and stop this datapoint
        result.error = f"Evaluierungsfehler: {e}"
        return result

    result.build_success = report.build_success
    result.completeness = report.completeness
    result.measures = report.measures
    result.inventions = report.inventions
    result.overall = report.overall()
    if not report.build_success and not result.error:
        # Without this the CSV shows a bare 0: the reason (SyntaxError, an
        # invented API call, ...) lives only in the matcher's report.
        violations = report.details.get("violations") or []
        if violations:
            result.error = _condense(str(violations[0]))
    return result


# ---------------------------------------------------------------------------
# Load the dataset
# ---------------------------------------------------------------------------


def load_datapoints(
    dataset_dir: str,
    regime_filter: str | None,
    id_filter: str | None,
    modality_filter: str | None,
    language_filter: str | None,
) -> list[tuple[str, str, dict[str, Any]]]:
    """
    Load datapoints from dataset/index.json and apply the filters.

    Returns: list of (dp_id, dp_path_abs, datapoint_dict)
    """
    index_path = os.path.join(dataset_dir, "index.json")
    if not os.path.exists(index_path):
        print(f"Fehler: {index_path} nicht gefunden. -> python make_index.py")
        sys.exit(1)

    with open(index_path, encoding="utf-8") as f:
        index = json.load(f)
    selected = []

    for entry in index.get("datapoints", []):
        dp_id = entry["id"]
        # filters
        if id_filter and id_filter.lower() not in dp_id.lower():
            continue
        if regime_filter and regime_filter != "all" and entry.get("regime") != regime_filter:
            continue
        if modality_filter and entry.get("modality") != modality_filter:
            continue
        if language_filter and entry.get("language") != language_filter:
            continue

        rel_path = entry["path"]
        abs_path = os.path.join(dataset_dir, rel_path)
        if not os.path.exists(abs_path):
            print(f"  Warnung: {abs_path} nicht gefunden, uebersprungen.")
            continue
        with open(abs_path, encoding="utf-8") as f:
            dp = json.load(f)
        selected.append((dp_id, abs_path, dp))

    return selected


# ---------------------------------------------------------------------------
# Save the results
# ---------------------------------------------------------------------------


def save_results(results: list[EvalResult], output_dir: str, model: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_slug = re.sub(r"[^a-zA-Z0-9_-]", "_", model)

    # CSV
    csv_path = os.path.join(output_dir, f"{model_slug}_{timestamp}.csv")
    rows = [r.as_dict() for r in results]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # store the generated code (optional, for inspection)
    code_dir = os.path.join(output_dir, f"{model_slug}_{timestamp}_code")
    os.makedirs(code_dir, exist_ok=True)
    for r in results:
        if r.generated_code:
            with open(os.path.join(code_dir, f"{r.dp_id}.py"), "w", encoding="utf-8") as fh:
                fh.write(r.generated_code)

    print(f"\nCSV:  {os.path.relpath(csv_path, REPO_ROOT)}")
    print(f"Code: {os.path.relpath(code_dir, REPO_ROOT)}/")


# ---------------------------------------------------------------------------
# Print the table
# ---------------------------------------------------------------------------


def print_table(results: list[EvalResult], model: str) -> None:
    hdr = (
        f"{'#':>2}  {'ID':<28} {'Reg.':<7} {'Mod.':<7} {'B':>1} "
        f"{'Vollst':>6} {'Masse':>6} {'Erfund':>6} {'Gesamt':>7}  {'O':>1} {'Fragen':>6}"
    )
    print(f"\nModell: {model}")
    print(hdr)
    print("-" * len(hdr))
    for i, r in enumerate(results, 1):
        b = "✓" if r.build_success else "✗"
        print(
            f"{i:>2}  {r.dp_id[:28]:<28} {r.regime[:7]:<7} {r.modality[:7]:<7} {b:>1} "
            f"{r.completeness:>6.1%} {r.measures:>6.1%} {r.inventions:>6.1%} {r.overall:>7.1%}  "
            f"{r.n_oracle_turns:>1} {r.n_questions:>6}"
        )
        if r.error:
            print(f"     !! {r.error}")

    ok = [r for r in results if r.build_success]
    print("-" * len(hdr))
    if ok:

        def avg(attr):
            return sum(getattr(r, attr) for r in ok) / len(ok)

        # column widths mirror the data rows above, so the summary lines up
        label = f"MITTEL (build OK) {len(ok)}/{len(results)}"
        total_questions = sum(r.n_questions for r in results)
        print(
            f"{'':>2}  {label:<28} {'':<7} {'':<7} {'':>1} "
            f"{avg('completeness'):>6.1%} {avg('measures'):>6.1%} {avg('inventions'):>6.1%} "
            f"{avg('overall'):>7.1%}  {'':>1} {'S' + str(total_questions):>6}"
        )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def _force_utf8_stdout() -> None:
    """Keep the report printable on a cp1252 console.

    The result table marks a failed build with U+2717, and German plant labels
    carry umlauts - both raise UnicodeEncodeError on a Windows console, which
    used to abort the run *after* the API calls and before the results were
    written. The output stream is therefore switched to UTF-8 up front.
    """
    for stream in (sys.stdout, sys.stderr):
        # an exotic stream may lack reconfigure() or refuse the encoding
        with contextlib.suppress(AttributeError, ValueError):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    _force_utf8_stdout()
    ap = argparse.ArgumentParser(
        description="LLM-Benchmark für PyADM1ODE-Codegenerierung.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", default="llama-3.3-70b-versatile", help="Groq-Modell-ID (Default: llama-3.3-70b-versatile)")
    ap.add_argument("--api-key", default=None, help="Groq API-Key (oder GROQ_API_KEY setzen)")
    ap.add_argument(
        "--regime",
        choices=["fully_specified", "underspecified", "all"],
        default="all",
        help="Datenpunkt-Filter: fully_specified | underspecified | all",
    )
    ap.add_argument(
        "--modality",
        choices=["text", "image", "hybrid", "pdf"],
        default=None,
        help="Filter: text | image | hybrid | pdf",
    )
    ap.add_argument("--language", choices=["de", "en"], default=None, help="Filter: de | en")
    ap.add_argument("--id", default=None, help="Filter: nur Datenpunkte, deren ID diesen String enthalten")
    ap.add_argument("--no-oracle", action="store_true", help="Oracle deaktivieren (LLM muss fehlende Werte raten)")
    ap.add_argument("--dataset", default=DATASET_DIR, help="Datensatz-Verzeichnis")
    ap.add_argument(
        "--output", default=os.path.join(REPO_ROOT, "benchmark", "results"), help="Ausgabeverzeichnis für CSV und Code"
    )
    ap.add_argument("--verbose", action="store_true", help="LLM-Antworten ausdrucken")
    ap.add_argument("--delay", type=float, default=1.0, help="Pause zwischen API-Aufrufen in Sekunden (Default: 1.0)")
    ap.add_argument(
        "--system-prompt",
        default=None,
        metavar="DATEI",
        help="Textdatei mit einem eigenen System-Prompt. Der eingebaute Prompt ist "
        "bewusst minimal (nur API-Signaturen, Variablenname, Ausgabeformat) — "
        "Modellierungshinweise und Standardwerte zu formulieren ist Teil der Aufgabe",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Antwortbudget je Aufruf. Reasoning-Modelle zaehlen ihre Denk-Token "
        "mit und liefern bei 4096 mitunter gar keinen Code mehr (Default: 4096)",
    )
    args = ap.parse_args()

    # load the datapoints
    datapoints = load_datapoints(
        args.dataset,
        regime_filter=args.regime,
        id_filter=args.id,
        modality_filter=args.modality,
        language_filter=args.language,
    )
    if not datapoints:
        print("Keine Datenpunkte nach Filter. Prüfe --regime / --id / --modality.")
        return 1

    system_prompt = None
    if args.system_prompt:
        if not os.path.exists(args.system_prompt):
            print(f"Fehler: System-Prompt-Datei nicht gefunden: {args.system_prompt}")
            return 1
        with open(args.system_prompt, encoding="utf-8") as fh:
            system_prompt = fh.read()

    print(f"\n{len(datapoints)} Datenpunkte geladen  |  Modell: {args.model}")
    print(
        "System-Prompt: "
        + (f"{args.system_prompt} ({len(system_prompt)} Zeichen)" if system_prompt else "eingebauter Minimal-Prompt")
    )
    if args.no_oracle:
        print("Oracle: deaktiviert (LLM nutzt Standardwerte)")

    # Groq-Client initialisieren
    client = _get_client(args.api_key)

    # Evaluation
    results: list[EvalResult] = []
    for i, (dp_id, dp_path, dp) in enumerate(datapoints, 1):
        dp_dir = os.path.dirname(dp_path)
        print(f"\n[{i}/{len(datapoints)}] {dp_id}  ({dp.get('regime','?')} / {dp['input'].get('modality','?')})")

        result = evaluate_datapoint(
            dp,
            dp_dir,
            client,
            args.model,
            use_oracle=not args.no_oracle,
            verbose=args.verbose,
            max_tokens=args.max_tokens,
            system_prompt=system_prompt,
        )
        results.append(result)

        status = "OK" if result.build_success else "FAIL"
        print(
            f"  -> {status}  |  Gesamt: {result.overall:.1%}"
            f"  |  Oracle-Turns: {result.n_oracle_turns}"
            f"  |  Nachgefragte Felder: {result.n_questions}"
        )
        if result.error:
            print(f"     !! {result.error}")

        if i < len(datapoints) and args.delay > 0:
            time.sleep(args.delay)

    # Ausgabe
    print_table(results, args.model)
    save_results(results, args.output, args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
