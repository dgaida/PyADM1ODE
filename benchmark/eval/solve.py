# benchmark/eval/solve.py
"""
LLM-Evaluation fuer den PyADM1ODE-Benchmark.

Testet, wie gut ein KI-Modell aus einer Anlagenbeschreibung lauffaehigen
PyADM1ODE-Code generiert. Unterstuetzt Oracle-Rueckfragen fuer Datenpunkte,
bei denen Informationen fehlen.

Ablauf je Datenpunkt:
  1. Prompt aufbauen (Text / Bild / Hybrid)
  2. LLM aufrufen  (Turn 1)
  3a. Falls Code extrahierbar  -> direkt bewerten
  3b. Falls Fragen gefunden    -> Oracle antwortet -> LLM (Turn 2) -> bewerten
  4. Ergebnisse speichern und Tabelle ausgeben

Benoetigt: pip install groq
API-Key:   GROQ_API_KEY Umgebungsvariable oder --api-key

CLI-Beispiele:

  # Nur vollstaendig spezifizierte Datenpunkte (kein Oracle noetig):
  python benchmark/eval/solve.py --regime fully_specified

  # Alle Datenpunkte mit Oracle-Unterstuetzung:
  python benchmark/eval/solve.py

  # Einzelnen Datenpunkt testen:
  python benchmark/eval/solve.py --id BGA2_text_de_full

  # Anderes Modell, eigenes Ausgabeverzeichnis:
  python benchmark/eval/solve.py --model openai/gpt-oss-120b --output results/gptoss

  # Ohne Oracle (LLM nutzt Default-Werte):
  python benchmark/eval/solve.py --no-oracle

Hinweis: Bild-/Hybrid-Datenpunkte (Skizzen) benoetigen ein vision-faehiges
Groq-Modell (z.B. ein Llama-4-Vision-Modell). Reine Textlaeufe via --modality text.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Pfade
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DATASET_DIR = os.path.join(HERE, "..", "dataset")
sys.path.insert(0, HERE)

from oracle import Oracle  # noqa: E402
from prompt import (  # noqa: E402
    SYSTEM_PROMPT,
    build_messages,
    add_oracle_answers,
    append_assistant,
)
from runner import evaluate_code  # noqa: E402

# ---------------------------------------------------------------------------
# Hilfsfunktionen: LLM-Antwort parsen
# ---------------------------------------------------------------------------


def extract_code(text: str) -> Optional[str]:
    """Extrahiert den ersten ```python … ```-Block aus einer LLM-Antwort."""
    m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback: kein Codeblock-Marker, aber import-Statement gefunden
    if "from pyadm1" in text or "import pyadm1" in text:
        return text.strip()
    return None


def extract_questions(text: str) -> Optional[List[Dict[str, str]]]:
    """
    Extrahiert open_questions aus einem ```json … ```-Block der LLM-Antwort.
    Gibt None zurueck, wenn kein JSON-Block gefunden oder leer.
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


# ---------------------------------------------------------------------------
# Groq API-Client (OpenAI-kompatibel; bei anderer API nur diese Sektion anpassen)
# ---------------------------------------------------------------------------


def _get_client(api_key: Optional[str]):
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


def call_llm(
    client,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int = 4096,
) -> str:
    """Sendet Messages an die Groq API (OpenAI-kompatibel) und gibt den Antworttext zurueck."""
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=full_messages,
    )
    return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# Ergebnisstruktur
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    dp_id: str
    regime: str
    modality: str
    language: str
    build_success: bool = False
    structure: float = 0.0
    measures: float = 0.0
    gaps: float = 0.0
    overall: float = 0.0
    n_oracle_turns: int = 0
    error: str = ""
    generated_code: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.dp_id,
            "regime": self.regime,
            "modality": self.modality,
            "language": self.language,
            "build_success": self.build_success,
            "structure": self.structure,
            "measures": self.measures,
            "gaps": self.gaps,
            "overall": self.overall,
            "oracle_turns": self.n_oracle_turns,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Kern-Logik: einen Datenpunkt auswerten
# ---------------------------------------------------------------------------


def evaluate_datapoint(
    dp: Dict[str, Any],
    dp_dir: str,
    client,
    model: str,
    *,
    use_oracle: bool = True,
    verbose: bool = False,
) -> EvalResult:
    """
    Fuehrt die vollstaendige Evaluation fuer einen Datenpunkt durch:
      1. Prompt aufbauen
      2. LLM Turn 1
      3. Falls Fragen: Oracle -> LLM Turn 2
      4. Code extrahieren und bewerten
    """
    dp_id = dp.get("id", "?")
    regime = dp.get("regime", "underspecified")
    inp = dp.get("input", {})
    modality = inp.get("modality", "text")
    language = inp.get("language", "?")

    result = EvalResult(dp_id=dp_id, regime=regime, modality=modality, language=language)

    # -- Prompt aufbauen --
    allow_questions = use_oracle and regime == "underspecified"
    try:
        messages = build_messages(dp, dp_dir, allow_questions=allow_questions)
    except FileNotFoundError as e:
        result.error = str(e)
        return result

    # -- Turn 1 --
    try:
        resp1 = call_llm(client, model, messages)
    except Exception as e:
        result.error = f"API-Fehler Turn 1: {e}"
        return result

    if verbose:
        print(f"\n  [Turn 1 Antwort]\n{resp1[:400]}{'...' if len(resp1) > 400 else ''}")

    # -- Code direkt in Turn 1? --
    code = extract_code(resp1)
    questions = extract_questions(resp1)

    # -- Oracle-Runde (Turn 2) --
    if code is None and questions and use_oracle and regime == "underspecified":
        result.n_oracle_turns = 1
        oracle = Oracle(dp)
        answer_text = oracle.answer(questions)

        if verbose:
            print(f"\n  [Oracle antwortet]\n{answer_text}")

        append_assistant(messages, resp1)
        add_oracle_answers(messages, answer_text)

        try:
            resp2 = call_llm(client, model, messages)
        except Exception as e:
            result.error = f"API-Fehler Turn 2: {e}"
            return result

        if verbose:
            print(f"\n  [Turn 2 Antwort]\n{resp2[:400]}{'...' if len(resp2) > 400 else ''}")

        code = extract_code(resp2)

    # -- Kein Code extrahierbar --
    if code is None:
        result.error = "Kein Python-Code in LLM-Antwort gefunden."
        return result

    result.generated_code = code

    # -- Bewerten --
    try:
        report = evaluate_code(dp, code)
    except Exception as e:
        result.error = f"Evaluierungsfehler: {e}"
        return result

    result.build_success = report.build_success
    result.structure = report.structure
    result.measures = report.measures
    result.gaps = report.gaps
    result.overall = report.overall()
    return result


# ---------------------------------------------------------------------------
# Datensatz laden
# ---------------------------------------------------------------------------


def load_datapoints(
    dataset_dir: str,
    regime_filter: Optional[str],
    id_filter: Optional[str],
    modality_filter: Optional[str],
    language_filter: Optional[str],
) -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    Laedt Datenpunkte aus dataset/index.json und wendet Filter an.

    Returns: list of (dp_id, dp_path_abs, datapoint_dict)
    """
    index_path = os.path.join(dataset_dir, "index.json")
    if not os.path.exists(index_path):
        print(f"Fehler: {index_path} nicht gefunden. -> python make_index.py")
        sys.exit(1)

    index = json.load(open(index_path, encoding="utf-8"))
    selected = []

    for entry in index.get("datapoints", []):
        dp_id = entry["id"]
        # Filter
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
        dp = json.load(open(abs_path, encoding="utf-8"))
        selected.append((dp_id, abs_path, dp))

    return selected


# ---------------------------------------------------------------------------
# Ergebnisse speichern
# ---------------------------------------------------------------------------


def save_results(results: List[EvalResult], output_dir: str, model: str) -> None:
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

    # Generierten Code speichern (optional, zur Inspektion)
    code_dir = os.path.join(output_dir, f"{model_slug}_{timestamp}_code")
    os.makedirs(code_dir, exist_ok=True)
    for r in results:
        if r.generated_code:
            with open(os.path.join(code_dir, f"{r.dp_id}.py"), "w", encoding="utf-8") as fh:
                fh.write(r.generated_code)

    print(f"\nCSV:  {os.path.relpath(csv_path, REPO_ROOT)}")
    print(f"Code: {os.path.relpath(code_dir, REPO_ROOT)}/")


# ---------------------------------------------------------------------------
# Tabelle ausgeben
# ---------------------------------------------------------------------------


def print_table(results: List[EvalResult], model: str) -> None:
    hdr = (
        f"{'#':>2}  {'ID':<28} {'Reg.':<7} {'Mod.':<7} {'B':>1} "
        f"{'Struk':>6} {'Masse':>6} {'Lücke':>6} {'Gesamt':>7}  {'O':>1}"
    )
    print(f"\nModell: {model}")
    print(hdr)
    print("-" * len(hdr))
    for i, r in enumerate(results, 1):
        b = "✓" if r.build_success else "✗"
        print(
            f"{i:>2}  {r.dp_id[:28]:<28} {r.regime[:7]:<7} {r.modality[:7]:<7} {b:>1} "
            f"{r.structure:>6.1%} {r.measures:>6.1%} {r.gaps:>6.1%} {r.overall:>7.1%}  "
            f"{r.n_oracle_turns:>1}"
        )
        if r.error:
            print(f"     !! {r.error}")

    ok = [r for r in results if r.build_success]
    print("-" * len(hdr))
    if ok:

        def avg(attr):
            return sum(getattr(r, attr) for r in ok) / len(ok)

        print(
            f"    {'MITTEL (build OK)':<28} {'':<7} {'':<7} {len(ok)}/{len(results):<4} "
            f"{avg('structure'):>6.1%} {avg('measures'):>6.1%} {avg('gaps'):>6.1%} "
            f"{avg('overall'):>7.1%}"
        )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
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
    ap.add_argument("--modality", choices=["text", "image", "hybrid"], default=None, help="Filter: text | image | hybrid")
    ap.add_argument("--language", choices=["de", "en"], default=None, help="Filter: de | en")
    ap.add_argument("--id", default=None, help="Filter: nur Datenpunkte, deren ID diesen String enthalten")
    ap.add_argument("--no-oracle", action="store_true", help="Oracle deaktivieren (LLM nutzt Standardwerte)")
    ap.add_argument("--dataset", default=DATASET_DIR, help="Datensatz-Verzeichnis")
    ap.add_argument(
        "--output", default=os.path.join(REPO_ROOT, "benchmark", "results"), help="Ausgabeverzeichnis für CSV und Code"
    )
    ap.add_argument("--verbose", action="store_true", help="LLM-Antworten ausdrucken")
    ap.add_argument("--delay", type=float, default=1.0, help="Pause zwischen API-Aufrufen in Sekunden (Default: 1.0)")
    args = ap.parse_args()

    # Datenpunkte laden
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

    print(f"\n{len(datapoints)} Datenpunkte geladen  |  Modell: {args.model}")
    if args.no_oracle:
        print("Oracle: deaktiviert (LLM nutzt Standardwerte)")

    # Groq-Client initialisieren
    client = _get_client(args.api_key)

    # Evaluation
    results: List[EvalResult] = []
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
        )
        results.append(result)

        status = "OK" if result.build_success else f"FAIL ({result.error[:60]})"
        print(f"  -> {status}  |  Gesamt: {result.overall:.1%}  |  Oracle-Turns: {result.n_oracle_turns}")

        if i < len(datapoints) and args.delay > 0:
            time.sleep(args.delay)

    # Ausgabe
    print_table(results, args.model)
    save_results(results, args.output, args.model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
