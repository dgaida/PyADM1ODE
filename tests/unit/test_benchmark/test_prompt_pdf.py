"""PDF-Datenpunkte: Textebene extrahieren und in den Prompt einbauen.

Der realistische Fall ist ein echtes Anlagendokument -- etwa ein Angebotsschreiben --,
aus dem die Anlagenstruktur erst herausgelesen werden muss. Getestet wird gegen ein
im Test erzeugtes PDF, damit die Suite keine Binaerdatei im Repo braucht.
"""

from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "..", "benchmark", "eval"))
sys.path.insert(0, EVAL_DIR)

from prompt import build_messages, extract_pdf_text  # noqa: E402

pypdf = pytest.importorskip("pypdf", reason="PDF-Datenpunkte brauchen pypdf")

OFFER_PAGE_1 = [
    "Angebot Nr. 2026-0473",
    "Biogasanlage Musterhof - schluesselfertige Errichtung",
    "Sehr geehrte Damen und Herren,",
    "gerne unterbreiten wir Ihnen folgendes Angebot:",
    "Pos. 1  Fermenter F1, D = 28 m, H = 6 m",
    "Pos. 2  Nachgaerer N1, D = 28 m, H = 6 m",
    "Pos. 3  BHKW, elektrische Nennleistung 500 kW",
]
OFFER_PAGE_2 = [
    "Zahlungsbedingungen: 30 % bei Auftrag, 60 % bei Lieferung",
    "Die Betriebstemperatur der Fermenter betraegt 40 Grad C.",
]


def _write_pdf(path: str, pages: list[list[str]]) -> None:
    """Minimales PDF mit Textebene ueber pypdf erzeugen."""
    from pypdf import PdfWriter
    from pypdf.generic import ArrayObject, DecodedStreamObject, DictionaryObject, NameObject, NumberObject

    writer = PdfWriter()
    for lines in pages:
        page = writer.add_blank_page(width=595, height=842)
        body = "BT /F1 11 Tf 50 780 Td 14 TL\n" + "".join(f"({ln}) Tj T*\n" for ln in lines) + "ET"
        stream = DecodedStreamObject()
        stream.set_data(body.encode("latin-1"))
        font = DictionaryObject()
        font.update(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        resources = DictionaryObject()
        fonts = DictionaryObject()
        fonts[NameObject("/F1")] = writer._add_object(font)
        resources[NameObject("/Font")] = fonts
        page[NameObject("/Resources")] = resources
        page[NameObject("/Contents")] = writer._add_object(stream)
        page[NameObject("/MediaBox")] = ArrayObject([NumberObject(0), NumberObject(0), NumberObject(595), NumberObject(842)])
    with open(path, "wb") as fh:
        writer.write(fh)


@pytest.fixture
def offer_pdf(tmp_path) -> str:
    path = os.path.join(tmp_path, "angebot.pdf")
    _write_pdf(path, [OFFER_PAGE_1, OFFER_PAGE_2])
    return path


class TestExtractPdfText:
    def test_reads_all_pages(self, offer_pdf: str) -> None:
        text = extract_pdf_text(offer_pdf)

        assert "Angebot Nr. 2026-0473" in text
        assert "Fermenter F1, D = 28 m" in text
        assert "Betriebstemperatur" in text  # Seite 2 ist mit drin
        assert "--- Seite 1 ---" in text and "--- Seite 2 ---" in text

    def test_long_document_is_not_truncated(self, tmp_path) -> None:
        """Kein Seiten- oder Zeichendeckel: Kürzen würde die Aufgabe verändern.

        Steht die entscheidende Angabe auf einer späten Seite, wäre der Datenpunkt
        nach einer Kürzung unlösbar — der Benchmark würde die Kürzung messen.
        """
        pages = [[f"Position {i}: Nebenposition ohne technischen Inhalt"] for i in range(1, 60)]
        pages.append(["Technischer Anhang: Fermenter V_liq = 3325 m3, T_ad = 40 Grad C"])
        path = os.path.join(tmp_path, "langes_angebot.pdf")
        _write_pdf(path, pages)

        text = extract_pdf_text(path)

        assert "--- Seite 60 ---" in text
        assert "V_liq = 3325 m3" in text  # letzte Seite kommt vollstaendig an
        assert "abgeschnitten" not in text

    def test_scan_without_text_layer_is_refused(self, tmp_path) -> None:
        from pypdf import PdfWriter

        path = os.path.join(tmp_path, "scan.pdf")
        writer = PdfWriter()
        writer.add_blank_page(width=595, height=842)
        with open(path, "wb") as fh:
            writer.write(fh)

        with pytest.raises(ValueError, match="keine Textebene"):
            extract_pdf_text(path)


class TestBuildMessagesPdf:
    def test_pdf_content_reaches_the_prompt(self, tmp_path, offer_pdf: str) -> None:
        dp = {
            "id": "X_pdf_de",
            "input": {"modality": "pdf", "language": "de", "document_path": "angebot.pdf"},
            "regime": "underspecified",
        }

        messages = build_messages(dp, str(tmp_path), allow_questions=False)

        text = "".join(part["text"] for part in messages[0]["content"] if part["type"] == "text")
        assert "angebot.pdf" in text
        assert "Fermenter F1, D = 28 m" in text
        assert "500 kW" in text
        # Kein Bild-Part: das PDF geht als Text raus, nicht als image_url.
        assert all(part["type"] == "text" for part in messages[0]["content"])

    def test_supplementary_content_is_appended(self, tmp_path, offer_pdf: str) -> None:
        dp = {
            "id": "X_pdf_de",
            "input": {
                "modality": "pdf",
                "language": "de",
                "document_path": "angebot.pdf",
                "content": "Das Gaerrestlager ist unbeheizt.",
            },
            "regime": "underspecified",
        }

        text = "".join(part["text"] for part in build_messages(dp, str(tmp_path))[0]["content"] if part["type"] == "text")
        assert "Fermenter F1" in text
        assert "Gaerrestlager ist unbeheizt" in text

    def test_missing_document_fails_loudly(self, tmp_path) -> None:
        dp = {"id": "X", "input": {"modality": "pdf", "document_path": "fehlt.pdf"}, "regime": "underspecified"}

        with pytest.raises(FileNotFoundError, match="PDF nicht gefunden"):
            build_messages(dp, str(tmp_path))

    def test_document_path_is_required_for_pdf(self, tmp_path) -> None:
        dp = {"id": "X", "input": {"modality": "pdf"}, "regime": "underspecified"}

        with pytest.raises(FileNotFoundError):
            build_messages(dp, str(tmp_path))


class TestSchemaAcceptsPdfDatapoint:
    def test_pdf_modality_and_document_path_validate(self) -> None:
        from validate import load_schema, validate_datapoint

        dp = {
            "id": "BGA9_pdf_de",
            "input": {"modality": "pdf", "language": "de", "document_path": "angebot.pdf"},
            "regime": "underspecified",
            "reference": {
                "components": [
                    {
                        "id": "F1",
                        "type": "Digester",
                        "obligation": "given",
                        "params": {"V_liq": {"value": 3325, "obligation": "given", "accept": {"min": 3291, "max": 3359}}},
                    }
                ],
                "connections": [],
            },
        }

        assert validate_datapoint(dp, load_schema()) == []
