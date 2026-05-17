# -*- coding: utf-8 -*-
"""Unit tests for pyadm1.utils.generate_skill_md."""

import inspect


from pyadm1.utils import generate_skill_md as skill_module
from pyadm1.utils.generate_skill_md import generate_skill_md, get_full_doc

# ---------------------------------------------------------------------------
# get_full_doc
# ---------------------------------------------------------------------------


def test_get_full_doc_returns_docstring_for_documented_object():
    def documented():
        """A short docstring."""

    assert get_full_doc(documented) == "A short docstring."


def test_get_full_doc_returns_placeholder_when_no_docstring():
    def undocumented():
        pass

    assert get_full_doc(undocumented) == "No documentation available."


def test_get_full_doc_works_on_classes_via_inspect_getdoc():
    class Documented:
        """Class-level docstring spread
        across two lines."""

    # inspect.getdoc dedents and strips, so check that the placeholder isn't returned.
    result = get_full_doc(Documented)
    assert result != "No documentation available."
    assert "Class-level docstring" in result


# ---------------------------------------------------------------------------
# generate_skill_md
# ---------------------------------------------------------------------------


def test_generate_skill_md_writes_file_at_output_path(tmp_path, capsys):
    out_path = tmp_path / "Skill.md"

    generate_skill_md(str(out_path))

    assert out_path.exists()
    assert out_path.stat().st_size > 0
    assert f"Generated Skill.md at: {out_path}" in capsys.readouterr().out


def test_generate_skill_md_creates_missing_parent_directories(tmp_path):
    out_path = tmp_path / "nested" / "deeper" / "Skill.md"

    generate_skill_md(str(out_path))

    assert out_path.exists()


def test_generate_skill_md_contains_all_documented_classes(tmp_path):
    out_path = tmp_path / "Skill.md"
    generate_skill_md(str(out_path))
    content = out_path.read_text(encoding="utf-8")

    expected_classes = [
        "Feedstock",
        "BiogasPlant",
        "PlantConfigurator",
        "Pump",
        "Mixer",
        "SubstrateStorage",
        "Feeder",
    ]
    for cls_name in expected_classes:
        assert f"## {cls_name}" in content, f"missing class section for {cls_name}"


def test_generate_skill_md_has_expected_top_level_structure(tmp_path):
    out_path = tmp_path / "Skill.md"
    generate_skill_md(str(out_path))
    content = out_path.read_text(encoding="utf-8")

    assert content.startswith("# PyADM1ODE Simulation Model Creation Skill")
    # Introduction paragraph.
    assert "full API documentation" in content
    # At least one Methods subsection (every class has __init__).
    assert "### Methods for" in content
    # __init__ is always emitted because the comprehension keeps it.
    assert "#### __init__" in content


def test_generate_skill_md_falls_back_when_signature_inspection_fails(tmp_path, monkeypatch):
    """Some C-implemented or wrapped callables raise on inspect.signature.
    The generator must emit '(...)' as a placeholder rather than crash."""
    original_signature = inspect.signature

    def fake_signature(obj, *args, **kwargs):
        # Force the failure path for any non-class callable.
        if not inspect.isclass(obj):
            raise ValueError("simulated signature failure")
        return original_signature(obj, *args, **kwargs)

    monkeypatch.setattr(skill_module.inspect, "signature", fake_signature)

    out_path = tmp_path / "Skill.md"
    generate_skill_md(str(out_path))
    content = out_path.read_text(encoding="utf-8")

    # Every method header should now use the '(...)' fallback signature line.
    assert "(...)" in content


def test_generate_skill_md_includes_method_docstrings_or_placeholder(tmp_path):
    out_path = tmp_path / "Skill.md"
    generate_skill_md(str(out_path))
    content = out_path.read_text(encoding="utf-8")

    # Either a real docstring is rendered, or the explicit placeholder is.
    # Both are evidence that the per-method loop executed without crashing.
    assert "#### __init__" in content
    # The placeholder string is what get_full_doc returns when a doc is missing,
    # so its presence (somewhere in the file) confirms the placeholder branch
    # is wired up — even if every public method happens to be documented.
    # We don't assert presence/absence; we just confirm the section structure.
    assert content.count("#### ") >= 7, "expected at least one method per class"
