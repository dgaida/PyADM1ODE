# -*- coding: utf-8 -*-
"""Unit tests for pyadm1.utils.fix_markdown_lists."""

from pathlib import Path

import pytest

from pyadm1.utils.fix_markdown_lists import fix_markdown_file, main

# ---------------------------------------------------------------------------
# fix_markdown_file
# ---------------------------------------------------------------------------


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def test_fix_markdown_file_appends_two_spaces_to_dash_bullet(tmp_path, capsys):
    md = tmp_path / "doc.md"
    _write(md, "- first item\n- second item\n")

    fix_markdown_file(str(md))

    assert md.read_text(encoding="utf-8") == "- first item  \n- second item  \n"
    assert f"Fixed lists in: {md}" in capsys.readouterr().out


@pytest.mark.parametrize("marker", ["-", "*", "+"])
def test_fix_markdown_file_handles_all_bullet_markers(tmp_path, marker):
    md = tmp_path / "doc.md"
    _write(md, f"{marker} item one\n{marker} item two\n")

    fix_markdown_file(str(md))

    assert md.read_text(encoding="utf-8") == f"{marker} item one  \n{marker} item two  \n"


def test_fix_markdown_file_handles_numbered_lists(tmp_path):
    md = tmp_path / "doc.md"
    _write(md, "1. step one\n2. step two\n10. step ten\n")

    fix_markdown_file(str(md))

    assert md.read_text(encoding="utf-8") == "1. step one  \n2. step two  \n10. step ten  \n"


def test_fix_markdown_file_handles_indented_lists(tmp_path):
    md = tmp_path / "doc.md"
    _write(md, "  - nested item\n    * deeply nested\n")

    fix_markdown_file(str(md))

    assert md.read_text(encoding="utf-8") == "  - nested item  \n    * deeply nested  \n"


def test_fix_markdown_file_skips_non_list_lines(tmp_path, capsys):
    md = tmp_path / "doc.md"
    content = "# Heading\n\nA paragraph of prose.\n\n```code\nblock\n```\n"
    _write(md, content)

    fix_markdown_file(str(md))

    assert md.read_text(encoding="utf-8") == content
    # No "Fixed lists in" print when nothing changed.
    assert "Fixed lists in" not in capsys.readouterr().out


def test_fix_markdown_file_skips_lines_already_ending_in_two_spaces(tmp_path, capsys):
    md = tmp_path / "doc.md"
    content = "- already padded  \n- and this one too  \n"
    _write(md, content)

    fix_markdown_file(str(md))

    assert md.read_text(encoding="utf-8") == content
    assert "Fixed lists in" not in capsys.readouterr().out


def test_fix_markdown_file_repads_line_with_single_trailing_space(tmp_path):
    """A list line with one trailing space (less than 2) gets repadded to 2."""
    md = tmp_path / "doc.md"
    _write(md, "- one trailing space \n")

    fix_markdown_file(str(md))

    assert md.read_text(encoding="utf-8") == "- one trailing space  \n"


def test_fix_markdown_file_leaves_three_trailing_spaces_untouched(tmp_path, capsys):
    """``endswith('  ')`` is satisfied by *any* line ending in ≥2 spaces, so
    lines with 3+ trailing spaces are not re-normalised to exactly 2."""
    md = tmp_path / "doc.md"
    content = "- three trailing spaces   \n"
    _write(md, content)

    fix_markdown_file(str(md))

    assert md.read_text(encoding="utf-8") == content
    assert "Fixed lists in" not in capsys.readouterr().out


def test_fix_markdown_file_mixes_list_and_prose(tmp_path):
    md = tmp_path / "doc.md"
    _write(md, "# Title\n\nIntro paragraph.\n\n- bullet one\n- bullet two\n\nMore prose.\n")

    fix_markdown_file(str(md))

    assert md.read_text(encoding="utf-8") == ("# Title\n\nIntro paragraph.\n\n- bullet one  \n- bullet two  \n\nMore prose.\n")


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def test_main_walks_directory_tree_and_only_processes_md(tmp_path, capsys):
    # .md at root, .md in subdir, plus non-.md files that must be ignored.
    (tmp_path / "root.md").write_text("- a\n", encoding="utf-8")
    (tmp_path / "ignore.txt").write_text("- not markdown\n", encoding="utf-8")
    (tmp_path / "README.MD").write_text("- uppercase ext should be ignored\n", encoding="utf-8")

    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "nested.md").write_text("- nested\n", encoding="utf-8")

    main(str(tmp_path))

    assert (tmp_path / "root.md").read_text(encoding="utf-8") == "- a  \n"
    assert (sub / "nested.md").read_text(encoding="utf-8") == "- nested  \n"

    # Non-.md files untouched.
    assert (tmp_path / "ignore.txt").read_text(encoding="utf-8") == "- not markdown\n"
    assert (tmp_path / "README.MD").read_text(encoding="utf-8") == "- uppercase ext should be ignored\n"

    out = capsys.readouterr().out
    assert "root.md" in out
    assert "nested.md" in out


def test_main_on_empty_directory_does_nothing(tmp_path, capsys):
    main(str(tmp_path))
    assert capsys.readouterr().out == ""
