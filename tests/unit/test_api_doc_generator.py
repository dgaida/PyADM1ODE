import io
import sys
from pathlib import Path

import pytest

import pyadm1.utils.api_doc_generator as api_doc_mod
from pyadm1.utils.api_doc_generator import APIDocGenerator


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture
def fake_package(tmp_path, monkeypatch):
    pkg_root = tmp_path / "fakepkg"
    _write(
        pkg_root / "__init__.py",
        '''"""Fake package doc.\n\nModules:\nbase\n"""\nfrom .base import BaseThing\nfrom .subpkg import SubThing\n''',
    )
    _write(
        pkg_root / "base.py",
        '''class BaseThing:\n    """BaseThing docs.\n\nAttributes:\n    x: demo attr\nExamples:\n    >>> 1 + 1\n    2\n"""\n    def __init__(self, a, b=2):\n        pass\n    def run(self, x=1):\n        """Run docs."""\n        return x\n    def _private(self):\n        return None\n''',
    )
    _write(
        pkg_root / "subpkg" / "__init__.py",
        '''"""Sub package doc."""\nfrom .mod import SubThing\n__all__ = ["SubThing", "_Ignore"]\n''',
    )
    _write(
        pkg_root / "subpkg" / "mod.py",
        '''class SubThing:\n    """Sub docs."""\n    def __init__(self):\n        pass\n    def ping(self):\n        """Ping docs."""\n        return "pong"\n''',
    )
    _write(pkg_root / "_private" / "__init__.py", "x = 1\n")
    _write(pkg_root / "nopy" / "readme.txt", "ignore\n")

    monkeypatch.syspath_prepend(str(tmp_path))
    yield "fakepkg"

    for key in list(sys.modules):
        if key == "fakepkg" or key.startswith("fakepkg."):
            sys.modules.pop(key, None)


def test_init_creates_output_dir_and_auto_title(tmp_path):
    out = tmp_path / "docs" / "api"
    gen = APIDocGenerator("pyadm1.test_pkg", output_dir=str(out))

    assert out.exists()
    assert gen.package_title == "Test Pkg"
    assert gen._get_main_filename() == "test_pkg.md"


def test_init_uses_explicit_package_title(tmp_path):
    gen = APIDocGenerator("pkg.name", output_dir=str(tmp_path / "out"), package_title="Custom Title")
    assert gen.package_title == "Custom Title"


def test_package_discovery_helpers(fake_package, tmp_path):
    gen = APIDocGenerator(fake_package, output_dir=str(tmp_path / "out"))
    pkg = __import__(fake_package)

    assert gen._get_subpackages(pkg) == ["subpkg"]
    assert gen._get_direct_package_classes(fake_package) == ["BaseThing"]
    assert gen._get_package_classes(f"{fake_package}.subpkg") == ["SubThing"]
    assert gen._get_package_classes("does.not.exist") == []
    assert gen._get_direct_package_classes("does.not.exist") == []


def test_docstring_preprocess_and_format_helpers(tmp_path):
    gen = APIDocGenerator("x.y", output_dir=str(tmp_path / "out"))
    raw = "Args:\nparam\n\nExample:\n>>> a = 1\n>>> a\n1\nReturns:\nvalue"
    pre = gen._preprocess_docstring(raw)
    fmt = gen._format_docstring(pre)

    assert "Args:\n\nparam" in pre
    assert "Example:\n\n```python" in fmt
    assert ">>> a = 1" in fmt
    assert "```" in fmt
    assert gen._format_title("my_module") == "My Module"
    assert gen._get_brief_description("\n\nFirst line\nSecond") == "First line"
    assert gen._get_brief_description("") == ""
    assert gen._preprocess_docstring("") == ""
    assert gen._format_docstring("") == ""
    assert gen._format_docstring("\n  \n") == ""
    assert gen._get_brief_description("  \n   ") == ""


def test_document_attributes_parses_attributes_section(tmp_path):
    gen = APIDocGenerator("x.y", output_dir=str(tmp_path / "out"))

    class Demo:
        """
        Demo class.

        Attributes:
            foo: first
            bar: second
        Example:
            >>> 1
        """

    buf = io.StringIO()
    gen._document_attributes(Demo, buf)
    text = buf.getvalue()

    assert "**Attributes:**" in text
    assert "- foo: first" in text
    assert "- bar: second" in text


def test_document_attributes_returns_for_missing_docstring(tmp_path):
    gen = APIDocGenerator("x.y", output_dir=str(tmp_path / "out"))

    class Demo:
        pass

    buf = io.StringIO()
    gen._document_attributes(Demo, buf)
    assert buf.getvalue() == ""


def test_document_class_success_and_error_paths(fake_package, tmp_path):
    gen = APIDocGenerator(fake_package, output_dir=str(tmp_path / "out"))
    buf = io.StringIO()

    gen._document_class(fake_package, "BaseThing", buf)
    text = buf.getvalue()

    assert "BaseThing docs." in text
    assert "**Signature:**" in text
    assert "a," in text
    assert "b=2" in text
    assert "**Methods:**" in text
    assert "`run()`" in text
    assert "Run docs." in text
    assert "_private" not in text

    err_buf = io.StringIO()
    gen._document_class(fake_package, "MissingClass", err_buf)
    assert "Error documenting MissingClass" in err_buf.getvalue()


def test_document_class_handles_signature_errors(monkeypatch, fake_package, tmp_path):
    gen = APIDocGenerator(fake_package, output_dir=str(tmp_path / "out"))
    real_signature = api_doc_mod.inspect.signature
    calls = {"n": 0}

    def flaky_signature(obj):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise ValueError("no sig")
        return real_signature(obj)

    monkeypatch.setattr(api_doc_mod.inspect, "signature", flaky_signature)
    buf = io.StringIO()
    gen._document_class(fake_package, "BaseThing", buf)
    text = buf.getvalue()

    assert "BaseThing docs." in text
    assert "**Methods:**" in text


def test_document_class_covers_dunder_skip_and_required_method_param(tmp_path):
    gen = APIDocGenerator("x.y", output_dir=str(tmp_path / "out"), exclude_private=False)

    class Demo:
        """Demo docs."""

        def __init__(self, a=1):
            pass

        def __repr__(self):
            return "x"

        def run_required(self, value):
            """Required arg docs."""
            return value

    module = type("M", (), {"Demo": Demo})
    buf = io.StringIO()

    real_import = api_doc_mod.importlib.import_module
    try:
        api_doc_mod.importlib.import_module = lambda name: module
        gen._document_class("fake.mod", "Demo", buf)
    finally:
        api_doc_mod.importlib.import_module = real_import

    text = buf.getvalue()
    assert "`run_required()`" in text
    assert "run_required(value)" in text
    assert "__repr__" not in text


def test_generate_main_and_subpackage_docs_write_files(fake_package, tmp_path):
    out = tmp_path / "docs_out"
    gen = APIDocGenerator(fake_package, output_dir=str(out))
    pkg = __import__(fake_package)

    gen._generate_main_file(pkg, ["subpkg"], ["BaseThing"], "fakepkg.md")
    gen._generate_subpackage_docs("subpkg")

    main_text = (out / "fakepkg.md").read_text(encoding="utf-8")
    sub_text = (out / "subpkg.md").read_text(encoding="utf-8")

    assert "# Fakepkg" in main_text
    assert "## Subpackages" in main_text
    assert "### [subpkg](subpkg.md)" in main_text
    assert "## Base Classes" in main_text
    assert "from fakepkg import BaseThing" in main_text
    assert "# Subpkg" in sub_text
    assert "## Classes" in sub_text
    assert "from fakepkg.subpkg import SubThing" in sub_text


def test_generate_main_file_ignores_subpackage_import_error(fake_package, monkeypatch, tmp_path):
    out = tmp_path / "docs_out"
    gen = APIDocGenerator(fake_package, output_dir=str(out))
    pkg = __import__(fake_package)
    real_import = api_doc_mod.importlib.import_module

    def fake_import(name):
        if name == "fakepkg.subpkg":
            raise ImportError("skip")
        return real_import(name)

    monkeypatch.setattr(api_doc_mod.importlib, "import_module", fake_import)
    gen._generate_main_file(pkg, ["subpkg"], [], "fakepkg.md")

    text = (out / "fakepkg.md").read_text(encoding="utf-8")
    assert "### [subpkg](subpkg.md)" not in text


def test_generate_subpackage_docs_import_error_prints_and_returns(monkeypatch, tmp_path, capsys):
    gen = APIDocGenerator("fakepkg", output_dir=str(tmp_path / "out"))
    monkeypatch.setattr(
        api_doc_mod.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError("boom")),
    )

    gen._generate_subpackage_docs("missing")

    assert "Could not import fakepkg.missing: boom" in capsys.readouterr().out


def test_get_package_classes_without___all___uses_inspect_branch(monkeypatch, tmp_path):
    gen = APIDocGenerator("x.y", output_dir=str(tmp_path / "out"))

    Public = type("Public", (), {})
    Public.__module__ = "pkg.mod"
    Other = type("Other", (), {})
    Other.__module__ = "other.pkg"
    _Private = type("_Private", (), {})
    _Private.__module__ = "pkg.mod"

    module = type("M", (), {})()
    if hasattr(module, "__all__"):
        delattr(module, "__all__")

    monkeypatch.setattr(api_doc_mod.importlib, "import_module", lambda name: module)
    monkeypatch.setattr(
        api_doc_mod.inspect,
        "getmembers",
        lambda pkg, pred=None: [
            ("Public", Public),
            ("Other", Other),
            ("_Private", _Private),
        ],
    )

    assert gen._get_package_classes("pkg.mod") == ["Public"]


def test_generate_all_handles_import_error(monkeypatch, tmp_path, capsys):
    gen = APIDocGenerator("missing.pkg", output_dir=str(tmp_path / "out"))
    monkeypatch.setattr(
        api_doc_mod.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError("x")),
    )

    gen.generate_all()

    out = capsys.readouterr().out
    assert "Generating API documentation for missing.pkg..." in out
    assert "Error importing missing.pkg: x" in out


def test_generate_all_success_flow(fake_package, monkeypatch, tmp_path):
    gen = APIDocGenerator(fake_package, output_dir=str(tmp_path / "out"))
    calls = []

    monkeypatch.setattr(gen, "_get_subpackages", lambda pkg: ["subpkg"])
    monkeypatch.setattr(gen, "_get_direct_package_classes", lambda pkg_name: ["BaseThing"])
    monkeypatch.setattr(gen, "_generate_main_file", lambda *args: calls.append(("main", args[3])))
    monkeypatch.setattr(gen, "_generate_subpackage_docs", lambda name: calls.append(("sub", name)))

    gen.generate_all()

    assert ("main", "fakepkg.md") in calls
    assert ("sub", "subpkg") in calls


def test_generate_api_docs_wrapper_and_generate_all_pyadm1_docs(monkeypatch):
    created = []
    generated = []

    class FakeGen:
        def __init__(self, package_name, output_dir, exclude_private=True, package_title=None):
            created.append((package_name, output_dir, exclude_private, package_title))

        def generate_all(self):
            generated.append(True)

    monkeypatch.setattr(api_doc_mod, "APIDocGenerator", FakeGen)
    api_doc_mod.generate_api_docs("out_dir", "pkg.name", "Title")

    assert created[-1] == ("pkg.name", "out_dir", True, "Title")
    assert generated[-1] is True

    wrapper_calls = []
    monkeypatch.setattr(
        api_doc_mod,
        "generate_api_docs",
        lambda out, pkg, title=None: wrapper_calls.append((out, pkg, title)),
    )
    api_doc_mod.generate_all_pyadm1_docs()

    assert len(wrapper_calls) == 5
    assert wrapper_calls[0][1] == "pyadm1.components"
    assert wrapper_calls[-1][1] == "pyadm1.core"


def test_format_docstring_covers_add_blank_line_and_code_stop_branches(tmp_path):
    gen = APIDocGenerator("x.y", output_dir=str(tmp_path / "out"))

    # Covers line 647 (no blank line after section header, code starts immediately)
    txt1 = "Example:\n>>> x = 1\n>>> x"
    out1 = gen._format_docstring(txt1)
    assert "Example:\n\n```python" in out1

    # Covers lines 659-665 (blank line inside code block followed by more code)
    txt2 = "Example:\n\n>>> x = 1\n\n>>> y = 2\nOverview:"
    out2 = gen._format_docstring(txt2)
    assert ">>> y = 2" in out2

    # Covers line 673 (generic top-level section marker stops code block)
    txt3 = "Example:\n\n>>> x = 1\nOverview:\ntext"
    out3 = gen._format_docstring(txt3)
    assert "```" in out3
    assert "Overview:" in out3


def test_format_docstring_stops_code_block_on_blank_line_before_non_code(tmp_path):
    gen = APIDocGenerator("x.y", output_dir=str(tmp_path / "out"))

    txt = "Example:\n\n>>> x = 1\n\nplain text after code"
    out = gen._format_docstring(txt)

    assert "```python" in out
    assert ">>> x = 1" in out
    assert "plain text after code" in out
