"""Unit tests for the substrate file loaders and the substrate registry.

All three on-disk formats (YAML, XML, TOML) feed the same
``_build_substrate_params`` path, so the tests are written as a cross-format
equivalence plus the per-format error handling.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pyadm1.substrates.feedstock import (
    SubstrateRegistry,
    load_substrate,
    load_substrate_toml,
    load_substrate_xml,
    load_substrate_yaml,
)

#: Every parameter that ``SubstrateParams`` has no default for.
_REQUIRED_PARAMS = {
    "TS": 300.0,
    "NH4": 1.5,
    "BGP": 600.0,
    "BMP": 320.0,
    "aXI": 0.2,
    "fOTSrf": 0.7,
    "fsOTS": 1.0,
    "ffOTS": 0.0,
    "aSi": 0.05,
    "fRF": 0.20,
    "fRP": 0.08,
    "fRFe": 0.03,
    "fRA": 0.04,
    "Temp": 20.0,
    "pH": 4.0,
    "KS43": 30.0,
    "FFS": 8.0,
}


def write_yaml(path: Path, name: str | None = "Test Silage") -> Path:
    lines = [] if name is None else [f"name: {name}"]
    lines += [f"{key}: {value}" for key, value in _REQUIRED_PARAMS.items()]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_xml(path: Path, name: str = "Test Silage") -> Path:
    params = "\n".join(f'  <param name="{k}" value="{v}"/>' for k, v in _REQUIRED_PARAMS.items())
    # Malformed entries must be skipped rather than abort the parse.
    params += '\n  <param name="no_value"/>\n  <param value="42"/>'
    path.write_text(
        f'<?xml version="1.0"?>\n<substrate name="{name}">\n{params}\n</substrate>',
        encoding="utf-8",
    )
    return path


def write_toml(path: Path, name: str = "Test Silage") -> Path:
    lines = [f'name = "{name}"'] + [f"{key} = {value}" for key, value in _REQUIRED_PARAMS.items()]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


class TestCrossFormatEquivalence:
    def test_all_three_formats_describe_the_same_substrate(self, tmp_path: Path) -> None:
        from_yaml = load_substrate(write_yaml(tmp_path / "s.yaml"))
        from_xml = load_substrate(write_xml(tmp_path / "s.xml"))
        from_toml = load_substrate(write_toml(tmp_path / "s.toml"))

        assert from_yaml == from_xml == from_toml
        assert from_yaml.name == "Test Silage"
        total_solids, vfa = from_yaml.TS, from_yaml.FFS
        assert total_solids == pytest.approx(300.0)
        assert vfa == pytest.approx(8.0)

    def test_defaults_fill_in_the_unspecified_parameters(self, tmp_path: Path) -> None:
        substrate = load_substrate(write_yaml(tmp_path / "s.yaml"))

        # The component densities are not in the file but have model defaults.
        assert substrate.roh_CH == pytest.approx(1550.0)
        assert substrate.roh_H2O == pytest.approx(1000.0)

    def test_the_yml_extension_is_recognised(self, tmp_path: Path) -> None:
        total_solids = load_substrate(write_yaml(tmp_path / "s.yml")).TS
        assert total_solids == pytest.approx(300.0)

    def test_the_extension_is_matched_case_insensitively(self, tmp_path: Path) -> None:
        write_toml(tmp_path / "s.TOML")

        assert load_substrate(tmp_path / "s.TOML").name == "Test Silage"

    def test_a_missing_name_falls_back_to_the_file_stem(self, tmp_path: Path) -> None:
        assert load_substrate(write_yaml(tmp_path / "fallback_name.yaml", name=None)).name == "fallback_name"

    def test_xml_entries_without_a_name_or_value_are_ignored(self, tmp_path: Path) -> None:
        """The XML writer above emits two malformed <param> elements on purpose."""
        total_solids = load_substrate_xml(write_xml(tmp_path / "s.xml")).TS
        assert total_solids == pytest.approx(300.0)


class TestLoaderErrors:
    @pytest.mark.parametrize(
        ("suffix", "loader"),
        [(".yaml", load_substrate_yaml), (".xml", load_substrate_xml), (".toml", load_substrate_toml)],
    )
    def test_a_missing_file_is_reported_per_format(self, tmp_path: Path, suffix: str, loader) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            loader(tmp_path / f"nope{suffix}")

    def test_an_unsupported_extension_lists_the_supported_ones(self, tmp_path: Path) -> None:
        path = tmp_path / "s.json"
        path.write_text("{}", encoding="utf-8")

        with pytest.raises(ValueError, match="Unsupported substrate file extension") as excinfo:
            load_substrate(path)

        message = str(excinfo.value)
        assert ".yaml" in message and ".xml" in message and ".toml" in message

    def test_a_yaml_that_is_not_a_mapping_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "list.yaml"
        path.write_text("- just\n- a\n- list\n", encoding="utf-8")

        with pytest.raises(ValueError, match="must be a mapping at the top level"):
            load_substrate_yaml(path)

    def test_an_empty_yaml_reports_the_first_missing_parameter(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.yaml"
        path.write_text("", encoding="utf-8")

        with pytest.raises(ValueError, match=r"Required parameter 'TS' not found in empty\.yaml"):
            load_substrate_yaml(path)

    def test_an_incomplete_definition_names_the_parameter_and_the_file(self, tmp_path: Path) -> None:
        path = tmp_path / "partial.yaml"
        path.write_text("name: Partial\nTS: 300.0\n", encoding="utf-8")

        with pytest.raises(ValueError, match=r"Required parameter 'NH4' not found in partial\.yaml"):
            load_substrate(path)


class TestSubstrateRegistry:
    def test_a_custom_directory_is_discovered(self, tmp_path: Path) -> None:
        write_yaml(tmp_path / "my_silage.yaml", name="My Silage")

        registry = SubstrateRegistry(data_dir=tmp_path)

        assert registry.available() == ["my_silage"]
        assert registry.get("my_silage").name == "My Silage"

    def test_all_supported_formats_are_discovered_side_by_side(self, tmp_path: Path) -> None:
        write_yaml(tmp_path / "a.yaml")
        write_xml(tmp_path / "b.xml")
        write_toml(tmp_path / "c.toml")

        assert SubstrateRegistry(data_dir=tmp_path).available() == ["a", "b", "c"]

    def test_the_legacy_xml_dir_alias_still_works(self, tmp_path: Path) -> None:
        write_xml(tmp_path / "legacy.xml", name="Legacy")

        assert SubstrateRegistry(xml_dir=tmp_path).get("legacy").name == "Legacy"

    def test_passing_both_directory_arguments_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="accepts 'data_dir' or 'xml_dir', not both"):
            SubstrateRegistry(data_dir=tmp_path, xml_dir=tmp_path)

    def test_a_repeated_lookup_is_served_from_the_cache(self, tmp_path: Path) -> None:
        path = write_yaml(tmp_path / "cached.yaml")
        registry = SubstrateRegistry(data_dir=tmp_path)

        first = registry.get("cached")
        path.unlink()  # the file is gone; only the cache can answer now

        assert registry.get("cached") is first
