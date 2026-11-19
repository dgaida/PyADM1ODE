"""The bundled substrate library and the three parameters added with it.

* The inert share fitted to a measured methane potential (``BMP``).
* The protein bound in the cell wall (``NDICP``), which enters the slow pool.
* The remuneration class ``substrate_class``.

The library files themselves are written in the measured schema; the files
they replace live on in ``data/substrates/legacy`` and are reachable as
``legacy:<id>``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pyadm1 import Feedstock
from pyadm1.substrates import (
    DEFAULT_BMP_RECOVERY,
    SubstrateParams,
    SubstrateRegistry,
    fit_degradability_to_bmp,
    load_substrate,
    params_from_measured,
)
from pyadm1.substrates.feedstock import _DEFAULT_DATA_DIR, _DEFAULT_SUBSTRATE_ORDER

LIBRARY_IDS = list(_DEFAULT_SUBSTRATE_ORDER)

# A maize silage with cell-wall analysis, in the units of the measured schema.
SILAGE = {
    "TS": 32.0,
    "VS": 95.0,
    "RP": 8.0,
    "RL": 3.0,
    "NDF": 45.0,
    "ADL": 3.0,
    "pH": 4.0,
    "Sac": 2.0,
}

# A slurry without cell-wall analysis: crude-fibre route.
SLURRY = {"TS": 8.0, "VS": 80.0, "RF": 18.0, "RP": 14.0, "RL": 4.0, "pH": 7.2, "fOTSrf": 0.7, "aSi": 0.05}

NATIVE_YAML = """\
name: Native silage
TS: 330.0
NH4: 0.8
BGP: 650.0
BMP: 340.0
{axi}
fOTSrf: 0.9
fsOTS: 0.0
ffOTS: 1.0
aSi: 0.0
fRF: 0.22
fRP: 0.08
fRFe: 0.03
fRA: 0.05
Temp: 15.0
pH: 4.0
KS43: 0.0
FFS: 2.0
{extra}
"""


def _measured(base: dict, **changes: object) -> SubstrateParams:
    raw = {**base, **changes}
    return params_from_measured("test", {key: value for key, value in raw.items() if value is not None})


def _potential(params: SubstrateParams) -> float:
    return Feedstock(params).bmp_theoretical()


def _pools(params: SubstrateParams) -> dict:
    return Feedstock(params).concentrations


class TestInertShareFromMethanePotential:
    def test_cell_wall_route_fits_the_degradation_level(self) -> None:
        p = _measured(SILAGE, BMP=330.0)

        assert 0.0 < p.D_VS < 1.0
        assert _potential(p) == pytest.approx(330.0, rel=1e-9)
        assert p.BMP == 330.0

    def test_unused_simba_fields_follow_the_fitted_level(self) -> None:
        p = _measured(SILAGE, BMP=330.0)

        assert p.aXI == pytest.approx(1.0 - p.D_VS)
        assert 0.0 < p.fOTSrf < 1.0

    def test_crude_fibre_route_fits_the_inert_share(self) -> None:
        p = _measured(SLURRY, BMP=210.0)

        assert p.D_VS is None
        assert 0.0 < p.aXI < 1.0
        assert _potential(p) == pytest.approx(210.0, rel=1e-9)

    def test_higher_potential_means_less_inert_matter(self) -> None:
        assert _measured(SILAGE, BMP=340.0).D_VS > _measured(SILAGE, BMP=300.0).D_VS
        assert _measured(SLURRY, BMP=250.0).aXI < _measured(SLURRY, BMP=200.0).aXI

    def test_recovery_raises_the_model_potential(self) -> None:
        p = _measured(SILAGE, BMP=300.0, BMP_recovery=0.9)

        assert _potential(p) == pytest.approx(300.0 / 0.9, rel=1e-9)

    def test_default_recovery_matches_the_potential_directly(self) -> None:
        assert DEFAULT_BMP_RECOVERY == 1.0

    def test_degradation_level_wins_over_the_potential(self) -> None:
        p = _measured(SILAGE, BMP=330.0, D_VS=0.7)

        level, potential = p.D_VS, p.BMP
        assert level == pytest.approx(0.7)
        assert potential == pytest.approx(330.0)
        assert _potential(p) != pytest.approx(330.0, rel=1e-3)

    def test_inert_share_wins_over_the_potential(self) -> None:
        assert _measured(SLURRY, BMP=210.0, aXI=0.3).aXI == 0.3

    def test_without_a_measured_potential_the_field_holds_the_model_value(self) -> None:
        p = _measured(SILAGE, D_VS=0.8)

        methane, biogas = p.BMP, p.BGP
        assert methane == pytest.approx(_potential(p))
        assert biogas == pytest.approx(methane / 0.55)

    def test_potential_out_of_reach_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="composition allows"):
            _measured(SILAGE, BMP=600.0)

    def test_function_takes_an_explicit_potential(self) -> None:
        base = _measured(SILAGE, D_VS=0.5)

        fitted = fit_degradability_to_bmp(base, bmp=320.0)

        assert _potential(fitted) == pytest.approx(320.0, rel=1e-9)
        assert base.D_VS == 0.5  # the input is left alone

    @pytest.mark.parametrize("recovery", [0.0, -0.5, 1.2])
    def test_recovery_out_of_range_is_rejected(self, recovery: float) -> None:
        with pytest.raises(ValueError, match="recovery"):
            fit_degradability_to_bmp(_measured(SILAGE, D_VS=0.8), bmp=300.0, recovery=recovery)

    def test_non_positive_potential_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            fit_degradability_to_bmp(_measured(SILAGE, D_VS=0.8), bmp=0.0)

    def test_native_file_without_inert_share_is_fitted(self, tmp_path: Path) -> None:
        file = tmp_path / "silage.yaml"
        file.write_text(NATIVE_YAML.format(axi="", extra=""), encoding="utf-8")

        p = load_substrate(file)

        assert 0.0 < p.aXI < 1.0
        assert _potential(p) == pytest.approx(340.0, rel=1e-9)

    def test_native_file_takes_a_recovery(self, tmp_path: Path) -> None:
        file = tmp_path / "silage.yaml"
        file.write_text(NATIVE_YAML.format(axi="", extra="BMP_recovery: 0.95"), encoding="utf-8")

        assert _potential(load_substrate(file)) == pytest.approx(340.0 / 0.95, rel=1e-9)

    def test_native_file_with_inert_share_keeps_it(self, tmp_path: Path) -> None:
        file = tmp_path / "silage.yaml"
        file.write_text(NATIVE_YAML.format(axi="aXI: 0.25", extra=""), encoding="utf-8")

        assert load_substrate(file).aXI == 0.25

    def test_native_file_without_inert_share_and_potential_is_rejected(self, tmp_path: Path) -> None:
        file = tmp_path / "silage.yaml"
        file.write_text(NATIVE_YAML.format(axi="", extra="").replace("BMP: 340.0\n", ""), encoding="utf-8")

        with pytest.raises(ValueError, match="Required parameter"):
            load_substrate(file)


class TestProteinBoundInTheCellWall:
    def test_bound_protein_enters_the_slow_pool(self) -> None:
        c = _pools(_measured(SILAGE, D_VS=0.8, NDICP=2.0))

        assert c["X_PS_pr"] / (c["X_PS_pr"] + c["X_PF_pr"]) == pytest.approx(2.0 / 8.0)

    def test_without_the_measurement_the_class_decides(self) -> None:
        c = _pools(_measured(SILAGE, D_VS=0.8))

        assert c["X_PS_pr"] == 0.0 and c["X_PF_pr"] > 0.0

    def test_bound_protein_is_taken_out_of_the_structural_carbohydrates(self) -> None:
        """It is weighed with the NDF, so the non-fibre carbohydrates grow by the same mass."""
        plain, bound = _measured(SILAGE, D_VS=1.0), _measured(SILAGE, D_VS=1.0, NDICP=2.0)
        c_plain, c_bound = _pools(plain), _pools(bound)
        mass = 0.02 * plain.TS / 1000.0 * Feedstock(plain).density / plain.M_Xch  # kgCOD/m³ of 2 % TS carbohydrate

        assert c_plain["X_PS_ch"] - c_bound["X_PS_ch"] == pytest.approx(mass)
        assert c_bound["X_PF_ch"] - c_plain["X_PF_ch"] == pytest.approx(mass)

    def test_protein_cod_is_unchanged(self) -> None:
        c_plain, c_bound = _pools(_measured(SILAGE, D_VS=0.8)), _pools(_measured(SILAGE, D_VS=0.8, NDICP=2.0))

        assert c_bound["X_PS_pr"] + c_bound["X_PF_pr"] == pytest.approx(c_plain["X_PS_pr"] + c_plain["X_PF_pr"])

    def test_lipids_keep_following_the_class(self) -> None:
        c = _pools(_measured(SILAGE, D_VS=0.8, NDICP=2.0, substrate_class="Rinderfestmist (EK II)"))

        assert c["X_PF_li"] == 0.0 and c["X_PS_li"] > 0.0
        assert c["X_PF_pr"] > 0.0  # the protein follows the measurement

    def test_bound_protein_cannot_exceed_the_protein(self) -> None:
        c = _pools(_measured(SILAGE, D_VS=0.8, NDICP=20.0))

        assert c["X_PF_pr"] == pytest.approx(0.0)
        assert c["X_PS_pr"] > 0.0

    def test_substrate_without_protein_is_handled(self) -> None:
        c = _pools(_measured(SILAGE, D_VS=0.8, RP=0.0, NDICP=1.0))

        assert c["X_PS_pr"] == 0.0 and c["X_PF_pr"] == 0.0

    def test_needs_the_cell_wall_analysis(self) -> None:
        with pytest.raises(ValueError, match="needs 'NDF' and 'ADL'"):
            _measured(SLURRY, aXI=0.3, NDICP=2.0)

    def test_native_field_is_a_fraction(self) -> None:
        base = _measured(SILAGE, D_VS=0.8)

        with pytest.raises(ValueError, match="fraction of TS"):
            replace(base, NDICP=2.0)
        with pytest.raises(ValueError, match="needs NDF and ADL"):
            replace(base, NDF=None, ADL=None, NDICP=0.02)


class TestSubstrateClass:
    def test_measured_schema_stores_the_class(self) -> None:
        assert _measured(SILAGE, D_VS=0.8, substrate_class="Mais (GPS) (EK I)").substrate_class == "Mais (GPS) (EK I)"

    def test_class_is_optional(self) -> None:
        assert _measured(SILAGE, D_VS=0.8).substrate_class is None

    def test_manure_class_puts_protein_and_lipids_into_the_slow_pool(self) -> None:
        p = _measured(SLURRY, aXI=0.3, substrate_class="Rindergülle (EK II)")

        assert (p.fsOTS, p.ffOTS) == (1.0, 0.0)

    def test_native_file_stores_the_class_as_text(self, tmp_path: Path) -> None:
        file = tmp_path / "silage.yaml"
        file.write_text(NATIVE_YAML.format(axi="aXI: 0.2", extra='substrate_class: "Mais (GPS) (EK I)"'), encoding="utf-8")

        assert load_substrate(file).substrate_class == "Mais (GPS) (EK I)"

    def test_class_does_not_change_the_native_calculation(self, tmp_path: Path) -> None:
        plain, classed = tmp_path / "plain.yaml", tmp_path / "classed.yaml"
        plain.write_text(NATIVE_YAML.format(axi="aXI: 0.2", extra=""), encoding="utf-8")
        classed.write_text(NATIVE_YAML.format(axi="aXI: 0.2", extra="substrate_class: Rindergülle"), encoding="utf-8")

        assert Feedstock(classed).concentrations == Feedstock(plain).concentrations


class TestSchemaName:
    @pytest.mark.parametrize("schema", ["measured", "gummersbach", "Measured"])
    def test_both_spellings_select_the_measured_parameters(self, tmp_path: Path, schema: str) -> None:
        file = tmp_path / "silage.yaml"
        lines = [f"schema: {schema}", "D_VS: 0.8"] + [f"{key}: {value}" for key, value in SILAGE.items()]
        file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        assert load_substrate(file).uses_cell_wall_analysis

    def test_unknown_schema_is_rejected(self, tmp_path: Path) -> None:
        file = tmp_path / "silage.yaml"
        file.write_text("schema: simba\nTS: 30\n", encoding="utf-8")

        with pytest.raises(ValueError, match="Unknown substrate schema"):
            load_substrate(file)


class TestSupersededFiles:
    def test_legacy_prefix_loads_the_superseded_file(self) -> None:
        feedstock = Feedstock("legacy:cattle_manure")

        assert feedstock.substrate == load_substrate(_DEFAULT_DATA_DIR / "legacy" / "cattle_manure.yaml")
        assert feedstock.substrate_ids == ["legacy:cattle_manure"]

    def test_superseded_and_current_file_differ(self) -> None:
        assert Feedstock("legacy:cattle_manure").substrate.TS != Feedstock("cattle_manure").substrate.TS

    def test_unknown_legacy_id_is_reported(self) -> None:
        with pytest.raises(FileNotFoundError, match="Legacy substrate 'no_such_substrate'"):
            Feedstock("legacy:no_such_substrate")

    def test_every_library_file_has_a_superseded_version(self) -> None:
        for sid in LIBRARY_IDS:
            assert (_DEFAULT_DATA_DIR / "legacy" / f"{sid}.yaml").exists()

    def test_superseded_files_are_not_part_of_the_registry(self) -> None:
        assert sorted(SubstrateRegistry().available()) == sorted(LIBRARY_IDS)


@pytest.mark.parametrize("sid", LIBRARY_IDS)
class TestBundledLibrary:
    def test_file_uses_the_measured_schema(self, sid: str) -> None:
        text = (_DEFAULT_DATA_DIR / f"{sid}.yaml").read_text(encoding="utf-8")

        assert "\nschema: measured\n" in text

    def test_substrate_has_a_class(self, sid: str) -> None:
        assert "(EK " in Feedstock(sid).substrate.substrate_class

    def test_model_potential_equals_the_methane_potential_of_the_file(self, sid: str) -> None:
        feedstock = Feedstock(sid)

        assert feedstock.substrate.BMP > 0.0
        assert feedstock.bmp_theoretical() == pytest.approx(feedstock.substrate.BMP, rel=1e-6)

    def test_inert_share_is_plausible(self, sid: str) -> None:
        p = Feedstock(sid).substrate

        assert 0.02 < p.aXI < 0.70
        if p.uses_cell_wall_analysis:
            assert p.aXI == pytest.approx(1.0 - p.D_VS)

    def test_composition_adds_up(self, sid: str) -> None:
        p = Feedstock(sid).substrate

        assert 0.0 < p.fRA < 0.35
        assert p.fRA + p.fRP + p.fRFe + (p.NDF if p.uses_cell_wall_analysis else p.fRF) < 1.0


class TestStandardYields:
    """The library reproduces the standard methane yields of the BiomasseV 2012 [m³ CH4/t FM]."""

    @pytest.mark.parametrize(
        ("sid", "standard"),
        [
            ("maize_silage_milk_ripeness", 106.0),
            ("maize_silage_gummersbach", 106.0),
            ("grass_silage", 100.0),
            ("green_rye_silage", 72.0),
            ("corn_cob_mix", 242.0),
            ("cattle_manure", 17.0),
            ("cattle_manure_solid", 53.0),
            ("onion_waste", 40.0),
        ],
    )
    def test_yield_per_fresh_matter(self, sid: str, standard: float) -> None:
        feedstock = Feedstock(sid)
        p = feedstock.substrate
        per_fresh_matter = p.BMP * p.TS / 1000.0 * (1.0 - p.fRA)

        assert per_fresh_matter == pytest.approx(standard, rel=0.06)
