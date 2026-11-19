"""Substrates described by the measured parameters of the legacy Gummersbach library.

Covers the optional measured fields of ``SubstrateParams`` (cell-wall analysis,
degradation level, single acids, titrated acid capacity, filtrate COD,
substrate-borne biomass), the legacy library loader and the ``schema:
gummersbach`` file format. The fractionation is checked against hand
calculations and against the two balances it has to keep: total COD and the
degraded share of the volatile solids.
"""

from __future__ import annotations

import dataclasses
from dataclasses import replace
from pathlib import Path

import pytest

from pyadm1 import Feedstock
from pyadm1.components.biological import Digester
from pyadm1.substrates import (
    SubstrateParams,
    gummersbach_substrate,
    load_gummersbach_library,
    load_substrate,
    params_from_measured,
)
from pyadm1.substrates.gummersbach import (
    DEFAULT_LIBRARY,
    fibre_degradability,
    is_manure_class,
    manure_biomass,
)

# Maize silage of the legacy library, in the units of the schema.
MAIZE = {
    "TS": 31.97,
    "VS": 96.25,
    "RF": 21.07,
    "RP": 8.69,
    "RL": 3.68,
    "NDF": 43.64,
    "ADF": 21.86,
    "ADL": 2.15,
    "D_VS": 0.80,
    "pH": 3.93,
    "Sac": 1.18,
    "Snh4": 0.66,
    "TAC": 6.0,
    "T": 13.0,
}

SLOW = ("X_PS_ch", "X_PS_pr", "X_PS_li")
FAST = ("X_PF_ch", "X_PF_pr", "X_PF_li")
HYDROLYSABLE = ("X_S_ch", "X_S_pr", "X_S_li")
ACIDS = ("S_ac", "S_pro", "S_bu", "S_va")
BIOMASS = ("X_su", "X_aa", "X_fa", "X_c4", "X_pro", "X_ac", "X_h2")

LIBRARY_IDS = ["maize", "swinemanure", "greenrye", "grass", "wheat", "gps", "ccm", "futterkalk", "cowmanure", "onions"]


def _measured(**changes: object) -> SubstrateParams:
    raw = dict(MAIZE)
    raw.update(changes)
    raw = {key: value for key, value in raw.items() if value is not None}
    return params_from_measured("test", raw)


def _conc(params: SubstrateParams) -> tuple[dict, float]:
    feedstock = Feedstock(params)
    return feedstock.concentrations, feedstock.density


def _total(conc: dict, keys: tuple[str, ...]) -> float:
    return sum(conc[key] for key in keys)


def _composition_cod(p: SubstrateParams, rho: float) -> float:
    """Organic COD from the composition alone [kgCOD/m³]; the fractionation must not change it."""
    m_ts = p.TS / 1000.0 * rho
    lignin = p.ADL or 0.0
    carbohydrates = 1.0 - p.fRA - p.fRP - p.fRFe - lignin
    return m_ts * (carbohydrates / p.M_Xch + p.fRP / p.M_Xpr + p.fRFe / p.M_Xli + lignin / p.M_Xlig)


def _degraded_vs_share(p: SubstrateParams, conc: dict, rho: float) -> float:
    """Degradable particulate mass per mass of volatile solids."""
    mass = (conc["X_PS_ch"] + conc["X_PF_ch"] + conc["X_S_ch"]) * p.M_Xch
    mass += (conc["X_PS_pr"] + conc["X_PF_pr"] + conc["X_S_pr"]) * p.M_Xpr
    mass += (conc["X_PS_li"] + conc["X_PF_li"] + conc["X_S_li"]) * p.M_Xli
    return mass / (p.TS / 1000.0 * rho * (1.0 - p.fRA))


class TestUnitConversion:
    def test_measured_values_arrive_in_model_units(self) -> None:
        p = _measured()

        assert pytest.approx(319.7) == p.TS  # % FM -> kg/t
        assert p.fRA == pytest.approx(0.0375)  # 1 - VS
        assert (p.fRF, p.fRP, p.fRFe) == pytest.approx((0.2107, 0.0869, 0.0368))
        assert pytest.approx((0.4364, 0.2186, 0.0215)) == (p.NDF, p.ADF, p.ADL)
        assert p.Temp == 13.0
        assert p.D_VS == 0.80

    def test_ammonium_uses_the_molar_mass_of_nh4_like_the_legacy_model(self) -> None:
        p = _measured()

        assert pytest.approx(0.66 * 14.0 / 18.0) == p.NH4
        assert _conc(p)[0]["S_nh4"] == pytest.approx(0.66 / 18.0)

    def test_simba_fields_carry_the_derived_equivalents(self) -> None:
        p = _measured()

        assert p.aXI == pytest.approx(0.20)
        assert p.fOTSrf == pytest.approx(fibre_degradability(43.64, 2.15, 96.25, 0.80))
        assert p.KS43 == 6.0
        assert pytest.approx(1.18) == p.FFS
        assert pytest.approx(Feedstock(p).bmp_theoretical()) == p.BMP
        assert pytest.approx(p.BMP / 0.55) == p.BGP

    def test_pool_of_protein_and_lipids_follows_the_substrate_class(self) -> None:
        crop = _measured(substrate_class="Mais (GPS) (EK I)")
        manure = _measured(substrate_class="Rindergülle (EK II)")

        assert (crop.fsOTS, crop.ffOTS) == (0.0, 1.0)
        assert (manure.fsOTS, manure.ffOTS) == (1.0, 0.0)

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Schweinegülle (EK II)", True),
            ("Schweineg�lle (EK II)", True),
            ("Rinderfestmist (EK II)", True),
            ("Pferdemist (EK II)", True),
            ("Mais (GPS) (EK I)", False),
            ("Miscellaneous", False),
            (None, False),
        ],
    )
    def test_manure_classes(self, text: str | None, expected: bool) -> None:
        assert is_manure_class(text) is expected


class TestCellWallWithDegradationLevel:
    """Koch et al. (2010): the non-degradable VS sit in the cell wall, lignin first."""

    def test_hand_calculation_for_maize(self) -> None:
        p = _measured()
        conc, rho = _conc(p)
        m_ts = 0.3197 * rho

        d = (43.64 - 96.25 * 0.20) / (43.64 - 2.15)
        structural = (0.4364 - 0.0215) * m_ts / p.M_Xch
        non_fibre = (0.9625 - 0.0869 - 0.0368 - 0.4364) * m_ts / p.M_Xch
        lignin = 0.0215 * m_ts / p.M_Xlig

        assert d == pytest.approx(0.588, abs=1e-3)
        assert conc["X_PS_ch"] == pytest.approx(d * structural)
        assert conc["X_PF_ch"] == pytest.approx(non_fibre)
        assert conc["X_PF_pr"] == pytest.approx(0.0869 * m_ts / p.M_Xpr)
        assert conc["X_PF_li"] == pytest.approx(0.0368 * m_ts / p.M_Xli)
        assert conc["X_PS_pr"] == conc["X_PS_li"] == 0.0
        assert conc["X_I"] == pytest.approx(lignin + (1.0 - d) * structural)
        assert conc["S_I"] == 0.0

    def test_lignin_cod_factor_matches_the_legacy_formula(self) -> None:
        # C10.92 H14.24 O5.76: 237.44 g/mol, 11.6 mol O2 per mol
        assert SubstrateParams.M_Xlig == pytest.approx(237.44 / (11.6 * 32.0))

    @pytest.mark.parametrize("substrate_id", LIBRARY_IDS)
    def test_degraded_share_of_vs_equals_d_vs(self, substrate_id: str) -> None:
        p = load_gummersbach_library(manure_age=None)[substrate_id]
        conc, rho = _conc(p)
        if p.fRA >= 1.0:
            pytest.skip("mineral substrate without volatile solids")

        assert _degraded_vs_share(p, conc, rho) == pytest.approx(p.D_VS, rel=1e-9)

    @pytest.mark.parametrize("substrate_id", LIBRARY_IDS)
    def test_total_cod_equals_the_cod_of_the_composition(self, substrate_id: str) -> None:
        p = gummersbach_substrate(substrate_id)
        feedstock = Feedstock(p)
        conc, rho = feedstock.concentrations, feedstock.density

        expected = _composition_cod(p, rho) + _total(conc, ACIDS)

        assert feedstock.total_cod() == pytest.approx(expected, rel=1e-9)

    def test_d_vs_only_moves_cod_between_slow_pool_and_inerts(self) -> None:
        low, high = _conc(_measured(D_VS=0.70))[0], _conc(_measured(D_VS=0.90))[0]

        assert high["X_PS_ch"] > low["X_PS_ch"]
        assert high["X_PF_ch"] == pytest.approx(low["X_PF_ch"])
        assert high["X_PS_ch"] + high["X_I"] == pytest.approx(low["X_PS_ch"] + low["X_I"])

    def test_non_degradable_beyond_the_cell_wall_reaches_the_cell_contents(self) -> None:
        # 60 % of the VS non-degradable, but the cell wall is only 43.64 % TS.
        p = _measured(D_VS=0.40)
        conc, rho = _conc(p)

        assert conc["X_PS_ch"] == 0.0
        assert _degraded_vs_share(p, conc, rho) == pytest.approx(0.40)
        assert Feedstock(p).total_cod() == pytest.approx(_composition_cod(p, rho) + conc["S_ac"])

    def test_lignin_stays_inert_even_if_d_vs_claims_more(self) -> None:
        p = _measured(D_VS=1.0)
        conc, rho = _conc(p)

        assert conc["X_I"] == pytest.approx(0.0215 * 0.3197 * rho / p.M_Xlig)
        assert _degraded_vs_share(p, conc, rho) == pytest.approx(1.0 - 2.15 / 96.25)

    def test_fibre_degradability_is_limited_like_the_pools(self) -> None:
        assert fibre_degradability(43.64, 2.15, 96.25, 0.40) == 0.0
        assert fibre_degradability(43.64, 2.15, 96.25, 1.0) == 1.0
        assert fibre_degradability(0.0, 0.0, 96.25, 0.8) == 0.0

    def test_ndf_above_the_carbohydrates_is_capped(self) -> None:
        """Params built in code bypass the file checks; the pools still stay non-negative."""
        conc, _ = _conc(dataclasses.replace(_measured(), NDF=0.95))

        assert conc["X_PF_ch"] == 0.0
        assert all(value >= 0.0 for value in conc.values())

    def test_a_file_with_more_fibre_than_organic_matter_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="more than the volatile solids"):
            _measured(NDF=95.0)

    def test_detergent_fibres_must_be_nested(self) -> None:
        with pytest.raises(ValueError, match="ADL <= ADF <= NDF"):
            _measured(ADF=50.0)


class TestCellWallWithoutDegradationLevel:
    """Without ``D_VS`` the inert share stays a parameter and lignin is taken out of it."""

    def test_axi_applies_to_everything_but_lignin(self) -> None:
        p = _measured(D_VS=None, aXI=0.10, aSi=0.02)
        conc, rho = _conc(p)
        m_ts = 0.3197 * rho

        structural = (0.4364 - 0.0215) * m_ts / p.M_Xch
        non_lignin = _composition_cod(p, rho) - 0.0215 * m_ts / p.M_Xlig

        assert conc["X_PS_ch"] == pytest.approx(structural * 0.88)
        assert conc["X_I"] == pytest.approx(0.0215 * m_ts / p.M_Xlig + 0.10 * non_lignin)
        assert conc["S_I"] == pytest.approx(0.02 * non_lignin)

    def test_inert_share_is_required(self) -> None:
        with pytest.raises(ValueError, match="'D_VS', 'aXI' or a measured 'BMP'"):
            _measured(D_VS=None)

    def test_crude_fibre_route_derives_axi_from_d_vs(self) -> None:
        p = _measured(NDF=None, ADL=None, ADF=None)

        assert not p.uses_cell_wall_analysis
        assert p.aXI == pytest.approx(0.20)
        assert p.fOTSrf == 1.0


class TestSingleAcids:
    def test_each_acid_enters_its_own_state(self) -> None:
        p = _measured(Sac=2.0, Spro=1.0, Sbu=0.5, Sva=0.25)
        conc, _ = _conc(p)

        assert conc["S_ac"] == pytest.approx(2.0 * 1.0667, rel=1e-3)
        assert conc["S_pro"] == pytest.approx(1.0 * 1.5135, rel=1e-3)
        assert conc["S_bu"] == pytest.approx(0.5 * 1.8182, rel=1e-3)
        assert conc["S_va"] == pytest.approx(0.25 * 2.0392, rel=1e-3)

    def test_ions_follow_the_ph(self) -> None:
        p = _measured(Spro=1.0, pH=4.88)  # pH = pKa of propionic acid
        conc, _ = _conc(p)

        assert conc["S_pro_ion"] == pytest.approx(0.5 * conc["S_pro"], rel=1e-3)

    def test_acids_count_in_cod_and_density(self) -> None:
        plain = Feedstock(_measured(Sac=0.0))
        acidic = Feedstock(_measured(Sac=0.0, Sbu=20.0))

        assert acidic.total_cod() - plain.total_cod() == pytest.approx(
            20.0 / acidic.substrate.M_Sbu
            + _composition_cod(acidic.substrate, acidic.density)
            - _composition_cod(plain.substrate, plain.density)
        )
        assert acidic.density != plain.density
        assert acidic.substrate.vfa_mass == 20.0

    def test_sum_parameter_still_counts_as_acetate(self) -> None:
        p = _measured(Sac=None, FFS=3.0)
        conc, _ = _conc(p)

        assert p.vfa_masses == (3.0, 0.0, 0.0, 0.0)
        assert conc["S_ac"] == pytest.approx(3.0 / p.M_Sac)
        assert conc["S_pro"] == conc["S_bu"] == conc["S_va"] == 0.0


class TestBicarbonateFromAcidCapacity:
    def test_without_acids_the_acid_capacity_is_bicarbonate(self) -> None:
        conc, _ = _conc(_measured(Sac=0.0, pH=7.0, TAC=100.0))

        assert conc["S_hco3_ion"] == pytest.approx(0.100)

    def test_titrated_acid_anions_are_subtracted(self) -> None:
        p = _measured(Sac=12.0, pH=7.1, TAC=212.0)
        conc, _ = _conc(p)

        acetate_kmol = 12.0 / 60.0  # kg/m3 over 60 g/mol
        alpha_start = p.Kaac / (p.Kaac + 10.0**-7.1)
        alpha_end = p.Kaac / (p.Kaac + 10.0**-4.3)

        assert conc["S_hco3_ion"] == pytest.approx(0.212 - acetate_kmol * (alpha_start - alpha_end), rel=1e-6)
        assert conc["S_hco3_ion"] < 0.212 / 2.0

    def test_bicarbonate_is_never_negative(self) -> None:
        conc, _ = _conc(_measured(Sac=30.0, pH=7.0, TAC=50.0))

        assert conc["S_hco3_ion"] == 0.0

    def test_silage_below_the_end_point_keeps_its_value(self) -> None:
        conc, _ = _conc(_measured())  # pH 3.93

        assert conc["S_hco3_ion"] == pytest.approx(0.006)

    def test_ks43_of_native_files_is_taken_directly(self) -> None:
        feedstock = Feedstock("swine_manure")

        assert feedstock.substrate.TAC is None
        assert feedstock.concentrations["S_hco3_ion"] == pytest.approx(feedstock.substrate.KS43 / 1000.0)


class TestSubstrateBorneBiomass:
    def test_legacy_values_for_manure_stored_ten_days(self) -> None:
        bacteria, methanogens = manure_biomass(10.0)

        assert bacteria == pytest.approx(0.2477, abs=1e-4)
        assert methanogens == pytest.approx(2.914e-4, rel=1e-3)
        assert manure_biomass(0.0)[0] == pytest.approx(2.477, abs=1e-3)

    def test_biomass_is_split_like_the_inoculum(self) -> None:
        conc, _ = _conc(_measured(X_bac=2.0, X_meth=1.0))

        assert [conc[key] for key in BIOMASS] == pytest.approx([0.5, 0.5, 0.3, 0.4, 0.3, 0.8, 0.2])

    def test_biomass_does_not_change_the_total_cod(self) -> None:
        without, with_biomass = Feedstock(_measured()), Feedstock(_measured(X_bac=2.0, X_meth=1.0))

        assert with_biomass.total_cod() == pytest.approx(without.total_cod())
        assert _total(with_biomass.concentrations, SLOW + FAST) == pytest.approx(
            _total(without.concentrations, SLOW + FAST) - 3.0
        )

    def test_library_feeds_biomass_only_with_manure(self) -> None:
        library = load_gummersbach_library()

        assert library["swinemanure"].X_bac == pytest.approx(0.2477, abs=1e-4)
        assert library["cowmanure"].X_meth > 0.0
        assert library["maize"].X_bac == library["maize"].X_meth == 0.0

    def test_biomass_can_be_switched_off(self) -> None:
        assert load_gummersbach_library(manure_age=None)["swinemanure"].X_bac == 0.0

    def test_native_substrates_carry_no_biomass(self) -> None:
        conc = Feedstock("swine_manure").concentrations

        assert _total(conc, BIOMASS) == 0.0


class TestFiltrateCodAndSolubleInert:
    def test_filtrate_cod_beyond_acids_and_inerts_is_hydrolysable(self) -> None:
        base, _ = _conc(_measured())
        conc, _ = _conc(_measured(COD_S=41.0, SIin=10.0))

        moved = 41.0 - conc["S_ac"] - 10.0
        assert _total(conc, HYDROLYSABLE) == pytest.approx(moved)
        assert _total(conc, SLOW + FAST) == pytest.approx(_total(base, SLOW + FAST) - moved)

    def test_classes_keep_their_proportion(self) -> None:
        base, _ = _conc(_measured())
        conc, _ = _conc(_measured(COD_S=41.0))

        share = _total(conc, HYDROLYSABLE) / _total(base, SLOW + FAST)
        assert conc["X_S_pr"] == pytest.approx(share * (base["X_PS_pr"] + base["X_PF_pr"]))
        assert conc["X_PS_ch"] / conc["X_PF_ch"] == pytest.approx(base["X_PS_ch"] / base["X_PF_ch"])

    def test_filtrate_cod_below_the_acids_moves_nothing(self) -> None:
        conc, _ = _conc(_measured(COD_S=0.5))

        assert _total(conc, HYDROLYSABLE) == 0.0

    def test_soluble_inert_comes_out_of_the_inert_cod(self) -> None:
        base, _ = _conc(_measured())
        conc, _ = _conc(_measured(SIin=10.0))

        assert conc["S_I"] == 10.0
        assert conc["X_I"] == pytest.approx(base["X_I"] - 10.0)

    def test_soluble_inert_is_capped_at_the_inert_cod(self) -> None:
        base, _ = _conc(_measured())
        conc, _ = _conc(_measured(SIin=1.0e6))

        assert conc["S_I"] == pytest.approx(base["X_I"])
        assert conc["X_I"] == 0.0

    def test_more_than_the_pools_hold_is_never_moved(self) -> None:
        feedstock = Feedstock(_measured(COD_S=1.0e6, X_bac=1.0e6))
        conc = feedstock.concentrations

        assert _total(conc, SLOW + FAST) == pytest.approx(0.0, abs=1e-9)
        assert feedstock.total_cod() == pytest.approx(Feedstock(_measured()).total_cod())

    def test_inoculum_accounts_for_hydrolysable_influent(self) -> None:
        params = _measured(COD_S=80.0)
        without = Digester("a", Feedstock(replace(params, COD_S=None)), V_liq=1000.0, V_gas=150.0)
        with_filtrate = Digester("b", Feedstock(params), V_liq=1000.0, V_gas=150.0)
        without.initialize({"Q_substrates": [20.0]})
        with_filtrate.initialize({"Q_substrates": [20.0]})

        index = 18  # X_S_ch
        assert with_filtrate.adm1_state[index] > without.adm1_state[index]


class TestLegacyLibrary:
    def test_all_substrates_are_read_in_file_order(self) -> None:
        assert list(load_gummersbach_library()) == LIBRARY_IDS

    def test_names_come_from_the_file(self) -> None:
        assert gummersbach_substrate("maize").name == "Silomais (Gummersbach library)"

    def test_cost_is_converted_to_euro_per_tonne(self) -> None:
        maize = gummersbach_substrate("maize")

        assert maize.cost == pytest.approx(40.0 * 1000.0 / Feedstock(maize).density)

    def test_substrate_id_with_prefix_resolves_everywhere(self) -> None:
        feedstock = Feedstock(["gummersbach:maize", "swine_manure"])

        assert feedstock.substrate_ids == ["gummersbach:maize", "swine_manure"]
        assert feedstock.substrates[0].uses_cell_wall_analysis
        assert Feedstock(feedstock.substrate_ids).densities == feedstock.densities

    def test_unknown_substrate_names_the_available_ones(self) -> None:
        with pytest.raises(KeyError, match="maize"):
            gummersbach_substrate("barley")

    def test_returned_parameters_are_copies(self) -> None:
        first = gummersbach_substrate("maize")
        first.TS = 1.0

        assert pytest.approx(319.7) == gummersbach_substrate("maize").TS

    def test_missing_file_is_reported(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_gummersbach_library(tmp_path / "nothing.xml")

    def test_incomplete_entries_are_skipped(self, tmp_path: Path) -> None:
        text = DEFAULT_LIBRARY.read_text(encoding="utf-8")
        text = text.replace("<substrates>", "<substrates><substrate><name>no id</name></substrate>", 1)
        text = text.replace("<Weender>", '<Weender><physValue symbol="RF"><unit>% TS</unit></physValue>', 1)
        file = tmp_path / "library.xml"
        file.write_text(text, encoding="utf-8")

        assert load_gummersbach_library(file) == load_gummersbach_library()

    def test_wrong_unit_is_rejected(self, tmp_path: Path) -> None:
        text = DEFAULT_LIBRARY.read_text(encoding="utf-8").replace("<unit>% FM</unit>", "<unit>kg/m^3</unit>", 1)
        file = tmp_path / "library.xml"
        file.write_text(text, encoding="utf-8")

        with pytest.raises(ValueError, match="TS is given in 'kg/m\\^3'"):
            load_gummersbach_library(file)


class TestSchemaFiles:
    YAML = """
name: Maize silage, measured
schema: gummersbach
TS: 31.97
VS: 96.25
RF: 21.07
RP: 8.69
RL: 3.68
NDF: 43.64
ADF: 21.86
ADL: 2.15
D_VS: 0.80
pH: 3.93
Sac: 1.18
Snh4: 0.66
TAC: 6.0
T: 13
kdis: 0.14
"""

    def test_yaml_file_equals_the_direct_call(self, tmp_path: Path) -> None:
        file = tmp_path / "maize_measured.yaml"
        file.write_text(self.YAML, encoding="utf-8")

        loaded = load_substrate(file)

        assert loaded == replace(_measured(), name="Maize silage, measured")
        assert Feedstock(file).concentrations == Feedstock(_measured()).concentrations

    def test_toml_file(self, tmp_path: Path) -> None:
        lines = [line for line in self.YAML.strip().splitlines()]
        toml = "\n".join(
            f'{key.strip()} = "{value.strip()}"' if key.strip() in ("name", "schema") else f"{key.strip()} = {value.strip()}"
            for key, value in (line.split(":", 1) for line in lines)
        )
        file = tmp_path / "maize_measured.toml"
        file.write_text(toml, encoding="utf-8")

        assert pytest.approx(0.4364) == load_substrate(file).NDF

    def test_xml_file(self, tmp_path: Path) -> None:
        params = "".join(f'<param name="{key}" value="{value}"/>' for key, value in MAIZE.items())
        file = tmp_path / "maize_measured.xml"
        file.write_text(f'<substrate name="m" schema="gummersbach">{params}</substrate>', encoding="utf-8")

        assert load_substrate(file) == replace(_measured(), name="m")

    def test_bundled_example_file_loads(self) -> None:
        example = DEFAULT_LIBRARY.parent.parent / "examples" / "maize_silage_measured.yaml"

        assert load_substrate(example).uses_cell_wall_analysis

    def test_unknown_schema_is_rejected(self, tmp_path: Path) -> None:
        file = tmp_path / "x.yaml"
        file.write_text("schema: simba9\nTS: 30\n", encoding="utf-8")

        with pytest.raises(ValueError, match="Unknown substrate schema"):
            load_substrate(file)

    def test_misspelled_parameter_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="ADl"):
            params_from_measured("x", {**MAIZE, "ADl": 2.0})

    @pytest.mark.parametrize("key", ["TS", "VS", "RP", "RL", "pH"])
    def test_required_parameters(self, key: str) -> None:
        raw = {k: v for k, v in MAIZE.items() if k != key}

        with pytest.raises(ValueError, match=f"'{key}'"):
            params_from_measured("x", raw)

    def test_crude_fibre_is_required_without_cell_wall_analysis(self) -> None:
        raw = {k: v for k, v in MAIZE.items() if k not in ("RF", "NDF", "ADL")}

        with pytest.raises(ValueError, match="'RF'"):
            params_from_measured("x", raw)

    def test_crude_fibre_is_optional_with_cell_wall_analysis(self) -> None:
        raw = {k: v for k, v in MAIZE.items() if k != "RF"}

        assert Feedstock(params_from_measured("x", raw)).concentrations == Feedstock(_measured()).concentrations

    def test_model_constants_can_be_overridden(self) -> None:
        heavy = params_from_measured("x", {**MAIZE, "roh_CH": 1800.0})

        assert heavy.roh_CH == 1800.0
        assert Feedstock(heavy).density > Feedstock(_measured()).density

    def test_percent_out_of_range_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="between 0 and 100"):
            _measured(TS=319.7)


class TestNativeFields:
    """The measured fields are also available in the native parameter set."""

    def _native(self, **changes: float | None) -> SubstrateParams:
        return replace(Feedstock("legacy:maize_silage_milk_ripeness").substrate, **changes)

    def test_fraction_given_in_percent_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="fraction of TS"):
            self._native(NDF=43.64, ADL=2.15)

    def test_ndf_and_adl_belong_together(self) -> None:
        with pytest.raises(ValueError, match="given together"):
            self._native(NDF=0.44)

    def test_d_vs_out_of_range_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="D_VS"):
            self._native(D_VS=1.2)

    def test_negative_acid_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Spro"):
            self._native(Spro=-1.0)

    def test_native_yaml_takes_the_measured_fields(self, tmp_path: Path) -> None:
        source = (DEFAULT_LIBRARY.parent / "maize_silage_milk_ripeness.yaml").read_text(encoding="utf-8")
        file = tmp_path / "maize.yaml"
        file.write_text(source + "\nNDF: 0.44\nADL: 0.02\nSpro: 0.4\nTAC: null\n", encoding="utf-8")

        loaded = load_substrate(file)

        assert loaded.uses_cell_wall_analysis
        assert loaded.TAC is None
        assert Feedstock(loaded).concentrations["S_pro"] > 0.0

    def test_superseded_native_files_use_the_crude_fibre_route(self) -> None:
        for file in sorted(DEFAULT_LIBRARY.parent.glob("*.yaml")):
            substrate = load_substrate(file)

            assert not substrate.uses_cell_wall_analysis
            assert substrate.D_VS is None and substrate.TAC is None and substrate.COD_S is None
