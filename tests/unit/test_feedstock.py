"""Unit tests for the pure-Python Feedstock class."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyadm1 import Feedstock
from pyadm1.core.adm1 import INFLUENT_COLUMNS
from pyadm1.substrates.feedstock import (
    SubstrateParams,
    SubstrateRegistry,
)
from tests.unit.test_substrate_loaders import write_yaml


def test_substrate_registry_lists_bundled_xml_substrates() -> None:
    reg = SubstrateRegistry()
    available = reg.available()
    assert "maize_silage_milk_ripeness" in available
    assert "swine_manure" in available


def test_load_substrate_xml_returns_dataclass() -> None:
    reg = SubstrateRegistry()
    sub = reg.get("swine_manure")
    assert isinstance(sub, SubstrateParams)
    assert sub.name  # XML carries a human-readable name; just check non-empty
    assert sub.TS > 0.0


class TestFeedstockSingleSubstrate:
    def test_single_substrate_construction_via_id(self) -> None:
        fs = Feedstock("maize_silage_milk_ripeness", feeding_freq=24, total_simtime=10)

        assert isinstance(fs.substrate, SubstrateParams)
        assert fs.density > 0.0

    def test_get_influent_dataframe_returns_correct_columns(self) -> None:
        fs = Feedstock("maize_silage_milk_ripeness", feeding_freq=24, total_simtime=5)
        df = fs.get_influent_dataframe(Q=15.0)

        assert list(df.columns) == INFLUENT_COLUMNS
        assert len(df) == 5  # 5 days × 24 h / 24 h-step
        assert df["Q"].iloc[0] > 0.0


class TestFeedstockMultiSubstrate:
    def test_construction_with_substrate_id_list(self) -> None:
        fs = Feedstock(
            ["maize_silage_milk_ripeness", "swine_manure"],
            feeding_freq=24,
            total_simtime=5,
        )
        assert len(fs.substrates) == 2

    def test_blended_density_is_weighted_average(self) -> None:
        fs = Feedstock(
            ["maize_silage_milk_ripeness", "swine_manure"],
            feeding_freq=24,
            total_simtime=5,
        )

        # Pure manure → close to 1000 kg/m³
        rho_pure_manure = fs.blended_density([0.0, 5.0, 0, 0, 0, 0, 0, 0, 0, 0])
        assert rho_pure_manure == pytest.approx(1000.0, abs=10.0)

        # Pure maize → > 1000 kg/m³
        rho_pure_maize = fs.blended_density([5.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0])
        assert rho_pure_maize > 1000.0

    def test_get_influent_dataframe_with_padded_q(self) -> None:
        fs = Feedstock(
            ["maize_silage_milk_ripeness", "swine_manure"],
            feeding_freq=24,
            total_simtime=5,
        )
        Q = [11.4, 6.1, 0, 0, 0, 0, 0, 0, 0, 0]
        df = fs.get_influent_dataframe(Q=Q)

        assert isinstance(df, pd.DataFrame)
        assert df["Q"].iloc[0] > 0.0  # blended actual flow

    def test_get_influent_dataframe_rejects_extra_nonzero_q(self) -> None:
        fs = Feedstock(["maize_silage_milk_ripeness"], feeding_freq=24, total_simtime=5)
        with pytest.raises(ValueError, match="non-zero"):
            fs.get_influent_dataframe(Q=[5.0, 5.0])

    def test_actual_q_applies_simba_conversion(self) -> None:
        fs = Feedstock("maize_silage_milk_ripeness", feeding_freq=24, total_simtime=5)
        # SIMBA# convention: actual Q = input Q × 1000 / ρ_FM (so for ρ > 1000, actual < input)
        actual = fs.actual_Q(11.4)
        assert actual[0] < 11.4

    def test_actual_q_unchanged_when_simba_convention_off(self) -> None:
        fs = Feedstock(
            "maize_silage_milk_ripeness",
            feeding_freq=24,
            total_simtime=5,
            simba_q_convention=False,
        )
        actual = fs.actual_Q(11.4)
        assert actual[0] == pytest.approx(11.4)


class TestFeedstockHelpers:
    def test_simtime_array_step_matches_feeding_freq(self) -> None:
        fs = Feedstock("swine_manure", feeding_freq=24, total_simtime=4)
        np.testing.assert_allclose(fs.simtime(), np.array([0.0, 1.0, 2.0, 3.0]))

    def test_header_returns_influent_columns(self) -> None:
        fs = Feedstock("swine_manure", feeding_freq=24)
        assert fs.header() == INFLUENT_COLUMNS

    def test_total_cod_is_positive(self) -> None:
        fs = Feedstock("maize_silage_milk_ripeness")
        assert fs.total_cod(0) > 0.0

    def test_vs_content_is_positive(self) -> None:
        fs = Feedstock("maize_silage_milk_ripeness")
        assert fs.vs_content(0) > 0.0

    def test_densities_are_returned_as_a_detached_copy(self) -> None:
        fs = Feedstock(["maize_silage_milk_ripeness", "swine_manure"], feeding_freq=24, total_simtime=2)

        densities = fs.densities
        assert len(densities) == 2
        assert all(d > 0.0 for d in densities)

        densities[0] = -1.0  # mutating the returned list must not corrupt the feedstock
        assert fs.densities[0] > 0.0

    def test_theoretical_bmp_is_positive_for_a_real_substrate(self) -> None:
        fs = Feedstock("maize_silage_milk_ripeness")

        assert fs.bmp_theoretical(0) > 0.0

    def test_theoretical_bmp_is_zero_without_volatile_solids(self) -> None:
        """An ash-only substrate has no VS, so a yield per t VS is undefined -> 0."""
        base = SubstrateRegistry().get("maize_silage_milk_ripeness")
        ash_only = replace(base, fRF=0.0, fRP=0.0, fRFe=0.0, fRA=1.0)
        fs = Feedstock([ash_only], feeding_freq=24, total_simtime=2)

        assert fs.vs_content(0) == pytest.approx(0.0)
        assert fs.bmp_theoretical(0) == 0.0


class TestFeedstockSubstrateResolution:
    """Substrates may be given as registry ids, file paths, or ready-made objects."""

    def test_a_substrate_params_object_is_used_directly(self) -> None:
        params = SubstrateRegistry().get("swine_manure")

        fs = Feedstock([params], feeding_freq=24, total_simtime=2)

        assert fs.substrate_ids == [params.name]
        assert fs.densities == [params.roh_H2O] or fs.densities[0] > 0.0

    def test_an_explicit_file_path_is_loaded(self, tmp_path: Path) -> None:
        path = write_yaml(tmp_path / "from_path.yaml", name="From Path")

        fs = Feedstock([str(path)], feeding_freq=24, total_simtime=2)

        assert fs.substrate_ids == ["from_path"]  # the file stem identifies the input
        assert fs.total_cod(0) > 0.0

    def test_objects_ids_and_paths_can_be_mixed(self, tmp_path: Path) -> None:
        params = SubstrateRegistry().get("swine_manure")
        path = write_yaml(tmp_path / "extra.yaml", name="Extra")

        fs = Feedstock([params, "maize_silage_milk_ripeness", str(path)], feeding_freq=24, total_simtime=2)

        assert fs.substrate_ids == [params.name, "maize_silage_milk_ripeness", "extra"]
        assert len(fs.densities) == 3

    def test_an_unknown_substrate_id_is_reported(self) -> None:
        with pytest.raises((KeyError, ValueError, FileNotFoundError)):
            Feedstock(["definitely_not_a_substrate"], feeding_freq=24, total_simtime=2)


class TestFeedstockDefaultSubstrateSet:
    """``Feedstock()`` without arguments loads the whole bundled substrate library."""

    def test_omitting_substrates_loads_the_bundled_library(self) -> None:
        fs = Feedstock(feeding_freq=24, total_simtime=2)

        assert len(fs.substrate_ids) == len(SubstrateRegistry().available())
        assert "swine_manure" in fs.substrate_ids

    def test_an_empty_substrate_directory_is_reported(self, tmp_path: Path, monkeypatch) -> None:
        import pyadm1.substrates.feedstock as feedstock_mod

        monkeypatch.setattr(feedstock_mod, "_DEFAULT_DATA_DIR", tmp_path)

        with pytest.raises(ValueError, match="No substrate files found in"):
            Feedstock(feeding_freq=24, total_simtime=2)


class TestFeedstockFlowHandling:
    def test_a_short_flow_vector_is_zero_padded(self) -> None:
        fs = Feedstock(["maize_silage_milk_ripeness", "swine_manure"], feeding_freq=24, total_simtime=2)

        # Only the first substrate is dosed; the second must default to zero flow.
        padded = fs.actual_Q([11.4])

        assert len(padded) == 2
        assert padded[0] > 0.0
        assert padded[1] == 0.0

    def test_a_substrate_with_zero_flow_does_not_enter_the_blend(self) -> None:
        fs = Feedstock(["maize_silage_milk_ripeness", "swine_manure"], feeding_freq=24, total_simtime=2)

        only_first = fs.get_influent_dataframe(Q=[11.4, 0.0]).iloc[0]
        both = fs.get_influent_dataframe(Q=[11.4, 6.1]).iloc[0]
        single = Feedstock(["maize_silage_milk_ripeness"], feeding_freq=24, total_simtime=2)
        alone = single.get_influent_dataframe(Q=[11.4]).iloc[0]

        liquid_cols = [c for c in INFLUENT_COLUMNS if c != "Q"]
        # Zero-flow substrate contributes nothing: the blend equals the single-substrate case.
        for col in liquid_cols:
            assert only_first[col] == pytest.approx(alone[col])
        # ...and adding real flow of the second substrate does change the mix.
        assert any(only_first[col] != pytest.approx(both[col]) for col in liquid_cols)
        assert fs.blended_density([11.4, 0.0]) == pytest.approx(fs.densities[0])

    def test_a_fully_zero_flow_yields_an_all_zero_influent_row(self) -> None:
        fs = Feedstock(["maize_silage_milk_ripeness", "swine_manure"], feeding_freq=24, total_simtime=2)

        row = fs.get_influent_dataframe(Q=[0.0, 0.0]).iloc[0]

        assert all(row[col] == 0.0 for col in INFLUENT_COLUMNS if col != "Q")
