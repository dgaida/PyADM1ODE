# -*- coding: utf-8 -*-
"""Unit tests for the pure-Python Feedstock class."""

import numpy as np
import pandas as pd
import pytest

from pyadm1 import Feedstock
from pyadm1.core.adm1 import INFLUENT_COLUMNS
from pyadm1.substrates.feedstock import (
    SubstrateParams,
    SubstrateRegistry,
)


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
        fs = Feedstock(["maize_silage_milk_ripeness", "swine_manure"], feeding_freq=24, total_simtime=5)
        assert len(fs.substrates) == 2

    def test_blended_density_is_weighted_average(self) -> None:
        fs = Feedstock(["maize_silage_milk_ripeness", "swine_manure"], feeding_freq=24, total_simtime=5)

        # Pure manure → close to 1000 kg/m³
        rho_pure_manure = fs.blended_density([0.0, 5.0, 0, 0, 0, 0, 0, 0, 0, 0])
        assert rho_pure_manure == pytest.approx(1000.0, abs=10.0)

        # Pure maize → > 1000 kg/m³
        rho_pure_maize = fs.blended_density([5.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0])
        assert rho_pure_maize > 1000.0

    def test_get_influent_dataframe_with_padded_q(self) -> None:
        fs = Feedstock(["maize_silage_milk_ripeness", "swine_manure"], feeding_freq=24, total_simtime=5)
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
