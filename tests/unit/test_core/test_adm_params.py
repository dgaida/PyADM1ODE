"""
Unit tests for ADM1 parameter functions.

This module tests the ADMParams class which provides stoichiometric
and kinetic parameters for the ADM1 model.
"""

import pytest
import numpy as np
from typing import Tuple

from pyadm1.core.adm_params import ADMParams


class TestADMParams:
    """Test suite for ADMParams class."""

    @pytest.fixture
    def standard_conditions(self) -> Tuple[float, float, float]:
        """
        Provide standard temperature and gas constant values.

        Returns:
            Tuple containing (R, T_base, T_ad) where:
                - R: Gas constant [bar·M^-1·K^-1]
                - T_base: Base temperature [K]
                - T_ad: Digester temperature [K]
        """
        R = 0.08314  # bar·M^-1·K^-1
        T_base = 298.15  # K (25°C)
        T_ad = 308.15  # K (35°C)
        return R, T_base, T_ad

    def test_getADMparams_returns_correct_number_of_values(self, standard_conditions: Tuple[float, float, float]) -> None:
        """
        Test that getADMparams returns exactly 87 parameter values.

        Args:
            standard_conditions: Fixture providing R, T_base, T_ad values.
        """
        R, T_base, T_ad = standard_conditions
        params = ADMParams.getADMparams(R, T_base, T_ad)

        assert len(params) == 87, "getADMparams should return 87 parameters"

    def test_getADMparams_all_positive_values(self, standard_conditions: Tuple[float, float, float]) -> None:
        """
        Test that all returned parameters are positive numbers.

        Args:
            standard_conditions: Fixture providing R, T_base, T_ad values.
        """
        R, T_base, T_ad = standard_conditions
        params = ADMParams.getADMparams(R, T_base, T_ad)

        for i, param in enumerate(params):
            assert param > 0, f"Parameter at index {i} should be positive, got {param}"
            assert np.isfinite(param), f"Parameter at index {i} should be finite"

    def test_getADMparams_stoichiometric_fractions_sum_to_one(self, standard_conditions: Tuple[float, float, float]) -> None:
        """
        Test that stoichiometric fractions from sugars sum to approximately 1.

        Args:
            standard_conditions: Fixture providing R, T_base, T_ad values.
        """
        R, T_base, T_ad = standard_conditions
        params = ADMParams.getADMparams(R, T_base, T_ad)

        # Extract f_h2_su, f_bu_su, f_pro_su, f_ac_su (indices 13-16)
        f_h2_su = params[13]
        f_bu_su = params[14]
        f_pro_su = params[15]
        f_ac_su = params[16]

        fraction_sum = f_h2_su + f_bu_su + f_pro_su + f_ac_su
        assert np.isclose(fraction_sum, 1.0, rtol=1e-6), f"Sugar fractions should sum to 1.0, got {fraction_sum}"

    def test_getADMparams_amino_acid_fractions_sum_to_one(self, standard_conditions: Tuple[float, float, float]) -> None:
        """
        Test that stoichiometric fractions from amino acids sum to approximately 1.

        Args:
            standard_conditions: Fixture providing R, T_base, T_ad values.
        """
        R, T_base, T_ad = standard_conditions
        params = ADMParams.getADMparams(R, T_base, T_ad)

        # Extract f_h2_aa, f_va_aa, f_bu_aa, f_pro_aa, f_ac_aa (indices 23-27)
        f_h2_aa = params[23]
        f_va_aa = params[24]
        f_bu_aa = params[25]
        f_pro_aa = params[26]
        f_ac_aa = params[27]

        fraction_sum = f_h2_aa + f_va_aa + f_bu_aa + f_pro_aa + f_ac_aa
        assert np.isclose(fraction_sum, 1.0, rtol=1e-6), f"Amino acid fractions should sum to 1.0, got {fraction_sum}"

    def test_getADMparams_yields_less_than_one(self, standard_conditions: Tuple[float, float, float]) -> None:
        """
        Test that all yield coefficients are between 0 and 1.

        Args:
            standard_conditions: Fixture providing R, T_base, T_ad values.
        """
        R, T_base, T_ad = standard_conditions
        params = ADMParams.getADMparams(R, T_base, T_ad)

        # Yield parameters: Y_su, Y_aa, Y_fa, Y_c4, Y_pro, Y_ac, Y_h2
        # Based on the code, Y_su is at index 22
        yield_indices = [22, 29, 30, 31, 32, 34, 35]

        for idx in yield_indices:
            yield_value = params[idx]
            assert 0 < yield_value < 1, f"Yield at index {idx} should be between 0 and 1, got {yield_value}"

    def test_getADMinhibitionparams_returns_six_values(self) -> None:
        """Test that getADMinhibitionparams returns exactly 6 values."""
        params = ADMParams.getADMinhibitionparams()

        assert len(params) == 6, "getADMinhibitionparams should return 6 values"

    def test_getADMinhibitionparams_positive_values(self) -> None:
        """Test that all inhibition parameters are positive."""
        params = ADMParams.getADMinhibitionparams()

        for i, param in enumerate(params):
            assert param > 0, f"Inhibition parameter at index {i} should be positive"
            assert np.isfinite(param), f"Inhibition parameter at index {i} should be finite"

    def test_getADMgasparams_returns_six_values(self, standard_conditions: Tuple[float, float, float]) -> None:
        """
        Test that getADMgasparams returns exactly 6 values.

        Args:
            standard_conditions: Fixture providing R, T_base, T_ad values.
        """
        R, T_base, T_ad = standard_conditions
        params = ADMParams.getADMgasparams(R, T_base, T_ad)

        assert len(params) == 6, "getADMgasparams should return 6 values"

    def test_getADMgasparams_henry_constants_positive(self, standard_conditions: Tuple[float, float, float]) -> None:
        """
        Test that Henry's law constants are positive.

        Args:
            standard_conditions: Fixture providing R, T_base, T_ad values.
        """
        R, T_base, T_ad = standard_conditions
        p_gas_h2o, k_p, k_L_a, K_H_co2, K_H_ch4, K_H_h2 = ADMParams.getADMgasparams(R, T_base, T_ad)

        assert K_H_co2 > 0, "Henry constant for CO2 should be positive"
        assert K_H_ch4 > 0, "Henry constant for CH4 should be positive"
        assert K_H_h2 > 0, "Henry constant for H2 should be positive"

    def test_getADMgasparams_water_vapor_pressure_reasonable(self, standard_conditions: Tuple[float, float, float]) -> None:
        """
        Test that water vapor pressure is in a reasonable range.

        Args:
            standard_conditions: Fixture providing R, T_base, T_ad values.
        """
        R, T_base, T_ad = standard_conditions
        p_gas_h2o, _, _, _, _, _ = ADMParams.getADMgasparams(R, T_base, T_ad)

        # At 35°C, water vapor pressure should be around 0.05-0.06 bar
        assert 0.03 < p_gas_h2o < 0.08, f"Water vapor pressure seems unreasonable: {p_gas_h2o} bar"

    def test_getADMgasparams_temperature_dependence(self) -> None:
        """Test that gas parameters change with temperature."""
        R = 0.08314
        T_base = 298.15
        T_ad_low = 298.15  # 25°C
        T_ad_high = 323.15  # 50°C

        params_low = ADMParams.getADMgasparams(R, T_base, T_ad_low)
        params_high = ADMParams.getADMgasparams(R, T_base, T_ad_high)

        # Water vapor pressure should increase with temperature
        assert params_high[0] > params_low[0], "Water vapor pressure should increase with temperature"

    def test_getADMKparams_equilibrium_constants(self, standard_conditions: Tuple[float, float, float]) -> None:
        """
        Test that acid-base equilibrium constants are in expected ranges.

        Args:
            standard_conditions: Fixture providing R, T_base, T_ad values.
        """
        R, T_base, T_ad = standard_conditions
        K_w, K_a_va, K_a_bu, K_a_pro, K_a_ac, K_a_co2, K_a_IN = ADMParams._getADMKparams(R, T_base, T_ad)

        # Water dissociation constant should be around 1e-14 at 25°C
        assert 1e-15 < K_w < 1e-13, f"K_w seems unreasonable: {K_w}"

        # Weak acid dissociation constants should be between 1e-10 and 1e-3
        for K_a, name in [
            (K_a_va, "valerate"),
            (K_a_bu, "butyrate"),
            (K_a_pro, "propionate"),
            (K_a_ac, "acetate"),
        ]:
            assert 1e-10 < K_a < 1e-3, f"K_a for {name} seems unreasonable: {K_a}"

    def test_getADMpHULLLparams_order(self) -> None:
        """Test that pH lower limits are less than upper limits."""
        pH_LL_aa, pH_UL_aa, pH_LL_ac, pH_UL_ac, pH_LL_h2, pH_UL_h2 = ADMParams._getADMpHULLLparams()

        assert pH_LL_aa < pH_UL_aa, "pH_LL_aa should be less than pH_UL_aa"
        assert pH_LL_ac < pH_UL_ac, "pH_LL_ac should be less than pH_UL_ac"
        assert pH_LL_h2 < pH_UL_h2, "pH_LL_h2 should be less than pH_UL_h2"

    def test_getADMpHULLLparams_reasonable_ranges(self) -> None:
        """Test that pH limits are in reasonable physiological ranges."""
        pH_LL_aa, pH_UL_aa, pH_LL_ac, pH_UL_ac, pH_LL_h2, pH_UL_h2 = ADMParams._getADMpHULLLparams()

        # All pH values should be between 0 and 14
        for pH_value in [pH_LL_aa, pH_UL_aa, pH_LL_ac, pH_UL_ac, pH_LL_h2, pH_UL_h2]:
            assert 0 < pH_value < 14, f"pH value {pH_value} is outside valid range"

        # Check that ranges make sense for anaerobic digestion (typically pH 4-8)
        assert 3 < pH_LL_aa < 10, "Lower limit for amino acid degraders seems unreasonable"
        assert 4 < pH_UL_aa < 10, "Upper limit for amino acid degraders seems unreasonable"

    def test_getADMk_mK_Sparams_returns_19_values(self) -> None:
        """Test that kinetic parameters function returns 19 values."""
        params = ADMParams._getADMk_mK_Sparams()

        assert len(params) == 19, "getADMk_mK_Sparams should return 19 values"

    def test_getADMk_mK_Sparams_all_positive(self) -> None:
        """Test that all kinetic parameters are positive."""
        params = ADMParams._getADMk_mK_Sparams()

        for i, param in enumerate(params):
            assert param > 0, f"Kinetic parameter at index {i} should be positive"

    def test_getADMYparams_returns_seven_values(self) -> None:
        """Test that yield parameters function returns 7 values."""
        params = ADMParams._getADMYparams()

        assert len(params) == 7, "getADMYparams should return 7 values"

    def test_getADMYparams_all_between_zero_and_one(self) -> None:
        """Test that all yield values are between 0 and 1."""
        Y_su, Y_aa, Y_fa, Y_c4, Y_pro, Y_ac, Y_h2 = ADMParams._getADMYparams()

        for yield_val, name in [
            (Y_su, "Y_su"),
            (Y_aa, "Y_aa"),
            (Y_fa, "Y_fa"),
            (Y_c4, "Y_c4"),
            (Y_pro, "Y_pro"),
            (Y_ac, "Y_ac"),
            (Y_h2, "Y_h2"),
        ]:
            assert 0 < yield_val < 1, f"{name} should be between 0 and 1, got {yield_val}"

    def test_getADMfaaparams_returns_five_values(self) -> None:
        """Test that amino acid fraction parameters return 5 values."""
        params = ADMParams._getADMfaaparams()

        assert len(params) == 5, "getADMfaaparams should return 5 values"

    def test_getADMfaaparams_sum_to_one(self) -> None:
        """Test that amino acid fractions sum to 1."""
        f_h2_aa, f_va_aa, f_bu_aa, f_pro_aa, f_ac_aa = ADMParams._getADMfaaparams()

        fraction_sum = f_h2_aa + f_va_aa + f_bu_aa + f_pro_aa + f_ac_aa
        assert np.isclose(fraction_sum, 1.0, rtol=1e-6), f"Amino acid fractions should sum to 1.0, got {fraction_sum}"

    def test_getADMfsuparams_returns_four_values(self) -> None:
        """Test that sugar fraction parameters return 4 values."""
        params = ADMParams._getADMfsuparams()

        assert len(params) == 4, "getADMfsuparams should return 4 values"

    def test_getADMfsuparams_sum_to_one(self) -> None:
        """Test that sugar fractions sum to 1."""
        f_h2_su, f_bu_su, f_pro_su, f_ac_su = ADMParams._getADMfsuparams()

        fraction_sum = f_h2_su + f_bu_su + f_pro_su + f_ac_su
        assert np.isclose(fraction_sum, 1.0, rtol=1e-6), f"Sugar fractions should sum to 1.0, got {fraction_sum}"


class TestADMParamsEdgeCases:
    """Test edge cases and boundary conditions for ADMParams."""

    def test_extreme_temperatures(self) -> None:
        """Test parameter calculation at extreme but valid temperatures."""
        R = 0.08314

        # Test low temperature (psychrophilic)
        T_base = 298.15
        T_ad_low = 283.15  # 10°C
        params_low = ADMParams.getADMparams(R, T_base, T_ad_low)
        assert all(np.isfinite(params_low)), "Parameters should be finite at low temperature"

        # Test high temperature (thermophilic)
        T_ad_high = 328.15  # 55°C
        params_high = ADMParams.getADMparams(R, T_base, T_ad_high)
        assert all(np.isfinite(params_high)), "Parameters should be finite at high temperature"

    def test_parameter_consistency(self) -> None:
        """Test that parameters are consistent across multiple calls."""
        R = 0.08314
        T_base = 298.15
        T_ad = 308.15

        params1 = ADMParams.getADMparams(R, T_base, T_ad)
        params2 = ADMParams.getADMparams(R, T_base, T_ad)

        assert np.allclose(params1, params2), "Parameters should be consistent across calls"
