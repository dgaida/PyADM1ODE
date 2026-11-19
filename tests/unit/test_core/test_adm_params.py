"""Unit tests for ADMParams (SIMBA# biogas model parameters)."""

import pytest

from pyadm1.core.adm_params import ADMParams


def test_get_stoichiometric_params_contains_expected_keys() -> None:
    p = ADMParams.get_stoichiometric_params()

    for k in ("C_su", "C_aa", "C_ch4", "N_bac", "N_aa", "f_ch_bac", "fXI_PS"):
        assert k in p


def test_get_kinetic_params_at_reference_temperature() -> None:
    k = ADMParams.get_kinetic_params()

    assert k["k_dis_PS"] == pytest.approx(0.04)
    assert k["k_dis_PF"] == pytest.approx(0.4)
    assert k["k_m_su"] == pytest.approx(30.0)
    # Decay rate for X_ac is doubled vs the others (SIMBA# convention)
    assert k["k_dec_ac"] == pytest.approx(0.04)
    assert k["k_dec_h2"] == pytest.approx(0.02)


def test_apply_temperature_corrections_at_reference_yields_identity() -> None:
    base = ADMParams.get_kinetic_params()
    theta = ADMParams.get_temperature_factors()

    corrected = ADMParams.apply_temperature_corrections(base, theta, T_ad=308.15)

    assert corrected["k_m_su"] == pytest.approx(base["k_m_su"])
    assert corrected["k_dis_PS"] == pytest.approx(base["k_dis_PS"])
    assert corrected["k_dec_ac"] == pytest.approx(base["k_dec_ac"])


def test_apply_temperature_corrections_above_reference_increases_rates() -> None:
    base = ADMParams.get_kinetic_params()
    theta = ADMParams.get_temperature_factors()

    hot = ADMParams.apply_temperature_corrections(base, theta, T_ad=315.15)

    assert hot["k_m_su"] > base["k_m_su"]
    assert hot["k_dis_PF"] > base["k_dis_PF"]


def test_get_inhibition_params_returns_acid_base_constants() -> None:
    R = 0.08314
    T_base = 298.15
    T_ad = 308.15

    ip = ADMParams.get_inhibition_params(R, T_base, T_ad)

    for key in ("K_w", "K_a_va", "K_a_bu", "K_a_pro", "K_a_ac", "K_a_co2", "K_a_IN"):
        assert key in ip
        assert ip[key] > 0.0


def test_get_admgasparams_returns_six_floats() -> None:
    R = 0.08314
    T_base = 298.15
    T_ad = 308.15

    p_gas_h2o, k_p, k_L_a, K_H_co2, K_H_ch4, K_H_h2 = ADMParams.getADMgasparams(R, T_base, T_ad)

    assert p_gas_h2o > 0.0
    assert k_p == pytest.approx(1.0e4)
    assert k_L_a == pytest.approx(200.0)
    assert K_H_co2 > 0.0
    assert K_H_ch4 > 0.0
    assert K_H_h2 > 0.0


def test_henry_constants_are_referenced_to_25_degrees() -> None:
    """At 25 °C the gas / liquid ratios follow from the tabulated solubilities."""
    R, T = 0.08314, 298.15

    _, _, _, K_H_co2, K_H_ch4, K_H_h2 = ADMParams.getADMgasparams(R, T, T)

    assert K_H_co2 == pytest.approx(1.0 / (0.035 * R * T))
    assert K_H_ch4 == pytest.approx(1.0 / (0.0014 * R * T))
    assert K_H_h2 == pytest.approx(1.0 / (7.8e-4 * R * T))
    assert ADMParams.get_nh3_henry_constant(R, T, T) == pytest.approx(1.0 / (60.0 * R * T))


def test_ammonia_is_less_soluble_in_a_warm_digester() -> None:
    R, T_base = 0.08314, 298.15

    cold = ADMParams.get_nh3_henry_constant(R, T_base, 308.15)
    warm = ADMParams.get_nh3_henry_constant(R, T_base, 328.15)

    assert warm > cold


def test_ammonia_inhibition_of_propionate_degraders_rises_with_temperature() -> None:
    """Temperature coefficient 0.061 per kelvin, as in the SIMBA# model file."""
    R, T_base = 0.08314, 298.15

    at_35 = ADMParams.get_inhibition_params(R, T_base, 308.15)["K_I_nh3_pro"]
    at_45 = ADMParams.get_inhibition_params(R, T_base, 318.15)["K_I_nh3_pro"]

    assert at_35 == pytest.approx(0.0019)
    assert at_45 / at_35 == pytest.approx(2.718281828459045**0.61)


def test_nitrogen_and_decay_products_follow_the_biomass_composition() -> None:
    """Protein holds 13.84 % nitrogen; biomass is 70 % protein; decay returns its three classes."""
    p = ADMParams.get_stoichiometric_params()

    assert p["N_aa"] == pytest.approx(0.7736 * 0.1384 / 14.0)
    assert p["N_bac"] == pytest.approx(0.70 * p["N_aa"])
    assert p["f_ch_bac"] + p["f_pr_bac"] + p["f_li_bac"] + p["f_p_bac"] == pytest.approx(1.0)
    # The inert share holds protein in the same proportion as the degradable share.
    assert p["N_I"] == pytest.approx(p["f_pr_bac"] / (1.0 - p["f_p_bac"]) * p["N_aa"])
    released = p["N_bac"] - p["f_pr_bac"] * p["N_aa"] - p["f_p_bac"] * p["N_I"]
    assert released == pytest.approx(-6.896e-5, rel=1e-3)
    assert p["C_aa"] == p["C_pr"]


def test_specific_volumes_are_mass_per_cod_over_density() -> None:
    p = ADMParams.get_stoichiometric_params()

    assert p["V_ch"] == pytest.approx((180.0 / 192.0) / 1550.0)
    assert p["V_pr"] == pytest.approx(0.7736 / 1370.0)
    assert p["V_li"] == pytest.approx((806.0 / 2320.0) / 920.0)
    assert p["V_ac"] == pytest.approx((60.0 / 64.0) / 1050.0)
    assert p["V_li"] < p["V_bac"] < p["V_ch"]
    assert p["V_li"] < p["V_I_x"] < p["V_ch"]
