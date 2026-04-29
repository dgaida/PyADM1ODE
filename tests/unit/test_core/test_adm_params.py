# -*- coding: utf-8 -*-
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
