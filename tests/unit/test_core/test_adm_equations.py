"""Unit tests for adm_equations helper functions."""

import pytest

from pyadm1.core.adm_equations import BiochemicalProcesses, InhibitionFunctions


def test_substrate_inhibition_monod_factor():
    """Cover substrate_inhibition helper."""
    value = InhibitionFunctions.substrate_inhibition(S=2.0, K_S=3.0)
    assert value == 2.0 / 5.0


def test_calculate_process_rates_applies_ts_based_hydrolysis_correction():
    """Hydro factors above 1 are treated as TS values and transformed."""
    state = [0.0] * 37
    state[12:23] = [1.5, 2.0, 3.0, 4.0, 0.5, 0.4, 0.3, 0.2, 0.1, 0.6, 0.7]
    inhibitions = (0.0,) * 8 + (1.0,) * 8
    kinetic_params = {
        "k_m_su": 30.0,
        "K_S_su": 0.5,
        "k_m_aa": 50.0,
        "K_S_aa": 0.3,
        "k_m_fa": 6.0,
        "K_S_fa": 0.4,
        "k_m_c4": 20.0,
        "K_S_c4": 0.2,
        "k_m_pro": 13.0,
        "K_S_pro": 0.1,
        "k_m_ac": 8.0,
        "K_S_ac": 0.15,
        "k_m_h2": 35.0,
        "K_S_h2": 7e-6,
        "k_dec_X_su": 0.02,
        "k_dec_X_aa": 0.02,
        "k_dec_X_fa": 0.02,
        "k_dec_X_c4": 0.02,
        "k_dec_X_pro": 0.02,
        "k_dec_X_ac": 0.02,
        "k_dec_X_h2": 0.02,
    }
    substrate_params = {
        "k_dis": 0.5,
        "k_hyd_ch": 10.0,
        "k_hyd_pr": 20.0,
        "k_hyd_li": 30.0,
    }

    rates = BiochemicalProcesses.calculate_process_rates(
        state=state,
        inhibitions=inhibitions,
        kinetic_params=kinetic_params,
        substrate_params=substrate_params,
        hydro_factor=11.0,
    )
    expected_factor = 1.0 / (1.0 + (11.0 / 5.5) ** 2.3)

    assert rates[0] == pytest.approx(0.75)
    assert rates[1] == pytest.approx(10.0 * 2.0 * expected_factor)
    assert rates[2] == pytest.approx(20.0 * 3.0 * expected_factor)
    assert rates[3] == pytest.approx(30.0 * 4.0 * expected_factor)


def test_calculate_process_rates_clamps_negative_hydro_factor_to_zero():
    """Negative hydro factors are clamped before hydrolysis rates are calculated."""
    state = [0.0] * 37
    state[12:23] = [1.5, 2.0, 3.0, 4.0, 0.5, 0.4, 0.3, 0.2, 0.1, 0.6, 0.7]
    inhibitions = (0.0,) * 8 + (1.0,) * 8
    kinetic_params = {
        "k_m_su": 30.0,
        "K_S_su": 0.5,
        "k_m_aa": 50.0,
        "K_S_aa": 0.3,
        "k_m_fa": 6.0,
        "K_S_fa": 0.4,
        "k_m_c4": 20.0,
        "K_S_c4": 0.2,
        "k_m_pro": 13.0,
        "K_S_pro": 0.1,
        "k_m_ac": 8.0,
        "K_S_ac": 0.15,
        "k_m_h2": 35.0,
        "K_S_h2": 7e-6,
        "k_dec_X_su": 0.02,
        "k_dec_X_aa": 0.02,
        "k_dec_X_fa": 0.02,
        "k_dec_X_c4": 0.02,
        "k_dec_X_pro": 0.02,
        "k_dec_X_ac": 0.02,
        "k_dec_X_h2": 0.02,
    }
    substrate_params = {
        "k_dis": 0.5,
        "k_hyd_ch": 10.0,
        "k_hyd_pr": 20.0,
        "k_hyd_li": 30.0,
    }

    rates = BiochemicalProcesses.calculate_process_rates(
        state=state,
        inhibitions=inhibitions,
        kinetic_params=kinetic_params,
        substrate_params=substrate_params,
        hydro_factor=-0.5,
    )

    assert rates[0] == pytest.approx(0.75)
    assert rates[1] == 0.0
    assert rates[2] == 0.0
    assert rates[3] == 0.0
