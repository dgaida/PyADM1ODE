# -*- coding: utf-8 -*-
"""Unit tests for the ADM1 (SIMBA# biogas) core model."""

import numpy as np
import pytest

from pyadm1 import Feedstock
from pyadm1.core.adm1 import ADM1, INFLUENT_COLUMNS, STATE_SIZE


@pytest.fixture
def feedstock() -> Feedstock:
    return Feedstock(
        ["maize_silage_milk_ripeness", "swine_manure"],
        feeding_freq=24,
        total_simtime=10,
    )


@pytest.fixture
def model(feedstock: Feedstock) -> ADM1:
    return ADM1(feedstock, V_liq=1200.0, V_gas=216.0, T_ad=315.15)


class TestADM1Construction:
    def test_state_size_is_41(self, model: ADM1) -> None:
        assert STATE_SIZE == 41
        assert model.get_state_size() == 41

    def test_model_name(self, model: ADM1) -> None:
        assert model.model_name == "ADM1"

    def test_volumes_and_temperature(self, model: ADM1) -> None:
        assert model.V_liq == 1200.0
        assert model._V_gas == 216.0
        assert model.T_ad == 315.15

    def test_kinetic_params_temperature_corrected(self, model: ADM1) -> None:
        # k_dis_PS reference value at 35 °C is 0.04; T_ad=315.15 K = 42 °C
        assert model._kinetic["k_dis_PS"] > 0.04

    def test_influent_columns_matches_state_size_minus_gas(self) -> None:
        # 37 liquid columns + Q
        assert len(INFLUENT_COLUMNS) == 38
        assert INFLUENT_COLUMNS[-1] == "Q"


class TestADM1ODE:
    def test_ode_returns_state_size_derivatives(self, model: ADM1, feedstock: Feedstock) -> None:
        model.set_influent_dataframe(feedstock.get_influent_dataframe(Q=[11.4, 6.1]))
        model.create_influent([11.4, 6.1], 0)

        state = [0.01] * STATE_SIZE
        # Avoid divide-by-zero on the gas phase
        state[37:41] = [1.0e-5, 0.65, 0.33, 0.65 + 0.33 + 1.0e-5]

        dydt = model.ADM_ODE(0.0, state)

        assert len(dydt) == STATE_SIZE
        assert all(np.isfinite(d) for d in dydt)

    def test_calc_gas_returns_five_values(self, model: ADM1) -> None:
        q_gas, q_ch4, q_co2, q_h2o, p_gas = model.calc_gas(1.0e-5, 0.65, 0.33, 0.98 + 1.0e-5)

        assert q_gas >= 0.0
        assert q_ch4 >= 0.0
        assert q_co2 >= 0.0
        assert q_h2o >= 0.0
        assert p_gas == pytest.approx(0.65 + 0.33 + 1.0e-5)

    def test_calc_gas_with_low_pressure_yields_zero_flow(self, model: ADM1) -> None:
        # Total pressure below external pressure → no flow
        q_gas, *_ = model.calc_gas(0.0, 0.0, 0.0, 0.0)
        assert q_gas == 0.0


class TestInfluentSetup:
    def test_set_influent_dataframe_validates_columns(self, model: ADM1) -> None:
        import pandas as pd

        bad_df = pd.DataFrame({"S_su": [0.0]})
        with pytest.raises(ValueError, match="missing columns"):
            model.set_influent_dataframe(bad_df)

    def test_create_influent_populates_state_input(self, model: ADM1, feedstock: Feedstock) -> None:
        model.set_influent_dataframe(feedstock.get_influent_dataframe(Q=[11.4, 6.1]))
        model.create_influent([11.4, 6.1], 0)

        assert model._state_input is not None
        assert len(model._state_input) == 37


class TestCalibrationParameters:
    def test_set_and_clear_calibration_parameters(self, model: ADM1) -> None:
        model.set_calibration_parameters({"k_p": 5.0e3})
        assert model.get_calibration_parameters()["k_p"] == 5.0e3

        model.clear_calibration_parameters()
        assert model.get_calibration_parameters() == {}

    def test_calibration_overrides_kp_in_calc_gas(self, model: ADM1) -> None:
        q_default, *_ = model.calc_gas(1.0e-5, 0.65, 0.33, 0.98 + 1.0e-5)

        model.set_calibration_parameters({"k_p": model._k_p / 2.0})
        q_lower, *_ = model.calc_gas(1.0e-5, 0.65, 0.33, 0.98 + 1.0e-5)

        # Lower k_p → lower outlet flow at the same pressure differential
        assert q_lower < q_default


class TestCalcPH:
    def test_neutral_charge_balance_yields_pH7(self) -> None:
        S_H = ADM1._calc_ph(
            S_nh4=0.0,
            S_nh3=0.0,
            S_hco3=0.0,
            S_ac_ion=0.0,
            S_pro_ion=0.0,
            S_bu_ion=0.0,
            S_va_ion=0.0,
            S_cation=0.0,
            S_anion=0.0,
            K_w=1.0e-14,
        )
        pH = -np.log10(S_H)
        assert pH == pytest.approx(7.0, abs=0.05)
