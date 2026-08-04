"""Unit tests for the ADM1 (SIMBA# biogas) core model."""

import numpy as np
import pytest

from pyadm1 import Feedstock
from pyadm1.core.adm1 import (
    _IDX_P_CH4,
    _IDX_P_CO2,
    _IDX_P_H2,
    _IDX_P_TOTAL,
    ADM1,
    INFLUENT_COLUMNS,
    STATE_SIZE,
    get_default_adm1_backend,
    get_state_zero_from_csv,
    set_default_adm1_backend,
)


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

    def test_calc_gas_species_split_follows_the_wet_partial_pressures(self, model: ADM1) -> None:
        p_h2, p_ch4, p_co2, pTOTAL = 1.0e-5, 0.65, 0.33, 1.3
        q_gas, q_ch4, q_co2, q_h2o, p_gas = model.calc_gas(p_h2, p_ch4, p_co2, pTOTAL)

        p_gas_wet = p_gas + model._p_gas_h2o
        assert q_ch4 == pytest.approx(q_gas * p_ch4 / p_gas_wet)
        assert q_co2 == pytest.approx(q_gas * p_co2 / p_gas_wet)
        assert q_h2o == pytest.approx(q_gas * model._p_gas_h2o / p_gas_wet)
        # The H2/H2O remainder means the species never exceed the total flow.
        assert q_ch4 + q_co2 + q_h2o <= q_gas + 1e-9

    def test_calc_gas_species_flows_vanish_for_a_non_positive_wet_pressure(self, model: ADM1) -> None:
        """Degenerate headspace: negative partial pressures below -p_h2o.

        The species split would divide by a non-positive number, so all three
        component flows are reported as zero while the (pressure-driven) total
        flow is still returned.
        """
        assert -0.15 + model._p_gas_h2o < 0.0  # precondition for the branch

        q_gas, q_ch4, q_co2, q_h2o, p_gas = model.calc_gas(-0.05, -0.05, -0.05, 1.3)

        assert p_gas == pytest.approx(-0.15)
        assert q_gas > 0.0
        assert (q_ch4, q_co2, q_h2o) == (0.0, 0.0, 0.0)


class TestBackendSelection:
    def test_default_backend_is_numpy(self) -> None:
        assert get_default_adm1_backend() == "numpy"
        assert ADM1(feedstock=None).backend == "numpy"

    def test_default_backend_applies_to_instances_built_without_an_explicit_one(self) -> None:
        previous = get_default_adm1_backend()
        try:
            set_default_adm1_backend("torch")

            assert get_default_adm1_backend() == "torch"
            assert ADM1(feedstock=None).backend == "torch"
            # An explicit argument still wins over the process-wide default.
            assert ADM1(feedstock=None, backend="numpy").backend == "numpy"
        finally:
            set_default_adm1_backend(previous)

    def test_setting_an_unknown_default_backend_is_rejected(self) -> None:
        before = get_default_adm1_backend()

        with pytest.raises(ValueError, match="Unknown ADM1 backend"):
            set_default_adm1_backend("jax")

        # The rejected value must not have been stored.
        assert get_default_adm1_backend() == before

    def test_rhs_callable_rejects_a_backend_swapped_in_after_construction(self) -> None:
        model = ADM1(feedstock=None)
        model.backend = "jax"  # bypasses the constructor's validation

        with pytest.raises(ValueError, match="Unknown ADM1 backend"):
            model.rhs_callable()


class TestStateZeroFromCsv:
    @staticmethod
    def _write_csv(path, liquid_value: float = 0.25) -> np.ndarray:
        import pandas as pd

        liquid_cols = INFLUENT_COLUMNS[:-1]  # 37 columns, Q dropped
        gas_cols = ("p_gas_h2", "p_gas_ch4", "p_gas_co2", "pTOTAL")
        expected = np.array(
            [liquid_value * (i + 1) for i in range(len(liquid_cols))] + [1.0e-5, 0.65, 0.33, 0.98],
            dtype=float,
        )
        row = dict(zip(list(liquid_cols) + list(gas_cols), expected))
        # A second row must be ignored: only the first is read.
        pd.DataFrame([row, dict.fromkeys(row, -999.0)]).to_csv(path, index=False)
        return expected

    def test_reads_the_first_row_into_a_full_state_vector(self, tmp_path) -> None:
        path = tmp_path / "state0.csv"
        expected = self._write_csv(path)

        state = get_state_zero_from_csv(str(path))

        assert len(state) == STATE_SIZE
        assert all(isinstance(v, float) for v in state)
        np.testing.assert_allclose(state, expected)

    def test_result_is_accepted_by_the_ode(self, tmp_path, model: ADM1, feedstock: Feedstock) -> None:
        path = tmp_path / "state0.csv"
        self._write_csv(path, liquid_value=0.01)
        model.set_influent_dataframe(feedstock.get_influent_dataframe(Q=[11.4, 6.1]))
        model.create_influent([11.4, 6.1], 0)

        dydt = model.ADM_ODE(0.0, get_state_zero_from_csv(str(path)))

        assert len(dydt) == STATE_SIZE
        assert all(np.isfinite(d) for d in dydt)

    def test_missing_column_is_reported(self, tmp_path) -> None:
        import pandas as pd

        path = tmp_path / "incomplete.csv"
        pd.DataFrame([{"S_su": 0.1}]).to_csv(path, index=False)

        with pytest.raises(KeyError):
            get_state_zero_from_csv(str(path))


class TestHistoryProperties:
    """The public history lists are live views onto the tracked trajectories."""

    def test_gas_and_ph_histories_are_filled_by_the_tracking_helper(self, model: ADM1) -> None:
        state = [0.01] * STATE_SIZE
        state[_IDX_P_H2], state[_IDX_P_CH4], state[_IDX_P_CO2], state[_IDX_P_TOTAL] = 1.0e-5, 0.65, 0.33, 1.3

        model.print_params_at_current_state(state)

        # The first call pads to three entries so downstream diff-based readers work.
        assert len(model.Q_GAS) == 3
        assert model.Q_GAS[0] > 0.0
        for history in (model.Q_CH4, model.Q_CO2, model.Q_H2O, model.P_GAS):
            assert len(history) == 3
        assert model.P_GAS[0] == pytest.approx(0.65 + 0.33 + 1.0e-5)
        assert len(model.pH_l) == 2
        assert 0.0 < model.pH_l[0] < 14.0

    def test_gas_histories_expose_the_underlying_lists(self, model: ADM1) -> None:
        assert model.Q_GAS is model._Q_GAS
        assert model.Q_CH4 is model._Q_CH4
        assert model.Q_CO2 is model._Q_CO2
        assert model.Q_H2O is model._Q_H2O
        assert model.P_GAS is model._P_GAS
        assert model.pH_l is model._pH_l

    def test_indicator_histories_start_empty(self, model: ADM1) -> None:
        # Declared history hooks; the core model itself never appends to them
        # (the Digester component computes these indicators on demand instead).
        assert model.VFA_TA == []
        assert model.AcvsPro == []
        assert model.VFA == []
        assert model.TAC == []

    def test_resume_from_broken_simulation_restores_the_methane_history(self, model: ADM1) -> None:
        model.resume_from_broken_simulation([1.0, 2.0, 3.0])

        assert model.Q_CH4 == [1.0, 2.0, 3.0]


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

    def test_kinetic_keys_are_written_into_the_live_kinetic_dict(self, model: ADM1) -> None:
        model.set_calibration_parameters({"k_dis_PS": 0.123})

        assert model._kinetic["k_dis_PS"] == pytest.approx(0.123)

        model.clear_calibration_parameters()
        assert model._kinetic["k_dis_PS"] == model._kinetic_default["k_dis_PS"]

    @pytest.mark.parametrize("key", ["K_H_co2", "K_H_ch4", "K_H_h2"])
    def test_each_henry_constant_can_be_overridden_and_restored(self, model: ADM1, key: str) -> None:
        attr = f"_{key}"
        original = getattr(model, attr)

        model.set_calibration_parameters({key: original * 1.5})
        assert getattr(model, attr) == pytest.approx(original * 1.5)

        model.clear_calibration_parameters()
        assert getattr(model, attr) == pytest.approx(original)

    def test_henry_overrides_are_independent_of_each_other(self, model: ADM1) -> None:
        co2, ch4, h2 = model._K_H_co2, model._K_H_ch4, model._K_H_h2

        model.set_calibration_parameters({"K_H_ch4": ch4 * 2.0})

        assert model._K_H_ch4 == pytest.approx(ch4 * 2.0)
        assert model._K_H_co2 == pytest.approx(co2)
        assert model._K_H_h2 == pytest.approx(h2)

    def test_unknown_keys_are_stored_but_do_not_touch_the_model(self, model: ADM1) -> None:
        kinetic_before = dict(model._kinetic)

        model.set_calibration_parameters({"k_L_a": 150.0, "not_a_parameter": 1.0})

        # k_L_a / k_p are looked up at their call sites, not written into _kinetic.
        assert model.get_calibration_parameters()["k_L_a"] == 150.0
        assert model._kinetic == kinetic_before

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
