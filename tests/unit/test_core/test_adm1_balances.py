"""Balance and structure tests of the ADM1da right-hand side.

Every test evaluates the conversion terms without feed and without sludge
outflow, so transport does not enter: what one state loses another must gain,
or it must leave with the gas.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from pyadm1 import Feedstock
from pyadm1.core.adm1 import ADM1

torch = pytest.importorskip("torch")

from pyadm1.core.adm1_torch import Adm1TorchParams, adm1da_rhs_torch  # noqa: E402

# A digester state close to a loaded maize / manure plant at 42 °C.
STATE = [
    0.005,  # S_su
    0.002,  # S_aa
    0.04,  # S_fa
    0.003,  # S_va
    0.006,  # S_bu
    0.10,  # S_pro
    3.8,  # S_ac
    1.0e-7,  # S_h2
    0.045,  # S_ch4
    0.138,  # S_co2 (total inorganic carbon)
    0.215,  # S_nh4 (total inorganic nitrogen)
    1.5,  # S_I
    12.8,  # X_PS_ch
    0.55,  # X_PS_pr
    0.31,  # X_PS_li
    3.3,  # X_PF_ch
    0.67,  # X_PF_pr
    0.50,  # X_PF_li
    0.47,  # X_S_ch
    0.11,  # X_S_pr
    0.055,  # X_S_li
    55.0,  # X_I
    4.7,  # X_su
    0.9,  # X_aa
    0.34,  # X_fa
    0.68,  # X_c4
    0.56,  # X_pro
    1.28,  # X_ac
    1.05,  # X_h2
    0.0,  # S_cation
    0.019,  # S_anion
    0.0033,  # S_va_ion
    0.0061,  # S_bu_ion
    0.11,  # S_pro_ion
    3.78,  # S_ac_ion
    0.127,  # S_hco3
    0.0082,  # S_nh3
    5.0e-6,  # p_h2
    0.49,  # p_ch4
    0.445,  # p_co2
    0.935005,  # pTOTAL (dry)
]

COD_STATES = [*range(9), 11, *range(12, 29)]
P_H2, P_CH4, P_CO2, P_TOTAL = 37, 38, 39, 40


@pytest.fixture
def model() -> ADM1:
    feedstock = Feedstock(["maize_silage_milk_ripeness", "cattle_manure"], feeding_freq=24, total_simtime=5)
    adm1 = ADM1(feedstock, V_liq=1050.0, V_gas=150.0, T_ad=315.15)
    # No sludge leaves: otherwise the model drains what a volume gain adds.
    adm1._Q_out_override = 0.0
    return adm1


def _rhs(model: ADM1, state: list[float] | None = None) -> np.ndarray:
    return np.asarray(model.ADM_ODE(0.0, list(STATE if state is None else state)), dtype=float)


def _gas_outflow(model: ADM1, state: list[float]) -> float:
    """Dry pressure the gas outlet removes from the headspace [bar/d]."""
    return max(model._k_p * (state[P_TOTAL] + model._p_gas_h2o - model._p_ext) * model.V_liq / model._V_gas, 0.0)


class TestElementBalances:
    def test_cod_is_conserved_between_liquid_gas_and_outlet(self, model: ADM1) -> None:
        """COD lost by the liquid shows up in the headspace or leaves through the outlet."""
        d = _rhs(model)
        per_liquid = model._V_gas / (model._RT * model.V_liq)

        liquid = sum(d[i] for i in COD_STATES)
        headspace = (16.0 * d[P_H2] + 64.0 * d[P_CH4]) * per_liquid
        outlet = (16.0 * STATE[P_H2] + 64.0 * STATE[P_CH4]) / STATE[P_TOTAL] * _gas_outflow(model, STATE) * per_liquid

        assert outlet > 0.0
        assert liquid + headspace == pytest.approx(-outlet, rel=1e-9)

    def test_carbon_is_conserved_in_the_liquid(self, model: ADM1) -> None:
        """Without gas transfer the conversions only move carbon between liquid states."""
        model._k_L_a = 0.0
        d = _rhs(model)
        c = model._stoich
        weights = {0: c["C_su"], 1: c["C_aa"], 2: c["C_fa"], 3: c["C_va"], 4: c["C_bu"], 5: c["C_pro"], 6: c["C_ac"]}
        weights.update({8: c["C_ch4"], 9: 1.0, 11: c["C_I_s"], 21: c["C_I_x"]})
        weights.update(dict.fromkeys((12, 15, 18), c["C_ch"]))
        weights.update(dict.fromkeys((13, 16, 19), c["C_pr"]))
        weights.update(dict.fromkeys((14, 17, 20), c["C_li"]))
        weights.update(dict.fromkeys(range(22, 29), c["C_bac"]))

        turnover = sum(abs(w * d[i]) for i, w in weights.items())
        assert turnover > 1.0e-3
        assert sum(w * d[i] for i, w in weights.items()) == pytest.approx(0.0, abs=1e-12 * turnover + 1e-15)

    def test_nitrogen_only_leaves_as_ammonia_in_the_gas(self, model: ADM1) -> None:
        """Nitrogen moves between ammonium, protein and biomass; the rest is the ammonia in the gas."""
        d = _rhs(model)
        n = model._stoich
        weights = {10: 1.0, 1: n["N_aa"], 13: n["N_aa"], 16: n["N_aa"], 19: n["N_aa"], 21: n["N_I"]}
        weights.update(dict.fromkeys(range(22, 29), n["N_bac"]))

        gas_m3_per_day = _gas_outflow(model, STATE) * model._V_gas / STATE[P_TOTAL]
        with_gas = gas_m3_per_day * model._K_H_nh3 * STATE[36] / model.V_liq

        assert with_gas > 0.0
        assert sum(w * d[i] for i, w in weights.items()) == pytest.approx(-with_gas, rel=1e-9)

    def test_decay_recycles_all_cod_and_carbon(self, model: ADM1) -> None:
        """Decay products add up to the decayed biomass; the inert share keeps its composition."""
        s = model._stoich
        assert s["f_ch_bac"] + s["f_pr_bac"] + s["f_li_bac"] + s["f_p_bac"] == pytest.approx(1.0, abs=1e-14)
        carbon = s["f_ch_bac"] * s["C_ch"] + s["f_pr_bac"] * s["C_pr"] + s["f_li_bac"] * s["C_li"] + s["f_p_bac"] * s["C_I_x"]
        assert carbon == pytest.approx(s["C_bac"], rel=1e-12)


class TestAmmoniaInTheGas:
    def test_ammonia_follows_the_gas_flow(self, model: ADM1) -> None:
        """Switching the ammonia off changes the total and the free ammonia by the same amount."""
        with_nh3 = _rhs(model)
        k_h = model._K_H_nh3
        model._K_H_nh3 = 0.0
        without = _rhs(model)

        gas_m3_per_day = _gas_outflow(model, STATE) * model._V_gas / STATE[P_TOTAL]
        expected = gas_m3_per_day * k_h * STATE[36] / model.V_liq
        changed = np.flatnonzero(with_nh3 != without)

        assert list(changed) == [10, 36]
        assert without[10] - with_nh3[10] == pytest.approx(expected, rel=1e-9)
        assert without[36] - with_nh3[36] == pytest.approx(expected, rel=1e-9)

    def test_gas_holds_a_few_hundred_ppm_ammonia(self, model: ADM1) -> None:
        """Order of magnitude: 115 mg/l free ammonia nitrogen give some hundred ppm in the gas."""
        ppm = model._K_H_nh3 * STATE[36] * model._RT / STATE[P_TOTAL] * 1.0e6
        assert 100.0 < ppm < 1000.0

    def test_no_gas_flow_no_ammonia_loss(self, model: ADM1) -> None:
        """Below the outlet pressure nothing leaves, whatever the dry pressure is."""
        state = list(STATE)
        state[P_TOTAL] = 0.5
        with_nh3 = _rhs(model, state)
        model._K_H_nh3 = 0.0

        assert _gas_outflow(model, state) == 0.0
        assert np.array_equal(with_nh3, _rhs(model, state))

    def test_torch_default_switches_the_ammonia_off(self, model: ADM1) -> None:
        """A hand-built parameter set without ``K_H_nh3`` behaves like the model without the loss."""
        params = Adm1TorchParams.from_adm1(model)
        x = torch.tensor(STATE, dtype=torch.float64)
        with_nh3 = adm1da_rhs_torch(x, params).numpy()
        without = adm1da_rhs_torch(x, dataclasses.replace(params, K_H_nh3=0.0)).numpy()

        assert params.K_H_nh3 == pytest.approx(model._K_H_nh3)
        assert Adm1TorchParams.__dataclass_fields__["K_H_nh3"].default == 0.0
        assert list(np.flatnonzero(with_nh3 != without)) == [10, 36]
        assert with_nh3 == pytest.approx(_rhs(model), rel=1e-9, abs=1e-12)


class TestHeadspacePressure:
    def test_wet_headspace_settles_at_the_external_pressure(self, model: ADM1) -> None:
        """Gas flows as soon as dry gas plus water vapour exceed ``p_ext``, not before."""
        threshold = model.p_ext - model._p_gas_h2o

        assert model.p_ext == pytest.approx(1.013)
        assert model.calc_gas(1.0e-5, 0.5, threshold - 0.5 - 1.0e-5 - 1.0e-6, threshold - 1.0e-6)[0] == 0.0
        assert model.calc_gas(1.0e-5, 0.5, threshold - 0.5 - 1.0e-5 + 1.0e-6, threshold + 1.0e-6)[0] > 0.0

    def test_external_pressure_does_not_depend_on_temperature(self, model: ADM1) -> None:
        before = model.p_ext
        model.T_ad = 328.15
        assert model.p_ext == before

    def test_external_pressure_can_be_raised(self, model: ADM1) -> None:
        """A gas system with back pressure moves the outlet threshold with it."""
        dry = model.p_ext - model._p_gas_h2o + 0.01
        assert model.calc_gas(1.0e-5, 0.5, dry - 0.5 - 1.0e-5, dry)[0] > 0.0

        model.p_ext = 1.05

        assert model.p_ext == 1.05
        assert model.calc_gas(1.0e-5, 0.5, dry - 0.5 - 1.0e-5, dry)[0] == 0.0
        assert Adm1TorchParams.from_adm1(model).p_ext == 1.05

    def test_external_pressure_must_leave_room_for_dry_gas(self, model: ADM1) -> None:
        with pytest.raises(ValueError, match="water vapour"):
            model.p_ext = 0.05


class TestSludgeVolume:
    def test_volume_loss_is_the_net_volume_of_the_converted_organics(self, model: ADM1) -> None:
        """Summing volume over the states gives the same loss as summing it over the processes."""
        d = _rhs(model)
        v = model._stoich
        weights = {0: v["V_ch"], 1: v["V_pr"], 2: v["V_li"], 3: v["V_va"], 4: v["V_bu"], 5: v["V_pro"], 6: v["V_ac"]}
        weights[21] = v["V_I_x"]
        weights.update(dict.fromkeys((12, 15, 18), v["V_ch"]))
        weights.update(dict.fromkeys((13, 16, 19), v["V_pr"]))
        weights.update(dict.fromkeys((14, 17, 20), v["V_li"]))
        weights.update(dict.fromkeys(range(22, 29), v["V_bac"]))

        expected = -model.V_liq * sum(w * d[i] for i, w in weights.items())

        assert model._q_S_loss_last > 0.0
        assert model._q_S_loss_last == pytest.approx(expected, rel=1e-9)

    def test_disintegration_and_hydrolysis_of_carbohydrates_keep_the_volume(self, model: ADM1) -> None:
        """Solids that only dissolve to sugars still take up the same room."""
        state = [0.0] * 41
        state[12], state[15], state[18] = 10.0, 5.0, 1.0
        state[10] = 0.1
        state[37:41] = [1.0e-6, 0.3, 0.2, 0.500001]
        _rhs(model, state)

        assert model._q_S_loss_last == pytest.approx(0.0, abs=1e-12)

    def test_volume_loss_follows_methane_formation(self, model: ADM1) -> None:
        """Acetate turned into methane and CO2 frees the room the acetate took."""
        state = [0.0] * 41
        state[6], state[34], state[27] = 2.0, 2.0, 1.0
        state[10] = 0.1
        state[29] = 2.0 / 64.0
        state[9], state[35] = 0.12, 0.10
        state[37:41] = [1.0e-6, 0.3, 0.2, 0.500001]
        d = _rhs(model, state)
        uptake = (d[27] + model._kinetic["k_dec_ac"] * state[27]) / model._kinetic["Y_ac"]
        decay = model._kinetic["k_dec_ac"] * state[27]
        v = model._stoich
        v_dec = -v["V_bac"] + v["f_ch_bac"] * v["V_ch"] + v["f_pr_bac"] * v["V_pr"] + v["f_li_bac"] * v["V_li"]
        v_dec += v["f_p_bac"] * v["V_I_x"]

        assert uptake > 0.0
        expected = model.V_liq * ((v["V_ac"] - model._kinetic["Y_ac"] * v["V_bac"]) * uptake - v_dec * decay)
        assert model._q_S_loss_last == pytest.approx(expected, rel=1e-9)


class TestInhibitionStructure:
    @staticmethod
    def _acid_state(substrate: int, biomass: int, s_h: float) -> list[float]:
        """Only one substrate and its degraders, at a proton concentration ``s_h``."""
        state = [0.0] * 41
        state[substrate], state[biomass] = 5.0, 1.0
        state[10] = 0.1
        state[30] = 0.1 + s_h  # anions balance the ammonium and leave ``s_h`` protons
        state[37:41] = [1.0e-6, 0.3, 0.2, 0.500001]
        return state

    @pytest.mark.parametrize(("substrate", "biomass", "name"), [(0, 22, "su"), (1, 23, "aa")])
    def test_sugar_and_amino_acid_uptake_use_the_squared_ph_term(
        self, model: ADM1, substrate: int, biomass: int, name: str
    ) -> None:
        """At twice the half-inhibition proton concentration the rate drops to one fifth, not one third."""
        k = model._kinetic
        k_ph = model._inhib_params["K_pH_aa"]
        d = _rhs(model, self._acid_state(substrate, biomass, 2.0 * k_ph))

        uptake = (d[biomass] + k[f"k_dec_{name}"]) / k[f"Y_{name}"]
        unlimited = k[f"k_m_{name}"] * 5.0 / (k[f"K_S_{name}"] + 5.0) * 0.1 / (model._inhib_params["K_S_IN"] + 0.1)

        assert uptake / unlimited == pytest.approx(0.2, rel=1e-3)

    def test_hydrogen_uptake_needs_dissolved_co2(self, model: ADM1) -> None:
        """Inorganic carbon that is all bicarbonate does not feed the hydrogenotrophs."""
        state = self._acid_state(7, 28, 1.0e-7)
        state[7] = 1.0e-5
        state[9], state[35] = 0.1, 0.1
        state[29] = 0.1  # cations balance the bicarbonate

        d = _rhs(model, state)
        assert d[28] == pytest.approx(-model._kinetic["k_dec_h2"], rel=1e-9)

        state[9] = 0.11
        assert _rhs(model, state)[28] > d[28]

    def test_nitrogen_limitation_reads_the_total(self, model: ADM1) -> None:
        """``S_nh3`` is part of ``S_nh4``; without ``S_nh4`` there is no nitrogen to grow on."""
        state = self._acid_state(0, 22, 1.0e-7)
        state[10], state[30], state[36] = 0.0, 1.0e-7, 0.05

        assert _rhs(model, state)[22] == pytest.approx(-model._kinetic["k_dec_su"], rel=1e-9)
