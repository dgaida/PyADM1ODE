# ============================================================================
# pyadm1/components/biological/adm1da_digester.py
# ============================================================================
"""
ADM1da-based digester component.

Mirrors the public interface of :class:`pyadm1.components.biological.Digester`
but wraps :class:`pyadm1.core.adm1da.ADM1da` (41-state SIMBA# biogas model)
instead of the legacy 37-state ADM1.  Interchangeable with the ADM1 Digester
at the component level, so plants can switch biological back-ends without
restructuring the topology.

Key differences from the ADM1 Digester
--------------------------------------
* State vector has 41 elements (vs 37) — indices 0-36 are liquid/charge-
  balance, 37-40 are gas partial pressures.
* Flow Q is at ``_state_input[37]`` (vs index 33 in ADM1).
* The ODE method is ``ADM_ODE`` (vs ``ADM1_ODE`` in ADM1); called directly
  via :func:`scipy.integrate.solve_ivp`, bypassing the ADM1-specific
  :class:`Simulator` wrapper.
* pH, VFA and TAC are computed natively from the ADM1da state vector (no
  C# DLL dependency).

Example
-------
    >>> from pyadm1.components.biological import ADM1daDigester
    >>> from pyadm1.substrates.adm1da_feedstock import ADM1daFeedstock
    >>> fs = ADM1daFeedstock(...)
    >>> dig = ADM1daDigester("dig1", fs, V_liq=1200, V_gas=216, T_ad=315.15)
    >>> dig.initialize()
"""

from typing import Any, Dict, Optional

import numpy as np
from scipy.integrate import solve_ivp

from .digester_base import DigesterBase
from ...core.adm1da import (
    ADM1da,
    STATE_SIZE as _ADM1DA_STATE_SIZE,
    _IDX_P_H2,
    _IDX_P_CH4,
    _IDX_P_CO2,
    _IDX_P_TOTAL,
    _IDX_S_NH4,
    _IDX_S_NH3,
    _IDX_S_HCO3,
    _IDX_S_AC_ION,
    _IDX_S_PRO_ION,
    _IDX_S_BU_ION,
    _IDX_S_VA_ION,
    _IDX_S_CATION,
    _IDX_S_ANION,
    _IDX_S_VA,
    _IDX_S_BU,
    _IDX_S_PRO,
    _IDX_S_AC,
)
from ...core.adm1da_params import ADM1daParams

# Q is the last column of the ADM1da influent vector (index 37)
_Q_IDX_ADM1DA = 37


class ADM1daDigester(DigesterBase):
    """
    Digester component wrapping the ADM1da (SIMBA# biogas) model.

    Public API is intentionally identical to
    :class:`pyadm1.components.biological.Digester` so the two classes are
    drop-in interchangeable at the plant-composition level.  Both derive
    from :class:`DigesterBase`, which supplies the shared geometry/storage
    scaffolding.

    Parameters
    ----------
    component_id : str
    feedstock : ADM1daFeedstock or Feedstock or None
        Feedstock used to derive the influent when ``create_influent`` is
        called without a pre-set DataFrame.  Pass ``None`` if you intend to
        drive the digester exclusively via ``set_influent_dataframe``.
    V_liq, V_gas, T_ad : float
        Reactor liquid volume [m³], gas headspace [m³] and temperature [K].
    name : str, optional
    """

    def __init__(
        self,
        component_id: str,
        feedstock,
        V_liq: float = 1977.0,
        V_gas: float = 304.0,
        T_ad: float = 308.15,
        name: Optional[str] = None,
    ):
        super().__init__(component_id, feedstock, V_liq, V_gas, T_ad, name)

        self.adm1 = ADM1da(feedstock=feedstock, V_liq=V_liq, V_gas=V_gas, T_ad=T_ad)

    # ------------------------------------------------------------------
    # Component lifecycle
    # ------------------------------------------------------------------

    def initialize(self, initial_state: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the digester state and attached gas storage.

        Keys accepted in *initial_state* mirror :class:`Digester`:

        * ``adm1_state`` : list of 41 floats (ADM1da state vector).
          If omitted and ``Q_substrates`` is given together with an
          :class:`ADM1daFeedstock`, a pre-inoculated steady-state vector at
          pH 7 is computed automatically from the blended influent.
        * ``Q_substrates`` : list of substrate feed rates [m³/d].  When
          supplied together with an :class:`ADM1daFeedstock`, the underlying
          ADM1da solver's influent DataFrame and density are set directly
          from the feedstock — no manual blending required.
        * ``gas_storage`` : dict forwarded to the attached gas storage
        """
        if initial_state is None:
            initial_state = {}

        # --- Substrate feed: auto-wire the influent DataFrame/density ---
        Q_substrates = initial_state.get("Q_substrates")
        if Q_substrates is not None:
            self.Q_substrates = list(Q_substrates)

            # Auto-wire the influent DataFrame and density from an ADM1daFeedstock
            from ...substrates.adm1da_feedstock import ADM1daFeedstock

            if isinstance(self.feedstock, ADM1daFeedstock):
                self.adm1.set_influent_dataframe(self.feedstock.get_influent_dataframe(Q=self.Q_substrates))
                self.adm1.set_influent_density(self.feedstock.blended_density(self.Q_substrates))

        # --- Biological state vector ---
        if "adm1_state" in initial_state and initial_state["adm1_state"] is not None:
            self.adm1_state = list(initial_state["adm1_state"])
        elif Q_substrates is not None:
            from ...substrates.adm1da_feedstock import ADM1daFeedstock

            if isinstance(self.feedstock, ADM1daFeedstock):
                self.adm1_state = self._build_pre_inoculated_state(self.Q_substrates)
            else:
                self.adm1_state = [0.01] * _ADM1DA_STATE_SIZE
        else:
            self.adm1_state = [0.01] * _ADM1DA_STATE_SIZE

        self.state = {
            "adm1_state": self.adm1_state,
            "Q_substrates": self.Q_substrates,
            "Q_gas": 0.0,
            "Q_ch4": 0.0,
            "Q_co2": 0.0,
            "pH": 7.0,
            "VFA": 0.0,
            "TAC": 0.0,
        }

        gs_state = initial_state.get("gas_storage")
        try:
            self.gas_storage.initialize(gs_state)
        except Exception:
            self.gas_storage.initialize()

        self._initialized = True

    def _build_pre_inoculated_state(self, Q_substrates) -> list:
        """
        Build a pre-inoculated initial ADM1da state from the feedstock blend.

        Particulate pools are seeded at their retention-factor steady state;
        dissolved species correspond to a healthy digester at pH 7; biomass
        concentrations sit above the bistability washout threshold so that
        methanogenesis is active from t = 0.  ``S_cation`` is computed from
        the charge balance to enforce pH = 7 exactly.
        """
        fs = self.feedstock
        conc = fs.blended_concentrations(Q_substrates)
        Q_total = float(np.sum(Q_substrates))
        V_liq = float(self.V_liq)
        T_ad = float(self.T_ad)

        # --- Acid-base constants (temperature-corrected to T_ad) ---
        R_gas = 0.08314  # bar·m³·kmol⁻¹·K⁻¹
        T_base = 298.15  # 25 °C reference
        K_a_va = 10.0**-4.86
        K_a_bu = 10.0**-4.82
        K_a_pro = 10.0**-4.88
        K_a_ac = 10.0**-4.76
        K_a_co2 = 10.0**-6.35
        K_a_IN = 10.0**-9.25
        K_w = 1.0e-14 * np.exp((55900.0 / (100.0 * R_gas)) * (1.0 / T_base - 1.0 / T_ad))

        # --- Target pH 7 and typical healthy-digester dissolved concentrations ---
        S_H_0 = 10.0**-7.0
        S_ac_0, S_pro_0, S_bu_0, S_va_0 = 0.10, 0.02, 0.01, 0.01  # kg COD/m³
        S_co2_0 = 0.18  # S_IC [kmol C/m³]
        S_nh4_0 = conc.get("S_nh4", 0.0) * 1.5

        # --- Ion concentrations at pH 7 ---
        S_ac_ion_0 = K_a_ac / (K_a_ac + S_H_0) * S_ac_0
        S_pro_ion_0 = K_a_pro / (K_a_pro + S_H_0) * S_pro_0
        S_bu_ion_0 = K_a_bu / (K_a_bu + S_H_0) * S_bu_0
        S_va_ion_0 = K_a_va / (K_a_va + S_H_0) * S_va_0
        S_nh3_0 = K_a_IN / (K_a_IN + S_H_0) * S_nh4_0
        S_hco3_0 = K_a_co2 * S_co2_0 / (S_H_0 + K_a_co2)

        vfa_kmol_0 = S_ac_ion_0 / 64.0 + S_pro_ion_0 / 112.0 + S_bu_ion_0 / 160.0 + S_va_ion_0 / 208.0

        # S_anion conservative tracer from influent; S_cation closes charge balance at pH 7.
        S_anion_0 = conc.get("S_anion", 0.0)
        S_cation_0 = S_anion_0 + S_hco3_0 + vfa_kmol_0 + K_w / S_H_0 - S_nh4_0 + S_nh3_0 - S_H_0

        # --- Particulate pools at retention-factor steady state ---
        # SIMBA# kinetics at 35 °C: k_dis_PF=0.4, k_dis_PS=0.04, k_hyd=4.0 d⁻¹.
        D = Q_total / V_liq if V_liq > 0.0 else 0.0
        dT = T_ad - 308.15
        k_dis_PF = 0.4 * (1.035**dT)
        k_dis_PS = 0.04 * (1.035**dT)
        k_hyd = 4.0 * (1.07**dT)

        ret_PF = D / (D + k_dis_PF) if (D + k_dis_PF) > 0.0 else 0.0
        ret_PS = D / (D + k_dis_PS) if (D + k_dis_PS) > 0.0 else 0.0

        X_PF_ch_ss = conc.get("X_PF_ch", 0.0) * ret_PF
        X_PF_pr_ss = conc.get("X_PF_pr", 0.0) * ret_PF
        X_PF_li_ss = conc.get("X_PF_li", 0.0) * ret_PF
        X_PS_ch_ss = conc.get("X_PS_ch", 0.0) * ret_PS
        X_PS_pr_ss = conc.get("X_PS_pr", 0.0) * ret_PS
        X_PS_li_ss = conc.get("X_PS_li", 0.0) * ret_PS

        # X_S pools at SS: produced by disintegration, consumed by hydrolysis (fXI = 0).
        denom_hyd = D + k_hyd
        X_S_ch_0 = (k_dis_PF * X_PF_ch_ss + k_dis_PS * X_PS_ch_ss) / denom_hyd
        X_S_pr_0 = (k_dis_PF * X_PF_pr_ss + k_dis_PS * X_PS_pr_ss) / denom_hyd
        X_S_li_0 = (k_dis_PF * X_PF_li_ss + k_dis_PS * X_PS_li_ss) / denom_hyd

        X_I_0 = conc.get("X_I", 0.0)

        # Gas phase (realistic partial pressures for a running digester)
        p_h2_0, p_ch4_0, p_co2_0 = 1.02e-5, 0.65, 0.33
        p_tot_0 = p_h2_0 + p_ch4_0 + p_co2_0

        return [
            0.01,
            0.001,
            0.05,  # 0-2   S_su, S_aa, S_fa
            S_va_0,
            S_bu_0,
            S_pro_0,
            S_ac_0,  # 3-6   S_va, S_bu, S_pro, S_ac
            1.0e-7,
            1.0e-4,  # 7-8   S_h2, S_ch4
            S_co2_0,
            S_nh4_0,
            0.0,  # 9-11  S_IC, S_nh4, S_I
            X_PS_ch_ss,
            X_PS_pr_ss,
            X_PS_li_ss,  # 12-14 X_PS_*
            X_PF_ch_ss,
            X_PF_pr_ss,
            X_PF_li_ss,  # 15-17 X_PF_*
            X_S_ch_0,
            X_S_pr_0,
            X_S_li_0,  # 18-20 X_S_*
            X_I_0,  # 21    X_I
            0.50,
            0.50,
            0.30,
            0.40,
            0.30,
            1.20,
            0.30,  # 22-28 biomass
            S_cation_0,
            S_anion_0,  # 29-30
            S_va_ion_0,
            S_bu_ion_0,
            S_pro_ion_0,
            S_ac_ion_0,
            S_hco3_0,
            S_nh3_0,  # 31-36
            p_h2_0,
            p_ch4_0,
            p_co2_0,
            p_tot_0,  # 37-40 gas phase
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _has_valid_state_input(self) -> bool:
        state_input = getattr(self.adm1, "_state_input", None)
        if state_input is None:
            return False
        try:
            float(state_input[_Q_IDX_ADM1DA])
        except (TypeError, ValueError, IndexError):
            return False
        return True

    def _compute_indicators(self) -> Dict[str, float]:
        """Compute pH and VFA directly from the ADM1da state (no DLL)."""
        st = self.adm1_state
        inhib = ADM1daParams.get_inhibition_params(self.adm1._R, self.adm1._T_base, self.adm1._T_ad)
        try:
            S_H = ADM1da._calc_ph(
                st[_IDX_S_NH4],
                st[_IDX_S_NH3],
                st[_IDX_S_HCO3],
                st[_IDX_S_AC_ION],
                st[_IDX_S_PRO_ION],
                st[_IDX_S_BU_ION],
                st[_IDX_S_VA_ION],
                st[_IDX_S_CATION],
                st[_IDX_S_ANION],
                inhib["K_w"],
            )
            pH = -np.log10(max(float(S_H), 1.0e-14))
        except Exception:
            pH = 7.0

        # VFA as total volatile fatty acids expressed in g HAc-eq/L
        # (sum of COD-based acids divided by their COD factors, then × M_HAc)
        # Simple COD-weighted approximation: just report sum(S_va..S_ac) in g COD/L
        vfa = float(st[_IDX_S_VA] + st[_IDX_S_BU] + st[_IDX_S_PRO] + st[_IDX_S_AC])

        # TAC not natively available in ADM1da; report HCO3⁻ as carbonate alk proxy
        # (kmol/m³ → g CaCO3/L: ×100 g/mol → ×100 kg/m³ → ×0.1 g/L equivalent)
        tac = float(st[_IDX_S_HCO3]) * 100.0  # rough proxy, g CaCO3/L
        return {"pH": pH, "VFA": vfa, "TAC": tac}

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, t: float, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate the ADM1da ODE by *dt* days and return outputs.

        Accepts the same inputs as :class:`Digester.step` and returns the
        same output schema (``Q_out``, ``state_out``, ``Q_gas``, ``Q_ch4``,
        ``Q_co2``, ``pH``, ``VFA``, ``TAC``, ``gas_storage``).
        """
        if "Q_substrates" in inputs:
            self.Q_substrates = inputs["Q_substrates"]

        if "state_in" in inputs and "Q_in" in inputs:
            Q_in = float(inputs["Q_in"])
            state_in = inputs["state_in"]

            self.adm1.create_influent(self.Q_substrates, int(t / dt))
            Q_sub = float(self.adm1._state_input[_Q_IDX_ADM1DA])
            Q_total = Q_in + Q_sub

            if Q_total > 0 and len(state_in) >= _Q_IDX_ADM1DA:
                for idx in range(_Q_IDX_ADM1DA):
                    c_sub = float(self.adm1._state_input[idx])
                    c_in = float(state_in[idx])
                    self.adm1._state_input[idx] = (c_sub * Q_sub + c_in * Q_in) / Q_total
                self.adm1._state_input[_Q_IDX_ADM1DA] = Q_total
        else:
            self.adm1.create_influent(self.Q_substrates, int(t / dt))

        # Integrate over [t, t+dt]
        result = solve_ivp(
            fun=self.adm1.ADM_ODE,
            t_span=(t, t + dt),
            y0=self.adm1_state,
            method="BDF",
            rtol=1.0e-6,
            atol=1.0e-8,
            max_step=max(0.1 * dt, 1.0e-3),
        )
        if not result.success:
            raise RuntimeError(f"ADM1da integration failed in '{self.component_id}': " f"{result.message}")
        self.adm1_state = list(result.y[:, -1])

        # Gas production from partial pressures at the tail of the state vector
        q_gas, q_ch4, q_co2, _, _ = self.adm1.calc_gas(
            self.adm1_state[_IDX_P_H2],
            self.adm1_state[_IDX_P_CH4],
            self.adm1_state[_IDX_P_CO2],
            self.adm1_state[_IDX_P_TOTAL],
        )

        indicators = self._compute_indicators()
        self.state["adm1_state"] = self.adm1_state
        self.state["Q_gas"] = q_gas
        self.state["Q_ch4"] = q_ch4
        self.state["Q_co2"] = q_co2
        self.state.update(indicators)

        Q_out = (
            float(self.adm1._state_input[_Q_IDX_ADM1DA]) if self._has_valid_state_input() else float(np.sum(self.Q_substrates))
        )

        gs_outputs = self.gas_storage.step(
            t=t,
            dt=dt,
            inputs={
                "Q_gas_in_m3_per_day": float(q_gas),
                "Q_gas_out_m3_per_day": 0.0,
                "vent_to_flare": True,
            },
        )
        self.gas_storage.outputs_data["Q_gas_in_m3_per_day"] = float(q_gas)

        self.outputs_data = {
            "Q_out": Q_out,
            "state_out": self.adm1_state,
            "Q_gas": float(q_gas),
            "Q_ch4": float(q_ch4),
            "Q_co2": float(q_co2),
            "pH": indicators["pH"],
            "VFA": indicators["VFA"],
            "TAC": indicators["TAC"],
            "Q_gas_to_storage_m3_per_day": float(q_gas),
            "gas_storage": {
                "component_id": self.gas_storage.component_id,
                "stored_volume_m3": gs_outputs["stored_volume_m3"],
                "pressure_bar": gs_outputs["pressure_bar"],
                "vented_volume_m3": gs_outputs["vented_volume_m3"],
                "Q_gas_supplied_m3_per_day": gs_outputs["Q_gas_supplied_m3_per_day"],
            },
        }
        return self.outputs_data

    @classmethod
    def from_dict(cls, config: Dict[str, Any], feedstock) -> "ADM1daDigester":
        digester = cls(
            component_id=config["component_id"],
            feedstock=feedstock,
            V_liq=config.get("V_liq", 1977.0),
            V_gas=config.get("V_gas", 304.0),
            T_ad=config.get("T_ad", 308.15),
            name=config.get("name"),
        )
        digester.inputs = config.get("inputs", [])
        digester.outputs = config.get("outputs", [])
        if "state" in config:
            digester.initialize(config["state"])
        return digester
