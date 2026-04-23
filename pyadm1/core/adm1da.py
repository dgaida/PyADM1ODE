# pyadm1/core/adm1da.py
"""
SIMBA# biogas ADM1da model implementation.

ADM1da (Schlattmann 2011) is an agricultural-biogas extension of ADM1 that
replaces the single composite variable (X_xc) with a two-pool sub-fraction
approach and adds temperature-dependent kinetics plus modified inhibition.

State vector (41 variables)
----------------------------
Dissolved (0–11):
    0  S_su       monosaccharides              [kg COD m⁻³]
    1  S_aa       amino acids                  [kg COD m⁻³]
    2  S_fa       long-chain fatty acids        [kg COD m⁻³]
    3  S_va       total valerate               [kg COD m⁻³]
    4  S_bu       total butyrate               [kg COD m⁻³]
    5  S_pro      total propionate             [kg COD m⁻³]
    6  S_ac       total acetate                [kg COD m⁻³]
    7  S_h2       dissolved hydrogen           [kg COD m⁻³]
    8  S_ch4      dissolved methane            [kg COD m⁻³]
    9  S_co2      inorganic carbon (S_IC)      [kmol C m⁻³]
   10  S_nh4      inorganic nitrogen (S_IN)    [kmol N m⁻³]
   11  S_I        soluble inerts               [kg COD m⁻³]

Particulate sub-fractions (12–21):
   12  X_PS_ch    slow-disint. carbohydrates   [kg COD m⁻³]
   13  X_PS_pr    slow-disint. proteins        [kg COD m⁻³]
   14  X_PS_li    slow-disint. lipids          [kg COD m⁻³]
   15  X_PF_ch    fast-disint. carbohydrates   [kg COD m⁻³]
   16  X_PF_pr    fast-disint. proteins        [kg COD m⁻³]
   17  X_PF_li    fast-disint. lipids          [kg COD m⁻³]
   18  X_S_ch     hydrolysable carbohydrates   [kg COD m⁻³]
   19  X_S_pr     hydrolysable proteins        [kg COD m⁻³]
   20  X_S_li     hydrolysable lipids          [kg COD m⁻³]
   21  X_I        particulate inerts           [kg COD m⁻³]

Biomass (22–28):
   22  X_su       sugar degraders              [kg COD m⁻³]
   23  X_aa       amino-acid degraders         [kg COD m⁻³]
   24  X_fa       LCFA degraders               [kg COD m⁻³]
   25  X_c4       valerate/butyrate degraders  [kg COD m⁻³]
   26  X_pro      propionate degraders         [kg COD m⁻³]
   27  X_ac       acetate degraders            [kg COD m⁻³]
   28  X_h2       hydrogen degraders           [kg COD m⁻³]

Charge balance (29–36):
   29  S_cation   strong-base cations          [kmol m⁻³]
   30  S_anion    strong-acid anions           [kmol m⁻³]
   31  S_va_ion   valerate ion                 [kg COD m⁻³]
   32  S_bu_ion   butyrate ion                 [kg COD m⁻³]
   33  S_pro_ion  propionate ion               [kg COD m⁻³]
   34  S_ac_ion   acetate ion                  [kg COD m⁻³]
   35  S_hco3_ion bicarbonate                  [kmol C m⁻³]
   36  S_nh3      free ammonia                 [kmol N m⁻³]

Gas phase (37–40):
   37  p_gas_h2   H₂ partial pressure         [bar]
   38  p_gas_ch4  CH₄ partial pressure        [bar]
   39  p_gas_co2  CO₂ partial pressure        [bar]
   40  pTOTAL     total gas pressure          [bar]

Influent format (set via set_influent_dataframe)
------------------------------------------------
DataFrame with columns:
    S_su … S_nh3 (indices 0–36), Q [m³ d⁻¹]
    → 38 columns total (no gas-phase states in influent).
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple

from pyadm1.core.adm_base import ADMBase
from pyadm1.core.adm1da_params import ADM1daParams
from pyadm1.core.adm_params import ADMParams
from pyadm1.substrates.feedstock import Feedstock

# INFLUENT_COLUMNS is defined in adm1da_feedstock to avoid a circular import:
#   adm1da_feedstock → pyadm1.core.adm1da → adm_base → substrates → adm1da_feedstock
from pyadm1.substrates.adm1da_feedstock import INFLUENT_COLUMNS  # noqa: E402

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# State-vector index constants
# --------------------------------------------------------------------------
_IDX_S_SU = 0
_IDX_S_AA = 1
_IDX_S_FA = 2
_IDX_S_VA = 3
_IDX_S_BU = 4
_IDX_S_PRO = 5
_IDX_S_AC = 6
_IDX_S_H2 = 7
_IDX_S_CH4 = 8
_IDX_S_CO2 = 9
_IDX_S_NH4 = 10
_IDX_S_I = 11
_IDX_X_PS_CH = 12
_IDX_X_PS_PR = 13
_IDX_X_PS_LI = 14
_IDX_X_PF_CH = 15
_IDX_X_PF_PR = 16
_IDX_X_PF_LI = 17
_IDX_X_S_CH = 18
_IDX_X_S_PR = 19
_IDX_X_S_LI = 20
_IDX_X_I = 21
_IDX_X_SU = 22
_IDX_X_AA = 23
_IDX_X_FA = 24
_IDX_X_C4 = 25
_IDX_X_PRO = 26
_IDX_X_AC = 27
_IDX_X_H2 = 28
_IDX_S_CATION = 29
_IDX_S_ANION = 30
_IDX_S_VA_ION = 31
_IDX_S_BU_ION = 32
_IDX_S_PRO_ION = 33
_IDX_S_AC_ION = 34
_IDX_S_HCO3 = 35
_IDX_S_NH3 = 36
_IDX_P_H2 = 37
_IDX_P_CH4 = 38
_IDX_P_CO2 = 39
_IDX_P_TOTAL = 40

STATE_SIZE = 41


def get_state_zero_from_csv(csv_file: str) -> List[float]:
    """
    Load an ADM1da initial state vector from a CSV file.

    The CSV must have exactly the columns listed in ``INFLUENT_COLUMNS``
    (without Q) plus the four gas-phase columns (p_gas_h2, p_gas_ch4,
    p_gas_co2, pTOTAL).

    Parameters
    ----------
    csv_file : str
        Path to the CSV file.

    Returns
    -------
    List[float]
        41-element state vector.
    """
    df = pd.read_csv(csv_file)
    state = []
    liquid_cols = INFLUENT_COLUMNS[:-1]  # drop Q
    for col in liquid_cols:
        state.append(float(df[col].iloc[0]))
    for col in ("p_gas_h2", "p_gas_ch4", "p_gas_co2", "pTOTAL"):
        state.append(float(df[col].iloc[0]))
    return state


class ADM1da(ADMBase):
    """
    SIMBA# biogas ADM1da model – 41-state ODE system.

    Extends ADMBase with the sub-fraction disintegration/hydrolysis approach,
    temperature-dependent kinetics, and modified inhibition kinetics described
    in Schlattmann (2011) / SIMBA# biogas 4.2 Tutorial.

    Usage
    -----
    >>> da = ADM1da(feedstock, V_liq=2000, T_ad=308.15)
    >>> da.set_influent_dataframe(influent_df)   # 38-column DataFrame
    >>> da.create_influent(Q=[15.0, 10.0], i=0)
    >>> state0 = [0.01] * 41
    >>> dydt = da.ADM_ODE(0.0, state0)
    """

    def __init__(
        self,
        feedstock: Feedstock,
        V_liq: float = 1977.0,
        V_gas: float = 304.0,
        T_ad: float = 308.15,
    ) -> None:
        """
        Initialize the ADM1da model.

        Parameters
        ----------
        feedstock : Feedstock
            Feedstock object (used only for ``create_influent`` if an external
            influent DataFrame is not provided via ``set_influent_dataframe``).
        V_liq : float
            Liquid digester volume [m³].
        V_gas : float
            Gas headspace volume [m³].
        T_ad : float
            Operating temperature [K] (default 308.15 K = 35 °C).
        """
        super().__init__(feedstock, V_liq, V_gas, T_ad)

        # Pre-compute temperature-corrected kinetics (reused across ODE calls)
        base_kinetic = ADM1daParams.get_kinetic_params()
        theta = ADM1daParams.get_temperature_factors()
        self._kinetic = ADM1daParams.apply_temperature_corrections(base_kinetic, theta, T_ad)

        # Stoichiometric and inhibition parameters
        self._stoich = ADM1daParams.get_stoichiometric_params()
        self._fractions = ADM1daParams.get_product_fractions()
        self._inhib_params = ADM1daParams.get_inhibition_params(self._R, self._T_base, T_ad)

        # Gas parameters (shared with ADM1)
        p_gas_h2o, k_p, k_L_a, K_H_co2, K_H_ch4, K_H_h2 = ADMParams.getADMgasparams(self._R, self._T_base, T_ad)
        self._p_gas_h2o = p_gas_h2o  # saturation water vapour pressure [bar]
        self._k_p = k_p
        self._k_L_a = k_L_a
        self._K_H_co2 = K_H_co2
        self._K_H_ch4 = K_H_ch4
        self._K_H_h2 = K_H_h2

        # SIMBA# gas volume reference conditions (theta = 20 °C, 1 atm)
        # NQ = 1000 * P_norm / (R * T_norm) [kmol/Nm³] — inverse molar volume at ref. T
        self._T_gas_norm = 293.15  # reference temperature [K] (20 °C)
        self._P_gas_norm = 1.01325  # reference pressure    [bar]
        self._NQ = 1000.0 * self._P_gas_norm / (self._R * self._T_gas_norm)  # ≈ 41.57 kmol/Nm³

        # Pressure threshold for linearised gas-outlet pipe formula [bar] (100 Pa)
        # The ADM1 outlet is already linear (q ∝ Δp), so no formula change is
        # needed; this is the minimum Δp = p_gas − p_ext for gas to flow.
        self._dP_min_pipe = 1.0e-3  # bar

        # Optional external influent DataFrame (set by set_influent_dataframe)
        self._influent_df: Optional[pd.DataFrame] = None

        # Influent and sludge densities for volume-balance hydraulics.
        # Default 1000 kg/m³ gives D_out == D_in (backward compatible).
        self._rho_in: float = 1000.0  # weighted influent density [kg/m³]
        self._rho_sludge: float = 1000.0  # sludge effluent density [kg/m³]

    # ------------------------------------------------------------------
    # ADMBase abstract interface
    # ------------------------------------------------------------------

    def get_state_size(self) -> int:
        """Return the number of state variables: 41 for ADM1da."""
        return STATE_SIZE

    @property
    def model_name(self) -> str:
        """Short identifier for this model variant."""
        return "ADM1da"

    def calc_gas(
        self,
        pi_Sh2: float,
        pi_Sch4: float,
        pi_Sco2: float,
        pTOTAL: float,
    ):
        """
        Calculate biogas production rates, normalised to SIMBA# reference
        conditions (theta = 20 °C, P = 1.01325 bar).

        Overrides adm_base.calc_gas to use the SIMBA# gas temperature reference
        instead of the 0 °C reference hardcoded in the base class.  Water
        vapour is included in the total pressure and mole fractions so that
        CH₄/CO₂ fractions match SIMBA# wet-basis reporting.

        Parameters
        ----------
        pi_Sh2, pi_Sch4, pi_Sco2 : float
            Partial pressures of H₂, CH₄, CO₂ [bar].
        pTOTAL : float
            Total dry gas pressure (H₂ + CH₄ + CO₂) [bar].

        Returns
        -------
        q_gas  : float   Total biogas flow [Nm³/d at theta = 20 °C, 1 atm]
        q_ch4  : float   CH₄ flow         [Nm³/d]
        q_co2  : float   CO₂ flow         [Nm³/d]
        q_h2o  : float   H₂O vapour flow  [Nm³/d]
        p_gas  : float   Dry partial pressure sum (H₂ + CH₄ + CO₂) [bar]
        """
        k_p = self._k_p
        if self._calibration_params.get("k_p") is not None:
            k_p = float(self._calibration_params["k_p"])

        # Wet total pressure includes saturated water vapour.
        # This drives the gas outlet in the same way as in SIMBA#.
        p_total_wet = pTOTAL + self._p_gas_h2o

        # Nm³/d at reference conditions (theta = 20 °C):
        q_gas = max(
            k_p * (p_total_wet - self._p_ext) / (self._RT / 1000.0 * self._NQ) * self.V_liq,
            0.0,
        )

        # Dry pressure sum (H₂ + CH₄ + CO₂) — returned for backward compatibility
        p_gas = pi_Sh2 + pi_Sch4 + pi_Sco2

        # Mole fractions on wet basis (includes H₂O)
        p_gas_wet = p_gas + self._p_gas_h2o
        if p_gas_wet > 0.0:
            q_ch4 = max(q_gas * (pi_Sch4 / p_gas_wet), 0.0)
            q_co2 = max(q_gas * (pi_Sco2 / p_gas_wet), 0.0)
            q_h2o = max(q_gas * (self._p_gas_h2o / p_gas_wet), 0.0)
        else:
            q_ch4 = 0.0
            q_co2 = 0.0
            q_h2o = 0.0

        return q_gas, q_ch4, q_co2, q_h2o, p_gas

    def create_influent(
        self,
        Q: List[float],
        i: int,
        rho: Optional[List[float]] = None,
    ) -> None:
        """
        Build the ADM1da influent vector for time step *i*.

        If an external influent DataFrame has been set via
        ``set_influent_dataframe()``, that DataFrame is used.  Otherwise the
        feedstock object is used to derive a baseline ADM1 influent and the
        particulate columns are mapped to the ADM1da sub-fraction layout.

        Parameters
        ----------
        Q : List[float]
            Volumetric substrate flow rates [m³ d⁻¹].
        i : int
            Time-step index.
        rho : List[float], optional
            Fresh-matter densities [kg/m³] for each substrate in *Q*.
            When provided the weighted-average influent density ``_rho_in``
            is updated, which feeds into the volume-balance Q_out calculation.
            If omitted, ``_rho_in`` retains its previous value (default 1000).
        """
        self._Q = Q
        if rho is not None and len(rho) == len(Q):
            q_total = sum(Q)
            if q_total > 0.0:
                self._rho_in = sum(q * r for q, r in zip(Q, rho)) / q_total

        if self._influent_df is not None:
            # Use the externally provided ADM1da influent DataFrame directly.
            max_i = len(self._influent_df) - 1
            row = self._influent_df.iloc[min(i, max_i)]
            self._state_input = row[INFLUENT_COLUMNS[:-1]].tolist()
        else:
            # Fall back: derive from ADM1 feedstock and remap particulates.
            adm1_df = self._feedstock.get_influent_dataframe(Q)
            self._state_input = self._map_adm1_influent_to_adm1da(adm1_df, i)

    def ADM_ODE(self, t: float, state: List[float]) -> Tuple[float, ...]:
        """
        Compute dy/dt for the 41-element ADM1da state vector.

        Parameters
        ----------
        t : float
            Current time [days] (system is autonomous; present for solver API).
        state : List[float]
            Current state vector (41 elements).

        Returns
        -------
        Tuple[float, ...]
            41 derivatives (dy/dt).
        """
        # ---- Unpack state ------------------------------------------------
        S_su = state[_IDX_S_SU]
        S_aa = state[_IDX_S_AA]
        S_fa = state[_IDX_S_FA]
        S_va = state[_IDX_S_VA]
        S_bu = state[_IDX_S_BU]
        S_pro = state[_IDX_S_PRO]
        S_ac = state[_IDX_S_AC]
        S_h2 = state[_IDX_S_H2]
        S_ch4 = state[_IDX_S_CH4]
        S_co2 = state[_IDX_S_CO2]
        S_nh4 = state[_IDX_S_NH4]
        S_I = state[_IDX_S_I]

        X_PS_ch = state[_IDX_X_PS_CH]
        X_PS_pr = state[_IDX_X_PS_PR]
        X_PS_li = state[_IDX_X_PS_LI]
        X_PF_ch = state[_IDX_X_PF_CH]
        X_PF_pr = state[_IDX_X_PF_PR]
        X_PF_li = state[_IDX_X_PF_LI]
        X_S_ch = state[_IDX_X_S_CH]
        X_S_pr = state[_IDX_X_S_PR]
        X_S_li = state[_IDX_X_S_LI]
        X_I = state[_IDX_X_I]

        X_su = state[_IDX_X_SU]
        X_aa = state[_IDX_X_AA]
        X_fa = state[_IDX_X_FA]
        X_c4 = state[_IDX_X_C4]
        X_pro = state[_IDX_X_PRO]
        X_ac = state[_IDX_X_AC]
        X_h2 = state[_IDX_X_H2]

        S_cation = state[_IDX_S_CATION]
        S_anion = state[_IDX_S_ANION]
        S_va_ion = state[_IDX_S_VA_ION]
        S_bu_ion = state[_IDX_S_BU_ION]
        S_pro_ion = state[_IDX_S_PRO_ION]
        S_ac_ion = state[_IDX_S_AC_ION]
        S_hco3 = state[_IDX_S_HCO3]
        S_nh3 = state[_IDX_S_NH3]

        p_gas_h2 = state[_IDX_P_H2]
        p_gas_ch4 = state[_IDX_P_CH4]
        p_gas_co2 = state[_IDX_P_CO2]
        pTOTAL = state[_IDX_P_TOTAL]

        # ---- Influent and hydraulic flow ---------------------------------
        q_ad = float(np.sum(self._Q)) if self._Q is not None else 0.0
        s_in = self._state_input if self._state_input is not None else [0.0] * 37

        # ---- Parameters --------------------------------------------------
        k = self._kinetic  # temperature-corrected kinetics
        st = self._stoich
        fr = self._fractions
        ip = self._inhib_params

        f_fa_li = st["f_fa_li"]
        f_ch_bac = st["f_ch_bac"]
        f_pr_bac = st["f_pr_bac"]
        f_li_bac = st["f_li_bac"]
        f_p_bac = st["f_p_bac"]
        fXI_PS = st["fXI_PS"]
        fXI_PF = st["fXI_PF"]
        fSI = st["fSI_hyd"]

        Y_su = k["Y_su"]
        Y_aa = k["Y_aa"]
        Y_fa = k["Y_fa"]
        Y_c4 = k["Y_c4"]
        Y_pro = k["Y_pro"]
        Y_ac = k["Y_ac"]
        Y_h2 = k["Y_h2"]

        # ---- pH (Newton–Raphson charge balance) --------------------------
        S_H = self._calc_ph(S_nh4, S_nh3, S_hco3, S_ac_ion, S_pro_ion, S_bu_ion, S_va_ion, S_cation, S_anion, ip["K_w"])
        S_H = max(S_H, 1.0e-14)

        # ---- Inhibition factors ------------------------------------------
        # Total inorganic nitrogen for N-limitation (ADM1da: S_IN = nh4 + nh3)
        S_IN = S_nh4 + S_nh3
        I_IN = S_IN / (ip["K_S_IN"] + S_IN + 1.0e-20)

        # pH inhibition (Hill function) – different exponents per group
        I_pH_aa = self._pH_inhib(S_H, ip["K_pH_aa"], n=1)
        I_pH_fa = self._pH_inhib(S_H, ip["K_pH_aa"], n=2)  # squared
        I_pH_c4 = self._pH_inhib(S_H, ip["K_pH_aa"], n=2)
        I_pH_pro = self._pH_inhib(S_H, ip["K_pH_aa"], n=2)
        I_pH_ac = self._pH_inhib(S_H, ip["K_pH_ac"], n=3)  # cubic
        I_pH_h2 = self._pH_inhib(S_H, ip["K_pH_h2"], n=1)

        # H2 inhibition
        I_h2_fa = ip["K_I_h2_fa"] / (ip["K_I_h2_fa"] + S_h2)
        I_h2_c4 = ip["K_I_h2_c4"] / (ip["K_I_h2_c4"] + S_h2)
        I_h2_pro = ip["K_I_h2_pro"] / (ip["K_I_h2_pro"] + S_h2)

        # NH3 inhibition (X_ac and X_pro separately) — squared Hill form per
        # SIMBA# biogas 4.2 tutorial §7.4–7.7: K^2/(K^2 + S_nh3^2). Sharper
        # cut-off above K than the linear ADM1 form; required to reproduce
        # SIMBA# acetate/propionate accumulation at thermophilic operation.
        K_nh3_ac_sq = ip["K_I_nh3"] ** 2
        K_nh3_pro_sq = ip["K_I_nh3_pro"] ** 2
        S_nh3_sq = S_nh3 * S_nh3
        I_nh3 = K_nh3_ac_sq / (K_nh3_ac_sq + S_nh3_sq)
        I_nh3_pro = K_nh3_pro_sq / (K_nh3_pro_sq + S_nh3_sq)

        # CO2 limitation for H2 methanogens (SIMBA#: inorganic carbon as CO2 source)
        I_co2_h2 = S_co2 / (ip["K_S_co2_h2"] + S_co2 + 1.0e-20)

        # Undissociated propionic acid inhibition [kmol m⁻³]
        Ka_pro = ip["K_a_pro"]
        S_HPr = (S_pro / 112.0) * S_H / (S_H + Ka_pro + 1.0e-20)
        I_HPr = ip["K_IH_pro"] / (ip["K_IH_pro"] + S_HPr + 1.0e-20)

        # Undissociated acetic acid inhibition [kmol m⁻³]
        Ka_ac = ip["K_a_ac"]
        S_HAc = (S_ac / 64.0) * S_H / (S_H + Ka_ac + 1.0e-20)
        I_HAc = ip["K_IH_ac"] / (ip["K_IH_ac"] + S_HAc + 1.0e-20)

        # Acetate competitive inhibition (for X_fa and X_c4 only per SIMBA#)
        I_ac_fa = ip["K_I_ac_xfa"] / (ip["K_I_ac_xfa"] + S_ac)
        I_ac_c4 = ip["K_I_ac_xc4"] / (ip["K_I_ac_xc4"] + S_ac)

        # Combined inhibition per process
        I_su = I_pH_aa * I_IN
        I_aa = I_pH_aa * I_IN
        I_fa = I_pH_fa * I_IN * I_h2_fa * I_ac_fa
        I_c4 = I_pH_c4 * I_IN * I_h2_c4 * I_HPr * I_ac_c4
        I_pro = I_pH_pro * I_IN * I_h2_pro * I_HPr * I_nh3_pro
        I_ac = I_pH_ac * I_IN * I_nh3 * I_HAc
        I_h2 = I_pH_h2 * I_IN * I_co2_h2

        # ---- Process rates -----------------------------------------------
        # Disintegration (first-order, temperature-corrected)
        Rho_dis_PS_ch = k["k_dis_PS"] * X_PS_ch
        Rho_dis_PS_pr = k["k_dis_PS"] * X_PS_pr
        Rho_dis_PS_li = k["k_dis_PS"] * X_PS_li
        Rho_dis_PF_ch = k["k_dis_PF"] * X_PF_ch
        Rho_dis_PF_pr = k["k_dis_PF"] * X_PF_pr
        Rho_dis_PF_li = k["k_dis_PF"] * X_PF_li

        # Hydrolysis (first-order, temperature-corrected)
        Rho_hyd_ch = k["k_hyd_ch"] * X_S_ch
        Rho_hyd_pr = k["k_hyd_pr"] * X_S_pr
        Rho_hyd_li = k["k_hyd_li"] * X_S_li

        # Uptake rates (Monod kinetics)
        Rho_su = k["k_m_su"] * S_su / (k["K_S_su"] + S_su + 1.0e-20) * X_su * I_su

        Rho_aa = k["k_m_aa"] * S_aa / (k["K_S_aa"] + S_aa + 1.0e-20) * X_aa * I_aa

        Rho_fa = k["k_m_fa"] * S_fa / (k["K_S_fa"] + S_fa + 1.0e-20) * X_fa * I_fa

        # X_c4 splits into valerate and butyrate uptake (competitive)
        S_vbu = S_va + S_bu + 1.0e-20
        Rho_c4_va = k["k_m_c4"] * S_va / (k["K_S_c4"] + S_va + 1.0e-20) * X_c4 * (S_va / S_vbu) * I_c4
        Rho_c4_bu = k["k_m_c4"] * S_bu / (k["K_S_c4"] + S_bu + 1.0e-20) * X_c4 * (S_bu / S_vbu) * I_c4

        Rho_pro = k["k_m_pro"] * S_pro / (k["K_S_pro"] + S_pro + 1.0e-20) * X_pro * I_pro

        Rho_ac = k["k_m_ac"] * S_ac / (k["K_S_ac"] + S_ac + 1.0e-20) * X_ac * I_ac

        Rho_h2 = k["k_m_h2"] * S_h2 / (k["K_S_h2"] + S_h2 + 1.0e-20) * X_h2 * I_h2

        # Decay (per-organism temperature-corrected rates, SIMBA# values)
        Rho_dec_su = k["k_dec_su"] * X_su
        Rho_dec_aa = k["k_dec_aa"] * X_aa
        Rho_dec_fa = k["k_dec_fa"] * X_fa
        Rho_dec_c4 = k["k_dec_c4"] * X_c4
        Rho_dec_pro = k["k_dec_pro"] * X_pro
        Rho_dec_ac = k["k_dec_ac"] * X_ac
        Rho_dec_h2 = k["k_dec_h2"] * X_h2
        sum_decay = Rho_dec_su + Rho_dec_aa + Rho_dec_fa + Rho_dec_c4 + Rho_dec_pro + Rho_dec_ac + Rho_dec_h2

        # ---- Acid-base rates --------------------------------------------
        k_AB = ip["k_A_B"]
        Rho_A_va = k_AB * (S_va_ion * S_H - ip["K_a_va"] * (S_va - S_va_ion))
        Rho_A_bu = k_AB * (S_bu_ion * S_H - ip["K_a_bu"] * (S_bu - S_bu_ion))
        Rho_A_pro = k_AB * (S_pro_ion * S_H - ip["K_a_pro"] * (S_pro - S_pro_ion))
        Rho_A_ac = k_AB * (S_ac_ion * S_H - ip["K_a_ac"] * (S_ac - S_ac_ion))
        Rho_A_co2 = k_AB * (S_hco3 * S_H - ip["K_a_co2"] * (S_co2 - S_hco3))
        Rho_A_IN = k_AB * (S_nh3 * S_H - ip["K_a_IN"] * (S_nh4 - S_nh3))

        # ---- Gas transfer rates -----------------------------------------
        k_L_a = self._k_L_a
        if self._calibration_params.get("k_L_a") is not None:
            k_L_a = float(self._calibration_params["k_L_a"])

        # S_eq = p * COD_per_mole / (RT * K_H)  (matches adm_equations.gas_transfer_rate)
        Rho_T_h2 = k_L_a * (S_h2 - 16.0 * p_gas_h2 / (self._RT * self._K_H_h2)) * (self.V_liq / self._V_gas)
        Rho_T_ch4 = k_L_a * (S_ch4 - 64.0 * p_gas_ch4 / (self._RT * self._K_H_ch4)) * (self.V_liq / self._V_gas)
        # CO2 gas transfer uses dissolved free CO2 only (not bicarbonate):
        #   S_co2_free = S_IC - S_hco3  (HCO3- is non-volatile)
        # This matches SIMBA#'s formulation and prevents spurious alkalinity loss
        # when starting from a buffered liquid with an air headspace.
        S_co2_free = max(S_co2 - S_hco3, 0.0)
        Rho_T_co2 = k_L_a * (S_co2_free - p_gas_co2 / (self._RT * self._K_H_co2)) * (self.V_liq / self._V_gas)

        # Total gas outlet (linearised pipe formula, SIMBA# dP_min_pipe = 100 Pa)
        # The outlet is linear in Δp so no singularity exists near Δp=0;
        # the dP_min_pipe threshold is satisfied by clamping at zero.
        # Water vapour pressure (constant at given T) contributes to the driving
        # pressure for gas flow, exactly as in SIMBA#.
        k_p = self._k_p
        if self._calibration_params.get("k_p") is not None:
            k_p = float(self._calibration_params["k_p"])
        Rho_T_11 = max(k_p * (pTOTAL + self._p_gas_h2o - self._p_ext) * (self.V_liq / self._V_gas), 0.0)

        # ---- Carbon stoichiometry coefficients for S_co2 balance --------
        C = st
        s_hyd_ch = -C["C_ch"] + (1.0 - fSI) * C["C_su"] + fSI * C["C_I_s"]
        s_hyd_pr = -C["C_pr"] + (1.0 - fSI) * C["C_aa"] + fSI * C["C_I_s"]
        # Lipid hydrolysis: → ffa*S_fa + (1-ffa)*S_su (+ SI fraction)
        s_hyd_li = -C["C_li"] + (1.0 - fSI) * (f_fa_li * C["C_fa"] + (1.0 - f_fa_li) * C["C_su"]) + fSI * C["C_I_s"]

        s_su = (
            -C["C_su"]
            + (1.0 - Y_su) * (fr["f_bu_su"] * C["C_bu"] + fr["f_pro_su"] * C["C_pro"] + fr["f_ac_su"] * C["C_ac"])
            + Y_su * C["C_bac"]
        )
        s_aa = (
            -C["C_aa"]
            + (1.0 - Y_aa)
            * (fr["f_va_aa"] * C["C_va"] + fr["f_bu_aa"] * C["C_bu"] + fr["f_pro_aa"] * C["C_pro"] + fr["f_ac_aa"] * C["C_ac"])
            + Y_aa * C["C_bac"]
        )
        s_fa = -C["C_fa"] + (1.0 - Y_fa) * 0.7 * C["C_ac"] + Y_fa * C["C_bac"]
        s_c4_va = -C["C_va"] + (1.0 - Y_c4) * (0.54 * C["C_pro"] + 0.31 * C["C_ac"]) + Y_c4 * C["C_bac"]
        s_c4_bu = -C["C_bu"] + (1.0 - Y_c4) * 0.8 * C["C_ac"] + Y_c4 * C["C_bac"]
        s_pro = -C["C_pro"] + (1.0 - Y_pro) * 0.57 * C["C_ac"] + Y_pro * C["C_bac"]
        s_ac = -C["C_ac"] + (1.0 - Y_ac) * C["C_ch4"] + Y_ac * C["C_bac"]
        s_h2 = (1.0 - Y_h2) * C["C_ch4"] + Y_h2 * C["C_bac"]
        # Decay: biomass → f_ch*XPS_ch + f_pr*XPS_pr + f_li*XPS_li + f_p*X_I
        s_dec = -C["C_bac"] + f_ch_bac * C["C_ch"] + f_pr_bac * C["C_pr"] + f_li_bac * C["C_li"] + f_p_bac * C["C_I_x"]

        Sigma = (
            s_hyd_ch * Rho_hyd_ch
            + s_hyd_pr * Rho_hyd_pr
            + s_hyd_li * Rho_hyd_li
            + s_su * Rho_su
            + s_aa * Rho_aa
            + s_fa * Rho_fa
            + s_c4_va * Rho_c4_va
            + s_c4_bu * Rho_c4_bu
            + s_pro * Rho_pro
            + s_ac * Rho_ac
            + s_h2 * Rho_h2
            + s_dec * sum_decay
        )

        # ---- Differential equations -------------------------------------
        D_in = q_ad / self.V_liq  # feed dilution rate [d⁻¹]

        # Sludge volume loss — SIMBA# biogas 4.2 §5.1 Approach 2:
        #   q_S_loss = V_liq × Σ(r_hyd_i × iM_i / ρ_i)
        # Hydrolysis is the step that converts solid fractions (density ρ)
        # to dissolved products.  Table 3 of the SIMBA# manual:
        #   CH: iM = 0.9375 kg kg⁻¹COD,  ρ = 1550 kg m⁻³
        #   PR: iM = 0.6125 kg kg⁻¹COD,  ρ = 1370 kg m⁻³
        #   LI: iM = 0.3474 kg kg⁻¹COD,  ρ =  920 kg m⁻³
        _q_S_loss = self.V_liq * (
            Rho_hyd_ch * (0.9375 / 1550.0) + Rho_hyd_pr * (0.6125 / 1370.0) + Rho_hyd_li * (0.3474 / 920.0)
        )
        _Q_out = max(q_ad - _q_S_loss, 0.0)
        D_out = _Q_out / self.V_liq  # washout dilution rate [d⁻¹]

        # --- Dissolved (0–11) ---
        diff_S_su = (
            D_in * s_in[0] - D_out * S_su + (1.0 - fSI) * Rho_hyd_ch + (1.0 - fSI) * (1.0 - f_fa_li) * Rho_hyd_li - Rho_su
        )

        diff_S_aa = D_in * s_in[1] - D_out * S_aa + (1.0 - fSI) * Rho_hyd_pr - Rho_aa

        diff_S_fa = D_in * s_in[2] - D_out * S_fa + (1.0 - fSI) * f_fa_li * Rho_hyd_li - Rho_fa

        diff_S_va = D_in * s_in[3] - D_out * S_va + (1.0 - Y_aa) * fr["f_va_aa"] * Rho_aa - Rho_c4_va

        diff_S_bu = (
            D_in * s_in[4]
            - D_out * S_bu
            + (1.0 - Y_su) * fr["f_bu_su"] * Rho_su
            + (1.0 - Y_aa) * fr["f_bu_aa"] * Rho_aa
            - Rho_c4_bu
        )

        diff_S_pro = (
            D_in * s_in[5]
            - D_out * S_pro
            + (1.0 - Y_su) * fr["f_pro_su"] * Rho_su
            + (1.0 - Y_aa) * fr["f_pro_aa"] * Rho_aa
            + (1.0 - Y_c4) * 0.54 * Rho_c4_va
            - Rho_pro
        )

        diff_S_ac = (
            D_in * s_in[6]
            - D_out * S_ac
            + (1.0 - Y_su) * fr["f_ac_su"] * Rho_su
            + (1.0 - Y_aa) * fr["f_ac_aa"] * Rho_aa
            + (1.0 - Y_fa) * 0.7 * Rho_fa
            + (1.0 - Y_c4) * 0.31 * Rho_c4_va
            + (1.0 - Y_c4) * 0.8 * Rho_c4_bu
            + (1.0 - Y_pro) * 0.57 * Rho_pro
            - Rho_ac
        )

        diff_S_h2 = (
            D_in * s_in[7]
            - D_out * S_h2
            + (1.0 - Y_su) * fr["f_h2_su"] * Rho_su
            + (1.0 - Y_aa) * fr["f_h2_aa"] * Rho_aa
            + (1.0 - Y_fa) * 0.3 * Rho_fa
            + (1.0 - Y_c4) * 0.15 * Rho_c4_va
            + (1.0 - Y_c4) * 0.2 * Rho_c4_bu
            + (1.0 - Y_pro) * 0.43 * Rho_pro
            - Rho_h2
            - self._V_gas / self.V_liq * Rho_T_h2
        )

        diff_S_ch4 = (
            D_in * s_in[8]
            - D_out * S_ch4
            + (1.0 - Y_ac) * Rho_ac
            + (1.0 - Y_h2) * Rho_h2
            - self._V_gas / self.V_liq * Rho_T_ch4
        )

        diff_S_co2 = D_in * s_in[9] - D_out * S_co2 - Sigma - self._V_gas / self.V_liq * Rho_T_co2 + Rho_A_co2

        # Nitrogen balance
        N_bac = st["N_bac"]
        N_aa = st["N_aa"]
        N_I = st["N_I"]
        # N released to S_NH4 from decay = N_bac - N_aa*f_pr_bac - N_I*f_p_bac
        # (ch and li fractions have no nitrogen; residue X_I has N_I content)
        diff_S_nh4 = (
            D_in * s_in[10]
            - D_out * S_nh4
            - Y_su * N_bac * Rho_su
            + (N_aa - Y_aa * N_bac) * Rho_aa
            - Y_fa * N_bac * Rho_fa
            - Y_c4 * N_bac * Rho_c4_va
            - Y_c4 * N_bac * Rho_c4_bu
            - Y_pro * N_bac * Rho_pro
            - Y_ac * N_bac * Rho_ac
            - Y_h2 * N_bac * Rho_h2
            + (N_bac - f_pr_bac * N_aa - f_p_bac * N_I) * sum_decay
            + Rho_A_IN
        )

        diff_S_I = D_in * s_in[11] - D_out * S_I + fSI * (Rho_hyd_ch + Rho_hyd_pr + Rho_hyd_li)

        # --- Particulate sub-fractions (12–21) ---
        diff_X_PS_ch = D_in * s_in[12] - D_out * X_PS_ch - Rho_dis_PS_ch + f_ch_bac * sum_decay

        diff_X_PS_pr = D_in * s_in[13] - D_out * X_PS_pr - Rho_dis_PS_pr + f_pr_bac * sum_decay

        diff_X_PS_li = D_in * s_in[14] - D_out * X_PS_li - Rho_dis_PS_li + f_li_bac * sum_decay

        diff_X_PF_ch = D_in * s_in[15] - D_out * X_PF_ch - Rho_dis_PF_ch

        diff_X_PF_pr = D_in * s_in[16] - D_out * X_PF_pr - Rho_dis_PF_pr

        diff_X_PF_li = D_in * s_in[17] - D_out * X_PF_li - Rho_dis_PF_li

        diff_X_S_ch = (
            D_in * s_in[18] - D_out * X_S_ch + (1.0 - fXI_PS) * Rho_dis_PS_ch + (1.0 - fXI_PF) * Rho_dis_PF_ch - Rho_hyd_ch
        )

        diff_X_S_pr = (
            D_in * s_in[19] - D_out * X_S_pr + (1.0 - fXI_PS) * Rho_dis_PS_pr + (1.0 - fXI_PF) * Rho_dis_PF_pr - Rho_hyd_pr
        )

        diff_X_S_li = (
            D_in * s_in[20] - D_out * X_S_li + (1.0 - fXI_PS) * Rho_dis_PS_li + (1.0 - fXI_PF) * Rho_dis_PF_li - Rho_hyd_li
        )

        diff_X_I = (
            D_in * s_in[21]
            - D_out * X_I
            + fXI_PS * (Rho_dis_PS_ch + Rho_dis_PS_pr + Rho_dis_PS_li)
            + fXI_PF * (Rho_dis_PF_ch + Rho_dis_PF_pr + Rho_dis_PF_li)
            + f_p_bac * sum_decay
        )

        # --- Biomass (22–28) ---
        diff_X_su = D_in * s_in[22] - D_out * X_su + Y_su * Rho_su - Rho_dec_su
        diff_X_aa = D_in * s_in[23] - D_out * X_aa + Y_aa * Rho_aa - Rho_dec_aa
        diff_X_fa = D_in * s_in[24] - D_out * X_fa + Y_fa * Rho_fa - Rho_dec_fa
        diff_X_c4 = D_in * s_in[25] - D_out * X_c4 + Y_c4 * Rho_c4_va + Y_c4 * Rho_c4_bu - Rho_dec_c4
        diff_X_pro = D_in * s_in[26] - D_out * X_pro + Y_pro * Rho_pro - Rho_dec_pro
        diff_X_ac = D_in * s_in[27] - D_out * X_ac + Y_ac * Rho_ac - Rho_dec_ac
        diff_X_h2 = D_in * s_in[28] - D_out * X_h2 + Y_h2 * Rho_h2 - Rho_dec_h2

        # --- Charge balance (29–36) ---
        diff_S_cation = D_in * s_in[29] - D_out * S_cation
        diff_S_anion = D_in * s_in[30] - D_out * S_anion
        diff_S_va_ion = D_in * s_in[31] - D_out * S_va_ion - Rho_A_va
        diff_S_bu_ion = D_in * s_in[32] - D_out * S_bu_ion - Rho_A_bu
        diff_S_pro_ion = D_in * s_in[33] - D_out * S_pro_ion - Rho_A_pro
        diff_S_ac_ion = D_in * s_in[34] - D_out * S_ac_ion - Rho_A_ac
        diff_S_hco3 = D_in * s_in[35] - D_out * S_hco3 - Rho_A_co2
        diff_S_nh3 = D_in * s_in[36] - D_out * S_nh3 - Rho_A_IN

        # --- Gas phase (37–40) ---
        diff_p_h2 = Rho_T_h2 * self._RT / 16.0 - p_gas_h2 / pTOTAL * Rho_T_11
        diff_p_ch4 = Rho_T_ch4 * self._RT / 64.0 - p_gas_ch4 / pTOTAL * Rho_T_11
        diff_p_co2 = Rho_T_co2 * self._RT - p_gas_co2 / pTOTAL * Rho_T_11
        diff_pTOT = self._RT / 16.0 * Rho_T_h2 + self._RT / 64.0 * Rho_T_ch4 + self._RT * Rho_T_co2 - Rho_T_11

        return (
            diff_S_su,
            diff_S_aa,
            diff_S_fa,
            diff_S_va,
            diff_S_bu,
            diff_S_pro,
            diff_S_ac,
            diff_S_h2,
            diff_S_ch4,
            diff_S_co2,
            diff_S_nh4,
            diff_S_I,
            diff_X_PS_ch,
            diff_X_PS_pr,
            diff_X_PS_li,
            diff_X_PF_ch,
            diff_X_PF_pr,
            diff_X_PF_li,
            diff_X_S_ch,
            diff_X_S_pr,
            diff_X_S_li,
            diff_X_I,
            diff_X_su,
            diff_X_aa,
            diff_X_fa,
            diff_X_c4,
            diff_X_pro,
            diff_X_ac,
            diff_X_h2,
            diff_S_cation,
            diff_S_anion,
            diff_S_va_ion,
            diff_S_bu_ion,
            diff_S_pro_ion,
            diff_S_ac_ion,
            diff_S_hco3,
            diff_S_nh3,
            diff_p_h2,
            diff_p_ch4,
            diff_p_co2,
            diff_pTOT,
        )

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def set_influent_dataframe(self, df: pd.DataFrame) -> None:
        """
        Store an external ADM1da influent DataFrame.

        The DataFrame must have columns matching ``INFLUENT_COLUMNS`` (37
        liquid-state columns + Q).  Once set, ``create_influent()`` reads
        from this DataFrame instead of deriving values from the feedstock.

        Parameters
        ----------
        df : pd.DataFrame
            Influent composition over time (one row per feeding event).

        Raises
        ------
        ValueError
            If any required column is missing.
        """
        missing = [c for c in INFLUENT_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Influent DataFrame missing columns: {missing}")
        self._influent_df = df.reset_index(drop=True)

    def set_influent_density(self, rho_in: float, rho_sludge: float = 1000.0) -> None:
        """
        Retained for API compatibility; no longer affects Q_out.

        Q_out is now computed from SIMBA# §5.1 Approach 2 (hydrolysis-rate
        volume balance) and does not depend on influent or sludge densities.

        Parameters
        ----------
        rho_in    : float  Weighted-average influent density [kg/m³] (unused).
        rho_sludge: float  Sludge effluent density [kg/m³] (unused).
        """
        self._rho_in = float(rho_in)
        self._rho_sludge = float(rho_sludge)

    def print_params_at_current_state(self, state: List[float]) -> None:
        """
        Calculate and store process indicators from the current state.

        Uses the pure-Python pH solver (no DLL dependency).

        Parameters
        ----------
        state : List[float]
            Current 41-element ADM1da state vector.
        """
        ip = self._inhib_params
        S_H = self._calc_ph(
            state[_IDX_S_NH4],
            state[_IDX_S_NH3],
            state[_IDX_S_HCO3],
            state[_IDX_S_AC_ION],
            state[_IDX_S_PRO_ION],
            state[_IDX_S_BU_ION],
            state[_IDX_S_VA_ION],
            state[_IDX_S_CATION],
            state[_IDX_S_ANION],
            ip["K_w"],
        )
        pH = -np.log10(max(S_H, 1.0e-14))
        self._track_pH(round(pH, 1))

        q_gas, q_ch4, q_co2, q_h2o, p_gas = self.calc_gas(
            state[_IDX_P_H2],
            state[_IDX_P_CH4],
            state[_IDX_P_CO2],
            state[_IDX_P_TOTAL],
        )
        self._track_gas(q_gas, q_ch4, q_co2, q_h2o, p_gas)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pH_inhib(S_H: float, K_pH: float, n: int = 1) -> float:
        """
        Hill-type pH inhibition factor.

        I = K_pH^n / (K_pH^n + S_H^n)

        Parameters
        ----------
        S_H  : float   [H+] concentration [kmol m⁻³]
        K_pH : float   Half-inhibition H+ concentration [kmol m⁻³]
        n    : int     Exponent (1=standard, 2=squared, 3=cubic)
        """
        Kn = K_pH**n
        SHn = S_H**n
        return Kn / (Kn + SHn)

    @staticmethod
    def _calc_ph(
        S_nh4: float,
        S_nh3: float,
        S_hco3: float,
        S_ac_ion: float,
        S_pro_ion: float,
        S_bu_ion: float,
        S_va_ion: float,
        S_cation: float,
        S_anion: float,
        K_w: float,
        max_iter: int = 50,
    ) -> float:
        """
        Solve charge balance for [H+] using Newton–Raphson iteration.

        Charge balance:
            S_cation + S_H + (S_nh4 − S_nh3)
            = S_anion + K_w/S_H + S_hco3
              + S_ac_ion/64 + S_pro_ion/112 + S_bu_ion/160 + S_va_ion/208

        All concentrations in kmol m⁻³ (VFA ions converted from kg COD m⁻³
        using their COD-equivalent molar masses).

        Parameters
        ----------
        S_nh4, S_nh3 : float   total and free ammonium-N [kmol N m⁻³]
        S_hco3       : float   bicarbonate [kmol C m⁻³]
        S_ac_ion     : float   acetate ion [kg COD m⁻³]
        S_pro_ion    : float   propionate ion [kg COD m⁻³]
        S_bu_ion     : float   butyrate ion [kg COD m⁻³]
        S_va_ion     : float   valerate ion [kg COD m⁻³]
        S_cation     : float   strong-base cations [kmol m⁻³]
        S_anion      : float   strong-acid anions [kmol m⁻³]
        K_w          : float   water dissociation constant [kmol² m⁻⁶]
        max_iter     : int     maximum Newton–Raphson iterations

        Returns
        -------
        float
            [H+] concentration [kmol m⁻³] (= [M]).
        """
        # Convert VFA ions from kg COD/m³ to kmol/m³
        vfa_anions = S_ac_ion / 64.0 + S_pro_ion / 112.0 + S_bu_ion / 160.0 + S_va_ion / 208.0

        # Net fixed-charge offset
        fixed = S_cation - S_anion + (S_nh4 - S_nh3) - S_hco3 - vfa_anions

        # f(S_H)  = fixed + S_H − K_w/S_H = 0
        # f'(S_H) = 1 + K_w/S_H²
        S_H = 1.0e-7  # initial guess (pH 7)
        for _ in range(max_iter):
            f = fixed + S_H - K_w / (S_H + 1.0e-30)
            df = 1.0 + K_w / (S_H + 1.0e-30) ** 2
            delta = -f / df
            S_H = max(1.0e-14, S_H + delta)
            if abs(delta) < 1.0e-15:
                break
        return S_H

    def _map_adm1_influent_to_adm1da(self, adm1_df: pd.DataFrame, i: int) -> List[float]:
        """
        Map an ADM1 influent DataFrame row to the ADM1da state-input layout.

        Particulate mapping (heuristic fallback):
          - X_ch, X_pr, X_li → X_S_ch, X_S_pr, X_S_li (already hydrolysable)
          - X_xc (composite) → split proportionally into X_PS_ch/pr/li
          - X_I → X_I (unchanged)
          - X_p → discarded (no particulate product in ADM1da)

        Parameters
        ----------
        adm1_df : pd.DataFrame
            ADM1-format influent DataFrame (from Feedstock.get_influent_dataframe).
        i : int
            Row index.

        Returns
        -------
        List[float]
            37-element ADM1da liquid-state influent (indices 0–36).
        """
        max_i = len(adm1_df) - 1
        row = adm1_df.iloc[min(i, max_i)]

        # ADM1 composites
        X_xc = float(row.get("X_xc", 0.0))
        X_ch = float(row.get("X_ch", 0.0))
        X_pr = float(row.get("X_pr", 0.0))
        X_li = float(row.get("X_li", 0.0))
        X_I = float(row.get("X_I", 0.0))

        # Map composite to XPS sub-fractions (slow pool) using ADM1 fractions
        total_xc = X_ch + X_pr + X_li + 1.0e-20
        X_PS_ch = X_xc * (X_ch / total_xc)
        X_PS_pr = X_xc * (X_pr / total_xc)
        X_PS_li = X_xc * (X_li / total_xc)

        return [
            float(row.get("S_su", 0.0)),  #  0
            float(row.get("S_aa", 0.0)),  #  1
            float(row.get("S_fa", 0.0)),  #  2
            float(row.get("S_va", 0.0)),  #  3
            float(row.get("S_bu", 0.0)),  #  4
            float(row.get("S_pro", 0.0)),  #  5
            float(row.get("S_ac", 0.0)),  #  6
            float(row.get("S_h2", 0.0)),  #  7
            float(row.get("S_ch4", 0.0)),  #  8
            float(row.get("S_co2", 0.0)),  #  9
            float(row.get("S_nh4", 0.0)),  # 10
            float(row.get("S_I", 0.0)),  # 11
            X_PS_ch,  # 12
            X_PS_pr,  # 13
            X_PS_li,  # 14
            0.0,  # 15  X_PF_ch (no fast pool from ADM1)
            0.0,  # 16  X_PF_pr
            0.0,  # 17  X_PF_li
            X_ch,  # 18  X_S_ch
            X_pr,  # 19  X_S_pr
            X_li,  # 20  X_S_li
            X_I,  # 21
            float(row.get("X_su", 0.0)),  # 22
            float(row.get("X_aa", 0.0)),  # 23
            float(row.get("X_fa", 0.0)),  # 24
            float(row.get("X_c4", 0.0)),  # 25
            float(row.get("X_pro", 0.0)),  # 26
            float(row.get("X_ac", 0.0)),  # 27
            float(row.get("X_h2", 0.0)),  # 28
            float(row.get("S_cation", 0.0)),  # 29
            float(row.get("S_anion", 0.0)),  # 30
            float(row.get("S_va_ion", 0.0)),  # 31
            float(row.get("S_bu_ion", 0.0)),  # 32
            float(row.get("S_pro_ion", 0.0)),  # 33
            float(row.get("S_ac_ion", 0.0)),  # 34
            float(row.get("S_hco3_ion", 0.0)),  # 35
            float(row.get("S_nh3", 0.0)),  # 36
        ]
