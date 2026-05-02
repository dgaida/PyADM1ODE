# pyadm1/core/adm1.py
"""
ADM1da model implementation.

This module implements **ADM1da** (Schlattmann 2011), an agricultural-biogas
extension of ADM1 (Batstone et al. 2002) that replaces the single composite
disintegration variable (X_xc) with a two-pool sub-fraction approach and adds
temperature-dependent kinetics plus modified inhibition.

Independent re-implementation; published parameter values are cited from
Schlattmann (2011) / SIMBA# biogas 4.2 Tutorial.

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

from pyadm1.core.adm_params import ADMParams

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Influent column names — defined here (not in feedstock) so that ADM1 can be
# imported without touching the substrates package, avoiding circular imports.
# --------------------------------------------------------------------------
INFLUENT_COLUMNS = [
    "S_su",
    "S_aa",
    "S_fa",
    "S_va",
    "S_bu",
    "S_pro",
    "S_ac",
    "S_h2",
    "S_ch4",
    "S_co2",
    "S_nh4",
    "S_I",
    "X_PS_ch",
    "X_PS_pr",
    "X_PS_li",
    "X_PF_ch",
    "X_PF_pr",
    "X_PF_li",
    "X_S_ch",
    "X_S_pr",
    "X_S_li",
    "X_I",
    "X_su",
    "X_aa",
    "X_fa",
    "X_c4",
    "X_pro",
    "X_ac",
    "X_h2",
    "S_cation",
    "S_anion",
    "S_va_ion",
    "S_bu_ion",
    "S_pro_ion",
    "S_ac_ion",
    "S_hco3_ion",
    "S_nh3",
    "Q",
]

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
    Load an ADM1 initial state vector from a CSV file.

    The CSV must have columns matching ``INFLUENT_COLUMNS`` (without ``Q``)
    plus the four gas-phase columns (p_gas_h2, p_gas_ch4, p_gas_co2, pTOTAL).
    """
    df = pd.read_csv(csv_file)
    state = []
    liquid_cols = INFLUENT_COLUMNS[:-1]  # drop Q
    for col in liquid_cols:
        state.append(float(df[col].iloc[0]))
    for col in ("p_gas_h2", "p_gas_ch4", "p_gas_co2", "pTOTAL"):
        state.append(float(df[col].iloc[0]))
    return state


class ADM1:
    """
    ADM1da – 41-state ODE system.

    Implements the sub-fraction disintegration/hydrolysis approach,
    temperature-dependent kinetics, and modified inhibition kinetics
    described in Schlattmann (2011); SIMBA# biogas 4.2 Tutorial.

    Usage
    -----
    >>> from pyadm1 import Feedstock
    >>> fs = Feedstock([...], feeding_freq=24)
    >>> adm = ADM1(fs, V_liq=1200, V_gas=216, T_ad=315.15)
    >>> adm.set_influent_dataframe(fs.get_influent_dataframe(Q=[11.4, 6.1]))
    >>> adm.create_influent(Q=[11.4, 6.1], i=0)
    >>> state0 = [0.01] * STATE_SIZE
    >>> dydt = adm.ADM_ODE(0.0, state0)
    """

    def __init__(
        self,
        feedstock,
        V_liq: float = 1977.0,
        V_gas: float = 304.0,
        T_ad: float = 308.15,
    ) -> None:
        """
        Initialize the ADM1 model.

        Parameters
        ----------
        feedstock : Feedstock
            Feedstock object (used by ``create_influent`` when no external
            influent DataFrame has been provided via ``set_influent_dataframe``).
        V_liq : float
            Liquid digester volume [m³].
        V_gas : float
            Gas headspace volume [m³].
        T_ad : float
            Operating temperature [K] (default 308.15 K = 35 °C).
        """
        # --- Reactor volumes ---
        self.V_liq = V_liq
        self._V_gas = V_gas
        self._V_ad = V_liq + V_gas

        # --- Temperature ---
        self._T_ad = T_ad

        # --- Physical constants ---
        self._R = 0.08314  # bar·m³·kmol⁻¹·K⁻¹
        self._T_base = 298.15  # 25 °C reference
        self._p_atm = 1.013

        self._RT = self._R * self._T_ad
        self._p_ext = self._p_atm - 0.0084147 * np.exp(0.054 * (self._T_ad - 273.15))

        # --- Feedstock / influent ---
        self._feedstock = feedstock
        self._Q: Optional[List[float]] = None
        self._state_input: Optional[List[float]] = None

        # --- Calibration overrides ---
        self._calibration_params: dict = {}

        # --- Result-tracking lists ---
        self._Q_GAS: List[float] = []
        self._Q_CH4: List[float] = []
        self._Q_CO2: List[float] = []
        self._Q_H2O: List[float] = []
        self._P_GAS: List[float] = []
        self._pH_l: List[float] = []
        self._FOSTAC: List[float] = []
        self._AcvsPro: List[float] = []
        self._VFA: List[float] = []
        self._TAC: List[float] = []

        # --- Temperature-corrected kinetics (reused across ODE calls) ---
        base_kinetic = ADMParams.get_kinetic_params()
        theta = ADMParams.get_temperature_factors()
        self._kinetic = ADMParams.apply_temperature_corrections(base_kinetic, theta, T_ad)

        # --- Stoichiometric / fraction / inhibition parameters ---
        self._stoich = ADMParams.get_stoichiometric_params()
        self._fractions = ADMParams.get_product_fractions()
        self._inhib_params = ADMParams.get_inhibition_params(self._R, self._T_base, T_ad)

        # --- Gas parameters ---
        p_gas_h2o, k_p, k_L_a, K_H_co2, K_H_ch4, K_H_h2 = ADMParams.getADMgasparams(self._R, self._T_base, T_ad)
        self._p_gas_h2o = p_gas_h2o
        self._k_p = k_p
        self._k_L_a = k_L_a
        self._K_H_co2 = K_H_co2
        self._K_H_ch4 = K_H_ch4
        self._K_H_h2 = K_H_h2

        # Gas volume reference conditions (theta = 20 °C, 1 atm) — ADM1da convention
        self._T_gas_norm = 293.15
        self._P_gas_norm = 1.01325
        self._NQ = 1000.0 * self._P_gas_norm / (self._R * self._T_gas_norm)

        # Pressure threshold for linearised gas-outlet pipe formula [bar].
        self._dP_min_pipe = 1.0e-3

        # Optional external influent DataFrame.
        self._influent_df: Optional[pd.DataFrame] = None

        # Influent / sludge densities (used by ``create_influent`` only).
        self._rho_in: float = 1000.0
        self._rho_sludge: float = 1000.0

    # ------------------------------------------------------------------
    # Public read-only properties
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """Short identifier for this model variant."""
        return "ADM1"

    @property
    def T_ad(self) -> float:
        """Operating temperature [K]."""
        return self._T_ad

    @property
    def feedstock(self):
        """Feedstock object."""
        return self._feedstock

    @property
    def Q_GAS(self) -> List[float]:
        """History of total biogas flow rate [m^3/d]."""
        return self._Q_GAS

    @property
    def Q_CH4(self) -> List[float]:
        """History of methane flow rate [m^3/d]."""
        return self._Q_CH4

    @property
    def Q_CO2(self) -> List[float]:
        """History of carbon dioxide flow rate [m^3/d]."""
        return self._Q_CO2

    @property
    def Q_H2O(self) -> List[float]:
        """History of water-vapour flow rate [m^3/d]."""
        return self._Q_H2O

    @property
    def P_GAS(self) -> List[float]:
        """History of total headspace pressure [bar]."""
        return self._P_GAS

    @property
    def pH_l(self) -> List[float]:
        """History of liquid-phase pH."""
        return self._pH_l

    @property
    def VFA_TA(self) -> List[float]:
        """History of the VFA/TA (FOS/TAC) ratio."""
        return self._FOSTAC

    @property
    def AcvsPro(self) -> List[float]:
        """History of the acetate-to-propionate ratio."""
        return self._AcvsPro

    @property
    def VFA(self) -> List[float]:
        """History of total volatile fatty acid concentration [kg HAc-eq/m^3]."""
        return self._VFA

    @property
    def TAC(self) -> List[float]:
        """History of total alkalinity (TAC)."""
        return self._TAC

    def get_state_size(self) -> int:
        """Return the number of state variables: 41."""
        return STATE_SIZE

    # ------------------------------------------------------------------
    # Calibration parameter API
    # ------------------------------------------------------------------

    def set_calibration_parameters(self, parameters: dict) -> None:
        """Set calibration overrides for kinetic / gas parameters."""
        self._calibration_params.update(parameters)

    def clear_calibration_parameters(self) -> None:
        """Clear all calibration parameters."""
        self._calibration_params = {}

    def get_calibration_parameters(self) -> dict:
        """Return a copy of the current calibration parameters."""
        return self._calibration_params.copy()

    # ------------------------------------------------------------------
    # Gas calculation
    # ------------------------------------------------------------------

    def calc_gas(
        self,
        pi_Sh2: float,
        pi_Sch4: float,
        pi_Sco2: float,
        pTOTAL: float,
    ) -> Tuple[float, float, float, float, float]:
        """
        Calculate biogas production rates, normalised to standard reference
        conditions (theta = 20 °C, P = 1.01325 bar; ADM1da convention).

        Returns
        -------
        q_gas, q_ch4, q_co2, q_h2o, p_gas
        """
        k_p = self._k_p
        if self._calibration_params.get("k_p") is not None:
            k_p = float(self._calibration_params["k_p"])

        p_total_wet = pTOTAL + self._p_gas_h2o
        q_gas = max(
            k_p * (p_total_wet - self._p_ext) / (self._RT / 1000.0 * self._NQ) * self.V_liq,
            0.0,
        )

        p_gas = pi_Sh2 + pi_Sch4 + pi_Sco2
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

    # ------------------------------------------------------------------
    # Influent setup
    # ------------------------------------------------------------------

    def set_influent_dataframe(self, df: pd.DataFrame) -> None:
        """
        Store an external ADM1 influent DataFrame.

        The DataFrame must have columns matching ``INFLUENT_COLUMNS`` (37
        liquid-state columns + Q).  Once set, ``create_influent()`` reads
        from this DataFrame instead of deriving values from the feedstock.
        """
        missing = [c for c in INFLUENT_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Influent DataFrame missing columns: {missing}")
        self._influent_df = df.reset_index(drop=True)

    def set_influent_density(self, rho_in: float, rho_sludge: float = 1000.0) -> None:
        """Retained for API compatibility; stored but no longer affects Q_out."""
        self._rho_in = float(rho_in)
        self._rho_sludge = float(rho_sludge)

    def create_influent(
        self,
        Q: List[float],
        i: int,
        rho: Optional[List[float]] = None,
    ) -> None:
        """
        Build the influent vector for time step *i*.

        If an external influent DataFrame has been set via
        ``set_influent_dataframe()``, that DataFrame is used.  Otherwise
        the feedstock object is used to derive the influent.
        """
        if hasattr(self._feedstock, "actual_Q"):
            Q_actual = self._feedstock.actual_Q(Q)
        else:
            Q_actual = list(Q)
        self._Q = Q_actual
        if rho is not None and len(rho) == len(Q_actual):
            q_total = sum(Q_actual)
            if q_total > 0.0:
                self._rho_in = sum(q * r for q, r in zip(Q_actual, rho)) / q_total

        if self._influent_df is not None:
            max_i = len(self._influent_df) - 1
            row = self._influent_df.iloc[min(i, max_i)]
            self._state_input = row[INFLUENT_COLUMNS[:-1]].tolist()
        else:
            df = self._feedstock.get_influent_dataframe(Q)
            max_i = len(df) - 1
            row = df.iloc[min(i, max_i)]
            self._state_input = row[INFLUENT_COLUMNS[:-1]].tolist()

    # ------------------------------------------------------------------
    # ODE system
    # ------------------------------------------------------------------

    def ADM_ODE(self, t: float, state: List[float]) -> Tuple[float, ...]:
        """
        Compute dy/dt for the 41-element ADM1 state vector.

        The system is autonomous; *t* is present only to satisfy the scipy
        solver interface.
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
        k = self._kinetic
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
        S_IN = S_nh4 + S_nh3
        I_IN = S_IN / (ip["K_S_IN"] + S_IN + 1.0e-20)

        I_pH_aa = self._pH_inhib(S_H, ip["K_pH_aa"], n=1)
        I_pH_fa = self._pH_inhib(S_H, ip["K_pH_aa"], n=2)
        I_pH_c4 = self._pH_inhib(S_H, ip["K_pH_aa"], n=2)
        I_pH_pro = self._pH_inhib(S_H, ip["K_pH_aa"], n=2)
        I_pH_ac = self._pH_inhib(S_H, ip["K_pH_ac"], n=3)
        I_pH_h2 = self._pH_inhib(S_H, ip["K_pH_h2"], n=3)

        I_h2_fa = ip["K_I_h2_fa"] / (ip["K_I_h2_fa"] + S_h2)
        I_h2_c4 = ip["K_I_h2_c4"] / (ip["K_I_h2_c4"] + S_h2)
        I_h2_pro = ip["K_I_h2_pro"] / (ip["K_I_h2_pro"] + S_h2)

        K_nh3_ac_sq = ip["K_I_nh3"] ** 2
        K_nh3_pro_sq = ip["K_I_nh3_pro"] ** 2
        S_nh3_sq = S_nh3 * S_nh3
        I_nh3 = K_nh3_ac_sq / (K_nh3_ac_sq + S_nh3_sq)
        I_nh3_pro = K_nh3_pro_sq / (K_nh3_pro_sq + S_nh3_sq)

        K_co2_h2_sq = ip["K_S_co2_h2"] * ip["K_S_co2_h2"]
        S_co2_sq = S_co2 * S_co2
        I_co2_h2 = S_co2_sq / (K_co2_h2_sq + S_co2_sq + 1.0e-30)

        Ka_pro = ip["K_a_pro"]
        S_HPr = (S_pro / 112.0) * S_H / (S_H + Ka_pro + 1.0e-20)
        I_HPr = ip["K_IH_pro"] / (ip["K_IH_pro"] + S_HPr + 1.0e-20)

        Ka_ac = ip["K_a_ac"]
        S_HAc = (S_ac / 64.0) * S_H / (S_H + Ka_ac + 1.0e-20)
        I_HAc = ip["K_IH_ac"] / (ip["K_IH_ac"] + S_HAc + 1.0e-20)

        I_su = I_pH_aa * I_IN
        I_aa = I_pH_aa * I_IN
        I_fa = I_pH_fa * I_IN * I_h2_fa
        I_c4 = I_pH_c4 * I_IN * I_h2_c4
        I_pro = I_pH_pro * I_IN * I_h2_pro * I_HPr * I_nh3_pro
        I_ac = I_pH_ac * I_IN * I_nh3 * I_HAc
        I_h2 = I_pH_h2 * I_IN * I_co2_h2

        # ---- Process rates -----------------------------------------------
        Rho_dis_PS_ch = k["k_dis_PS"] * X_PS_ch
        Rho_dis_PS_pr = k["k_dis_PS"] * X_PS_pr
        Rho_dis_PS_li = k["k_dis_PS"] * X_PS_li
        Rho_dis_PF_ch = k["k_dis_PF"] * X_PF_ch
        Rho_dis_PF_pr = k["k_dis_PF"] * X_PF_pr
        Rho_dis_PF_li = k["k_dis_PF"] * X_PF_li

        Rho_hyd_ch = k["k_hyd_ch"] * X_S_ch
        Rho_hyd_pr = k["k_hyd_pr"] * X_S_pr
        Rho_hyd_li = k["k_hyd_li"] * X_S_li

        Rho_su = k["k_m_su"] * S_su / (k["K_S_su"] + S_su + 1.0e-20) * X_su * I_su
        Rho_aa = k["k_m_aa"] * S_aa / (k["K_S_aa"] + S_aa + 1.0e-20) * X_aa * I_aa
        Rho_fa = k["k_m_fa"] * S_fa / (k["K_S_fa"] + S_fa + 1.0e-20) * X_fa * I_fa

        S_vbu = S_va + S_bu + 1.0e-20
        Rho_c4_va = k["k_m_c4"] * S_va / (k["K_S_c4"] + S_va + 1.0e-20) * X_c4 * (S_va / S_vbu) * I_c4
        Rho_c4_bu = k["k_m_c4"] * S_bu / (k["K_S_c4"] + S_bu + 1.0e-20) * X_c4 * (S_bu / S_vbu) * I_c4

        Rho_pro = k["k_m_pro"] * S_pro / (k["K_S_pro"] + S_pro + 1.0e-20) * X_pro * I_pro
        Rho_ac = k["k_m_ac"] * S_ac / (k["K_S_ac"] + S_ac + 1.0e-20) * X_ac * I_ac
        Rho_h2 = k["k_m_h2"] * S_h2 / (k["K_S_h2"] + S_h2 + 1.0e-20) * X_h2 * I_h2

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

        Rho_T_h2 = k_L_a * (S_h2 - 16.0 * p_gas_h2 / (self._RT * self._K_H_h2)) * (self.V_liq / self._V_gas)
        Rho_T_ch4 = k_L_a * (S_ch4 - 64.0 * p_gas_ch4 / (self._RT * self._K_H_ch4)) * (self.V_liq / self._V_gas)
        S_co2_free = max(S_co2 - S_hco3, 0.0)
        Rho_T_co2 = k_L_a * (S_co2_free - p_gas_co2 / (self._RT * self._K_H_co2)) * (self.V_liq / self._V_gas)

        k_p = self._k_p
        if self._calibration_params.get("k_p") is not None:
            k_p = float(self._calibration_params["k_p"])
        Rho_T_11 = max(k_p * (pTOTAL + self._p_gas_h2o - self._p_ext) * (self.V_liq / self._V_gas), 0.0)

        # ---- Carbon stoichiometry coefficients for S_co2 balance --------
        C = st
        s_hyd_ch = -C["C_ch"] + (1.0 - fSI) * C["C_su"] + fSI * C["C_I_s"]
        s_hyd_pr = -C["C_pr"] + (1.0 - fSI) * C["C_aa"] + fSI * C["C_I_s"]
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
        D_in = q_ad / self.V_liq

        # Sludge volume loss (hydrolysis-rate volume balance, after Schlattmann 2011 §5.1)
        _q_S_loss = self.V_liq * (
            Rho_hyd_ch * (0.9375 / 1550.0) + Rho_hyd_pr * (0.6125 / 1370.0) + Rho_hyd_li * (0.3474 / 920.0)
        )
        _Q_out = max(q_ad - _q_S_loss, 0.0)
        D_out = _Q_out / self.V_liq

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

        N_bac = st["N_bac"]
        N_aa = st["N_aa"]
        N_I = st["N_I"]
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
        diff_X_PS_ch = D_in * s_in[12] - D_out * X_PS_ch - Rho_dis_PS_ch
        diff_X_PS_pr = D_in * s_in[13] - D_out * X_PS_pr - Rho_dis_PS_pr
        diff_X_PS_li = D_in * s_in[14] - D_out * X_PS_li - Rho_dis_PS_li
        diff_X_PF_ch = D_in * s_in[15] - D_out * X_PF_ch - Rho_dis_PF_ch
        diff_X_PF_pr = D_in * s_in[16] - D_out * X_PF_pr - Rho_dis_PF_pr
        diff_X_PF_li = D_in * s_in[17] - D_out * X_PF_li - Rho_dis_PF_li
        diff_X_S_ch = (
            D_in * s_in[18]
            - D_out * X_S_ch
            + (1.0 - fXI_PS) * Rho_dis_PS_ch
            + (1.0 - fXI_PF) * Rho_dis_PF_ch
            + f_ch_bac * sum_decay
            - Rho_hyd_ch
        )
        diff_X_S_pr = (
            D_in * s_in[19]
            - D_out * X_S_pr
            + (1.0 - fXI_PS) * Rho_dis_PS_pr
            + (1.0 - fXI_PF) * Rho_dis_PF_pr
            + f_pr_bac * sum_decay
            - Rho_hyd_pr
        )
        diff_X_S_li = (
            D_in * s_in[20]
            - D_out * X_S_li
            + (1.0 - fXI_PS) * Rho_dis_PS_li
            + (1.0 - fXI_PF) * Rho_dis_PF_li
            + f_li_bac * sum_decay
            - Rho_hyd_li
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
    # Result-tracking helper
    # ------------------------------------------------------------------

    def print_params_at_current_state(self, state: List[float]) -> None:
        """Compute and store process indicators (pH, gas) from the current state."""
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

    def resume_from_broken_simulation(self, Q_CH4: List[float]) -> None:
        """Re-populate the methane tracking list after a simulation restart."""
        for q in Q_CH4:
            self._Q_CH4.append(q)

    def _track_pH(self, pH: float) -> None:
        """Append pH to the history; pad on the very first call to keep length >= 2."""
        self._pH_l.append(pH)
        if len(self._pH_l) < 2:
            self._pH_l.append(self._pH_l[-1])

    def _track_gas(self, q_gas, q_ch4, q_co2, q_h2o, p_gas) -> None:
        """Append gas flows and headspace pressure to their history lists; pad on the first call."""
        self._Q_GAS.append(q_gas)
        self._Q_CH4.append(q_ch4)
        self._Q_CO2.append(q_co2)
        self._Q_H2O.append(q_h2o)
        self._P_GAS.append(p_gas)
        if len(self._Q_GAS) < 2:
            for _ in range(2):
                self._Q_GAS.append(q_gas)
                self._Q_CH4.append(q_ch4)
                self._Q_CO2.append(q_co2)
                self._Q_H2O.append(q_h2o)
                self._P_GAS.append(p_gas)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pH_inhib(S_H: float, K_pH: float, n: int = 1) -> float:
        """Hill-type pH inhibition factor: I = K_pH^n / (K_pH^n + S_H^n)."""
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
        """Solve the charge balance for [H+] using Newton–Raphson iteration."""
        vfa_anions = S_ac_ion / 64.0 + S_pro_ion / 112.0 + S_bu_ion / 160.0 + S_va_ion / 208.0
        fixed = S_cation - S_anion + (S_nh4 - S_nh3) - S_hco3 - vfa_anions

        S_H = 1.0e-7
        for _ in range(max_iter):
            f = fixed + S_H - K_w / (S_H + 1.0e-30)
            df = 1.0 + K_w / (S_H + 1.0e-30) ** 2
            delta = -f / df
            S_H = max(1.0e-14, S_H + delta)
            if abs(delta) < 1.0e-15:
                break
        return S_H
