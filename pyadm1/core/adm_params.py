"""
Parameter definitions for the ADM1da model.

Reference: Schlattmann (2011); SIMBA# biogas (ifak), model file ``adm1da``.

The model implemented in :mod:`pyadm1.core.adm1` is **ADM1da** (Schlattmann
2011), an agricultural-biogas extension of ADM1.  Compared to the classical
Batstone et al. (2002) formulation this variant adds:

  - A two-pool sub-fraction approach: XPS (slow) and XPF (fast) disintegration
    pools, each split into CH/PR/LI sub-types that produce XS (hydrolysable) +
    XI.
  - Temperature-dependent kinetics (Arrhenius θ-corrections per organism
    group).
  - Modified inhibition: squared pH inhibition for all acidogenic and
    acetogenic groups, cubic for X_ac and X_h2; inhibition by undissociated
    propionic and acetic acid (KIHPRO, KIHAC); squared free-ammonia inhibition
    of X_pro and X_ac with temperature-dependent constants.
  - Decay products recycled to the X_S fractions; the inert share keeps the
    composition of the biomass. Decay rate 0.02 d⁻¹, 0.04 d⁻¹ for X_ac.
  - Ammonia leaving with the biogas and a sludge volume that shrinks with the
    net volume of the converted organic matter.
"""

from __future__ import annotations

import numpy as np

# Mass-to-COD ratios [kg / kg COD] and densities [kg / m³] of the organic
# classes, as in the SIMBA# ADM1da model file.
_M_CH, _RHO_CH = 180.0 / 192.0, 1550.0
_M_PR, _RHO_PR = 0.7736, 1370.0
_M_LI, _RHO_LI = 806.0 / 2320.0, 920.0
_M_VA, _RHO_VA = 102.0 / 208.0, 940.0
_M_BU, _RHO_BU = 88.0 / 160.0, 960.0
_M_PRO, _RHO_PRO = 74.0 / 112.0, 990.0
_M_AC, _RHO_AC = 60.0 / 64.0, 1050.0

# Carbon content [kmol C / kg COD] of the three classes.
_C_CH, _C_PR, _C_LI = 0.0313, 0.0306, 0.022

# Biomass: 20 % carbohydrate, 70 % protein, 10 % lipid.
_BM_CH, _BM_PR, _BM_LI = 0.20, 0.70, 0.10
_M_BAC = _BM_CH * _M_CH + _BM_PR * _M_PR + _BM_LI * _M_LI
_RHO_BAC = _BM_CH * _RHO_CH + _BM_PR * _RHO_PR + _BM_LI * _RHO_LI

# Shares of the three classes in the decay products of the biomass (sum 1).
_F_CH_XB = _BM_CH * _M_CH / _M_BAC
_F_PR_XB = _BM_PR * _M_PR / _M_BAC
_F_LI_XB = _BM_LI * _M_LI / _M_BAC

# Nitrogen: proteins hold 13.84 % nitrogen by mass.
_N_AA = _M_PR * 0.1384 / 14.0
_N_BAC = _BM_PR * _N_AA
_C_BAC = (_BM_CH * _M_CH * _C_CH + _BM_PR * _M_PR * _C_PR + _BM_LI * _M_LI * _C_LI) / _M_BAC

# Share of the decayed biomass that ends up inert.
_F_P_BAC = 0.20

# Specific volumes [m³ / kg COD].
_V_CH, _V_PR, _V_LI = _M_CH / _RHO_CH, _M_PR / _RHO_PR, _M_LI / _RHO_LI


class ADMParams:
    """Static parameter class for the ADM1da model."""

    # *** PUBLIC STATIC GET methods ***

    @staticmethod
    def get_stoichiometric_params() -> dict:
        """Return composition, decay-product and volume constants.

        The inert share of the decay products keeps the composition of the
        biomass, so ``N_I``, ``C_I_x`` and ``V_I_x`` follow from the protein,
        carbohydrate and lipid shares of the biomass.

        Returns:
            dict: Carbon content [kmol C / kg COD], nitrogen content
                [kmol N / kg COD], product fractions [-] and specific volumes
                [m³ / kg COD] (ADM1da defaults; SIMBA# model file ``adm1da``).
        """
        return {
            # --- Carbon content [kmol C / kg COD] ---
            "C_su": _C_CH,
            "C_aa": _C_PR,
            "C_fa": 0.0217,
            "C_va": 0.024,
            "C_bu": 0.025,
            "C_pro": 0.0268,
            "C_ac": 0.0313,
            "C_ch4": 0.0156,
            "C_bac": _C_BAC,
            "C_ch": _C_CH,
            "C_pr": _C_PR,
            "C_li": _C_LI,
            "C_I_s": 0.03,
            "C_I_x": _F_CH_XB * _C_CH + _F_PR_XB * _C_PR + _F_LI_XB * _C_LI,
            # --- Nitrogen content [kmol N / kg COD] ---
            "N_bac": _N_BAC,
            "N_aa": _N_AA,
            "N_I": _F_PR_XB * _N_AA,
            # --- Lipid hydrolysis fraction ---
            "f_fa_li": 0.95,
            # --- Biomass decay product fractions ---
            "f_ch_bac": _F_CH_XB * (1.0 - _F_P_BAC),
            "f_pr_bac": _F_PR_XB * (1.0 - _F_P_BAC),
            "f_li_bac": _F_LI_XB * (1.0 - _F_P_BAC),
            "f_p_bac": _F_P_BAC,
            # --- Disintegration inert fractions ---
            "fXI_PS": 0.0,
            "fXI_PF": 0.0,
            # --- Hydrolysis soluble inert fraction ---
            "fSI_hyd": 0.0,
            # --- Specific volumes [m³ / kg COD] ---
            "V_ch": _V_CH,
            "V_pr": _V_PR,
            "V_li": _V_LI,
            "V_va": _M_VA / _RHO_VA,
            "V_bu": _M_BU / _RHO_BU,
            "V_pro": _M_PRO / _RHO_PRO,
            "V_ac": _M_AC / _RHO_AC,
            "V_bac": _M_BAC / _RHO_BAC,
            "V_I_x": _F_CH_XB * _V_CH + _F_PR_XB * _V_PR + _F_LI_XB * _V_LI,
        }

    @staticmethod
    def get_kinetic_params() -> dict:
        """
        Return kinetic parameters at the reference temperature (35 °C).
        """
        return {
            # --- Disintegration rate constants [d⁻¹] ---
            "k_dis_PS": 0.04,
            "k_dis_PF": 0.4,
            # --- Hydrolysis rate constants [d⁻¹] ---
            "k_hyd_ch": 4.0,
            "k_hyd_pr": 4.0,
            "k_hyd_li": 4.0,
            # --- Maximum uptake rates [d⁻¹] ---
            "k_m_su": 30.0,
            "k_m_aa": 50.0,
            "k_m_fa": 6.0,
            "k_m_c4": 20.0,
            "k_m_pro": 13.0,
            "k_m_ac": 8.0,
            "k_m_h2": 35.0,
            # --- Half-saturation constants [kg COD m⁻³] ---
            "K_S_su": 0.5,
            "K_S_aa": 0.3,
            "K_S_fa": 0.4,
            "K_S_c4": 0.2,
            "K_S_pro": 0.1,
            "K_S_ac": 0.15,
            "K_S_h2": 7.0e-6,
            # --- Per-organism decay rates [d⁻¹] ---
            "k_dec_su": 0.02,
            "k_dec_aa": 0.02,
            "k_dec_fa": 0.02,
            "k_dec_c4": 0.02,
            "k_dec_pro": 0.02,
            "k_dec_ac": 0.04,
            "k_dec_h2": 0.02,
            # --- Yield coefficients ---
            "Y_su": 0.10,
            "Y_aa": 0.08,
            "Y_fa": 0.06,
            "Y_c4": 0.06,
            "Y_pro": 0.04,
            "Y_ac": 0.05,
            "Y_h2": 0.06,
        }

    @staticmethod
    def get_temperature_factors() -> dict:
        """
        Return Arrhenius θ correction factors per organism/process group.
        """
        return {
            "theta_dis": np.exp(0.024),
            "theta_hyd": np.exp(0.024),
            "theta_su": np.exp(0.069),
            "theta_aa": np.exp(0.069),
            "theta_fa": np.exp(0.055),
            "theta_c4": np.exp(0.055),
            "theta_pro": np.exp(0.055),
            "theta_ac": np.exp(0.055),
            "theta_h2": np.exp(0.069),
            "theta_dec_su_aa_h2": np.exp(0.069),
            "theta_dec_fa_c4_pro_ac": np.exp(0.055),
        }

    @staticmethod
    def get_product_fractions() -> dict:
        """Return fermentation product fractions."""
        return {
            "f_h2_su": 0.19,
            "f_bu_su": 0.13,
            "f_pro_su": 0.27,
            "f_ac_su": 0.41,
            "f_h2_aa": 0.06,
            "f_va_aa": 0.23,
            "f_bu_aa": 0.26,
            "f_pro_aa": 0.05,
            "f_ac_aa": 0.40,
        }

    @staticmethod
    def get_inhibition_params(R: float, T_base: float, T_ad: float) -> dict:
        """
        Return inhibition-related parameters and acid-base constants.

        Parameters
        ----------
        R : float
            Gas constant [bar m³ kmol⁻¹ K⁻¹].
        T_base : float
            Reference temperature [K] (25 °C = 298.15 K).
        T_ad : float
            Operating temperature [K].
        """
        pH_LL_aa, pH_UL_aa = 4.0, 5.5
        pH_LL_ac, pH_UL_ac = 6.0, 7.0
        pH_LL_h2, pH_UL_h2 = 5.0, 6.0

        K_pH_aa = 10.0 ** (-(pH_LL_aa + pH_UL_aa) / 2.0)
        K_pH_ac = 10.0 ** (-(pH_LL_ac + pH_UL_ac) / 2.0)
        K_pH_h2 = 10.0 ** (-(pH_LL_h2 + pH_UL_h2) / 2.0)

        K_w = 10.0**-14.0 * np.exp((55900.0 / (100.0 * R)) * (1.0 / T_base - 1.0 / T_ad))
        K_a_va = 10.0**-4.86
        K_a_bu = 10.0**-4.82
        K_a_pro = 10.0**-4.88
        K_a_ac = 10.0**-4.76
        K_a_co2 = 10.0**-6.35 * np.exp((7646.0 / (100.0 * R)) * (1.0 / T_base - 1.0 / T_ad))
        K_a_IN = 10.0**-9.25 * np.exp((51965.0 / (100.0 * R)) * (1.0 / T_base - 1.0 / T_ad))

        dT_C = T_ad - 308.15

        return {
            "K_pH_aa": K_pH_aa,
            "K_pH_ac": K_pH_ac,
            "K_pH_h2": K_pH_h2,
            "K_w": K_w,
            "K_a_va": K_a_va,
            "K_a_bu": K_a_bu,
            "K_a_pro": K_a_pro,
            "K_a_ac": K_a_ac,
            "K_a_co2": K_a_co2,
            "K_a_IN": K_a_IN,
            "k_A_B": 1.0e8,
            "K_S_IN": 1.0e-4,
            "K_I_nh3": 0.0018 * np.exp(0.086 * dT_C),
            "K_I_h2_fa": 5.0e-6 * np.exp(0.080 * dT_C),
            "K_I_h2_c4": 1.0e-5 * np.exp(0.080 * dT_C),
            "K_I_h2_pro": 3.5e-6 * np.exp(0.080 * dT_C),
            "K_IH_pro": 8.0e-4,
            "K_IH_ac": 2.417e-3,
            "K_I_nh3_pro": 0.0019 * np.exp(0.061 * dT_C),
            "K_S_co2_h2": 5.0e-5,
            "K_I_ac_xfa": 4.0 * np.exp(0.080 * dT_C),
            "K_I_ac_xc4": 4.0 * np.exp(0.080 * dT_C),
        }

    @staticmethod
    def apply_temperature_corrections(kinetic: dict, theta: dict, T_ad: float) -> dict:
        """
        Return a copy of *kinetic* with rates corrected to *T_ad*.

        Uses: k(T) = k(35 °C) · θ^(T[°C] − 35)
        """
        dT = T_ad - 308.15
        corrected = dict(kinetic)

        corrected["k_dis_PS"] = kinetic["k_dis_PS"] * theta["theta_dis"] ** dT
        corrected["k_dis_PF"] = kinetic["k_dis_PF"] * theta["theta_dis"] ** dT
        corrected["k_hyd_ch"] = kinetic["k_hyd_ch"] * theta["theta_hyd"] ** dT
        corrected["k_hyd_pr"] = kinetic["k_hyd_pr"] * theta["theta_hyd"] ** dT
        corrected["k_hyd_li"] = kinetic["k_hyd_li"] * theta["theta_hyd"] ** dT
        corrected["k_m_su"] = kinetic["k_m_su"] * theta["theta_su"] ** dT
        corrected["k_m_aa"] = kinetic["k_m_aa"] * theta["theta_aa"] ** dT
        corrected["k_m_fa"] = kinetic["k_m_fa"] * theta["theta_fa"] ** dT
        corrected["k_m_c4"] = kinetic["k_m_c4"] * theta["theta_c4"] ** dT
        corrected["k_m_pro"] = kinetic["k_m_pro"] * theta["theta_pro"] ** dT
        corrected["k_m_ac"] = kinetic["k_m_ac"] * theta["theta_ac"] ** dT
        corrected["k_m_h2"] = kinetic["k_m_h2"] * theta["theta_h2"] ** dT
        for dec_key in ("k_dec_su", "k_dec_aa", "k_dec_h2"):
            corrected[dec_key] = kinetic[dec_key] * theta["theta_dec_su_aa_h2"] ** dT
        for dec_key in ("k_dec_fa", "k_dec_c4", "k_dec_pro", "k_dec_ac"):
            corrected[dec_key] = kinetic[dec_key] * theta["theta_dec_fa_c4_pro_ac"] ** dT

        return corrected

    @staticmethod
    def getADMgasparams(R: float, T_base: float, T_ad: float) -> tuple[float, float, float, float, float, float]:
        """
        Get gas phase parameters including Henry constants.

        Parameters
        ----------
        R : float
            Gas constant [bar·m³·kmol⁻¹·K⁻¹]
        T_base : float
            Reference temperature [K] (298.15)
        T_ad : float
            Digester temperature [K]

        Returns
        -------
        Tuple[float, float, float, float, float, float]
            p_gas_h2o, k_p, k_L_a, K_H_co2, K_H_ch4, K_H_h2
        """
        p_gas_h2o = 0.0313 * np.exp(5290 * (1 / T_base - 1 / T_ad))
        k_p = 1.0e4
        k_L_a = 200.0

        # Henry's law solubility coefficients [kmol m⁻³ bar⁻¹] at 25 °C with
        # van't Hoff T-correction.
        R_J = 100.0 * R
        dH_co2, dH_ch4, dH_h2 = 19410.0, 14240.0, 4180.0
        H_co2 = 0.035 * np.exp(-dH_co2 / R_J * (1.0 / T_base - 1.0 / T_ad))
        H_ch4 = 0.0014 * np.exp(-dH_ch4 / R_J * (1.0 / T_base - 1.0 / T_ad))
        H_h2 = 7.8e-4 * np.exp(-dH_h2 / R_J * (1.0 / T_base - 1.0 / T_ad))
        K_H_co2 = 1 / (H_co2 * R * T_ad)
        K_H_ch4 = 1 / (H_ch4 * R * T_ad)
        K_H_h2 = 1 / (H_h2 * R * T_ad)

        return p_gas_h2o, k_p, k_L_a, K_H_co2, K_H_ch4, K_H_h2

    @staticmethod
    def get_nh3_henry_constant(R: float, T_base: float, T_ad: float) -> float:
        """Return the gas-to-liquid concentration ratio of free ammonia.

        Ammonia is far more soluble than the main biogas components, so its
        partial pressure follows the dissolved concentration without delay:
        the gas phase holds ``K_H_nh3 * S_nh3`` kmol N per m³.

        Args:
            R (float): Gas constant [bar m³ kmol⁻¹ K⁻¹].
            T_base (float): Reference temperature [K] (298.15).
            T_ad (float): Digester temperature [K].

        Returns:
            float: ``K_H_nh3 = 1 / (H_nh3 R T)`` [m³ liquid / m³ gas], with the
                solubility ``H_nh3`` = 60 kmol m⁻³ bar⁻¹ at 25 °C.
        """
        H_nh3 = 60.0 * np.exp(-36584.0 / (100.0 * R) * (1.0 / T_base - 1.0 / T_ad))
        return 1.0 / (H_nh3 * R * T_ad)
