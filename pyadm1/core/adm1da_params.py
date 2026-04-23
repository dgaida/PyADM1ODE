# pyadm1/core/adm1da_params.py
"""
Parameter definitions for the SIMBA# biogas ADM1da model.

ADM1da (Schlattmann 2011) extends ADM1 with:
  - Sub-fraction approach: XPS (slow) and XPF (fast) disintegration pools,
    each split into CH/PR/LI sub-types that produce XS (hydrolysable) + XI.
  - Temperature-dependent kinetics (Arrhenius θ-corrections per organism group).
  - Modified inhibition: squared pH inhibition for X_fa/X_c4/X_pro,
    cubic for X_ac; undissociated acid inhibition (KIHPRO, KIHAC);
    acetate competitive inhibition; S_IN = S_nh4 + S_nh3 for N limitation.
  - Doubled decay rate (k_dec = 0.04 d⁻¹) and decay products recycled
    to XPS fractions (f_ch_bac=0.16, f_pr_bac=0.56, f_li_bac=0.08).
  - NH3 inhibition constants (K_I_nh3, K_I_nh3_pro) use reference-temperature
    values without T-correction — see get_inhibition_params() for details.

Reference: SIMBA# biogas 4.2 Tutorial, ifak e.V. Magdeburg
"""

import numpy as np


class ADM1daParams:
    """Static parameter class for the ADM1da model variant."""

    # *** PUBLIC STATIC GET methods ***

    @staticmethod
    def get_stoichiometric_params() -> dict:
        """
        Return carbon/nitrogen content and disintegration/hydrolysis fractions.

        Returns
        -------
        dict
            Stoichiometric constants for ADM1da (SIMBA# biogas 4.2 defaults).
        """
        return {
            # --- Carbon content [kmol C / kg COD] ---
            "C_su": 0.0313,  # monosaccharides
            "C_aa": 0.03,  # amino acids
            "C_fa": 0.0217,  # long-chain fatty acids
            "C_va": 0.024,  # valerate
            "C_bu": 0.025,  # butyrate
            "C_pro": 0.0268,  # propionate
            "C_ac": 0.0313,  # acetate
            "C_ch4": 0.0156,  # methane
            "C_bac": 0.030381,  # active biomass (SIMBA# value)
            "C_ch": 0.0313,  # carbohydrate sub-fractions (XPS_ch, XPF_ch, XS_ch)
            "C_pr": 0.0306,  # protein sub-fractions (SIMBA# value)
            "C_li": 0.022,  # lipid sub-fractions
            "C_I_s": 0.03,  # soluble inerts
            "C_I_x": 0.03,  # particulate inerts
            # --- Nitrogen content [kmol N / kg COD] ---
            "N_bac": 0.005353,  # active biomass (SIMBA# value)
            "N_aa": 0.0076,  # amino acids / proteins (ADM1da Table 3 value)
            "N_I": 0.06 / 14,  # inerts
            # --- Lipid hydrolysis fraction ---
            "f_fa_li": 0.95,  # LCFA fraction from lipid hydrolysis
            # --- Biomass decay product fractions (f_ch + f_pr + f_li + f_p = 1) ---
            # SIMBA#: fBM_CH=0.20, fBM_PR=0.70, fBM_LI=0.10, fP=0.20
            # Net fractions to XPS pools = (1 - fP) * fBM_*
            "f_ch_bac": 0.16,  # = (1-0.20)*0.20 → XPS_ch
            "f_pr_bac": 0.56,  # = (1-0.20)*0.70 → XPS_pr
            "f_li_bac": 0.08,  # = (1-0.20)*0.10 → XPS_li
            "f_p_bac": 0.20,  # endogenous residue fraction → X_I
            # --- Disintegration inert fractions (SIMBA#: zero) ---
            "fXI_PS": 0.0,  # particulate inert fraction from slow disintegration (XPS)
            "fXI_PF": 0.0,  # particulate inert fraction from fast disintegration (XPF)
            # --- Hydrolysis soluble inert fraction (SIMBA#: zero) ---
            "fSI_hyd": 0.0,  # soluble inert produced per unit hydrolysis
        }

    @staticmethod
    def get_kinetic_params() -> dict:
        """
        Return kinetic parameters at the reference temperature (35 °C).

        These values are corrected to the operating temperature via
        ``apply_temperature_corrections()``.

        Returns
        -------
        dict
            Rate constants and half-saturation coefficients.
        """
        return {
            # --- Disintegration rate constants [d⁻¹] (SIMBA# values) ---
            "k_dis_PS": 0.04,  # slow disintegration pool (XPS)
            "k_dis_PF": 0.4,  # fast disintegration pool (XPF)
            # --- Hydrolysis rate constants [d⁻¹] (SIMBA# value: 4 d⁻¹) ---
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
            # --- Half-saturation constants [kg COD m⁻³] unless noted ---
            "K_S_su": 0.5,
            "K_S_aa": 0.3,
            "K_S_fa": 0.4,
            "K_S_c4": 0.2,
            "K_S_pro": 0.1,
            "K_S_ac": 0.15,
            "K_S_h2": 7.0e-6,
            # --- Per-organism decay rates [d⁻¹] (SIMBA# Table 6) ---
            # X_su/aa/fa/c4/pro/h2 = 0.02; only X_ac = 0.04 per SIMBA# parameter table
            "k_dec_su": 0.02,
            "k_dec_aa": 0.02,
            "k_dec_fa": 0.02,
            "k_dec_c4": 0.02,
            "k_dec_pro": 0.02,
            "k_dec_ac": 0.04,  # X_ac only — confirmed 0.04 in SIMBA#
            "k_dec_h2": 0.02,
            # --- Yield coefficients [kg COD_X / kg COD_S] ---
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

        Temperature correction: k(T) = k(35 °C) · θ^(T[°C] − 35)
        where θ = exp(θ_exp) using the SIMBA# Table 6 exponent coefficients:
          - Disintegration & hydrolysis: θ_exp = 0.024
          - X_su, X_aa, X_h2 (µ and decay): θ_exp = 0.069
          - X_fa, X_c4, X_pro, X_ac (µ and decay): θ_exp = 0.055

        Returns
        -------
        dict
            θ values for each process group.
        """
        return {
            "theta_dis": np.exp(0.024),  # θ_exp=0.024 for disintegration
            "theta_hyd": np.exp(0.024),  # θ_exp=0.024 for hydrolysis
            "theta_su": np.exp(0.069),  # θ_exp=0.069: X_su/X_aa/X_h2 group
            "theta_aa": np.exp(0.069),
            "theta_fa": np.exp(0.055),  # θ_exp=0.055: X_fa/X_c4/X_pro/X_ac group
            "theta_c4": np.exp(0.055),
            "theta_pro": np.exp(0.055),
            "theta_ac": np.exp(0.055),
            "theta_h2": np.exp(0.069),
            "theta_dec_su_aa_h2": np.exp(0.069),  # decay θ for X_su, X_aa, X_h2
            "theta_dec_fa_c4_pro_ac": np.exp(0.055),  # decay θ for X_fa, X_c4, X_pro, X_ac
        }

    @staticmethod
    def get_product_fractions() -> dict:
        """
        Return fermentation product fractions (same values as ADM1).

        Returns
        -------
        dict
            Fraction coefficients for sugar and amino-acid fermentation.
        """
        return {
            # From sugar fermentation (X_su)
            "f_h2_su": 0.19,
            "f_bu_su": 0.13,
            "f_pro_su": 0.27,
            "f_ac_su": 0.41,
            # From amino-acid fermentation (X_aa)
            "f_h2_aa": 0.06,
            "f_va_aa": 0.23,
            "f_bu_aa": 0.26,
            "f_pro_aa": 0.05,
            "f_ac_aa": 0.40,
        }

    @staticmethod
    def get_inhibition_params(R: float, T_base: float, T_ad: float) -> dict:
        """
        Return all inhibition-related parameters including acid-base constants.

        Differences from ADM1:
          - N limitation uses S_IN = S_nh4 + S_nh3 (not S_nh4 alone).
          - pH inhibition exponents: n=2 for X_fa/X_c4/X_pro, n=3 for X_ac.
          - Additional: undissociated propionic acid (K_IH_pro) and acetic
            acid (K_IH_ac) inhibition; acetate competitive inhibition (K_I_ac_*).

        Parameters
        ----------
        R : float
            Gas constant [bar m³ kmol⁻¹ K⁻¹].
        T_base : float
            Reference temperature [K] (25 °C = 298.15 K).
        T_ad : float
            Operating temperature [K].

        Returns
        -------
        dict
            Inhibition and acid-base equilibrium constants.
        """
        # --- pH inhibition half-concentrations ---
        pH_LL_aa, pH_UL_aa = 4.0, 5.5
        pH_LL_ac, pH_UL_ac = 6.0, 7.0
        pH_LL_h2, pH_UL_h2 = 5.0, 6.0

        K_pH_aa = 10.0 ** (-(pH_LL_aa + pH_UL_aa) / 2.0)
        K_pH_ac = 10.0 ** (-(pH_LL_ac + pH_UL_ac) / 2.0)
        K_pH_h2 = 10.0 ** (-(pH_LL_h2 + pH_UL_h2) / 2.0)

        # --- Acid-base equilibrium constants ---
        K_w = 10.0**-14.0 * np.exp((55900.0 / (100.0 * R)) * (1.0 / T_base - 1.0 / T_ad))
        K_a_va = 10.0**-4.86
        K_a_bu = 10.0**-4.82
        K_a_pro = 10.0**-4.88
        K_a_ac = 10.0**-4.76
        K_a_co2 = 10.0**-6.35 * np.exp((7646.0 / (100.0 * R)) * (1.0 / T_base - 1.0 / T_ad))
        K_a_IN = 10.0**-9.25 * np.exp((51965.0 / (100.0 * R)) * (1.0 / T_base - 1.0 / T_ad))

        # Temperature offset from reference 35 °C [K = °C offset]
        dT_C = T_ad - 308.15

        return {
            # pH half-concentrations [M = kmol m⁻³]
            "K_pH_aa": K_pH_aa,
            "K_pH_ac": K_pH_ac,
            "K_pH_h2": K_pH_h2,
            # Acid-base equilibrium constants
            "K_w": K_w,
            "K_a_va": K_a_va,
            "K_a_bu": K_a_bu,
            "K_a_pro": K_a_pro,
            "K_a_ac": K_a_ac,
            "K_a_co2": K_a_co2,
            "K_a_IN": K_a_IN,
            # Acid-base kinetic constant [kmol⁻¹ m³ d⁻¹] (same for all pairs)
            "k_A_B": 1.0e8,
            # N-limitation half-saturation constant [kmol N m⁻³]
            "K_S_IN": 1.0e-4,
            # NH3 inhibition constant for X_ac [kmol N m⁻³] — T-corrected per
            # SIMBA# biogas 4.2 tutorial §6: K_I_NH3_XAC ~ exp(0.086·(T-35)).
            # Reference value 0.0018 at 35 °C (Batstone 2002 / SIMBA# Table 6).
            "K_I_nh3": 0.0018 * np.exp(0.086 * dT_C),
            # H2 inhibition constants [kg COD m⁻³] — T-corrected (θ_exp=0.080, SIMBA# Table 6)
            "K_I_h2_fa": 5.0e-6 * np.exp(0.080 * dT_C),
            "K_I_h2_c4": 1.0e-5 * np.exp(0.080 * dT_C),
            "K_I_h2_pro": 3.5e-6 * np.exp(0.080 * dT_C),
            # --- ADM1da-specific inhibition constants ---
            # Undissociated acid inhibition [kmol m⁻³] (no T-correction in SIMBA# table)
            # KI_hpro = 0.0896 kg COD/m³ / 112 kg COD/kmol = 8.0e-4 kmol/m³
            # KI_hac  = 0.1547 kg COD/m³ /  64 kg COD/kmol = 2.417e-3 kmol/m³
            "K_IH_pro": 8.0e-4,  # propionic acid → inhibits X_c4 and X_pro
            "K_IH_ac": 2.417e-3,  # acetic acid    → inhibits X_ac
            # NH3 inhibition for propionate degraders [kmol N m⁻³] — T-corrected
            # per SIMBA# biogas 4.2 tutorial §6: K_I_NH3_XPRO ~ exp(0.060·(T-35)).
            # Reference value 0.0019 at 35 °C (SIMBA# Table 6).
            "K_I_nh3_pro": 0.0019 * np.exp(0.060 * dT_C),
            # CO2 half-saturation for H2 methanogens [kmol C m⁻³]
            "K_S_co2_h2": 5.0e-5,
            # Acetate competitive inhibition [kg COD m⁻³] — T-corrected (θ_exp=0.080, SIMBA# Table 6)
            # Note: K_IAC,XPRO from SIMBA# maps to K_IHPRO (undissociated propionate), not S_ac-based
            "K_I_ac_xfa": 4.0 * np.exp(0.080 * dT_C),  # for X_fa (LCFA degraders)
            "K_I_ac_xc4": 4.0 * np.exp(0.080 * dT_C),  # for X_c4 (valerate/butyrate degraders)
        }

    @staticmethod
    def apply_temperature_corrections(kinetic: dict, theta: dict, T_ad: float) -> dict:
        """
        Return a copy of *kinetic* with rates corrected to *T_ad*.

        Uses: k(T) = k(35 °C) · θ^(T[°C] − 35)

        Parameters
        ----------
        kinetic : dict
            Base kinetic parameters at 35 °C (from ``get_kinetic_params()``).
        theta : dict
            θ factors (from ``get_temperature_factors()``).
        T_ad : float
            Operating temperature [K].

        Returns
        -------
        dict
            Corrected kinetic parameters.
        """
        dT = T_ad - 308.15  # offset from reference 35 °C in Kelvin (= °C offset)
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
        # Per-organism decay rates (split by organism group per SIMBA# Table 6)
        for dec_key in ("k_dec_su", "k_dec_aa", "k_dec_h2"):
            corrected[dec_key] = kinetic[dec_key] * theta["theta_dec_su_aa_h2"] ** dT
        for dec_key in ("k_dec_fa", "k_dec_c4", "k_dec_pro", "k_dec_ac"):
            corrected[dec_key] = kinetic[dec_key] * theta["theta_dec_fa_c4_pro_ac"] ** dT

        return corrected
