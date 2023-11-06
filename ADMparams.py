# -*- coding: utf-8 -*-
"""
Created on Fri Nov 04 09:56:06 2023

@author: Daniel Gaida
"""
"""
contains functions returning constant ADM1 params

Most code is taken from the PyADM1 implementation of:

@article {Sadrimajd2021.03.03.433746,
	author = {Sadrimajd, Peyman and Mannion, Patrick and Howley, Enda and Lens, Piet N. L.},
	title = {PyADM1: a Python implementation of Anaerobic Digestion Model No. 1},
	elocation-id = {2021.03.03.433746},
	year = {2021},
	doi = {10.1101/2021.03.03.433746},
	URL = {https://www.biorxiv.org/content/early/2021/03/04/2021.03.03.433746},
	eprint = {https://www.biorxiv.org/content/early/2021/03/04/2021.03.03.433746.full.pdf},
	journal = {bioRxiv}
}
"""

import numpy as np

# the params fH2_FA, fH2_VA, fPRO_VA, fH2_BU, fH2_PRO
# are coded in PyADM1.py directly as numbers, so they are not defined here

class ADMparams:
    # *** CONSTRUCTORS ***

    # *** PUBLIC SET methods ***

    # *** PUBLIC GET methods ***

    # *** PUBLIC methods ***

    # *** PUBLIC STATIC/CLASS GET methods ***

    @staticmethod
    def getADMparams(R, T_base, T_ad):
        ##parameter definition from the Rosen et al (2006) BSM2 report bmadm1_report
        # Stoichiometric parameter
        # they are substrate dependent and therefore calculated from the current substrate mix, directly in PyADM1.py
        #f_sI_xc =  0.1      # OK
        #f_xI_xc =  0.2      # in C# split into fXI_XC and fXP_XC
        #f_ch_xc =  0.2      # OK
        #f_pr_xc =  0.2      # OK
        #f_li_xc =  0.3      # OK

        N_xc =  0.0376 / 14                 # OK
        N_I =  0.06 / 14 #kmole N.kg^-1COD  # OK
        N_aa =  0.007 #kmole N.kg^-1COD         OK
        C_xc =  0.02786 #kmole C.kg^-1COD       in C# 0.03
        C_sI =  0.03 #kmole C.kg^-1COD          OK
        C_ch =  0.0313 #kmole C.kg^-1COD        C_Xch OK
        C_pr =  0.03 #kmole C.kg^-1COD          C_Xpr OK
        C_li =  0.022 #kmole C.kg^-1COD         C_Xli OK
        C_xI =  0.03 #kmole C.kg^-1COD          OK
        C_su =  0.0313 #kmole C.kg^-1COD        OK
        C_aa =  0.03 #kmole C.kg^-1COD          OK
        f_fa_li =  0.95                         # fFA_Xli OK
        C_fa =  0.0217 #kmole C.kg^-1COD        # C_Sfa OK

        f_h2_su, f_bu_su, f_pro_su, f_ac_su = ADMparams._getADMfsuparams()

        N_bac =  0.08 / 14 #kmole N.kg^-1COD        # N_XB OK
        C_bu =  0.025 #kmole C.kg^-1COD         C_Sbu OK
        C_pro =  0.0268 #kmole C.kg^-1COD       C_Spro OK
        C_ac =  0.0313 #kmole C.kg^-1COD        C_Sac OK
        C_bac =  0.0313 #kmole C.kg^-1COD       C_XB OK

        C_va = 0.024  # kmole C.kg^-1COD        # C_Sva OK
        C_ch4 = 0.0156  # kmole C.kg^-1COD      # C_Sch4 OK

        (Y_su, Y_aa, Y_fa, Y_c4, Y_pro, Y_ac, Y_h2) = ADMparams._getADMYparams()

        f_h2_aa, f_va_aa, f_bu_aa, f_pro_aa, f_ac_aa = ADMparams._getADMfaaparams()

        # they are substrate dependent and therefore calculated from the current substrate mix, directly in PyADM1.py
        # Biochemical parameter values from the Rosen et al (2006) BSM2 report
        #k_dis =  0.5 #d^-1              OK
        #k_hyd_ch =  10 #d^-1            OK
        #k_hyd_pr =  10 #d^-1            OK
        #k_hyd_li =  10 #d^-1            OK

        (K_S_IN, k_m_su, K_S_su, k_m_aa, K_S_aa, k_m_fa,
         K_S_fa, K_I_h2_fa, k_m_c4, K_S_c4, K_I_h2_c4, k_m_pro, K_S_pro, K_I_h2_pro,
         k_m_ac, K_S_ac, K_I_nh3, k_m_h2, K_S_h2) = ADMparams._getADMk_mK_Sparams()

        pH_LL_aa, pH_UL_aa, pH_LL_ac, pH_UL_ac, pH_LL_h2, pH_UL_h2 = ADMparams._getADMpHULLLparams()

        # decay rates
        k_dec_X_su =  0.02 #d^-1            # OK
        k_dec_X_aa =  0.02 #d^-1            # OK
        k_dec_X_fa =  0.02 #d^-1            # OK
        k_dec_X_c4 =  0.02 #d^-1            # OK
        k_dec_X_pro =  0.02 #d^-1           # OK
        k_dec_X_ac =  0.02 #d^-1            # OK
        k_dec_X_h2 =  0.02 #d^-1            # OK
        ## M is kmole m^-3

        K_w, K_a_va, K_a_bu, K_a_pro, K_a_ac, K_a_co2, K_a_IN = ADMparams._getADMKparams(R, T_base, T_ad)

        # acid-base kinetic parameters
        # those values all are 1e8 kmole/d in C# implementation
        k_A_B_va =  1e8 # 10 ** 10 #M^-1 * d^-1
        k_A_B_bu =  1e8 #10 ** 10 #M^-1 * d^-1
        k_A_B_pro =  1e8 #10 ** 10 #M^-1 * d^-1
        k_A_B_ac =  1e8 #10 ** 10 #M^-1 * d^-1
        k_A_B_co2 =  1e8 #10 ** 10 #M^-1 * d^-1
        k_A_B_IN =  1e8 #10 ** 10 #M^-1 * d^-1

        p_gas_h2o, k_p, k_L_a, K_H_co2, K_H_ch4, K_H_h2 = ADMparams.getADMgasparams(R, T_base, T_ad)

        return (N_xc, N_I, N_aa,
                C_xc, C_sI, C_ch, C_pr, C_li, C_xI, C_su, C_aa, f_fa_li, C_fa,
                f_h2_su, f_bu_su, f_pro_su, f_ac_su, N_bac, C_bu, C_pro, C_ac, C_bac,
                Y_su, f_h2_aa, f_va_aa, f_bu_aa, f_pro_aa, f_ac_aa, C_va,
                Y_aa, Y_fa, Y_c4, Y_pro, C_ch4, Y_ac, Y_h2,
                K_S_IN, k_m_su, K_S_su, pH_UL_aa, pH_LL_aa, k_m_aa, K_S_aa, k_m_fa,
                K_S_fa, K_I_h2_fa, k_m_c4, K_S_c4, K_I_h2_c4, k_m_pro, K_S_pro, K_I_h2_pro,
                k_m_ac, K_S_ac, K_I_nh3, pH_UL_ac, pH_LL_ac, k_m_h2, K_S_h2,
                pH_UL_h2, pH_LL_h2, k_dec_X_su, k_dec_X_aa, k_dec_X_fa, k_dec_X_c4, k_dec_X_pro,
                k_dec_X_ac, k_dec_X_h2, K_w, K_a_va, K_a_bu, K_a_pro, K_a_ac, K_a_co2, K_a_IN,
                k_A_B_va, k_A_B_bu, k_A_B_pro, k_A_B_ac, k_A_B_co2, k_A_B_IN, p_gas_h2o, k_p, k_L_a,
                K_H_co2, K_H_ch4, K_H_h2)

    @staticmethod
    def getADMinhibitionparams():
        pH_LL_aa, pH_UL_aa, pH_LL_ac, pH_UL_ac, pH_LL_h2, pH_UL_h2 = ADMparams._getADMpHULLLparams()

        # related to pH inhibition taken from BSM2 report, they are global variables to avoid repeating them in DAE part
        K_pH_aa = (10 ** (-1 * (pH_LL_aa + pH_UL_aa) / 2.0))  # OK
        # we need a difference between N_aa and n_aa to avoid typos and nn_aa refers to the n_aa in BSM2 report
        # in C# just set to 2, the calculation here is also 2, so OK
        nn_aa = (3.0 / (
                pH_UL_aa - pH_LL_aa))
        K_pH_ac = (10 ** (-1 * (pH_LL_ac + pH_UL_ac) / 2.0))  # OK
        n_ac = (3.0 / (pH_UL_ac - pH_LL_ac))  # OK
        K_pH_h2 = (10 ** (-1 * (pH_LL_h2 + pH_UL_h2) / 2.0))  # OK
        n_h2 = (3.0 / (pH_UL_h2 - pH_LL_h2))  # OK

        return K_pH_aa, nn_aa, K_pH_ac, n_ac, K_pH_h2, n_h2

    @staticmethod
    def getADMgasparams(R, T_base, T_ad):
        p_gas_h2o = 0.0313 * np.exp(5290 * (1 / T_base - 1 / T_ad))  # bar #0.0557
        # in C# this value is 10000, but in m^3/(m^3*d)
        # k_p = 5 * 10 ** 4  # m^3.d^-1.bar^-1 #only for BSM2 AD conditions, recalibrate for other AD cases #gas outlet friction
        k_p = 10000
        k_L_a = 200.0  # d^-1 OK in C# implementation there is klAH2, klaCH4, klaCO2, but all are equal to 200
        # Henry constants.
        # in C# implementation different calculations but also different units
        # K_H_co2 = 0.035 * np.exp((-19410 / (100 * R)) * (1 / T_base - 1 / T_ad))  # Mliq.bar^-1 #0.0271
        # K_H_ch4 = 0.0014 * np.exp((-14240 / (100 * R)) * (1 / T_base - 1 / T_ad))  # Mliq.bar^-1 #0.00116
        # K_H_h2 = 7.8 * 10 ** -4 * np.exp(-4180 / (100 * R) * (1 / T_base - 1 / T_ad))  # Mliq.bar^-1 #7.38*10^-4

        # C# implementations
        K_H_co2 = 1 / (0.0271 * 0.08314 * T_ad)
        K_H_ch4 = 1 / (0.00116 * 0.08314 * T_ad)
        K_H_h2 = 1 / (7.38e-4 * 0.08314 * T_ad)

        return p_gas_h2o, k_p, k_L_a, K_H_co2, K_H_ch4, K_H_h2

    # *** PRIVATE methods ***

    @staticmethod
    def _getADMKparams(R, T_base, T_ad):
        """
        Return acid-base equilibrium coefficients
        :param R: gas constant
        :param T_base: outside temperature
        :param T_ad: temperature inside the digester
        :return:
        """
        K_w = 10 ** -14.0 * np.exp((55900 / (100 * R)) * (1 / T_base - 1 / T_ad))  # M #2.08 * 10 ^ -14  OK

        K_a_va = 10 ** -4.86  # M  ADM1 value = 1.38 * 10 ^ -5      OK
        K_a_bu = 10 ** -4.82  # M #1.5 * 10 ^ -5                    OK
        K_a_pro = 10 ** -4.88  # M #1.32 * 10 ^ -5                  OK
        K_a_ac = 10 ** -4.76  # M #1.74 * 10 ^ -5                   OK

        K_a_co2 = 10 ** -6.35 * np.exp((7646 / (100 * R)) * (1 / T_base - 1 / T_ad))  # M #4.94 * 10 ^ -7 OK
        K_a_IN = 10 ** -9.25 * np.exp((51965 / (100 * R)) * (1 / T_base - 1 / T_ad))  # M #1.11 * 10 ^ -9 OK

        return K_w, K_a_va, K_a_bu, K_a_pro, K_a_ac, K_a_co2, K_a_IN

    @staticmethod
    def _getADMpHULLLparams():
        """
        Return upper and lower limits for inhibiton by pH
        :return:
        """
        pH_UL_aa = 5.5                  # OK
        pH_LL_aa = 4                    # OK
        pH_UL_ac = 7                    # OK
        pH_LL_ac = 6                    # OK
        pH_UL_h2 = 6                    # OK
        pH_LL_h2 = 5                    # OK

        return pH_LL_aa, pH_UL_aa, pH_LL_ac, pH_UL_ac, pH_LL_h2, pH_UL_h2

    @staticmethod
    def _getADMk_mK_Sparams():
        K_S_IN = 10 ** -4  # M              OK
        k_m_su = 30  # d^-1                 OK
        K_S_su = 0.5  # kgCOD.m^-3          OK
        k_m_aa = 50  # d^-1                 OK
        K_S_aa = 0.3  ##kgCOD.m^-3          OK
        k_m_fa = 6  # d^-1                  OK
        K_S_fa = 0.4  # kgCOD.m^-3          OK
        K_I_h2_fa = 5 * 10 ** -6  # kgCOD.m^-3      OK
        k_m_c4 = 20  # d^-1                     OK
        # in C# 0.3 Quelle: Eignung des Anaerobic Digestion Model No.1 (ADM 1] zur Prozesssteuerung
        # landwirtschaftlicher Biogasanlagen, Gülzower Gespräche
        K_S_c4 = 0.3 # 0.2  # kgCOD.m^-3
        K_I_h2_c4 = 10 ** -5  # kgCOD.m^-3      OK
        k_m_pro = 13  # d^-1                    OK
        K_S_pro = 0.1  # kgCOD.m^-3             OK
        K_I_h2_pro = 3.5 * 10 ** -6  # kgCOD.m^-3       OK
        k_m_ac = 8  # kgCOD.m^-3                    OK
        K_S_ac = 0.15  # kgCOD.m^-3                  OK
        # in C# 0.002, Quelle: 09 monofermentation of grass silage under mesophilic conditions - measurements
        # and mathematical modeling with adm 1.pdf
        K_I_nh3 = 0.002 #0.0018  # M
        k_m_h2 = 35  # d^-1             OK
        K_S_h2 = 7 * 10 ** -6  # kgCOD.m^-3         OK

        return (K_S_IN, k_m_su, K_S_su, k_m_aa, K_S_aa, k_m_fa,
                K_S_fa, K_I_h2_fa, k_m_c4, K_S_c4, K_I_h2_c4, k_m_pro, K_S_pro, K_I_h2_pro,
                k_m_ac, K_S_ac, K_I_nh3, k_m_h2, K_S_h2)

    @staticmethod
    def _getADMYparams():
        """
        Return yield uptake parameters
        :return:
        """
        Y_su = 0.1              # OK
        Y_aa = 0.08             # OK
        Y_fa = 0.06             # OK
        Y_c4 = 0.06             # OK
        Y_pro = 0.04            # OK
        Y_ac = 0.05             # OK
        Y_h2 = 0.06             # OK

        return (Y_su, Y_aa, Y_fa, Y_c4, Y_pro, Y_ac, Y_h2)

    @staticmethod
    def _getADMfaaparams():
        """
        Return ... from amino acids parameters
        :return:
        """
        f_h2_aa = 0.06          # OK
        f_va_aa = 0.23          # OK
        f_bu_aa = 0.26          # OK
        f_pro_aa = 0.05         # OK
        f_ac_aa = 0.40          # OK

        return f_h2_aa, f_va_aa, f_bu_aa, f_pro_aa, f_ac_aa

    @staticmethod
    def _getADMfsuparams():
        """
        Return ... from sugars parameters
        :return:
        """
        f_h2_su = 0.19          # OK
        f_bu_su = 0.13          # OK
        f_pro_su = 0.27         # OK
        f_ac_su = 0.41          # OK

        return f_h2_su, f_bu_su, f_pro_su, f_ac_su

    # *** PUBLIC properties ***

    # *** PRIVATE variables ***

