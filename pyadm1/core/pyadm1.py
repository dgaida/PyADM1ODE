# -*- coding: utf-8 -*-
"""
Created on Fri Nov 04 09:56:06 2023

The ADM1ODE implementation is based on:

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

It is changed to an ODE only implementation with no DAEs similar to the implementation in Simba (ifak e.V., 2010).
More information about the implementation and characterisation of the ADM1 input stream can be found in:
Gaida, D., Dynamic real-time substrate feed optimization of anaerobic co-digestion plants, PhD thesis, Leiden, 2014.

This file has dependencies to feedstock.py and ADMparams.py

@author: Daniel Gaida
"""

import clr
import numpy as np
import pandas as pd
from typing import List, Tuple

from pyadm1.core.adm_params import ADMparams
from pyadm1.substrates.feedstock import Feedstock

# CLR reference must be added before importing from DLL
clr.AddReference("pyadm1/dlls/plant")
from biogas import ADMstate  # noqa: E402  # type: ignore


def get_state_zero_from_initial_state(csv_file: str) -> List[float]:
    """
    Read CSV file and return the initial ADM1 state vector as a list.

    Parameters
    ----------
    csv_file : str
        Path to file containing initial ADM1 state vector

    Returns
    -------
    List[float]
        Initial ADM1 state vector (37 dimensions)
    """
    initial_state = pd.read_csv(csv_file)

    # initiate variables (initial values for the reactor state at the initial time (t0)
    S_su = initial_state["S_su"][0]  # kg COD.m^-3 monosaccharides
    S_aa = initial_state["S_aa"][0]  # kg COD.m^-3 amino acids
    S_fa = initial_state["S_fa"][0]  # kg COD.m^-3 total long chain fatty acids
    S_va = initial_state["S_va"][0]  # kg COD.m^-3 total valerate
    S_bu = initial_state["S_bu"][0]  # kg COD.m^-3 total butyrate
    S_pro = initial_state["S_pro"][0]  # kg COD.m^-3 total propionate
    S_ac = initial_state["S_ac"][0]  # kg COD.m^-3 total acetate
    S_h2 = initial_state["S_h2"][0]  # kg COD.m^-3 hydrogen gas
    S_ch4 = initial_state["S_ch4"][0]  # kg COD.m^-3 methane gas
    S_co2 = initial_state["S_co2"][0]  # kmole C.m^-3 inorganic carbon
    S_nh4_ion = initial_state["S_nh4"][0]  # kmole N.m^-3 inorganic nitrogen
    S_I = initial_state["S_I"][0]  # kg COD.m^-3 soluble inerts

    X_xc = initial_state["X_xc"][0]  # kg COD.m^-3 composites
    X_ch = initial_state["X_ch"][0]  # kg COD.m^-3 carbohydrates
    X_pr = initial_state["X_pr"][0]  # kg COD.m^-3 proteins
    X_li = initial_state["X_li"][0]  # kg COD.m^-3 lipids
    X_su = initial_state["X_su"][0]  # kg COD.m^-3 sugar degraders
    X_aa = initial_state["X_aa"][0]  # kg COD.m^-3 amino acid degraders
    X_fa = initial_state["X_fa"][0]  # kg COD.m^-3 LCFA degraders
    X_c4 = initial_state["X_c4"][0]  # kg COD.m^-3 valerate and butyrate degraders
    X_pro = initial_state["X_pro"][0]  # kg COD.m^-3 propionate degraders
    X_ac = initial_state["X_ac"][0]  # kg COD.m^-3 acetate degraders
    X_h2 = initial_state["X_h2"][0]  # kg COD.m^-3 hydrogen degraders
    X_I = initial_state["X_I"][0]  # kg COD.m^-3 particulate inerts
    # X_p
    X_p = initial_state["X_p"][0]
    S_cation = initial_state["S_cation"][0]  # kmole.m^-3 cations (metallic ions, strong base)
    S_anion = initial_state["S_anion"][0]  # kmole.m^-3 anions (metallic ions, strong acid)
    S_va_ion = initial_state["S_va_ion"][0]  # kg COD.m^-3 valerate
    S_bu_ion = initial_state["S_bu_ion"][0]  # kg COD.m^-3 butyrate
    S_pro_ion = initial_state["S_pro_ion"][0]  # kg COD.m^-3 propionate
    S_ac_ion = initial_state["S_ac_ion"][0]  # kg COD.m^-3 acetate
    S_hco3_ion = initial_state["S_hco3_ion"][0]  # kmole C.m^-3 bicarbonate
    S_nh3 = initial_state["S_nh3"][0]  # kmole N.m^-3 ammonia
    pi_Sh2 = initial_state["pi_Sh2"][0]  # kg COD.m^-3 hydrogen concentration in gas phase
    pi_Sch4 = initial_state["pi_Sch4"][0]  # kg COD.m^-3 methane concentration in gas phase
    pi_Sco2 = initial_state["pi_Sco2"][0]  # kmole C.m^-3 carbon dioxide concentration in gas phas
    pTOTAL = initial_state["pTOTAL"][0]

    state_zero = [
        S_su,
        S_aa,
        S_fa,
        S_va,
        S_bu,
        S_pro,
        S_ac,
        S_h2,
        S_ch4,
        S_co2,
        S_nh4_ion,
        S_I,
        X_xc,
        X_ch,
        X_pr,
        X_li,
        X_su,
        X_aa,
        X_fa,
        X_c4,
        X_pro,
        X_ac,
        X_h2,
        X_I,
        X_p,
        S_cation,
        S_anion,
        S_va_ion,
        S_bu_ion,
        S_pro_ion,
        S_ac_ion,
        S_hco3_ion,
        S_nh3,
        pi_Sh2,
        pi_Sch4,
        pi_Sco2,
        pTOTAL,
    ]

    return state_zero


"""
Class mainly contains the ODE only implementation of the ADM1. Also contains some methods that can be used to set
the input stream or to calculate and store a couple of process values after the simulation.

"""


class PyADM1:
    """
    Main class containing the ODE implementation of ADM1.

    Attributes
    ----------
    V_liq : float
        Liquid volume of digester [m³]
    """

    # *** CONSTRUCTORS ***
    def __init__(self, feedstock: Feedstock) -> None:
        """
        Initialize PyADM1 model.

        Parameters
        ----------
        feedstock : Feedstock
            Feedstock object containing substrate information
        """
        # Physical parameter values used in BSM2 from the Rosen et al (2006) BSM2 report
        self.V_liq = 1977  # 3000 # 1977 #m^3
        self._V_gas = 304  # m^3
        self._V_ad = self.V_liq + self._V_gas  # m^-3

        self._RT = self._T_ad * self._R
        # in C# p_atm = 1.04
        self._pext = self._p_atm - 0.0084147 * np.exp(0.054 * (self._T_ad - 273.15))

        self._feedstock = feedstock

    # *** PUBLIC SET methods ***

    # *** PUBLIC GET methods ***

    # *** PUBLIC methods ***

    def createInfluent(self, Q: List[float], i: int) -> None:
        """
        Create ADM1 input stream from volumetric flow rate vector Q and set self._state_input so that the ADM1 knows
        the current input.

        Parameters
        ----------
        Q : List[float]
            list of volumetric flow rates for each substrate [m³/d]. Length of list must be equal to number of
            substrates available on the plant, e.g.: [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]
        i : int
            Time step index
        """
        self._Q = Q

        influent_state = self._feedstock.get_influent_dataframe(Q)

        self._setInfluent(influent_state, i)

    def save_final_state_in_csv(
        self,
        simulate_results: List[List[float]],
        filename: str = "digester_initial6.csv",
    ) -> None:
        """
        Save final ADM1 state vector to CSV file.

        Parameters
        ----------
        simulate_results : List[List[float]]
            List of ADM1 state vectors collected during simulation
        filename : str, optional
            Output filename, by default "digester_initial6.csv"
        """
        columns = [
            *self._feedstock.header()[:-1],
            "pi_Sh2",
            "pi_Sch4",
            "pi_Sco2",
            "pTOTAL",
        ]

        simulate_results_df = pd.DataFrame.from_records(simulate_results)
        simulate_results_df.columns = columns

        # Keep only the last row
        last_simulated_result = simulate_results_df[-1:]
        last_simulated_result.to_csv(filename, index=False)

    def calc_gas(self, pi_Sh2: float, pi_Sch4: float, pi_Sco2: float, pTOTAL: float) -> Tuple[float, float, float, float]:
        """
        Calculate biogas production rates from partial pressures.

        Parameters
        ----------
        pi_Sh2 : float
            Partial pressure of hydrogen [bar]
        pi_Sch4 : float
            Partial pressure of methane [bar]
        pi_Sco2 : float
            Partial pressure of carbon dioxide [bar]
        pTOTAL : float
            Total pressure [bar]

        Returns
        -------
        Tuple[float, float, float, float]
            q_gas, q_ch4, q_co2, p_gas [m³/d, m³/d, m³/d, bar]
        """
        p_gas_h2o, k_p, k_L_a, K_H_co2, K_H_ch4, K_H_h2 = ADMparams.getADMgasparams(self._R, self._T_base, self._T_ad)

        NQ = 44.64300

        q_gas = k_p * (pTOTAL - self._pext) / (self._RT / 1000 * NQ) * self.V_liq

        p_gas = pi_Sh2 + pi_Sch4 + pi_Sco2

        if (q_gas < 0).all():
            print(pi_Sch4, p_gas, pi_Sco2, pi_Sh2, pTOTAL, q_gas, self._pext)

        if (q_gas < 0).any():
            if not isinstance(q_gas, np.float64) and (len(q_gas) > 0):
                q_gas[q_gas < 0] = 0
            else:
                q_gas = 0

        q_ch4 = q_gas * (pi_Sch4 / p_gas)  # methane flow
        if (q_ch4 < 0).any():
            if not isinstance(q_ch4, np.float64) and len(q_ch4) > 0:
                q_ch4[q_ch4 < 0] = 0
            else:
                q_ch4 = 0

        q_co2 = q_gas * (pi_Sco2 / p_gas)  # co2 flow
        if (q_co2 < 0).any():
            if len(q_co2) > 0:
                q_co2[q_co2 < 0] = 0
            else:
                q_co2 = 0

        return q_gas, q_ch4, q_co2, p_gas

    def resume_from_broken_simulation(self, Q_CH4):
        for Qch4 in Q_CH4:
            self._Q_CH4.append(Qch4)

    def print_params_at_current_state(self, state_ADM1xp: List[float]) -> None:
        """
        Calculate and store process values from current state.

        Parameters
        ----------
        state_ADM1xp : List[float]
            Current ADM1 state vector
        """
        self._pH_l.append(np.round(ADMstate.calcPHOfADMstate(state_ADM1xp), 1))
        self._FOSTAC.append(np.round(ADMstate.calcFOSTACOfADMstate(state_ADM1xp).Value, 2))
        self._AcvsPro.append(np.round(ADMstate.calcAcetic_vs_PropionicOfADMstate(state_ADM1xp).Value, 1))
        self._VFA.append(np.round(ADMstate.calcVFAOfADMstate(state_ADM1xp, "gHAceq/l").Value, 2))
        self._TAC.append(np.round(ADMstate.calcTACOfADMstate(state_ADM1xp, "gCaCO3eq/l").Value, 1))

        # to get at least 3 values into the list, because the last three values go to the controller.
        # I am assuming here that we start from a steady state
        if len(self._pH_l) < 2:
            self._pH_l.append(np.round(ADMstate.calcPHOfADMstate(state_ADM1xp), 1))
            self._FOSTAC.append(np.round(ADMstate.calcFOSTACOfADMstate(state_ADM1xp).Value, 2))
            self._AcvsPro.append(np.round(ADMstate.calcAcetic_vs_PropionicOfADMstate(state_ADM1xp).Value, 1))
            self._VFA.append(np.round(ADMstate.calcVFAOfADMstate(state_ADM1xp, "gHAceq/l").Value, 2))
            self._TAC.append(np.round(ADMstate.calcTACOfADMstate(state_ADM1xp, "gCaCO3eq/l").Value, 1))

        print("pH(lib) = {0}".format(self._pH_l))
        print("FOS/TAC = {0}".format(self._FOSTAC))
        print("VFA = {0}".format(self._VFA))
        print("TAC = {0}".format(self._TAC))
        # print('SS = {0}'.format(ADMstate.calcSSOfADMstate(state_ADM1xp).printValue()))
        # print('VS = {0}'.format(ADMstate.calcVSOfADMstate(state_ADM1xp, 'kgCOD/m^3').printValue()))
        print("Ac/Pro = {0}".format(self._AcvsPro))
        # print('Biomass = {0}'.format(ADMstate.calcBiomassOfADMstate(state_ADM1xp).printValue()))

        # calc biogas production rates from state vector
        q_gas, q_ch4, q_co2, p_gas = self._calc_gas(state_ADM1xp)

        self._Q_GAS.append(q_gas)
        self._Q_CH4.append(q_ch4)
        self._Q_CO2.append(q_co2)
        self._P_GAS.append(p_gas)

        # to get at least 3 values into the list, because the last three values go to controller.
        # I am assuming here that we start from a steady state
        if len(self._Q_GAS) < 2:
            for i in range(0, 2):  # to have at least 4 values in the lists
                self._Q_GAS.append(q_gas)
                self._Q_CH4.append(q_ch4)
                self._Q_CO2.append(q_co2)
                self._P_GAS.append(p_gas)

        print("Q_gas = {0} m^3/d".format(self._Q_GAS))
        print("Q_ch4 = {0} m^3/d".format(self._Q_CH4))

    def ADM1_ODE(self, t: float, state_zero: List[float]) -> Tuple[float, ...]:
        """
        Calculate derivatives for ADM1 system of equations.

        Parameters
        ----------
        t : float
            Time (model is not time-dependent, but required by ODE solver)
        state_zero : List[float]
            Current ADM1 state vector (37 dimensions)

        Returns
        -------
        Tuple[float, ...]
            Derivatives dx/dt (37 dimensions)
        """
        (
            N_xc,
            N_I,
            N_aa,
            C_xc,
            C_sI,
            C_ch,
            C_pr,
            C_li,
            C_xI,
            C_su,
            C_aa,
            f_fa_li,
            C_fa,
            f_h2_su,
            f_bu_su,
            f_pro_su,
            f_ac_su,
            N_bac,
            C_bu,
            C_pro,
            C_ac,
            C_bac,
            Y_su,
            f_h2_aa,
            f_va_aa,
            f_bu_aa,
            f_pro_aa,
            f_ac_aa,
            C_va,
            Y_aa,
            Y_fa,
            Y_c4,
            Y_pro,
            C_ch4,
            Y_ac,
            Y_h2,
            K_S_IN,
            k_m_su,
            K_S_su,
            pH_UL_aa,
            pH_LL_aa,
            k_m_aa,
            K_S_aa,
            k_m_fa,
            K_S_fa,
            K_I_h2_fa,
            k_m_c4,
            K_S_c4,
            K_I_h2_c4,
            k_m_pro,
            K_S_pro,
            K_I_h2_pro,
            k_m_ac,
            K_S_ac,
            K_I_nh3,
            pH_UL_ac,
            pH_LL_ac,
            k_m_h2,
            K_S_h2,
            pH_UL_h2,
            pH_LL_h2,
            k_dec_X_su,
            k_dec_X_aa,
            k_dec_X_fa,
            k_dec_X_c4,
            k_dec_X_pro,
            k_dec_X_ac,
            k_dec_X_h2,
            K_w,
            K_a_va,
            K_a_bu,
            K_a_pro,
            K_a_ac,
            K_a_co2,
            K_a_IN,
            k_A_B_va,
            k_A_B_bu,
            k_A_B_pro,
            k_A_B_ac,
            k_A_B_co2,
            k_A_B_IN,
            p_gas_h2o,
            k_p,
            k_L_a,
            K_H_co2,
            K_H_ch4,
            K_H_h2,
        ) = ADMparams.getADMparams(self._R, self._T_base, self._T_ad)

        (
            f_ch_xc,
            f_pr_xc,
            f_li_xc,
            f_xI_xc,
            f_sI_xc,
            f_xp_xc,
            k_dis,
            k_hyd_ch,
            k_hyd_pr,
            k_hyd_li,
            k_m_c4,
            k_m_pro,
            k_m_ac,
            k_m_h2,
        ) = self._get_substrate_dependent_params()

        q_ad = np.sum(self._Q)

        f_p = 0.08  # number from ADM_private.cs
        f_ch_xb = f_ch_xc / (f_ch_xc + f_pr_xc + f_li_xc) * (1 - f_p)
        f_pr_xb = f_pr_xc / (f_ch_xc + f_pr_xc + f_li_xc) * (1 - f_p)
        f_li_xb = f_li_xc / (f_ch_xc + f_pr_xc + f_li_xc) * (1 - f_p)

        K_pH_aa, nn_aa, K_pH_ac, n_ac, K_pH_h2, n_h2 = ADMparams.getADMinhibitionparams()

        S_su = state_zero[0]
        S_aa = state_zero[1]
        S_fa = state_zero[2]
        S_va = state_zero[3]
        S_bu = state_zero[4]
        S_pro = state_zero[5]
        S_ac = state_zero[6]
        S_h2 = state_zero[7]
        S_ch4 = state_zero[8]
        # S_co2
        S_co2 = state_zero[9]
        # S_nh4
        S_nh4_ion = state_zero[10]
        S_I = state_zero[11]
        X_xc = state_zero[12]
        X_ch = state_zero[13]
        X_pr = state_zero[14]
        X_li = state_zero[15]
        X_su = state_zero[16]
        X_aa = state_zero[17]
        X_fa = state_zero[18]
        X_c4 = state_zero[19]
        X_pro = state_zero[20]
        X_ac = state_zero[21]
        X_h2 = state_zero[22]
        X_I = state_zero[23]
        # X_p: 24
        X_p = state_zero[24]
        S_cation = state_zero[25]
        S_anion = state_zero[26]
        S_va_ion = state_zero[27]
        S_bu_ion = state_zero[28]
        S_pro_ion = state_zero[29]
        S_ac_ion = state_zero[30]
        S_hco3_ion = state_zero[31]
        S_nh3 = state_zero[32]
        p_gas_h2 = state_zero[33]
        p_gas_ch4 = state_zero[34]
        p_gas_co2 = state_zero[35]
        pTOTAL = state_zero[36]

        pH = ADMstate.calcPHOfADMstate(state_zero)
        S_H_ion = 10 ** (-pH)

        S_su_in = self._state_input[0]
        S_aa_in = self._state_input[1]
        S_fa_in = self._state_input[2]
        S_va_in = self._state_input[3]
        S_bu_in = self._state_input[4]
        S_pro_in = self._state_input[5]
        S_ac_in = self._state_input[6]
        S_h2_in = self._state_input[7]
        S_ch4_in = self._state_input[8]
        S_co2_in = self._state_input[9]
        S_nh4_in = self._state_input[10]
        S_I_in = self._state_input[11]
        X_xc_in = self._state_input[12]
        X_ch_in = self._state_input[13]
        X_pr_in = self._state_input[14]
        X_li_in = self._state_input[15]
        X_su_in = self._state_input[16]
        X_aa_in = self._state_input[17]
        X_fa_in = self._state_input[18]
        X_c4_in = self._state_input[19]
        X_pro_in = self._state_input[20]
        X_ac_in = self._state_input[21]
        X_h2_in = self._state_input[22]
        X_I_in = self._state_input[23]
        X_p_in = self._state_input[24]
        S_cation_in = self._state_input[25]
        S_anion_in = self._state_input[26]
        S_va_ion_in = self._state_input[27]
        S_bu_ion_in = self._state_input[28]
        S_pro_ion_in = self._state_input[29]
        S_ac_ion_in = self._state_input[30]
        S_hco3_ion_in = self._state_input[31]
        S_nh3_in = self._state_input[32]

        I_pH_aa = (K_pH_aa**nn_aa) / (S_H_ion**nn_aa + K_pH_aa**nn_aa)  # OK
        I_pH_ac = (K_pH_ac**n_ac) / (S_H_ion**n_ac + K_pH_ac**n_ac)  # OK
        I_pH_h2 = (K_pH_h2**n_h2) / (S_H_ion**n_h2 + K_pH_h2**n_h2)  # OK
        I_IN_lim = 1 / (1 + (K_S_IN / (S_nh4_ion + S_nh3)))  # OK
        I_h2_fa = 1 / (1 + (S_h2 / K_I_h2_fa))  # OK
        I_h2_c4 = 1 / (1 + (S_h2 / K_I_h2_c4))  # OK
        I_h2_pro = 1 / (1 + (S_h2 / K_I_h2_pro))  # OK
        I_nh3 = 1 / (1 + (S_nh3 / K_I_nh3))  # OK

        I_5 = I_pH_aa * I_IN_lim  # OK
        I_6 = I_5  # OK
        I_7 = I_pH_aa * I_IN_lim * I_h2_fa  # OK
        I_8 = I_pH_aa * I_IN_lim * I_h2_c4  # OK
        I_9 = I_8  # OK
        I_10 = I_pH_aa * I_IN_lim * I_h2_pro  # OK
        I_11 = I_pH_ac * I_IN_lim * I_nh3  # OK
        I_12 = I_pH_h2 * I_IN_lim  # OK

        # this extension introduces instabilities into the simulation, so it is outcommented. the TS value calculated in
        # calcTS is also not very accurate
        # Erweiterung der hydrolyse, abhängigkeit von TS gehalt im fermenter, s. diss. von Koch 2010, S. 62
        # TS = digester.calcTS(state_zero, self.feedstock.mySubstrates, self.Q)
        # TS_digester = TS.Value

        # Khyd = 5.5 # 2.5
        # nhyd = 2.3

        # hydro_koch = 1.0 / (1.0 + math.pow(TS_digester/Khyd, nhyd))

        # by setting hydro_koch to 1, we are not using it
        hydro_koch = 1.0
        # print(hydro_koch)

        # biochemical process rates from Rosen et al (2006) BSM2 report
        Rho_1 = k_dis * X_xc  # Disintegration        OK
        # In Koch et al. hydrolysis is multiplied by a TS dependent factor
        Rho_2 = (k_hyd_ch * X_ch) * hydro_koch  # Hydrolysis of carbohydrates
        Rho_3 = (k_hyd_pr * X_pr) * hydro_koch  # Hydrolysis of proteins
        Rho_4 = (k_hyd_li * X_li) * hydro_koch  # Hydrolysis of lipids
        Rho_5 = k_m_su * S_su / (K_S_su + S_su) * X_su * I_5  # Uptake of sugars OK
        Rho_6 = k_m_aa * (S_aa / (K_S_aa + S_aa)) * X_aa * I_6  # Uptake of amino-acids OK
        # Uptake of LCFA (long-chain fatty acids)
        Rho_7 = k_m_fa * (S_fa / (K_S_fa + S_fa)) * X_fa * I_7  # OK
        # Uptake of valerate
        Rho_8 = k_m_c4 * (S_va / (K_S_c4 + S_va)) * X_c4 * (S_va / (S_bu + S_va + 1e-6)) * I_8  # OK
        # Uptake of butyrate
        Rho_9 = k_m_c4 * (S_bu / (K_S_c4 + S_bu)) * X_c4 * (S_bu / (S_bu + S_va + 1e-6)) * I_9  # OK
        Rho_10 = k_m_pro * (S_pro / (K_S_pro + S_pro)) * X_pro * I_10  # Uptake of propionate OK
        Rho_11 = k_m_ac * (S_ac / (K_S_ac + S_ac)) * X_ac * I_11  # Uptake of acetate OK
        Rho_12 = k_m_h2 * (S_h2 / (K_S_h2 + S_h2)) * X_h2 * I_12  # Uptake of hydrogen OK
        Rho_13 = k_dec_X_su * X_su  # Decay of X_su        OK
        Rho_14 = k_dec_X_aa * X_aa  # Decay of X_aa        OK
        Rho_15 = k_dec_X_fa * X_fa  # Decay of X_fa        OK
        Rho_16 = k_dec_X_c4 * X_c4  # Decay of X_c4        OK
        Rho_17 = k_dec_X_pro * X_pro  # Decay of X_pro     OK
        Rho_18 = k_dec_X_ac * X_ac  # Decay of X_ac        OK
        Rho_19 = k_dec_X_h2 * X_h2  # Decay of X_h2        OK

        # acid-base rates for the BSM2 ODE implementation from Rosen et al (2006) BSM2 report
        Rho_A_4 = k_A_B_va * (S_va_ion * S_H_ion - K_a_va * (S_va - S_va_ion))  # OK
        Rho_A_5 = k_A_B_bu * (S_bu_ion * S_H_ion - K_a_bu * (S_bu - S_bu_ion))  # OK
        Rho_A_6 = k_A_B_pro * (S_pro_ion * S_H_ion - K_a_pro * (S_pro - S_pro_ion))  # OK
        Rho_A_7 = k_A_B_ac * (S_ac_ion * S_H_ion - K_a_ac * (S_ac - S_ac_ion))  # OK
        Rho_A_10 = k_A_B_co2 * (S_hco3_ion * S_H_ion - K_a_co2 * S_co2)  # OK
        Rho_A_11 = k_A_B_IN * (S_nh3 * S_H_ion - K_a_IN * S_nh4_ion)  # OK

        # gas transfer rates from Rosen et al (2006) BSM2 report
        Rho_T_8 = k_L_a * (S_h2 - p_gas_h2 * 16 / self._RT / K_H_h2) * self.V_liq / self._V_gas
        Rho_T_9 = k_L_a * (S_ch4 - p_gas_ch4 * 64 / self._RT / K_H_ch4) * self.V_liq / self._V_gas
        Rho_T_10 = k_L_a * (S_co2 - p_gas_co2 * 1 / self._RT / K_H_co2) * self.V_liq / self._V_gas
        Rho_T_11 = k_p * (pTOTAL - self._pext) * self.V_liq / self._V_gas

        ##differential equaitons from Rosen et al (2006) BSM2 report
        # differential equations 1 to 12 (soluble matter)
        diff_S_su = q_ad / self.V_liq * (S_su_in - S_su) + Rho_2 + (1 - f_fa_li) * Rho_4 - Rho_5  # eq1    OK

        diff_S_aa = q_ad / self.V_liq * (S_aa_in - S_aa) + Rho_3 - Rho_6  # eq2                OK

        diff_S_fa = q_ad / self.V_liq * (S_fa_in - S_fa) + (f_fa_li * Rho_4) - Rho_7  # eq3        OK

        diff_S_va = q_ad / self.V_liq * (S_va_in - S_va) + (1 - Y_aa) * f_va_aa * Rho_6 - Rho_8  # eq4     OK

        diff_S_bu = (
            q_ad / self.V_liq * (S_bu_in - S_bu) + (1 - Y_su) * f_bu_su * Rho_5 + (1 - Y_aa) * f_bu_aa * Rho_6 - Rho_9
        )  # eq5                             OK

        diff_S_pro = (
            q_ad / self.V_liq * (S_pro_in - S_pro)
            + (1 - Y_su) * f_pro_su * Rho_5
            + (1 - Y_aa) * f_pro_aa * Rho_6
            + (1 - Y_c4) * 0.54 * Rho_8
            - Rho_10
        )  # eq6                              OK, factor 0.54 = fpro_va

        diff_S_ac = (
            q_ad / self.V_liq * (S_ac_in - S_ac)
            + (1 - Y_su) * f_ac_su * Rho_5
            + (1 - Y_aa) * f_ac_aa * Rho_6
            + (1 - Y_fa) * 0.7 * Rho_7
            + (1 - Y_c4) * 0.31 * Rho_8
            + (1 - Y_c4) * 0.8 * Rho_9
            + (1 - Y_pro) * 0.57 * Rho_10
            - Rho_11
        )  # eq7         OK factors are fac_fa, fac_va, fac_bu, fac_pro

        diff_S_h2 = (
            q_ad / self.V_liq * (S_h2_in - S_h2)
            + (1 - Y_su) * f_h2_su * Rho_5
            + (1 - Y_aa) * f_h2_aa * Rho_6
            + (1 - Y_fa) * 0.3 * Rho_7
            + (1 - Y_c4) * 0.15 * Rho_8
            + (1 - Y_c4) * 0.2 * Rho_9
            + (1 - Y_pro) * 0.43 * Rho_10
            - Rho_12
            - self._V_gas / self.V_liq * Rho_T_8
        )

        # eq9
        diff_S_ch4 = (
            q_ad / self.V_liq * (S_ch4_in - S_ch4)
            + (1 - Y_ac) * Rho_11
            + (1 - Y_h2) * Rho_12
            - self._V_gas / self.V_liq * Rho_T_9
        )

        ## eq10 start##
        s_1 = (
            -1 * C_xc + f_sI_xc * C_sI + f_ch_xc * C_ch + f_pr_xc * C_pr + f_li_xc * C_li + f_xI_xc * C_xI
        )  # OK, equals -fco2Xc except that in C# we also have Xp
        s_2 = -1 * C_ch + C_su  # TODO: missing in C#
        s_3 = -1 * C_pr + C_aa  # TODO: missing in C#
        s_4 = -1 * C_li + (1 - f_fa_li) * C_su + f_fa_li * C_fa  # OK, equals -fco2,xli
        s_5 = (
            -1 * C_su + (1 - Y_su) * (f_bu_su * C_bu + f_pro_su * C_pro + f_ac_su * C_ac) + Y_su * C_bac
        )  # OK, equals -fco2,su
        s_6 = (
            -1 * C_aa + (1 - Y_aa) * (f_va_aa * C_va + f_bu_aa * C_bu + f_pro_aa * C_pro + f_ac_aa * C_ac) + Y_aa * C_bac
        )  # OK, equals -fco2,aa
        s_7 = -1 * C_fa + (1 - Y_fa) * 0.7 * C_ac + Y_fa * C_bac  # OK, equals -fco2,fa, 0.7 = fac,fa
        # OK, equals -fco2,va, factors are fpro,va and facva
        s_8 = -1 * C_va + (1 - Y_c4) * 0.54 * C_pro + (1 - Y_c4) * 0.31 * C_ac + Y_c4 * C_bac
        s_9 = -1 * C_bu + (1 - Y_c4) * 0.8 * C_ac + Y_c4 * C_bac  # OK, equals -fco2,bu, factor equals fac,bu
        s_10 = -1 * C_pro + (1 - Y_pro) * 0.57 * C_ac + Y_pro * C_bac  # OK, equals -fco2,pro, factor equals fac,pro
        s_11 = -1 * C_ac + (1 - Y_ac) * C_ch4 + Y_ac * C_bac  # OK, equals -fco2,ac
        s_12 = (1 - Y_h2) * C_ch4 + Y_h2 * C_bac  # OK, equals -fco2,h2
        s_13 = -1 * C_bac + C_xc  # OK, equals -fco2,xb

        Sigma = (
            s_1 * Rho_1
            + s_2 * Rho_2
            + s_3 * Rho_3
            + s_4 * Rho_4
            + s_5 * Rho_5
            + s_6 * Rho_6
            + s_7 * Rho_7
            + s_8 * Rho_8
            + s_9 * Rho_9
            + s_10 * Rho_10
            + s_11 * Rho_11
            + s_12 * Rho_12
            + s_13 * (Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19)
        )  # OK, except s_2, s_3

        diff_S_co2 = q_ad / self.V_liq * (S_co2_in - S_co2) - Sigma - self._V_gas / self.V_liq * Rho_T_10 + Rho_A_10
        ## eq10 end##

        # TODO: in C# last term is different using fsin,xb
        diff_S_nh4_ion = (
            q_ad / self.V_liq * (S_nh4_in - S_nh4_ion)  # + (N_xc - f_xI_xc * N_I - f_sI_xc * N_I-f_pr_xc * N_aa) * Rho_1 -
            - Y_su * N_bac * Rho_5
            + (N_aa - Y_aa * N_bac) * Rho_6
            - Y_fa * N_bac * Rho_7
            - Y_c4 * N_bac * Rho_8
            - Y_c4 * N_bac * Rho_9
            - Y_pro * N_bac * Rho_10
            - Y_ac * N_bac * Rho_11
            - Y_h2 * N_bac * Rho_12
            + (N_bac - N_xc) * (Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19)
        ) + Rho_A_11  # eq11

        diff_S_I = q_ad / self.V_liq * (S_I_in - S_I) + f_sI_xc * Rho_1  # eq12            OK

        # Differential equations 13 to 24 (particulate matter)
        diff_X_xc = q_ad / self.V_liq * (X_xc_in - X_xc) - Rho_1  # eq13

        diff_X_ch = (
            q_ad / self.V_liq * (X_ch_in - X_ch)
            + f_ch_xc * Rho_1
            - Rho_2
            + f_ch_xb * (Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19)
        )  # eq14

        diff_X_pr = (
            q_ad / self.V_liq * (X_pr_in - X_pr)
            + f_pr_xc * Rho_1
            - Rho_3
            + f_pr_xb * (Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19)
        )  # eq15

        diff_X_li = (
            q_ad / self.V_liq * (X_li_in - X_li)
            + f_li_xc * Rho_1
            - Rho_4
            + f_li_xb * (Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19)
        )  # eq16

        diff_X_su = q_ad / self.V_liq * (X_su_in - X_su) + Y_su * Rho_5 - Rho_13  # eq17       OK

        diff_X_aa = q_ad / self.V_liq * (X_aa_in - X_aa) + Y_aa * Rho_6 - Rho_14  # eq18       OK

        diff_X_fa = q_ad / self.V_liq * (X_fa_in - X_fa) + Y_fa * Rho_7 - Rho_15  # eq19       OK

        diff_X_c4 = q_ad / self.V_liq * (X_c4_in - X_c4) + Y_c4 * Rho_8 + Y_c4 * Rho_9 - Rho_16  # eq20    OK

        diff_X_pro = q_ad / self.V_liq * (X_pro_in - X_pro) + Y_pro * Rho_10 - Rho_17  # eq21  OK

        diff_X_ac = q_ad / self.V_liq * (X_ac_in - X_ac) + Y_ac * Rho_11 - Rho_18  # eq22      OK

        diff_X_h2 = q_ad / self.V_liq * (X_h2_in - X_h2) + Y_h2 * Rho_12 - Rho_19  # eq23      OK

        diff_X_I = q_ad / self.V_liq * (X_I_in - X_I) + f_xI_xc * Rho_1  # eq24        OK

        # X_p : as in Simba implementation 2010
        diff_X_p = (
            q_ad / self.V_liq * (X_p_in - X_p)
            - f_xp_xc * Rho_1
            + f_p * (Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19)
        )

        # Differential equations 25 and 26 (cations and anions)
        diff_S_cation = q_ad / self.V_liq * (S_cation_in - S_cation)  # eq25           OK

        diff_S_anion = q_ad / self.V_liq * (S_anion_in - S_anion)  # eq26              OK

        # Differential equations 27 to 32 (ion states, only for ODE implementation)
        diff_S_va_ion = q_ad / self.V_liq * (S_va_ion_in - S_va_ion) - Rho_A_4  # eq27

        diff_S_bu_ion = q_ad / self.V_liq * (S_bu_ion_in - S_bu_ion) - Rho_A_5  # eq28

        diff_S_pro_ion = q_ad / self.V_liq * (S_pro_ion_in - S_pro_ion) - Rho_A_6  # eq29

        diff_S_ac_ion = q_ad / self.V_liq * (S_ac_ion_in - S_ac_ion) - Rho_A_7  # eq30

        diff_S_hco3_ion = q_ad / self.V_liq * (S_hco3_ion_in - S_hco3_ion) - Rho_A_10  # eq31

        diff_S_nh3 = q_ad / self.V_liq * (S_nh3_in - S_nh3) - Rho_A_11  # eq32

        # Gas phase equations: Differential equations 33 to 35
        diff_p_gas_h2 = Rho_T_8 * self._RT / 16 - p_gas_h2 / pTOTAL * Rho_T_11  # eq33

        diff_p_gas_ch4 = Rho_T_9 * self._RT / 64 - p_gas_ch4 / pTOTAL * Rho_T_11  # eq34

        diff_p_gas_co2 = Rho_T_10 * self._RT - p_gas_co2 / pTOTAL * Rho_T_11  # eq35

        diff_pTOTAL = self._RT / 16 * Rho_T_8 + self._RT / 64 * Rho_T_9 + self._RT * Rho_T_10 - Rho_T_11

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
            diff_S_nh4_ion,
            diff_S_I,
            diff_X_xc,
            diff_X_ch,
            diff_X_pr,
            diff_X_li,
            diff_X_su,
            diff_X_aa,
            diff_X_fa,
            diff_X_c4,
            diff_X_pro,
            diff_X_ac,
            diff_X_h2,
            diff_X_I,
            diff_X_p,
            diff_S_cation,
            diff_S_anion,
            diff_S_va_ion,
            diff_S_bu_ion,
            diff_S_pro_ion,
            diff_S_ac_ion,
            diff_S_hco3_ion,
            diff_S_nh3,
            diff_p_gas_h2,
            diff_p_gas_ch4,
            diff_p_gas_co2,
            diff_pTOTAL,
        )

    # *** PRIVATE methods ***

    def _setInfluent(self, influent_state: pd.DataFrame, i: int) -> None:
        """
        Set influent values for ADM1 at current simulation step. Sets the
        self._state_input variable that is accessed by the ADM1_ODE method.

        Parameters
        ----------
        influent_state : pd.DataFrame
            DataFrame containing ADM1 input variables over n time steps
        i : int
            the ith value is taken out of the given influent_state dictionary.
        """
        ##variable definition
        # Input values (influent/feed)
        S_su_in = influent_state["S_su"][i]  # kg COD.m^-3
        S_aa_in = influent_state["S_aa"][i]  # kg COD.m^-3
        S_fa_in = influent_state["S_fa"][i]  # kg COD.m^-3
        S_va_in = influent_state["S_va"][i]  # kg COD.m^-3
        S_bu_in = influent_state["S_bu"][i]  # kg COD.m^-3
        S_pro_in = influent_state["S_pro"][i]  # kg COD.m^-3
        S_ac_in = influent_state["S_ac"][i]  # kg COD.m^-3
        S_h2_in = influent_state["S_h2"][i]  # kg COD.m^-3
        S_ch4_in = influent_state["S_ch4"][i]  # kg COD.m^-3
        S_IC_in = influent_state["S_co2"][i]  # kmole C.m^-3
        S_IN_in = influent_state["S_nh4"][i]  # kmole N.m^-3
        S_I_in = influent_state["S_I"][i]  # kg COD.m^-3

        X_xc_in = influent_state["X_xc"][i]  # kg COD.m^-3
        X_ch_in = influent_state["X_ch"][i]  # kg COD.m^-3
        X_pr_in = influent_state["X_pr"][i]  # kg COD.m^-3
        X_li_in = influent_state["X_li"][i]  # kg COD.m^-3
        X_su_in = influent_state["X_su"][i]  # kg COD.m^-3
        X_aa_in = influent_state["X_aa"][i]  # kg COD.m^-3
        X_fa_in = influent_state["X_fa"][i]  # kg COD.m^-3
        X_c4_in = influent_state["X_c4"][i]  # kg COD.m^-3
        X_pro_in = influent_state["X_pro"][i]  # kg COD.m^-3
        X_ac_in = influent_state["X_ac"][i]  # kg COD.m^-3
        X_h2_in = influent_state["X_h2"][i]  # kg COD.m^-3
        X_I_in = influent_state["X_I"][i]  # kg COD.m^-3
        X_p_in = influent_state["X_p"][i]

        S_cation_in = influent_state["S_cation"][i]  # kmole.m^-3
        S_anion_in = influent_state["S_anion"][i]  # kmole.m^-3
        S_va_ion_in = influent_state["S_va_ion"][i]  # kg COD.m^-3
        S_bu_ion_in = influent_state["S_bu_ion"][i]  # kg COD.m^-3
        S_pro_ion_in = influent_state["S_pro_ion"][i]  # kg COD.m^-3
        S_ac_ion_in = influent_state["S_ac_ion"][i]  # kg COD.m^-3
        S_hco3_ion_in = influent_state["S_hco3_ion"][i]  # kg COD.m^-3
        S_nh3_in = influent_state["S_nh3"][i]  # kg COD.m^-3

        q_ad = influent_state["Q"][i]  # m^3/d

        self._state_input = [
            S_su_in,
            S_aa_in,
            S_fa_in,
            S_va_in,
            S_bu_in,
            S_pro_in,
            S_ac_in,
            S_h2_in,
            S_ch4_in,
            S_IC_in,
            S_IN_in,
            S_I_in,
            X_xc_in,
            X_ch_in,
            X_pr_in,
            X_li_in,
            X_su_in,
            X_aa_in,
            X_fa_in,
            X_c4_in,
            X_pro_in,
            X_ac_in,
            X_h2_in,
            X_I_in,
            X_p_in,
            S_cation_in,
            S_anion_in,
            S_va_ion_in,
            S_bu_ion_in,
            S_pro_ion_in,
            S_ac_ion_in,
            S_hco3_ion_in,
            S_nh3_in,
            q_ad,
        ]

    def _get_substrate_dependent_params(self) -> Tuple[float, ...]:
        """
        Get substrate-dependent ADM1 parameters via C# DLL methods. The parameters
        used in these calculations are defined in substrate_...xml. Documentation in PhD thesis of D. Gaida, 2014

        Returns
        -------
        Tuple[float, ...]
            f_ch_xc, f_pr_xc, f_li_xc, f_xI_xc, f_sI_xc, f_xp_xc, k_dis,
            k_hyd_ch, k_hyd_pr, k_hyd_li, k_m_c4, k_m_pro, k_m_ac, k_m_h2
        """
        f_ch_xc, f_pr_xc, f_li_xc, f_xI_xc, f_sI_xc, f_xp_xc = self._feedstock.mySubstrates().calcfFactors(self._Q)
        f_xp_xc = max(f_xp_xc, 0)

        k_dis = self._feedstock.mySubstrates().calcDisintegrationParam(self._Q)

        k_hyd_ch, k_hyd_pr, k_hyd_li = self._feedstock.mySubstrates().calcHydrolysisParams(self._Q)

        k_m_c4, k_m_pro, k_m_ac, k_m_h2 = self._feedstock.mySubstrates().calcMaxUptakeRateParams(self._Q)

        return (
            f_ch_xc,
            f_pr_xc,
            f_li_xc,
            f_xI_xc,
            f_sI_xc,
            f_xp_xc,
            k_dis,
            k_hyd_ch,
            k_hyd_pr,
            k_hyd_li,
            k_m_c4,
            k_m_pro,
            k_m_ac,
            k_m_h2,
        )

    def _calc_gas(self, state_ADM1: List[float]) -> Tuple[float, float, float, float]:
        """
        Calculate biogas production rates from ADM1 state vector.

        Parameters
        ----------
        state_ADM1 : List[float]
            ADM1 state vector (37 dimensions)

        Returns
        -------
        Tuple[float, float, float, float]
            q_gas, q_ch4, q_co2, p_gas
        """
        pi_Sh2, pi_Sch4 = state_ADM1[-4], state_ADM1[-3]
        pi_Sco2, pTOTAL = state_ADM1[-2], state_ADM1[-1]

        q_gas, q_ch4, q_co2, p_gas = self.calc_gas(pi_Sh2, pi_Sch4, pi_Sco2, pTOTAL)

        return q_gas, q_ch4, q_co2, p_gas

    # *** PRIVATE STATIC/CLASS methods ***

    # *** PUBLIC properties ***

    def T_ad(self) -> float:
        """
        Returns temperature inside the digester
        :return: temperature inside the digester
        """
        return self._T_ad

    def feedstock(self) -> Feedstock:
        """
        Returns object of the feedstock class
        :return: object of the feedstock class
        """
        return self._feedstock

    def Q_GAS(self) -> List[float]:
        """Biogas production rates over all simulations [m³/d]."""
        return self._Q_GAS

    def Q_CH4(self) -> List[float]:
        """Methane production rates over all simulations [m³/d]."""
        return self._Q_CH4

    def Q_CO2(self) -> List[float]:
        """CO2 production rates over all simulations [m³/d]."""
        return self._Q_CO2

    def P_GAS(self) -> List[float]:
        """Gas pressures over all simulations [bar]."""
        return self._P_GAS

    def pH_l(self) -> List[float]:
        """pH values over all simulations."""
        return self._pH_l

    def VFA_TA(self) -> List[float]:
        """VFA/TA ratios over all simulations."""
        return self._FOSTAC

    def AcvsPro(self) -> List[float]:
        """Acetic/Propionic acid ratios over all simulations."""
        return self._AcvsPro

    def VFA(self) -> List[float]:
        """VFA concentrations over all simulations [g/L]."""
        return self._VFA

    def TAC(self) -> List[float]:
        """TA concentrations over all simulations [g CaCO3 eq/L]."""
        return self._TAC

    # TODO: those properties should be private as well

    V_liq = None

    # *** PRIVATE variables ***

    _Q_GAS = []  # produced biogas over all simulations
    _Q_CH4 = []  # produced methane over all simulations
    _Q_CO2 = []  # produced co2 over all simulations
    _P_GAS = []  # gas pressures over all simulations
    _pH_l = []  # pH values over all simulations
    _FOSTAC = []  # ratio of VFA over TA over all simulations
    _AcvsPro = []  # ratio of acetic over propionic acid over all simulations
    _VFA = []  # VFA concentrations over all simulations
    _TAC = []  # TA concentrations over all simulations

    # vector of volumetric flow rates of the substrates. Length must be equal to the number of substrates defined in xml
    _Q = None

    # object of the feedstock class. E.g. used to calculate ADM1 input stream
    _feedstock = None

    # contains ADM1 input stream as a 34dim vector
    _state_input = None

    # gas volume of digester
    _V_gas = None
    # total volume of digester: liquid + gas volume
    _V_ad = None

    ## unit for each parameter is commented after it is declared (inline)
    ## if the suggested value for the parameter is different -
    ## The original default value from the original ADM1 report by Batstone et al (2002), is commented after each unit (inline)
    # temperature inside the digester
    _T_ad = 308.15  # 308.15  # k ##T_ad #=35 C

    # outside temperature at the biogas plant in K
    _T_base = 295.15  # 304.15 # 298.15  # K = 17 °C

    # R * T_ad
    _RT = None
    _pext = None  # external pressure

    _R = 0.08313999999  # 0.083145 #bar.M^-1.K^-1
    # atmospheric pressure
    _p_atm = 1.04  # 1.013 #bar      got 1.04 from C# implementation
