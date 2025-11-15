# pyadm1/core/adm1.py
"""
Anaerobic Digestion Model No. 1 (ADM1) - Main Implementation

This module contains the core ADM1 class implementing the complete ODE system
for anaerobic digestion simulation, adapted for agricultural biogas plants.

The implementation is based on:
- Batstone et al. (2002): ADM1 IWA Task Group model
- Rosen et al. (2006): BSM2 implementation
- Gaida (2014): Agricultural substrate characterization

AND

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

References:
    Sadrimajd, P., et al. (2021). PyADM1: a Python implementation of
        Anaerobic Digestion Model No. 1. bioRxiv.
    Gaida, D. (2014). Dynamic real-time substrate feed optimization of
        anaerobic co-digestion plants. PhD thesis, Leiden University.

Example:
    >>> from pyadm1.core import ADM1
    >>> from pyadm1.substrates import Feedstock
    >>>
    >>> feedstock = Feedstock(feeding_freq=48)
    >>> adm1 = ADM1(feedstock, V_liq=2000, T_ad=308.15)
    >>>
    >>> # Create influent stream
    >>> Q = [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]  # m³/d
    >>> adm1.create_influent(Q, 0)
    >>>
    >>> # Calculate gas production
    >>> state = [0.01] * 37  # Initial state
    >>> q_gas, q_ch4, q_co2, p_gas = adm1.calc_gas(*state[33:37])
"""

import os
import clr
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

from pyadm1.core.adm_params import ADMParams
from pyadm1.core.adm_equations import BiochemicalProcesses
from pyadm1.substrates.feedstock import Feedstock

# CLR reference must be added before importing from DLL
dll_path = os.path.join(os.path.dirname(__file__), "..", "dlls")
clr.AddReference(os.path.join(dll_path, "plant"))
from biogas import ADMstate  # noqa: E402  # type: ignore


def get_state_zero_from_initial_state(csv_file: str) -> List[float]:
    """
    Load initial ADM1 state vector from CSV file.

    The CSV file should contain a single row with 37 columns representing
    the complete ADM1 state vector.

    Args:
        csv_file: Path to CSV file containing initial state

    Returns:
        Initial ADM1 state vector (37 elements)

    Raises:
        FileNotFoundError: If CSV file does not exist
        ValueError: If CSV format is invalid

    Example:
        >>> state = get_state_zero_from_initial_state('data/digester_initial.csv')
        >>> len(state)
        37
    """
    initial_state = pd.read_csv(csv_file)

    # Extract state variables in correct order
    state_zero = [
        initial_state["S_su"][0],  # 0: kg COD.m^-3 monosaccharides
        initial_state["S_aa"][0],  # 1: kg COD.m^-3 amino acids
        initial_state["S_fa"][0],  # 2: kg COD.m^-3 total long chain fatty acids
        initial_state["S_va"][0],  # 3: kg COD.m^-3 total valerate
        initial_state["S_bu"][0],  # 4: kg COD.m^-3 total butyrate
        initial_state["S_pro"][0],  # 5: kg COD.m^-3 total propionate
        initial_state["S_ac"][0],  # 6: kg COD.m^-3 total acetate
        initial_state["S_h2"][0],  # 7: kg COD.m^-3 hydrogen gas
        initial_state["S_ch4"][0],  # 8: kg COD.m^-3 methane gas
        initial_state["S_co2"][0],  # 9: kmole C.m^-3 inorganic carbon
        initial_state["S_nh4"][0],  # 10: kmole N.m^-3 inorganic nitrogen
        initial_state["S_I"][0],  # 11: kg COD.m^-3 soluble inerts
        initial_state["X_xc"][0],  # 12: kg COD.m^-3 composites
        initial_state["X_ch"][0],  # 13: kg COD.m^-3 carbohydrates
        initial_state["X_pr"][0],  # 14: kg COD.m^-3 proteins
        initial_state["X_li"][0],  # 15: kg COD.m^-3 lipids
        initial_state["X_su"][0],  # 16: kg COD.m^-3 sugar degraders
        initial_state["X_aa"][0],  # 17: kg COD.m^-3 amino acid degraders
        initial_state["X_fa"][0],  # 18: kg COD.m^-3 LCFA degraders
        initial_state["X_c4"][0],  # 19: kg COD.m^-3 valerate and butyrate degraders
        initial_state["X_pro"][0],  # 20: kg COD.m^-3 propionate degraders
        initial_state["X_ac"][0],  # 21: kg COD.m^-3 acetate degraders
        initial_state["X_h2"][0],  # 22: kg COD.m^-3 hydrogen degraders
        initial_state["X_I"][0],  # 23: kg COD.m^-3 particulate inerts
        initial_state["X_p"][0],  # 24: kg COD.m^-3 particulate products
        initial_state["S_cation"][0],  # 25: kmole.m^-3 cations (metallic ions, strong base)
        initial_state["S_anion"][0],  # 26: kmole.m^-3 anions (metallic ions, strong acid)
        initial_state["S_va_ion"][0],  # 27: kg COD.m^-3 valerate ion
        initial_state["S_bu_ion"][0],  # 28: kg COD.m^-3 butyrate ion
        initial_state["S_pro_ion"][0],  # 29: kg COD.m^-3 propionate ion
        initial_state["S_ac_ion"][0],  # 30: kg COD.m^-3 acetate ion
        initial_state["S_hco3_ion"][0],  # 31: kmole C.m^-3 bicarbonate
        initial_state["S_nh3"][0],  # 32: kmole N.m^-3 ammonia
        initial_state["pi_Sh2"][0],  # 33: kg COD.m^-3 hydrogen concentration in gas phase
        initial_state["pi_Sch4"][0],  # 34: kg COD.m^-3 methane concentration in gas phase
        initial_state["pi_Sco2"][0],  # 35: kmole C.m^-3 carbon dioxide concentration in gas phase
        initial_state["pTOTAL"][0],  # 36: bar total pressure
    ]

    return state_zero


class ADM1:
    """
    Main class implementing ADM1 as pure ODE system.

    This class manages the ADM1 state, parameters, and provides methods for
    simulation including influent stream creation, gas production calculation,
    and state tracking.

    Attributes:
        V_liq: Liquid volume [m³]
        T_ad: Operating temperature [K]
        feedstock: Feedstock object for substrate management

    Example:
        >>> feedstock = Feedstock(feeding_freq=48)
        >>> adm1 = ADM1(feedstock, V_liq=2000, T_ad=308.15)
        >>> adm1.create_influent([15, 10, 0, 0, 0, 0, 0, 0, 0, 0], 0)
    """

    def __init__(self, feedstock: Feedstock, V_liq: float = 1977.0, V_gas: float = 304.0, T_ad: float = 308.15) -> None:
        """
        Initialize ADM1 model.

        Args:
            feedstock: Feedstock object containing substrate information. E.g. used to calculate ADM1 input stream.
            V_liq: Liquid volume [m³]
            V_gas: Gas volume [m³]
            T_ad: Operating temperature [K] (default: 308.15 = 35°C)
        """
        # Physical parameters
        self.V_liq = V_liq  # liquid volume of digester
        self._V_gas = V_gas  # gas volume of digester
        self._V_ad = self.V_liq + self._V_gas  # total volume of digester: liquid + gas volume
        self._T_ad = T_ad  # temperature inside the digester

        # Constants
        self._R = 0.08313999999  # 0.083145  Gas constant [bar·M^-1·K^-1]
        self._T_base = 295.15  # Base temperature (outside temperature at the biogas plant, 304.15, 298.15) [K]
        self._p_atm = 1.04  # Atmospheric pressure [bar] (1.013 bar; got 1.04 from C# implementation)

        # Calculated parameters
        self._RT = self._R * self._T_ad  # R * T_ad
        # external pressures
        self._pext = self._p_atm - 0.0084147 * np.exp(0.054 * (self._T_ad - 273.15))

        # Feedstock and state
        self._feedstock = feedstock
        # vector of volumetric flow rates of the substrates. Length must be equal to the number of
        # substrates defined in xml
        self._Q: Optional[List[float]] = None
        self._state_input: Optional[List[float]] = None  # contains ADM1 input stream as a 34dim vector

        # Result tracking lists
        # Result tracking lists
        self._Q_GAS: List[float] = []  # produced biogas over all simulations [m³/d]
        self._Q_CH4: List[float] = []  # produced methane over all simulations [m³/d]
        self._Q_CO2: List[float] = []  # produced CO2 over all simulations [m³/d]
        self._P_GAS: List[float] = []  # gas pressures over all simulations [bar]
        self._pH_l: List[float] = []  # pH values over all simulations [-]
        self._FOSTAC: List[float] = []  # ratio of VFA over TA over all simulations [-]
        self._AcvsPro: List[float] = []  # ratio of acetic over propionic acid over all simulations [-]
        self._VFA: List[float] = []  # VFA concentrations over all simulations [g/L]
        self._TAC: List[float] = []  # TA concentrations over all simulations [g CaCO3/L]

    def create_influent(self, Q: List[float], i: int) -> None:
        """
        Create ADM1 input stream from volumetric flow rates.

        Calculates the ADM1 influent state by mixing substrate streams according
        to their volumetric flow rates. The resulting influent composition is
        stored internally for use in ODE calculations.

        Args:
            Q: Volumetric flow rates for each substrate [m³/d]
               Length must equal number of substrates in feedstock
            i: Time step index for accessing influent dataframe

        Example:
            >>> adm1.create_influent([15, 10, 0, 0, 0, 0, 0, 0, 0, 0], 0)
        """
        self._Q = Q
        influent_state = self._feedstock.get_influent_dataframe(Q)
        self._set_influent(influent_state, i)

    def calc_gas(self, pi_Sh2: float, pi_Sch4: float, pi_Sco2: float, pTOTAL: float) -> Tuple[float, float, float, float]:
        """
        Calculate biogas production rates from partial pressures.

        Uses the ideal gas law and Henry's constants to calculate gas flow rates
        from the gas phase partial pressures.

        Args:
            pi_Sh2: Hydrogen partial pressure [bar]
            pi_Sch4: Methane partial pressure [bar]
            pi_Sco2: CO2 partial pressure [bar]
            pTOTAL: Total gas pressure [bar]

        Returns:
            Tuple containing:
                - q_gas: Total biogas flow rate [m³/d]
                - q_ch4: Methane flow rate [m³/d]
                - q_co2: CO2 flow rate [m³/d]
                - p_gas: Total gas partial pressure (excl. H2O) [bar]

        Example:
            >>> q_gas, q_ch4, q_co2, p_gas = adm1.calc_gas(5e-6, 0.55, 0.42, 0.98)
            >>> print(f"Biogas: {q_gas:.1f} m³/d, Methane: {q_ch4:.1f} m³/d")
        """
        _, k_p, _, _, _, _ = ADMParams.getADMgasparams(self._R, self._T_base, self._T_ad)

        # Ideal gas law constant for conversion
        NQ = 44.643

        # Total biogas flow from pressure difference
        q_gas = k_p * (pTOTAL - self._pext) / (self._RT / 1000 * NQ) * self.V_liq

        # Total gas partial pressure (excluding water vapor)
        p_gas = pi_Sh2 + pi_Sch4 + pi_Sco2

        # Ensure non-negative gas flows
        if isinstance(q_gas, np.ndarray):
            q_gas = np.maximum(q_gas, 0.0)
        else:
            q_gas = max(q_gas, 0.0)

        # Calculate component flows from partial pressures
        if p_gas > 0:
            q_ch4 = q_gas * (pi_Sch4 / p_gas)
            q_co2 = q_gas * (pi_Sco2 / p_gas)
        else:
            q_ch4 = 0.0
            q_co2 = 0.0

        # Ensure non-negative
        if isinstance(q_ch4, np.ndarray):
            q_ch4 = np.maximum(q_ch4, 0.0)
            q_co2 = np.maximum(q_co2, 0.0)
        else:
            q_ch4 = max(q_ch4, 0.0)
            q_co2 = max(q_co2, 0.0)

        return q_gas, q_ch4, q_co2, p_gas

    def resume_from_broken_simulation(self, Q_CH4):
        for Qch4 in Q_CH4:
            self._Q_CH4.append(Qch4)

    def save_final_state_in_csv(self, simulate_results: List[List[float]], filename: str = "digester_final.csv") -> None:
        """
        Save final ADM1 state vector to CSV file.

        Exports only the last state from simulation results, which can be used
        as initial state for subsequent simulations.

        Args:
            simulate_results: List of ADM1 state vectors from simulation
            filename: Output CSV filename

        Example:
            >>> results = [[0.01]*37, [0.02]*37, [0.03]*37]
            >>> adm1.save_final_state_in_csv(results, 'final_state.csv')
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

    def print_params_at_current_state(self, state_ADM1xp: List[float]) -> None:
        """
        Calculate and print process parameters from current state.

        Computes and displays key process indicators including pH, VFA, TAC,
        and gas production rates. Also stores values in tracking lists.

        Args:
            state_ADM1xp: Current ADM1 state vector (37 elements)

        Example:
            >>> adm1.print_params_at_current_state(state_vector)
            pH(lib) = [7.2, 7.3]
            FOS/TAC = [0.25, 0.26]
            ...
        """
        # Calculate process indicators using DLL
        self._pH_l.append(np.round(ADMstate.calcPHOfADMstate(state_ADM1xp), 1))
        self._FOSTAC.append(np.round(ADMstate.calcFOSTACOfADMstate(state_ADM1xp).Value, 2))
        self._AcvsPro.append(np.round(ADMstate.calcAcetic_vs_PropionicOfADMstate(state_ADM1xp).Value, 1))
        self._VFA.append(np.round(ADMstate.calcVFAOfADMstate(state_ADM1xp, "gHAceq/l").Value, 2))
        self._TAC.append(np.round(ADMstate.calcTACOfADMstate(state_ADM1xp, "gCaCO3eq/l").Value, 1))

        # Ensure at least 2 values in lists (because the last three values go to the controller)
        # I am assuming here that we start from a steady state
        if len(self._pH_l) < 2:
            self._pH_l.append(self._pH_l[-1])
            self._FOSTAC.append(self._FOSTAC[-1])
            self._AcvsPro.append(self._AcvsPro[-1])
            self._VFA.append(self._VFA[-1])
            self._TAC.append(self._TAC[-1])

        # Print process values
        print(f"pH(lib) = {self._pH_l}")
        # print(f"FOS/TAC = {self._FOSTAC}")
        # print(f"VFA = {self._VFA}")
        # print(f"TAC = {self._TAC}")
        # print(f"Ac/Pro = {self._AcvsPro}")

        # Calculate and store gas production
        q_gas, q_ch4, q_co2, p_gas = self.calc_gas(state_ADM1xp[33], state_ADM1xp[34], state_ADM1xp[35], state_ADM1xp[36])

        self._Q_GAS.append(q_gas)
        self._Q_CH4.append(q_ch4)
        self._Q_CO2.append(q_co2)
        self._P_GAS.append(p_gas)

        # Ensure at least 2 values, because the last three values go to controller.
        # I am assuming here that we start from a steady state
        if len(self._Q_GAS) < 2:
            for _ in range(2):  # to have at least 4 values in the lists
                self._Q_GAS.append(q_gas)
                self._Q_CH4.append(q_ch4)
                self._Q_CO2.append(q_co2)
                self._P_GAS.append(p_gas)

        print(f"Q_gas = {self._Q_GAS} m^3/d")
        print(f"Q_ch4 = {self._Q_CH4} m^3/d")

    def ADM1_ODE(self, t: float, state_zero: List[float]) -> Tuple[float, ...]:
        """
        Calculate derivatives for ADM1 ODE system.

        This is the main ODE function that computes dy/dt for all 37 state
        variables. Uses process rate equations and stoichiometric relationships.

        Args:
            t: Current time [days] (not used, system is autonomous)
            state_zero: Current ADM1 state vector (37 elements)

        Returns:
            Tuple of 37 derivatives (dy/dt)

        Note:
            This method is called by the ODE solver and should not be called
            directly by users.
        """
        # Get all ADM1 parameters
        params_tuple = ADMParams.getADMparams(self._R, self._T_base, self._T_ad)

        # Get substrate-dependent parameters
        substrate_params = self._get_substrate_dependent_params()

        # Unpack frequently used parameters
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
        ) = params_tuple[:36]

        # Get kinetic parameters (starting from index 36)
        kinetic_params = {
            "K_S_IN": params_tuple[36],
            "k_m_su": params_tuple[37],
            "K_S_su": params_tuple[38],
            "k_m_aa": params_tuple[41],
            "K_S_aa": params_tuple[42],
            "k_m_fa": params_tuple[43],
            "K_S_fa": params_tuple[44],
            "K_I_h2_fa": params_tuple[45],
            "k_m_c4": params_tuple[46],
            "K_S_c4": params_tuple[47],
            "K_I_h2_c4": params_tuple[48],
            "k_m_pro": params_tuple[49],
            "K_S_pro": params_tuple[50],
            "K_I_h2_pro": params_tuple[51],
            "k_m_ac": params_tuple[52],
            "K_S_ac": params_tuple[53],
            "K_I_nh3": params_tuple[54],
            "k_m_h2": params_tuple[57],
            "K_S_h2": params_tuple[58],
            "k_dec_X_su": params_tuple[61],
            "k_dec_X_aa": params_tuple[62],
            "k_dec_X_fa": params_tuple[63],
            "k_dec_X_c4": params_tuple[64],
            "k_dec_X_pro": params_tuple[65],
            "k_dec_X_ac": params_tuple[66],
            "k_dec_X_h2": params_tuple[67],
        }

        # Get acid-base and gas parameters
        acid_base_params = {
            "K_w": params_tuple[68],
            "K_a_va": params_tuple[69],
            "K_a_bu": params_tuple[70],
            "K_a_pro": params_tuple[71],
            "K_a_ac": params_tuple[72],
            "K_a_co2": params_tuple[73],
            "K_a_IN": params_tuple[74],
            "k_A_B_va": params_tuple[75],
            "k_A_B_bu": params_tuple[76],
            "k_A_B_pro": params_tuple[77],
            "k_A_B_ac": params_tuple[78],
            "k_A_B_co2": params_tuple[79],
            "k_A_B_IN": params_tuple[80],
        }

        gas_params = {
            "k_L_a": params_tuple[82],
            "K_H_co2": params_tuple[83],
            "K_H_ch4": params_tuple[84],
            "K_H_h2": params_tuple[85],
            "k_p": params_tuple[81],
        }

        # Get pH and inhibition parameters
        K_pH_aa, nn_aa, K_pH_ac, n_ac, K_pH_h2, n_h2 = ADMParams.getADMinhibitionparams()

        # Calculate pH and H+ concentration
        pH = ADMstate.calcPHOfADMstate(state_zero)
        S_H_ion = 10 ** (-pH)

        # Unpack state variables
        S_su, S_aa, S_fa, S_va, S_bu, S_pro, S_ac, S_h2 = state_zero[0:8]
        S_ch4, S_co2, S_nh4_ion, S_I = state_zero[8:12]
        X_xc, X_ch, X_pr, X_li = state_zero[12:16]
        X_su, X_aa, X_fa, X_c4, X_pro, X_ac, X_h2, X_I, X_p = state_zero[16:25]
        S_cation, S_anion = state_zero[25:27]
        S_va_ion, S_bu_ion, S_pro_ion, S_ac_ion = state_zero[27:31]
        S_hco3_ion, S_nh3 = state_zero[31:33]
        p_gas_h2, p_gas_ch4, p_gas_co2, pTOTAL = state_zero[33:37]

        # Get influent concentrations
        q_ad = np.sum(self._Q) if self._Q is not None else 0.0
        state_in = self._state_input if self._state_input is not None else [0.0] * 34

        # Calculate inhibition factors
        bio = BiochemicalProcesses()
        inhibitions = bio.calculate_inhibition_factors(
            S_H_ion,
            S_h2,
            S_nh4_ion,
            S_nh3,
            K_pH_aa,
            nn_aa,
            K_pH_ac,
            n_ac,
            K_pH_h2,
            n_h2,
            kinetic_params["K_S_IN"],
            kinetic_params["K_I_h2_fa"],
            kinetic_params["K_I_h2_c4"],
            kinetic_params["K_I_h2_pro"],
            kinetic_params["K_I_nh3"],
        )

        # Calculate biochemical process rates
        process_rates = bio.calculate_process_rates(state_zero, inhibitions, kinetic_params, substrate_params)
        Rho_1, Rho_2, Rho_3, Rho_4, Rho_5, Rho_6, Rho_7, Rho_8, Rho_9 = process_rates[:9]
        Rho_10, Rho_11, Rho_12, Rho_13, Rho_14, Rho_15, Rho_16 = process_rates[9:16]
        Rho_17, Rho_18, Rho_19 = process_rates[16:19]

        # Calculate acid-base rates
        acid_base_rates = bio.calculate_acid_base_rates(state_zero, acid_base_params)
        Rho_A_4, Rho_A_5, Rho_A_6, Rho_A_7, Rho_A_10, Rho_A_11 = acid_base_rates

        # Calculate gas transfer rates
        gas_rates = bio.calculate_gas_transfer_rates(state_zero, gas_params, self._RT, self.V_liq, self._V_gas)
        Rho_T_8, Rho_T_9, Rho_T_10, Rho_T_11 = gas_rates

        # Extract substrate-dependent fractions
        (f_ch_xc, f_pr_xc, f_li_xc, f_xI_xc, f_sI_xc, f_xp_xc, _, _, _, _, _, _, _, _) = substrate_params.values()

        # Calculate biomass decay fractions
        f_p = 0.08
        f_ch_xb = f_ch_xc / (f_ch_xc + f_pr_xc + f_li_xc) * (1 - f_p)
        f_pr_xb = f_pr_xc / (f_ch_xc + f_pr_xc + f_li_xc) * (1 - f_p)
        f_li_xb = f_li_xc / (f_ch_xc + f_pr_xc + f_li_xc) * (1 - f_p)

        # Differential equations for soluble components (1-12)
        diff_S_su = q_ad / self.V_liq * (state_in[0] - S_su) + Rho_2 + (1 - f_fa_li) * Rho_4 - Rho_5

        diff_S_aa = q_ad / self.V_liq * (state_in[1] - S_aa) + Rho_3 - Rho_6

        diff_S_fa = q_ad / self.V_liq * (state_in[2] - S_fa) + f_fa_li * Rho_4 - Rho_7

        diff_S_va = q_ad / self.V_liq * (state_in[3] - S_va) + (1 - Y_aa) * f_va_aa * Rho_6 - Rho_8

        diff_S_bu = (
            q_ad / self.V_liq * (state_in[4] - S_bu) + (1 - Y_su) * f_bu_su * Rho_5 + (1 - Y_aa) * f_bu_aa * Rho_6 - Rho_9
        )

        diff_S_pro = (
            q_ad / self.V_liq * (state_in[5] - S_pro)
            + (1 - Y_su) * f_pro_su * Rho_5
            + (1 - Y_aa) * f_pro_aa * Rho_6
            + (1 - Y_c4) * 0.54 * Rho_8
            - Rho_10
        )

        diff_S_ac = (
            q_ad / self.V_liq * (state_in[6] - S_ac)
            + (1 - Y_su) * f_ac_su * Rho_5
            + (1 - Y_aa) * f_ac_aa * Rho_6
            + (1 - Y_fa) * 0.7 * Rho_7
            + (1 - Y_c4) * 0.31 * Rho_8
            + (1 - Y_c4) * 0.8 * Rho_9
            + (1 - Y_pro) * 0.57 * Rho_10
            - Rho_11
        )

        diff_S_h2 = (
            q_ad / self.V_liq * (state_in[7] - S_h2)
            + (1 - Y_su) * f_h2_su * Rho_5
            + (1 - Y_aa) * f_h2_aa * Rho_6
            + (1 - Y_fa) * 0.3 * Rho_7
            + (1 - Y_c4) * 0.15 * Rho_8
            + (1 - Y_c4) * 0.2 * Rho_9
            + (1 - Y_pro) * 0.43 * Rho_10
            - Rho_12
            - self._V_gas / self.V_liq * Rho_T_8
        )

        diff_S_ch4 = (
            q_ad / self.V_liq * (state_in[8] - S_ch4)
            + (1 - Y_ac) * Rho_11
            + (1 - Y_h2) * Rho_12
            - self._V_gas / self.V_liq * Rho_T_9
        )

        # CO2 balance with carbon stoichiometry
        s_1 = -C_xc + f_sI_xc * C_sI + f_ch_xc * C_ch + f_pr_xc * C_pr + f_li_xc * C_li + f_xI_xc * C_xI
        s_2 = -C_ch + C_su
        s_3 = -C_pr + C_aa
        s_4 = -C_li + (1 - f_fa_li) * C_su + f_fa_li * C_fa
        s_5 = -C_su + (1 - Y_su) * (f_bu_su * C_bu + f_pro_su * C_pro + f_ac_su * C_ac) + Y_su * C_bac
        s_6 = -C_aa + (1 - Y_aa) * (f_va_aa * C_va + f_bu_aa * C_bu + f_pro_aa * C_pro + f_ac_aa * C_ac) + Y_aa * C_bac
        s_7 = -C_fa + (1 - Y_fa) * 0.7 * C_ac + Y_fa * C_bac
        s_8 = -C_va + (1 - Y_c4) * 0.54 * C_pro + (1 - Y_c4) * 0.31 * C_ac + Y_c4 * C_bac
        s_9 = -C_bu + (1 - Y_c4) * 0.8 * C_ac + Y_c4 * C_bac
        s_10 = -C_pro + (1 - Y_pro) * 0.57 * C_ac + Y_pro * C_bac
        s_11 = -C_ac + (1 - Y_ac) * C_ch4 + Y_ac * C_bac
        s_12 = (1 - Y_h2) * C_ch4 + Y_h2 * C_bac
        s_13 = -C_bac + C_xc

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
        )

        diff_S_co2 = q_ad / self.V_liq * (state_in[9] - S_co2) - Sigma - self._V_gas / self.V_liq * Rho_T_10 + Rho_A_10

        # Nitrogen balance
        diff_S_nh4_ion = (
            q_ad / self.V_liq * (state_in[10] - S_nh4_ion)
            - Y_su * N_bac * Rho_5
            + (N_aa - Y_aa * N_bac) * Rho_6
            - Y_fa * N_bac * Rho_7
            - Y_c4 * N_bac * Rho_8
            - Y_c4 * N_bac * Rho_9
            - Y_pro * N_bac * Rho_10
            - Y_ac * N_bac * Rho_11
            - Y_h2 * N_bac * Rho_12
            + (N_bac - N_xc) * (Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19)
            + Rho_A_11
        )

        diff_S_I = q_ad / self.V_liq * (state_in[11] - S_I) + f_sI_xc * Rho_1

        # Differential equations for particulate components (13-24)
        diff_X_xc = q_ad / self.V_liq * (state_in[12] - X_xc) - Rho_1

        diff_X_ch = (
            q_ad / self.V_liq * (state_in[13] - X_ch)
            + f_ch_xc * Rho_1
            - Rho_2
            + f_ch_xb * (Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19)
        )

        diff_X_pr = (
            q_ad / self.V_liq * (state_in[14] - X_pr)
            + f_pr_xc * Rho_1
            - Rho_3
            + f_pr_xb * (Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19)
        )

        diff_X_li = (
            q_ad / self.V_liq * (state_in[15] - X_li)
            + f_li_xc * Rho_1
            - Rho_4
            + f_li_xb * (Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19)
        )

        diff_X_su = q_ad / self.V_liq * (state_in[16] - X_su) + Y_su * Rho_5 - Rho_13
        diff_X_aa = q_ad / self.V_liq * (state_in[17] - X_aa) + Y_aa * Rho_6 - Rho_14
        diff_X_fa = q_ad / self.V_liq * (state_in[18] - X_fa) + Y_fa * Rho_7 - Rho_15
        diff_X_c4 = q_ad / self.V_liq * (state_in[19] - X_c4) + Y_c4 * Rho_8 + Y_c4 * Rho_9 - Rho_16
        diff_X_pro = q_ad / self.V_liq * (state_in[20] - X_pro) + Y_pro * Rho_10 - Rho_17
        diff_X_ac = q_ad / self.V_liq * (state_in[21] - X_ac) + Y_ac * Rho_11 - Rho_18
        diff_X_h2 = q_ad / self.V_liq * (state_in[22] - X_h2) + Y_h2 * Rho_12 - Rho_19
        diff_X_I = q_ad / self.V_liq * (state_in[23] - X_I) + f_xI_xc * Rho_1

        diff_X_p = (
            q_ad / self.V_liq * (state_in[24] - X_p)
            - f_xp_xc * Rho_1
            + f_p * (Rho_13 + Rho_14 + Rho_15 + Rho_16 + Rho_17 + Rho_18 + Rho_19)
        )

        # Differential equations for ions (25-32)
        diff_S_cation = q_ad / self.V_liq * (state_in[25] - S_cation)
        diff_S_anion = q_ad / self.V_liq * (state_in[26] - S_anion)
        diff_S_va_ion = q_ad / self.V_liq * (state_in[27] - S_va_ion) - Rho_A_4
        diff_S_bu_ion = q_ad / self.V_liq * (state_in[28] - S_bu_ion) - Rho_A_5
        diff_S_pro_ion = q_ad / self.V_liq * (state_in[29] - S_pro_ion) - Rho_A_6
        diff_S_ac_ion = q_ad / self.V_liq * (state_in[30] - S_ac_ion) - Rho_A_7
        diff_S_hco3_ion = q_ad / self.V_liq * (state_in[31] - S_hco3_ion) - Rho_A_10
        diff_S_nh3 = q_ad / self.V_liq * (state_in[32] - S_nh3) - Rho_A_11

        # Differential equations for gas phase (33-36)
        diff_p_gas_h2 = Rho_T_8 * self._RT / 16 - p_gas_h2 / pTOTAL * Rho_T_11
        diff_p_gas_ch4 = Rho_T_9 * self._RT / 64 - p_gas_ch4 / pTOTAL * Rho_T_11
        diff_p_gas_co2 = Rho_T_10 * self._RT - p_gas_co2 / pTOTAL * Rho_T_11
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

    def _set_influent(self, influent_state: pd.DataFrame, i: int) -> None:
        """
        Set influent values from dataframe.

        Internal method to extract influent state at time index i and store
        it for use in ODE calculations.

        Args:
            influent_state: DataFrame with ADM1 input variables over time
            i: Time step index (uses last row if i exceeds available data)
        """
        # Handle index out of bounds by using last row (steady-state assumption)
        max_index = len(influent_state) - 1
        if i > max_index:
            i = max_index

        # Extract all state variables at time index i
        self._state_input = [
            influent_state["S_su"].iloc[i],  # kg COD.m^-3
            influent_state["S_aa"].iloc[i],  # kg COD.m^-3
            influent_state["S_fa"].iloc[i],  # kg COD.m^-3
            influent_state["S_va"].iloc[i],  # kg COD.m^-3
            influent_state["S_bu"].iloc[i],  # kg COD.m^-3
            influent_state["S_pro"].iloc[i],  # kg COD.m^-3
            influent_state["S_ac"].iloc[i],  # kg COD.m^-3
            influent_state["S_h2"].iloc[i],  # kg COD.m^-3
            influent_state["S_ch4"].iloc[i],  # kg COD.m^-3
            influent_state["S_co2"].iloc[i],  # kmole C.m^-3 (S_IC_in)
            influent_state["S_nh4"].iloc[i],  # kmole N.m^-3 (S_IN_in)
            influent_state["S_I"].iloc[i],  # kg COD.m^-3
            influent_state["X_xc"].iloc[i],  # kg COD.m^-3
            influent_state["X_ch"].iloc[i],  # kg COD.m^-3
            influent_state["X_pr"].iloc[i],  # kg COD.m^-3
            influent_state["X_li"].iloc[i],  # kg COD.m^-3
            influent_state["X_su"].iloc[i],  # kg COD.m^-3
            influent_state["X_aa"].iloc[i],  # kg COD.m^-3
            influent_state["X_fa"].iloc[i],  # kg COD.m^-3
            influent_state["X_c4"].iloc[i],  # kg COD.m^-3
            influent_state["X_pro"].iloc[i],  # kg COD.m^-3
            influent_state["X_ac"].iloc[i],  # kg COD.m^-3
            influent_state["X_h2"].iloc[i],  # kg COD.m^-3
            influent_state["X_I"].iloc[i],  # kg COD.m^-3
            influent_state["X_p"].iloc[i],  # kg COD.m^-3
            influent_state["S_cation"].iloc[i],  # kmole.m^-3
            influent_state["S_anion"].iloc[i],  # kmole.m^-3
            influent_state["S_va_ion"].iloc[i],  # kg COD.m^-3
            influent_state["S_bu_ion"].iloc[i],  # kg COD.m^-3
            influent_state["S_pro_ion"].iloc[i],  # kg COD.m^-3
            influent_state["S_ac_ion"].iloc[i],  # kg COD.m^-3
            influent_state["S_hco3_ion"].iloc[i],  # kg COD.m^-3
            influent_state["S_nh3"].iloc[i],  # kg COD.m^-3
            influent_state["Q"].iloc[i],  # m^3/d (q_ad)
        ]

    def _get_substrate_dependent_params(self) -> dict:
        """
        Get substrate-dependent parameters from C# DLL. The parameters
        used in these calculations are defined in substrate_...xml. Documentation in PhD thesis of D. Gaida, 2014

        Calculates ADM1 parameters that depend on substrate composition
        using weighted averaging based on volumetric flow rates.

        Returns:
            Dictionary with substrate-dependent parameters:
                - f_ch_xc, f_pr_xc, f_li_xc: Composite fractions
                - f_xI_xc, f_sI_xc, f_xp_xc: Inert and product fractions
                - k_dis: Disintegration rate [1/d]
                - k_hyd_ch, k_hyd_pr, k_hyd_li: Hydrolysis rates [1/d]
                - k_m_c4, k_m_pro, k_m_ac, k_m_h2: Uptake rates [1/d]
        """
        if self._Q is None:
            # Return default values if Q not set
            return {
                "f_ch_xc": 0.2,
                "f_pr_xc": 0.2,
                "f_li_xc": 0.3,
                "f_xI_xc": 0.2,
                "f_sI_xc": 0.1,
                "f_xp_xc": 0.0,
                "k_dis": 0.5,
                "k_hyd_ch": 10.0,
                "k_hyd_pr": 10.0,
                "k_hyd_li": 10.0,
                "k_m_c4": 20.0,
                "k_m_pro": 13.0,
                "k_m_ac": 8.0,
                "k_m_h2": 35.0,
            }

        # Calculate weighted substrate parameters
        f_ch_xc, f_pr_xc, f_li_xc, f_xI_xc, f_sI_xc, f_xp_xc = self._feedstock.mySubstrates().calcfFactors(self._Q)
        f_xp_xc = max(f_xp_xc, 0.0)

        k_dis = self._feedstock.mySubstrates().calcDisintegrationParam(self._Q)
        k_hyd_ch, k_hyd_pr, k_hyd_li = self._feedstock.mySubstrates().calcHydrolysisParams(self._Q)
        k_m_c4, k_m_pro, k_m_ac, k_m_h2 = self._feedstock.mySubstrates().calcMaxUptakeRateParams(self._Q)

        return {
            "f_ch_xc": f_ch_xc,
            "f_pr_xc": f_pr_xc,
            "f_li_xc": f_li_xc,
            "f_xI_xc": f_xI_xc,
            "f_sI_xc": f_sI_xc,
            "f_xp_xc": f_xp_xc,
            "k_dis": k_dis,
            "k_hyd_ch": k_hyd_ch,
            "k_hyd_pr": k_hyd_pr,
            "k_hyd_li": k_hyd_li,
            "k_m_c4": k_m_c4,
            "k_m_pro": k_m_pro,
            "k_m_ac": k_m_ac,
            "k_m_h2": k_m_h2,
        }

    # Properties for accessing model parameters and results
    @property
    def T_ad(self) -> float:
        """Operating temperature [K]."""
        return self._T_ad

    @property
    def feedstock(self) -> Feedstock:
        """Feedstock object."""
        return self._feedstock

    @property
    def Q_GAS(self) -> List[float]:
        """Biogas production rates over all simulations [m³/d]."""
        return self._Q_GAS

    @property
    def Q_CH4(self) -> List[float]:
        """Methane production rates over all simulations [m³/d]."""
        return self._Q_CH4

    @property
    def Q_CO2(self) -> List[float]:
        """CO2 production rates over all simulations [m³/d]."""
        return self._Q_CO2

    @property
    def P_GAS(self) -> List[float]:
        """Gas pressures over all simulations [bar]."""
        return self._P_GAS

    @property
    def pH_l(self) -> List[float]:
        """pH values over all simulations."""
        return self._pH_l

    @property
    def VFA_TA(self) -> List[float]:
        """VFA/TA ratios over all simulations."""
        return self._FOSTAC

    @property
    def AcvsPro(self) -> List[float]:
        """Acetic/Propionic acid ratios over all simulations."""
        return self._AcvsPro

    @property
    def VFA(self) -> List[float]:
        """VFA concentrations over all simulations [g/L]."""
        return self._VFA

    @property
    def TAC(self) -> List[float]:
        """TA concentrations over all simulations [g CaCO3 eq/L]."""
        return self._TAC
