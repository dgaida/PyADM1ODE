# pyadm1/core/adm_equations.py
"""
Process rate equations, inhibition functions, and biochemical transformations
for the ADM1 model.

This module separates the biochemical equations from the main ADM1 class for
clarity and easier modification. It includes:
- Inhibition functions (pH, hydrogen, ammonia)
- Biochemical process rates (disintegration, hydrolysis, uptake, decay)
- Acid-base kinetics
- Gas transfer rates
"""

import numpy as np
from typing import Tuple, List


class InhibitionFunctions:
    """pH and substrate inhibition functions for ADM1 processes."""

    @staticmethod
    def pH_inhibition(S_H_ion: float, K_pH: float, n: float) -> float:
        """
        Calculate pH inhibition factor.

        Args:
            S_H_ion: Hydrogen ion concentration [M]
            K_pH: pH inhibition constant [M]
            n: Hill coefficient for pH inhibition [-]

        Returns:
            Inhibition factor between 0 and 1
        """
        return (K_pH**n) / (S_H_ion**n + K_pH**n)

    @staticmethod
    def substrate_inhibition(S: float, K_S: float) -> float:
        """
        Calculate Monod substrate limitation factor.

        Args:
            S: Substrate concentration [kg COD/m³]
            K_S: Half-saturation constant [kg COD/m³]

        Returns:
            Limitation factor between 0 and 1
        """
        return S / (K_S + S)

    @staticmethod
    def hydrogen_inhibition(S_h2: float, K_I_h2: float) -> float:
        """
        Calculate non-competitive hydrogen inhibition.

        Args:
            S_h2: Hydrogen concentration [kg COD/m³]
            K_I_h2: Hydrogen inhibition constant [kg COD/m³]

        Returns:
            Inhibition factor between 0 and 1
        """
        return 1.0 / (1.0 + (S_h2 / K_I_h2))

    @staticmethod
    def ammonia_inhibition(S_nh3: float, K_I_nh3: float) -> float:
        """
        Calculate ammonia inhibition.

        Args:
            S_nh3: Free ammonia concentration [M]
            K_I_nh3: Ammonia inhibition constant [M]

        Returns:
            Inhibition factor between 0 and 1
        """
        return 1.0 / (1.0 + (S_nh3 / K_I_nh3))

    @staticmethod
    def nitrogen_limitation(S_nh4_ion: float, S_nh3: float, K_S_IN: float) -> float:
        """
        Calculate inorganic nitrogen limitation factor.

        Args:
            S_nh4_ion: Ammonium ion concentration [M]
            S_nh3: Free ammonia concentration [M]
            K_S_IN: Nitrogen half-saturation constant [M]

        Returns:
            Limitation factor between 0 and 1
        """
        S_IN_total = S_nh4_ion + S_nh3
        return 1.0 / (1.0 + (K_S_IN / S_IN_total))


class ProcessRates:
    """Biochemical process rate calculations for ADM1."""

    @staticmethod
    def disintegration_rate(k_dis: float, X_xc: float) -> float:
        """
        Calculate disintegration rate of composites.

        Args:
            k_dis: Disintegration rate constant [1/d]
            X_xc: Composite concentration [kg COD/m³]

        Returns:
            Disintegration rate [kg COD/(m³·d)]
        """
        return k_dis * X_xc

    @staticmethod
    def hydrolysis_rate(k_hyd: float, X_substrate: float, hydro_factor: float = 1.0) -> float:
        """
        Calculate hydrolysis rate (carbohydrates, proteins, lipids).

        Args:
            k_hyd: Hydrolysis rate constant [1/d]
            X_substrate: Particulate substrate concentration [kg COD/m³]
            hydro_factor: Optional TS-dependent factor [-]

        Returns:
            Hydrolysis rate [kg COD/(m³·d)]
        """
        return k_hyd * X_substrate * hydro_factor

    @staticmethod
    def uptake_rate(k_m: float, S_substrate: float, K_S: float, X_biomass: float, I_combined: float) -> float:
        """
        Calculate Monod uptake rate with inhibition.

        Args:
            k_m: Maximum uptake rate constant [1/d]
            S_substrate: Substrate concentration [kg COD/m³]
            K_S: Half-saturation constant [kg COD/m³]
            X_biomass: Biomass concentration [kg COD/m³]
            I_combined: Combined inhibition factor [-]

        Returns:
            Uptake rate [kg COD/(m³·d)]
        """
        return k_m * (S_substrate / (K_S + S_substrate)) * X_biomass * I_combined

    @staticmethod
    def decay_rate(k_dec: float, X_biomass: float) -> float:
        """
        Calculate biomass decay rate.

        Args:
            k_dec: Decay rate constant [1/d]
            X_biomass: Biomass concentration [kg COD/m³]

        Returns:
            Decay rate [kg COD/(m³·d)]
        """
        return k_dec * X_biomass


class AcidBaseKinetics:
    """Acid-base equilibrium kinetics for ADM1."""

    @staticmethod
    def acid_base_rate(k_AB: float, S_ion: float, S_H_ion: float, K_a: float, S_undissociated: float) -> float:
        """
        Calculate acid-base reaction rate.

        Implements: S_ion + H+ <-> S_undissociated

        Args:
            k_AB: Acid-base kinetic constant [M^-1·d^-1]
            S_ion: Ionized form concentration [M or kg COD/m³]
            S_H_ion: Hydrogen ion concentration [M]
            K_a: Acid dissociation constant [M]
            S_undissociated: Undissociated form concentration [M or kg COD/m³]

        Returns:
            Acid-base reaction rate [M/d or kg COD/(m³·d)]
        """
        forward = S_ion * S_H_ion
        backward = K_a * S_undissociated
        return k_AB * (forward - backward)


class GasTransfer:
    """Gas-liquid transfer and gas outlet calculations."""

    @staticmethod
    def gas_transfer_rate(
        k_L_a: float, S_gas_liq: float, p_gas: float, K_H: float, RT: float, COD_per_mole: float, V_liq: float, V_gas: float
    ) -> float:
        """
        Calculate gas-liquid transfer rate.

        Args:
            k_L_a: Gas-liquid transfer coefficient [1/d]
            S_gas_liq: Gas concentration in liquid [kg COD/m³ or kmol/m³]
            p_gas: Partial pressure in gas phase [bar]
            K_H: Henry's law constant [M/bar]
            RT: Gas constant × temperature [bar·m³/kmol]
            COD_per_mole: COD per mole of gas [kg COD/kmol]
            V_liq: Liquid volume [m³]
            V_gas: Gas volume [m³]

        Returns:
            Gas transfer rate to gas phase [kg COD/(m³_gas·d) or kmol/(m³_gas·d)]
        """
        S_gas_equilibrium = p_gas * COD_per_mole / (RT * K_H)
        return k_L_a * (S_gas_liq - S_gas_equilibrium) * (V_liq / V_gas)

    @staticmethod
    def gas_outlet_rate(k_p: float, p_total: float, p_ext: float, V_liq: float, V_gas: float) -> float:
        """
        Calculate gas outlet flow rate.

        Args:
            k_p: Gas outlet friction coefficient [m³/(m³·d·bar)]
            p_total: Total gas pressure [bar]
            p_ext: External pressure [bar]
            V_liq: Liquid volume [m³]
            V_gas: Gas volume [m³]

        Returns:
            Gas outlet rate [1/d]
        """
        return k_p * (p_total - p_ext) * (V_liq / V_gas)


class BiochemicalProcesses:
    """
    Combined biochemical process calculations for ADM1.

    This class orchestrates the calculation of all process rates including
    inhibition factors and stoichiometric relationships.
    """

    @staticmethod
    def calculate_inhibition_factors(
        S_H_ion: float,
        S_h2: float,
        S_nh4_ion: float,
        S_nh3: float,
        K_pH_aa: float,
        nn_aa: float,
        K_pH_ac: float,
        n_ac: float,
        K_pH_h2: float,
        n_h2: float,
        K_S_IN: float,
        K_I_h2_fa: float,
        K_I_h2_c4: float,
        K_I_h2_pro: float,
        K_I_nh3: float,
    ) -> Tuple[float, float, float, float, float, float, float, float, float]:
        """
        Calculate all inhibition factors for ADM1 processes.

        Args:
            S_H_ion: Hydrogen ion concentration [M]
            S_h2: Hydrogen gas concentration [kg COD/m³]
            S_nh4_ion: Ammonium concentration [M]
            S_nh3: Free ammonia concentration [M]
            K_pH_aa: pH inhibition constant for amino acid degraders [M]
            nn_aa: Hill coefficient for aa pH inhibition [-]
            K_pH_ac: pH inhibition constant for acetate degraders [M]
            n_ac: Hill coefficient for ac pH inhibition [-]
            K_pH_h2: pH inhibition constant for hydrogen degraders [M]
            n_h2: Hill coefficient for h2 pH inhibition [-]
            K_S_IN: Nitrogen half-saturation constant [M]
            K_I_h2_fa: H2 inhibition constant for LCFA degraders [kg COD/m³]
            K_I_h2_c4: H2 inhibition constant for C4 degraders [kg COD/m³]
            K_I_h2_pro: H2 inhibition constant for propionate degraders [kg COD/m³]
            K_I_nh3: Ammonia inhibition constant [M]

        Returns:
            Tuple of inhibition factors (I_pH_aa, I_pH_ac, I_pH_h2, I_IN_lim,
            I_h2_fa, I_h2_c4, I_h2_pro, I_nh3, I_5 through I_12)
        """
        inh = InhibitionFunctions()

        # pH inhibition factors
        I_pH_aa = inh.pH_inhibition(S_H_ion, K_pH_aa, nn_aa)
        I_pH_ac = inh.pH_inhibition(S_H_ion, K_pH_ac, n_ac)
        I_pH_h2 = inh.pH_inhibition(S_H_ion, K_pH_h2, n_h2)

        # Nitrogen limitation
        I_IN_lim = inh.nitrogen_limitation(S_nh4_ion, S_nh3, K_S_IN)

        # Hydrogen inhibition
        I_h2_fa = inh.hydrogen_inhibition(S_h2, K_I_h2_fa)
        I_h2_c4 = inh.hydrogen_inhibition(S_h2, K_I_h2_c4)
        I_h2_pro = inh.hydrogen_inhibition(S_h2, K_I_h2_pro)

        # Ammonia inhibition
        I_nh3 = inh.ammonia_inhibition(S_nh3, K_I_nh3)

        # Combined inhibition factors for different processes
        I_5 = I_pH_aa * I_IN_lim  # Sugar uptake
        I_6 = I_5  # Amino acid uptake
        I_7 = I_pH_aa * I_IN_lim * I_h2_fa  # LCFA uptake
        I_8 = I_pH_aa * I_IN_lim * I_h2_c4  # Valerate uptake
        I_9 = I_8  # Butyrate uptake
        I_10 = I_pH_aa * I_IN_lim * I_h2_pro  # Propionate uptake
        I_11 = I_pH_ac * I_IN_lim * I_nh3  # Acetate uptake
        I_12 = I_pH_h2 * I_IN_lim  # Hydrogen uptake

        return (
            I_pH_aa,
            I_pH_ac,
            I_pH_h2,
            I_IN_lim,
            I_h2_fa,
            I_h2_c4,
            I_h2_pro,
            I_nh3,
            I_5,
            I_6,
            I_7,
            I_8,
            I_9,
            I_10,
            I_11,
            I_12,
        )

    @staticmethod
    def calculate_process_rates(
        state: List[float],
        inhibitions: Tuple[float, ...],
        kinetic_params: dict,
        substrate_params: dict,
        hydro_factor: float = 1.0,
    ) -> Tuple[float, ...]:
        """
        Calculate all 19 biochemical process rates for ADM1.

        Args:
            state: ADM1 state vector (37 elements)
            inhibitions: Tuple of inhibition factors from calculate_inhibition_factors
            kinetic_params: Dictionary of kinetic parameters (k_m, K_S, k_dec, etc.)
            substrate_params: Dictionary of substrate-dependent parameters
                (k_dis, k_hyd_ch, k_hyd_pr, k_hyd_li)
            hydro_factor: Optional TS-dependent hydrolysis factor [-]

        Returns:
            Tuple of 19 process rates (Rho_1 through Rho_19)
        """
        # Unpack state
        S_su, S_aa, S_fa, S_va, S_bu, S_pro, S_ac, S_h2 = state[0:8]
        X_xc, X_ch, X_pr, X_li = state[12:16]
        X_su, X_aa, X_fa, X_c4, X_pro, X_ac, X_h2 = state[16:23]

        # Unpack inhibition factors (using indices 8-15 for combined factors)
        I_5, I_6, I_7, I_8, I_9, I_10, I_11, I_12 = inhibitions[8:16]

        # Unpack parameters
        k_dis = substrate_params["k_dis"]
        k_hyd_ch = substrate_params["k_hyd_ch"]
        k_hyd_pr = substrate_params["k_hyd_pr"]
        k_hyd_li = substrate_params["k_hyd_li"]

        # TODO
        # this extension introduces instabilities into the simulation, so it is outcommented. the TS value calculated in
        # calcTS is also not very accurate
        # Erweiterung der hydrolyse, abhängigkeit von TS gehalt im fermenter, s. diss. von Koch 2010, S. 62
        # TS = digester.calcTS(state_zero, self.feedstock.mySubstrates, self.Q)
        # TS_digester = TS.Value

        # Khyd = 5.5 # 2.5
        # nhyd = 2.3

        # hydro_factor = 1.0 / (1.0 + math.pow(TS_digester/Khyd, nhyd))

        # by setting hydro_factor to 1, we are not using it
        # hydro_factor = 1.0
        # print(hydro_factor)

        proc = ProcessRates()

        # Process rates (Rosen et al. 2006, BSM2)
        Rho_1 = proc.disintegration_rate(k_dis, X_xc)
        Rho_2 = proc.hydrolysis_rate(k_hyd_ch, X_ch, hydro_factor)
        Rho_3 = proc.hydrolysis_rate(k_hyd_pr, X_pr, hydro_factor)
        Rho_4 = proc.hydrolysis_rate(k_hyd_li, X_li, hydro_factor)

        # Uptake rates
        Rho_5 = proc.uptake_rate(kinetic_params["k_m_su"], S_su, kinetic_params["K_S_su"], X_su, I_5)
        Rho_6 = proc.uptake_rate(kinetic_params["k_m_aa"], S_aa, kinetic_params["K_S_aa"], X_aa, I_6)
        Rho_7 = proc.uptake_rate(kinetic_params["k_m_fa"], S_fa, kinetic_params["K_S_fa"], X_fa, I_7)

        # Valerate and butyrate uptake (with competition)
        competition_va = S_va / (S_bu + S_va + 1e-6)
        competition_bu = S_bu / (S_bu + S_va + 1e-6)
        Rho_8 = kinetic_params["k_m_c4"] * (S_va / (kinetic_params["K_S_c4"] + S_va)) * X_c4 * competition_va * I_8
        Rho_9 = kinetic_params["k_m_c4"] * (S_bu / (kinetic_params["K_S_c4"] + S_bu)) * X_c4 * competition_bu * I_9

        Rho_10 = proc.uptake_rate(kinetic_params["k_m_pro"], S_pro, kinetic_params["K_S_pro"], X_pro, I_10)
        Rho_11 = proc.uptake_rate(kinetic_params["k_m_ac"], S_ac, kinetic_params["K_S_ac"], X_ac, I_11)
        Rho_12 = proc.uptake_rate(kinetic_params["k_m_h2"], S_h2, kinetic_params["K_S_h2"], X_h2, I_12)

        # Decay rates
        Rho_13 = proc.decay_rate(kinetic_params["k_dec_X_su"], X_su)
        Rho_14 = proc.decay_rate(kinetic_params["k_dec_X_aa"], X_aa)
        Rho_15 = proc.decay_rate(kinetic_params["k_dec_X_fa"], X_fa)
        Rho_16 = proc.decay_rate(kinetic_params["k_dec_X_c4"], X_c4)
        Rho_17 = proc.decay_rate(kinetic_params["k_dec_X_pro"], X_pro)
        Rho_18 = proc.decay_rate(kinetic_params["k_dec_X_ac"], X_ac)
        Rho_19 = proc.decay_rate(kinetic_params["k_dec_X_h2"], X_h2)

        return (
            Rho_1,
            Rho_2,
            Rho_3,
            Rho_4,
            Rho_5,
            Rho_6,
            Rho_7,
            Rho_8,
            Rho_9,
            Rho_10,
            Rho_11,
            Rho_12,
            Rho_13,
            Rho_14,
            Rho_15,
            Rho_16,
            Rho_17,
            Rho_18,
            Rho_19,
        )

    @staticmethod
    def calculate_acid_base_rates(
        state: List[float], acid_base_params: dict
    ) -> Tuple[float, float, float, float, float, float]:
        """
        Calculate acid-base reaction rates for ODE implementation.

        Args:
            state: ADM1 state vector (37 elements)
            acid_base_params: Dictionary containing K_a and k_AB values

        Returns:
            Tuple of 6 acid-base rates (Rho_A_4 through Rho_A_11)
        """
        # Unpack state
        S_va, S_bu, S_pro, S_ac, S_co2, S_nh4_ion = (state[3], state[4], state[5], state[6], state[9], state[10])
        S_va_ion, S_bu_ion, S_pro_ion, S_ac_ion = state[27:31]
        S_hco3_ion, S_nh3 = state[31], state[32]

        # Get pH to calculate H+ concentration
        # Note: In actual implementation, pH should be calculated from state
        # For now, we'll use a placeholder that should be replaced
        from biogas import ADMstate

        # Convert state to 2D array for DLL methods that expect double[,]
        S_H_ion = 10 ** (-ADMstate.calcPHOfADMstate(np.atleast_2d(state)))

        ab = AcidBaseKinetics()

        # Acid-base rates
        Rho_A_4 = ab.acid_base_rate(
            acid_base_params["k_A_B_va"], S_va_ion, S_H_ion, acid_base_params["K_a_va"], S_va - S_va_ion
        )

        Rho_A_5 = ab.acid_base_rate(
            acid_base_params["k_A_B_bu"], S_bu_ion, S_H_ion, acid_base_params["K_a_bu"], S_bu - S_bu_ion
        )

        Rho_A_6 = ab.acid_base_rate(
            acid_base_params["k_A_B_pro"], S_pro_ion, S_H_ion, acid_base_params["K_a_pro"], S_pro - S_pro_ion
        )

        Rho_A_7 = ab.acid_base_rate(
            acid_base_params["k_A_B_ac"], S_ac_ion, S_H_ion, acid_base_params["K_a_ac"], S_ac - S_ac_ion
        )

        Rho_A_10 = ab.acid_base_rate(acid_base_params["k_A_B_co2"], S_hco3_ion, S_H_ion, acid_base_params["K_a_co2"], S_co2)

        Rho_A_11 = ab.acid_base_rate(acid_base_params["k_A_B_IN"], S_nh3, S_H_ion, acid_base_params["K_a_IN"], S_nh4_ion)

        return (Rho_A_4, Rho_A_5, Rho_A_6, Rho_A_7, Rho_A_10, Rho_A_11)

    @staticmethod
    def calculate_gas_transfer_rates(
        state: List[float], gas_params: dict, RT: float, V_liq: float, V_gas: float
    ) -> Tuple[float, float, float, float]:
        """
        Calculate gas-liquid transfer and gas outlet rates.

        Args:
            state: ADM1 state vector (37 elements)
            gas_params: Dictionary containing k_L_a, K_H constants, k_p
            RT: Gas constant × temperature [bar·m³/kmol]
            V_liq: Liquid volume [m³]
            V_gas: Gas volume [m³]

        Returns:
            Tuple of 4 rates (Rho_T_8, Rho_T_9, Rho_T_10, Rho_T_11)
        """
        # Unpack state
        S_h2, S_ch4, S_co2 = state[7:10]
        p_gas_h2, p_gas_ch4, p_gas_co2, pTOTAL = state[33:37]

        # External pressure (should be passed as parameter in real implementation)
        p_ext = 1.04 - 0.0084147 * np.exp(0.054 * (RT / 0.08314 - 273.15))

        gt = GasTransfer()

        # Gas transfer rates
        Rho_T_8 = gt.gas_transfer_rate(gas_params["k_L_a"], S_h2, p_gas_h2, gas_params["K_H_h2"], RT, 16.0, V_liq, V_gas)

        Rho_T_9 = gt.gas_transfer_rate(gas_params["k_L_a"], S_ch4, p_gas_ch4, gas_params["K_H_ch4"], RT, 64.0, V_liq, V_gas)

        Rho_T_10 = gt.gas_transfer_rate(gas_params["k_L_a"], S_co2, p_gas_co2, gas_params["K_H_co2"], RT, 1.0, V_liq, V_gas)

        # Gas outlet rate
        Rho_T_11 = gt.gas_outlet_rate(gas_params["k_p"], pTOTAL, p_ext, V_liq, V_gas)

        return (Rho_T_8, Rho_T_9, Rho_T_10, Rho_T_11)
