# pyadm1/core/adm_base.py
"""
Abstract base class for Anaerobic Digestion Model implementations.

Defines the shared interface and common infrastructure used by all ADM
variants (ADM1, ADM1da, ...). Each variant must implement the abstract
methods while inheriting the shared physical setup and gas calculation logic.

Subclasses:
    ADM1   -- standard 37-state implementation (Batstone et al. 2002)
    ADM1da -- extended implementation with sub-fractions and temperature
              dependency (SIMBA#biogas ADM1da, Schlattmann 2011)
"""

import logging
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from pyadm1.core.adm_params import ADMParams
from pyadm1.substrates.feedstock import Feedstock

logger = logging.getLogger(__name__)


class ADMBase(ABC):
    """
    Abstract base for ADM model variants.

    Provides shared physical setup (reactor volumes, temperature, gas
    constants), result-tracking lists, and the gas production calculation
    that is identical across all variants. Subclasses implement the ODE
    system and influent creation for their specific state vector.

    Attributes:
        V_liq: Liquid volume [m³]
        T_ad:  Operating temperature [K]
    """

    def __init__(
        self,
        feedstock: Feedstock,
        V_liq: float = 1977.0,
        V_gas: float = 304.0,
        T_ad: float = 308.15,
    ) -> None:
        """
        Initialize shared ADM infrastructure.

        Args:
            feedstock: Feedstock object providing substrate compositions and
                       influent flow calculations.
            V_liq:     Liquid volume of the digester [m³].
            V_gas:     Gas headspace volume [m³].
            T_ad:      Operating temperature [K] (default 308.15 K = 35 °C).
        """
        # --- Reactor volumes ---
        self.V_liq = V_liq
        self._V_gas = V_gas
        self._V_ad = V_liq + V_gas

        # --- Temperature ---
        self._T_ad = T_ad

        # --- Physical constants ---
        self._R = 0.08314  # Gas constant [bar·m³·kmol⁻¹·K⁻¹]
        self._T_base = 298.15  # Reference temperature [K] (25 °C)
        self._p_atm = 1.013  # Atmospheric pressure [bar]

        # Pre-computed temperature-dependent scalars
        self._RT = self._R * self._T_ad
        # External pressure: atm minus water-vapour partial pressure
        self._p_ext = self._p_atm - 0.0084147 * np.exp(0.054 * (self._T_ad - 273.15))

        # --- Feedstock / influent ---
        self._feedstock = feedstock
        self._Q: Optional[List[float]] = None  # Substrate flow rates [m³/d]
        self._state_input: Optional[List[float]] = None  # ADM influent vector

        # --- Optional calibration overrides ---
        # Subclasses and external code may populate this dict to override
        # specific parameter values without subclassing further.
        # Example: adm._calibration_params = {"k_p": 5e4}
        self._calibration_params: dict = {}

        # --- Result-tracking lists (appended by print_params_at_current_state) ---
        self._Q_GAS: List[float] = []  # Total biogas flow rate   [m³/d]
        self._Q_CH4: List[float] = []  # Methane flow rate        [m³/d]
        self._Q_CO2: List[float] = []  # CO₂ flow rate            [m³/d]
        self._Q_H2O: List[float] = []  # Water vapour flow rate   [m³/d]
        self._P_GAS: List[float] = []  # Total gas pressure       [bar]
        self._pH_l: List[float] = []  # pH                       [-]
        self._FOSTAC: List[float] = []  # VFA/TAC ratio            [-]
        self._AcvsPro: List[float] = []  # Acetic/propionic ratio   [-]
        self._VFA: List[float] = []  # VFA concentration        [g/L]
        self._TAC: List[float] = []  # Total alkalinity         [g CaCO₃/L]

    # ------------------------------------------------------------------
    # Abstract interface -- every subclass must implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def ADM_ODE(self, t: float, state: List[float]) -> Tuple[float, ...]:
        """
        Compute dy/dt for the model's complete state vector.

        This is the function passed directly to the ODE solver.

        Args:
            t:     Current time [days].  The system is autonomous, so t is
                   only present to satisfy the scipy interface.
            state: Current state vector.  Length must equal get_state_size().

        Returns:
            Tuple of derivatives, one per state variable.
        """

    @abstractmethod
    def create_influent(self, Q: List[float], i: int) -> None:
        """
        Build the internal influent vector from substrate flow rates.

        Called once per simulation step before the ODE is integrated.
        Subclasses populate self._state_input and self._Q.

        Args:
            Q: Volumetric flow rates for each substrate defined in the
               feedstock XML [m³/d].  Length must match feedstock.
            i: Time-step index used to select the appropriate row from
               the influent dataframe.
        """

    @abstractmethod
    def get_state_size(self) -> int:
        """
        Return the number of state variables in this model variant.

        Examples:
            ADM1   → 37
            ADM1da → depends on sub-fraction count (typically ~50)
        """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Short identifier for this model variant.

        Examples:  "ADM1", "ADM1da"
        Used for logging, CSV headers, and component configuration.
        """

    # ------------------------------------------------------------------
    # Shared concrete methods
    # ------------------------------------------------------------------

    def calc_gas(
        self,
        pi_Sh2: float,
        pi_Sch4: float,
        pi_Sco2: float,
        pTOTAL: float,
    ) -> Tuple[float, float, float, float, float]:
        """
        Calculate biogas production rates from gas-phase partial pressures.

        The gas model (Henry constants, outlet friction coefficient k_p) is
        identical for all ADM variants.  The result depends only on the
        current gas-phase state, which all variants carry at the tail of
        their state vector.

        Args:
            pi_Sh2:  Hydrogen partial pressure  [bar]
            pi_Sch4: Methane partial pressure   [bar]
            pi_Sco2: CO₂ partial pressure       [bar]
            pTOTAL:  Total gas pressure         [bar]

        Returns:
            q_gas:  Total biogas flow rate                    [m³/d]
            q_ch4:  Methane flow rate                         [m³/d]
            q_co2:  CO₂ flow rate                             [m³/d]
            q_h2o:  Water vapour flow rate                    [m³/d]  (0 in base)
            p_gas:  Partial pressure sum (H₂ + CH₄ + CO₂)   [bar]
        """
        _, k_p, _, _, _, _ = ADMParams.getADMgasparams(self._R, self._T_base, self._T_ad)

        if self._calibration_params.get("k_p") is not None:
            k_p = float(self._calibration_params["k_p"])

        # Ideal-gas volumetric conversion factor [Nm³/kmol at 0 °C, 1 bar]
        NQ = 44.643

        q_gas = k_p * (pTOTAL - self._p_ext) / (self._RT / 1000 * NQ) * self.V_liq
        q_gas = np.maximum(q_gas, 0.0)

        p_gas = pi_Sh2 + pi_Sch4 + pi_Sco2

        if p_gas > 0:
            q_ch4 = np.maximum(q_gas * (pi_Sch4 / p_gas), 0.0)
            q_co2 = np.maximum(q_gas * (pi_Sco2 / p_gas), 0.0)
        else:
            q_ch4 = 0.0
            q_co2 = 0.0

        return q_gas, q_ch4, q_co2, 0.0, p_gas

    def resume_from_broken_simulation(self, Q_CH4: List[float]) -> None:
        """
        Re-populate the methane tracking list after a simulation restart.

        Call this when loading a partially completed simulation so that
        the tracking list is consistent with the stored results.

        Args:
            Q_CH4: Methane flow rates from the completed portion [m³/d].
        """
        for q in Q_CH4:
            self._Q_CH4.append(q)

    # ------------------------------------------------------------------
    # Calibration parameter API
    # ------------------------------------------------------------------

    def set_calibration_parameters(self, parameters: dict) -> None:
        """
        Set calibration parameters that override substrate-dependent calculations.

        Args:
            parameters: Parameter values as {param_name: value}.

        Example:
            >>> adm.set_calibration_parameters({
            ...     'k_dis': 0.55,
            ...     'k_hyd_ch': 11.0,
            ...     'Y_su': 0.105
            ... })
        """
        self._calibration_params.update(parameters)

    def clear_calibration_parameters(self) -> None:
        """Clear all calibration parameters and revert to substrate-dependent calculations."""
        self._calibration_params = {}

    def get_calibration_parameters(self) -> dict:
        """
        Get currently set calibration parameters.

        Returns:
            dict: Current calibration parameters as {param_name: value}.
        """
        return self._calibration_params.copy()

    # ------------------------------------------------------------------
    # Result-tracking helpers (used by print_params_at_current_state)
    # ------------------------------------------------------------------

    def _track_pH(self, pH: float) -> None:
        """
        Append *pH* to the tracking list, bootstrapping a second value on first call.

        The bootstrap mirrors the "assume we start from steady state" contract:
        downstream controllers read the last few entries of the list, so we need
        at least two values after the very first sample.
        """
        self._pH_l.append(pH)
        if len(self._pH_l) < 2:
            self._pH_l.append(self._pH_l[-1])

    def _track_gas(
        self,
        q_gas: float,
        q_ch4: float,
        q_co2: float,
        q_h2o: float,
        p_gas: float,
    ) -> None:
        """
        Append gas-production values to their tracking lists.

        On the very first call the values are appended three times so that the
        lists contain at least four entries (the controller reads the last
        three). This assumes the simulation starts from a steady state.
        """
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
    # Shared read-only properties
    # ------------------------------------------------------------------

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
    def Q_H2O(self) -> List[float]:
        """Water vapour flow rates over all simulations [m³/d]."""
        return self._Q_H2O

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

    # ------------------------------------------------------------------
    # Internal helpers available to subclasses
    # ------------------------------------------------------------------

    def _set_influent(self, influent_state, i: int) -> None:
        """
        Store the influent vector for a given time step.

        Shared implementation: subclasses call this from create_influent()
        after computing the mixed influent from feedstock data.

        Args:
            influent_state: DataFrame returned by feedstock.get_influent_dataframe().
            i:              Row index (time step).
        """
        row = influent_state.iloc[i]
        self._state_input = row.tolist()

    def _effective_param(self, params: dict, key: str) -> float:
        """
        Return calibration override for *key* if present, else params[key].

        Convenience helper for use inside ADM_ODE implementations.

        Args:
            params: Dictionary of default parameter values.
            key:    Parameter name to look up.

        Returns:
            Calibrated value if available, otherwise the default value.
        """
        if key in self._calibration_params:
            return float(self._calibration_params[key])
        return params[key]
