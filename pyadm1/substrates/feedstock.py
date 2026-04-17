# -*- coding: utf-8 -*-
"""
Created on Fri Nov 04 09:56:06 2023

Feedstock class for substrate management and ADM1 input stream calculation.

Substrate parameters are defined in XML files and accessed via C# DLLs.

@author: Daniel Gaida
"""

import os
import numpy as np
import pandas as pd
from typing import List
from pathlib import Path

# DLLs are centrally loaded in pyadm1/__init__.py
from biogas import substrates, ADMstate  # noqa: E402  # type: ignore

"""
this class returns parameters of substrates and also creates the input stream for the ADM1 model. The substrate
parameters are defined in the XML file substrate_...xml. This xml file can be accessed via the DLLs in the dlls
subfolder. Inside the DLLs is also defined how the ADM1 input stream is calculated for a mix of different substrates.
More information on how the substrate mix is calculated can be found in:
Gaida, D., Dynamic real-time substrate feed optimization of anaerobic co-digestion plants, PhD thesis, Leiden, 2014.

It is expected that the file 'substrate_gummersbach.xml' is in the same folder as this *.py file.
"""


class Feedstock:
    """
    Manages substrate information and creates ADM1 input streams.

    Substrate parameters are loaded from XML files and processed via C# DLLs
    to generate ADM1-compatible input streams.
    """

    # *** CONSTRUCTORS ***
    def __init__(
        self,
        feeding_freq: int,
        total_simtime: int = 60,
        substrate_xml: str = "substrate_gummersbach.xml",
    ) -> None:
        """
        Initialize Feedstock with feeding frequency and simulation time.

        Parameters
        ----------
        feeding_freq : int
            Sample time between feeding events [hours]
        total_simtime : int, optional
            Total simulation time [days]. Default is 60.
        substrate_xml : str, optional
            Name of the XML file containing substrate parameters.
            Default is 'substrate_gummersbach.xml'.
        """
        self._feeding_freq = feeding_freq
        self._total_simtime = total_simtime
        self._substrate_xml = substrate_xml

        # Resolve path to substrate XML
        # 1. Check if it's an absolute path
        if os.path.isabs(substrate_xml):
            self._xml_path = substrate_xml
        else:
            # 2. Check if it's in the data/substrates directory
            data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "substrates"))
            self._xml_path = os.path.join(data_dir, substrate_xml)

            # 3. Fallback to current directory
            if not os.path.exists(self._xml_path):
                self._xml_path = os.path.abspath(substrate_xml)

        if not os.path.exists(self._xml_path):
            raise FileNotFoundError(f"Substrate XML file not found at {self._xml_path}")

        # Initialize substrates from DLL
        self._mySubstrates = substrates(self._xml_path)

        # array specifying the total simulation time of the complete experiment in days
        self._simtime = np.arange(0, total_simtime, float(feeding_freq / 24))

        # Storage for calculation results
        self._adm_input = None
        self._Q = None

    def header(self) -> List[str]:
        """Get ADM1 state vector header."""
        return [
            "S_su", "S_aa", "S_fa", "S_va", "S_bu", "S_pro", "S_ac", "S_h2",
            "S_ch4", "S_co2", "S_nh4", "S_I", "X_xc", "X_ch", "X_pr", "X_li",
            "X_su", "X_aa", "X_fa", "X_c4", "X_pro", "X_ac", "X_h2", "X_I",
            "X_p", "S_cation", "S_anion", "S_va_ion", "S_bu_ion", "S_pro_ion",
            "S_ac_ion", "S_hco3_ion", "S_nh3", "Q"
        ]

    def get_influent_dataframe(self, Q: List[float] = None) -> pd.DataFrame:
        """
        Generate ADM1 input stream as DataFrame for entire simulation.

        Parameters
        ----------
        Q : List[float], optional
            Volumetric flow rates [m³/d]. If provided, calls create_inputstream first.
            If None, uses the last calculated adm_input.

        Returns
        -------
        pd.DataFrame
            ADM1 input stream with columns: time, S_su, S_aa, ..., Q
        """
        if Q is not None:
            self.create_inputstream(Q, len(self._simtime))

        if self._adm_input is None:
            return pd.DataFrame()

        # Create the data object with time column
        data = []
        for i, time in enumerate(self._simtime):
            row = [time] + list(self._adm_input[i])
            data.append(row)

        header = ["time"] + self.header()
        return pd.DataFrame(data, columns=header)

    def get_substrate_names(self) -> List[str]:
        """
        Get list of all substrate names defined in the XML file.

        Returns
        -------
        List[str]
            List of substrate names.
        """
        names = []
        for i in range(self._mySubstrates.getNumSubstrates()):
            names.append(self._mySubstrates.get_name_solids(i))
        return names

    def create_inputstream(self, Q: List[float], n_steps: int) -> np.ndarray:
        """
        Create ADM1 input stream for a given substrate mix and number of steps.

        Parameters
        ----------
        Q : List[float]
            Volumetric flow rates for each substrate [m^3/d].
        n_steps : int
            Number of simulation steps.

        Returns
        -------
        np.ndarray
            Matrix of ADM1 input states (n_steps x 34).
        """
        self._Q = Q

        # Check dimension of Q
        n_substrates = self._mySubstrates.getNumSubstrates()
        if len(Q) != n_substrates:
            raise ValueError(f"Flow rate list Q must have {n_substrates} elements, but has {len(Q)}.")

        # Preferred path for pythonnet 3.x on Linux/Mono: pass explicit System.Double[,]
        # Using _mixADMstreams internal logic for consistency
        mixed_state = self._mixADMstreams(Q)

        # Create input matrix
        self._adm_input = np.tile(mixed_state, (n_steps, 1))

        return self._adm_input

    def get_substrate_params(self, Q: List[float]) -> dict:
        """
        Calculate substrate-dependent ADM1 parameters for a mix.

        Parameters
        ----------
        Q : List[float]
            Volumetric flow rates for each substrate [m^3/d].

        Returns
        -------
        dict
            Substrate-dependent parameters.
        """
        # Create input stream first to populate internal state
        self.create_inputstream(Q, 1)

        # Preferred path for pythonnet 3.x on Linux/Mono: pass explicit System.Double[,]
        try:
            from System import Array, Double

            q_values = [float(q) for q in Q]
            q_2d = Array.CreateInstance(Double, 1, len(q_values))
            for idx, value in enumerate(q_values):
                q_2d[0, idx] = value

            q_arg = q_2d
        except Exception:
            # Fallback path: numpy 2D array
            q_arg = np.atleast_2d(np.asarray(Q, dtype=float))

        # Get factors from DLL
        f_ch_xc, f_pr_xc, f_li_xc, f_xI_xc, f_sI_xc, f_xp_xc = self._mySubstrates.calcfFactors(q_arg)

        # Get kinetic parameters from DLL
        k_dis = self._mySubstrates.calcDisintegrationParam(q_arg)
        k_hyd_ch, k_hyd_pr, k_hyd_li = self._mySubstrates.calcHydrolysisParams(q_arg)
        k_m_c4, k_m_pro, k_m_ac, k_m_h2 = self._mySubstrates.calcMaxUptakeRateParams(q_arg)

        return {
            "f_ch_xc": f_ch_xc,
            "f_pr_xc": f_pr_xc,
            "f_li_xc": f_li_xc,
            "f_xI_xc": f_xI_xc,
            "f_sI_xc": f_sI_xc,
            "f_xp_xc": max(f_xp_xc, 0.0),
            "k_dis": k_dis,
            "k_hyd_ch": k_hyd_ch,
            "k_hyd_pr": k_hyd_pr,
            "k_hyd_li": k_hyd_li,
            "k_m_c4": k_m_c4,
            "k_m_pro": k_m_pro,
            "k_m_ac": k_m_ac,
            "k_m_h2": k_m_h2,
        }

    @staticmethod
    def get_substrate_feed_mixtures(Q, n=13):
        """Generate variations of substrate feed mixtures for optimization/sensitivity."""
        Qnew = [[q for q in Q] for i in range(0, n)]
        active = [i for i, q in enumerate(Q) if q > 0]

        for idx in active:
            Qnew[1][idx] = Q[idx] + 1.5

        if n > 2:
            for idx in active:
                Qnew[2][idx] = max(0.0, Q[idx] - 1.5)

        for i in range(3, n):
            for idx in active:
                Qnew[i][idx] = max(0.0, Q[idx] + np.random.uniform() * 3.0 - 1.5)

        return Qnew

    def calc_OLR_fromTOC(self, Q: List[float], V_liq: float) -> float:
        """Calculate Organic Loading Rate (OLR) from TOC [kg COD/(m³·d)]."""
        OLR = 0
        for i in range(1, self._mySubstrates.getNumSubstrates() + 1):
            TOC_i = self._get_TOC(self._mySubstrates.getIDs()[i-1]).Value
            OLR += TOC_i * Q[i - 1]
        return OLR / V_liq

    def get_substrate_params_string(self, substrate_id: str) -> str:
        """Get formatted string of substrate parameters."""
        mySubstrate = self._mySubstrates.get(substrate_id)
        pH = self._mySubstrates.get_param_of(substrate_id, "pH")
        TS = self._mySubstrates.get_param_of(substrate_id, "TS")
        VS = self._mySubstrates.get_param_of(substrate_id, "VS")
        BMP = np.round(self._mySubstrates.get_param_of(substrate_id, "BMP"), 3)
        TKN = np.round(self._mySubstrates.get_param_of(substrate_id, "TKN"), 2)

        Xc = mySubstrate.calcXc()
        return (
            "pH value: {0} \n"
            "Dry matter: {1} %FM \n"
            "Volatile solids content: {2} %TS \n"
            "Particulate chemical oxygen demand: {3} \n"
            "Particulate disintegrated chemical oxygen demand: {4} \n"
            "Total organic carbon: {5} \n"
            "Carbon-to-Nitrogen ratio: {6} \n"
            "Biochemical methane potential: {7} l/gFM \n"
            "Total Kjeldahl Nitrogen: {8} %FM"
        ).format(
            pH, TS, VS, Xc.printValue(),
            mySubstrate.calcCOD_SX().printValue(),
            self._get_TOC(substrate_id).printValue(),
            np.round(mySubstrate.calcCtoNratio(), 2),
            BMP, TKN,
        )

    def _get_TOC(self, substrate_id):
        """Get total organic carbon (TOC) of the given substrate."""
        mySubstrate = self._mySubstrates.get(substrate_id)
        return mySubstrate.calcTOC()

    def _mixADMstreams(self, Q: List[float]) -> List[float]:
        """Calculate weighted ADM1 input stream from substrate mix using DLL."""
        # Check dimension of Q
        n_substrates = self._mySubstrates.getNumSubstrates()
        if len(Q) != n_substrates:
            raise ValueError(f"Flow rate list Q must have {n_substrates} elements, but has {len(Q)}.")

        # Preferred path for pythonnet 3.x on Linux/Mono: pass explicit System.Double[,]
        try:
            from System import Array, Double

            # Create 2D array [1, n_substrates]
            q_values = [float(q) for q in Q]
            q_2d = Array.CreateInstance(Double, 1, n_substrates)
            for idx, value in enumerate(q_values):
                q_2d[0, idx] = value

            # Use DLL to calculate mixed ADM1 state
            mixed_state = self._mySubstrates.mixADMstreams(q_2d)
        except Exception:
            # Fallback path: numpy 2D array works on some local runtimes
            try:
                q_2d_np = np.atleast_2d(np.asarray(Q, dtype=float))
                mixed_state = self._mySubstrates.mixADMstreams(q_2d_np)
            except Exception:
                # Final fallback for legacy runtimes that still accept 1D arrays
                q_1d = [float(q) for q in Q]
                mixed_state = self._mySubstrates.mixADMstreams(q_1d)

        return [float(val) for val in mixed_state]

    # *** PROPERTIES ***
    @property
    def mySubstrates(self):
        """Reference to the C# Substrates object."""
        return self._mySubstrates

    @property
    def adm_input(self) -> np.ndarray:
        """The calculated ADM1 input stream matrix."""
        return self._adm_input

    @property
    def Q(self) -> List[float]:
        """Volumetric flow rates used for calculation."""
        return self._Q

    @property
    def feeding_freq(self) -> int:
        """Feeding frequency in hours."""
        return self._feeding_freq

    @property
    def total_simtime(self) -> int:
        """Total simulation time in days."""
        return self._total_simtime

    @property
    def simtime(self) -> np.ndarray:
        """Simulation time array [days]."""
        return self._simtime

    @property
    def xml_path(self) -> str:
        """Full path to the substrate XML file."""
        return self._xml_path
