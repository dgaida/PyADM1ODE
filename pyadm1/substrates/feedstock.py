# -*- coding: utf-8 -*-
"""
Created on Fri Nov 04 09:56:06 2023

Feedstock class for substrate management and ADM1 input stream calculation.

Substrate parameters are defined in XML files and accessed via C# DLLs.

@author: Daniel Gaida
"""

import clr

# import os
import numpy as np
import pandas as pd
from typing import List
from pathlib import Path

# CLR reference must be added before importing from DLL
clr.AddReference("pyadm1/dlls/substrates")
clr.AddReference("pyadm1/dlls/biogas")
clr.AddReference("pyadm1/dlls/plant")
clr.AddReference("pyadm1/dlls/physchem")

from biogas import substrates, ADMstate  # noqa: E402  # type: ignore


data_path = Path(__file__).parent.parent / "data" / "substrates"


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
            Total simulation time [days], by default 60
        substrate_xml : str, optional
            Path to substrate XML file, by default "substrate_gummersbach.xml"
        """
        # the length of the total experiment here is 60 days
        self._simtime = np.arange(0, total_simtime, float(feeding_freq / 24))

        print(data_path / "substrate_gummersbach.xml")

    # *** PUBLIC SET methods ***

    # *** PUBLIC GET methods ***

    def get_influent_dataframe(self, Q: List[float]) -> pd.DataFrame:
        """
        Generate ADM1 input stream as DataFrame for entire simulation.

        The input stream is constant over the simulation duration and depends
        on the volumetric flow rate of each substrate.

        Parameters
        ----------
        Q : List[float]
            Volumetric flow rates [m³/d], e.g., [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]

        Returns
        -------
        pd.DataFrame
            ADM1 input stream with columns: time, S_su, S_aa, ..., Q
        """
        ADMstreamMix = self._mixADMstreams(Q)

        # Create the data object
        data = [[i, *ADMstreamMix] for i in self._simtime]

        header = ["time", *self._header]

        # Check if the data rows match the header length
        if any(len(row) != len(header) for row in data):
            raise ValueError("Data rows do not match the header length")

        df = pd.DataFrame(data, columns=header)

        return df

    # *** PUBLIC methods ***

    # *** PUBLIC STATIC/CLASS GET methods ***

    @staticmethod
    def get_substrate_feed_mixtures(Q, n=13):
        Qnew = [[q for q in Q] for i in range(0, n)]
        # TODO: assumes that we only have two substrates
        # 2nd simulation runs with a volumetric flow rate of Q + 1.5 m^3/d
        Qnew[1][0] = Q[0] + 1.5
        Qnew[1][1] = Q[1] + 1.5

        if n > 2:
            # 3rd simulation runs with a volumetric flow rate of Q - 1.5 m^3/d
            Qnew[2][0] = Q[0] - 1.5
            Qnew[2][1] = Q[1] - 1.5

        # create n - 3 random flow rates
        for i in range(3, n):
            Qnew[i][0] = Q[0] + np.random.uniform() * 3.0 - 1.5
            Qnew[i][1] = Q[1] + np.random.uniform() * 3.0 - 1.5

        return Qnew

    @classmethod
    def calc_OLR_fromTOC(cls, Q: List[float], V_liq: float) -> float:
        """
        Calculate Organic Loading Rate (OLR) from substrate mix given by Q and the liquid volume of the digester.

        Parameters
        ----------
        Q : List[float]
            Volumetric flow rates [m³/d], e.g.: Q = [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]
        V_liq : float
            Liquid volume of digester [m³]

        Returns
        -------
        float
            Organic loading rate [kg COD/(m³·d)]
        """
        OLR = 0

        for i in range(1, cls._mySubstrates.getNumSubstrates() + 1):
            TOC_i = cls._get_TOC(cls._mySubstrates.getID(i)).Value

            OLR += TOC_i * Q[i - 1]

        OLR = OLR / V_liq

        return OLR

    @classmethod
    def get_substrate_params_string(cls, substrate_id: str) -> str:
        """
        Get substrate parameters of substrate substrate_id that are stored in substrate_...xml as formatted string.

        Parameters
        ----------
        substrate_id : str
            Substrate ID as defined in XML file: substrate_...xml

        Returns
        -------
        str
            Formatted string containing substrate parameters
        """
        mySubstrate = cls._mySubstrates.get(substrate_id)

        pH = cls._mySubstrates.get_param_of(substrate_id, "pH")
        TS = cls._mySubstrates.get_param_of(substrate_id, "TS")
        VS = cls._mySubstrates.get_param_of(substrate_id, "VS")
        BMP = np.round(cls._mySubstrates.get_param_of(substrate_id, "BMP"), 3)
        TKN = np.round(cls._mySubstrates.get_param_of(substrate_id, "TKN"), 2)

        Xc = mySubstrate.calcXc()

        params = (
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
            pH,
            TS,
            VS,
            Xc.printValue(),
            mySubstrate.calcCOD_SX().printValue(),
            cls._get_TOC(substrate_id).printValue(),
            np.round(mySubstrate.calcCtoNratio(), 2),
            BMP,
            TKN,
        )

        return params

    # *** PRIVATE STATIC/CLASS methods ***

    @classmethod
    def _get_TOC(cls, substrate_id):
        """
        Get total organic carbon (TOC) of the given substrate substrate_id. needed to calculate the
        organic loading rate of the digester.

        Parameters
        ----------
        substrate_id : str
            Substrate ID

        Returns
        -------
        PhysValue
            TOC value with units
        """
        mySubstrate = cls._mySubstrates.get(substrate_id)
        TOC = mySubstrate.calcTOC()
        return TOC

    @classmethod
    def _mixADMstreams(cls, Q: List[float]) -> List[float]:
        """
        Calculate weighted ADM1 input stream from substrate mix.

        Calls C# DLL methods (ADMstate.calcADMstream) to calculate ADM1 stream for each substrate
        and weighs them according to volumetric flow rates.

        How the input stream is calculated is defined in the
        PhD thesis Gaida: Dynamic Real-Time Substrate feed optimization of anaerobic co-digestion plants, 2014

        Parameters
        ----------
        Q : List[float]
            Volumetric flow rates [m³/d]. length of Q must be equal to number of substrates defined in
            substrate_...xml file

        Returns
        -------
        List[float]
            Mixed ADM1 input stream (34 dimensions)
        """
        ADMstreamAllSubstrates = []

        for i in range(1, cls._mySubstrates.getNumSubstrates() + 1):
            ADMstream = ADMstate.calcADMstream(cls._mySubstrates.get(i), Q[i - 1])

            myData_l = [row for row in ADMstream]

            ADMstreamAllSubstrates.append(myData_l)

        ADMstreamAllSubstrates = np.ravel(ADMstreamAllSubstrates)

        ADMstreamMix = ADMstate.mixADMstreams(ADMstreamAllSubstrates)

        ADMstreamMix = [row for row in ADMstreamMix]

        return ADMstreamMix

    # *** PRIVATE methods ***

    # *** PUBLIC properties ***

    def mySubstrates(self):
        """Substrates object from C# DLL."""
        return self._mySubstrates

    def header(self) -> List[str]:
        """Names of ADM1 input stream components."""
        return self._header

    def simtime(self) -> np.ndarray:
        """Simulation time array [days]."""
        return self._simtime

    # *** PRIVATE variables ***

    _mySubstrates = substrates("data/substrates/substrate_gummersbach.xml")
    # _mySubstrates = substrates(os.path.join(data_path, "substrate_gummersbach.xml"))

    # names of ADM1 input stream components
    _header = [
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
        "X_xc",
        "X_ch",
        "X_pr",
        "X_li",
        "X_su",
        "X_aa",
        "X_fa",
        "X_c4",
        "X_pro",
        "X_ac",
        "X_h2",
        "X_I",
        "X_p",
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

    # array specifying the total simulation time of the complete experiment in days, has to start at 0 and include
    # the timesteps where the substrate feed may change. Example [0, 2, 4, 6, ..., 50]. This means every 2 days the
    # substrate feed may change and the total simulation duration is 50 days
    _simtime = None
