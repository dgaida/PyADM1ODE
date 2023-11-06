# -*- coding: utf-8 -*-
"""
Created on Fri Nov 04 09:56:06 2023

@author: Daniel Gaida
"""

import clr
import numpy as np
import pandas as pd

# have to add DLLs as references as they are used here
clr.AddReference('dlls/substrates')
clr.AddReference('dlls/biogas')
clr.AddReference('dlls/plant')
clr.AddReference('dlls/physchem')

from biogas import substrates
from biogas import ADMstate


"""
this class returns parameters of substrates and also creates the input stream for the ADM1 model. The substrate
parameters are defined in the XML file substrate_...xml. This xml file can be accessed via the DLLs in the dlls
subfolder. Inside the DLLs is also defined how the ADM1 input stream is calculated for a mix of different substrates.
More information on how the substrate mix is calculated can be found in:
Gaida, D., Dynamic real-time substrate feed optimization of anaerobic co-digestion plants, PhD thesis, Leiden, 2014.

It is expected that the file 'substrate_gummersbach.xml' is in the same folder as this *.py file. 
"""
class Feedstock:
    # *** CONSTRUCTORS ***
    def __init__(self, feeding_freq, total_simtime=60):
        """
        Standard constructor

        :param feeding_freq: the sample time between each feeding event, measured in hours
        :param total_simtime: total simulation time in days
        """
        # the length of the total experiment here is 60 days
        self._simtime = np.arange(0, total_simtime, float(feeding_freq / 24))

    # *** PUBLIC SET methods ***

    # *** PUBLIC GET methods ***

    def get_influent_dataframe(self, Q):
        """
        return the ADM input stream over the complete experiment duration as a pandas DataFrame. The input stream
        is constant over the complete simulation duration and depends on the volume flow of each substrate defined by Q.

        :param Q: volume flow, e.g.: Q = [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]
        :return: pandas DataFrame
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

    @classmethod
    def calc_OLR_fromTOC(cls, Q, V_liq):
        """
        Calculates Organic loading rate for given substrate mix given by Q and the liquid volume of the digester
        :param Q: volumetric flow rate of substrates as array such as: Q = [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]
        :param V_liq: liquid volume of digester
        :return:
        """
        OLR = 0

        for i in range(1, cls._mySubstrates.getNumSubstrates() + 1):
            TOC_i = cls._get_TOC(cls._mySubstrates.getID(i)).Value

            OLR += TOC_i * Q[i - 1]

        OLR = OLR / V_liq

        return OLR

    @classmethod
    def get_substrate_params_string(cls, substrate_id):
        """
        Gets a couple of substrate parameters of substrate substrate_id that are stored in substrate_...xml and returns
        the parameters as a string

        :param substrate_id: ID of a substrate defined in substrate_...xml
        :return: a string containing the values of a couple of substrate parameters of substrate_id
        """
        mySubstrate = cls._mySubstrates.get(substrate_id)

        pH = cls._mySubstrates.get_param_of(substrate_id, 'pH')
        TS = cls._mySubstrates.get_param_of(substrate_id, 'TS')
        VS = cls._mySubstrates.get_param_of(substrate_id, 'VS')
        BMP = np.round(cls._mySubstrates.get_param_of(substrate_id, 'BMP'), 3)

        Xc = mySubstrate.calcXc()

        params = ('pH value: {0} \n'
                  'Dry matter: {1} %FM \n'
                  'Volatile solids content: {2} %TS \n'
                  'Particulate chemical oxygen demand: {3} \n'
                  'Particulate disintegrated chemical oxygen demand: {4} \n'
                  'Total organic carbon: {5} \n'
                  'Carbon-to-Nitrogen ratio: {6} \n'
                  'Biochemical methane potential: {7} l/gFM').format(pH, TS, VS, Xc.printValue(),
                                                                     mySubstrate.calcCOD_SX().printValue(),
                                                                     cls._get_TOC(substrate_id).printValue(),
                                                                     np.round(mySubstrate.calcCtoNratio(), 2), BMP)

        return params

    # *** PRIVATE STATIC/CLASS methods ***

    @classmethod
    def _get_TOC(cls, substrate_id):
        """
        Returns the total organic carbon of the given substrate substrate_id. needed to calculate the
        organic loading rate of the digester

        :param substrate_id: ID of a substrate defined in substrate_...xml
        :return: TOC value of given substrate
        """
        mySubstrate = cls._mySubstrates.get(substrate_id)
        TOC = mySubstrate.calcTOC()
        return TOC

    @classmethod
    def _mixADMstreams(cls, Q):
        """
        Calculates ADM1 input stream for all substrates and weighs them according to the volumetric flow rate of each
        substrate defined in Q
        This method calls ADMstate.calcADMstream defined in a DLL. How the input stream is calculated is defined in the
        PhD thesis Gaida: Dynamic Real-Time Substrate feed optimization of anaerobic co-digestion plants, 2014
        :param Q: array of volumetric flow rates for all substrates measured in m^3/d. length of Q must be equal to
        number of substrates defined in substrate_...xml file
        :return: the ADM1 input stream as a 34 dimensional vector
        """
        ADMstreamAllSubstrates = []

        for i in range(1,cls._mySubstrates.getNumSubstrates() + 1):
          ADMstream = ADMstate.calcADMstream(cls._mySubstrates.get(i), Q[i-1])

          myData_l = [row for row in ADMstream]

          ADMstreamAllSubstrates.append(myData_l)

        ADMstreamAllSubstrates = np.ravel(ADMstreamAllSubstrates)

        ADMstreamMix = ADMstate.mixADMstreams(ADMstreamAllSubstrates)

        ADMstreamMix = [row for row in ADMstreamMix]

        return ADMstreamMix

    # *** PRIVATE methods ***

    # *** PUBLIC properties ***

    def mySubstrates(self):
        return self._mySubstrates

    def header(self):
        return self._header

    def simtime(self):
        return self._simtime

    # *** PRIVATE variables ***

    _mySubstrates = substrates('substrate_gummersbach.xml')

    # names of ADM1 input stream components
    _header = ["S_su", "S_aa", "S_fa", "S_va", "S_bu", "S_pro", "S_ac", "S_h2", "S_ch4", "S_co2", "S_nh4", "S_I",
              "X_xc", "X_ch", "X_pr", "X_li", "X_su", "X_aa", "X_fa", "X_c4", "X_pro", "X_ac", "X_h2", "X_I", "X_p",
              "S_cation", "S_anion", "S_va_ion", "S_bu_ion", "S_pro_ion", "S_ac_ion",
              "S_hco3_ion", "S_nh3", "Q"]

    # array specifying the total simulation time of the complete experiment in days, has to start at 0 and include
    # the timesteps where the substrate feed may change. Example [0, 2, 4, 6, ..., 50]. This means every 2 days the
    # substrate feed may change and the total simulation duration is 50 days
    _simtime = None
