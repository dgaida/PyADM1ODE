# -*- coding: utf-8 -*-
"""
Created on Fri Nov 04 09:56:06 2023

@author: Daniel Gaida
"""

import numpy as np
import scipy.integrate


"""
This class is used to do simulations with the ADM1 using an ODE solver. It contains two public methods:

- determineBestFeedbyNSims: Do n simulations with ADM1 with different substrate feeds and return that 
substrate feed that let to a methane production rate that is closest to a given set point.  
- simulateADplant: Simulate ADM1 for given duration starting at a given state and returns the state

This class only works with the class PyADM1 that implements the ADM1. 
"""
class Simulator:

    # *** CONSTRUCTORS ***
    def __init__(self, adm1):
        """

        :param adm1: object of the PyADM1 class
        """
        self._adm1 = adm1

    # *** PUBLIC SET methods ***

    # *** PUBLIC GET methods ***

    # *** PUBLIC methods ***

    def determineBestFeedbyNSims(self, state_zero, Q, Qch4sp, feeding_freq, n=13):
        """
        From the state state_zero n simulations are run for 7 days with random volumetric flow rates around the given flowrate Q.
        The first simulation will be run with Q, the 2nd and 3rd with Q + 1.5 m³/d and Q - 1.5 m^3/d, respectively.
        The remaining simulations are random flow rates between Q +- 1.5 m^3/d.
        It returns the flowrate that yields a methane production rate in 7 days that is closest to the given Qch4sp. It
        also returns the corresponding biogas and methane production rates after 7 days and after feeding_freq/24 days.

        :param state_zero: ADM1 state vector where to start the simulation
        :param Q: volumetric flow rate of each substrate, e.g.: [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]
        :param Qch4sp: set point of methane flowrate in m^3/d
        :param feeding_freq: specifies how often substrate mix can be changed in days
        :param n: number of simulations to run. Must be >= 3.
        :return:
        """
        Q_CH4_7d = [0] * n
        Q_Gas_7d = [0] * n

        Qnew = [[q for q in Q] for i in range(0, n)]
        # TODO: assumes that we only have two substrates
        # 2nd simulation runs with a volumetric flow rate of Q + 1.5 m^3/d
        Qnew[1][0] = Q[0] + 1.5
        Qnew[1][1] = Q[1] + 1.5
        # 3rd simulation runs with a volumetric flow rate of Q - 1.5 m^3/d
        Qnew[2][0] = Q[0] - 1.5
        Qnew[2][1] = Q[1] - 1.5

        # create n - 3 random flow rates
        for i in range(3, n):
            Qnew[i][0] = Q[0] + np.random.uniform() * 3.0 - 1.5
            Qnew[i][1] = Q[1] + np.random.uniform() * 3.0 - 1.5

        for i,q in enumerate(Qnew):
            Q_Gas_7d[i], Q_CH4_7d[i] = self._simulate_wosavinglaststate([0, 7], state_zero, q)

        ii = np.argmin([np.sum((q - Qch4sp)**2) for q in Q_CH4_7d])
        Qbest = Qnew[ii]

        # simulate with Q for feeding_freq/24 days. The resulting gas flow rates can be helpful
        # information
        q_gas_2d, q_ch4_2d = self._simulate_wosavinglaststate([0, feeding_freq/24], state_zero, Qnew[0])

        # simulate with best substrate feed for feeding_freq/24 days. The resulting gas flow rates can be helpful
        # information
        q_gas_best_2d, q_ch4_best_2d = self._simulate_wosavinglaststate([0, feeding_freq / 24], state_zero, Qbest)

        return (Q_Gas_7d[ii][-1], Q_CH4_7d[ii][-1], Qbest, Q_Gas_7d[0][-1], Q_CH4_7d[0][-1], Qnew[0],
                q_gas_best_2d[-1], q_ch4_best_2d[-1], q_gas_2d[-1], q_ch4_2d[-1])

    def simulateADplant(self, tstep, state_zero):
        """
        Simulate ADM1 starting at state state_zero for time tstep. Returns final state. Also saves a couple of simulated
        values into lists. These lists are used to give biogas plant operator information about the last process values
        of the biogas plant
        :param tstep: [t_start, t_end]: duration of simulation in days
        :param state_zero: ADM1 state vector where to start the simulation
        :return: the final ADM1 state vector after the simulation
        """
        final_state = self._simulate_returnlaststate(tstep, state_zero)

        self._adm1.print_params_at_current_state(final_state)

        return final_state

    # *** PRIVATE methods ***

    def _simulate_wosavinglaststate(self, tstep, state_zero, Q):
        """
        Run a simulation with the ADM1 for tstep starting at the state state_zero and with the volumetric flow rate Q.
        The final state is not saved nor returned.

        :param tstep: Specifies when to start and end the simulation: [t_start, t_end]
        :param state_zero: ADM1 state vector where to start the simulation
        :param Q: volumetric flow rate of each substrate, e.g.: [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]
        :return: produced biogas and methane production rate
        """
        # create ADM1 input stream out of given Q
        self._adm1.createInfluent(Q, 0)

        # Solve and store ODE results to calculate biogas production rate
        (sim_S_su, sim_S_aa, sim_S_fa, sim_S_va, sim_S_bu, sim_S_pro, sim_S_ac, sim_S_h2, sim_S_ch4, sim_S_IC, sim_S_IN,
         sim_S_I, sim_X_xc, sim_X_ch, sim_X_pr, sim_X_li, sim_X_su, sim_X_aa, sim_X_fa, sim_X_c4, sim_X_pro, sim_X_ac,
         sim_X_h2, sim_X_I, sim_X_p, sim_S_cation, sim_S_anion, sim_S_va_ion, sim_S_bu_ion, sim_S_pro_ion,
         sim_S_ac_ion, sim_S_hco3_ion, sim_S_nh3, sim_pi_Sh2, sim_pi_Sch4, sim_pi_Sco2, sim_pTOTAL) = (
            self._simulate(tstep, state_zero))

        # Store final ODE simulation result states
        pi_Sh2, pi_Sch4, pi_Sco2, pTOTAL = sim_pi_Sh2[:-1], sim_pi_Sch4[:-1], sim_pi_Sco2[:-1], sim_pTOTAL[:-1]

        # calc final biogas production flow rates
        q_gas, q_ch4, q_co2, p_gas = self._adm1.calc_gas(pi_Sh2, pi_Sch4, pi_Sco2, pTOTAL)

        return q_gas, q_ch4

    def _simulate_returnlaststate(self, tstep, state_zero):
        """
        Simulate ADM1 and return final state

        :param tstep: Specifies when to start and end the simulation: [t_start, t_end]
        :param state_zero: ADM1 state vector where to start the simulation
        :return: ADM1 state vector after the simulation ends
        """
        # Solve and store ODE results for next step
        (sim_S_su, sim_S_aa, sim_S_fa, sim_S_va, sim_S_bu, sim_S_pro, sim_S_ac, sim_S_h2, sim_S_ch4, sim_S_co2, sim_S_nh4,
         sim_S_I, sim_X_xc, sim_X_ch, sim_X_pr, sim_X_li, sim_X_su, sim_X_aa, sim_X_fa, sim_X_c4, sim_X_pro, sim_X_ac,
         sim_X_h2, sim_X_I, sim_X_p, sim_S_cation, sim_S_anion, sim_S_va_ion, sim_S_bu_ion, sim_S_pro_ion, sim_S_ac_ion,
         sim_S_hco3_ion, sim_S_nh3, sim_pi_Sh2, sim_pi_Sch4, sim_pi_Sco2, sim_pTOTAL) = (
            self._simulate(tstep, state_zero))

        # Store ODE simulation result states
        state_vector = [sim_S_su[-1], sim_S_aa[-1], sim_S_fa[-1], sim_S_va[-1],
                        sim_S_bu[-1], sim_S_pro[-1], sim_S_ac[-1], sim_S_h2[-1], sim_S_ch4[-1], sim_S_co2[-1],
                        sim_S_nh4[-1], sim_S_I[-1], sim_X_xc[-1], sim_X_ch[-1], sim_X_pr[-1], sim_X_li[-1],
                        sim_X_su[-1], sim_X_aa[-1], sim_X_fa[-1], sim_X_c4[-1],
        sim_X_pro[-1], sim_X_ac[-1], sim_X_h2[-1], sim_X_I[-1], sim_X_p[-1], sim_S_cation[-1], sim_S_anion[-1],
        sim_S_va_ion[-1], sim_S_bu_ion[-1], sim_S_pro_ion[-1], sim_S_ac_ion[-1], sim_S_hco3_ion[-1],
        sim_S_nh3[-1], sim_pi_Sh2[-1], sim_pi_Sch4[-1], sim_pi_Sco2[-1], sim_pTOTAL[-1]]

        return state_vector

    def _simulate(self, t_step, state_zero):
        """
        Function for integration of ADM1 differential equations

        :param tstep: Specifies when to start and end the simulation: [t_start, t_end]
        :param state_zero: ADM1 state vector where to start the simulation
        :return: ADM1 state vector after the simulation ends, as 37 individual numbers
        """
        # the stuff here might also work. To use solve_ivp as a oneliner seemed to be easier
        # r = scipy.integrate.ode(adm1.ADM1_ODE).set_integrator('vode', method='bdf')

        # r.set_initial_value(state_zero, t_step[0])

        # dt = 0.05

        # while r.successful() and r.t < t_step[1]:
        #    r.integrate(r.t + dt)

        # a minimum step length of 0.05 was necessary to get accurate simulation results. Whether smaller numbers lead
        # to better results was not yet tested
        r = scipy.integrate.solve_ivp(self._adm1.ADM1_ODE, t_step, state_zero, method=self._solvermethod,
                                      t_eval=np.arange(t_step[0], t_step[1], 0.05))
        return r.y

    # *** PRIVATE STATIC/CLASS methods ***

    # *** PUBLIC properties ***

    # *** PRIVATE variables ***

    # Setting the solver method for the simulate function. The BDF solver seemed to be the only leading
    # to stable simulation runs. Remark: the ADM1 is a stiff ODE, so solver for non-stiff ODEs do not work
    _solvermethod = 'BDF'  #

    # object of the PyADM1 class
    _adm1 = None
