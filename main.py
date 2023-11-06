# -*- coding: utf-8 -*-
"""
Created on Fri Nov 04 09:56:06 2023

@author: Daniel Gaida
"""

from PyADM1 import get_state_zero_from_initial_state, PyADM1

from feedstock import Feedstock

from Simulator import Simulator

feeding_freq = 48   # every feeding_freq hours the controller can change the substrate feed of the digester

myfeedstock = Feedstock(feeding_freq)

adm1 = PyADM1(myfeedstock)
mySimulator = Simulator(adm1)

# initial substrate feed for all substrates. At the moment only values for the first two substrates may be changed, rest 0
# first value: corn silage, 2nd value: liquid manure, both in m^3/d
Q = [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]
#Q = [69, 64, 0, 0, 0, 0, 0, 0, 0, 0]

# initial ADM1 state vector where to start the simulation
state_zero = get_state_zero_from_initial_state("digester_initial8.csv")

## time array definition
t = myfeedstock.simtime()

# Initiate the cache data frame for storing simulation results
simulate_results = [state_zero]

t0=0

## Dynamic simulation
# Loop for simulating at each time step and feeding the results to the next time step
for n, u in enumerate(t[1:], 1):
    # you could change Q here to simulate with dynamically changing substrate feed

    adm1.createInfluent(Q, n)

    # Span for next time step
    tstep = [t0,u]

    state_zero = mySimulator.simulateADplant(tstep, state_zero)

    simulate_results.append(state_zero)

    t0 = u

    if n % 100 == 0:
        print("Simulated {0} of {1} steps.".format(n, len(t)))

# save final ADM1 state vector
adm1.save_final_state_in_csv(simulate_results)
