# benchmark/dataset/BGA2/gold.py
"""
Reference solution ("Gold") for datapoint BGA2 as executable PyADM1ODE code.

Convention: the variable ``plant`` (a ``BiogasPlant``) must exist at the end.
"""

from pyadm1 import BiogasPlant, Feedstock
from pyadm1.configurator.plant_configurator import PlantConfigurator

# Substrates are not scored — any valid Feedstock suffices.
feedstock = Feedstock(["cattle_manure"], feeding_freq=24, total_simtime=30)

plant = BiogasPlant("BGA2")
cfg = PlantConfigurator(plant, feedstock)

# V_liq = pi/4 * D^2 * fill_fraction * H_wall
cfg.add_digester("F1", V_liq=2244, V_gas=300, T_ad=313.15, name="Fermenter 1")
cfg.add_digester("N1", V_liq=2244, V_gas=300, T_ad=313.15, name="Nachgaerer")
cfg.add_digester("G1", V_liq=3817, V_gas=500, T_ad=293.15, name="Gaerrestlager")

# BHKW 250 kW — add_chp auto-creates bhkw_flare (= Notfackel)
cfg.add_chp("bhkw", P_el_nom=250.0, eta_el=0.40, eta_th=0.45, name="BHKW 250 kW")

# Liquid cascade
cfg.connect("F1", "N1", "liquid")
cfg.connect("N1", "G1", "liquid")

# Gas storages -> BHKW; bhkw -> bhkw_flare is wired automatically by add_chp
cfg.auto_connect_digester_to_chp("F1", "bhkw")
cfg.auto_connect_digester_to_chp("N1", "bhkw")
cfg.auto_connect_digester_to_chp("G1", "bhkw")

plant.initialize()
