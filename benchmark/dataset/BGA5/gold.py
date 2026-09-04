# benchmark/dataset/BGA5/gold.py
"""
Reference solution ("Gold") for datapoint BGA5 as executable PyADM1ODE code.

BGA5 is a small slurry-based farm plant (Güllekleinanlage, 75 kW): pre-pit and a
small vertical-mixer feeder charge the single digester, which overflows straight
into the gas-tight digestate store — there is no post-digester.

Convention: the variable ``plant`` (a ``BiogasPlant``) must exist at the end.
"""

from pyadm1 import BiogasPlant, Feedstock
from pyadm1.configurator.plant_configurator import PlantConfigurator

# Substrates are not scored, but the feed mirrors the task description:
# 20 m3/d cattle slurry (pre-pit) plus 3 t/d solid cattle manure and 0.8 t/d
# grass silage (solid feeder) — all into F1.
feedstock = Feedstock(
    ["cattle_manure", "cattle_manure_solid", "grass_silage"],
    feeding_freq=24,
    total_simtime=30,
)

plant = BiogasPlant("BGA5")
cfg = PlantConfigurator(plant, feedstock)

# V_liq = pi/4 * D^2 * H_wall * 0.90 (Fuellgrad); V_gas = gas space incl. dome.
# Q_substrates [m3/d] per slot.
cfg.add_digester("F1", V_liq=831, V_gas=160, T_ad=311.15, name="Fermenter 1", Q_substrates=[20, 3, 0.8])
cfg.add_digester("G1", V_liq=1414, V_gas=355, T_ad=293.15, name="Gaerrestlager")

# BHKW 75 kW — add_chp auto-creates bhkw_flare even though this small plant is
# secured by pressure relief valves only.
cfg.add_chp("bhkw", P_el_nom=75.0, eta_el=0.37, eta_th=0.44, name="BHKW 75 kW")

# Liquid cascade: the digester overflows directly into the store (no N1).
cfg.connect("F1", "G1", "liquid")

# Gas storages -> BHKW; bhkw -> bhkw_flare is wired automatically by add_chp
cfg.auto_connect_digester_to_chp("F1", "bhkw")
cfg.auto_connect_digester_to_chp("G1", "bhkw")

plant.initialize()
