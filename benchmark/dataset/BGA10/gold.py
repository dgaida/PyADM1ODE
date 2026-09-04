# benchmark/dataset/BGA10/gold.py
"""
Reference solution ("Gold") for datapoint BGA10 as executable PyADM1ODE code.

BGA10 is the first plant of the benchmark with TWO PARALLEL LINES that stay
separate all the way down to the shared store — and with five tanks:

    Dos 1 + Vorgrube 1 -> F1 -> N1 --.
                                      >-- G1
    Dos 2 + Vorgrube 2 -> F2 -> N2 --'

Not to be confused with BGA1/BGA3, where two fermenters feed ONE post-digester:
here each line has its own post-digester and only the digestate store is shared.

Convention: the variable ``plant`` (a ``BiogasPlant``) must exist at the end.
"""

from pyadm1 import BiogasPlant, Feedstock
from pyadm1.configurator.plant_configurator import PlantConfigurator

# Substrates are not scored, but the feed mirrors the task description:
# 20 t/d maize silage plus 15 m3/d cattle slurry per line.
feedstock = Feedstock(["maize_silage_milk_ripeness", "cattle_manure"], feeding_freq=24, total_simtime=30)

plant = BiogasPlant("BGA10")
cfg = PlantConfigurator(plant, feedstock)

# V_liq = pi/4 * D^2 * H_wall * 0.90 (Fuellgrad). Fermenters and post-digesters
# are structurally identical; the plant runs at 39 C, not at the usual 40 C.
cfg.add_digester("F1", V_liq=3092, V_gas=830, T_ad=312.15, name="Fermenter 1", Q_substrates=[20, 15])
cfg.add_digester("F2", V_liq=3092, V_gas=830, T_ad=312.15, name="Fermenter 2", Q_substrates=[20, 15])
cfg.add_digester("N1", V_liq=3092, V_gas=830, T_ad=312.15, name="Nachgaerer 1")
cfg.add_digester("N2", V_liq=3092, V_gas=830, T_ad=312.15, name="Nachgaerer 2")
cfg.add_digester("G1", V_liq=5791, V_gas=1450, T_ad=293.15, name="Gaerrestlager")

# BHKW 750 kW — add_chp auto-creates bhkw_flare (= Notfackel)
cfg.add_chp("bhkw", P_el_nom=750.0, eta_el=0.42, eta_th=0.43, name="BHKW 750 kW")

# Two separate lines; they meet only in the shared store.
cfg.connect("F1", "N1", "liquid")
cfg.connect("F2", "N2", "liquid")
cfg.connect("N1", "G1", "liquid")
cfg.connect("N2", "G1", "liquid")

# Gas storages -> BHKW (all five tanks are gas-tight)
for digester in ("F1", "F2", "N1", "N2", "G1"):
    cfg.auto_connect_digester_to_chp(digester, "bhkw")

plant.initialize()
