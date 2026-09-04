# benchmark/dataset/BGA8/gold.py
"""
Reference solution ("Gold") for datapoint BGA8 as executable PyADM1ODE code.

BGA8 is a four-stage plant with a strictly LINEAR cascade — this is what sets it
apart from BGA3, which has the same component types but two fermenters feeding
one post-digester in parallel:

    Dosierer + Vorgrube -> F1 -> N1 -> N2 -> G1

Both post-digesters are structurally identical and connected in series; the gas
of all four tanks goes to the single 300 kW CHP.

Convention: the variable ``plant`` (a ``BiogasPlant``) must exist at the end.
"""

from pyadm1 import BiogasPlant, Feedstock
from pyadm1.configurator.plant_configurator import PlantConfigurator

# Substrates are not scored, but the feed mirrors the task description:
# 12 t/d maize silage and 3 t/d grass silage (solid feeder) plus 12 m3/d cattle
# slurry (pre-pit), all into F1.
feedstock = Feedstock(
    ["maize_silage_milk_ripeness", "grass_silage", "cattle_manure"],
    feeding_freq=24,
    total_simtime=30,
)

plant = BiogasPlant("BGA8")
cfg = PlantConfigurator(plant, feedstock)

# V_liq = pi/4 * D^2 * H_wall * 0.90 (Fuellgrad); the plant runs at 41 C, not at
# the usual 40 C — 313.15 K would be outside the acceptance band.
cfg.add_digester("F1", V_liq=1696, V_gas=385, T_ad=314.15, name="Fermenter 1", Q_substrates=[12, 3, 12])
cfg.add_digester("N1", V_liq=2443, V_gas=610, T_ad=314.15, name="Nachgaerer 1")
cfg.add_digester("N2", V_liq=2443, V_gas=610, T_ad=314.15, name="Nachgaerer 2")
cfg.add_digester("G1", V_liq=2867, V_gas=719, T_ad=293.15, name="Gaerrestlager")

# BHKW 300 kW — add_chp auto-creates bhkw_flare (= Notfackel)
cfg.add_chp("bhkw", P_el_nom=300.0, eta_el=0.40, eta_th=0.45, name="BHKW 300 kW")

# Strictly serial liquid cascade — no parallel branch anywhere.
cfg.connect("F1", "N1", "liquid")
cfg.connect("N1", "N2", "liquid")
cfg.connect("N2", "G1", "liquid")

# Gas storages -> BHKW; bhkw -> bhkw_flare is wired automatically by add_chp
for digester in ("F1", "N1", "N2", "G1"):
    cfg.auto_connect_digester_to_chp(digester, "bhkw")

plant.initialize()
