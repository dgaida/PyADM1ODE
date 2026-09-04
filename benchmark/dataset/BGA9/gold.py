# benchmark/dataset/BGA9/gold.py
"""
Reference solution ("Gold") for datapoint BGA9 as executable PyADM1ODE code.

BGA9 is a four-stage plant with a linear cascade:

    Dosierer + Vorgrube -> F1 -> N1 -> G1 -> G2

What distinguishes it from BGA7: BOTH digestate stores are gas-tight, so both
feed the CHP — in BGA7 the second store is open and its gas is not captured.

Convention: the variable ``plant`` (a ``BiogasPlant``) must exist at the end.
"""

from pyadm1 import BiogasPlant, Feedstock
from pyadm1.configurator.plant_configurator import PlantConfigurator

# Substrates are not scored, but the feed mirrors the task description:
# 16 t/d maize silage and 1.5 t/d corn-cob mix (solid feeder) plus 20 m3/d pig
# slurry (pre-pit), all into F1.
feedstock = Feedstock(
    ["maize_silage_milk_ripeness", "corn_cob_mix", "swine_manure"],
    feeding_freq=24,
    total_simtime=30,
)

plant = BiogasPlant("BGA9")
cfg = PlantConfigurator(plant, feedstock)

# V_liq = pi/4 * D^2 * H_wall * 0.90 (Fuellgrad).
cfg.add_digester("F1", V_liq=2651, V_gas=680, T_ad=313.15, name="Fermenter 1", Q_substrates=[16, 1.5, 20])
cfg.add_digester("N1", V_liq=2651, V_gas=680, T_ad=313.15, name="Nachgaerer 1")
cfg.add_digester("G1", V_liq=3325, V_gas=799, T_ad=293.15, name="Gaerrestlager 1")
cfg.add_digester("G2", V_liq=3325, V_gas=799, T_ad=293.15, name="Gaerrestlager 2")

# BHKW 400 kW — add_chp auto-creates bhkw_flare (= Notfackel)
cfg.add_chp("bhkw", P_el_nom=400.0, eta_el=0.41, eta_th=0.44, name="BHKW 400 kW")

# Linear liquid cascade through both stores
cfg.connect("F1", "N1", "liquid")
cfg.connect("N1", "G1", "liquid")
cfg.connect("G1", "G2", "liquid")

# Gas storages -> BHKW; both stores are gas-tight, so all four tanks feed it.
for digester in ("F1", "N1", "G1", "G2"):
    cfg.auto_connect_digester_to_chp(digester, "bhkw")

plant.initialize()
