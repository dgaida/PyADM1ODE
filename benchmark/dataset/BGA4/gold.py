# benchmark/dataset/BGA4/gold.py
"""
Reference solution ("Gold") for datapoint BGA4 as executable PyADM1ODE code.

BGA4 is the plant of the attached offer ("Individualanlage 250 kW"):
pre-pit + solid feeder -> Fermenter 1 -> Nachgaerer 1 -> Gaerproduktlager 1,
one 250 kW CHP, one biogas flare.

Convention: the variable ``plant`` (a ``BiogasPlant``) must exist at the end.
"""

from pyadm1 import BiogasPlant, Feedstock
from pyadm1.configurator.plant_configurator import PlantConfigurator

# Substrates are not scored, but the feed mirrors the annual quantities of the
# offer divided by 365 d: 4793 m3/a cattle slurry (pre-pit) plus 1729 t/a maize
# silage, 911 t/a solid cattle manure, 615 t/a dry chicken manure and 130 t/a
# grass silage (solid feeder) — all into F1.
feedstock = Feedstock(
    [
        "maize_silage_milk_ripeness",
        "cattle_manure",
        "cattle_manure_solid",
        "chicken_manure_dry",
        "grass_silage",
    ],
    feeding_freq=24,
    total_simtime=30,
)

plant = BiogasPlant("BGA4")
cfg = PlantConfigurator(plant, feedstock)

# V_liq = pi/4 * D^2 * H_wall * 0.90 (Fuellgrad); V_gas = gas storage volume of
# the quarter-sphere double-membrane roof as quoted in the offer.
# Q_substrates [m3/d] per slot.
cfg.add_digester(
    "F1",
    V_liq=1531,
    V_gas=599,
    T_ad=313.15,
    name="Fermenter 1",
    Q_substrates=[4.7, 13.1, 2.5, 1.7, 0.4],
)
cfg.add_digester("N1", V_liq=1531, V_gas=599, T_ad=313.15, name="Nachgaerer 1")
cfg.add_digester("G1", V_liq=6537, V_gas=3432, T_ad=293.15, name="Gaerproduktlager 1")

# BHKW 250 kW — add_chp auto-creates bhkw_flare (= Biogas-Fackel)
cfg.add_chp("bhkw", P_el_nom=250.0, eta_el=0.40, eta_th=0.45, name="BHKW 250 kW")

# Liquid cascade
cfg.connect("F1", "N1", "liquid")
cfg.connect("N1", "G1", "liquid")

# Gas storages -> BHKW; bhkw -> bhkw_flare is wired automatically by add_chp
cfg.auto_connect_digester_to_chp("F1", "bhkw")
cfg.auto_connect_digester_to_chp("N1", "bhkw")
cfg.auto_connect_digester_to_chp("G1", "bhkw")

plant.initialize()
