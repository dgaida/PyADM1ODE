# benchmark/dataset/BGA11/gold.py
"""
Reference solution ("Gold") for datapoint BGA11 as executable PyADM1ODE code.

BGA11 is the first plant of the benchmark with TWO gas consumers of DIFFERENT
type: a heat-led CHP for the plant's own process heat and a biogas upgrading
unit for the remaining gas stream.

    Dos 1 + Vorgrube -> F1 --.
                              >-- N1 -> G1
    Dos 2 + Vorgrube -> F2 --'

    gas of all four tanks --> BHKW 250 kW  (heat-led base load)
                          --> BGAA 350 m3/h (amine scrubbing -> grid injection)

Both helpers create their own flare, so the plant carries ``bhkw_flare`` AND
``bgaa_flare`` even though the description mentions a single flare.

Convention: the variable ``plant`` (a ``BiogasPlant``) must exist at the end.
"""

from pyadm1 import BiogasPlant, Feedstock
from pyadm1.configurator.plant_configurator import PlantConfigurator

# Substrates are not scored, but the feed mirrors the task description:
# per line 21 t/d maize silage, 2 t/d grass silage and 4 t/d solid cattle manure
# through the feeders, plus 15 m3/d cattle slurry from the shared pre-pit.
feedstock = Feedstock(
    ["maize_silage_milk_ripeness", "grass_silage", "cattle_manure_solid", "cattle_manure"],
    feeding_freq=24,
    total_simtime=30,
)

plant = BiogasPlant("BGA11")
cfg = PlantConfigurator(plant, feedstock)

# V_liq = pi/4 * D^2 * H_wall * 0.90 (Fuellgrad).
cfg.add_digester("F1", V_liq=2651, V_gas=680, T_ad=313.15, name="Fermenter 1", Q_substrates=[21, 2, 4, 15])
cfg.add_digester("F2", V_liq=2651, V_gas=680, T_ad=313.15, name="Fermenter 2", Q_substrates=[21, 2, 4, 15])
cfg.add_digester("N1", V_liq=3567, V_gas=995, T_ad=313.15, name="Nachgaerer 1")
cfg.add_digester("G1", V_liq=4619, V_gas=1400, T_ad=293.15, name="Gaerrestlager")

# Two consumers of different type — each creates its own flare.
cfg.add_chp("bhkw", P_el_nom=250.0, eta_el=0.39, eta_th=0.46, name="BHKW 250 kW")
cfg.add_bgaa(
    "bgaa",
    capacity_m3h=350.0,
    ch4_recovery=0.99,
    ch4_content_out=0.97,
    name="Biogasaufbereitung 350 m³/h",
)

# Liquid cascade: both fermenters into the shared post-digester, then the store.
cfg.connect("F1", "N1", "liquid")
cfg.connect("F2", "N1", "liquid")
cfg.connect("N1", "G1", "liquid")

# Every gas storage feeds BOTH consumers — the split happens in the gas pipe.
for digester in ("F1", "F2", "N1", "G1"):
    cfg.auto_connect_digester_to_chp(digester, "bhkw")
    cfg.auto_connect_digester_to_bgaa(digester, "bgaa")

plant.initialize()
