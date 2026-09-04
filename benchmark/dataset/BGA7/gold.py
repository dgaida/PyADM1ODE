# benchmark/dataset/BGA7/gold.py
"""
Reference solution ("Gold") for datapoint BGA7 as executable PyADM1ODE code.

BGA7 is a flexibilised plant with a serial fermenter cascade and two CHPs:

    Dosierer + Vorgrube -> F1 -> F2 -> sep -> G1 (gas-tight) -> G2 (open)
    gas from F1, F2 and G1 -> BHKW 1 (250 kW, base load) and BHKW 2 (550 kW, flex)

The whole digestate of F2 runs through the screw press; the solid fraction is
sold as bedding and fertiliser and leaves the model, the liquid phase flows on
into G1. PyADM1ODE routes that because the Separator emits ``Q_out``/
``state_out`` for the liquid phase (particulates depleted by the separation
efficiency, dissolved components unchanged).

Two modelling notes:

* ``add_chp`` creates one flare per engine, so the plant carries ``bhkw1_flare``
  and ``bhkw2_flare`` even though the description mentions a single flare.
* G2 has no gas capture. PyADM1ODE always attaches a gas storage to a digester,
  so ``G2_storage`` exists but stays unconnected — the matcher reports that as a
  dead gas path, which is exactly what an open store is.

Convention: the variable ``plant`` (a ``BiogasPlant``) must exist at the end.
"""

from pyadm1 import BiogasPlant, Feedstock
from pyadm1.components.biological.separator import Separator
from pyadm1.configurator.plant_configurator import PlantConfigurator

# Substrates are not scored, but the feed mirrors the task description:
# 20 t/d maize silage, 4 t/d sugar-beet silage (not in the substrate library —
# green rye silage stands in), 3 t/d grass silage and 15 m3/d pig slurry, all
# into F1.
feedstock = Feedstock(
    ["maize_silage_milk_ripeness", "green_rye_silage", "grass_silage", "swine_manure"],
    feeding_freq=24,
    total_simtime=30,
)

plant = BiogasPlant("BGA7")
cfg = PlantConfigurator(plant, feedstock)

# V_liq = pi/4 * D^2 * H_wall * 0.90 (Fuellgrad). G2 is open: its gas space is
# just the 10 % freeboard, not a dome.
cfg.add_digester("F1", V_liq=2443, V_gas=610, T_ad=313.15, name="Fermenter 1", Q_substrates=[20, 4, 3, 15])
cfg.add_digester("F2", V_liq=2443, V_gas=610, T_ad=313.15, name="Fermenter 2")
cfg.add_digester("G1", V_liq=4343, V_gas=1290, T_ad=293.15, name="Gaerrestlager 1")
cfg.add_digester("G2", V_liq=5497, V_gas=611, T_ad=293.15, name="Gaerrestlager 2 (offen)")

# Two engines: base load + flexible peaking unit. Each one gets its own flare.
cfg.add_chp("bhkw1", P_el_nom=250.0, eta_el=0.40, eta_th=0.45, name="BHKW 1 (250 kW)")
cfg.add_chp("bhkw2", P_el_nom=550.0, eta_el=0.42, eta_th=0.43, name="BHKW 2 (550 kW)")

plant.add_component(Separator("sep", separator_type="screw_press", name="Separator"))

# Serial liquid cascade; the digestate of F2 passes through the press before it
# reaches the store, so the solid fraction really leaves the liquid path.
cfg.connect("F1", "F2", "liquid")
cfg.connect("F2", "sep", "liquid")
cfg.connect("sep", "G1", "liquid")
cfg.connect("G1", "G2", "liquid")

# Gas from F1, F2 and G1 feeds both engines; G2 has no gas capture.
for digester in ("F1", "F2", "G1"):
    cfg.auto_connect_digester_to_chp(digester, "bhkw1")
    cfg.auto_connect_digester_to_chp(digester, "bhkw2")

plant.initialize()
