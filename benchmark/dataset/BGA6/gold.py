# benchmark/dataset/BGA6/gold.py
"""
Reference solution ("Gold") for datapoint BGA6 as executable PyADM1ODE code.

BGA6 is a two-stage energy-crop plant: a horizontal plug-flow digester running
thermophilically, a mesophilic post-digester, a screw-press separator at the
post-digester outlet and an unheated digestate store.

The digestate leaves the post-digester through the separator, which splits it:

    F1 -> N1 -> sep --(38 %)--> F1     press water, ~10 m3/d back for mashing
                    --(62 %)--> G1     the rest into the store

The recirculation closes a loop F1 -> N1 -> sep -> F1. PyADM1ODE routes it
because the Separator hands the liquid phase on as ``Q_out``/``state_out`` —
particulates depleted by the separation efficiency, dissolved components
unchanged — and ``split_fraction`` on the connection carries only part of the
stream. The press water therefore needs no substrate-library entry: it brings
its own ADM1 state, which the digester mixes flow-weighted with the fresh feed.

Still NOT wired: the gas transfer F1 -> gas space of N1. Gas storages only sum
the ``Q_gas`` of the digesters connected to them, so storage-to-storage carries
nothing. Each digester keeps its own gas storage and feeds the CHP.

Convention: the variable ``plant`` (a ``BiogasPlant``) must exist at the end.
"""

from pyadm1 import BiogasPlant, Feedstock
from pyadm1.components.biological.separator import Separator
from pyadm1.configurator.plant_configurator import PlantConfigurator

# Substrates are not scored, but the feed mirrors the task description:
# 14 t/d maize silage, 4 t/d whole-crop silage and 3 t/d solid cattle manure
# through the solid feeder; the ~10 m3/d press water arrives through the
# recirculation edge, not as a feedstock slot.
feedstock = Feedstock(
    ["maize_silage_milk_ripeness", "cereal_gps_silage", "cattle_manure_solid"],
    feeding_freq=24,
    total_simtime=30,
)

plant = BiogasPlant("BGA6")
cfg = PlantConfigurator(plant, feedstock)

# F1: plug-flow tank 32 x 6 x 4.5 m -> 864 m3 gross, 90 % fill, headspace only.
# N1/G1: round tanks, V_liq = pi/4 * D^2 * H_wall * 0.90.
cfg.add_digester("F1", V_liq=778, V_gas=86, T_ad=325.15, name="Pfropfenstromfermenter", Q_substrates=[14, 4, 3])
cfg.add_digester("N1", V_liq=2053, V_gas=505, T_ad=315.15, name="Nachgaerer 1")
cfg.add_digester("G1", V_liq=2867, V_gas=719, T_ad=293.15, name="Gaerrestlager")

# BHKW 360 kW — add_chp auto-creates bhkw_flare (= Notfackel)
cfg.add_chp("bhkw", P_el_nom=360.0, eta_el=0.41, eta_th=0.43, name="BHKW 360 kW")

plant.add_component(Separator("sep", separator_type="screw_press", name="Separator"))

# Liquid path: the whole digestate of N1 runs through the press, 38 % of the
# press water returns to F1 (~10 m3/d in steady state), the rest goes to G1.
cfg.connect("F1", "N1", "liquid")
cfg.connect("N1", "sep", "liquid")
cfg.connect("sep", "F1", "liquid", split_fraction=0.38)
cfg.connect("sep", "G1", "liquid", split_fraction=0.62)

# Gas storages -> BHKW; bhkw -> bhkw_flare is wired automatically by add_chp
cfg.auto_connect_digester_to_chp("F1", "bhkw")
cfg.auto_connect_digester_to_chp("N1", "bhkw")
cfg.auto_connect_digester_to_chp("G1", "bhkw")

plant.initialize()
