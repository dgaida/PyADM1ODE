# benchmark/dataset/BGA1/gold.py
"""
Reference solution ("Gold") for datapoint BGA1 (see BGA1.json) as
executable PyADM1ODE code.

Convention: the variable ``plant`` (a ``BiogasPlant``) must exist at the end.
"""

from pyadm1 import BiogasPlant, Feedstock
from pyadm1.configurator.plant_configurator import PlantConfigurator
from pyadm1.components.biological.separator import Separator

# Substrates are not scored, but the feed mirrors the task description:
# ~80 t/d maize silage (solid feeder) split across F1+F2, plus 20 m3/d cattle
# slurry (pre-pit) into F1+F2. Slot 0 = maize silage, slot 1 = cattle slurry.
feedstock = Feedstock(["maize_silage_milk_ripeness", "cattle_manure"], feeding_freq=24, total_simtime=30)

plant = BiogasPlant("BGA1")
cfg = PlantConfigurator(plant, feedstock)

# Q_substrates [m3/d] per slot; 80 t/d maize + 20 m3/d slurry split over F1+F2.
cfg.add_digester("F1", V_liq=3325, V_gas=870, T_ad=313.15, name="Fermenter 1", Q_substrates=[40, 10])
cfg.add_digester("F2", V_liq=3325, V_gas=870, T_ad=313.15, name="Fermenter 2", Q_substrates=[40, 10])
cfg.add_digester("N1", V_liq=3325, V_gas=870, T_ad=313.15, name="Nachgaerer")
cfg.add_digester("G1", V_liq=3817, V_gas=994, T_ad=293.15, name="Gaerrestlager")

# BGAA (500 m³/h) — auto-creates bgaa_flare (= Notfackel) for capacity overflow
cfg.add_bgaa("bgaa", capacity_m3h=500.0, name="Biogasaufbereitung 500 m³/h")

plant.add_component(Separator("sep", separator_type="screw_press", name="Separator"))

# Liquid cascade + Separator
cfg.connect("F1", "N1", "liquid")
cfg.connect("F2", "N1", "liquid")
cfg.connect("N1", "G1", "liquid")
cfg.connect("N1", "sep", "liquid")

# Gas storages -> BGAA; bgaa -> bgaa_flare is wired automatically by add_bgaa
cfg.auto_connect_digester_to_bgaa("F1", "bgaa")
cfg.auto_connect_digester_to_bgaa("F2", "bgaa")
cfg.auto_connect_digester_to_bgaa("N1", "bgaa")
cfg.auto_connect_digester_to_bgaa("G1", "bgaa")

plant.initialize()
