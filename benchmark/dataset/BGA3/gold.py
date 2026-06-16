from pyadm1 import BiogasPlant, Feedstock
from pyadm1.configurator.plant_configurator import PlantConfigurator

# Substrates are not scored, but the feed mirrors the task description:
# 50 t/d maize silage each via Dos1->F1 and Dos2->F2, plus 20 m3/d cattle slurry
# each from two pre-pits into F1+F2. Slot 0 = maize silage, slot 1 = cattle slurry.
feedstock = Feedstock(["maize_silage_milk_ripeness", "cattle_manure"], feeding_freq=24, total_simtime=30)
plant = BiogasPlant("BGA3")
cfg = PlantConfigurator(plant, feedstock)

# Q_substrates [m3/d] per slot.
cfg.add_digester("F1", V_liq=2867, V_gas=719, T_ad=313.15, name="Fermenter 1", Q_substrates=[50, 20])
cfg.add_digester("F2", V_liq=2867, V_gas=719, T_ad=313.15, name="Fermenter 2", Q_substrates=[50, 20])
cfg.add_digester("N1", V_liq=2867, V_gas=719, T_ad=313.15, name="Nachgaerer")
cfg.add_digester("G1", V_liq=3325, V_gas=799, T_ad=293.15, name="Gaerrestlager")

cfg.add_chp("bhkw", P_el_nom=500.0, eta_el=0.40, eta_th=0.45, name="BHKW 500 kW")

cfg.connect("F1", "N1", "liquid")
cfg.connect("F2", "N1", "liquid")
cfg.connect("N1", "G1", "liquid")

cfg.auto_connect_digester_to_chp("F1", "bhkw")
cfg.auto_connect_digester_to_chp("F2", "bhkw")
cfg.auto_connect_digester_to_chp("N1", "bhkw")
cfg.auto_connect_digester_to_chp("G1", "bhkw")

plant.initialize()
