from pyadm1 import BiogasPlant, Feedstock
from pyadm1.configurator.plant_configurator import PlantConfigurator

feedstock = Feedstock(["cattle_manure"], feeding_freq=24, total_simtime=30)
plant = BiogasPlant("BGA3")
cfg = PlantConfigurator(plant, feedstock)

cfg.add_digester("F1", V_liq=2867, V_gas=400, T_ad=313.15, name="Fermenter 1")
cfg.add_digester("F2", V_liq=2867, V_gas=400, T_ad=313.15, name="Fermenter 2")
cfg.add_digester("N1", V_liq=2867, V_gas=400, T_ad=313.15, name="Nachgaerer")
cfg.add_digester("G1", V_liq=3325, V_gas=430, T_ad=293.15, name="Gaerrestlager")

cfg.add_chp("bhkw", P_el_nom=500.0, eta_el=0.40, eta_th=0.45, name="BHKW 500 kW")

cfg.connect("F1", "N1", "liquid")
cfg.connect("F2", "N1", "liquid")
cfg.connect("N1", "G1", "liquid")

cfg.auto_connect_digester_to_chp("F1", "bhkw")
cfg.auto_connect_digester_to_chp("F2", "bhkw")
cfg.auto_connect_digester_to_chp("N1", "bhkw")
cfg.auto_connect_digester_to_chp("G1", "bhkw")

plant.initialize()
