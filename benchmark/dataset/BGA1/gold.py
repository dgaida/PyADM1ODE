# benchmark/dataset/BGA1/gold.py
"""
Referenz-Loesung ("Gold") fuer den Datenpunkt BGA1 (siehe BGA1.json) — als
ausfuehrbarer PyADM1ODE-Code. Dient dem Runner als Kandidat, der ~100 % erreicht.

Konvention fuer Kandidaten-Code: am Ende existiert die Variable ``plant``
(eine ``BiogasPlant``).
"""

from pyadm1 import BiogasPlant, Feedstock
from pyadm1.configurator.plant_configurator import PlantConfigurator
from pyadm1.components.biological.separator import Separator

# Substrate werden nicht bewertet — irgendeine gueltige Feedstock genuegt.
feedstock = Feedstock(["maize_silage_milk_ripeness", "cattle_manure"], feeding_freq=24, total_simtime=30)

plant = BiogasPlant("BGA1")
cfg = PlantConfigurator(plant, feedstock)

cfg.add_digester("F1", V_liq=3325, V_gas=500, T_ad=313.15, name="Fermenter 1")
cfg.add_digester("F2", V_liq=3325, V_gas=500, T_ad=313.15, name="Fermenter 2")
cfg.add_digester("N1", V_liq=3325, V_gas=500, T_ad=313.15, name="Nachgaerer")
cfg.add_digester("G1", V_liq=3817, V_gas=570, T_ad=293.15, name="Gaerrestlager")

# BHKW (erfragt, ~500 kW) -> erzeugt automatisch chp_flare (= Fackel)
cfg.add_chp("chp", P_el_nom=500.0, eta_el=0.40, eta_th=0.45, name="BHKW 500 kW")

# Heizungen (erfragt) fuer die mesophilen Stufen
cfg.add_heating("heating_F1", target_temperature=313.15, name="Heizung F1")
cfg.add_heating("heating_F2", target_temperature=313.15, name="Heizung F2")
cfg.add_heating("heating_N1", target_temperature=313.15, name="Heizung N1")

# Separator manuell (kein Helper)
plant.add_component(Separator("sep", separator_type="screw_press", name="Separator"))

# Liquid-Kaskade + Separator
cfg.connect("F1", "N1", "liquid")
cfg.connect("F2", "N1", "liquid")
cfg.connect("N1", "G1", "liquid")
cfg.connect("N1", "sep", "liquid")

# Gas -> BHKW: jede GasStorage braucht einen Abnehmer, damit das Gas bei
# Bedarf weitergeleitet wird (auch das Restgas des Lagers G1).
cfg.auto_connect_digester_to_chp("F1", "chp")
cfg.auto_connect_digester_to_chp("F2", "chp")
cfg.auto_connect_digester_to_chp("N1", "chp")
cfg.auto_connect_digester_to_chp("G1", "chp")

# Heizverbindungen BHKW -> Heizungen
cfg.auto_connect_chp_to_heating("chp", "heating_F1")
cfg.auto_connect_chp_to_heating("chp", "heating_F2")
cfg.auto_connect_chp_to_heating("chp", "heating_N1")

plant.initialize()
