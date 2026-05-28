# API für LLM

Diese Seite bietet eine strukturierte Referenz der Methoden und Klassen, die benötigt werden, um ein PyADM1ODE-Simulationsmodell automatisch zu erstellen. Diese Dokumentation ist darauf optimiert, von Large Language Models (LLMs) gelesen zu werden, um Biogasanlagenkonfigurationen zu generieren.

!!! abstract "Skill für LLMs"
    Sie können die vollständige API-Dokumentation für LLMs als "Skill"-Datei hier herunterladen: [Skill.md](Skill.md) (wird während des Dokumentations-Builds dynamisch generiert)

## Kern-Workflow

Um eine Simulation zu erstellen, folgen Sie diesem Ablauf:  
1. Erstellen Sie ein `Feedstock`-Objekt.  
2. Erstellen Sie ein `BiogasPlant`-Objekt.  
3. Verwenden Sie den `PlantConfigurator`, um Komponenten hinzuzufügen und zu verbinden.  
4. Initialisieren Sie die Anlage und starten Sie die Simulation.  

## 1. Substrat-Konfiguration (Feedstock)

Das `Feedstock`-Objekt definiert, welche Substrate in der Anlage verwendet werden.

```python
from pyadm1.substrates import Feedstock

# Erstellt ein Feedstock mit Maissilage und Schweinegülle
feedstock = Feedstock(
    substrates=["maize_silage_milk_ripeness", "swine_manure"],
    feeding_freq=24,   # Fütterungsfrequenz pro Tag
    total_simtime=30   # Gesamte Simulationsdauer in Tagen
)
```

## 2. Anlagen-Basis (BiogasPlant)

```python
from pyadm1.configurator import BiogasPlant

plant = BiogasPlant("Name der Anlage")
```

## 3. Anlagen-Konfigurator (PlantConfigurator)

Der `PlantConfigurator` ist das Hauptwerkzeug zum Aufbau der Topologie.

```python
from pyadm1.configurator import PlantConfigurator

configurator = PlantConfigurator(plant, feedstock)
```

### Fermenter hinzufügen

```python
# Fügt einen Fermenter hinzu. Erstellt automatisch einen Gasspeicher.
# Q_substrates gibt die Menge der im Feedstock definierten Substrate in [m³/d] an.
digester, state_info = configurator.add_digester(
    digester_id="main_digester",
    V_liq=2000.0,              # Flüssigkeitsvolumen [m³]
    V_gas=300.0,               # Gasraum [m³]
    T_ad=308.15,               # Temperatur [K] (35°C = 308.15K)
    Q_substrates=[15.0, 10.0]  # Mengen korrespondierend zur Feedstock-Liste
)
```

### Energie-Komponenten

```python
# Blockheizkraftwerk (BHKW) hinzufügen
configurator.add_chp(
    chp_id="chp_1",
    P_el_nom=500.0  # Nominale elektrische Leistung [kW]
)

# Heizung hinzufügen
configurator.add_heating(
    heating_id="heating_1",
    target_temperature=308.15  # Zieltemperatur [K]
)
```

### Mechanische Komponenten

Mechanische Komponenten müssen direkt zur `plant` hinzugefügt werden:

```python
from pyadm1.components.mechanical import Pump, Mixer
from pyadm1.components.feeding import SubstrateStorage, Feeder

# Pumpe
pump = Pump("pump1", pump_type="progressive_cavity", Q_nom=15.0)
plant.add_component(pump)

# Rührwerk
mixer = Mixer("mix1", tank_volume=2000.0, intermittent=True, on_time_fraction=0.25)
plant.add_component(mixer)

# Substratlager
storage = SubstrateStorage("silo1", storage_type="vertical_silo", capacity=1000.0)
plant.add_component(storage)

# Dosierer
feeder = Feeder("feed1", feeder_type="screw", Q_max=20.0)
plant.add_component(feeder)
```

### Verbindungen herstellen

```python
# Manuelle Verbindung
configurator.connect("source_id", "target_id", connection_type="liquid") # oder "gas", "heat", "default"

# Automatische Verbindungs-Helfer (empfohlen)
configurator.auto_connect_digester_to_chp("main_digester", "chp_1")
configurator.auto_connect_chp_to_heating("chp_1", "heating_1")
```

## 4. Simulation ausführen

```python
plant.initialize()
results = plant.simulate(
    duration=30,      # Dauer in Tagen
    dt=1/24,          # Zeitschritt (z.B. stündlich)
    save_interval=1.0 # Intervall für Ergebnis-Snapshots
)
```

## Zusammenfassung für LLM-Prompts

Verwenden Sie diese Klassen für den Aufbau:  
- **Biologische Stufe**: `configurator.add_digester()`  
- **Energie**: `configurator.add_chp()`, `configurator.add_heating()`  
- **Mechanik**: `Pump`, `Mixer`  
- **Logistik**: `SubstrateStorage`, `Feeder`  
- **Infrastruktur**: `PlantConfigurator.connect()`  
