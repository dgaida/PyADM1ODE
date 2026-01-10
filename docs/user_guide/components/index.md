# Components Guide

PyADM1 verwendet eine modulare, komponentenbasierte Architektur. Dieser Leitfaden behandelt alle verfügbaren Komponenten, ihre Parameter und Verwendungsmuster.

## Komponentenarchitektur

### Basis-Komponentenstruktur

Alle Komponenten erben von der `Component` Basisklasse und implementieren:

```python
class Component(ABC):
    def __init__(self, component_id, component_type, name):
        """Initialisiere Komponente mit eindeutiger ID und Typ."""

    def initialize(self, initial_state):
        """Setze initialen Zustand vor der Simulation."""

    def step(self, t, dt, inputs):
        """Führe einen Simulationszeitschritt aus."""

    def to_dict(self):
        """Serialisiere zu Dictionary für JSON-Export."""

    @classmethod
    def from_dict(cls, config):
        """Erstelle Komponente aus Konfigurations-Dictionary."""
```

### Komponentenlebenszyklus

```
Erstellen → Initialisieren → Simulieren (step-Schleife) → Speichern/Exportieren
   ↓                            ↑
   └────────────────────────────┘
        (kann reinitialisiert werden)
```

## Komponentenübersicht

PyADM1 bietet verschiedene Komponentenkategorien:

### [Biologische Komponenten](biological.md)

Komponenten für biologische Umwandlungsprozesse:

- **Digester**: Hauptfermenter mit ADM1-Modell für anaerobe Vergärung
- **Hydrolysis**: Vorbehandlungstank für Hydrolyseprozesse
- **Separator**: Fest-Flüssig-Trennung für Gärrestaufbereitung

### [Energiekomponenten](energy.md)

Komponenten für Energieerzeugung und -speicherung:

- **CHP**: Blockheizkraftwerk zur Strom- und Wärmeerzeugung
- **Heating**: Heizsystem zur Temperaturkontrolle
- **GasStorage**: Biogasspeicher mit Druckmanagement
- **Flare**: Sicherheitsfackel für Überschussgas

### [Mechanische Komponenten](mechanical.md)

Mechanische Anlagenkomponenten:

- **Pump**: Pumpen für Substratförderung und Rezirkulation
- **Mixer**: Rührwerke zur Homogenisierung im Fermenter

### [Fütterungskomponenten](feeding.md)

Substrathandhabung und -dosierung:

- **SubstrateStorage**: Substratlagerbehälter mit Qualitätsverfolgung
- **Feeder**: Automatische Dosiersysteme

### [Sensoren](sensors.md)

Mess- und Überwachungskomponenten (in Entwicklung)

## Verbindungstypen

### Flüssigkeitsverbindungen

Übertragen Gärrest zwischen Fermentern:

```python
configurator.connect("digester_1", "digester_2", "liquid")
```

**Datenübertragung:**
- `Q_out`: Flüssigkeitsdurchfluss [m³/d]
- `state_out`: Vollständiger ADM1-Zustandsvektor

### Gasverbindungen

Übertragen Biogas vom Speicher zum BHKW:

```python
configurator.connect("digester_1_storage", "chp_1", "gas")
```

**Datenübertragung:**
- `Q_gas_supplied_m3_per_day`: Verfügbares Gas [m³/d]
- Gaszusammensetzung (CH4%, CO2%)

### Wärmeverbindungen

Übertragen Abwärme vom BHKW zur Heizung:

```python
configurator.connect("chp_1", "heating_1", "heat")
```

**Datenübertragung:**
- `P_th`: Verfügbare thermische Leistung [kW]
- Temperaturniveaus

### Auto-Verbindungshelfer

```python
# Automatisches Gas-Routing: Fermenter → Speicher → BHKW → Fackel
configurator.auto_connect_digester_to_chp("dig1", "chp1")

# Automatisches Wärme-Routing: BHKW → Heizung
configurator.auto_connect_chp_to_heating("chp1", "heat1")
```

## Komponentenmuster

### Muster 1: Einstufige Anlage

```python
configurator.add_digester("dig1", V_liq=2000, Q_substrates=[15,10,0,0,0,0,0,0,0,0])
configurator.add_chp("chp1", P_el_nom=500)
configurator.add_heating("heat1", target_temperature=308.15)

configurator.auto_connect_digester_to_chp("dig1", "chp1")
configurator.auto_connect_chp_to_heating("chp1", "heat1")
```

**Topologie:**
```
[Fermenter] → [Gasspeicher] → [BHKW] → [Fackel]
                                 ↓
                             [Heizung]
```

### Muster 2: Zweistufige Reihe

```python
# Stufe 1: Hydrolyse (thermophil)
configurator.add_digester("hydro", V_liq=500, T_ad=318.15,
                         Q_substrates=[15,10,0,0,0,0,0,0,0,0])

# Stufe 2: Methanogenese (mesophil)
configurator.add_digester("main", V_liq=2000, T_ad=308.15,
                         Q_substrates=[0,0,0,0,0,0,0,0,0,0])

# Verbinde Flüssigkeitsstrom
configurator.connect("hydro", "main", "liquid")

# Einzelnes BHKW für beide
configurator.add_chp("chp1", P_el_nom=500)
configurator.auto_connect_digester_to_chp("hydro", "chp1")
configurator.auto_connect_digester_to_chp("main", "chp1")

# Separate Heizung für jede Stufe
configurator.add_heating("heat1", target_temperature=318.15)
configurator.add_heating("heat2", target_temperature=308.15)
configurator.auto_connect_chp_to_heating("chp1", "heat1")
configurator.auto_connect_chp_to_heating("chp1", "heat2")
```

**Topologie:**
```
[Hydrolyse] → [Speicher] ↘
                          → [BHKW] → [Heizung 1]
[Haupt] → [Speicher] ↗         ↓
                            [Heizung 2]
```

### Muster 3: Parallele Fermenter

```python
# Mehrere Fermenter speisen ein BHKW
for i in range(3):
    configurator.add_digester(
        f"dig{i+1}",
        V_liq=1000,
        Q_substrates=[10, 5, 0, 0, 0, 0, 0, 0, 0, 0]
    )

configurator.add_chp("chp1", P_el_nom=1000)

for i in range(3):
    configurator.auto_connect_digester_to_chp(f"dig{i+1}", "chp1")
```

## Vollständiges Integrationsbeispiel

### Komplette Fütterungskette

```python
from pyadm1.configurator import BiogasPlant, PlantConfigurator
from pyadm1.components.feeding import SubstrateStorage, Feeder
from pyadm1.components.mechanical import Pump, Mixer
from pyadm1.substrates import Feedstock

# Setup
feedstock = Feedstock(feeding_freq=48)
plant = BiogasPlant("Komplette Anlage")
config = PlantConfigurator(plant, feedstock)

# 1. Substratspeicher
storage = SubstrateStorage(
    "silo1",
    storage_type="vertical_silo",
    substrate_type="corn_silage",
    capacity=1000,
    initial_level=800
)
plant.add_component(storage)

# 2. Dosierer
feeder = Feeder(
    "feed1",
    feeder_type="screw",
    Q_max=20.0,
    substrate_type="solid"
)
plant.add_component(feeder)

# 3. Förderpumpe
pump = Pump(
    "pump1",
    pump_type="progressive_cavity",
    Q_nom=15.0,
    pressure_head=50.0
)
plant.add_component(pump)

# 4. Fermenter
digester, _ = config.add_digester(
    "main_digester",
    V_liq=2000,
    Q_substrates=[15, 10, 0, 0, 0, 0, 0, 0, 0, 0]
)

# 5. Rührwerk
mixer = Mixer(
    "mix1",
    mixer_type="propeller",
    tank_volume=2000,
    mixing_intensity="medium",
    intermittent=True,
    on_time_fraction=0.25
)
plant.add_component(mixer)

# 6. BHKW und Heizung
config.add_chp("chp1", P_el_nom=500)
config.add_heating("heat1", target_temperature=308.15)

# Komponenten verbinden
config.connect("silo1", "feed1", "default")
config.connect("feed1", "pump1", "default")
config.connect("pump1", "main_digester", "liquid")
config.auto_connect_digester_to_chp("main_digester", "chp1")
config.auto_connect_chp_to_heating("chp1", "heat1")

# Initialisieren und simulieren
plant.initialize()
results = plant.simulate(duration=30, dt=1/24, save_interval=1.0)

# Ergebnisse analysieren
final = results[-1]
print("\nEndergebnisse:")
print(f"Speicherstand: {final['components']['silo1']['current_level']:.1f} t")
print(f"Dosierer-Durchsatz: {final['components']['feed1']['total_mass_fed']:.1f} t")
print(f"Pumpenenergie: {final['components']['pump1']['energy_consumed']:.1f} kWh")
print(f"Rührwerkenergie: {final['components']['mix1']['energy_consumed']:.1f} kWh")
print(f"Biogas: {final['components']['main_digester']['Q_gas']:.1f} m³/d")
```

### Energieanalyse

```python
def calculate_parasitic_load(results):
    """Berechne gesamten parasitären Energieverbrauch"""
    final = results[-1]
    components = final['components']

    # Mechanische Komponenten
    pump_energy = components.get('pump1', {}).get('energy_consumed', 0)
    mixer_energy = components.get('mix1', {}).get('energy_consumed', 0)
    feeder_power = components.get('feed1', {}).get('P_consumed', 0)

    # BHKW-Produktion
    chp_energy = components.get('chp1', {}).get('P_el', 0) * 30 * 24  # kWh

    parasitic_total = pump_energy + mixer_energy
    parasitic_fraction = parasitic_total / chp_energy

    return {
        'pump_energy': pump_energy,
        'mixer_energy': mixer_energy,
        'total_parasitic': parasitic_total,
        'chp_production': chp_energy,
        'parasitic_fraction': parasitic_fraction,
        'net_energy': chp_energy - parasitic_total
    }

analysis = calculate_parasitic_load(results)
print(f"\nEnergieanalyse:")
print(f"BHKW-Produktion: {analysis['chp_production']:.0f} kWh")
print(f"Pumpenverbrauch: {analysis['pump_energy']:.0f} kWh")
print(f"Rührwerkverbrauch: {analysis['mixer_energy']:.0f} kWh")
print(f"Parasitäre Last: {analysis['parasitic_fraction']:.1%}")
print(f"Nettoproduktion: {analysis['net_energy']:.0f} kWh")
```

## Fehlerbehebung

### Häufige Probleme

**Problem**: Pumpe liefert keinen Durchfluss

**Lösung**: Prüfe Druckhöhe und Drehzahleinstellungen
```python
result = pump.step(0, 1/24, {
    'Q_setpoint': 15.0,
    'enable_pump': True,
    'pressure_head': 50.0  # Stelle ausreichende Druckhöhe sicher
})

if result['Q_actual'] < 0.5 * result['Q_setpoint']:
    print("Prüfe: Druckhöhe, Blockaden, Stromversorgung")
```

**Problem**: Rührwerk verbraucht zu viel Energie

**Lösung**: Nutze intermittierenden Betrieb
```python
# Anstelle von kontinuierlich (360 kWh/Tag):
mixer_continuous = Mixer("mix1", intermittent=False)

# Nutze intermittierend (90 kWh/Tag):
mixer_optimal = Mixer(
    "mix1",
    intermittent=True,
    on_time_fraction=0.25  # 75% Energieeinsparung
)
```

**Problem**: Dosierer-Genauigkeit zu niedrig

**Lösung**: Nutze präziseren Dosierertyp oder deaktiviere Rauschen
```python
# Weniger präzise: Schnecke (±5%)
feeder_screw = Feeder("feed1", feeder_type="screw")

# Präziser: Kolben (±1%)
feeder_piston = Feeder("feed1", feeder_type="piston")

# Oder deaktiviere realistisches Rauschen für idealisierte Simulation
feeder_ideal = Feeder(
    "feed1",
    feeder_type="screw",
    enable_dosing_noise=False
)
```

**Problem**: Lagerqualität verschlechtert sich zu schnell

**Lösung**: Prüfe Temperatur und Lagertyp
```python
# Schlecht: Miete bei 20°C
storage_poor = SubstrateStorage(
    "clamp1",
    storage_type="clamp",        # Hohe Degradation
    temperature=293.15           # Warm
)
# Degradation: ~0.003/d → 91% Qualität nach 30 Tagen

# Besser: Silo bei 15°C
storage_good = SubstrateStorage(
    "silo1",
    storage_type="vertical_silo", # Niedrige Degradation
    temperature=288.15            # Kühl
)
# Degradation: ~0.0005/d → 98.5% Qualität nach 30 Tagen
```

## Komponenten-Übersichtstabelle

| Komponente | Zweck | Hauptparameter | Typische Leistung | Hinweise |
|-----------|---------|----------------|---------------|-------|
| **Pump** | Materialtransfer | Q_nom, pressure_head | 2-10 kW | Dimensionierung für 80-90% max. Durchfluss |
| **Mixer** | Homogenisierung | mixing_intensity, on_time | 5-20 kW | Nutze intermittierend (25% Einschaltzeit) |
| **Storage** | Substratlagerung | capacity, storage_type | 0 kW | Überwache Qualitätsverschlechterung |
| **Feeder** | Dosierung | Q_max, feeder_type | 1-5 kW | Aktiviere Dosierrauschen für Realismus |

## Nächste Schritte

- **Beispiele**: Siehe detaillierte Komponentenguides für vollständige Implementierungen
- **Optimierung**: Nutze Parameterstudien zur Optimierung der Komponentendimensionierung
- **[API-Referenz](../api_reference/components/components.md)**: Siehe detaillierte Klassendokumentation für erweiterte Funktionen

## Weiterführende Dokumentation

- [Biologische Komponenten](biological.md)
- [Energiekomponenten](energy.md)
- [Mechanische Komponenten](mechanical.md)
- [Fütterungskomponenten](feeding.md)
- [Sensoren](sensors.md)
