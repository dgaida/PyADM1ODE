# Schnellstart-Anleitung

Diese Anleitung bringt Sie in wenigen Minuten mit PyADM1ODE zum Laufen.

## Inhaltsverzeichnis

1. [Ihre erste Simulation](#ihre-erste-simulation)  
2. [Einfacher Einzelfermenter](#einfacher-einzelfermenter)  
3. [Komplette Anlage mit BHKW und Heizung](#komplette-anlage-mit-bhkw-und-heizung)  
4. [Zweistufige Vergärung](#zweistufige-vergärung)  
5. [Arbeiten mit verschiedenen Substraten](#arbeiten-mit-verschiedenen-substraten)  
6. [Konfigurationen speichern und laden](#konfigurationen-speichern-und-laden)  
7. [Ergebnisse verstehen](#ergebnisse-verstehen)  
8. [Gängige Muster](#gängige-muster)  
9. [Nächste Schritte](#nächste-schritte)  

## Ihre erste Simulation

Simulieren wir einen einfachen einstufigen Biogasfermenter mit Maissilage und Gülle.

```python
from pathlib import Path
from pyadm1.configurator.plant_builder import BiogasPlant
from pyadm1.substrates.feedstock import Feedstock
from pyadm1.core.adm1 import get_state_zero_from_csv
from pyadm1.configurator.plant_configurator import PlantConfigurator

# 1. Feedstock-Manager erstellen
feedstock = Feedstock(feeding_freq=48)  # Fütterung alle 48 Stunden änderbar

# 2. Anfangszustand laden (Steady-State-Werte, 41-State ADM1da-Vektor)
data_path = Path("data/initial_states")
initial_state_file = data_path / "digester_initial8.csv"
adm1_state = get_state_zero_from_csv(str(initial_state_file))

# 3. Substrat-Fütterungsraten definieren [m³/Tag]
# [corn_silage, manure, rye, grass, wheat, gps, ccm, feed_lime, cow_manure, onions]
Q_substrates = [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]

# 4. Anlage erstellen und konfigurieren
plant = BiogasPlant("Meine erste Biogasanlage")
configurator = PlantConfigurator(plant, feedstock)

configurator.add_digester(
    digester_id="main_digester",
    V_liq=2000.0,        # Flüssigvolumen [m³]
    V_gas=300.0,         # Gasraum [m³]
    T_ad=308.15,         # Temperatur [K] = 35 °C
    Q_substrates=Q_substrates
)

# 5. Initialisieren und simulieren
plant.initialize()

results = plant.simulate(
    duration=10.0,       # Simulationszeit [Tage]
    dt=1.0/24.0,        # Zeitschritt [Tage] = 1 Stunde
    save_interval=1.0   # Ergebnisse täglich speichern
)

# 6. Ergebnisse anzeigen
for result in results:
    time = result["time"]
    digester = result["components"]["main_digester"]

    print(f"Tag {time:.0f}:")
    print(f"  Biogas:  {digester['Q_gas']:.1f} m³/d")
    print(f"  Methan:  {digester['Q_ch4']:.1f} m³/d")
    print(f"  pH:      {digester['pH']:.2f}")
    print(f"  VFA:     {digester['VFA']:.2f} g/L")
```

**Erwartete Ausgabe:**

```text
Tag 1:
  Biogas:  1245.3 m³/d
  Methan:  748.2 m³/d
  pH:      7.32
  VFA:     2.45 g/L
...
```

## Einfacher Einzelfermenter

Das vollständige Beispiel finden Sie in [`examples/01_basic_digester.py`](https://github.com/dgaida/PyADM1ODE/blob/master/examples/01_basic_digester.py).

### Systemarchitektur

Der einfache Fermenter besteht aus:

```text
[Substratfütterung] → [Fermenter] → [Gasspeicher] → [Biogasausgang]
   15 m³/d Mais         2000 m³        300 m³         ~1250 m³/d
   10 m³/d Gülle        @ 35 °C        Membran        60 % CH₄
```

### Kernfunktionen

- **Einstufiger CSTR**: Continuously stirred tank reactor  
- **Automatischer Gasspeicher**: Pro Fermenter erstellt (Niederdruckmembran)  
- **Substratmischung**: Maissilage + Rindergülle  
- **Temperaturregelung**: Mesophil (35 °C)  

### Details zur Anlagenkonfiguration

```python
# Die Methode add_digester erzeugt automatisch:
# 1. Fermenter-Komponente mit den angegebenen Parametern
# 2. Gasspeicher (Membrantyp), dimensioniert auf V_gas
# 3. Verbindung: Fermenter → Gasspeicher

configurator.add_digester(
    digester_id="main_digester",
    V_liq=2000.0,               # Flüssigvolumen [m³]
    V_gas=300.0,                # Gasraum [m³]
    T_ad=308.15,                # 35 °C mesophil
    name="Hauptfermenter",
    load_initial_state=True,    # Steady-State-CSV laden
    Q_substrates=[15, 10, 0, 0, 0, 0, 0, 0, 0, 0]
)
```

### Die Ausgabe verstehen

**Verhalten des Gasspeichers:**

```python
# Aus den Ergebnissen:
'gas_storage': {
    'stored_volume_m3': 150.0,      # Aktuelles Volumen [m³ STP]
    'pressure_bar': 1.01,            # Aktueller Druck [bar]
    'vented_volume_m3': 0.0,         # In diesem Schritt entlüftet [m³]
    'utilization': 0.50,             # 50 % gefüllt
    'Q_gas_supplied_m3_per_day': 1250.0  # Verfügbar für Verbraucher
}
```

**Wenn der Speicher gefüllt wird:**

- Druck steigt von 0.95 auf 1.05 bar  
- Bei 1.05 bar (voll) wird Überschussgas entlüftet  
- Das Entlüften verhindert Überdruck und Anlagenschäden  
- In realen Anlagen wird entlüftetes Gas zur Fackel geleitet  

### Prozessstabilitäts-Indikatoren

```python
# Prozessstabilität prüfen
final = results[-1]["components"]["main_digester"]

# pH sollte für stabilen Betrieb bei 6.8–7.5 liegen
if 6.8 <= final['pH'] <= 7.5:
    print("✓ pH stabil")
else:
    print(f"⚠ pH instabil: {final['pH']:.2f}")

# VFA sollte < 5 g/L sein
if final['VFA'] < 5.0:
    print("✓ VFA akzeptabel")
else:
    print(f"⚠ Hohe VFA: {final['VFA']:.2f} g/L")

# FOS/TAC-Verhältnis sollte < 0.3 sein
if final['TAC'] > 0:
    fos_tac = final['VFA'] / final['TAC']
    if fos_tac < 0.3:
        print(f"✓ FOS/TAC stabil: {fos_tac:.3f}")
    else:
        print(f"⚠ FOS/TAC hoch: {fos_tac:.3f}")
```

**Typische Leistungsmetriken:**

| Metrik | Wert | Bewertung |
|--------|------|-----------|
| Biogasproduktion | ~1250 m³/d | Gut |
| Methangehalt | ~60 % | Typisch für landwirtschaftliche Substrate |
| Spezifische Gasausbeute | ~50 m³/m³ Einsatz | Gut für Maissilage + Gülle |
| pH | 7.28–7.30 | Optimal (stabil) |
| VFA | 2.3–2.4 g/L | Gut (< 3 g/L Grenzwert) |
| TAC | 8.4–8.5 g CaCO₃/L | Ausgezeichneter Puffer |
| FOS/TAC | ~0.27 | Stabil (< 0.3) |

## Komplette Anlage mit BHKW und Heizung

Jetzt ergänzen wir Stromerzeugung und Heizung zu einer kompletten Biogasanlage:

```python
from pyadm1.configurator.plant_builder import BiogasPlant
from pyadm1.substrates.feedstock import Feedstock
from pyadm1.configurator.plant_configurator import PlantConfigurator

# Setup
feedstock = Feedstock(feeding_freq=48)
plant = BiogasPlant("Komplette Biogasanlage")
configurator = PlantConfigurator(plant, feedstock)

# Fermenter mit automatischem Gasspeicher hinzufügen
configurator.add_digester(
    digester_id="main_digester",
    V_liq=2000.0,
    V_gas=300.0,
    T_ad=308.15,
    Q_substrates=[15, 10, 0, 0, 0, 0, 0, 0, 0, 0]
)

# BHKW hinzufügen (erstellt automatisch eine Fackel)
configurator.add_chp(
    chp_id="chp_main",
    P_el_nom=500.0,      # Elektrische Leistung [kW]
    eta_el=0.40,         # Elektrischer Wirkungsgrad 40 %
    eta_th=0.45          # Thermischer Wirkungsgrad 45 %
)

# Heizsystem hinzufügen
configurator.add_heating(
    heating_id="heating_main",
    target_temperature=308.15,
    heat_loss_coefficient=0.5
)

# Komponenten automatisch verbinden
configurator.auto_connect_digester_to_chp("main_digester", "chp_main")
configurator.auto_connect_chp_to_heating("chp_main", "heating_main")

# Initialisieren und simulieren
plant.initialize()
results = plant.simulate(duration=10.0, dt=1.0/24.0, save_interval=1.0)

# Endergebnisse analysieren
final = results[-1]
digester = final["components"]["main_digester"]
chp = final["components"]["chp_main"]
heating = final["components"]["heating_main"]

print(f"\nEndergebnisse (Tag {final['time']:.0f}):")
print(f"\nFermenter-Leistung:")
print(f"  Biogas:  {digester['Q_gas']:.1f} m³/d")
print(f"  Methan:  {digester['Q_ch4']:.1f} m³/d")
print(f"  CH4-Gehalt: {digester['Q_ch4']/digester['Q_gas']*100:.1f}%")
print(f"\nBHKW-Leistung:")
print(f"  Elektrische Leistung: {chp['P_el']:.1f} kW")
print(f"  Thermische Leistung:  {chp['P_th']:.1f} kW")
print(f"  Gasverbrauch:         {chp['Q_gas_consumed']:.1f} m³/d")
print(f"\nHeizung:")
print(f"  Bereitgestellte Wärme: {heating['Q_heat_supplied']:.1f} kW")
print(f"  Zusatzwärme:           {heating['P_aux_heat']:.1f} kW")
```

**Automatische Komponentenerstellung:**

PlantConfigurator erstellt und verbindet automatisch:

- **Gasspeicher**: Einer pro Fermenter (Membran, dimensioniert auf V_gas)  
- **Fackel**: Eine pro BHKW (Sicherheitsverbrennung, 98 % CH₄-Zerstörung)  

**Verbindungskette:**

```text
Fermenter → Gasspeicher → BHKW → Fackel
                          ↓
                       Heizung
```

## Zweistufige Vergärung

Das vollständige Beispiel finden Sie in [`examples/02_two_stage_plant.py`](https://github.com/dgaida/PyADM1ODE/blob/master/examples/02_two_stage_plant.py).

### Systemarchitektur

```text
[Einsatz] → [Hydrolyse] → [Speicher 1] ↘
            500 m³         304 m³        → [BHKW] → [Fackel]
            @ 45 °C                     ↗  500 kW    98 %
                                       ↓
[Ablauf]  → [Haupt]  → [Speicher 2] ↗  [Heizung 1] + [Heizung 2]
            1000 m³     150 m³           45 °C         35 °C
            @ 35 °C
```

### Kernfunktionen

- **Temperaturphasiert**: Thermophil (45 °C) + Mesophil (35 °C)  
- **Verstärkte Hydrolyse**: Höhere Temperatur in der ersten Stufe  
- **Stabile Methanogenese**: Optimierte Bedingungen in der zweiten Stufe  
- **Mechanische Komponenten**: Pumpen und Rührwerke für die Materialhandhabung  
- **Energieintegration**: Kraft-Wärme-Kopplung mit Abwärmenutzung  

### Konfiguration

```python
# Stufe 1: Hydrolyse (thermophil, 45 °C)
configurator.add_digester(
    digester_id="hydrolysis_tank",
    V_liq=500.0,
    V_gas=100.0,
    T_ad=318.15,  # 45 °C für schnellere Hydrolyse
    Q_substrates=[15, 10, 0, 0, 0, 0, 0, 0, 0, 0]
)

# Stufe 2: Methanogenese (mesophil, 35 °C)
configurator.add_digester(
    digester_id="main_digester",
    V_liq=2000.0,
    V_gas=300.0,
    T_ad=308.15,  # 35 °C für Methanogenese
    Q_substrates=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Speisung aus Hydrolyse
)

# In Serie verbinden (Flüssigkeitsfluss)
configurator.connect("hydrolysis_tank", "main_digester", "liquid")

# BHKW und Heizung für beide Stufen hinzufügen
configurator.add_chp("chp_main", P_el_nom=500.0)
configurator.add_heating("heating_1", target_temperature=318.15)
configurator.add_heating("heating_2", target_temperature=308.15)

# Gas- und Wärmeflüsse verbinden
configurator.auto_connect_digester_to_chp("hydrolysis_tank", "chp_main")
configurator.auto_connect_digester_to_chp("main_digester", "chp_main")
configurator.auto_connect_chp_to_heating("chp_main", "heating_1")
configurator.auto_connect_chp_to_heating("chp_main", "heating_2")

# Mechanische Komponenten hinzufügen
from pyadm1.components.mechanical import Pump, Mixer

# Förderpumpe
feed_pump = Pump(
    component_id="feed_pump",
    pump_type="progressive_cavity",
    Q_nom=30.0,
    pressure_head=5.0
)
plant.add_component(feed_pump)

# Transferpumpe (Fermenter 1 → Fermenter 2)
transfer_pump = Pump(
    component_id="transfer_pump",
    pump_type="progressive_cavity",
    Q_nom=25.0,
    pressure_head=8.0
)
plant.add_component(transfer_pump)

# Rührwerke für beide Fermenter
mixer_1 = Mixer(
    component_id="mixer_1",
    mixer_type="propeller",
    tank_volume=500.0,
    mixing_intensity="high",
    power_installed=15.0,
    intermittent=True,
    on_time_fraction=0.25
)
plant.add_component(mixer_1)

mixer_2 = Mixer(
    component_id="mixer_2",
    mixer_type="propeller",
    tank_volume=2000.0,
    mixing_intensity="medium",
    power_installed=10.0,
    intermittent=True,
    on_time_fraction=0.25
)
plant.add_component(mixer_2)
```

### Drei-Pass-Gasflusssimulation

Die Simulation verwendet ein Drei-Pass-Ausführungsmodell für ein realistisches Gasmanagement:

**Pass 1 – Gasproduktion:**

```python
Fermenter 1: Q_gas = 850 m³/d → Speicher 1
Fermenter 2: Q_gas = 400 m³/d → Speicher 2
```

**Pass 2 – Speicheraktualisierung:**

```python
Speicher 1: erhält 850 m³/d, aktualisiert Druck/Volumen
Speicher 2: erhält 400 m³/d, aktualisiert Druck/Volumen
# Bei voll: Überschuss in die Atmosphäre entlüften
```

**Pass 3 – Gasverbrauch:**

```python
BHKW-Bedarf: 1150 m³/d Biogas
Speicher 1 liefert: ~675 m³/d
Speicher 2 liefert: ~475 m³/d
BHKW arbeitet mit tatsächlicher Versorgung
Überschuss zur Fackel: (Lieferung - Verbrauch)
```

### Vorteile des zweistufigen Designs

| Aspekt | Einstufig | Zweistufig | Verbesserung |
|--------|-----------|------------|--------------|
| **Hydrolyse** | Begrenzt durch mesophile Temperatur | Verstärkt bei 45 °C | Schneller |
| **Methanogenese** | Muss VFA-Spitzen tolerieren | Stabil, vorgepuffertes Substrat | Stabiler |
| **OLR-Kapazität** | 3–4 kg CSB/(m³·d) | 5–8 kg CSB/(m³·d) | +100 % |
| **Biogasausbeute** | 1150 m³/d | 1253 m³/d | +9 % |
| **CH₄-Gehalt** | 58 % | 60 % | +3.4 % |

### Erwartete Ergebnisse

**Energiebilanz:**

```text
Energieerzeugung:
  Elektrisch (brutto):       480.5 kW
  Thermisch:                 540.6 kW

Parasitäre Last:
  Rührwerk 1:                  3.75 kW
  Rührwerk 2:                  2.50 kW
  Pumpen (geschätzt):          2.00 kW
  Gesamt parasitär:            8.25 kW

Netto-Elektrizität:         472.3 kW

Wärmenutzung:
  Wärmebedarf:               125.4 kW
  BHKW-Wärmebereitstellung:  540.6 kW
  Wärmedeckung:              431.0 %

Gasmanagement:
  Gesamtproduktion:         1253.1 m³/d
  BHKW-Verbrauch:           1150.0 m³/d
  Zur Fackel:                103.1 m³/d (8.2 %)
```

## Arbeiten mit verschiedenen Substraten

### Verfügbare Substrate

PyADM1 enthält 10 vorkonfigurierte landwirtschaftliche Substrate:

1. **Maissilage (maize)** – Energiepflanze, hohe Biogasausbeute  
2. **Schweinegülle (swinemanure)** – Hoher Stickstoffgehalt  
3. **Grünroggen (greenrye)** – Früherntliche Energiepflanze  
4. **Grassilage (grass)** – Grünlandbiomasse  
5. **Weizen (wheat)** – Getreidekultur  
6. **GPS (gps)** – Ganzpflanzensilage  
7. **CCM (ccm)** – Corn-Cob-Mix  
8. **Futterkalk (futterkalk)** – pH-Pufferzusatz  
9. **Rindergülle (cowmanure)** – Gülle aus Milchviehhaltung  
10. **Zwiebeln (onions)** – Gemüseabfall  

### Beispiele für Substratfütterung

```python
# Energiereiche Mischung (Mais + Gülle)
Q = [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]

# Grasbasiert (erneuerbar, extensive Landwirtschaft)
Q = [0, 5, 0, 20, 0, 0, 0, 0, 0, 0]

# Abfallbasiert (Gülle + Gemüse)
Q = [0, 15, 0, 0, 0, 0, 0, 0, 10, 5]

# Fokus auf Energiepflanzen
Q = [20, 5, 10, 0, 0, 0, 0, 0, 0, 0]
```

### Substratinformationen

Detaillierte Substrateigenschaften abrufen:

```python
from pyadm1.substrates.feedstock import Feedstock

# Substratparameter anzeigen
params = Feedstock.get_substrate_params_string("maize")
print(params)
```

Ausgabe:

```text
pH value: 3.93
Dry matter: 31.97 %FM
Volatile solids content: 96.25 %TS
Particulate chemical oxygen demand: ...
Biochemical methane potential: 0.xxx l/gFM
```

## Konfigurationen speichern und laden

Speichern Sie Ihre Anlagenkonfiguration zur Wiederverwendung:

```python
# Konfiguration speichern
plant.to_json("my_plant_config.json")

# Konfiguration später laden
from pyadm1.configurator.plant_builder import BiogasPlant
from pyadm1.substrates.feedstock import Feedstock

feedstock = Feedstock(feeding_freq=48)
plant = BiogasPlant.from_json("my_plant_config.json", feedstock)

# Simulation fortsetzen
plant.initialize()
results = plant.simulate(duration=10.0, dt=1.0/24.0)
```

## Ergebnisse verstehen

### Wichtige Ausgabevariablen

#### Fermenter-Ausgaben

- `Q_gas` – Gesamte Biogasproduktion [m³/d]  
- `Q_ch4` – Methanproduktion [m³/d]  
- `Q_co2` – CO2-Produktion [m³/d]  
- `pH` – pH-Wert [-]  
- `VFA` – Flüchtige Fettsäuren [g HAceq/L]  
- `TAC` – Gesamtalkalinität [g CaCO3/L]  

#### BHKW-Ausgaben

- `P_el` – Elektrische Leistung [kW]  
- `P_th` – Thermische Leistung [kW]  
- `Q_gas_consumed` – Gasverbrauch [m³/d]  
- `load_factor` – Betriebspunkt [0–1]  

#### Heizungsausgaben

- `Q_heat_supplied` – Gelieferte Wärme [kW]  
- `P_th_used` – Genutzte BHKW-Wärme [kW]  
- `P_aux_heat` – Benötigte Zusatzwärme [kW]  

#### Gasspeicher-Ausgaben

- `stored_volume_m3` – Aktuelles Volumen [m³ STP]  
- `pressure_bar` – Aktueller Druck [bar]  
- `utilization` – Füllstand [0–1]  
- `vented_volume_m3` – Entlüftetes Gas [m³]  
- `Q_gas_supplied_m3_per_day` – Verfügbares Gas [m³/d]  

## Gängige Muster

### Muster 1: Parameterstudie

Verschiedene Substratmengen testen:

```python
feed_rates = [10, 15, 20, 25]
results_all = []

for feed in feed_rates:
    Q = [feed, 10, 0, 0, 0, 0, 0, 0, 0, 0]

    plant = BiogasPlant(f"Plant_Feed_{feed}")
    configurator = PlantConfigurator(plant, feedstock)
    configurator.add_digester("dig1", V_liq=2000, Q_substrates=Q)

    plant.initialize()
    results = plant.simulate(duration=10.0, dt=1.0/24.0)

    final = results[-1]["components"]["dig1"]
    results_all.append({
        'feed': feed,
        'biogas': final['Q_gas'],
        'methane': final['Q_ch4']
    })

for r in results_all:
    print(f"Einsatz {r['feed']} m³/d → CH4 {r['methane']:.1f} m³/d")
```

### Muster 2: Zeitreihenanalyse

Entwicklung über die Zeit verfolgen:

```python
import matplotlib.pyplot as plt

# Zeitreihen extrahieren
times = [r['time'] for r in results]
biogas = [r['components']['main_digester']['Q_gas'] for r in results]
methane = [r['components']['main_digester']['Q_ch4'] for r in results]
pH = [r['components']['main_digester']['pH'] for r in results]

# Plotten
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(times, biogas, 'b-', label='Biogas')
ax1.plot(times, methane, 'g-', label='Methan')
ax1.set_ylabel('Produktion [m³/d]')
ax1.legend()
ax1.grid(True)

ax2.plot(times, pH, 'r-')
ax2.set_xlabel('Zeit [Tage]')
ax2.set_ylabel('pH')
ax2.axhline(y=7.0, color='k', linestyle='--', alpha=0.3)
ax2.grid(True)

plt.tight_layout()
plt.savefig('simulation_results.png')
```

## Nächste Schritte

Nachdem Sie Ihre ersten Simulationen ausgeführt haben:

1. **Komponenten kennenlernen**: [Komponenten-Leitfaden](components/index.md)  
2. **Fortgeschrittene Beispiele erkunden**:  
   - [`examples/01_basic_digester.py`](https://github.com/dgaida/PyADM1ODE/blob/master/examples/01_basic_digester.py)  
   - [`examples/02_two_stage_plant.py`](https://github.com/dgaida/PyADM1ODE/blob/master/examples/02_two_stage_plant.py)  
   - [`examples/parallel_two_stage_simulation.py`](https://github.com/dgaida/PyADM1ODE/blob/master/examples/parallel_two_stage_simulation.py)  
3. **Parallele Simulationen ausprobieren**: Mehrere Szenarien gleichzeitig  
4. **MCP-Server erkunden**: [PyADM1ODE_mcp](https://github.com/dgaida/PyADM1ODE_mcp) für LLM-gesteuerte Anlagenauslegung  
5. **Modell kalibrieren**: [PyADM1ODE_calibration](https://github.com/dgaida/PyADM1ODE_calibration) für Parameteranpassung  
6. **API-Dokumentation lesen**: Vollständige Referenz für alle Klassen  

## Schnellreferenz

### Häufige Befehle

```python
# Anlage erstellen
plant = BiogasPlant("Meine Anlage")
configurator = PlantConfigurator(plant, feedstock)

# Komponenten hinzufügen
configurator.add_digester(id, V_liq, V_gas, T_ad, Q_substrates)
configurator.add_chp(id, P_el_nom, eta_el, eta_th)
configurator.add_heating(id, target_temperature)

# Verbinden
configurator.connect(from_id, to_id, type)
configurator.auto_connect_digester_to_chp(dig_id, chp_id)
configurator.auto_connect_chp_to_heating(chp_id, heat_id)

# Simulieren
plant.initialize()
results = plant.simulate(duration, dt, save_interval)

# Speichern/Laden
plant.to_json(filepath)
plant = BiogasPlant.from_json(filepath, feedstock)
```

### Temperaturumrechnungen

```python
# Gängige Temperaturen
T_mesophil = 308.15     # 35 °C
T_thermophil = 328.15   # 55 °C
T_psychrophil = 298.15  # 25 °C

# °C in K umrechnen
T_K = T_celsius + 273.15
```

### Standardparameter

```python
# Typische Fermentergrößen
V_liq_small = 500      # Kleiner Hof [m³]
V_liq_medium = 2000    # Mittlerer Hof [m³]
V_liq_large = 5000     # Großer Hof [m³]

# BHKW-Größen
P_el_small = 150       # Klein [kW]
P_el_medium = 500      # Mittel [kW]
P_el_large = 1000      # Groß [kW]

# Substratfütterung
Q_low = 10             # Niedrige Belastung [m³/d]
Q_medium = 20          # Mittlere Belastung [m³/d]
Q_high = 40            # Hohe Belastung [m³/d]
```

## Fehlerbehebung

### Problem: Simulation instabil

**Symptome**: pH fällt, VFA steigt, Methanproduktion sinkt

**Lösungen**:

- Substratfütterungsrate reduzieren  
- Verweilzeit erhöhen (größeres V_liq)  
- Puffermaterial hinzufügen (Futterkalk)  
- Substratzusammensetzung prüfen  

### Problem: Geringe Gasproduktion

**Lösungen**:

- Substratfütterung erhöhen  
- Abbaubarkeit des Substrats prüfen  
- Optimale Temperatur sicherstellen  
- Ausreichende Durchmischung gewährleisten (im Modell implizit)  

### Problem: Langsame Simulation

**Lösungen**:

- Zeitschritt `dt` erhöhen (aber < 0.1 Tage halten)  
- `save_interval` für weniger Ausgabe reduzieren  
- Parallele Simulation für Parameterstudien verwenden  

Weitere Hilfe finden Sie in der [Installationsanleitung](installation.md) oder kontaktieren Sie <daniel.gaida@th-koeln.de>.

## Referenzen

- **ADM1-Modell**: Batstone et al. (2002). *Anaerobic Digestion Model No. 1*. IWA Publishing.  
- **Leitfaden Biogas**: FNR (2016). <https://mediathek.fnr.de/leitfaden-biogas.html>  
