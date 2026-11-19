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
from pyadm1.configurator.plant_builder import BiogasPlant
from pyadm1.substrates.feedstock import Feedstock
from pyadm1.configurator.plant_configurator import PlantConfigurator

# 1. Feedstock-Manager erstellen (lädt die mitgelieferte Substratbibliothek)
feedstock = Feedstock(feeding_freq=48)  # Zeit zwischen Fütterungen [Stunden]

# 2. Substrat-Fütterungsraten definieren [t FM/d], ein Eintrag je Substrat-Slot
# Slot 0 ist Maissilage, Slot 1 ist Rindergülle (siehe "Verfügbare Substrate")
Q_substrates = [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]

# 3. Anlage erstellen und konfigurieren
plant = BiogasPlant("Meine erste Biogasanlage")
configurator = PlantConfigurator(plant, feedstock)

configurator.add_digester(
    digester_id="main_digester",
    V_liq=2000.0,        # Flüssigvolumen [m³]
    V_gas=300.0,         # Gasraum [m³]
    T_ad=308.15,         # Temperatur [K] = 35 °C
    Q_substrates=Q_substrates
)

# 4. Initialisieren und simulieren
plant.initialize()

results = plant.simulate(
    duration=10.0,       # Simulationszeit [Tage]
    dt=1.0/24.0,        # Zeitschritt [Tage] = 1 Stunde
    save_interval=1.0   # Ergebnisse täglich speichern
)

# 5. Ergebnisse anzeigen
for result in results:
    time = result["time"]
    digester = result["components"]["main_digester"]

    print(f"Tag {time:.0f}:")
    print(f"  Biogas:  {digester['Q_gas']:.1f} Nm³/d")
    print(f"  Methan:  {digester['Q_ch4']:.1f} Nm³/d")
    print(f"  pH:      {digester['pH']:.2f}")
    print(f"  VFA:     {digester['VFA']:.2f} g/L")
```

**Erwartete Ausgabe:**

```text
Simulated 100/240 steps
Simulated 200/240 steps
Tag 0:
  Biogas:  4601.8 Nm³/d
  Methan:  992.4 Nm³/d
  pH:      7.16
  VFA:     0.05 g/L
Tag 1:
  Biogas:  2559.6 Nm³/d
  Methan:  1247.4 Nm³/d
  pH:      7.34
  VFA:     0.15 g/L
...
Tag 10:
  Biogas:  2561.9 Nm³/d
  Methan:  1349.9 Nm³/d
  pH:      7.37
  VFA:     0.20 g/L
```

Die beiden Zeilen `Simulated ...` sind Fortschrittsmeldungen von `simulate()`.
`Tag 0` ist das Ergebnis nach dem ersten Stundenschritt, in dem sich der
Gasstrom noch vom automatisch erzeugten Anfangszustand einschwingt.

Die Fütterungsraten sind Tonnen Frischmasse pro Tag (t FM/d). Gasströme sind
trockenes Gas im Normzustand (Nm³/d bei 0 °C und 1,01325 bar), und der
Methangehalt ist der des trockenen Gases: 1349,9 / 2561,9 = 52,7 % an Tag 10.

## Einfacher Einzelfermenter

Ein vollständiges Skript für einen Einzelfermenter ist [`examples/01_basic_digester.py`](https://github.com/dgaida/PyADM1ODE/blob/master/examples/01_basic_digester.py); es verwendet eine eigene Fermentergröße und Substratmischung.

### Systemarchitektur

Der einfache Fermenter besteht aus:

```text
[Substratfütterung] → [Fermenter] → [Gasspeicher] → [Biogasausgang]
   15 t/d Mais          2000 m³        300 m³         ~2560 Nm³/d
   10 t/d Gülle         @ 35 °C        Membran        ~53 % CH₄
```

### Kernfunktionen

- **Einstufiger CSTR**: Continuously stirred tank reactor  
- **Automatischer Gasspeicher**: Pro Fermenter erstellt (Niederdruckmembran)  
- **Substratmischung**: Maissilage + Rindergülle  
- **Temperaturregelung**: Mesophil (35 °C)  

### Details zur Anlagenkonfiguration

Der Fermenter-Aufruf der ersten Simulation, auf einer neuen Anlage und mit einem lesbaren Namen:

```python
plant = BiogasPlant("Meine erste Biogasanlage")
configurator = PlantConfigurator(plant, feedstock)

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
    Q_substrates=[15, 10, 0, 0, 0, 0, 0, 0, 0, 0]
)

print(plant.get_summary())
```

Ausgabe:

```text
=== Meine erste Biogasanlage ===
Simulation time: 0.00 days

Components (2):
  - Hauptfermenter (digester)  
  - Hauptfermenter Gas Storage (storage)  

Connections (1):
  - Hauptfermenter -> Hauptfermenter Gas Storage (gas)  
```

Der Fermenter hat keinen internen Gasspeicher: Der Speicher ist eine eigene
Komponente mit der ID `<digester_id>_storage`, hier `main_digester_storage`.

Der Anfangszustand des Fermenters wird automatisch aus dem Feedstock und den
Fütterungsraten erzeugt. Für einen eigenen Startzustand übergeben Sie einen
Zustandsvektor mit 41 Elementen als `adm1_state`, zum Beispiel einen mit
`pyadm1.core.adm1.get_state_zero_from_csv` geladenen.

### Die Ausgabe verstehen

**Verhalten des Gasspeichers:**

```python
# results[-1]["components"]["main_digester_storage"] der ersten Simulation
# an Tag 10 (Werte gerundet):
{
    'stored_volume_m3': 300.0,           # Aktuelles Volumen [m³ STP]
    'pressure_bar': 1.05,                # Aktueller Druck [bar]
    'utilization': 1.0,                  # Füllstand (1.0 = voll)
    'vented_volume_m3': 106.7,           # In diesem Schritt entlüftet [m³]
    'cumulative_vented_m3': 25513.2,     # Seit dem Start entlüftet [m³]
    'Q_gas_supplied_m3_per_day': 0.0,    # An Verbraucher geliefert [m³/d]
    'Q_ch4_supplied_m3_per_day': 0.0,    # Methan in dieser Lieferung [m³/d]
    'CH4_fraction': 0.527,               # CH4-Anteil des gespeicherten Gases
    'CO2_fraction': 0.473,               # CO2-Anteil des gespeicherten Gases
    'pressure_setpoint_bar': None        # Drucksollwert, falls gesetzt
}
```

Die erste Simulation hat keinen Gasverbraucher. Der Speicher ist deshalb
innerhalb des ersten Tages voll, und die gesamte weitere Produktion wird
entlüftet:

- Der Druck steigt mit dem Füllstand von 1,013 bar (leer) auf 1,05 bar (voll)  
- Gas über der Kapazität wird entlüftet (`vented_volume_m3` je Schritt, `cumulative_vented_m3` insgesamt)  
- Ohne Fackel entweicht das entlüftete Gas unverbrannt  
- Mit einem angeschlossenen BHKW wird das entlüftete Gas in der Fackel des BHKW verbrannt (nächster Abschnitt)  

### Prozessstabilitäts-Indikatoren

pH, VFA und TAC gehören zu den Fermenter-Ergebnissen. Die Grenzwerte in dieser
Prüfung sind gängige Faustregeln, keine von PyADM1ODE festgelegten Werte;
passen Sie sie an Ihre Anlage an.

```python
# Prozessstabilität am Ende der ersten Simulation prüfen
final = results[-1]["components"]["main_digester"]

# pH zwischen 6.8 und 7.5
if 6.8 <= final['pH'] <= 7.5:
    print("✓ pH stabil")
else:
    print(f"⚠ pH instabil: {final['pH']:.2f}")

# VFA unter 5 g/L
if final['VFA'] < 5.0:
    print("✓ VFA akzeptabel")
else:
    print(f"⚠ Hohe VFA: {final['VFA']:.2f} g/L")

# FOS/TAC-Verhältnis (VFA / TAC) unter 0.3
if final['TAC'] > 0:
    fos_tac = final['VFA'] / final['TAC']
    if fos_tac < 0.3:
        print(f"✓ FOS/TAC stabil: {fos_tac:.3f}")
    else:
        print(f"⚠ FOS/TAC hoch: {fos_tac:.3f}")
```

Ausgabe:

```text
✓ pH stabil
✓ VFA akzeptabel
✓ FOS/TAC stabil: 0.036
```

**Leistungskennzahlen der ersten Simulation (Tag 10):**

| Metrik | Wert |
|--------|------|
| Biogasproduktion | 2561,9 Nm³/d |
| Methanproduktion | 1349,9 Nm³/d |
| Methangehalt | 52,7 % |
| Spezifische Gasausbeute | 102,5 Nm³/t FM |
| pH | 7,37 |
| VFA | 0,20 g/L |
| TAC | 5,57 g CaCO₃/L |
| FOS/TAC | 0,036 |

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
print(f"  Biogas:  {digester['Q_gas']:.1f} Nm³/d")
print(f"  Methan:  {digester['Q_ch4']:.1f} Nm³/d")
print(f"  CH4-Gehalt: {digester['Q_ch4']/digester['Q_gas']*100:.1f}%")
print(f"\nBHKW-Leistung:")
print(f"  Elektrische Leistung: {chp['P_el']:.1f} kW")
print(f"  Thermische Leistung:  {chp['P_th']:.1f} kW")
print(f"  Gasverbrauch:         {chp['Q_gas_consumed']:.1f} Nm³/d")
print(f"\nHeizung:")
print(f"  Bereitgestellte Wärme: {heating['Q_heat_supplied']:.1f} kW")
print(f"  Zusatzwärme:           {heating['P_aux_heat']:.1f} kW")
```

**Erwartete Ausgabe:**

```text
Simulated 100/240 steps
Simulated 200/240 steps

Endergebnisse (Tag 10):

Fermenter-Leistung:
  Biogas:  2561.9 Nm³/d
  Methan:  1349.9 Nm³/d
  CH4-Gehalt: 52.7%

BHKW-Leistung:
  Elektrische Leistung: 225.0 kW
  Thermische Leistung:  253.1 kW
  Gasverbrauch:         2561.9 Nm³/d

Heizung:
  Bereitgestellte Wärme: 10.0 kW
  Zusatzwärme:           0.0 kW
```

Das BHKW liefert 225,0 kW seiner 500 kW Nennleistung: Es verbrennt das gesamte
Gas, das der Fermenter erzeugt, der Speicher wird in jedem Schritt geleert und
nichts gelangt zur Fackel.

Der Wärmebedarf von 10,0 kW ist nur der pauschale Wärmeverlust-Term
(0,5 kW/K × 20 K zwischen 35 °C und der voreingestellten Umgebungstemperatur
von 15 °C). Übergeben Sie `digester_id="main_digester"` an `add_heating`, um
die Wärme zum Aufheizen der Substrate einzubeziehen; der Bedarf beträgt dann
36,1 kW.

**Automatische Komponentenerstellung:**

PlantConfigurator erstellt und verbindet:

- **Gasspeicher**: Einer pro Fermenter (Membran, dimensioniert auf V_gas)  
- **Fackel**: Eine pro BHKW (Sicherheitsverbrennung, 98 % CH₄-Zerstörung)  

**Verbindungskette:**

```text
Fermenter → Gasspeicher → BHKW → Fackel
                          ↓
                       Heizung
```

## Zweistufige Vergärung

Ein vollständiges zweistufiges Skript ist [`examples/02_two_stage_plant.py`](https://github.com/dgaida/PyADM1ODE/blob/master/examples/02_two_stage_plant.py); es verwendet eigene Behältergrößen und eine eigene Substratmischung.

### Systemarchitektur

```text
[Einsatz] → [Hydrolyse] → [Speicher 1] ↘
            500 m³         100 m³        → [BHKW] → [Fackel]
            @ 45 °C                     ↗  500 kW    98 %
                                       ↓
[Ablauf]  → [Haupt]  → [Speicher 2] ↗  [Heizung 1] + [Heizung 2]
            2000 m³     300 m³           45 °C         35 °C
            @ 35 °C
```

### Kernfunktionen

- **Temperaturphasiert**: 45 °C in der ersten Stufe, 35 °C in der zweiten  
- **Zwei Fermenter in Serie**: Der Ablauf der ersten Stufe speist die zweite  
- **Ein Gasspeicher pro Fermenter**: Beide Speicher versorgen dasselbe BHKW  
- **Mechanische Komponenten**: Pumpen und Rührwerke bringen ihren Strombedarf ein  
- **Energieintegration**: Das BHKW liefert Strom und die Wärme für beide Heizsysteme  

### Konfiguration

```python
plant = BiogasPlant("Zweistufige Anlage")
configurator = PlantConfigurator(plant, feedstock)

# Stufe 1: Hydrolysebehälter (45 °C)
configurator.add_digester(
    digester_id="hydrolysis_tank",
    V_liq=500.0,
    V_gas=100.0,
    T_ad=318.15,  # 45 °C für schnellere Hydrolyse
    Q_substrates=[15, 10, 0, 0, 0, 0, 0, 0, 0, 0]
)

# Stufe 2: Hauptfermenter (mesophil, 35 °C)
configurator.add_digester(
    digester_id="main_digester",
    V_liq=2000.0,
    V_gas=300.0,
    T_ad=308.15,  # 35 °C
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

# Initialisieren und simulieren
plant.initialize()
results = plant.simulate(duration=10.0, dt=1.0/24.0, save_interval=1.0)
```

Pumpen und Rührwerke werden ohne Verbindungen hinzugefügt: Sie laufen in ihrem
Nennbetriebspunkt und tragen nur ihren Strombedarf bei.
`configurator.add_mixer(mixer_id, digester_id)` baut ein Rührwerk dagegen in
einen Fermenter ein und übernimmt die Behältergeometrie von ihm (siehe
[Mechanische Komponenten](components/mechanical.md)).

### Gasfluss in vier Durchläufen

Jeder Anlagenschritt läuft in vier Durchläufen ab, damit jede Komponente einmal
mit dem Gas gerechnet wird, das sie tatsächlich erhält:

1. **Fermenter** und alle übrigen Komponenten außerhalb der Gasstrecke: Die Fermenter erzeugen Gas  
2. **Gasspeicher**: Jeder Speicher nimmt die Produktion seines Fermenters und den Bedarf seiner Verbraucher gleichzeitig auf; Gas über der Kapazität wird entlüftet  
3. **Gasverbraucher**: Das BHKW läuft mit dem tatsächlich gelieferten Gas, danach nutzen die Heizsysteme seine Wärme  
4. **Fackeln**: verbrennen das Gas, das die Speicher entlüftet haben  

Die Ströme des letzten Schritts (Tag 10):

```python
final = results[-1]["components"]

for digester_id in ("hydrolysis_tank", "main_digester"):
    storage = final[f"{digester_id}_storage"]
    print(f"{digester_id}:")
    print(f"  Erzeugt:          {final[digester_id]['Q_gas']:7.1f} Nm³/d")
    print(f"  Speicher liefert: {storage['Q_gas_supplied_m3_per_day']:7.1f} Nm³/d")
    print(f"  Entlüftet:        {storage['vented_volume_m3']:7.1f} m³")

print("chp_main:")
print(f"  Verbraucht:       {final['chp_main']['Q_gas_consumed']:7.1f} Nm³/d")
print(f"  Abgefackelt:      {final['chp_main_flare']['vented_volume_m3']:7.1f} m³")
```

Ausgabe:

```text
hydrolysis_tank:
  Erzeugt:           2087.8 Nm³/d
  Speicher liefert:  2087.8 Nm³/d
  Entlüftet:            0.0 m³
main_digester:
  Erzeugt:            397.1 Nm³/d
  Speicher liefert:   397.1 Nm³/d
  Entlüftet:            0.0 m³
chp_main:
  Verbraucht:        2484.9 Nm³/d
  Abgefackelt:          0.0 m³
```

### Vergleich mit der einstufigen Anlage

Beide Anlagen erhalten dieselbe Fütterung. Werte an Tag 10:

| Aspekt | Einstufig | Zweistufig |
|--------|-----------|------------|
| **Fermentervolumen** | 2000 m³ | 500 m³ + 2000 m³ |
| **Biogasproduktion** | 2561,9 Nm³/d | 2484,9 Nm³/d |
| **Methanproduktion** | 1349,9 Nm³/d | 1315,6 Nm³/d |
| **CH₄-Gehalt** | 52,7 % | 52,9 % |
| **Elektrische BHKW-Leistung** | 225,0 kW | 219,3 kW |

Zehn Tage sind kurz gegenüber der Verweilzeit (86 Tage im einstufigen
Fermenter); die Werte beschreiben deshalb den Beginn des Betriebs, keinen
stationären Zustand.

### Erwartete Ergebnisse

**Energiebilanz:**

```python
chp = final["chp_main"]
balance = plant.get_electricity_balance()

print("Energieerzeugung:")
print(f"  Elektrisch (brutto): {balance['P_el_produced']:7.1f} kW")
print(f"  Thermisch:           {chp['P_th']:7.1f} kW")

print("\nEigenverbrauch:")
for component_id, power in balance["consumers"].items():
    print(f"  {component_id + ':':<20} {power:7.2f} kW")
print(f"  {'Gesamt:':<20} {balance['P_el_consumed']:7.2f} kW")

print(f"\nNetto-Stromerzeugung:  {balance['P_el_net']:7.1f} kW")

heat_demand = final["heating_1"]["Q_heat_supplied"] + final["heating_2"]["Q_heat_supplied"]
print("\nWärmenutzung:")
print(f"  Wärmebedarf:         {heat_demand:7.1f} kW")
print(f"  BHKW-Wärme:          {chp['P_th']:7.1f} kW")
```

Ausgabe:

```text
Energieerzeugung:
  Elektrisch (brutto):   219.3 kW
  Thermisch:             246.7 kW

Eigenverbrauch:
  chp_main:               6.58 kW
  feed_pump:              0.03 kW
  transfer_pump:          0.04 kW
  mixer_1:                3.75 kW
  mixer_2:                2.50 kW
  Gesamt:                12.90 kW

Netto-Stromerzeugung:    206.4 kW

Wärmenutzung:
  Wärmebedarf:            25.0 kW
  BHKW-Wärme:            246.7 kW
```

## Arbeiten mit verschiedenen Substraten

### Verfügbare Substrate

PyADM1ODE enthält 13 vorkonfigurierte landwirtschaftliche Substrate.
`Feedstock()` ohne Substratliste lädt alle in dieser Reihenfolge; der Slot ist
die Position in `Q_substrates`:

| Slot | Substrat-ID | Substrat |
|------|-------------|----------|
| 0 | `maize_silage_milk_ripeness` | Maissilage (Milchreife) |
| 1 | `cattle_manure` | Rindergülle |
| 2 | `swine_manure` | Schweinegülle |
| 3 | `corn_cob_mix` | Corn-Cob-Mix (CCM) |
| 4 | `grass_silage` | Grassilage |
| 5 | `green_rye_silage` | Grünroggensilage |
| 6 | `cereal_gps_silage` | Getreide-Ganzpflanzensilage (GPS) |
| 7 | `onion_waste` | Zwiebelabfälle |
| 8 | `maize_silage_gummersbach` | Maissilage (Anlage Gummersbach) |
| 9 | `cattle_manure_solid` | Rinderfestmist |
| 10 | `chicken_manure_dry` | Hühnertrockenkot (HTK) |
| 11 | `swine_manure_gummersbach` | Schweinegülle (Anlage Gummersbach) |
| 12 | `wheat_whole_plant_silage` | Weizen-Ganzpflanzensilage |

`Q_substrates` darf kürzer sein als die Liste; fehlende Slots zählen als null.
Um die Slots selbst festzulegen, benennen Sie die Substrate:
`Feedstock(["maize_silage_milk_ripeness", "cattle_manure"])` mit
`Q_substrates=[15, 10]` liefert dieselben Ergebnisse wie die erste Simulation.
Die Substrate sind in der [Substratbibliothek](../substrates/library.md)
beschrieben.

### Beispiele für Substratfütterung

```python
# Energiereiche Mischung (Maissilage + Rindergülle)
Q = [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]

# Grasbasiert (Grassilage + Rindergülle)
Q = [0, 5, 0, 0, 20, 0, 0, 0, 0, 0]

# Abfallbasiert (Gülle + Gemüse)
Q = [0, 10, 15, 0, 0, 0, 0, 5, 0, 0]

# Fokus auf Energiepflanzen (Maissilage + Grünroggensilage + Rindergülle)
Q = [20, 5, 0, 0, 0, 10, 0, 0, 0, 0]
```

Diese Arrays zeigen nur, wie die Slots angesprochen werden. Ob eine Mischung
stabil läuft, hängt von der Anlage ab, siehe [Fehlerbehebung](#fehlerbehebung).

### Substratinformationen

Detaillierte Substrateigenschaften abrufen:

```python
from pyadm1.substrates import SubstrateRegistry

# Substratparameter anzeigen
substrate = SubstrateRegistry().get("maize_silage_milk_ripeness")
print(f"Name: {substrate.name}")
print(f"pH-Wert: {substrate.pH}")
print(f"Trockensubstanz: {substrate.TS / 10:.1f} %FM")
print(f"Organische Trockensubstanz: {(1 - substrate.fRA) * 100:.1f} %TS")
print(f"Biochemisches Methanpotenzial: {substrate.BMP:.0f} Nm³ CH4/t oTS")
```

Ausgabe:

```text
Name: Maize silage (milk ripeness)
pH-Wert: 3.9
Trockensubstanz: 32.6 %FM
Organische Trockensubstanz: 94.0 %TS
Biochemisches Methanpotenzial: 338 Nm³ CH4/t oTS
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

Die Datei enthält die Konfiguration und den aktuellen Zustand der Anlage; die
geladene Anlage setzt deshalb bei der Simulationszeit fort, bei der sie
gespeichert wurde.

## Ergebnisse verstehen

### Wichtige Ausgabevariablen

#### Fermenter-Ausgaben

- `Q_gas` – Biogasproduktion, trockenes Gas [Nm³/d]  
- `Q_ch4` – Methanproduktion [Nm³/d]  
- `Q_co2` – CO2-Produktion [Nm³/d]  
- `pH` – pH-Wert [-]  
- `VFA` – Flüchtige Fettsäuren [g HAceq/L]  
- `TAC` – Gesamtalkalinität [g CaCO3/L]  

#### BHKW-Ausgaben

- `P_el` – Elektrische Leistung [kW]  
- `P_th` – Thermische Leistung [kW]  
- `Q_gas_consumed` – Gasverbrauch [Nm³/d]  
- `CH4_fraction` – Methananteil des verbrannten Gases [0–1]  
- `P_consumed` – Strombedarf der BHKW-Nebenaggregate [kW]  

#### Heizungsausgaben

- `Q_heat_supplied` – Gelieferte Wärme [kW]  
- `P_th_used` – Genutzte BHKW-Wärme [kW]  
- `P_aux_heat` – Benötigte Zusatzwärme [kW]  

#### Gasspeicher-Ausgaben

- `stored_volume_m3` – Aktuelles Volumen [m³ STP]  
- `pressure_bar` – Aktueller Druck [bar]  
- `utilization` – Füllstand [0–1]  
- `vented_volume_m3` – In diesem Schritt entlüftetes Gas [m³]  
- `Q_gas_supplied_m3_per_day` – An Verbraucher geliefertes Gas [m³/d]  

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
    configurator.add_digester("main_digester", V_liq=2000, Q_substrates=Q)

    plant.initialize()
    results = plant.simulate(duration=10.0, dt=1.0/24.0)

    final = results[-1]["components"]["main_digester"]
    results_all.append({
        'feed': feed,
        'biogas': final['Q_gas'],
        'methane': final['Q_ch4']
    })

for r in results_all:
    print(f"Einsatz {r['feed']} t/d → CH4 {r['methane']:.1f} Nm³/d")
```

Ausgabe (nach den Fortschrittsmeldungen):

```text
...
Einsatz 10 t/d → CH4 989.4 Nm³/d
Einsatz 15 t/d → CH4 1376.3 Nm³/d
Einsatz 20 t/d → CH4 1750.4 Nm³/d
Einsatz 25 t/d → CH4 2113.1 Nm³/d
```

### Muster 2: Zeitreihenanalyse

Entwicklung über die Zeit verfolgen. `results` ist die von `plant.simulate()`
zurückgegebene Liste:

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
ax1.set_ylabel('Produktion [Nm³/d]')
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
   - [`examples/05_parallel_two_stage_simulation.py`](https://github.com/dgaida/PyADM1ODE/blob/master/examples/05_parallel_two_stage_simulation.py)  
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
configurator.add_digester(
    "main_digester", V_liq=2000.0, V_gas=300.0, T_ad=308.15,
    Q_substrates=[15, 10, 0, 0, 0, 0, 0, 0, 0, 0]
)
configurator.add_chp("chp_main", P_el_nom=500.0, eta_el=0.40, eta_th=0.45)
configurator.add_heating("heating_main", target_temperature=308.15)

# Verbinden
configurator.auto_connect_digester_to_chp("main_digester", "chp_main")
configurator.auto_connect_chp_to_heating("chp_main", "heating_main")
# Allgemeine Form: configurator.connect(from_id, to_id, connection_type)

# Simulieren
plant.initialize()
results = plant.simulate(duration=10.0, dt=1.0/24.0, save_interval=1.0)

# Speichern/Laden
plant.to_json("my_plant_config.json")
plant = BiogasPlant.from_json("my_plant_config.json", feedstock)
```

### Temperaturumrechnungen

```python
# Gängige Temperaturen
T_mesophil = 308.15     # 35 °C
T_thermophil = 328.15   # 55 °C
T_psychrophil = 298.15  # 25 °C

# °C in K umrechnen
T_celsius = 35.0
T_K = T_celsius + 273.15  # 308.15
```

### Standardparameter

Die Werte, die gelten, wenn ein Argument weggelassen wird:

```python
# configurator.add_digester
V_liq = 1050.0         # Flüssigvolumen [m³]
V_gas = 150.0          # Gasraum [m³]
T_ad = 315.15          # Temperatur [K] = 42 °C

# configurator.add_chp
P_el_nom = 500.0       # Elektrische Leistung [kW]
eta_el = 0.40          # Elektrischer Wirkungsgrad
eta_th = 0.45          # Thermischer Wirkungsgrad

# configurator.add_heating
target_temperature = 315.15    # Sollwert [K] = 42 °C
heat_loss_coefficient = None   # [kW/K]; None = vom Fermenter, sonst 0.5

# plant.simulate
dt = 1.0 / 24.0        # Zeitschritt [Tage] = 1 Stunde
save_interval = None   # None = jeden Schritt speichern
```

## Fehlerbehebung

### Problem: Simulation instabil

**Symptome**: pH fällt, VFA steigt, Methanproduktion sinkt

**Lösungen**:

- Substratfütterungsrate reduzieren  
- Verweilzeit erhöhen (größeres V_liq)  
- Substratzusammensetzung prüfen  

Beispiel: Mit der grasbasierten Mischung aus
[Beispiele für Substratfütterung](#beispiele-für-substratfütterung) im
2000-m³-Fermenter bei 35 °C steigen die VFA von 0,5 g/L an Tag 10 auf
10,3 g/L an Tag 60. Die halbe Fütterung oder das doppelte Volumen verzögert
das nur (3,7 g/L an Tag 60). Die Maismischung der ersten Simulation bleibt
über dieselben 60 Tage bei etwa 0,2 g/L.

### Problem: Geringe Gasproduktion

**Lösungen**:

- Substratfütterung erhöhen  
- Abbaubarkeit des Substrats prüfen  
- Fermentertemperatur prüfen  

### Problem: Langsame Simulation

**Lösungen**:

- Zeitschritt `dt` erhöhen  
- `save_interval` erhöhen, um weniger Ergebnisse zu speichern  
- Parallele Simulation für Parameterstudien verwenden  

Die Fermentergleichungen werden innerhalb jedes Schritts mit einem adaptiven
Löser integriert; `dt` legt fest, wie oft die Komponenten ihre Werte
austauschen. Die erste Simulation liefert an Tag 10 mit `dt=1.0` dieselbe
Biogasproduktion (2561,9 Nm³/d) wie mit Stundenschritten.

Weitere Hilfe finden Sie in der [Installationsanleitung](installation.md) oder kontaktieren Sie <daniel.gaida@th-koeln.de>.

## Referenzen

- **ADM1-Modell**: Batstone et al. (2002). *Anaerobic Digestion Model No. 1*. IWA Publishing.  
- **Leitfaden Biogas**: FNR (2016). <https://mediathek.fnr.de/leitfaden-biogas.html>  
