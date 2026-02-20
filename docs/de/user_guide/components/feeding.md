# Fütterungskomponenten

Komponenten für Substrathandhabung, Lagerung und Dosierung in Biogasanlagen.

## SubstrateStorage

Lagerbehälter für verschiedene Substrattypen mit Bestandsverfolgung und Qualitätsmanagement.

### Parameter

```python
from pyadm1.components.feeding import SubstrateStorage

storage = SubstrateStorage(
    component_id="silo1",
    storage_type="vertical_silo",    # Siehe Tabelle unten
    substrate_type="corn_silage",    # Siehe Tabelle unten
    capacity=1000.0,                 # Max. Kapazität [t oder m³]
    initial_level=800.0,             # Anfangsbestand
    degradation_rate=None,           # Auto-berechnet
    temperature=288.15,              # Lagertemperatur [K] (15°C)
    name="Maissilagelager"
)
```

### Lagertypen

| Typ | Degradation [1/d] | Beste Anwendung | Typische Größe | Investition |
|------|------------------|----------|--------------|------------|
| **Vertical Silo** | 0.0005 | Mais-/Grassilage | 500-2000 t | Hoch |
| **Horizontal Silo** | 0.0008 | Großbetriebe | 1000-3000 t | Mittel |
| **Bunker Silo** | 0.001 | Überfahrbarer Zugang | 1000-5000 t | Mittel |
| **Clamp** | 0.0025 | Saisonale Lagerung | 500-2000 t | Niedrig |
| **Above-ground Tank** | 0.0002 | Flüssiggülle | 500-3000 m³ | Hoch |
| **Below-ground Tank** | 0.0001 | Flüssige Lagerung | 1000-5000 m³ | Sehr hoch |

### Substrattypen

| Substrat | Dichte [kg/m³] | TM [%] | oTS [% der TM] | Typische Lagerung |
|-----------|----------------|--------|--------------|-----------------|
| Maissilage | 650 | 35 | 95 | Silo |
| Grassilage | 700 | 30 | 92 | Silo |
| Ganzpflanzensilage | 680 | 32 | 94 | Silo/Bunker |
| Flüssiggülle | 1020 | 8 | 80 | Tank |
| Festmist | 850 | 25 | 75 | Miete |
| Bioabfall | 1000 | 20 | 90 | Tank |

### Qualitätsverschlechterung

Lagerqualität verschlechtert sich über Zeit:

```python
# Qualitätsfaktor zum Zeitpunkt t:
quality(t) = quality(0) × exp(-k × t)

# Wobei:
# k = degradation_rate [1/d]
# t = storage_time [Tage]

# Beispiel: Maissilage im Hochsilo
# Nach 30 Tagen: quality = 1.0 × exp(-0.0005 × 30) = 0.985 (98.5%)
# Nach 90 Tagen: quality = 1.0 × exp(-0.0005 × 90) = 0.956 (95.6%)
```

**Temperatureffekt:**

Temperatur beeinflusst Degradation (Q10 = 2):

```python
# Degradation steigt mit Temperatur
T_ref = 288.15  # 15°C Referenz
k_ref = 0.0005  # Basisrate

# Bei 20°C (293.15 K):
k_20C = k_ref × 2^((293.15-288.15)/10) = 0.0007

# Bei 10°C (283.15 K):
k_10C = k_ref × 2^((283.15-288.15)/10) = 0.0004
```

### Ausgaben

```python
{
    'current_level': 750.0,      # Bestand [t oder m³]
    'utilization': 0.75,         # Füllstand (0-1)
    'quality_factor': 0.95,      # Qualität (0-1)
    'available_mass': 712.5,     # Nutzbare Masse
    'degradation_rate': 0.0005,  # Aktuelle Rate
    'losses_this_step': 0.4,     # Verluste [t oder m³]
    'withdrawn_this_step': 15.0, # Entnommen [t oder m³]
    'is_empty': False,
    'is_full': False,
    'storage_time': 25.5,        # Tage gelagert
    'dry_matter': 35.0,          # TM [%]
    'vs_content': 95.0           # oTS [% der TM]
}
```

### Verwendungsbeispiel

```python
# Maissilagelagerung
storage = SubstrateStorage(
    component_id="silo1",
    storage_type="vertical_silo",
    substrate_type="corn_silage",
    capacity=1000,
    initial_level=800
)

storage.initialize()

# Täglicher Betrieb
result = storage.step(
    t=10,
    dt=1,
    inputs={
        'withdrawal_rate': 15,    # m³/d oder t/d
        'refill_amount': 0,
        'temperature': 288.15
    }
)

print(f"Füllstand: {result['current_level']:.1f} t")
print(f"Qualität: {result['quality_factor']:.3f}")
print(f"Verfügbar: {result['available_mass']:.1f} t")
print(f"Verluste: {result['losses_this_step']:.2f} t")
```

### Lagermanagement-Strategie

```python
def should_refill(storage_result, safety_days=7):
    """Bestimme ob Nachfüllung erforderlich ist"""
    level = storage_result['current_level']
    daily_usage = 15  # t/d
    days_remaining = level / daily_usage

    return days_remaining < safety_days

def check_quality(storage_result, min_quality=0.90):
    """Alarm bei zu niedriger Qualität"""
    quality = storage_result['quality_factor']
    if quality < min_quality:
        print(f"Warnung: Qualität bei {quality:.1%}")
        return False
    return True
```

## Feeder

Automatische Dosiersysteme für präzise Substratfütterung.

### Parameter

```python
from pyadm1.components.feeding import Feeder

feeder = Feeder(
    component_id="feed1",
    feeder_type="screw",             # Auto-gewählt wenn None
    Q_max=20.0,                      # Max. Durchfluss [m³/d oder t/d]
    substrate_type="solid",          # "solid", "slurry", "liquid", "fibrous"
    dosing_accuracy=None,            # Auto-berechnet
    power_installed=None,            # Auto-berechnet
    enable_dosing_noise=True,        # Realistische Varianz
    name="Schneckendosierer"
)
```

### Dosierertypen

| Typ | Genauigkeit [±%] | Beste Anwendung | Drehzahlregelung | Leistung [kW/m³/h] |
|------|--------------|----------|---------------|----------------|
| **Screw** | 5 | Feste Substrate | Gut | 0.8 |
| **Twin Screw** | 3 | Bessere Kontrolle | Ausgezeichnet | 1.0 |
| **Progressive Cavity** | 2 | Viskose Schlämme | Gut | 1.2 |
| **Piston** | 1 | Präzise Dosierung | Ausgezeichnet | 1.5 |
| **Centrifugal Pump** | 8 | Niedrige Viskosität | Mittelmäßig | 0.5 |
| **Mixer Wagon** | 10 | Chargen-Fütterung | N/A | 2.0 |

### Dosiergenauigkeit

Echte Dosierer haben Varianz um Sollwerte:

```python
# Mit aktiviertem dosing_noise:
# Tatsächlicher Durchfluss = Sollwert + Rauschen
# Wobei Rauschen ~ Normal(0, accuracy × Sollwert)

# Beispiel: Schneckendosierer (5% Genauigkeit) bei 15 m³/d
# Typischer Bereich: 14.25 - 15.75 m³/d
# Gelegentlich: 13.5 - 16.5 m³/d (±2σ)

feeder = Feeder("feed1", Q_max=20, dosing_accuracy=0.05)
```

### Leistungsanforderungen

Leistung hängt vom Substrattyp ab:

| Substrat | Basisleistung [kW/m³/h] | Modifikator | Gesamt |
|-----------|---------------------|----------|-------|
| Flüssig | 0.5 | ×0.7 | 0.35 |
| Schlämme | 0.8 | ×1.0 | 0.80 |
| Fest | 0.8 | ×1.4 | 1.12 |
| Faserreich | 0.8 | ×1.8 | 1.44 |

```python
# Beispiel: 15 m³/h Schneckendosierer für faserreiches Substrat
Q_nom_h = 15 / 24  # = 0.625 m³/h
P = 0.8 * 0.625 * 1.8 * 1.3  # [Basis × Q × Modifikator × Sicherheit]
  = 1.17 kW
```

### Ausgaben

```python
{
    'Q_actual': 14.8,            # Tatsächlicher Durchfluss [m³/d]
    'is_running': True,
    'load_factor': 0.74,         # Last (0-1)
    'P_consumed': 2.5,           # Leistung [kW]
    'blockage_detected': False,  # Alarm
    'dosing_error': 1.3,         # Fehler [%]
    'speed_fraction': 0.95,      # Drehzahl (0-1)
    'dosing_accuracy': 0.05,     # Genauigkeit
    'total_mass_fed': 1250.0     # Kumulativ [t]
}
```

### Verwendungsbeispiel

```python
# Schneckendosierer für feste Substrate
feeder = Feeder(
    component_id="feed1",
    feeder_type="screw",
    Q_max=20.0,
    substrate_type="solid",
    enable_dosing_noise=True
)

feeder.initialize()

result = feeder.step(
    t=0,
    dt=1/24,
    inputs={
        'Q_setpoint': 15.0,
        'enable_feeding': True,
        'substrate_available': 500,
        'speed_setpoint': 1.0
    }
)

print(f"Ziel: 15.0 m³/d")
print(f"Tatsächlich: {result['Q_actual']:.2f} m³/d")
print(f"Fehler: {result['dosing_error']:.1f}%")
print(f"Leistung: {result['P_consumed']:.2f} kW")
```

### Blockadenerkennung

Dosierer können Blockaden erkennen und handhaben:

```python
# Automatische Handhabung
if result['blockage_detected']:
    print("Blockade erkannt!")
    # Dosierer reduziert automatisch Durchfluss auf 10%
    # Weiter überwachen

# Überwache kumulative Blockaden
print(f"Gesamt-Blockaden: {feeder.state['n_blockages']}")
```

## Vollständige Fütterungskette

### Integriertes System

```python
from pyadm1.configurator import BiogasPlant, PlantConfigurator
from pyadm1.components.feeding import SubstrateStorage, Feeder
from pyadm1.components.mechanical import Pump
from pyadm1.substrates import Feedstock

# Setup
feedstock = Feedstock(feeding_freq=48)
plant = BiogasPlant("Komplettes Fütterungssystem")
config = PlantConfigurator(plant, feedstock)

# 1. Mehrfache Substratlagerbehälter
corn_storage = SubstrateStorage(
    "corn_silo",
    storage_type="vertical_silo",
    substrate_type="corn_silage",
    capacity=1000,
    initial_level=800
)
plant.add_component(corn_storage)

manure_storage = SubstrateStorage(
    "manure_tank",
    storage_type="above_ground_tank",
    substrate_type="manure_liquid",
    capacity=500,
    initial_level=400
)
plant.add_component(manure_storage)

# 2. Dosierer für jedes Substrat
corn_feeder = Feeder(
    "corn_feeder",
    feeder_type="screw",
    Q_max=15.0,
    substrate_type="solid"
)
plant.add_component(corn_feeder)

manure_feeder = Feeder(
    "manure_feeder",
    feeder_type="progressive_cavity",
    Q_max=10.0,
    substrate_type="slurry"
)
plant.add_component(manure_feeder)

# 3. Mischpumpe
mix_pump = Pump(
    "mix_pump",
    pump_type="progressive_cavity",
    Q_nom=25.0,
    pressure_head=50.0
)
plant.add_component(mix_pump)

# 4. Fermenter
digester, storage = config.add_digester(
    "main_digester",
    V_liq=2000,
    Q_substrates=[15, 10, 0, 0, 0, 0, 0, 0, 0, 0]
)

# 5. Energiesystem
config.add_chp("chp1", P_el_nom=500)
config.add_heating("heat1", target_temperature=308.15)

# Verbindungen
config.connect("corn_silo", "corn_feeder", "default")
config.connect("manure_tank", "manure_feeder", "default")
config.connect("corn_feeder", "mix_pump", "default")
config.connect("manure_feeder", "mix_pump", "default")
config.connect("mix_pump", "main_digester", "liquid")
config.auto_connect_digester_to_chp("main_digester", "chp1")
config.auto_connect_chp_to_heating("chp1", "heat1")

# Simulieren
plant.initialize()
results = plant.simulate(duration=30, dt=1/24, save_interval=1.0)

# Fütterungsanalyse
def feeding_system_analysis(results):
    """Analysiere Fütterungssystem-Leistung"""
    final = results[-1]
    comp = final['components']

    # Lagerbestand
    corn_level = comp['corn_silo']['current_level']
    manure_level = comp['manure_tank']['current_level']

    # Qualität
    corn_quality = comp['corn_silo']['quality_factor']
    manure_quality = comp['manure_tank']['quality_factor']

    # Durchsatz
    corn_fed = comp['corn_feeder']['total_mass_fed']
    manure_fed = comp['manure_feeder']['total_mass_fed']

    # Energieverbrauch
    corn_feeder_energy = comp['corn_feeder'].get('energy_consumed', 0)
    manure_feeder_energy = comp['manure_feeder'].get('energy_consumed', 0)
    pump_energy = comp['mix_pump']['energy_consumed']

    total_feed_energy = corn_feeder_energy + manure_feeder_energy + pump_energy

    return {
        'corn_remaining': corn_level,
        'manure_remaining': manure_level,
        'corn_quality': corn_quality,
        'manure_quality': manure_quality,
        'total_corn_fed': corn_fed,
        'total_manure_fed': manure_fed,
        'feeding_energy': total_feed_energy
    }

analysis = feeding_system_analysis(results)
print("\nFütterungssystem-Analyse:")
print(f"Mais verbleibend: {analysis['corn_remaining']:.0f} t (Qualität: {analysis['corn_quality']:.1%})")
print(f"Gülle verbleibend: {analysis['manure_remaining']:.0f} m³ (Qualität: {analysis['manure_quality']:.1%})")
print(f"Gesamt Mais gefüttert: {analysis['total_corn_fed']:.0f} t")
print(f"Gesamt Gülle gefüttert: {analysis['total_manure_fed']:.0f} m³")
print(f"Fütterungsenergie: {analysis['feeding_energy']:.0f} kWh")
```

## Optimierungsstrategien

### 1. Substratmischoptimierung

```python
def optimize_substrate_mix(available_substrates, target_vs_loading):
    """Optimiere Substratmischung für Ziel-oTS-Belastung"""

    # Beispiel-Substrate
    substrates = {
        'corn_silage': {'vs': 0.33, 'cost': 30},  # 33% oTS, 30 €/t
        'manure': {'vs': 0.06, 'cost': 0},        # 6% oTS, kostenlos
        'biowaste': {'vs': 0.17, 'cost': -20}     # 17% oTS, Gate Fee
    }

    # Einfache Mischberechnung (kann erweitert werden für Optimierung)
    corn_fraction = 0.60
    manure_fraction = 0.30
    biowaste_fraction = 0.10

    mix_vs = (corn_fraction * substrates['corn_silage']['vs'] +
              manure_fraction * substrates['manure']['vs'] +
              biowaste_fraction * substrates['biowaste']['vs'])

    mix_cost = (corn_fraction * substrates['corn_silage']['cost'] +
                manure_fraction * substrates['manure']['cost'] +
                biowaste_fraction * substrates['biowaste']['cost'])

    print(f"Optimierte Mischung:")
    print(f"- Mais: {corn_fraction:.0%}")
    print(f"- Gülle: {manure_fraction:.0%}")
    print(f"- Bioabfall: {biowaste_fraction:.0%}")
    print(f"Resultierende oTS: {mix_vs:.1%}")
    print(f"Kosten: {mix_cost:.1f} €/t")

    return {
        'corn': corn_fraction,
        'manure': manure_fraction,
        'biowaste': biowaste_fraction,
        'total_vs': mix_vs,
        'cost': mix_cost
    }

optimized_mix = optimize_substrate_mix({}, 0.20)
```

### 2. Bestandsmanagement

```python
def manage_inventory(storage_results, forecast_days=30):
    """Verwalte Lagerbestand mit Vorhersage"""

    for name, result in storage_results.items():
        level = result['current_level']
        capacity = result.get('capacity', 1000)
        daily_usage = 15  # Beispiel

        days_remaining = level / daily_usage

        print(f"\n{name}:")
        print(f"- Aktueller Bestand: {level:.0f} ({level/capacity:.1%} Kapazität)")
        print(f"- Verbleibende Tage: {days_remaining:.1f}")

        if days_remaining < 7:
            print("- AKTION: Dringend nachfüllen!")
            refill_amount = capacity * 0.8 - level
            print(f"- Empfohlene Nachfüllung: {refill_amount:.0f}")
        elif days_remaining < 14:
            print("- WARNUNG: Nachfüllung planen")

        # Qualitätsprüfung
        quality = result['quality_factor']
        if quality < 0.90:
            print(f"- QUALITÄT: Niedrig ({quality:.1%}) - erwäge Verwendungsreihenfolge")

# Beispielverwendung
storage_results = {
    'corn_silo': final['components']['corn_silo'],
    'manure_tank': final['components']['manure_tank']
}
manage_inventory(storage_results)
```

### 3. Dosiergenauigkeitsoptimierung

```python
def optimize_dosing_accuracy(substrate_value, process_sensitivity):
    """Wähle Dosierertyp basierend auf Anforderungen"""

    # Hochwertige Substrate oder sensible Prozesse benötigen hohe Genauigkeit
    if substrate_value > 40 or process_sensitivity == "high":
        recommended_type = "piston"
        accuracy = 0.01
    elif substrate_value > 20 or process_sensitivity == "medium":
        recommended_type = "progressive_cavity"
        accuracy = 0.02
    else:
        recommended_type = "screw"
        accuracy = 0.05

    print(f"Empfohlener Dosierertyp: {recommended_type}")
    print(f"Erwartete Genauigkeit: ±{accuracy:.0%}")

    return recommended_type, accuracy

# Beispiel: Hochwertige Energiepflanzen
feeder_type, accuracy = optimize_dosing_accuracy(substrate_value=35,
                                                  process_sensitivity="medium")
```

## Fehlerbehebung

### Problem: Schneller Qualitätsverlust

**Diagnose:**
```python
storage_result = storage.step(t, dt, inputs)

if storage_result['quality_factor'] < 0.95 and storage_result['storage_time'] < 30:
    print("Schneller Qualitätsverlust erkannt:")
    print(f"- Qualität: {storage_result['quality_factor']:.1%}")
    print(f"- Lagerzeit: {storage_result['storage_time']:.1f} Tage")
    print(f"- Degradationsrate: {storage_result['degradation_rate']:.4f} 1/d")
    print(f"- Temperatur: {storage.temperature:.1f} K")
```

**Lösungen:**
```python
# Option 1: Verbessere Lagertyp
storage_improved = SubstrateStorage(
    "silo1",
    storage_type="vertical_silo",  # Von "clamp"
    substrate_type="corn_silage",
    capacity=1000
)

# Option 2: Reduziere Temperatur
storage.temperature = 283.15  # 10°C statt 15°C

# Option 3: Schnellere Verwendung (reduziere Lagerzeit)
increase_daily_usage = True
```

### Problem: Dosierer-Blockaden

**Diagnose:**
```python
if feeder.state['n_blockages'] > 5:
    print(f"Häufige Blockaden erkannt: {feeder.state['n_blockages']}")
    print("Mögliche Ursachen:")
    print("- Faserreiches Substrat für Dosierertyp ungeeignet")
    print("- Fremdkörper im Substrat")
    print("- Verschleiß oder Wartung erforderlich")
```

**Lösungen:**
- Wechsle zu robusterem Dosierertyp (Twin Screw)
- Verbessere Substratvorbereitung
- Implementiere Wartungsplan

### Problem: Inkonsistente Dosierung

**Diagnose:**
```python
dosing_errors = [r['components']['feed1']['dosing_error']
                for r in results]
avg_error = sum(dosing_errors) / len(dosing_errors)

if avg_error > 10:
    print(f"Hoher durchschn. Dosierfehler: {avg_error:.1f}%")
    print("Empfehlungen:")
    print("- Erwäge präziseren Dosierertyp")
    print("- Prüfe Kalibrierung")
    print("- Deaktiviere dosing_noise für idealisierte Simulation")
```

## Best Practices

1. **Qualitätsüberwachung implementieren**
   - Verfolge quality_factor über Zeit
   - Alarmiere bei <90% Qualität
   - Plane FIFO-Rotation

2. **Bestandssicherheit aufrechterhalten**
   - 7-14 Tage Mindestsicherheitsbestand
   - Plane Nachfüllungen im Voraus
   - Berücksichtige saisonale Verfügbarkeit

3. **Optimiere Energieverbrauch**
   - Nutze FU für variable Dosierung
   - Minimiere Leerl aufzeiten
   - Richtige Dosierer-Dimensionierung

4. **Substratmischung optimieren**
   - Ausgewogene oTS-Belastung
   - Kostenoptimierung
   - Nährstoffbilanzierung

5. **Wartung planen**
   - Überwache Verschleißteile
   - Präventive Wartung für Dosierer
   - Lagerbehälter-Inspektionen

## Nächste Schritte

- [Biologische Komponenten](biological.md): Fermenter und Prozesssteuerung
- [Energiekomponenten](energy.md): BHKW und Wärmesysteme
- [Mechanische Komponenten](mechanical.md): Pumpen und Rührwerke
- [API-Referenz](../../api_reference/components/feeding.md): Detaillierte Klassendokumentation
