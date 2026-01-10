# Mechanische Komponenten

Mechanische Anlagenkomponenten für Materialhandhabung und Prozesssteuerung.

## Pump

Pumpen für Substratförderung, Rezirkulation und Gärrestverarbeitung in Biogasanlagen.

### Parameter

```python
from pyadm1.components.mechanical import Pump

pump = Pump(
    component_id="pump1",
    pump_type="progressive_cavity",  # oder "centrifugal", "piston"
    Q_nom=15.0,                      # Nenn-Durchflussrate [m³/h]
    pressure_head=50.0,              # Auslegungsdruck [m]
    efficiency=None,                 # Auto-berechnet wenn None
    motor_efficiency=0.90,           # Motorwirkungsgrad (0-1)
    fluid_density=1020.0,            # Fluiddichte [kg/m³]
    speed_control=True,              # Frequenzumrichter
    name="Förderpumpe"
)
```

### Pumpentypen-Vergleich

| Typ | Beste Anwendung | Wirkungsgrad | Vorteile | Nachteile |
|------|-----------------|------------|------------|---------------|
| **Kreiselpumpe** | Niederviskose Flüssigkeiten | 65-75% | • Hohe Durchflussraten<br>• Robust<br>• Wartungsarm | • Nicht selbstansaugend<br>• Schlecht bei hoher Viskosität<br>• Wirkungsgrad sinkt bei Feststoffen |
| **Exzenterschneckenpumpe** | Viskose Schlämme | 50-70% | • Handhabt hohe Feststoffgehalte<br>• Selbstansaugend<br>• Schonende Förderung | • Niedrigerer Wirkungsgrad<br>• Höherer Wartungsaufwand<br>• Drehzahlabhängiger Druck |
| **Kolbenpumpe** | Hochdruckanwendungen | 70-85% | • Hohe Druckfähigkeit<br>• Präzise Durchflusskontrolle<br>• Guter Wirkungsgrad | • Höhere Kosten<br>• Komplexer<br>• Empfindlich gegen Feststoffe |

### Dimensionierungsrichtlinien

**Durchflussraten-Auswahl:**

| Anlagengröße | Substratfütterung [m³/d] | Pumpe Q_nom [m³/h] | Typischer Druck [m] |
|------------|----------------------|-------------------|---------------------|
| Klein | 10-25 | 5-15 | 30-50 |
| Mittel | 25-75 | 15-40 | 40-60 |
| Groß | 75-200 | 40-100 | 50-80 |

**Druckhöhen-Überlegungen:**

```python
# Berechne erforderliche Druckhöhe
H_static = 5.0      # Vertikaler Hub [m]
H_friction = 8.0    # Rohrreibungsverluste [m]
H_process = 2.0     # Prozessdruck [m]
H_safety = 1.2      # Sicherheitsfaktor

H_required = (H_static + H_friction + H_process) * H_safety
# = 18.0 m

pump = Pump("pump1", Q_nom=15, pressure_head=H_required)
```

### Ausgaben

```python
{
    'P_consumed': 8.5,           # Leistungsaufnahme [kW]
    'Q_actual': 10.0,            # Tatsächliche Durchflussrate [m³/h]
    'is_running': True,          # Betriebszustand
    'efficiency': 0.68,          # Aktueller Wirkungsgrad
    'pressure_actual': 48.5,     # Tatsächlicher Druck [m]
    'speed_fraction': 1.0,       # Drehzahl (0-1)
    'specific_energy': 0.85      # Energie pro Volumen [kWh/m³]
}
```

### Verwendungsbeispiel

```python
# Exzenterschneckenpumpe für Substratförderung
pump = Pump(
    component_id="feed_pump",
    pump_type="progressive_cavity",
    Q_nom=15.0,
    pressure_head=50.0,
    speed_control=True
)

pump.initialize()

# Betrieb bei 80% Kapazität
result = pump.step(
    t=0,
    dt=1/24,
    inputs={
        'Q_setpoint': 12.0,
        'enable_pump': True,
        'fluid_density': 1020,
        'pressure_head': 50
    }
)

print(f"Leistung: {result['P_consumed']:.1f} kW")
print(f"Durchfluss: {result['Q_actual']:.1f} m³/h")
print(f"Wirkungsgrad: {result['efficiency']:.1%}")
```

### Leistungsaufnahme

Pumpen berechnen Leistung basierend auf hydraulischer Formel:

```
P_hydraulic = ρ × g × Q × H / 1000  [kW]
P_shaft = P_hydraulic / η_pump
P_electrical = P_shaft / η_motor
```

Wobei:
- ρ = Fluiddichte [kg/m³]
- g = 9.81 m/s²
- Q = Durchflussrate [m³/s]
- H = Druckhöhe [m]
- η = Wirkungsgrad

**Typische Leistungsaufnahme:**

| Durchfluss [m³/h] | Druckhöhe [m] | Pumpentyp | Leistung [kW] |
|-------------|----------|-----------|------------|
| 10 | 30 | Kreiselpumpe | 2.5 |
| 10 | 50 | Exzenterschnecke | 4.5 |
| 15 | 50 | Exzenterschnecke | 6.8 |
| 20 | 60 | Kolbenpumpe | 10.5 |

## Mixer

Rührwerke und Agitatoren zur Aufrechterhaltung der Homogenität in anaeroben Fermentern.

### Parameter

```python
from pyadm1.components.mechanical import Mixer

mixer = Mixer(
    component_id="mix1",
    mixer_type="propeller",          # oder "paddle", "jet"
    tank_volume=2000.0,              # Tankvolumen [m³]
    tank_diameter=None,              # Auto-berechnet wenn None
    mixing_intensity="medium",       # "low", "medium", "high"
    power_installed=None,            # Auto-berechnet wenn None
    intermittent=True,               # Intermittierender Betrieb
    on_time_fraction=0.25,           # 25% Einschaltzeit
    name="Hauptrührwerk"
)
```

### Rührwerkstypen

| Typ | Strömungsmuster | Beste Anwendung | Leistungsfaktor | Typische Drehzahl [rpm] |
|------|--------------|----------|--------------|-------------------|
| **Propeller** | Axial | Große Tanks, flüssige Substrate | 1.0× | 40-100 |
| **Paddelrührwerk** | Radial | Hohe Feststoffe, faseriges Material | 1.2× | 20-60 |
| **Strahlmischer** | Hydraulisch | Rezirkulationsmischung | 1.5× | N/A (Pumpe) |

### Mischintensität

| Intensität | Spezifische Leistung [W/m³] | Mischzeit [min] | Anwendung |
|-----------|----------------------|-------------------|-------------|
| **Niedrig** | 3 | 15-30 | Flüssige Gülle, niedrige Feststoffe |
| **Mittel** | 5 | 8-15 | Standardbetrieb, Energiepflanzen |
| **Hoch** | 8 | 3-8 | Hohe Feststoffe, faserreiche Substrate |

### Dimensionierungsbeispiel

```python
# Für 2000 m³ Fermenter mit mittlerer Intensität
tank_volume = 2000  # m³
specific_power = 5  # W/m³ für mittlere Intensität

P_required = (tank_volume * specific_power) / 1000  # kW
P_required *= 1.2  # Sicherheitsfaktor für faserreiches Material
# = 12 kW

mixer = Mixer(
    "mix1",
    tank_volume=2000,
    mixing_intensity="medium",
    power_installed=15  # Auf Standardgröße aufrunden
)
```

### Ausgaben

```python
{
    'P_consumed': 12.5,          # Aktuelle Leistung [kW]
    'P_average': 3.1,            # Zeitgemittelt [kW]
    'is_running': True,          # Betriebszustand
    'mixing_quality': 0.85,      # Qualitätsindex (0-1)
    'reynolds_number': 15000,    # Strömungsregime-Indikator
    'power_number': 0.32,        # Dimensionslose Leistungszahl
    'mixing_time': 8.5,          # Zeit bis Homogenität [min]
    'shear_rate': 45.2,          # Durchschn. Scherrate [1/s]
    'specific_power': 6.25,      # Leistungsdichte [kW/m³]
    'tip_speed': 2.8             # Rührblatt-Spitzengeschwindigkeit [m/s]
}
```

### Verwendungsbeispiel

```python
# Propellerrührwerk mittlerer Intensität
mixer = Mixer(
    component_id="mix1",
    mixer_type="propeller",
    tank_volume=2000,
    mixing_intensity="medium",
    intermittent=True,
    on_time_fraction=0.25  # 6 Stunden pro Tag
)

mixer.initialize()

result = mixer.step(
    t=0,
    dt=1/24,
    inputs={
        'enable_mixing': True,
        'speed_setpoint': 1.0,
        'fluid_viscosity': 0.05  # Pa·s
    }
)

print(f"Leistung: {result['P_consumed']:.1f} kW")
print(f"Durchschn. Leistung: {result['P_average']:.1f} kW")
print(f"Mischqualität: {result['mixing_quality']:.2f}")
print(f"Mischzeit: {result['mixing_time']:.1f} min")
```

### Intermittierender Betrieb

Intermittierende Mischung reduziert Energieverbrauch:

```python
# Vergleich kontinuierlich vs. intermittierend
# Kontinuierlich: 15 kW × 24 h = 360 kWh/Tag
# Intermittierend (25%): 15 kW × 6 h = 90 kWh/Tag
# Einsparung: 270 kWh/Tag (75%)

mixer_continuous = Mixer(
    "mix_cont",
    tank_volume=2000,
    intermittent=False
)

mixer_intermittent = Mixer(
    "mix_int",
    tank_volume=2000,
    intermittent=True,
    on_time_fraction=0.25
)

# Beide erreichen ähnliche Mischqualität
```

**Empfohlene Einschaltzeiten:**

| Substrattyp | Einschaltzeit | Gesamtstunden/Tag | Energieeinsparung |
|----------------|---------|-----------------|----------------|
| Flüssige Gülle | 15-20% | 3.6-4.8 h | 80-85% |
| Energiepflanzen | 20-30% | 4.8-7.2 h | 70-80% |
| Hohe Feststoffe | 25-35% | 6.0-8.4 h | 65-75% |
| Faserreiches Material | 30-40% | 7.2-9.6 h | 60-70% |

### Mischleistungsberechnung

Rührwerke berechnen Leistung basierend auf Leistungszahl-Korrelation:

```python
# Mechanische Leistung aus Leistungszahl-Korrelation
# P = Np * ρ * N³ * D⁵

N = operating_speed / 60.0  # Hz (Umdrehungen/s)
D = impeller_diameter       # m

P_mech = power_number * fluid_density * N**3 * D**5 / 1000.0  # kW

# Berücksichtige Motorwirkungsgrad (typisch 85-95%)
motor_efficiency = 0.90
P_electrical = P_mech / motor_efficiency

# Limitiere auf installierte Leistung
P_actual = min(P_electrical, power_installed)
```

**Reynolds-Zahl für Mischen:**
```
Re = ρ * N * D² / μ

wobei:
- ρ = Fluiddichte [kg/m³]
- N = Drehzahl [Hz]
- D = Rührblattdurchmesser [m]
- μ = Viskosität [Pa·s]
```

**Leistungszahl (abhängig von Rührwerkstyp und Reynolds-Zahl):**

- **Propeller:**
  - Laminar (Re < 100): Np = 14.0 * Re^(-0.67)
  - Übergang (100 < Re < 10000): Np = 1.2 * Re^(-0.15)
  - Turbulent (Re > 10000): Np = 0.32

- **Paddelrührwerk:**
  - Laminar (Re < 10): Np = 300.0 / Re
  - Übergang (10 < Re < 10000): Np = 8.0 * Re^(-0.25)
  - Turbulent (Re > 10000): Np = 5.0

### Mischzeit-Schätzung

Basierend auf Nienow-Korrelation:

```
θ_mix = C * (D_T/D)^α * (H/D_T)^β / N

wobei:
- C, α, β = konstanten abhängig vom Rührwerkstyp
- D_T = Tankdurchmesser [m]
- D = Rührblattdurchmesser [m]
- H = Tankhöhe [m]
- N = Drehzahl [Hz]
```

**Typische Konstanten:**

| Rührwerkstyp | C | α | β |
|---------|---|-----|-----|
| Propeller | 5.3 | 2.0 | 0.5 |
| Paddelrührwerk | 6.5 | 2.5 | 0.7 |
| Strahlmischer | 4.0 | 1.5 | 0.3 |

## Integrationsbeispiel

### Komplette Pumpen- und Mischkette

```python
from pyadm1.configurator import BiogasPlant, PlantConfigurator
from pyadm1.components.mechanical import Pump, Mixer
from pyadm1.substrates import Feedstock

# Setup
feedstock = Feedstock(feeding_freq=48)
plant = BiogasPlant("Mechanische Systemanlage")
config = PlantConfigurator(plant, feedstock)

# 1. Substratförderungspumpe
feed_pump = Pump(
    "feed_pump",
    pump_type="progressive_cavity",
    Q_nom=15.0,
    pressure_head=50.0,
    speed_control=True
)
plant.add_component(feed_pump)

# 2. Fermenter
digester, storage = config.add_digester(
    "main_digester",
    V_liq=2000,
    Q_substrates=[15, 10, 0, 0, 0, 0, 0, 0, 0, 0]
)

# 3. Hauptrührwerk
main_mixer = Mixer(
    "main_mixer",
    mixer_type="propeller",
    tank_volume=2000,
    mixing_intensity="medium",
    intermittent=True,
    on_time_fraction=0.25
)
plant.add_component(main_mixer)

# 4. Rezirkulationspumpe
recirc_pump = Pump(
    "recirc_pump",
    pump_type="centrifugal",
    Q_nom=50.0,  # Höherer Durchfluss für Rezirkulation
    pressure_head=10.0,  # Niedrigerer Druck
    speed_control=True
)
plant.add_component(recirc_pump)

# 5. Gärrestpumpe
digestate_pump = Pump(
    "digestate_pump",
    pump_type="progressive_cavity",
    Q_nom=20.0,
    pressure_head=30.0
)
plant.add_component(digestate_pump)

# Energiesystem
config.add_chp("chp1", P_el_nom=500)
config.add_heating("heat1", target_temperature=308.15)

# Verbindungen
config.connect("feed_pump", "main_digester", "liquid")
config.auto_connect_digester_to_chp("main_digester", "chp1")
config.auto_connect_chp_to_heating("chp1", "heat1")

# Simulieren
plant.initialize()
results = plant.simulate(duration=30, dt=1/24, save_interval=1.0)

# Mechanische Energieanalyse
def mechanical_energy_analysis(results):
    """Analysiere mechanischen Energieverbrauch"""
    final = results[-1]
    comp = final['components']

    # Pumpenenergie
    feed_pump_energy = comp['feed_pump']['energy_consumed']
    recirc_pump_energy = comp['recirc_pump']['energy_consumed']
    digestate_pump_energy = comp['digestate_pump']['energy_consumed']
    total_pump_energy = feed_pump_energy + recirc_pump_energy + digestate_pump_energy

    # Rührwerkenergie
    mixer_energy = comp['main_mixer']['energy_consumed']

    # Gesamtmechanische Energie
    total_mech_energy = total_pump_energy + mixer_energy

    # BHKW-Produktion
    chp_energy = comp['chp1']['P_el'] * 30 * 24  # kWh

    # Parasitäre Last
    parasitic_fraction = total_mech_energy / chp_energy if chp_energy > 0 else 0

    return {
        'feed_pump': feed_pump_energy,
        'recirc_pump': recirc_pump_energy,
        'digestate_pump': digestate_pump_energy,
        'total_pump': total_pump_energy,
        'mixer': mixer_energy,
        'total_mechanical': total_mech_energy,
        'chp_production': chp_energy,
        'parasitic_fraction': parasitic_fraction,
        'net_energy': chp_energy - total_mech_energy
    }

analysis = mechanical_energy_analysis(results)
print("\nMechanische Energieanalyse:")
print(f"Förderpumpe: {analysis['feed_pump']:.0f} kWh")
print(f"Rezirkulationspumpe: {analysis['recirc_pump']:.0f} kWh")
print(f"Gärrestpumpe: {analysis['digestate_pump']:.0f} kWh")
print(f"Gesamt Pumpen: {analysis['total_pump']:.0f} kWh")
print(f"Rührwerk: {analysis['mixer']:.0f} kWh")
print(f"Gesamt mechanisch: {analysis['total_mechanical']:.0f} kWh")
print(f"Parasitäre Last: {analysis['parasitic_fraction']:.1%}")
print(f"Nettoenergieproduktion: {analysis['net_energy']:.0f} kWh")
```

## Optimierungsstrategien

### 1. Pumpenoptimierung

```python
def optimize_pump_sizing(Q_required, H_required, pump_type="progressive_cavity"):
    """Optimiere Pumpendimensionierung für Effizienz"""

    # Dimensioniere für 80-90% Nennlast (höchster Wirkungsgrad)
    Q_nom = Q_required / 0.85

    # Füge Sicherheitsmarge für Druckhöhe hinzu
    H_nom = H_required * 1.2

    pump = Pump(
        "optimized_pump",
        pump_type=pump_type,
        Q_nom=Q_nom,
        pressure_head=H_nom,
        speed_control=True  # FU für Teillastoptimierung
    )

    return pump

# Beispiel: 12 m³/h erforderlich bei 40 m Druckhöhe
optimized = optimize_pump_sizing(12.0, 40.0)
print(f"Optimierte Pumpengröße: {optimized.Q_nom:.1f} m³/h")
```

### 2. Mischoptimierung

```python
def optimize_mixing_strategy(substrate_type, tank_volume):
    """Wähle optimale Mischstrategie basierend auf Substrat"""

    strategies = {
        'liquid_manure': {
            'intensity': 'low',
            'on_time_fraction': 0.20,
            'mixer_type': 'propeller'
        },
        'energy_crops': {
            'intensity': 'medium',
            'on_time_fraction': 0.25,
            'mixer_type': 'propeller'
        },
        'high_solids': {
            'intensity': 'medium',
            'on_time_fraction': 0.30,
            'mixer_type': 'paddle'
        },
        'fibrous': {
            'intensity': 'high',
            'on_time_fraction': 0.35,
            'mixer_type': 'paddle'
        }
    }

    strategy = strategies.get(substrate_type, strategies['energy_crops'])

    mixer = Mixer(
        "optimized_mixer",
        mixer_type=strategy['mixer_type'],
        tank_volume=tank_volume,
        mixing_intensity=strategy['intensity'],
        intermittent=True,
        on_time_fraction=strategy['on_time_fraction']
    )

    return mixer, strategy

# Beispiel: Faserreiche Substrate
mixer, strategy = optimize_mixing_strategy('fibrous', 2000)
print(f"Optimierte Mischstrategie:")
print(f"- Typ: {strategy['mixer_type']}")
print(f"- Intensität: {strategy['intensity']}")
print(f"- Einschaltzeit: {strategy['on_time_fraction']:.0%}")
```

### 3. Energieminimierung

```python
def minimize_mechanical_energy(plant_config):
    """Strategien zur Minimierung mechanischer Energie"""

    strategies = []

    # 1. Nutze intermittierend Mischen
    strategies.append({
        'name': 'Intermittierendes Mischen',
        'saving': 0.70,  # 70% Einsparung
        'implementation': 'on_time_fraction=0.25'
    })

    # 2. Frequenzumrichter für Pumpen
    strategies.append({
        'name': 'FU für Teillastbetrieb',
        'saving': 0.30,  # 30% Einsparung bei Teillast
        'implementation': 'speed_control=True'
    })

    # 3. Richtige Dimensionierung
    strategies.append({
        'name': 'Optimale Dimensionierung',
        'saving': 0.15,  # 15% durch Wirkungsgradoptimierung
        'implementation': 'Q_nom = Q_required / 0.85'
    })

    # 4. Niedrigere Mischintensität wo möglich
    strategies.append({
        'name': 'Angepasste Mischintensität',
        'saving': 0.40,  # 40% durch niedrigere Intensität
        'implementation': 'mixing_intensity="low" für Flüssigsubstrate'
    })

    total_potential = sum(s['saving'] for s in strategies)

    print("Energieminimierungsstrategien:")
    for s in strategies:
        print(f"- {s['name']}: {s['saving']:.0%} Einsparung")
        print(f"  Umsetzung: {s['implementation']}")

    return strategies

strategies = minimize_mechanical_energy({})
```

## Fehlerbehebung

### Problem: Pumpe liefert unzureichenden Durchfluss

**Diagnose:**
```python
pump_result = pump.step(0, 1/24, {'Q_setpoint': 15, 'enable_pump': True})

if pump_result['Q_actual'] < 0.8 * pump_result.get('Q_setpoint', 15):
    print("Niedriger Pumpendurchfluss - prüfe:")
    print(f"- Aktuelle Effizienz: {pump_result['efficiency']:.1%}")
    print(f"- Druckhöhe: {pump_result['pressure_actual']:.1f} m")
    print(f"- Ist Pumpe richtig dimensioniert für Anwendung?")

    # Prüfe ob Überlast
    if pump.speed_fraction > 1.0:
        print("- WARNUNG: Pumpe überlastet!")
```

**Lösungen:**
- Erhöhe Pumpengröße wenn konsistent überlastet
- Reduziere Reibungsverluste in Rohrleitungen
- Prüfe auf Blockaden oder Verschleiß

### Problem: Rührwerk verbraucht zu viel Energie

**Diagnose:**
```python
mixer_result = mixer.step(0, 1/24, {})

specific_power = mixer_result['P_consumed'] / mixer.tank_volume  # kW/m³

if specific_power > 6.0:  # Obergrenze für mittlere Intensität
    print(f"Hohe spezifische Leistung: {specific_power:.1f} W/m³")
    print("Optimierungsoptionen:")

    if not mixer.intermittent:
        print("- Aktiviere intermittierenden Betrieb (70% Einsparung)")

    if mixer.mixing_intensity == "high":
        print("- Reduziere auf mittlere Intensität wenn möglich")
```

**Lösungen:**
```python
# Implementiere intermittierenden Betrieb
mixer_optimized = Mixer(
    "mix1",
    tank_volume=mixer.tank_volume,
    mixing_intensity="medium",
    intermittent=True,
    on_time_fraction=0.25  # 75% Energieeinsparung
)
```

### Problem: Schlechte Mischqualität

**Diagnose:**
```python
if mixer_result['mixing_quality'] < 0.7:
    print(f"Niedrige Mischqualität: {mixer_result['mixing_quality']:.2f}")
    print(f"Mischzeit: {mixer_result['mixing_time']:.1f} min")
    print(f"Reynolds-Zahl: {mixer_result['reynolds_number']:.0f}")

    if mixer_result['reynolds_number'] < 1000:
        print("- Laminare Strömung - erhöhe Drehzahl oder Rührblattgröße")

    if mixer_result['mixing_time'] > 30:
        print("- Lange Mischzeit - erhöhe Intensität oder Einschaltzeit")
```

**Lösungen:**
- Erhöhe Mischintensität für schwierige Substrate
- Verlängere Einschaltzeit bei intermittierendem Betrieb
- Erwäge größeres Rührblatt oder höhere Drehzahl

## Best Practices

1. **Pumpen für optimalen Wirkungsgrad dimensionieren**
   - Betreibe bei 80-90% Nennlast
   - Nutze FU für variable Lastanforderungen

2. **Implementiere intermittierendes Mischen**
   - 25% Einschaltzeit für die meisten Anwendungen
   - Passe an Substrattyp an

3. **Regelmäßige Wartung**
   - Überwache Pumpenwirkungsgrad über Zeit
   - Prüfe Rührwerkverschleiß

4. **Optimiere Systemdesign**
   - Minimiere Rohrleitungsverlu ste
   - Richtige Pumpenplatzierung

5. **Überwache Energieverbrauch**
   - Verfolge parasitäre Last
   - Ziel: <10% der BHKW-Produktion

## Nächste Schritte

- [Biologische Komponenten](biological.md): Fermenter und Prozesssteuerung
- [Energiekomponenten](energy.md): BHKW und Wärmesysteme
- [Fütterungskomponenten](feeding.md): Lagerung und Dosierung
- [API-Referenz](../../api_reference/components/mechanical.md): Detaillierte Klassendokumentation
