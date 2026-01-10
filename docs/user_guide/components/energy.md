# Energiekomponenten

Komponenten für Energieerzeugung, -umwandlung und -speicherung in Biogasanlagen.

## CHP (Combined Heat and Power)

Blockheizkraftwerk zur Umwandlung von Biogas in Strom und Wärme.

### Parameter

```python
configurator.add_chp(
    chp_id="chp_main",
    P_el_nom=500.0,        # Elektrische Nennleistung [kW]
    eta_el=0.40,           # Elektrischer Wirkungsgrad (40%)
    eta_th=0.45,           # Thermischer Wirkungsgrad (45%)
    name="Haupt-BHKW"
)
```

### Typische BHKW-Spezifikationen

| Typ | Größe [kW_el] | η_el | η_th | Gasbedarf [m³/d @ 60% CH4] |
|------|--------------|------|------|---------------------------|
| Klein | 100-250 | 0.38 | 0.48 | 600-1500 |
| Mittel | 250-750 | 0.40 | 0.45 | 1500-4500 |
| Groß | 750-2000 | 0.42 | 0.43 | 4500-12000 |

### Technologieoptionen

```python
# Gasmotor (am häufigsten)
chp_engine = configurator.add_chp(
    "chp1", P_el_nom=500, eta_el=0.40, eta_th=0.45
)

# Mikroturbine (100-500 kW)
chp_turbine = configurator.add_chp(
    "chp2", P_el_nom=250, eta_el=0.30, eta_th=0.55
)

# Hocheffizient (>1 MW)
chp_large = configurator.add_chp(
    "chp3", P_el_nom=1500, eta_el=0.42, eta_th=0.43
)
```

### Ausgaben

```python
{
    'P_el': 450.0,              # Elektrische Leistung [kW]
    'P_th': 506.3,              # Thermische Leistung [kW]
    'Q_gas_consumed': 2700.0,   # Gasverbrauch [m³/d]
    'load_factor': 0.90         # Betriebspunkt (0-1)
}
```

### Erweiterte BHKW-Steuerung

```python
# Variable Laststeuerung
inputs = {
    'Q_ch4': 800.0,           # Verfügbares Methan [m³/d]
    'load_setpoint': 0.75     # Betrieb bei 75% Kapazität
}
result = chp.step(t, dt, inputs)
```

### Leistungsberechnung

BHKW berechnen Leistung basierend auf verfügbarem Methan:

```python
# Methan-Energieinhalt: ~10 kWh/m³
E_ch4 = 10.0  # kWh/m³

# Verfügbare Leistung aus Methan
P_available = Q_ch4_available / 24.0 * E_ch4  # kW

# Elektrische Leistung
P_el = min(P_el_nom, P_available * eta_el)

# Thermische Leistung
P_th = P_el * eta_th / eta_el

# Gasverbrauch
Q_ch4_consumed = P_el / eta_el * 24.0 / E_ch4  # m³/d
```

## Heizsystem

Hält Fermentertemperatur mittels BHKW-Abwärme und Zusatzheizung.

### Parameter

```python
configurator.add_heating(
    heating_id="heating_main",
    target_temperature=308.15,      # Zieltemperatur [K]
    heat_loss_coefficient=0.5,      # Wärmeverlustkoeffizient [kW/K]
    name="Hauptfermenter-Heizung"
)
```

### Wärmeverlustkoeffizienten

| Isolierung | k [kW/K] | Beschreibung |
|------------|----------|-------------|
| Ausgezeichnet | 0.3-0.4 | Modern, gut isoliert |
| Gut | 0.4-0.6 | Standard-Isolierung |
| Schlecht | 0.6-1.0 | Alt oder minimale Isolierung |

### Ausgaben

```python
{
    'Q_heat_supplied': 125.5,    # Gesamt bereitgestellte Wärme [kW]
    'P_th_used': 110.0,          # Genutzte BHKW-Wärme [kW]
    'P_aux_heat': 15.5           # Benötigte Zusatzheizung [kW]
}
```

### Wärmebedarfsberechnung

```python
# Wärmebedarf = Wärmeverlust + Prozesswärme
Q_loss = k * (T_target - T_ambient)  # [kW]
Q_process = Q_feed * c_p * ΔT        # [kW]
Q_total = Q_loss + Q_process         # [kW]

# Nutze zuerst BHKW-Wärme, dann Zusatzheizung
if Q_total <= P_th_available:
    P_aux = 0
else:
    P_aux = Q_total - P_th_available
```

### Beispiel: Mehrstufige Heizung

```python
from pyadm1.configurator import BiogasPlant, PlantConfigurator
from pyadm1.substrates import Feedstock

feedstock = Feedstock(feeding_freq=48)
plant = BiogasPlant("Zweistufige Anlage")
config = PlantConfigurator(plant, feedstock)

# Zwei Fermenter mit unterschiedlichen Temperaturen
config.add_digester("hydro", V_liq=500, T_ad=328.15)  # 55°C
config.add_digester("main", V_liq=2000, T_ad=308.15)  # 35°C

# Ein BHKW
config.add_chp("chp1", P_el_nom=500)

# Separate Heizungen für jeden Fermenter
config.add_heating("heat_hydro", target_temperature=328.15, heat_loss_coefficient=0.3)
config.add_heating("heat_main", target_temperature=308.15, heat_loss_coefficient=0.5)

# Verbinde BHKW mit beiden Heizungen
config.auto_connect_chp_to_heating("chp1", "heat_hydro")
config.auto_connect_chp_to_heating("chp1", "heat_main")

# Simulieren
plant.initialize()
results = plant.simulate(duration=30, dt=1/24, save_interval=1.0)

# Analysiere Wärmeverteilung
final = results[-1]
print(f"Hydrolyse-Heizung: {final['components']['heat_hydro']['Q_heat_supplied']:.1f} kW")
print(f"Haupt-Heizung: {final['components']['heat_main']['Q_heat_supplied']:.1f} kW")
print(f"Zusatzheizung Hydrolyse: {final['components']['heat_hydro']['P_aux_heat']:.1f} kW")
print(f"Zusatzheizung Haupt: {final['components']['heat_main']['P_aux_heat']:.1f} kW")
```

## Gasspeicher

Biogasspeicher mit Druckmanagement (wird automatisch pro Fermenter erstellt).

### Typen

```python
from pyadm1.components.energy import GasStorage

# Niederdruckmembranspeicher (am häufigsten)
storage_membrane = GasStorage(
    component_id="storage1",
    storage_type="membrane",
    capacity_m3=1000.0,      # Kapazität bei STP [m³]
    p_min_bar=0.95,          # Minimaldruck [bar]
    p_max_bar=1.05,          # Maximaldruck [bar]
    initial_fill_fraction=0.1
)

# Domespeicher
storage_dome = GasStorage(
    component_id="storage2",
    storage_type="dome",
    capacity_m3=500.0,
    p_min_bar=0.98,
    p_max_bar=1.02
)

# Hochdruck-Komprimiertspeicher
storage_compressed = GasStorage(
    component_id="storage3",
    storage_type="compressed",
    capacity_m3=100.0,
    p_min_bar=10.0,
    p_max_bar=200.0
)
```

### Ausgaben

```python
{
    'stored_volume_m3': 450.0,       # Aktueller Speicherstand [m³ STP]
    'pressure_bar': 1.01,            # Aktueller Druck [bar]
    'utilization': 0.45,             # Füllstand (0-1)
    'vented_volume_m3': 0.0,         # Abgefackeltes Gas dieses Schritts [m³]
    'Q_gas_supplied_m3_per_day': 2700.0  # Bereitgestelltes Gas [m³/d]
}
```

### Druckmodell

Der Speicher schätzt den Druck basierend auf gespeichertem Volumen:

**Niederdruck (membrane/dome):**
```
p = p_atm + frac * (p_max - p_atm)
```

**Hochdruck (compressed):**
```
p = p_min + frac^α * (p_max - p_min)  # α > 1 für nichtlinearen Anstieg
```

### Sicherheitsventing

Bei Überdruck wird Gas automatisch zur Fackel geleitet:

```python
# Speicher überwacht Druck
if pressure > p_max:
    # Berechne zu entfernendes Volumen
    vent = stored_volume - target_volume
    # Leite zur Fackel
    vented_volume += vent
```

## Fackel

Sicherheitssystem zur Verbrennung von überschüssigem Biogas.

```python
from pyadm1.components.energy import Flare

flare = Flare(
    component_id="flare1",
    destruction_efficiency=0.98,  # 98% CH4 zerstört
    name="Notfackel"
)
```

### Ausgaben

```python
{
    'vented_volume_m3': 0.0,         # Verbranntes Volumen dieses Schritts [m³]
    'cumulative_vented_m3': 125.5,   # Kumulativ verbranntes Volumen [m³]
    'CH4_destroyed_m3': 0.0          # Zerstörtes CH4 dieses Schritts [m³]
}
```

### Fackelsteuerung

Die Fackel wird automatisch aktiviert, wenn:
- Gasspeicher Überdruck erreicht
- BHKW weniger Gas verbraucht als produziert
- Notabschaltung erforderlich ist

```python
# Gas vom Speicher zur Fackel
flare_inputs = {
    'Q_gas_in_m3_per_day': vented_gas,
    'CH4_fraction': 0.6  # 60% Methan im Biogas
}

result = flare.step(t, dt, flare_inputs)
print(f"Zerstörtes CH4: {result['CH4_destroyed_m3']:.2f} m³")
```

## Vollständiges Energiesystem

### Integrierte Energiekette

```python
from pyadm1.configurator import BiogasPlant, PlantConfigurator
from pyadm1.substrates import Feedstock

# Setup
feedstock = Feedstock(feeding_freq=48)
plant = BiogasPlant("Energieoptimierte Anlage")
config = PlantConfigurator(plant, feedstock)

# Fermenter
digester, storage = config.add_digester(
    "main_digester",
    V_liq=2000,
    V_gas=300,
    Q_substrates=[15, 10, 0, 0, 0, 0, 0, 0, 0, 0]
)

# BHKW
config.add_chp("chp1", P_el_nom=500, eta_el=0.40, eta_th=0.45)

# Heizung
config.add_heating("heat1", target_temperature=308.15, heat_loss_coefficient=0.5)

# Auto-Verbindungen (erstellt auch automatisch Fackel)
config.auto_connect_digester_to_chp("main_digester", "chp1")
config.auto_connect_chp_to_heating("chp1", "heat1")

# Simulieren
plant.initialize()
results = plant.simulate(duration=30, dt=1/24, save_interval=1.0)

# Energiebilanz
def energy_balance(results):
    """Berechne Energiebilanz der Anlage"""
    final = results[-1]
    comp = final['components']

    # Gasproduktion
    Q_gas = comp['main_digester']['Q_gas']  # m³/d
    Q_ch4 = comp['main_digester']['Q_ch4']  # m³/d
    E_gas = Q_ch4 * 10.0  # kWh/d (10 kWh/m³ CH4)

    # BHKW-Ausgabe
    P_el = comp['chp1']['P_el']  # kW
    P_th = comp['chp1']['P_th']  # kW
    E_el = P_el * 24  # kWh/d
    E_th = P_th * 24  # kWh/d

    # Wärmebedarf
    Q_heat = comp['heat1']['Q_heat_supplied']  # kW
    E_heat_needed = Q_heat * 24  # kWh/d

    # Wirkungsgrade
    eta_el_actual = E_el / E_gas if E_gas > 0 else 0
    eta_th_actual = E_th / E_gas if E_gas > 0 else 0
    eta_total = (E_el + E_th) / E_gas if E_gas > 0 else 0

    # Wärmenutzung
    heat_utilization = E_heat_needed / E_th if E_th > 0 else 0

    return {
        'E_gas': E_gas,
        'E_el': E_el,
        'E_th': E_th,
        'E_heat_needed': E_heat_needed,
        'eta_el': eta_el_actual,
        'eta_th': eta_th_actual,
        'eta_total': eta_total,
        'heat_utilization': heat_utilization,
        'excess_heat': max(0, E_th - E_heat_needed)
    }

balance = energy_balance(results)
print("\nEnergiebilanz:")
print(f"Gasenergie: {balance['E_gas']:.0f} kWh/d")
print(f"Strom: {balance['E_el']:.0f} kWh/d (η={balance['eta_el']:.1%})")
print(f"Wärme: {balance['E_th']:.0f} kWh/d (η={balance['eta_th']:.1%})")
print(f"Gesamtwirkungsgrad: {balance['eta_total']:.1%}")
print(f"Wärmenutzung: {balance['heat_utilization']:.1%}")
print(f"Überschusswärme: {balance['excess_heat']:.0f} kWh/d")
```

## Optimierungsstrategien

### 1. Wärmenutzungsoptimierung

```python
def optimize_heat_utilization(plant, results):
    """Optimiere Wärmenutzung durch Lastmanagement"""

    # Analysiere Wärmeüberschuss
    excess_heat = []
    for result in results:
        P_th = result['components']['chp1']['P_th']
        Q_heat = result['components']['heat1']['Q_heat_supplied']
        excess_heat.append(max(0, P_th - Q_heat))

    avg_excess = sum(excess_heat) / len(excess_heat)

    if avg_excess > 50:  # kW
        print(f"Durchschnittlicher Wärmeüberschuss: {avg_excess:.1f} kW")
        print("Optimierungsoptionen:")
        print("- Reduziere BHKW-Größe")
        print("- Füge zusätzliche Wärmenutzung hinzu (Trocknung, etc.)")
        print("- Nutze Wärmespeicher")

    return avg_excess

optimize_heat_utilization(plant, results)
```

### 2. Lastfolgebetrieb

```python
# BHKW-Laststeuerung basierend auf Gasproduktion
def load_following_control(Q_gas_available, P_el_nom):
    """Passe BHKW-Last an verfügbares Gas an"""

    # Minimale Last: 40% für stabile Verbrennung
    min_load = 0.4

    # Berechne optimale Last
    E_gas = Q_gas_available / 24 * 10  # kW
    load = min(1.0, max(min_load, E_gas / (P_el_nom / 0.40)))

    return load

# In Simulation anwenden
load_setpoint = load_following_control(Q_gas_available, 500)
chp_inputs = {
    'Q_gas_supplied_m3_per_day': Q_gas_available,
    'load_setpoint': load_setpoint
}
```

### 3. Gasspeicher-Management

```python
def manage_gas_storage(storage_state, chp_demand):
    """Optimiere Gasspeicher-Füllstand"""

    utilization = storage_state['utilization']
    pressure = storage_state['pressure_bar']

    # Ziel: 30-70% Füllstand für Flexibilität
    if utilization < 0.3:
        print("Niedriger Speicherstand - erhöhe Gasproduktion oder reduziere BHKW-Last")
        adjust_load = 0.8
    elif utilization > 0.7:
        print("Hoher Speicherstand - erhöhe BHKW-Last oder bereite Venting vor")
        adjust_load = 1.2
    else:
        adjust_load = 1.0

    return adjust_load

# In Simulation anwenden
load_adjustment = manage_gas_storage(storage.outputs_data, chp_demand)
```

## Leistungsmetriken

### BHKW-Verfügbarkeit

```python
def calculate_chp_availability(results):
    """Berechne BHKW-Verfügbarkeit und Nutzung"""

    total_hours = len(results) / 24  # Tage * 24 h
    running_hours = sum(1 for r in results if r['components']['chp1']['P_el'] > 0) / 24

    availability = running_hours / total_hours

    # Lastfaktor
    load_factors = [r['components']['chp1']['load_factor'] for r in results]
    avg_load = sum(load_factors) / len(load_factors)

    return {
        'availability': availability,
        'running_hours': running_hours,
        'avg_load': avg_load,
        'full_load_hours': running_hours * avg_load
    }

metrics = calculate_chp_availability(results)
print(f"BHKW-Verfügbarkeit: {metrics['availability']:.1%}")
print(f"Betriebsstunden: {metrics['running_hours']:.0f} h")
print(f"Durchschn. Last: {metrics['avg_load']:.1%}")
print(f"Volllaststunden: {metrics['full_load_hours']:.0f} h")
```

## Fehlerbehebung

### Problem: BHKW läuft nicht

**Diagnose:**
```python
chp_outputs = chp.step(t, dt, inputs)

if chp_outputs['P_el'] == 0:
    print("BHKW läuft nicht - prüfe:")
    print(f"- Verfügbares Gas: {inputs.get('Q_gas_supplied_m3_per_day', 0):.1f} m³/d")
    print(f"- Mindestgasbedarf: {P_el_nom / eta_el * 24 / 10:.1f} m³/d")
    print(f"- Speicherdruck: {storage.outputs_data['pressure_bar']:.2f} bar")
```

### Problem: Übermäßiges Venting

**Ursache:** Gasproduktion > BHKW-Verbrauch

**Lösung:**
```python
# Option 1: Erhöhe BHKW-Kapazität
config.add_chp("chp1", P_el_nom=750)  # Von 500 auf 750 kW

# Option 2: Füge zweites BHKW hinzu
config.add_chp("chp2", P_el_nom=250)

# Option 3: Vergrößere Gasspeicher
storage = GasStorage("storage1", capacity_m3=1500)  # Von 1000 auf 1500
```

### Problem: Unzureichende Wärme

**Diagnose:**
```python
heat_outputs = heating.step(t, dt, inputs)

if heat_outputs['P_aux_heat'] > 50:  # kW Zusatzheizung
    print("Hoher Zusatzheizungsbedarf:")
    print(f"- BHKW-Wärme: {heat_outputs['P_th_used']:.1f} kW")
    print(f"- Zusatzheizung: {heat_outputs['P_aux_heat']:.1f} kW")
    print("Lösungen:")
    print("- Verbessere Isolierung (reduziere k)")
    print("- Vergrößere BHKW")
    print("- Reduziere Fermentertemperatur")
```

## Nächste Schritte

- [Biologische Komponenten](biological.md): Fermenter und Prozesssteuerung
- [Mechanische Komponenten](mechanical.md): Pumpen und Rührwerke
- [Fütterungskomponenten](feeding.md): Lagerung und Dosierung
- [API-Referenz](../../api_reference/components/energy.md): Detaillierte Klassendokumentation
