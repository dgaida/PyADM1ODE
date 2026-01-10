# Biologische Komponenten

Komponenten für biologische Umwandlungsprozesse in Biogasanlagen.

## Digester

Der Hauptfermenter, der das ADM1-Modell für anaerobe Vergärung implementiert.

### Parameter

```python
from pyadm1.configurator.plant_configurator import PlantConfigurator

configurator.add_digester(
    digester_id="main_digester",      # Eindeutige Kennung
    V_liq=2000.0,                     # Flüssigvolumen [m³]
    V_gas=300.0,                      # Gasraum [m³]
    T_ad=308.15,                      # Betriebstemperatur [K]
    name="Hauptfermenter",            # Lesbarer Name
    load_initial_state=True,          # Lade Steady-State-Initialisierung
    initial_state_file=None,          # Benutzerdefinierter Anfangszustand CSV (optional)
    Q_substrates=[15, 10, 0, ...]    # Substrat-Fütterungsraten [m³/d]
)
```

### Dimensionierungsrichtlinien

| Anlagengröße | V_liq [m³] | V_gas [m³] | Fütterungsrate [m³/d] | HRT [Tage] |
|--------------|------------|------------|------------------|------------|
| Klein        | 300-800    | 50-120     | 10-25            | 20-40      |
| Mittel       | 1000-3000  | 150-450    | 25-75            | 25-45      |
| Groß         | 3000-8000  | 450-1200   | 75-200           | 30-50      |

### Temperaturoptionen

```python
# Psychrophil (selten in der Praxis)
T_psychro = 298.15  # 25°C

# Mesophil (am häufigsten)
T_meso = 308.15     # 35°C

# Thermophil (faserreiche Substrate)
T_thermo = 328.15   # 55°C
```

### Ausgaben

```python
outputs = digester.step(t, dt, inputs)
# Rückgabe:
{
    'Q_out': 25.0,              # Auslaufdurchfluss [m³/d]
    'state_out': [...],         # ADM1-Zustand für nächsten Fermenter
    'Q_gas': 1250.5,           # Biogasproduktion [m³/d]
    'Q_ch4': 750.3,            # Methanproduktion [m³/d]
    'Q_co2': 475.2,            # CO2-Produktion [m³/d]
    'pH': 7.32,                # pH-Wert
    'VFA': 2.45,               # Flüchtige Fettsäuren [g/L]
    'TAC': 8.50,               # Gesamtalkalinität [g CaCO3/L]
    'gas_storage': {           # Angeschlossene Gasspeicherinfo
        'stored_volume_m3': 150.0,
        'pressure_bar': 1.02,
        'vented_volume_m3': 0.0
    }
}
```

### Erweiterte Verwendung

**Mehrere Fermenter in Reihe:**
```python
# Hydrolyse + Methanogenese
configurator.add_digester("hydro", V_liq=500, T_ad=318.15,
                         Q_substrates=[15, 10, 0, 0, 0, 0, 0, 0, 0, 0])
configurator.add_digester("main", V_liq=2000, T_ad=308.15,
                         Q_substrates=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
configurator.connect("hydro", "main", "liquid")
```

**Benutzerdefinierter Anfangszustand:**
```python
import pandas as pd

# Erstelle benutzerdefinierten Zustand
initial = pd.DataFrame({
    'S_su': [0.01], 'S_aa': [0.001], # ... alle 37 Zustandsvariablen
})
initial.to_csv('custom_state.csv', index=False)

# Verwende im Fermenter
configurator.add_digester(
    "dig1", V_liq=2000,
    initial_state_file='custom_state.csv'
)
```

### Kalibrierungsparameter

Der Fermenter unterstützt die Anwendung von Kalibrierungsparametern zur Modellanpassung:

```python
from pyadm1.components.biological import Digester

digester = Digester("dig1", feedstock, V_liq=2000)

# Kalibrierungsparameter anwenden
digester.apply_calibration_parameters({
    'k_dis': 0.55,      # Desintegrationsrate
    'Y_su': 0.105,      # Ertragskoeffizient für Zucker
    'k_hyd_ch': 11.0    # Hydrolyserate für Kohlenhydrate
})

# Aktuelle Parameter abrufen
params = digester.get_calibration_parameters()
print(params)

# Parameter löschen (zurück zu Standardwerten)
digester.clear_calibration_parameters()
```

## Hydrolysis

Vorbehandlungstank für hydrolysedominierte Prozesse (Stub für zukünftige Implementierung).

```python
from pyadm1.components.biological import Hydrolysis

hydrolysis = Hydrolysis(
    component_id="hydro1",
    feedstock=feedstock,
    V_liq=500.0,
    T_ad=318.15  # Höhere Temperatur für schnellere Hydrolyse
)
```

**Anwendung**: Nützlich für Substrate mit hohem lignozellulosem Gehalt, kann bei anderen Temperaturen und Verweilzeiten als der Hauptfermenter betrieben werden.

## Separator

Fest-Flüssig-Trennung für Gärrestverarbeitung (Stub für zukünftige Implementierung).

```python
from pyadm1.components.biological import Separator

separator = Separator(
    component_id="sep1",
    separation_efficiency=0.95  # 95% Feststoffabtrennung
)
```

**Anwendung**: Modelliert mechanische (Schneckenpresse, Zentrifuge) oder Schwerkrafttrennung mit konfigurierbarer Trenneffizienz.

## Beispiel: Zweistufiges Vergärungssystem

```python
from pyadm1.configurator import BiogasPlant, PlantConfigurator
from pyadm1.substrates import Feedstock

# Setup
feedstock = Feedstock(feeding_freq=48)
plant = BiogasPlant("Zweistufige Anlage")
config = PlantConfigurator(plant, feedstock)

# Stufe 1: Thermophile Hydrolyse
hydro, hydro_storage = config.add_digester(
    "hydrolysis",
    V_liq=500,
    V_gas=75,
    T_ad=328.15,  # 55°C
    Q_substrates=[15, 10, 0, 0, 0, 0, 0, 0, 0, 0]
)

# Stufe 2: Mesophile Methanogenese
main, main_storage = config.add_digester(
    "methanogenesis",
    V_liq=2000,
    V_gas=300,
    T_ad=308.15,  # 35°C
    Q_substrates=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Nur Input aus Stufe 1
)

# Verbinde Fermenter
config.connect("hydrolysis", "methanogenesis", "liquid")

# Energiesystem
config.add_chp("chp1", P_el_nom=500)
config.add_heating("heat_hydro", target_temperature=328.15, heat_loss_coefficient=0.3)
config.add_heating("heat_main", target_temperature=308.15, heat_loss_coefficient=0.5)

# Auto-Verbindungen
config.auto_connect_digester_to_chp("hydrolysis", "chp1")
config.auto_connect_digester_to_chp("methanogenesis", "chp1")
config.auto_connect_chp_to_heating("chp1", "heat_hydro")
config.auto_connect_chp_to_heating("chp1", "heat_main")

# Simulieren
plant.initialize()
results = plant.simulate(duration=100, dt=1/24, save_interval=1.0)

# Ergebnisse analysieren
final = results[-1]
print(f"Hydrolyse-Biogas: {final['components']['hydrolysis']['Q_gas']:.1f} m³/d")
print(f"Haupt-Biogas: {final['components']['methanogenesis']['Q_gas']:.1f} m³/d")
print(f"Gesamt-Methan: {final['components']['hydrolysis']['Q_ch4'] + final['components']['methanogenesis']['Q_ch4']:.1f} m³/d")
print(f"pH Hydrolyse: {final['components']['hydrolysis']['pH']:.2f}")
print(f"pH Haupt: {final['components']['methanogenesis']['pH']:.2f}")
```

## Prozessüberwachung

### Wichtige Prozessindikatoren

```python
def monitor_digester_health(results):
    """Überwache Fermentergesundheit anhand von Prozessindikatoren"""

    for result in results:
        digester_data = result['components']['main_digester']

        # pH-Wert prüfen
        pH = digester_data['pH']
        if pH < 6.8:
            print(f"Warnung: Niedriger pH ({pH:.2f}) - Übersäuerungsrisiko")
        elif pH > 8.0:
            print(f"Warnung: Hoher pH ({pH:.2f}) - Mögliche Ammoniakinhibition")

        # VFA/TAC-Verhältnis
        VFA = digester_data['VFA']  # g HAc-eq/L
        TAC = digester_data['TAC']  # g CaCO3-eq/L

        # Umrechnung für VFA/TAC-Verhältnis
        VFA_TAC = VFA / TAC if TAC > 0 else 0

        if VFA_TAC > 0.4:
            print(f"Warnung: Hohes VFA/TAC-Verhältnis ({VFA_TAC:.2f}) - Prozessinstabilität")

        # Gasproduktion
        Q_gas = digester_data['Q_gas']
        if Q_gas < 500:  # Beispielschwelle
            print(f"Warnung: Niedrige Gasproduktion ({Q_gas:.1f} m³/d)")

monitor_digester_health(results)
```

### Optimale Betriebsbereiche

| Parameter | Optimal | Akzeptabel | Kritisch |
|-----------|---------|------------|----------|
| pH | 7.0-7.5 | 6.8-8.0 | <6.8 oder >8.0 |
| VFA [g/L] | 0.5-2.0 | 2.0-4.0 | >4.0 |
| VFA/TAC | 0.2-0.3 | 0.3-0.4 | >0.4 |
| TAC [g CaCO3/L] | 5.0-10.0 | 4.0-12.0 | <4.0 |
| Temp. mesophil [°C] | 35-38 | 32-40 | <30 oder >42 |
| Temp. thermophil [°C] | 52-55 | 48-58 | <45 oder >60 |

## Fehlerbehebung

### Niedriger pH-Wert

**Ursachen:**
- Zu hohe organische Raumbelastung (OLR)
- Unzureichende Pufferkapazität
- Plötzliche Substratänderung

**Lösungen:**
```python
# Reduziere organische Belastung
Q = [10, 8, 0, 0, 0, 0, 0, 0, 0, 0]  # Reduziert von [15, 10, ...]

# Oder füge Kalkpuffer hinzu
Q = [15, 10, 0, 0, 0, 0, 0, 1, 0, 0]  # 1 m³/d Kalk
```

### Schaumbildung

**Ursachen:**
- Zu hoher Proteingehalt im Substrat
- Plötzliche pH-Änderungen
- Hohe VFA-Konzentrationen

**Lösungen:**
- Reduziere proteinreiche Substrate
- Stabilisiere pH-Wert durch Pufferung
- Implementiere Anti-Schaum-Maßnahmen

### Geringe Gasproduktion

**Ursachen:**
- Niedrige organische Belastung
- Substrat niedriger Qualität
- Inhibition (NH3, H2S, Schwermetalle)
- Zu kurze Verweilzeit

**Diagnose:**
```python
def diagnose_low_gas_production(digester_outputs):
    """Diagnostiziere Ursachen für niedrige Gasproduktion"""

    Q_gas = digester_outputs['Q_gas']
    Q_in = sum(Q_substrates)  # Gesamter Input

    # Spezifische Gasproduktion
    specific_gas = Q_gas / Q_in if Q_in > 0 else 0

    if specific_gas < 0.5:  # m³ Biogas / m³ Input
        print("Niedrige spezifische Gasausbeute - mögliche Ursachen:")
        print("- Substrat niedriger Qualität")
        print("- Inhibition")
        print("- Prozessinstabilität")

    # Prüfe Methangehalt
    CH4_content = digester_outputs['Q_ch4'] / Q_gas if Q_gas > 0 else 0

    if CH4_content < 0.55:
        print(f"Niedriger Methangehalt ({CH4_content:.1%}) - mögliche Lufteinträge oder CO2-Stripping")

diagnose_low_gas_production(digester.outputs_data)
```

## Best Practices

1. **Starten Sie mit realistischen Betriebsparametern**
   - Nutze typische HRT-Werte (30-40 Tage)
   - Beginne mit moderater OLR (2-4 kg VS/m³/d)

2. **Überwache kritische Parameter**
   - pH-Wert sollte stabil sein (±0.2)
   - VFA/TAC-Verhältnis < 0.4
   - Methangehalt > 55%

3. **Implementiere Puffersysteme**
   - Füge Kalk oder andere Puffer bei niedrigem pH hinzu
   - Halte TAC > 4 g CaCO3/L

4. **Nutze zweistufige Systeme für schwierige Substrate**
   - Thermophile Hydrolyse für faserreiche Substrate
   - Mesophile Methanogenese für stabile Gasproduktion

5. **Kalibriere das Modell mit realen Daten**
   - Nutze Kalibrierungsparameter für genauere Vorhersagen
   - Validiere mit Betriebsdaten

## Nächste Schritte

- [Energiekomponenten](energy.md): BHKW und Wärmesysteme
- [Mechanische Komponenten](mechanical.md): Pumpen und Rührwerke
- [Fütterungskomponenten](feeding.md): Lagerung und Dosierung
- [API-Referenz](../../api_reference/components/biological.md): Detaillierte Klassendokumentation
