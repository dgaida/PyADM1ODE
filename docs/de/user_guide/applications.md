# Typische Anwendungen

PyADM1ODE kann für eine Vielzahl von Aufgaben eingesetzt werden, vom Anlagendesign bis hin zur Echtzeitoptimierung.

## 1. Anlagendesign und Optimierung

Testen Sie verschiedene Anlagenkonfigurationen, um das optimale Setup für Ihre Bedürfnisse zu finden.

```python
from pyadm1.configurator import BiogasPlant, PlantConfigurator
from pyadm1.substrates import Feedstock

# Test verschiedener Fermentergrößen
for V_liq in [1500, 2000, 2500]:
    plant = BiogasPlant(f"Anlage_{V_liq}")
    feedstock = Feedstock()
    configurator = PlantConfigurator(plant, feedstock)
    configurator.add_digester("dig1", V_liq=V_liq, Q_substrates=[15, 10, 0, 0, 0, 0, 0, 0, 0, 0])

    plant.initialize()
    results = plant.simulate(duration=30, dt=1/24)

    final = results[-1]["components"]["dig1"]
    print(f"V={V_liq} m³ → CH4={final['Q_ch4']:.1f} m³/d")
```

## 2. Substratoptimierung

Vergleichen Sie verschiedene Substratbelegungen, um die Methanproduktion zu maximieren oder die Kosten zu minimieren.

```python
# Vergleich verschiedener Substratbelegungen
mixes = {
    'high_energy': [20, 5, 0, 0, 0, 0, 0, 0, 0, 0],
    'balanced': [15, 10, 0, 0, 0, 0, 0, 0, 0, 0],
    'waste_based': [0, 15, 0, 0, 0, 0, 0, 0, 10, 5]
}

for name, Q in mixes.items():
    # ... konfigurieren und simulieren ...
    print(f"{name}: {final['Q_ch4']:.1f} m³/d Methan")
```

## 3. Energiebilanzanalyse

Analysieren Sie die Nettoenergieerzeugung und den Eigenverbrauch Ihrer Anlage.

```python
# Berechnung der Nettoenergieerzeugung
chp_power = results[-1]["components"]["chp_main"]["P_el"]
mixer_power = results[-1]["components"]["mixer_1"]["P_consumed"]
pump_power = results[-1]["components"]["pump_1"]["P_consumed"]

eigenverbrauch = mixer_power + pump_power
netto_leistung = chp_power - eigenverbrauch

print(f"Nettoleistung: {netto_leistung:.1f} kW")
print(f"Eigenverbrauchsquote: {eigenverbrauch/chp_power:.1%}")
```

## 4. Zweistufiges Prozessdesign

Modellieren Sie fortgeschrittene Anlagendesigns wie die temperaturgestufte anaerobe Vergärung (TPAD).

```python
# Temperaturgestufte anaerobe Vergärung (TPAD)
configurator.add_digester("hydrolyse", V_liq=500, T_ad=318.15)  # 45°C
configurator.add_digester("hauptfermenter", V_liq=2000, T_ad=308.15)       # 35°C
configurator.connect("hydrolyse", "hauptfermenter", "liquid")

# Verbesserte Hydrolyse in Stufe 1, stabile Methanogenese in Stufe 2
```

## Forschungsanwendungen

Dieses Framework unterstützt die Forschung in den folgenden Bereichen:

- **Prozessoptimierung**: Substratfütterungsstrategien, Verweilzeit.  
- **Regelungssysteme**: Modellprädiktive Regelung, Feedback-Regler.  
- **Anlagendesign**: Komponentendimensionierung, Layoutoptimierung.  
- **Energiemanagement**: BHKW-Einsatzplanung, Wärmeintegration.  
- **Substratbewertung**: Bewertung des Biogaspotenzials.  
