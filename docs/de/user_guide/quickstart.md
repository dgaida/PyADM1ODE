# Schnellstart-Anleitung

Diese Anleitung hilft Ihnen, PyADM1ODE in wenigen Minuten in Betrieb zu nehmen.

## Ihre erste Simulation

Lassen Sie uns einen einfachen einstufigen Biogasfermenter mit Maissilage und Gülle simulieren.

```python
from pathlib import Path
from pyadm1.configurator.plant_builder import BiogasPlant
from pyadm1.substrates.feedstock import Feedstock
from pyadm1.core.adm1 import get_state_zero_from_initial_state
from pyadm1.configurator.plant_configurator import PlantConfigurator

# 1. Feedstock-Manager erstellen
feedstock = Feedstock(feeding_freq=48)

# 2. Anfangszustand laden (Steady-State-Werte)
data_path = Path("data/initial_states")
initial_state_file = data_path / "digester_initial8.csv"
adm1_state = get_state_zero_from_initial_state(str(initial_state_file))

# 3. Substrat-Fütterungsraten definieren [m³/Tag]
Q_substrates = [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]

# 4. Anlage erstellen und konfigurieren
plant = BiogasPlant("Meine erste Biogasanlage")
configurator = PlantConfigurator(plant, feedstock)

configurator.add_digester(
    digester_id="main_digester",
    V_liq=2000.0,        # Flüssigvolumen [m³]
    V_gas=300.0,         # Gasvolumen [m³]
    T_ad=308.15,         # Temperatur [K] = 35°C
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
    print(f"Tag {time:.0f}: Biogas: {digester['Q_gas']:.1f} m³/d, pH: {digester['pH']:.2f}")
```

## Nächste Schritte

- Erfahren Sie mehr über die [Komponenten](components/index.md).
- Sehen Sie sich die [Beispiele](../examples/basic_digester.md) an.
