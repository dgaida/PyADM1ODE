# PyADM1ODE Dokumentation

Willkommen bei PyADM1ODE - Einem Python-Framework zur Modellierung, Simulation und Optimierung von landwirtschaftlichen Biogasanlagen basierend auf dem Anaerobic Digestion Model No. 1 (ADM1).

## 🎯 Quick Links

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } __Schnellstart__

    ---

    Starten Sie in wenigen Minuten mit Ihrer ersten Biogasanlagensimulation

    [:octicons-arrow-right-24: Schnellstart-Anleitung](user_guide/quickstart.md)

-   :material-download:{ .lg .middle } __Installation__

    ---

    Installieren Sie PyADM1ODE auf Windows, Linux oder macOS

    [:octicons-arrow-right-24: Installations-Anleitung](user_guide/installation.md)

-   :material-book-open-variant:{ .lg .middle } __Komponenten-Leitfaden__

    ---

    Erfahren Sie mehr über Fermenter, BHKWs, Pumpen und mehr

    [:octicons-arrow-right-24: Komponentendokumentation](user_guide/components/index.md)

-   :material-code-braces:{ .lg .middle } __Beispiele__

    ---

    Praxisbeispiele von einfachen bis zu fortgeschrittenen Anlagen

    [:octicons-arrow-right-24: Beispiele](examples/basic_digester.md)

</div>

## Was ist PyADM1ODE?

PyADM1ODE ist ein umfassendes Python-Framework für die Modellierung landwirtschaftlicher Biogasanlagen, das Folgendes kombiniert:

- **Wissenschaftliche Genauigkeit**: Basierend auf dem ADM1-Modell der IWA, dem internationalen Standard für die anaerobe Vergärung.
- **Modulare Architektur**: Kombinieren Sie Komponenten (Fermenter, BHKW, Pumpen, Rührwerke), um jede beliebige Anlagenkonfiguration zu erstellen.
- **Praxisnähe**: Validiert mit Daten von in Betrieb befindlichen Biogasanlagen.
- **Python-Ökosystem**: Integriert mit NumPy, SciPy, Pandas und Visualisierungsbibliotheken.

### Hauptmerkmale

✨ **Umfassende Komponentenbibliothek**

- Biologisch: Ein-/mehrstufige Fermenter, Hydrolysetanks, Separatoren
- Energie: BHKW-Einheiten, Heizsysteme, Gasspeicher, Fackeln
- Mechanisch: Pumpen, Rührwerke mit realistischem Stromverbrauch
- Fütterung: Substratlagerung, automatisierte Dosiersysteme

🔧 **Flexible Anlagenkonfiguration**

- Erstellen Sie komplexe Anlagen programmatisch oder über Vorlagen
- Automatische Komponentenverbindung und Validierung
- Speichern/Laden von Konfigurationen als JSON

📊 **Fortgeschrittene Simulation**

- Parallele Ausführung für Parameterstudien und Monte-Carlo-Analysen
- Adaptive ODE-Solver, optimiert für steife Biogassysteme
- Zeitreihen-Datenverarbeitung und Ergebnisanalyse

🎓 **Bildung & Professionell**

- Geeignet für die Lehre im Bereich Biogasanlagendesign
- Forschungswerkzeug zur Prozessoptimierung
- Engineering-Anwendungen für die Anlagenplanung

## Systemarchitektur

```
┌─────────────────────────────────────────────────────────────────┐
│                     PyADM1ODE Framework                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Biologische  │  │   Energie-   │  │ Mechanische  │         │
│  │ Komponenten  │  │ Komponenten  │  │ Komponenten  │         │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤         │
│  │ • Fermenter  │  │ • BHKW       │  │ • Pumpen     │         │
│  │ • Hydrolyse  │  │ • Heizung    │  │ • Rührwerke  │         │
│  │ • Separatoren│  │ • Speicher   │  │              │         │
│  │              │  │ • Fackeln    │  │              │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Fütterung  │  │   Sensoren   │  │ Konfigurator │         │
│  │ Komponenten  │  │  (geplant)   │  │              │         │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤         │
│  │ • Lagerung   │  │ • pH         │  │ • Builder    │         │
│  │ • Dosierer   │  │ • VFA        │  │ • Vorlagen   │         │
│  │              │  │ • Gas        │  │ • Validator  │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                       Kern-ADM1-Engine                           │
│  • 37 Zustandsvariablen • pH-Dynamik • Gas-Flüssig-Transfer     │
│  • Temperaturabhängige Kinetik • Inhibitionsmodellierung        │
├─────────────────────────────────────────────────────────────────┤
│                    Substratmanagement                            │
│  • 10 vorkonfigurierte landwirtschaftliche Substrate            │
│  • Automatische ADM1-Input-Strom-Generierung                    │
│  • Zeitlich variierende Fütterungspläne                         │
└─────────────────────────────────────────────────────────────────┘
```

## Kurzes Beispiel

Erstellen und simulieren Sie eine komplette Biogasanlage in nur wenigen Zeilen:

```python
from pyadm1.configurator import BiogasPlant, PlantConfigurator
from pyadm1.substrates import Feedstock

# Anlage erstellen
feedstock = Feedstock(feeding_freq=48)
plant = BiogasPlant("Meine Biogasanlage")
configurator = PlantConfigurator(plant, feedstock)

# Fermenter hinzufügen (erstellt automatisch Gasspeicher)
configurator.add_digester(
    digester_id="main_digester",
    V_liq=2000.0,              # 2000 m³ Flüssigvolumen
    V_gas=300.0,               # 300 m³ Gasraum
    T_ad=308.15,               # 35°C mesophil
    Q_substrates=[15, 10, 0, 0, 0, 0, 0, 0, 0, 0]  # Maissilage + Gülle
)

# BHKW und Heizung hinzufügen (erstellt automatisch Fackel)
configurator.add_chp("chp_main", P_el_nom=500.0)
configurator.add_heating("heating_main", target_temperature=308.15)

# Komponenten verbinden
configurator.auto_connect_digester_to_chp("main_digester", "chp_main")
configurator.auto_connect_chp_to_heating("chp_main", "heating_main")

# Simulieren
plant.initialize()
results = plant.simulate(duration=30, dt=1/24, save_interval=1.0)

# Analysieren
final = results[-1]["components"]["main_digester"]
print(f"Biogas: {final['Q_gas']:.1f} m³/d")
print(f"Methan: {final['Q_ch4']:.1f} m³/d")
print(f"pH: {final['pH']:.2f}")
```

**Ausgabe:**
```
Biogas: 1245.3 m³/d
Methan: 748.2 m³/d
pH: 7.28
```

## Typische Anwendungen

### 1. Anlagendesign und Optimierung

### 2. Substratoptimierung

### 3. Energiebilanzanalyse

### 4. Zweistufiges Prozessdesign

(Details siehe englische Version oder Unterseiten)
