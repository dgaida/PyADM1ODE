# PyADM1ODE Dokumentation

Willkommen bei PyADM1ODE - Einem Python-Framework zur Modellierung, Simulation und Optimierung von landwirtschaftlichen Biogasanlagen basierend auf dem Anaerobic Digestion Model No. 1 (ADM1).

## 🎯 Quick Links
<div align="center">
  <a href="https://colab.research.google.com/github/dgaida/PyADM1ODE/blob/master/examples/colab_01_basic_digester.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Basic Digester Example"></a>
  &nbsp;
  <a href="https://colab.research.google.com/github/dgaida/PyADM1ODE/blob/master/examples/colab_02_complex_plant.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Complex Plant Example"></a>
</div>

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } __Schnellstart__  

    ---

    Starten Sie in wenigen Minuten mit Ihrer ersten Biogasanlagensimulation

    [:octicons-arrow-right-24: Schnellstart-Anleitung](user_guide/quickstart.md)

-   :material-download:{ .lg .middle } __Installation__  

    ---

    Installieren Sie PyADM1ODE auf Windows, Linux oder macOS

    [:octicons-arrow-right-24: Installations-Anleitung](user_guide/installation.md)

-   :material-book-open-variant:{ .lg .middle } __Handbuch__  

    ---

    Erfahren Sie mehr über das Framework, Komponenten und Substrate

    [:octicons-arrow-right-24: Handbuch](user_guide/adm1_implementation.md)

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

---

## Community und Support

- **GitHub Repository**: [dgaida/PyADM1ODE](https://github.com/dgaida/PyADM1ODE)  
- **Issue Tracker**: [Bugs melden oder Features anfragen](https://github.com/dgaida/PyADM1ODE/issues)  
- **Discussions**: [Fragen stellen und Ideen austauschen](https://github.com/dgaida/PyADM1ODE/discussions)  

## Lizenz

PyADM1ODE ist Open-Source-Software unter der MIT-Lizenz.
