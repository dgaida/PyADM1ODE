# PyADM1ODE

[![PyPI version](https://badge.fury.io/py/PyADM1ODE.svg)](https://badge.fury.io/py/PyADM1ODE)
[![Documentation Status](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://dgaida.github.io/PyADM1ODE/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/dgaida/PyADM1ODE/branch/master/graph/badge.svg)](https://codecov.io/gh/dgaida/PyADM1ODE)
[![Code Quality](https://github.com/dgaida/PyADM1ODE/actions/workflows/lint.yml/badge.svg)](https://github.com/dgaida/PyADM1ODE/actions/workflows/lint.yml)
[![Tests](https://github.com/dgaida/PyADM1ODE/actions/workflows/tests.yml/badge.svg)](https://github.com/dgaida/PyADM1ODE/actions/workflows/tests.yml)
[![CodeQL](https://github.com/dgaida/PyADM1ODE/actions/workflows/codeql.yml/badge.svg)](https://github.com/dgaida/PyADM1ODE/actions/workflows/codeql.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Version](https://img.shields.io/github/v/tag/dgaida/PyADM1ODE?label=version)](https://github.com/dgaida/PyADM1ODE/tags)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/dgaida/PyADM1ODE/graphs/commit-activity)
![Last commit](https://img.shields.io/github/last-commit/dgaida/PyADM1ODE)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://dgaida.github.io/PyADM1ODE/)

PyADM1ODE ist ein Python-Framework zur Modellierung, Simulation und Optimierung von landwirtschaftlichen Biogasanlagen basierend auf dem Anaerobic Digestion Model No. 1 (ADM1).

## Projektstruktur

```text
PyADM1ODE/
├── data/                    # Datenbanken (Substrate, Anfangszustände)
├── docs/                    # Dokumentationsquellen (DE/EN)
├── examples/                # Anwendungsbeispiele & Jupyter Notebooks
├── pyadm1/                  # Quellcode des Frameworks
│   ├── components/          # Anlagenkomponenten (Fermenter, BHKW, etc.)
│   ├── configurator/        # Anlagenbau und Konfiguration
│   ├── core/                # ADM1-Kern (Gleichungen, Solver, Parameter)
│   ├── dlls/                # Integrierte C#-DLLs für Physico-Chemicals
│   ├── simulation/          # Simulations-Engine und Parallelisierung
│   └── substrates/          # Substratmanagement und Feedstock-Berechnung
├── tests/                   # Unit- und Integrationstests
├── mkdocs.yml               # Konfiguration der Dokumentation
└── pyproject.toml           # Projekt-Metadaten und Abhängigkeiten
```

## Hauptmerkmale

- **Modulare Architektur**: Komponentenbasierter Aufbau von Biogasanlagen (Fermenter, BHKW, Pumpen, Rührwerke, etc.).  
- **Wissenschaftlich fundiert**: Implementierung des ADM1da-Modells für realistische Simulationen landwirtschaftlicher Anlagen.  
- **Einfache Konfiguration**: Programmatische Konfiguration oder über JSON-Dateien.  
- **Drei-Pass-Simulation**: Realistische Modellierung von Gasflüssen und Pufferspeichern.  
- **Bilingualität**: Vollständige Dokumentation in Deutsch und Englisch.  

## Dokumentation

Die vollständige Dokumentation finden Sie unter [dgaida.github.io/PyADM1ODE](https://dgaida.github.io/PyADM1ODE/).

- [Installation](https://dgaida.github.io/PyADM1ODE/latest/user_guide/installation/)  
- [Schnellstart](https://dgaida.github.io/PyADM1ODE/latest/user_guide/quickstart/)  
- [ADM1-Implementierung](https://dgaida.github.io/PyADM1ODE/latest/user_guide/adm1_implementation/)  
- [Komponenten](https://dgaida.github.io/PyADM1ODE/latest/user_guide/components/)  
- [Validierung](https://dgaida.github.io/PyADM1ODE/latest/user_guide/validation/)  

## Schnellstart

### Interaktive Beispiele (Google Colab)

Probieren Sie PyADM1ODE direkt im Browser aus:

- **Basis-Fermenter**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dgaida/PyADM1ODE/blob/master/examples/colab_01_basic_digester.ipynb)  
- **Zweistufige Anlage**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dgaida/PyADM1ODE/blob/master/examples/colab_02_complex_plant.ipynb)  

### Minimalbeispiel

```python
from pyadm1 import BiogasPlant
from pyadm1.configurator import PlantConfigurator
from pyadm1.substrates import Feedstock

# Setup
feedstock = Feedstock(feeding_freq=48)
plant = BiogasPlant("Meine Biogasanlage")
configurator = PlantConfigurator(plant, feedstock)

# Komponenten hinzufügen
configurator.add_digester("main_digester", V_liq=2000, V_gas=300, Q_substrates=[15, 10, 0, 0, 0, 0, 0, 0, 0, 0])
configurator.add_chp("chp1", P_el_nom=500)

# Verbinden und simulieren
configurator.auto_connect_digester_to_chp("main_digester", "chp1")
plant.initialize()
results = plant.simulate(duration=30, dt=1/24)
```

## Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert - siehe [LICENSE](LICENSE) für Details.

## Kontakt

**Daniel Gaida**  
- E-Mail: daniel.gaida@th-koeln.de  
- GitHub: [@dgaida](https://github.com/dgaida)  
- Institution: TH Köln - University of Applied Sciences  
