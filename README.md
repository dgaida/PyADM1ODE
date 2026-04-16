# PyADM1ODE - Advanced Biogas Plant Simulation Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/dgaida/PyADM1ODE/branch/master/graph/badge.svg)](https://codecov.io/gh/dgaida/PyADM1ODE)
[![Code Quality](https://github.com/dgaida/PyADM1ODE/actions/workflows/lint.yml/badge.svg)](https://github.com/dgaida/PyADM1ODE/actions/workflows/lint.yml)
[![Tests](https://github.com/dgaida/PyADM1ODE/actions/workflows/tests.yml/badge.svg)](https://github.com/dgaida/PyADM1ODE/actions/workflows/tests.yml)
[![CodeQL](https://github.com/dgaida/PyADM1ODE/actions/workflows/codeql.yml/badge.svg)](https://github.com/dgaida/PyADM1ODE/actions/workflows/codeql.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Version](https://img.shields.io/github/v/tag/dgaida/PyADM1ODE?label=version)](https://github.com/dgaida/PyADM1ODE/tags)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://dgaida.github.io/PyADM1ODE/)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://github.com/dgaida/PyADM1ODE/graphs/commit-activity)
![Last commit](https://img.shields.io/github/last-commit/dgaida/PyADM1ODE)

A comprehensive Python framework for modeling, simulating, and optimizing agricultural biogas plants based on the Anaerobic Digestion Model No. 1 (ADM1).

## Overview

PyADM1 provides a modular, extensible platform for:

- **Component-based plant modeling**: Build complex biogas plant configurations from modular components (digesters, CHP units, pumps, mixers, feeders, gas storage, etc.)  
- **High-fidelity ADM1 simulation**: Agricultural biogas-specific implementation of ADM1 as pure ODE system  
- **Automated model configuration**: Build plant models programmatically via MCP server for LLM integration (see [PyADM1ODE_mcp](https://github.com/dgaida/PyADM1ODE_mcp))  
- **Parallel scenario simulation**: Run multiple simulations with varying parameters simultaneously  
- **Online calibration**: Automatic parameter calibration and re-calibration during plant operation (see [PyADM1ODE_calibration](https://github.com/dgaida/PyADM1ODE_calibration))  
- **Validation framework**: Comprehensive testing against established models (SIMBA#, [ADM1F](https://github.com/lanl/ADM1F))  

## Key Features

### 🧩 Modular Component System  
- **Biological**: Digesters with integrated gas storage, hydrolysis tanks, separators  
- **Mechanical**: Pumps, mixers, valves, heat exchangers  
- **Energy**: CHP units, boilers, gas storage (low/high pressure), flares  
- **Feeding**: Substrate storage with quality degradation, dosing systems with accuracy modeling, mixer wagons  
- **Sensors**: Physical, chemical, and gas sensors with realistic measurement characteristics  

### 🔧 Plant Configurator  
- Template-based plant design (single-stage, two-stage, custom)  
- JSON-based configuration with validation  
- Component registry for dynamic loading  
- Connection management with type safety  
- Automatic gas storage creation for each digester  
- Integrated flare for CHP units  

### 🤖 MCP Server Integration
See [PyADM1ODE_mcp](https://github.com/dgaida/PyADM1ODE_mcp).

### ⚡ High-Performance Simulation  
- Pure ODE implementation (no DAEs) for numerical stability  
- Three-pass execution model for gas flow management:  
  1. Execute digesters → produce gas to storage  
  2. Execute CHPs → determine gas demand from storage  
  3. Execute storages → supply gas based on actual demand and availability  
- Parallel execution of multiple scenarios  
- Parameter sweeps for sensitivity analysis  
- Time-series data management (see [PyADM1ODE_calibration](https://github.com/dgaida/PyADM1ODE_calibration))  

### 🎯 Calibration Framework
See [PyADM1ODE_calibration](https://github.com/dgaida/PyADM1ODE_calibration).

### ✅ Validation & Testing  
- Comparison with SIMBA# and ADM1F models  
- Measurement data validation  
- Comprehensive test suite (unit, integration, validation)  

## Project Structure
```
PyADM1/
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
├── CONTRIBUTING.md
├── CHANGELOG.md
│
├── docs/                              # Documentation
│   ├── conf.py
│   ├── index.rst
│   ├── user_guide/
│   │   ├── installation.md
│   │   ├── quickstart.md
│   │   └── components
│   ├── api_reference/
│   │   ├── core.rst
│   │   ├── components.rst
│   │   ├── configurator.rst
│   │   ├── components
│   │   ├── configurator
│   │   └── core
│   ├── examples/
│   │   ├── basic_digester.md
│   │   ├── two_stage_plant.md
│   │   └── parallel_simulation.md
│   └── development/
│       ├── architecture.md
│       ├── adding_components.md
│       └── TESTING.md
│
├── pyadm1/                           # Main package
│   ├── __init__.py
│   ├── __version__.py
│   │
│   ├── core/                         # Core ADM1 implementation
│   │   ├── __init__.py
│   │   ├── adm1.py                  # Main ADM1 ODE system
│   │   ├── adm_params.py            # ADM1 parameters
│   │   ├── adm_equations.py         # Process rates, inhibitions
│   │   └── solver.py                # ODE solver wrapper
│   │
│   ├── components/                   # Modular components
│   │   ├── __init__.py
│   │   ├── base.py                  # Base classes for all components
│   │   ├── registry.py              # Component registry
│   │   │
│   │   ├── biological/              # Biological processes
│   │   │   ├── __init__.py
│   │   │   ├── digester.py         # Fermenter component with integrated gas storage
│   │   │   ├── hydrolysis.py       # Hydrolysis tank
│   │   │   └── separator.py        # Solid-liquid separation
│   │   │
│   │   ├── mechanical/              # Mechanical components
│   │   │   ├── __init__.py
│   │   │   ├── pump.py             # Pumps
│   │   │   ├── mixer.py            # Agitators/stirrers
│   │   │   ├── valve.py            # Valves
│   │   │   └── heat_exchanger.py   # Heat exchangers
│   │   │
│   │   ├── energy/                  # Energy components
│   │   │   ├── __init__.py
│   │   │   ├── chp.py              # Combined heat and power with automatic flare
│   │   │   ├── boiler.py           # Boilers
│   │   │   ├── gas_storage.py      # Gas storage (low/high pressure)
│   │   │   └── flare.py            # Flares
│   │   │
│   │   ├── feeding/                 # Substrate components
│   │   │   ├── __init__.py
│   │   │   ├── substrate_storage.py # Substrate storage with degradation
│   │   │   ├── feeder.py           # Dosing systems with accuracy modeling
│   │   │   └── mixer_wagon.py      # Mixer wagons
│   │   │
│   │   └── sensors/                 # Sensor components
│   │       ├── __init__.py
│   │       ├── physical.py         # pH, T, pressure, etc.
│   │       ├── chemical.py         # VFA, NH4, etc.
│   │       └── gas.py              # CH4, CO2, H2S
│   │
│   ├── substrates/                   # Substrate management
│   │   ├── __init__.py
│   │   ├── feedstock.py            # Feedstock class
│   │   ├── substrate_db.py         # Substrate database
│   │   ├── xml_loader.py           # XML parser for substrates
│   │   └── characterization.py     # Substrate characterization
│   │
│   ├── configurator/                 # Model configurator
│   │   ├── __init__.py
│   │   ├── plant_builder.py        # Plant builder with three-pass simulation
│   │   ├── plant_configurator.py   # High-level configuration helpers
│   │   ├── connection_manager.py   # Connection management
│   │   ├── validation.py           # Model validation
│   │   └── templates/              # Plant templates
│   │       ├── __init__.py
│   │       ├── single_stage.py
│   │       ├── two_stage.py
│   │       └── custom.py
│   │
│   ├── simulation/                   # Simulation engine
│   │   ├── __init__.py
│   │   ├── simulator.py            # Main simulator
│   │   ├── parallel.py             # Parallel simulation
│   │   ├── scenarios.py            # Scenario management
│   │   ├── time_series.py          # Time series handling
│   │   └── results.py              # Result management
│   │
│   ├── utils/                       # Utility functions
│   │   ├── __init__.py
│   │   ├── math_helpers.py         # Mathematical helpers
│   │   ├── unit_conversion.py      # Unit conversion
│   │   ├── logging.py              # Logging configuration
│   │   └── validators.py           # Validation functions
│   │
│   └── dlls/                        # C# DLLs
│       ├── plant.dll
│       ├── substrates.dll
│       ├── biogas.dll
│       └── physchem.dll
│
├── data/                            # Data directory
│   ├── substrates/
│   │   ├── substrate_gummersbach.xml
│   │   └── substrate_database.json
│   ├── initial_states/
│   │   └── digester_initial*.csv
│   ├── plant_templates/            # Plant templates
│   │   ├── standard_single_stage.json
│   │   └── standard_two_stage.json
│   └── validation_data/            # Validation data
│       ├── simba_comparison/
│       └── adm1f_comparison/
│
├── examples/                        # Examples
│   ├── __init__.py
│   ├── 01_basic_digester.py
│   ├── 02_two_stage_plant.py
│   ├── 04_substrate_optimization.py
│   ├── 05_parallel_simulation.py
│   └── notebooks/
│       ├── tutorial_basic.ipynb
│       └── tutorial_calibration.ipynb
│
├── tests/                           # Tests
│   ├── __init__.py
│   ├── conftest.py
│   │
│   ├── unit/                        # Unit tests
│   │   ├── test_core/
│   │   │   ├── test_adm1.py
│   │   │   └── test_adm_params.py
│   │   ├── test_components/
│   │   │   ├── test_digester.py
│   │   │   ├── test_chp.py
│   │   │   └── test_pump.py
│   │   └── test_configurator/
│   │      └── test_plant_builder.py
│   │
│   ├── integration/                 # Integration tests
│   │   ├── test_plant_simulation.py
│   │   └── test_parallel_sim.py
│   │
│   └── validation/                  # Validation tests
│       ├── test_simba_comparison.py
│       ├── test_adm1f_comparison.py
│       └── test_measurement_data.py
│
├── benchmarks/                      # Performance tests
│   ├── __init__.py
│   ├── benchmark_adm1.py
│   └── benchmark_parallel.py
│
├── scripts/                         # Helper scripts
│   ├── setup_dev_env.sh
│   └── generate_validation_data.py
│
└── .github/                         # GitHub CI/CD
    ├── workflows/
    │   ├── tests.yml
    │   ├── lint.yml
    │   ├── docs.yml
    │   └── release.yml
    └── ISSUE_TEMPLATE/
        ├── bug_report.md
        └── feature_request.md
```

## Installation

See [Installation](docs/user_guide/installation.md).

## Quick Start

See [Quickstart](docs/user_guide/quickstart.md).

### Basic Usage
```python
from pyadm1 import BiogasPlant
from pyadm1.components.biological import Digester
from pyadm1.components.energy import CHP
from pyadm1.substrates import Feedstock

# Create feedstock
feedstock = Feedstock(feeding_freq=48)

# Build plant using PlantConfigurator
from pyadm1.configurator.plant_configurator import PlantConfigurator

plant = BiogasPlant("My Biogas Plant")
configurator = PlantConfigurator(plant, feedstock)

# Add digester (automatically creates gas storage)
configurator.add_digester(
    "main_digester",
    V_liq=2000,
    V_gas=300,
    Q_substrates=[15, 10, 0, 0, 0, 0, 0, 0, 0, 0]
)

# Add CHP (automatically creates flare)
configurator.add_chp("chp1", P_el_nom=500)

# Connect components (gas storage automatically connected)
configurator.auto_connect_digester_to_chp("main_digester", "chp1")

# Initialize and simulate
plant.initialize()
results = plant.simulate(duration=30, dt=1/24)

# Save configuration
plant.to_json("my_plant.json")
```

### Using the MCP Server

See [PyADM1ODE_mcp](https://github.com/dgaida/PyADM1ODE_mcp).

### Parallel Simulation
```python
from pyadm1.simulation import ParallelSimulator

# Create scenarios with parameter variations
scenarios = [
    {"k_dis": 0.5, "Y_su": 0.1},
    {"k_dis": 0.6, "Y_su": 0.11},
    {"k_dis": 0.7, "Y_su": 0.12},
]

# Run parallel simulations
sim = ParallelSimulator(plant)
results = sim.run_scenarios(scenarios, duration=30)
```

### Model Calibration

See [PyADM1ODE_calibration](https://github.com/dgaida/PyADM1ODE_calibration).

## Core Concepts

### Components

All plant components inherit from `Component` base class and provide:  
- `step(t, dt, inputs)`: Perform one simulation time step  
- `initialize(state)`: Initialize component state  
- `to_dict()` / `from_dict()`: Serialization  

### Plant Configuration

Plants are configured through:  
- **Programmatic API**: Direct component instantiation using PlantConfigurator  
- **JSON files**: Load/save complete configurations  
- **Templates**: Pre-defined plant layouts  
- **MCP Server**: LLM-driven configuration from natural language  

### Gas Flow Management

The simulation uses a three-pass execution model:

1. **Pass 1**: Digesters produce gas → Gas storages  
2. **Pass 2**: Gas storages receive production (no demand yet)  
3. **Pass 3**: CHPs request gas → Storages supply → CHPs re-execute with actual supply  

This ensures:  
- Realistic gas buffering in storage tanks  
- Pressure management in gas systems  
- CHP operates with available gas, not idealized supply  
- Excess gas venting to flare when storage is full  

### Automatic Component Creation

PlantConfigurator automatically creates and connects:  
- **Gas Storage**: One per digester (membrane type, sized to gas volume)  
- **Flare**: One per CHP unit (safety combustion of excess gas)  

This reduces configuration complexity while ensuring realistic plant behavior.

### Substrate Management

Substrates are characterized by:  
- Weender analysis (fiber, protein, lipids)  
- Van Soest fractions (NDF, ADF, ADL)  
- Physical properties (pH, TS, VS, COD)  
- Kinetic parameters (disintegration, hydrolysis rates)  

### Simulation Engine

The simulation engine:  
- Uses BDF solver for stiff ODEs  
- Supports variable time steps  
- Manages component dependencies  
- Handles liquid, gas, heat, and power flows  

## Research Applications

This framework supports research in:

- **Process optimization**: Substrate feed strategies, retention time  
- **Control systems**: Model predictive control, feedback controllers  
- **Plant design**: Component sizing, layout optimization  
- **Energy management**: CHP scheduling, heat integration  
- **Substrate evaluation**: Biogas potential assessment  

## Validation

The framework has been validated against:  
- **SIMBA#**: Commercial biogas simulation software  
- **[ADM1F](https://github.com/lanl/ADM1F)**: LANL's Fortran ADM1 implementation  
- **Real plant data**: Multiple agricultural biogas plants  

## Development Status

- ✅ Core ADM1 implementation  
- ✅ Basic components (Digester with integrated storage, CHP with flare, Heating)  
- ✅ Plant configuration and JSON I/O  
- ✅ Three-pass gas flow simulation  
- ✅ Mechanical components (Pumps, Mixers)  
- ✅ Feeding components (Storage with degradation, Feeders with accuracy)  
- 🚧 Extended component library (in progress)  
- 🚧 Parallel simulation (in progress)  
- 📋 Validation framework (planned)  

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas where we need help:  
- Additional component implementations  
- Validation data from real plants  
- Performance optimization  
- Documentation and examples  
- Integration with other tools  

## Citation

If you use PyADM1ODE in your research, please cite:
```bibtex
@software{pyadm1,
  author = {Gaida, Daniel},
  title = {PyADM1: Advanced Biogas Plant Simulation Framework},
  year = {2026},
  url = {https://github.com/dgaida/PyADM1}
}

@phdthesis{gaida2014dynamic,
  title={Dynamic real-time substrate feed optimization of anaerobic co-digestion plants},
  author={Gaida, Daniel},
  year={2014},
  school={Universiteit Leiden}
}
```

## Related Publications

- **Gaida, D. (2024).** *Synergizing Language Models and Biogas Plant Control: A GPT-4 Approach.* 18th IWA World Conference on Anaerobic Digestion, Istanbul, Turkey.  

- **Batstone, D.J., et al. (2002).** *Anaerobic Digestion Model No. 1 (ADM1).* IWA Publishing, London.  

- **Sadrimajd, P., Mannion, P., Howley, E., & Lens, P.N.L. (2021).** *PyADM1: a Python implementation of Anaerobic Digestion Model No. 1.* bioRxiv. DOI: [10.1101/2021.03.03.433746](https://doi.org/10.1101/2021.03.03.433746)  

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- Original [PyADM1](https://github.com/CaptainFerMag/PyADM1) implementation by Peyman Sadrimajd et al. that motivated me to create this project  
- ADM1 development by IWA Task Group  
- SIMBA implementation by ifak e.V.  

## Contact

**Daniel Gaida**  
- Email: daniel.gaida@th-koeln.de  
- GitHub: [@dgaida](https://github.com/dgaida)  
- Institution: TH Köln - University of Applied Sciences  

---

**Note**: This is an active research project. APIs may change as development progresses. For production use, please use tagged releases.
