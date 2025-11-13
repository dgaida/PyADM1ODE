# PyADM1ODE

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Quality](https://github.com/dgaida/PyADM1ODE/actions/workflows/lint.yml/badge.svg)](https://github.com/dgaida/PyADM1ODE/actions/workflows/lint.yml)
[![Tests](https://github.com/dgaida/PyADM1ODE/actions/workflows/tests.yml/badge.svg)](https://github.com/dgaida/PyADM1ODE/actions/workflows/tests.yml)
[![CodeQL](https://github.com/dgaida/PyADM1ODE/actions/workflows/codeql.yml/badge.svg)](https://github.com/dgaida/PyADM1ODE/actions/workflows/codeql.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A Python implementation of the Anaerobic Digestion Model No. 1 (ADM1) as a system of Ordinary Differential Equations (ODEs) without differential algebraic equations (DAEs).

## Overview

PyADM1ODE is specifically designed for simulating agricultural co-digestion plants. The ADM1 input stream is calculated from agricultural substrates, making this implementation particularly useful for:

- Biogas plant process simulation
- Substrate feed optimization
- Process control development
- Research and education in anaerobic digestion

## Features

- **Pure ODE Implementation**: Simplified ADM1 without DAEs for better numerical stability
- **Agricultural Substrates**: Direct calculation of ADM1 inputs from substrate characteristics
- **Flexible Substrate Mixing**: Support for multiple substrate types with dynamic mixing ratios
- **Process Monitoring**: Built-in calculation of key process indicators (pH, VFA, TAC, biogas production)
- **Extensible Architecture**: Modular design for easy customization and extension

## Installation

### Prerequisites

- Python 3.8 or higher
- .NET Framework or Mono (for C# DLL interoperability via pythonnet)

### From Source

```bash
# Clone the repository
git clone https://github.com/dgaida/PyADM1ODE.git
cd PyADM1ODE

# Install in development mode
pip install -e .

# Or with development dependencies
pip install -e ".[dev]"
```

### Dependencies

Core dependencies:
- `numpy>=1.20.0` - Numerical computations
- `pandas>=1.3.0` - Data handling
- `scipy>=1.7.0` - ODE solver
- `pythonnet>=3.0.0` - C# interoperability

## Quick Start

```python
from pyadm1 import PyADM1, Feedstock, Simulator, get_state_zero_from_initial_state

# Configuration
feeding_freq = 48  # hours between feeding events

# Initialize components
feedstock = Feedstock(feeding_freq)
adm1 = PyADM1(feedstock)
simulator = Simulator(adm1)

# Define substrate feed [m³/d]
# [corn silage, liquid manure, substrate3, ..., substrate10]
Q = [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]

# Load initial digester state
state_zero = get_state_zero_from_initial_state("data/initial_states/digester_initial8.csv")

# Run simulation
t = feedstock.simtime()
simulate_results = [state_zero]

for n, t_end in enumerate(t[1:], 1):
    adm1.createInfluent(Q, n)
    state_zero = simulator.simulateADplant([t[n-1], t_end], state_zero)
    simulate_results.append(state_zero)

    if n % 100 == 0:
        print(f"Simulated {n} of {len(t)} steps.")

# Save final state
adm1.save_final_state_in_csv(simulate_results, "digester_final.csv")
```

## Project Structure

```
PyADM1ODE/
├── pyadm1/                      # Main package
│   ├── __init__.py
│   ├── core/                    # Core ADM1 implementation
│   │   ├── __init__.py
│   │   ├── adm_params.py        # ADM1 parameters
│   │   ├── pyadm1.py            # Main ADM1 ODE model
│   │   └── simulator.py         # Simulation orchestration
│   ├── substrates/              # Substrate management
│   │   ├── __init__.py
│   │   └── feedstock.py         # Substrate characterization
│   ├── utils/                   # Utility functions
│   │   └── __init__.py
│   └── dlls/                    # C# DLL dependencies
│       ├── plant.dll
│       ├── substrates.dll
│       ├── biogas.dll
│       └── physchem.dll
├── data/                        # Data files
│   ├── initial_states/          # Initial digester states
│   └── substrates/              # Substrate definitions (XML)
├── output/                      # data created during simulation
├── tests/                       # Unit tests
├── LICENSE
├── main.py
├── README.md
├── pyproject.toml              # Project configuration
└── requirements.txt            # Dependencies
```

## Model Description

### ADM1 State Variables

The model tracks 37 state variables:

**Soluble components** (12):
- Monosaccharides (S_su), amino acids (S_aa), long chain fatty acids (S_fa)
- Valerate (S_va), butyrate (S_bu), propionate (S_pro), acetate (S_ac)
- Hydrogen (S_h2), methane (S_ch4), inorganic carbon (S_co2)
- Inorganic nitrogen (S_nh4), soluble inerts (S_I)

**Particulate components** (13):
- Composites (X_xc), carbohydrates (X_ch), proteins (X_pr), lipids (X_li)
- Seven bacterial populations (X_su, X_aa, X_fa, X_c4, X_pro, X_ac, X_h2)
- Particulate inerts (X_I), particulate products (X_p)

**Acid-base components** (8):
- Cations and anions (S_cation, S_anion)
- Ionized forms of VFAs and inorganic species

**Gas phase** (4):
- Partial pressures of H₂, CH₄, CO₂, and total pressure

### Key Processes

1. **Disintegration**: Complex particulate matter breakdown
2. **Hydrolysis**: Carbohydrates, proteins, and lipids to monomers
3. **Acidogenesis**: Sugars and amino acids to VFAs
4. **Acetogenesis**: VFAs to acetate and hydrogen
5. **Methanogenesis**: Acetate and hydrogen to methane
6. **Gas transfer**: Liquid-gas phase equilibrium

## Substrate Configuration

Substrates are defined in XML files (e.g., `substrate_gummersbach.xml`) with parameters including:

- Weender analysis (crude fiber, protein, lipids)
- Van Soest fractions (NDF, ADF, ADL)
- Physical properties (pH, TS, VS, COD)
- Kinetic parameters (disintegration, hydrolysis, uptake rates)

Example substrate definition:
```xml
<substrate id="maize">
    <name>Silomais</name>
    <Weender>
        <physValue symbol="RF"><value>21.07</value><unit>% TS</unit></physValue>
        <physValue symbol="RP"><value>8.69</value><unit>% TS</unit></physValue>
        <!-- ... -->
    </Weender>
    <!-- ... -->
</substrate>
```

## Advanced Usage

### Custom Substrate Feeds

```python
# Optimize substrate feed for target methane production
Qch4_setpoint = 1500  # m³/d
best_feed = simulator.determineBestFeedbyNSims(
    state_zero=current_state,
    Q=[15, 10, 0, 0, 0, 0, 0, 0, 0, 0],
    Qch4sp=Qch4_setpoint,
    feeding_freq=48,
    n=13  # number of scenarios to test
)
```

### Process Monitoring

```python
# Access process indicators
pH = adm1.pH_l()              # pH history
vfa = adm1.VFA()              # VFA concentration [g/L]
tac = adm1.TAC()              # Total alkalinity [g CaCO3/L]
fos_tac = adm1.VFA_TA()       # FOS/TAC ratio
biogas = adm1.Q_GAS()         # Biogas production [m³/d]
methane = adm1.Q_CH4()        # Methane production [m³/d]
```

### Custom Initial States

```python
import pandas as pd

# Create custom initial state
initial_state = pd.DataFrame({
    'S_su': [0.01], 'S_aa': [0.001], 'S_fa': [0.04],
    # ... (37 columns total)
})
initial_state.to_csv('custom_initial.csv', index=False)

# Load for simulation
state_zero = get_state_zero_from_initial_state('custom_initial.csv')
```

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pyadm1

# Run specific test file
pytest tests/test_pyadm1.py -v
```

## Publications

This implementation is based on research documented in:

- **Gaida, D. (2014).** *Dynamic real-time substrate feed optimization of anaerobic co-digestion plants.* PhD thesis, Universiteit Leiden. [Link](https://scholarlypublications.universiteitleiden.nl/handle/1887/29085)

- **Gaida, D. (2024).** *Synergizing Language Models and Biogas Plant Control: A GPT-4 Approach.* 18th IWA World Conference on Anaerobic Digestion, Istanbul, Turkey.

The original ADM1 model:

- **Batstone, D.J., et al. (2002).** *Anaerobic Digestion Model No. 1 (ADM1).* IWA Task Group for Mathematical Modelling of Anaerobic Digestion Processes. IWA Publishing, London.

This implementation also draws from:

- **Sadrimajd, P., Mannion, P., Howley, E., & Lens, P.N.L. (2021).** *PyADM1: a Python implementation of Anaerobic Digestion Model No. 1.* bioRxiv. DOI: [10.1101/2021.03.03.433746](https://doi.org/10.1101/2021.03.03.433746)

- **Rosen, C., et al. (2006).** *Benchmark Simulation Model No. 2 (BSM2).* IWA Task Group on Benchmarking of Control Strategies for WWTPs.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original [PyADM1](https://github.com/CaptainFerMag/PyADM1) implementation by Peyman Sadrimajd et al.
- ADM1 development by the IWA Task Group
- Simba implementation by ifak e.V. (2010)

## Contact

**Daniel Gaida**
- GitHub: [@dgaida](https://github.com/dgaida)

## Citation

If you use this software in your research, please cite:

```bibtex
@phdthesis{gaida2014dynamic,
  title={Dynamic real-time substrate feed optimization of anaerobic co-digestion plants},
  author={Gaida, Daniel},
  year={2014},
  school={Universiteit Leiden}
}

@software{pyadm1ode,
  author={Gaida, Daniel},
  title={PyADM1ODE: Python implementation of Anaerobic Digestion Model No. 1},
  year={2024},
  url={https://github.com/dgaida/PyADM1ODE}
}
```

---

**Note**: This implementation requires C# DLLs for substrate characterization. Ensure the `dlls/` directory exists and contains the required DLLs.

Source code to create the DLL files can be found at [matlab_toolboxes](https://github.com/dgaida/matlab_toolboxes).
