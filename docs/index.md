# PyADM1ODE Documentation

Welcome to PyADM1ODE - A Python framework for modeling, simulating, and optimizing agricultural biogas plants based on the Anaerobic Digestion Model No. 1 (ADM1).

## 🎯 Quick Links

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } __Quick Start__

    ---

    Get started in minutes with your first biogas plant simulation

    [:octicons-arrow-right-24: Quickstart Guide](user_guide/quickstart.md)

-   :material-download:{ .lg .middle } __Installation__

    ---

    Install PyADM1ODE on Windows, Linux, or macOS

    [:octicons-arrow-right-24: Installation Guide](user_guide/installation.md)

-   :material-book-open-variant:{ .lg .middle } __Components Guide__

    ---

    Learn about digesters, CHP units, pumps, and more

    [:octicons-arrow-right-24: Component Documentation](user_guide/components/index.md)

-   :material-code-braces:{ .lg .middle } __Examples__

    ---

    Real-world examples from basic to advanced plants

    [:octicons-arrow-right-24: Examples](examples/basic_digester.md)

</div>

## What is PyADM1ODE?

PyADM1ODE is a comprehensive Python framework for agricultural biogas plant modeling that combines:

- **Scientific accuracy**: Based on IWA's ADM1 model, the international standard for anaerobic digestion
- **Modular architecture**: Mix and match components (digesters, CHP units, pumps, mixers) to build any plant configuration
- **Real-world applicability**: Validated with data from operating biogas plants
- **Python ecosystem**: Integrates with NumPy, SciPy, Pandas, and visualization libraries

### Key Features

✨ **Comprehensive Component Library**  
- Biological: Single/multi-stage digesters, hydrolysis tanks, separators  
- Energy: CHP units, heating systems, gas storage, flares  
- Mechanical: Pumps, mixers with realistic power consumption  
- Feeding: Substrate storage, automated dosing systems

🔧 **Flexible Plant Configuration**  
- Build complex plants programmatically or via templates
- Automatic component connection and validation
- Save/load configurations as JSON

📊 **Advanced Simulation**
- Parallel execution for parameter sweeps and Monte Carlo analysis
- Adaptive ODE solvers optimized for stiff biogas systems
- Time-series data handling and result analysis

🎓 **Educational & Professional**
- Suitable for teaching biogas plant design
- Research tool for process optimization
- Engineering applications for plant planning

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     PyADM1ODE Framework                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Biological │  │    Energy    │  │  Mechanical  │         │
│  │  Components  │  │  Components  │  │  Components  │         │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤         │
│  │ • Digesters  │  │ • CHP Units  │  │ • Pumps      │         │
│  │ • Hydrolysis │  │ • Heating    │  │ • Mixers     │         │
│  │ • Separators │  │ • Storage    │  │              │         │
│  │              │  │ • Flares     │  │              │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │   Feeding    │  │   Sensors    │  │ Configurator │         │
│  │  Components  │  │  (planned)   │  │              │         │
│  ├──────────────┤  ├──────────────┤  ├──────────────┤         │
│  │ • Storage    │  │ • pH         │  │ • Builder    │         │
│  │ • Feeders    │  │ • VFA        │  │ • Templates  │         │
│  │              │  │ • Gas        │  │ • Validator  │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                       Core ADM1 Engine                           │
│  • 37 state variables • pH dynamics • Gas-liquid transfer       │
│  • Temperature-dependent kinetics • Inhibition modeling         │
├─────────────────────────────────────────────────────────────────┤
│                    Substrate Management                          │
│  • 10 pre-configured agricultural substrates                    │
│  • Automatic ADM1 input stream generation                       │
│  • Time-varying feed schedules                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Example

Build and simulate a complete biogas plant in just a few lines:

```python
from pyadm1.configurator import BiogasPlant, PlantConfigurator
from pyadm1.substrates import Feedstock

# Create plant
feedstock = Feedstock(feeding_freq=48)
plant = BiogasPlant("My Biogas Plant")
configurator = PlantConfigurator(plant, feedstock)

# Add digester (automatically creates gas storage)
configurator.add_digester(
    digester_id="main_digester",
    V_liq=2000.0,              # 2000 m³ liquid volume
    V_gas=300.0,               # 300 m³ gas headspace
    T_ad=308.15,               # 35°C mesophilic
    Q_substrates=[15, 10, 0, 0, 0, 0, 0, 0, 0, 0]  # Corn silage + manure
)

# Add CHP and heating (automatically creates flare)
configurator.add_chp("chp_main", P_el_nom=500.0)
configurator.add_heating("heating_main", target_temperature=308.15)

# Connect components
configurator.auto_connect_digester_to_chp("main_digester", "chp_main")
configurator.auto_connect_chp_to_heating("chp_main", "heating_main")

# Simulate
plant.initialize()
results = plant.simulate(duration=30, dt=1/24, save_interval=1.0)

# Analyze
final = results[-1]["components"]["main_digester"]
print(f"Biogas: {final['Q_gas']:.1f} m³/d")
print(f"Methane: {final['Q_ch4']:.1f} m³/d")
print(f"pH: {final['pH']:.2f}")
```

**Output:**
```
Biogas: 1245.3 m³/d
Methane: 748.2 m³/d
pH: 7.28
```

## Typical Applications

### 1. Plant Design and Optimization

```python
# Test different digester sizes
for V_liq in [1500, 2000, 2500]:
    plant = BiogasPlant(f"Plant_{V_liq}")
    configurator = PlantConfigurator(plant, feedstock)
    configurator.add_digester("dig1", V_liq=V_liq, Q_substrates=[15, 10, 0, 0, 0, 0, 0, 0, 0, 0])

    plant.initialize()
    results = plant.simulate(duration=30, dt=1/24)

    final = results[-1]["components"]["dig1"]
    print(f"V={V_liq} m³ → CH4={final['Q_ch4']:.1f} m³/d")
```

### 2. Substrate Optimization

```python
# Compare different substrate mixes
mixes = {
    'high_energy': [20, 5, 0, 0, 0, 0, 0, 0, 0, 0],
    'balanced': [15, 10, 0, 0, 0, 0, 0, 0, 0, 0],
    'waste_based': [0, 15, 0, 0, 0, 0, 0, 0, 10, 5]
}

for name, Q in mixes.items():
    # ... configure and simulate ...
    print(f"{name}: {final['Q_ch4']:.1f} m³/d methane")
```

### 3. Energy Balance Analysis

```python
# Calculate net energy production
chp_power = results[-1]["components"]["chp_main"]["P_el"]
mixer_power = results[-1]["components"]["mixer_1"]["P_consumed"]
pump_power = results[-1]["components"]["pump_1"]["P_consumed"]

parasitic_load = mixer_power + pump_power
net_power = chp_power - parasitic_load

print(f"Net power: {net_power:.1f} kW")
print(f"Parasitic ratio: {parasitic_load/chp_power:.1%}")
```

### 4. Two-Stage Process Design

```python
# Temperature-phased anaerobic digestion (TPAD)
configurator.add_digester("hydrolysis", V_liq=500, T_ad=318.15)  # 45°C
configurator.add_digester("main", V_liq=2000, T_ad=308.15)       # 35°C
configurator.connect("hydrolysis", "main", "liquid")

# Enhanced hydrolysis in stage 1, stable methanogenesis in stage 2
```

## Component Categories

### Biological Components

Model the core biological processes:

- **[Digester](user_guide/components/biological.md#digester)**: Main fermenter with ADM1 model
  - Single or multi-stage configurations
  - Temperature control (psychrophilic, mesophilic, thermophilic)
  - Automatic gas storage creation
  - Calibration parameter support

- **[Hydrolysis](user_guide/components/biological.md#hydrolysis)**: Pre-treatment tank (planned)
- **[Separator](user_guide/components/biological.md#separator)**: Digestate processing (planned)

### Energy Components

Complete energy integration:

- **[CHP](user_guide/components/energy.md#chp)**: Combined heat and power generation
  - Variable efficiency curves
  - Load-following operation
  - Automatic flare creation

- **[Heating](user_guide/components/energy.md#heating)**: Temperature control systems
  - CHP waste heat utilization
  - Auxiliary heating calculation

- **[Gas Storage](user_guide/components/energy.md#gasstorage)**: Biogas buffering
  - Low-pressure (membrane, dome) and high-pressure options
  - Automatic pressure management
  - Safety venting

- **[Flare](user_guide/components/energy.md#flare)**: Safety gas combustion
  - 98% methane destruction efficiency
  - Automatic activation on overpressure

### Mechanical Components

Material handling and process control:

- **[Pump](user_guide/components/mechanical.md#pump)**: Substrate and digestate transfer
  - Progressive cavity, centrifugal, piston types
  - Power consumption modeling
  - Variable frequency drive support

- **[Mixer](user_guide/components/mechanical.md#mixer)**: Digester agitation
  - Propeller, paddle, jet mixer types
  - Intermittent operation for energy savings
  - Reynolds number and mixing time calculation

### Feeding Components

Substrate management:

- **[Substrate Storage](user_guide/components/feeding.md#substratestorage)**: Material inventory
  - Multiple storage types (silos, tanks, bunkers)
  - Quality degradation modeling
  - Capacity and utilization tracking

- **[Feeder](user_guide/components/feeding.md#feeder)**: Automated dosing
  - Screw, piston, progressive cavity feeders
  - Realistic dosing accuracy and noise
  - Blockage detection

## Pre-configured Substrates

PyADM1ODE includes 10 agricultural substrates with literature-validated parameters:

| Substrate | Type | Typical Use | Biogas Potential |
|-----------|------|-------------|------------------|
| **Corn silage** | Energy crop | Main feedstock | High (600-700 L/kg VS) |
| **Liquid manure** | Animal waste | Co-substrate | Medium (200-400 L/kg VS) |
| **Green rye** | Energy crop | Early harvest | Medium-High |
| **Grass silage** | Grassland | Renewable | Medium (400-550 L/kg VS) |
| **Wheat** | Cereal | Energy crop | High |
| **GPS** | Grain silage | Whole-crop | High |
| **CCM** | Corn-cob-mix | Energy crop | High |
| **Feed lime** | Additive | pH buffer | N/A |
| **Cow manure** | Animal waste | Co-substrate | Medium (200-350 L/kg VS) |
| **Onions** | Waste | Vegetable waste | Medium-High |

All substrates are characterized with:
- Dry matter (DM) and volatile solids (VS) content
- ADM1 fractionation (carbohydrates, proteins, lipids)
- Biochemical methane potential (BMP)
- pH and alkalinity

## Advanced Features

### Parallel Simulation

Run multiple scenarios concurrently:

```python
from pyadm1.simulation import ParallelSimulator

# Parameter sweep
parallel = ParallelSimulator(adm1, n_workers=4)
scenarios = [
    {"k_dis": 0.5, "Q": [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]},
    {"k_dis": 0.6, "Q": [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]},
    {"k_dis": 0.7, "Q": [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]}
]

results = parallel.run_scenarios(scenarios, duration=30, initial_state=state)
```

### Model Calibration

Fit model parameters to measurement data:

```python
from pyadm1.components.biological import Digester

digester = Digester("dig1", feedstock, V_liq=2000)

# Apply calibrated parameters
digester.apply_calibration_parameters({
    'k_dis': 0.55,
    'Y_su': 0.105,
    'k_hyd_ch': 11.0
})

# Get current parameters
params = digester.get_calibration_parameters()
```

### Configuration Management

Save and reuse plant designs:

```python
# Save configuration
plant.to_json("two_stage_plant.json")

# Load later
plant = BiogasPlant.from_json("two_stage_plant.json", feedstock)
plant.initialize()
results = plant.simulate(duration=30, dt=1/24)
```

## Scientific Foundation

PyADM1ODE is based on the **Anaerobic Digestion Model No. 1 (ADM1)**, developed by the International Water Association (IWA) Task Group:

- **37 state variables**: Complete representation of liquid and gas phases
- **19 biochemical processes**: Disintegration, hydrolysis, acidogenesis, acetogenesis, methanogenesis
- **Temperature-dependent kinetics**: Arrhenius relationships for all rate constants
- **pH dynamics**: Full acid-base equilibrium with 6 ionic species
- **Gas-liquid transfer**: Henry's law implementation for H₂, CH₄, CO₂
- **Inhibition modeling**: pH, ammonia, and hydrogen inhibition

**Key References:**

- Batstone, D.J., et al. (2002). *Anaerobic Digestion Model No. 1 (ADM1)*. IWA Publishing.
- Sadrimajd, P., et al. (2021). *PyADM1: a Python implementation of Anaerobic Digestion Model No. 1*. bioRxiv.
- Gaida, D. (2014). *Dynamic real-time substrate feed optimization of anaerobic co-digestion plants*. PhD thesis, Leiden University.

## Installation

Install PyADM1ODE via pip (not yet existing):

```bash
pip install pyadm1ode
```

For development or the latest features:

```bash
git clone https://github.com/dgaida/PyADM1ODE.git
cd PyADM1ODE
pip install -e .
```

**Platform-specific requirements:**
- **Linux/macOS**: Mono runtime (for C# DLLs)
- **Windows**: .NET Framework (usually pre-installed)

See the [Installation Guide](user_guide/installation.md) for detailed instructions.

## Getting Started

1. **[Install PyADM1ODE](user_guide/installation.md)** on your system
2. **[Follow the Quickstart](user_guide/quickstart.md)** to run your first simulation
3. **[Explore Components](user_guide/components/index.md)** to understand available building blocks
4. **[Study Examples](examples/basic_digester.md)** for real-world applications

## Extension Packages

### PyADM1ODE_mcp - LLM-Driven Plant Design

Natural language interface for biogas plant modeling:

```bash
git clone https://github.com/dgaida/PyADM1ODE_mcp.git
cd PyADM1ODE_mcp
pip install -e .
```

**Features:**  
- Interact with Claude or other LLMs to design plants via natural language  
- MCP server for seamless LLM integration  
- Automated configuration parsing and validation

**Use case:** *"Design a two-stage biogas plant with 2000 m³ main digester, 500 m³ hydrolysis tank at 45°C, and a 500 kW CHP unit. Use corn silage and cattle manure as substrates."*

### PyADM1ODE_calibration - Parameter Estimation

Automated model calibration from measurement data:

```bash
git clone https://github.com/dgaida/PyADM1ODE_calibration.git
cd PyADM1ODE_calibration
pip install -e .
```

**Features:**  
- Initial calibration from historical data
- Online re-calibration during operation
- Multiple optimization algorithms (Differential Evolution, PSO, Nelder-Mead)
- Comprehensive validation metrics

**Use case:** Fit ADM1 parameters to real plant measurements for accurate predictions.

## Community and Support

- **GitHub Repository**: [dgaida/PyADM1ODE](https://github.com/dgaida/PyADM1ODE)
- **Issue Tracker**: [Report bugs or request features](https://github.com/dgaida/PyADM1ODE/issues)
- **Discussions**: [Ask questions and share ideas](https://github.com/dgaida/PyADM1ODE/discussions)
- **Email**: daniel.gaida@th-koeln.de

## License

PyADM1ODE is open-source software licensed under the MIT License.

## Citation

If you use PyADM1ODE in your research, please cite:

```bibtex
@software{gaida2025pyadm1ode,
  author = {Gaida, Daniel},
  title = {PyADM1ODE: Python Framework for Agricultural Biogas Plant Modeling},
  year = {2025},
  url = {https://github.com/dgaida/PyADM1ODE}
}
```

## Acknowledgments

PyADM1ODE builds upon:

- **IWA ADM1 Task Group** - Original model development
- **PyADM1** by Sadrimajd et al. - Initial Python implementation
- **SIMBA#biogas** - Substrate characterization and validation data

---

**Ready to start?** Head over to the [Quickstart Guide](user_guide/quickstart.md) and build your first biogas plant in minutes! 🚀
