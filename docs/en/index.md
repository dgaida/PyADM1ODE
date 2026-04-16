# PyADM1ODE Documentation

Welcome to PyADM1ODE - A Python framework for modeling, simulating, and optimizing agricultural biogas plants based on the Anaerobic Digestion Model No. 1 (ADM1).

## 🎯 Quick Links
<div align="center">
  <a href="https://colab.research.google.com/github/dgaida/PyADM1ODE/blob/master/examples/colab_01_basic_digester.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
</div>

<div class="grid cards" markdown>

-   :material-clock-fast:{ .lg .middle } __Quick Start__

    ---

    Get started in minutes with your first biogas plant simulation

    [:octicons-arrow-right-24: Quickstart Guide](user_guide/quickstart.md)

-   :material-download:{ .lg .middle } __Installation__

    ---

    Install PyADM1ODE on Windows, Linux, or macOS

    [:octicons-arrow-right-24: Installation Guide](user_guide/installation.md)

-   :material-book-open-variant:{ .lg .middle } __User Guide__

    ---

    Learn about the framework, components, and substrates

    [:octicons-arrow-right-24: Handbuch](user_guide/adm1_implementation.md)

-   :material-code-braces:{ .lg .middle } __Examples__

    ---

    Real-world examples from basic to advanced plants

    [:octicons-arrow-right-24: Examples](examples/basic_digester.md)

</div>

## What is PyADM1ODE?

PyADM1ODE is a comprehensive Python framework for agricultural biogas plant modeling that combines:

- **Scientific accuracy**: Based on IWA's ADM1 model, the international standard for anaerobic digestion.
- **Modular architecture**: Mix and match components (digesters, CHP units, pumps, mixers) to build any plant configuration.
- **Real-world applicability**: Validated with data from operating biogas plants.
- **Python ecosystem**: Integrates with NumPy, SciPy, Pandas, and visualization libraries.

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

---

## Community and Support

- **GitHub Repository**: [dgaida/PyADM1ODE](https://github.com/dgaida/PyADM1ODE)
- **Issue Tracker**: [Report bugs or request features](https://github.com/dgaida/PyADM1ODE/issues)
- **Discussions**: [Ask questions and share ideas](https://github.com/dgaida/PyADM1ODE/discussions)

## License

PyADM1ODE is open-source software licensed under the MIT License.
