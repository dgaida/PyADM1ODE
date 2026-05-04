# Typical Applications

PyADM1ODE can be used for a wide range of tasks, from plant design to real-time optimization.

## 1. Plant Design and Optimization

Test different plant configurations to find the optimal setup for your needs.

```python
from pyadm1.configurator import BiogasPlant, PlantConfigurator
from pyadm1.substrates import Feedstock

# Test different digester sizes
for V_liq in [1500, 2000, 2500]:
    plant = BiogasPlant(f"Plant_{V_liq}")
    feedstock = Feedstock()
    configurator = PlantConfigurator(plant, feedstock)
    configurator.add_digester("dig1", V_liq=V_liq, Q_substrates=[15, 10, 0, 0, 0, 0, 0, 0, 0, 0])

    plant.initialize()
    results = plant.simulate(duration=30, dt=1/24)

    final = results[-1]["components"]["dig1"]
    print(f"V={V_liq} m³ → CH4={final['Q_ch4']:.1f} m³/d")
```

## 2. Substrate Optimization

Compare different substrate mixes to maximize methane production or minimize costs.

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

## 3. Energy Balance Analysis

Analyze the net energy production and parasitic loads of your plant.

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

## 4. Two-Stage Process Design

Model advanced plant designs like Temperature-Phased Anaerobic Digestion (TPAD).

```python
# Temperature-phased anaerobic digestion (TPAD)
configurator.add_digester("hydrolysis", V_liq=500, T_ad=318.15)  # 45°C
configurator.add_digester("main", V_liq=2000, T_ad=308.15)       # 35°C
configurator.connect("hydrolysis", "main", "liquid")

# Enhanced hydrolysis in stage 1, stable methanogenesis in stage 2
```

## Research Applications

This framework supports research in:

- **Process optimization**: Substrate feed strategies, retention time.
- **Control systems**: Model predictive control, feedback controllers.
- **Plant design**: Component sizing, layout optimization.
- **Energy management**: CHP scheduling, heat integration.
- **Substrate evaluation**: Biogas potential assessment.
