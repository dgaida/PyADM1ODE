# Quickstart Guide

This guide will get you up and running with PyADM1ODE in minutes.

## Table of Contents

1. [Your First Simulation](#your-first-simulation)
2. [Basic Single Digester](#basic-single-digester)
3. [Complete Plant with CHP and Heating](#complete-plant-with-chp-and-heating)
4. [Two-Stage Digestion](#two-stage-digestion)
5. [Working with Different Substrates](#working-with-different-substrates)
6. [Save and Load Configurations](#save-and-load-configurations)
7. [Understanding Results](#understanding-results)
8. [Common Patterns](#common-patterns)
9. [Next Steps](#next-steps)

## Your First Simulation

Let's simulate a simple single-stage biogas digester with corn silage and manure.

```python
from pathlib import Path
from pyadm1.configurator.plant_builder import BiogasPlant
from pyadm1.substrates.feedstock import Feedstock
from pyadm1.core.adm1 import get_state_zero_from_initial_state
from pyadm1.configurator.plant_configurator import PlantConfigurator

# 1. Create feedstock manager
feedstock = Feedstock(feeding_freq=48)  # Can change feed every 48 hours

# 2. Load initial state (steady-state values)
data_path = Path("data/initial_states")
initial_state_file = data_path / "digester_initial8.csv"
adm1_state = get_state_zero_from_initial_state(str(initial_state_file))

# 3. Define substrate feed rates [m³/day]
# [corn_silage, manure, rye, grass, wheat, gps, ccm, feed_lime, cow_manure, onions]
Q_substrates = [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]

# 4. Create and configure plant
plant = BiogasPlant("My First Biogas Plant")
configurator = PlantConfigurator(plant, feedstock)

configurator.add_digester(
    digester_id="main_digester",
    V_liq=2000.0,        # Liquid volume [m³]
    V_gas=300.0,         # Gas volume [m³]
    T_ad=308.15,         # Temperature [K] = 35°C
    Q_substrates=Q_substrates
)

# 5. Initialize and simulate
plant.initialize()

results = plant.simulate(
    duration=10.0,       # Simulation time [days]
    dt=1.0/24.0,        # Time step [days] = 1 hour
    save_interval=1.0   # Save results daily
)

# 6. View results
for result in results:
    time = result["time"]
    digester = result["components"]["main_digester"]

    print(f"Day {time:.0f}:")
    print(f"  Biogas:  {digester['Q_gas']:.1f} m³/d")
    print(f"  Methane: {digester['Q_ch4']:.1f} m³/d")
    print(f"  pH:      {digester['pH']:.2f}")
    print(f"  VFA:     {digester['VFA']:.2f} g/L")
```

**Expected Output:**
```
Day 1:
  Biogas:  1245.3 m³/d
  Methane: 748.2 m³/d
  pH:      7.32
  VFA:     2.45 g/L
...
```

## Basic Single Digester

See the complete example in [`examples/01_basic_digester.py`](../../examples/01_basic_digester.py).

### System Architecture

The basic digester consists of:

```
[Substrate Feed] → [Digester] → [Gas Storage] → [Biogas Output]
   15 m³/d corn        2000 m³      300 m³         ~1250 m³/d
   10 m³/d manure      @ 35°C       membrane       60% CH₄
```

### Key Features

- **Single-stage CSTR**: Continuously stirred tank reactor
- **Automatic gas storage**: Created per digester (low-pressure membrane)
- **Substrate mixing**: Corn silage + cattle manure
- **Temperature control**: Mesophilic (35°C)

### Plant Configuration Details

```python
# The add_digester method automatically creates:
# 1. Digester component with specified parameters
# 2. Gas storage (membrane type) sized to V_gas
# 3. Connection: Digester → Gas Storage

configurator.add_digester(
    digester_id="main_digester",
    V_liq=2000.0,               # Liquid volume [m³]
    V_gas=300.0,                # Gas volume [m³]
    T_ad=308.15,                # 35°C mesophilic
    name="Main Digester",
    load_initial_state=True,    # Load steady-state CSV
    Q_substrates=[15, 10, 0, 0, 0, 0, 0, 0, 0, 0]
)
```

### Understanding the Output

**Gas Storage Behavior:**

```python
# From results:
'gas_storage': {
    'stored_volume_m3': 150.0,      # Current volume [m³ STP]
    'pressure_bar': 1.01,            # Current pressure [bar]
    'vented_volume_m3': 0.0,         # Vented this step [m³]
    'utilization': 0.50,             # 50% full
    'Q_gas_supplied_m3_per_day': 1250.0  # Available for consumers
}
```

**When storage fills up:**
- Pressure increases from 0.95 to 1.05 bar
- At 1.05 bar (full), excess gas is vented
- Venting prevents overpressure and equipment damage
- In real plants, vented gas goes to flare

### Process Stability Indicators

```python
# Check process stability
final = results[-1]["components"]["main_digester"]

# pH should be 6.8-7.5 for stable operation
if 6.8 <= final['pH'] <= 7.5:
    print("✓ pH stable")
else:
    print(f"⚠ pH unstable: {final['pH']:.2f}")

# VFA should be < 5 g/L
if final['VFA'] < 5.0:
    print("✓ VFA acceptable")
else:
    print(f"⚠ High VFA: {final['VFA']:.2f} g/L")

# FOS/TAC ratio should be < 0.3
if final['TAC'] > 0:
    fos_tac = final['VFA'] / final['TAC']
    if fos_tac < 0.3:
        print(f"✓ FOS/TAC stable: {fos_tac:.3f}")
    else:
        print(f"⚠ FOS/TAC high: {fos_tac:.3f}")
```

**Typical Performance Metrics:**

| Metric | Value | Assessment |
|--------|-------|------------|
| Biogas production | ~1250 m³/d | Good |
| Methane content | ~60% | Typical for agricultural substrates |
| Specific gas yield | ~50 m³/m³ feed | Good for corn silage + manure |
| pH | 7.28-7.30 | Optimal (stable) |
| VFA | 2.3-2.4 g/L | Good (< 3 g/L limit) |
| TAC | 8.4-8.5 g CaCO₃/L | Excellent buffer |
| FOS/TAC | ~0.27 | Stable (< 0.3) |

## Complete Plant with CHP and Heating

Now let's add power generation and heating to create a complete biogas plant:

```python
from pyadm1.configurator.plant_builder import BiogasPlant
from pyadm1.substrates.feedstock import Feedstock
from pyadm1.configurator.plant_configurator import PlantConfigurator

# Setup
feedstock = Feedstock(feeding_freq=48)
plant = BiogasPlant("Complete Biogas Plant")
configurator = PlantConfigurator(plant, feedstock)

# Add digester with automatic gas storage
configurator.add_digester(
    digester_id="main_digester",
    V_liq=2000.0,
    V_gas=300.0,
    T_ad=308.15,
    Q_substrates=[15, 10, 0, 0, 0, 0, 0, 0, 0, 0]
)

# Add CHP unit (automatically creates flare)
configurator.add_chp(
    chp_id="chp_main",
    P_el_nom=500.0,      # Electrical power [kW]
    eta_el=0.40,         # Electrical efficiency 40%
    eta_th=0.45          # Thermal efficiency 45%
)

# Add heating system
configurator.add_heating(
    heating_id="heating_main",
    target_temperature=308.15,
    heat_loss_coefficient=0.5
)

# Connect components automatically
configurator.auto_connect_digester_to_chp("main_digester", "chp_main")
configurator.auto_connect_chp_to_heating("chp_main", "heating_main")

# Initialize and simulate
plant.initialize()
results = plant.simulate(duration=10.0, dt=1.0/24.0, save_interval=1.0)

# Analyze final results
final = results[-1]
digester = final["components"]["main_digester"]
chp = final["components"]["chp_main"]
heating = final["components"]["heating_main"]

print(f"\nFinal Results (Day {final['time']:.0f}):")
print(f"\nDigester Performance:")
print(f"  Biogas:  {digester['Q_gas']:.1f} m³/d")
print(f"  Methane: {digester['Q_ch4']:.1f} m³/d")
print(f"  CH4 Content: {digester['Q_ch4']/digester['Q_gas']*100:.1f}%")
print(f"\nCHP Performance:")
print(f"  Electrical Power: {chp['P_el']:.1f} kW")
print(f"  Thermal Power:    {chp['P_th']:.1f} kW")
print(f"  Gas Consumed:     {chp['Q_gas_consumed']:.1f} m³/d")
print(f"\nHeating:")
print(f"  Heat Supplied:    {heating['Q_heat_supplied']:.1f} kW")
print(f"  Auxiliary Heat:   {heating['P_aux_heat']:.1f} kW")
```

**Automatic Component Creation:**

PlantConfigurator creates and connects:
- **Gas Storage**: One per digester (membrane, sized to V_gas)
- **Flare**: One per CHP (safety combustion, 98% CH₄ destruction)

**Connection Chain:**
```
Digester → Gas Storage → CHP → Flare
                          ↓
                      Heating
```

## Two-Stage Digestion

See the complete example in [`examples/02_two_stage_plant.py`](../../examples/02_two_stage_plant.py).

### System Architecture

```
[Feed] → [Hydrolysis] → [Storage 1] ↘
         500 m³          304 m³       → [CHP] → [Flare]
         @ 45°C                      ↗  500 kW    98%
                                    ↓
[Effluent] → [Main] → [Storage 2] ↗  [Heat 1] + [Heat 2]
             1000 m³    150 m³        45°C       35°C
             @ 35°C
```

### Key Features

- **Temperature-phased**: Thermophilic (45°C) + Mesophilic (35°C)
- **Enhanced hydrolysis**: Higher temp in first stage
- **Stable methanogenesis**: Optimized conditions in second stage
- **Mechanical components**: Pumps and mixers for material handling
- **Energy integration**: Combined heat and power with waste heat recovery

### Configuration

```python
# Stage 1: Hydrolysis (thermophilic, 45°C)
configurator.add_digester(
    digester_id="hydrolysis_tank",
    V_liq=500.0,
    V_gas=100.0,
    T_ad=318.15,  # 45°C for faster hydrolysis
    Q_substrates=[15, 10, 0, 0, 0, 0, 0, 0, 0, 0]
)

# Stage 2: Methanogenesis (mesophilic, 35°C)
configurator.add_digester(
    digester_id="main_digester",
    V_liq=2000.0,
    V_gas=300.0,
    T_ad=308.15,  # 35°C for methanogenesis
    Q_substrates=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Fed from hydrolysis
)

# Connect in series (liquid flow)
configurator.connect("hydrolysis_tank", "main_digester", "liquid")

# Add CHP and heating for both stages
configurator.add_chp("chp_main", P_el_nom=500.0)
configurator.add_heating("heating_1", target_temperature=318.15)
configurator.add_heating("heating_2", target_temperature=308.15)

# Connect gas and heat flows
configurator.auto_connect_digester_to_chp("hydrolysis_tank", "chp_main")
configurator.auto_connect_digester_to_chp("main_digester", "chp_main")
configurator.auto_connect_chp_to_heating("chp_main", "heating_1")
configurator.auto_connect_chp_to_heating("chp_main", "heating_2")

# Add mechanical components
from pyadm1.components.mechanical import Pump, Mixer

# Feed pump
feed_pump = Pump(
    component_id="feed_pump",
    pump_type="progressive_cavity",
    Q_nom=30.0,
    pressure_head=5.0
)
plant.add_component(feed_pump)

# Transfer pump (digester 1 → digester 2)
transfer_pump = Pump(
    component_id="transfer_pump",
    pump_type="progressive_cavity",
    Q_nom=25.0,
    pressure_head=8.0
)
plant.add_component(transfer_pump)

# Mixers for both digesters
mixer_1 = Mixer(
    component_id="mixer_1",
    mixer_type="propeller",
    tank_volume=500.0,
    mixing_intensity="high",
    power_installed=15.0,
    intermittent=True,
    on_time_fraction=0.25
)
plant.add_component(mixer_1)

mixer_2 = Mixer(
    component_id="mixer_2",
    mixer_type="propeller",
    tank_volume=2000.0,
    mixing_intensity="medium",
    power_installed=10.0,
    intermittent=True,
    on_time_fraction=0.25
)
plant.add_component(mixer_2)
```

### Three-Pass Gas Flow Simulation

The simulation uses a three-pass execution model for realistic gas management:

**Pass 1 - Gas Production:**
```python
Digester 1: Q_gas = 850 m³/d → Storage 1
Digester 2: Q_gas = 400 m³/d → Storage 2
```

**Pass 2 - Storage Update:**
```python
Storage 1: receives 850 m³/d, updates pressure/volume
Storage 2: receives 400 m³/d, updates pressure/volume
# If full: vent excess to atmosphere
```

**Pass 3 - Gas Consumption:**
```python
CHP demand: 1150 m³/d biogas
Storage 1 supplies: ~675 m³/d
Storage 2 supplies: ~475 m³/d
CHP operates with actual supply
Excess to flare: (supply - consumption)
```

### Advantages of Two-Stage Design

| Aspect | Single-Stage | Two-Stage | Improvement |
|--------|-------------|-----------|-------------|
| **Hydrolysis** | Limited by mesophilic temp | Enhanced at 45°C | Faster |
| **Methanogenesis** | Must tolerate VFA spikes | Stable, pre-buffered feed | More stable |
| **OLR capacity** | 3-4 kg COD/(m³·d) | 5-8 kg COD/(m³·d) | +100% |
| **Biogas yield** | 1150 m³/d | 1253 m³/d | +9% |
| **CH₄ content** | 58% | 60% | +3.4% |

### Expected Results

**Energy Balance:**

```
Energy Production:
  Electrical (gross):       480.5 kW
  Thermal:                  540.6 kW

Parasitic Load:
  Mixer 1:                    3.75 kW
  Mixer 2:                    2.50 kW
  Pumps (estimated):          2.00 kW
  Total parasitic:            8.25 kW

Net Electrical Output:      472.3 kW

Heat Utilization:
  Heating demand:           125.4 kW
  CHP thermal supply:       540.6 kW
  Heat coverage:            431.0 %

Gas Management:
  Total production:        1253.1 m³/d
  CHP consumption:         1150.0 m³/d
  To flare:                 103.1 m³/d (8.2%)
```

## Working with Different Substrates

### Available Substrates

PyADM1 includes 10 pre-configured agricultural substrates:

1. **Corn silage (maize)** - Energy crop, high biogas yield
2. **Liquid manure (swinemanure)** - High nitrogen content
3. **Green rye (greenrye)** - Early-harvest energy crop
4. **Grass silage (grass)** - Grassland biomass
5. **Wheat (wheat)** - Cereal crop
6. **GPS (gps)** - Whole-crop grain silage
7. **CCM (ccm)** - Corn-cob-mix
8. **Feed lime (futterkalk)** - pH buffer additive
9. **Cow manure (cowmanure)** - Dairy farm manure
10. **Onions (onions)** - Vegetable waste

### Substrate Feed Examples

```python
# High-energy mix (corn + manure)
Q = [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]

# Grass-based (renewable, extensive farming)
Q = [0, 5, 0, 20, 0, 0, 0, 0, 0, 0]

# Waste-based (manure + vegetables)
Q = [0, 15, 0, 0, 0, 0, 0, 0, 10, 5]

# Energy crop focus
Q = [20, 5, 10, 0, 0, 0, 0, 0, 0, 0]
```

### Substrate Information

Get detailed substrate properties:

```python
from pyadm1.substrates.feedstock import Feedstock

# View substrate parameters
params = Feedstock.get_substrate_params_string("maize")
print(params)
```

Output:
```
pH value: 3.93
Dry matter: 31.97 %FM
Volatile solids content: 96.25 %TS
Particulate chemical oxygen demand: ...
Biochemical methane potential: 0.xxx l/gFM
```

## Save and Load Configurations

Save your plant configuration for reuse:

```python
# Save configuration
plant.to_json("my_plant_config.json")

# Load configuration later
from pyadm1.configurator.plant_builder import BiogasPlant
from pyadm1.substrates.feedstock import Feedstock

feedstock = Feedstock(feeding_freq=48)
plant = BiogasPlant.from_json("my_plant_config.json", feedstock)

# Continue simulation
plant.initialize()
results = plant.simulate(duration=10.0, dt=1.0/24.0)
```

## Understanding Results

### Key Output Variables

#### Digester Outputs
- `Q_gas` - Total biogas production [m³/d]
- `Q_ch4` - Methane production [m³/d]
- `Q_co2` - CO2 production [m³/d]
- `pH` - pH value [-]
- `VFA` - Volatile fatty acids [g HAceq/L]
- `TAC` - Total alkalinity [g CaCO3/L]

#### CHP Outputs
- `P_el` - Electrical power [kW]
- `P_th` - Thermal power [kW]
- `Q_gas_consumed` - Gas consumption [m³/d]
- `load_factor` - Operating point [0-1]

#### Heating Outputs
- `Q_heat_supplied` - Heat delivered [kW]
- `P_th_used` - CHP heat used [kW]
- `P_aux_heat` - Auxiliary heat needed [kW]

#### Gas Storage Outputs
- `stored_volume_m3` - Current volume [m³ STP]
- `pressure_bar` - Current pressure [bar]
- `utilization` - Fill level [0-1]
- `vented_volume_m3` - Gas vented [m³]
- `Q_gas_supplied_m3_per_day` - Gas available [m³/d]

## Common Patterns

### Pattern 1: Parameter Sweep

Test different substrate amounts:

```python
feed_rates = [10, 15, 20, 25]
results_all = []

for feed in feed_rates:
    Q = [feed, 10, 0, 0, 0, 0, 0, 0, 0, 0]

    plant = BiogasPlant(f"Plant_Feed_{feed}")
    configurator = PlantConfigurator(plant, feedstock)
    configurator.add_digester("dig1", V_liq=2000, Q_substrates=Q)

    plant.initialize()
    results = plant.simulate(duration=10.0, dt=1.0/24.0)

    final = results[-1]["components"]["dig1"]
    results_all.append({
        'feed': feed,
        'biogas': final['Q_gas'],
        'methane': final['Q_ch4']
    })

for r in results_all:
    print(f"Feed {r['feed']} m³/d → CH4 {r['methane']:.1f} m³/d")
```

### Pattern 2: Time Series Analysis

Track evolution over time:

```python
import matplotlib.pyplot as plt

# Extract time series
times = [r['time'] for r in results]
biogas = [r['components']['main_digester']['Q_gas'] for r in results]
methane = [r['components']['main_digester']['Q_ch4'] for r in results]
pH = [r['components']['main_digester']['pH'] for r in results]

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(times, biogas, 'b-', label='Biogas')
ax1.plot(times, methane, 'g-', label='Methane')
ax1.set_ylabel('Production [m³/d]')
ax1.legend()
ax1.grid(True)

ax2.plot(times, pH, 'r-')
ax2.set_xlabel('Time [days]')
ax2.set_ylabel('pH')
ax2.axhline(y=7.0, color='k', linestyle='--', alpha=0.3)
ax2.grid(True)

plt.tight_layout()
plt.savefig('simulation_results.png')
```

## Next Steps

Now that you've run your first simulations:

1. **Learn about components**: [Components Guide](components/index.md)
2. **Explore advanced examples**:
   - [`examples/01_basic_digester.py`](../../examples/01_basic_digester.py)
   - [`examples/02_two_stage_plant.py`](../../examples/02_two_stage_plant.py)
   - [`examples/parallel_two_stage_simulation.py`](../../examples/parallel_two_stage_simulation.py)
3. **Try parallel simulations**: Multiple scenarios simultaneously
4. **Explore MCP server**: [PyADM1ODE_mcp](https://github.com/dgaida/PyADM1ODE_mcp) for LLM-driven plant design
5. **Calibrate your model**: [PyADM1ODE_calibration](https://github.com/dgaida/PyADM1ODE_calibration) for parameter fitting
6. **Read API documentation**: Full reference for all classes

## Quick Reference

### Common Commands

```python
# Create plant
plant = BiogasPlant("My Plant")
configurator = PlantConfigurator(plant, feedstock)

# Add components
configurator.add_digester(id, V_liq, V_gas, T_ad, Q_substrates)
configurator.add_chp(id, P_el_nom, eta_el, eta_th)
configurator.add_heating(id, target_temperature)

# Connect
configurator.connect(from_id, to_id, type)
configurator.auto_connect_digester_to_chp(dig_id, chp_id)
configurator.auto_connect_chp_to_heating(chp_id, heat_id)

# Simulate
plant.initialize()
results = plant.simulate(duration, dt, save_interval)

# Save/Load
plant.to_json(filepath)
plant = BiogasPlant.from_json(filepath, feedstock)
```

### Temperature Conversions

```python
# Common temperatures
T_mesophilic = 308.15  # 35°C
T_thermophilic = 328.15  # 55°C
T_psychrophilic = 298.15  # 25°C

# Convert °C to K
T_K = T_celsius + 273.15
```

### Default Parameters

```python
# Typical digester sizes
V_liq_small = 500      # Small farm [m³]
V_liq_medium = 2000    # Medium farm [m³]
V_liq_large = 5000     # Large farm [m³]

# CHP sizes
P_el_small = 150       # Small [kW]
P_el_medium = 500      # Medium [kW]
P_el_large = 1000      # Large [kW]

# Substrate feeds
Q_low = 10             # Low loading [m³/d]
Q_medium = 20          # Medium loading [m³/d]
Q_high = 40            # High loading [m³/d]
```

## Troubleshooting

### Issue: Simulation unstable

**Symptoms**: pH drops, VFA rises, methane production decreases

**Solutions**:
- Reduce substrate feed rate
- Increase retention time (larger V_liq)
- Add buffer material (feed lime)
- Check substrate composition

### Issue: Low gas production

**Solutions**:
- Increase substrate feed
- Check substrate degradability
- Verify temperature is optimal
- Ensure adequate mixing (implicit in model)

### Issue: Slow simulation

**Solutions**:
- Increase time step `dt` (but keep < 0.1 days)
- Reduce `save_interval` for less output
- Use parallel simulation for parameter sweeps

For more help, see the [Installation Guide](installation.md) or contact daniel.gaida@th-koeln.de.

## References

- **ADM1 Model**: Batstone et al. (2002). *Anaerobic Digestion Model No. 1*. IWA Publishing.
- **Leitfaden Biogas**: FNR (2016). https://mediathek.fnr.de/leitfaden-biogas.html
