# Components Guide

PyADM1 uses a modular, component-based architecture. This guide covers all available components, their parameters, and usage patterns.

## Component Architecture

### Base Component Structure

All components inherit from the `Component` base class and implement:

```python
class Component(ABC):
    def __init__(self, component_id, component_type, name):
        """Initialize component with unique ID and type."""

    def initialize(self, initial_state):
        """Set initial state before simulation."""

    def step(self, t, dt, inputs):
        """Execute one simulation time step."""

    def to_dict(self):
        """Serialize to dictionary for JSON export."""

    @classmethod
    def from_dict(cls, config):
        """Create component from configuration dictionary."""
```

### Component Lifecycle

```text
Create → Initialize → Simulate (step loop) → Save/Export
  ↓                       ↑
  └───────────────────────┘
        (can be re-initialized)
```

## Component Overview

PyADM1 provides several categories of components:

### [Biological Components](biological.md)

Components for biological conversion processes:

- **Digester**: Main fermenter with the ADM1 model for anaerobic digestion. A hydrolysis pre-tank is simply a `Digester` with higher temperature and shorter HRT — there is no separate class.  
- **Separator**: Solid–liquid separation for digestate processing.  

### [Energy Components](energy.md)

Components for energy generation and storage:

- **CHP**: Combined heat and power unit for electricity and heat generation.  
- **Heating**: Heating system for temperature control.  
- **GasStorage**: Biogas storage with pressure management.  
- **Flare**: Safety flare for excess gas.  

### [Mechanical Components](mechanical.md)

Mechanical plant components:

- **Pump**: Pumps for substrate transport and recirculation.  
- **Mixer**: Agitators for homogenization in the fermenter.  

### [Feeding Components](feeding.md)

Substrate handling and dosing:

- **SubstrateStorage**: Substrate storage tanks with quality tracking.  
- **Feeder**: Automated dosing systems.  

### [Sensors](sensors.md)

Measurement and monitoring components.

## Connection Types

### Liquid Connections

Transfer digestate between digesters:

```python
configurator.connect("digester_1", "digester_2", "liquid")
```

**Data transferred:**  
- `Q_out`: Liquid flow rate [m³/d]  
- `state_out`: Complete ADM1 state vector  

### Gas Connections

Transfer biogas from storage to CHP:

```python
configurator.connect("digester_1_storage", "chp_1", "gas")
```

**Data transferred:**  
- `Q_gas_supplied_m3_per_day`: Available gas [m³/d]  
- Gas composition (CH4%, CO2%)  

### Heat Connections

Transfer waste heat from CHP to heating system:

```python
configurator.connect("chp_1", "heating_1", "heat")
```

**Data transferred:**  
- `P_th`: Available thermal power [kW]  
- Temperature levels  

### Auto-Connection Helpers

```python
# Automatic gas routing: Digester → Storage → CHP → Flare
configurator.auto_connect_digester_to_chp("dig1", "chp1")

# Automatic heat routing: CHP → Heating
configurator.auto_connect_chp_to_heating("chp1", "heat1")
```

## Component Patterns

### Pattern 1: Single-Stage Plant

```python
configurator.add_digester("dig1", V_liq=2000, Q_substrates=[15,10,0,0,0,0,0,0,0,0])
configurator.add_chp("chp1", P_el_nom=500)
configurator.add_heating("heat1", target_temperature=308.15)

configurator.auto_connect_digester_to_chp("dig1", "chp1")
configurator.auto_connect_chp_to_heating("chp1", "heat1")
```

**Topology:**

```text
[Digester] → [Gas storage] → [CHP] → [Flare]
                              ↓
                           [Heating]
```

### Pattern 2: Two-Stage Series

```python
# Stage 1: Hydrolysis (thermophilic)
configurator.add_digester("hydro", V_liq=500, T_ad=318.15,
                         Q_substrates=[15,10,0,0,0,0,0,0,0,0])

# Stage 2: Methanogenesis (mesophilic)
configurator.add_digester("main", V_liq=2000, T_ad=308.15,
                         Q_substrates=[0,0,0,0,0,0,0,0,0,0])

# Connect liquid flow
configurator.connect("hydro", "main", "liquid")

# Single CHP for both
configurator.add_chp("chp1", P_el_nom=500)
configurator.auto_connect_digester_to_chp("hydro", "chp1")
configurator.auto_connect_digester_to_chp("main", "chp1")

# Separate heating for each stage
configurator.add_heating("heat1", target_temperature=318.15)
configurator.add_heating("heat2", target_temperature=308.15)
configurator.auto_connect_chp_to_heating("chp1", "heat1")
configurator.auto_connect_chp_to_heating("chp1", "heat2")
```

**Topology:**

```text
[Hydrolysis] → [Storage] ↘
                          → [CHP] → [Heating 1]
[Main]       → [Storage] ↗     ↓
                            [Heating 2]
```

### Pattern 3: Parallel Digesters

```python
# Multiple digesters feeding one CHP
for i in range(3):
    configurator.add_digester(
        f"dig{i+1}",
        V_liq=1000,
        Q_substrates=[10, 5, 0, 0, 0, 0, 0, 0, 0, 0]
    )

configurator.add_chp("chp1", P_el_nom=1000)

for i in range(3):
    configurator.auto_connect_digester_to_chp(f"dig{i+1}", "chp1")
```

## Complete Integration Example

### Full Feeding Chain

```python
from pyadm1.configurator import BiogasPlant, PlantConfigurator
from pyadm1.components.feeding import SubstrateStorage, Feeder
from pyadm1.components.mechanical import Pump, Mixer
from pyadm1.substrates import Feedstock

# Setup
feedstock = Feedstock(feeding_freq=48)
plant = BiogasPlant("Complete Plant")
config = PlantConfigurator(plant, feedstock)

# 1. Substrate storage
storage = SubstrateStorage(
    "silo1",
    storage_type="vertical_silo",
    substrate_type="corn_silage",
    capacity=1000,
    initial_level=800
)
plant.add_component(storage)

# 2. Feeder
feeder = Feeder(
    "feed1",
    feeder_type="screw",
    Q_max=20.0,
    substrate_type="solid"
)
plant.add_component(feeder)

# 3. Transfer pump
pump = Pump(
    "pump1",
    pump_type="progressive_cavity",
    Q_nom=15.0,
    pressure_head=50.0
)
plant.add_component(pump)

# 4. Digester
digester, _ = config.add_digester(
    "main_digester",
    V_liq=2000,
    Q_substrates=[15, 10, 0, 0, 0, 0, 0, 0, 0, 0]
)

# 5. Mixer
mixer = Mixer(
    "mix1",
    mixer_type="propeller",
    tank_volume=2000,
    mixing_intensity="medium",
    intermittent=True,
    on_time_fraction=0.25
)
plant.add_component(mixer)

# 6. CHP and heating
config.add_chp("chp1", P_el_nom=500)
config.add_heating("heat1", target_temperature=308.15)

# Connect components
config.connect("silo1", "feed1", "default")
config.connect("feed1", "pump1", "default")
config.connect("pump1", "main_digester", "liquid")
config.auto_connect_digester_to_chp("main_digester", "chp1")
config.auto_connect_chp_to_heating("chp1", "heat1")

# Initialize and simulate
plant.initialize()
results = plant.simulate(duration=30, dt=1/24, save_interval=1.0)

# Analyze results
final = results[-1]
print("\nFinal results:")
print(f"Storage level: {final['components']['silo1']['current_level']:.1f} t")
print(f"Feeder throughput: {final['components']['feed1']['total_mass_fed']:.1f} t")
print(f"Pump energy: {final['components']['pump1']['energy_consumed']:.1f} kWh")
print(f"Mixer energy: {final['components']['mix1']['energy_consumed']:.1f} kWh")
print(f"Biogas: {final['components']['main_digester']['Q_gas']:.1f} m³/d")
```

### Energy Analysis

```python
def calculate_parasitic_load(results):
    """Compute total parasitic energy consumption."""
    final = results[-1]
    components = final['components']

    # Mechanical components
    pump_energy = components.get('pump1', {}).get('energy_consumed', 0)
    mixer_energy = components.get('mix1', {}).get('energy_consumed', 0)
    feeder_power = components.get('feed1', {}).get('P_consumed', 0)

    # CHP production
    chp_energy = components.get('chp1', {}).get('P_el', 0) * 30 * 24  # kWh

    parasitic_total = pump_energy + mixer_energy
    parasitic_fraction = parasitic_total / chp_energy

    return {
        'pump_energy': pump_energy,
        'mixer_energy': mixer_energy,
        'total_parasitic': parasitic_total,
        'chp_production': chp_energy,
        'parasitic_fraction': parasitic_fraction,
        'net_energy': chp_energy - parasitic_total
    }

analysis = calculate_parasitic_load(results)
print(f"\nEnergy analysis:")
print(f"CHP production: {analysis['chp_production']:.0f} kWh")
print(f"Pump consumption: {analysis['pump_energy']:.0f} kWh")
print(f"Mixer consumption: {analysis['mixer_energy']:.0f} kWh")
print(f"Parasitic load: {analysis['parasitic_fraction']:.1%}")
print(f"Net production: {analysis['net_energy']:.0f} kWh")
```

## Troubleshooting

### Common Issues

**Problem:** Pump delivers no flow

**Solution:** Check pressure head and speed settings

```python
result = pump.step(0, 1/24, {
    'Q_setpoint': 15.0,
    'enable_pump': True,
    'pressure_head': 50.0  # Ensure sufficient head
})

if result['Q_actual'] < 0.5 * result['Q_setpoint']:
    print("Check: pressure head, blockages, power supply")
```

**Problem:** Mixer consumes too much energy

**Solution:** Use intermittent operation

```python
# Instead of continuous (360 kWh/day):
mixer_continuous = Mixer("mix1", intermittent=False)

# Use intermittent (90 kWh/day):
mixer_optimal = Mixer(
    "mix1",
    intermittent=True,
    on_time_fraction=0.25  # 75% energy savings
)
```

**Problem:** Feeder accuracy too low

**Solution:** Use a more precise feeder type or disable noise

```python
# Less precise: screw (±5%)
feeder_screw = Feeder("feed1", feeder_type="screw")

# More precise: piston (±1%)
feeder_piston = Feeder("feed1", feeder_type="piston")

# Or disable realistic noise for idealized simulation
feeder_ideal = Feeder(
    "feed1",
    feeder_type="screw",
    enable_dosing_noise=False
)
```

**Problem:** Storage quality degrades too quickly

**Solution:** Check temperature and storage type

```python
# Bad: clamp at 20 °C
storage_poor = SubstrateStorage(
    "clamp1",
    storage_type="clamp",        # High degradation
    temperature=293.15           # Warm
)
# Degradation: ~0.003/d → 91% quality after 30 days

# Better: silo at 15 °C
storage_good = SubstrateStorage(
    "silo1",
    storage_type="vertical_silo", # Low degradation
    temperature=288.15            # Cool
)
# Degradation: ~0.0005/d → 98.5% quality after 30 days
```

## Component Overview Table

| Component | Purpose | Main parameters | Typical power | Notes |
|-----------|---------|-----------------|---------------|-------|
| **Pump** | Material transfer | Q_nom, pressure_head | 2–10 kW | Size for 80–90% of max flow |
| **Mixer** | Homogenization | mixing_intensity, on_time | 5–20 kW | Use intermittent (25% on-time) |
| **Storage** | Substrate storage | capacity, storage_type | 0 kW | Monitor quality degradation |
| **Feeder** | Dosing | Q_max, feeder_type | 1–5 kW | Enable dosing noise for realism |

## Next Steps

- **Examples**: See the detailed component guides for full implementations  
- **Optimization**: Use parameter studies to optimize component sizing  
- **[API Reference](../../api/biological.md)**: Detailed class documentation for advanced features  

## Related Documentation

- [Biological Components](biological.md)  
- [Energy Components](energy.md)  
- [Mechanical Components](mechanical.md)  
- [Feeding Components](feeding.md)  
- [Sensors](sensors.md)  

## Core Concepts

### Component Base Class

All plant components inherit from the `Component` base class and provide:  
- `step(t, dt, inputs)`: Execute one simulation time step.  
- `initialize(state)`: Initialize the component state.  
- `to_dict()` / `from_dict()`: Serialization for JSON export/import.  

### Automatic Component Creation

The `PlantConfigurator` automatically creates and connects:  
- **Gas storage**: One per digester (membrane type, sized to gas volume).  
- **Flare**: One per CHP unit (safety combustion of excess gas).  

This reduces configuration complexity while ensuring realistic plant behavior.
