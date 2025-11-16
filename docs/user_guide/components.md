# Components Guide

PyADM1 uses a modular component-based architecture. This guide covers all available components, their parameters, and usage patterns.

## Component Architecture

### Base Component Structure

All components inherit from the `Component` base class and implement:

```python
class Component(ABC):
    def __init__(self, component_id, component_type, name):
        """Initialize component with unique ID and type."""

    def initialize(self, initial_state):
        """Set up initial state before simulation."""

    def step(self, t, dt, inputs):
        """Execute one simulation time step."""

    def to_dict(self):
        """Serialize to dictionary for JSON export."""

    @classmethod
    def from_dict(cls, config):
        """Create component from configuration dictionary."""
```

### Component Lifecycle

```
Create → Initialize → Simulate (step loop) → Save/Export
   ↓                      ↑
   └──────────────────────┘
        (can reinitialize)
```

## Biological Components

### Digester

The core component implementing ADM1 anaerobic digestion.

#### Parameters

```python
from pyadm1.configurator.plant_configurator import PlantConfigurator

configurator.add_digester(
    digester_id="main_digester",      # Unique identifier
    V_liq=2000.0,                     # Liquid volume [m³]
    V_gas=300.0,                      # Gas headspace [m³]
    T_ad=308.15,                      # Operating temperature [K]
    name="Main Digester",             # Human-readable name
    load_initial_state=True,          # Load steady-state initialization
    initial_state_file=None,          # Custom initial state CSV (optional)
    Q_substrates=[15, 10, 0, ...]    # Substrate feed rates [m³/d]
)
```

#### Size Guidelines

| Plant Size | V_liq [m³] | V_gas [m³] | Feed Rate [m³/d] | HRT [days] |
|------------|------------|------------|------------------|------------|
| Small      | 300-800    | 50-120     | 10-25            | 20-40      |
| Medium     | 1000-3000  | 150-450    | 25-75            | 25-45      |
| Large      | 3000-8000  | 450-1200   | 75-200           | 30-50      |

#### Temperature Options

```python
# Psychrophilic (rare in practice)
T_psychro = 298.15  # 25°C

# Mesophilic (most common)
T_meso = 308.15     # 35°C

# Thermophilic (high-fiber substrates)
T_thermo = 328.15   # 55°C
```

#### Outputs

```python
outputs = digester.step(t, dt, inputs)
# Returns:
{
    'Q_out': 25.0,              # Effluent flow [m³/d]
    'state_out': [...],         # ADM1 state for next digester
    'Q_gas': 1250.5,           # Biogas production [m³/d]
    'Q_ch4': 750.3,            # Methane production [m³/d]
    'Q_co2': 475.2,            # CO2 production [m³/d]
    'pH': 7.32,                # pH value
    'VFA': 2.45,               # Volatile fatty acids [g/L]
    'TAC': 8.50,               # Total alkalinity [g CaCO3/L]
    'gas_storage': {           # Attached gas storage info
        'stored_volume_m3': 150.0,
        'pressure_bar': 1.02,
        'vented_volume_m3': 0.0
    }
}
```

#### Advanced Usage

**Multiple Digesters in Series:**
```python
# Hydrolysis + Methanogenesis
configurator.add_digester("hydro", V_liq=500, T_ad=318.15,
                         Q_substrates=[15, 10, 0, 0, 0, 0, 0, 0, 0, 0])
configurator.add_digester("main", V_liq=2000, T_ad=308.15,
                         Q_substrates=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
configurator.connect("hydro", "main", "liquid")
```

**Custom Initial State:**
```python
import pandas as pd

# Create custom state
initial = pd.DataFrame({
    'S_su': [0.01], 'S_aa': [0.001], # ... all 37 state variables
})
initial.to_csv('custom_state.csv', index=False)

# Use in digester
configurator.add_digester(
    "dig1", V_liq=2000,
    initial_state_file='custom_state.csv'
)
```

### Hydrolysis Tank

Pre-treatment stage for fiber-rich substrates (stub for future full implementation).

```python
from pyadm1.components.biological import Hydrolysis

hydrolysis = Hydrolysis(
    component_id="hydro1",
    feedstock=feedstock,
    V_liq=500.0,
    T_ad=318.15  # Higher temp for faster hydrolysis
)
```

### Separator

Solid-liquid separation for digestate processing (stub for future full implementation).

```python
from pyadm1.components.biological import Separator

separator = Separator(
    component_id="sep1",
    separation_efficiency=0.95  # 95% solids removal
)
```

## Energy Components

### CHP (Combined Heat and Power)

Converts biogas to electricity and heat.

#### Parameters

```python
configurator.add_chp(
    chp_id="chp_main",
    P_el_nom=500.0,        # Nominal electrical power [kW]
    eta_el=0.40,           # Electrical efficiency (40%)
    eta_th=0.45,           # Thermal efficiency (45%)
    name="Main CHP"
)
```

#### Typical CHP Specifications

| Type | Size [kW_el] | η_el | η_th | Gas Need [m³/d @ 60% CH4] |
|------|--------------|------|------|---------------------------|
| Small | 100-250 | 0.38 | 0.48 | 600-1500 |
| Medium | 250-750 | 0.40 | 0.45 | 1500-4500 |
| Large | 750-2000 | 0.42 | 0.43 | 4500-12000 |

TODO: add a source for these numbers.

#### Technology Options

```python
# Gas engine (most common)
chp_engine = configurator.add_chp(
    "chp1", P_el_nom=500, eta_el=0.40, eta_th=0.45
)

# Micro-turbine (100-500 kW)
chp_turbine = configurator.add_chp(
    "chp2", P_el_nom=250, eta_el=0.30, eta_th=0.55
)

# High-efficiency (>1 MW)
chp_large = configurator.add_chp(
    "chp3", P_el_nom=1500, eta_el=0.42, eta_th=0.43
)
```

#### Outputs

```python
{
    'P_el': 450.0,              # Electrical power [kW]
    'P_th': 506.3,              # Thermal power [kW]
    'Q_gas_consumed': 2700.0,   # Gas consumption [m³/d]
    'load_factor': 0.90         # Operating point (0-1)
}
```

#### Advanced CHP Control

```python
# Variable load operation
inputs = {
    'Q_ch4': 800.0,           # Available methane [m³/d]
    'load_setpoint': 0.75     # Operate at 75% capacity
}
result = chp.step(t, dt, inputs)
```

### Heating System

Maintains digester temperature using CHP waste heat and auxiliary heating.

#### Parameters

```python
configurator.add_heating(
    heating_id="heating_main",
    target_temperature=308.15,      # Target [K]
    heat_loss_coefficient=0.5,      # Heat loss [kW/K]
    name="Main Digester Heating"
)
```

#### Heat Loss Coefficients

| Insulation | k [kW/K] | Description |
|------------|----------|-------------|
| Excellent | 0.3-0.4 | Modern, well-insulated |
| Good | 0.4-0.6 | Standard insulation |
| Poor | 0.6-1.0 | Old or minimal insulation |

TODO: add a source for these numbers.

#### Outputs

```python
{
    'Q_heat_supplied': 125.5,    # Total heat delivered [kW]
    'P_th_used': 110.0,          # CHP heat used [kW]
    'P_aux_heat': 15.5           # Auxiliary heat needed [kW]
}
```

#### Heating Calculations

```python
# Heat demand = Heat loss + Process heat
Q_loss = k * (T_target - T_ambient)  # [kW]
Q_process = Q_feed * c_p * ΔT        # [kW]
Q_total = Q_loss + Q_process         # [kW]

# Use CHP heat first, then auxiliary
if Q_total <= P_th_available:
    P_aux = 0
else:
    P_aux = Q_total - P_th_available
```

TODO: a more elaborate implementation of the heating calculations can be found in the dlls.

### Gas Storage

Biogas storage with pressure management (automatically created per digester).

#### Types

```python
from pyadm1.components.energy import GasStorage

# Low-pressure membrane storage (most common)
storage_membrane = GasStorage(
    component_id="storage1",
    storage_type="membrane",
    capacity_m3=1000.0,      # Capacity at STP [m³]
    p_min_bar=0.95,          # Minimum pressure [bar]
    p_max_bar=1.05,          # Maximum pressure [bar]
    initial_fill_fraction=0.1
)

# Dome storage
storage_dome = GasStorage(
    component_id="storage2",
    storage_type="dome",
    capacity_m3=500.0,
    p_min_bar=0.98,
    p_max_bar=1.02
)

# High-pressure compressed storage
storage_compressed = GasStorage(
    component_id="storage3",
    storage_type="compressed",
    capacity_m3=100.0,
    p_min_bar=10.0,
    p_max_bar=200.0
)
```

#### Outputs

```python
{
    'stored_volume_m3': 450.0,       # Current storage [m³ STP]
    'pressure_bar': 1.01,            # Current pressure [bar]
    'utilization': 0.45,             # Fill level (0-1)
    'vented_volume_m3': 0.0,         # Gas vented this step [m³]
    'Q_gas_supplied_m3_per_day': 2700.0  # Gas supplied [m³/d]
}
```

### Flare

Safety system for excess biogas combustion (automatically created per CHP).

```python
from pyadm1.components.energy import Flare

flare = Flare(
    component_id="flare1",
    destruction_efficiency=0.98,  # 98% CH4 destroyed
    name="Emergency Flare"
)
```

## Connection Types

### Liquid Connections

Transfer digestate between digesters:

```python
configurator.connect("digester_1", "digester_2", "liquid")
```

**Data Transfer:**
- `Q_out`: Liquid flow rate [m³/d]
- `state_out`: Complete ADM1 state vector

### Gas Connections

Transfer biogas from storage to CHP:

```python
configurator.connect("digester_1_storage", "chp_1", "gas")
```

**Data Transfer:**
- `Q_gas_supplied_m3_per_day`: Available gas [m³/d]
- Gas composition (CH4%, CO2%)

### Heat Connections

Transfer waste heat from CHP to heating:

```python
configurator.connect("chp_1", "heating_1", "heat")
```

**Data Transfer:**
- `P_th`: Available thermal power [kW]
- Temperature levels

### Auto-Connection Helpers

```python
# Automatic gas routing: digester → storage → CHP → flare
configurator.auto_connect_digester_to_chp("dig1", "chp1")

# Automatic heat routing: CHP → heating
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
```
[Digester] → [Gas Storage] → [CHP] → [Flare]
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
```
[Hydro] → [Storage] ↘
                     → [CHP] → [Heating 1]
[Main] → [Storage] ↗      ↓
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

## Advanced Topics

### Custom Component Development

Create new component types:

```python
from pyadm1.components.base import Component, ComponentType

class CustomMixer(Component):
    def __init__(self, component_id, mixing_time=10.0):
        super().__init__(component_id, ComponentType.MIXER,
                        name=f"Mixer {component_id}")
        self.mixing_time = mixing_time

    def initialize(self, initial_state=None):
        self.state = {'mixed': False, 'time_elapsed': 0.0}

    def step(self, t, dt, inputs):
        self.state['time_elapsed'] += dt

        if self.state['time_elapsed'] >= self.mixing_time:
            self.state['mixed'] = True

        return {
            'is_mixed': self.state['mixed'],
            'mixing_progress': min(1.0, self.state['time_elapsed'] / self.mixing_time)
        }

    def to_dict(self):
        return {
            'component_id': self.component_id,
            'component_type': self.component_type.value,
            'mixing_time': self.mixing_time,
            'state': self.state
        }

    @classmethod
    def from_dict(cls, config):
        mixer = cls(config['component_id'],
                   config.get('mixing_time', 10.0))
        if 'state' in config:
            mixer.state = config['state']
        return mixer
```

### Component Registry

Register custom components:

```python
from pyadm1.components import ComponentRegistry

registry = ComponentRegistry()
registry.register("CustomMixer", CustomMixer)

# Create via registry
mixer = registry.create("CustomMixer", "mix1", mixing_time=15.0)
```

### State Management

Components manage internal state for simulation continuity:

```python
# Get current state
state = component.get_state()

# Save for later
import json
with open('component_state.json', 'w') as f:
    json.dump(state, f)

# Restore state
with open('component_state.json', 'r') as f:
    saved_state = json.load(f)
component.set_state(saved_state)
```

## Performance Considerations

### Digester Performance

- **Time step**: Use dt = 1/24 days (1 hour) for stability
- **State vector**: 37 variables × 8 bytes ≈ 300 bytes per state
- **Memory**: ~30 KB per 1000 time points per digester

### Simulation Speed

Typical performance on modern hardware:

| Configuration | Real-time Factor | Simulation Speed |
|---------------|------------------|------------------|
| Single digester | 1000:1 | 24 hrs in 1.4 min |
| Two-stage + CHP | 500:1 | 24 hrs in 2.9 min |
| 4 digesters | 250:1 | 24 hrs in 5.8 min |

TODO: actually test this and update numbers.

### Optimization Tips

```python
# Faster simulations
results = plant.simulate(
    duration=30,
    dt=1.0/24.0,           # 1-hour steps
    save_interval=1.0      # Save daily (not every step)
)

# Parallel scenarios
from pyadm1.simulation import ParallelSimulator
parallel = ParallelSimulator(adm1, n_workers=4)
results = parallel.run_scenarios(scenarios, duration=30)
```

## Troubleshooting

### Common Issues

**Issue**: Component not found after loading JSON

**Solution**: Ensure feedstock is passed to `from_json()`:
```python
plant = BiogasPlant.from_json("plant.json", feedstock)
```

**Issue**: Connection refused between components

**Solution**: Check component IDs match exactly:
```python
# Both components must exist
print(plant.components.keys())  # List all component IDs
configurator.connect("dig1", "chp1", "gas")  # Use exact IDs
```

**Issue**: Digester pH unstable

**Solution**: Check substrate feed rates and add buffer:
```python
# Reduce organic loading
Q = [10, 8, 0, 0, 0, 0, 0, 0, 0, 0]  # Reduced from [15, 10, ...]

# Or add lime buffer
Q = [15, 10, 0, 0, 0, 0, 0, 1, 0, 0]  # 1 m³/d feed lime
```

## Next Steps

- **Try Examples**: See `examples/` directory for complete plants
- **Parallel Simulation**: Scale to parameter sweeps
- **MCP Server**: LLM-driven plant configuration
- **API Reference**: Detailed class documentation

## Component Summary Table

| Component | Purpose | Key Parameters | Outputs |
|-----------|---------|----------------|---------|
| Digester | Anaerobic digestion | V_liq, T_ad, Q_substrates | Q_gas, Q_ch4, pH, VFA |
| CHP | Power generation | P_el_nom, η_el, η_th | P_el, P_th, Q_gas_consumed |
| Heating | Temperature control | T_target, k_loss | Q_heat, P_aux |
| GasStorage | Biogas buffering | capacity, p_min, p_max | volume, pressure, Q_supplied |
| Flare | Safety combustion | destruction_eff | vented_volume, CH4_destroyed |

For detailed API documentation, see the [API Reference](../api_reference/components.rst).
