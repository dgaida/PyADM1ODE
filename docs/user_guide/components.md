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

## Mechanical Components

### Pump

Pumps transfer substrates, digestate, and other fluids throughout the biogas plant.

#### Parameters

```python
from pyadm1.components.mechanical import Pump

pump = Pump(
    component_id="pump1",
    pump_type="progressive_cavity",  # or "centrifugal", "piston"
    Q_nom=15.0,                      # Nominal flow rate [m³/h]
    pressure_head=50.0,              # Design pressure [m]
    efficiency=None,                 # Auto-calculated if None
    motor_efficiency=0.90,           # Motor efficiency (0-1)
    fluid_density=1020.0,            # Fluid density [kg/m³]
    speed_control=True,              # Variable speed drive
    name="Feed Pump"
)
```

#### Pump Types Comparison

| Type | Best Application | Efficiency | Advantages | Disadvantages |
|------|-----------------|------------|------------|---------------|
| **Centrifugal** | Low viscosity liquids | 65-75% | • High flow rates<br>• Robust<br>• Low maintenance | • Not self-priming<br>• Poor with high viscosity<br>• Efficiency drops with slurries |
| **Progressive Cavity** | Viscous slurries | 50-70% | • Handles high solids<br>• Self-priming<br>• Gentle conveying | • Lower efficiency<br>• Higher maintenance<br>• Speed-dependent pressure |
| **Piston** | High pressure applications | 70-85% | • High pressure capability<br>• Precise flow control<br>• Good efficiency | • Higher cost<br>• More complex<br>• Sensitive to solids |

#### Sizing Guidelines

**Flow Rate Selection:**

| Plant Size | Substrate Feed [m³/d] | Pump Q_nom [m³/h] | Typical Pressure [m] |
|------------|----------------------|-------------------|---------------------|
| Small | 10-25 | 5-15 | 30-50 |
| Medium | 25-75 | 15-40 | 40-60 |
| Large | 75-200 | 40-100 | 50-80 |

% TODO: add sources for those numbers

**Pressure Head Considerations:**

```python
# Calculate required pressure head
H_static = 5.0      # Vertical lift [m]
H_friction = 8.0    # Pipe friction losses [m]
H_process = 2.0     # Process pressure [m]
H_safety = 1.2      # Safety factor

H_required = (H_static + H_friction + H_process) * H_safety
# = 18.0 m

pump = Pump("pump1", Q_nom=15, pressure_head=H_required)
```

#### Outputs

```python
{
    'P_consumed': 8.5,           # Power consumption [kW]
    'Q_actual': 10.0,            # Actual flow rate [m³/h]
    'is_running': True,          # Operating state
    'efficiency': 0.68,          # Current efficiency
    'pressure_actual': 48.5,     # Actual pressure [m]
    'speed_fraction': 1.0,       # Speed (0-1)
    'specific_energy': 0.85      # Energy per volume [kWh/m³]
}
```

#### Usage Example

```python
# Progressive cavity pump for substrate feeding
pump = Pump(
    component_id="feed_pump",
    pump_type="progressive_cavity",
    Q_nom=15.0,
    pressure_head=50.0,
    speed_control=True
)

pump.initialize()

# Operate at 80% capacity
result = pump.step(
    t=0,
    dt=1/24,
    inputs={
        'Q_setpoint': 12.0,
        'enable_pump': True,
        'fluid_density': 1020,
        'pressure_head': 50
    }
)

print(f"Power: {result['P_consumed']:.1f} kW")
print(f"Flow: {result['Q_actual']:.1f} m³/h")
print(f"Efficiency: {result['efficiency']:.1%}")
```

#### Power Consumption

Pumps calculate power based on hydraulic formula:

```
P_hydraulic = ρ × g × Q × H / 1000  [kW]
P_shaft = P_hydraulic / η_pump
P_electrical = P_shaft / η_motor
```

Where:
- ρ = fluid density [kg/m³]
- g = 9.81 m/s²
- Q = flow rate [m³/s]
- H = pressure head [m]
- η = efficiency

**Typical Power Consumption:**

| Flow [m³/h] | Head [m] | Pump Type | Power [kW] |
|-------------|----------|-----------|------------|
| 10 | 30 | Centrifugal | 2.5 |
| 10 | 50 | Progressive Cavity | 4.5 |
| 15 | 50 | Progressive Cavity | 6.8 |
| 20 | 60 | Piston | 10.5 |

% TODO: add sources for those numbers

### Mixer

Mixers maintain homogeneity in digesters, preventing stratification and optimizing substrate-bacteria contact.

#### Parameters

```python
from pyadm1.components.mechanical import Mixer

mixer = Mixer(
    component_id="mix1",
    mixer_type="propeller",          # or "paddle", "jet"
    tank_volume=2000.0,              # Tank volume [m³]
    tank_diameter=None,              # Auto-calculated if None
    mixing_intensity="medium",       # "low", "medium", "high"
    power_installed=None,            # Auto-calculated if None
    intermittent=True,               # Intermittent operation
    on_time_fraction=0.25,           # 25% on-time
    name="Main Mixer"
)
```

#### Mixer Types

| Type | Flow Pattern | Best For | Power Factor | Typical Speed [rpm] |
|------|--------------|----------|--------------|-------------------|
| **Propeller** | Axial | Large tanks, liquid substrates | 1.0× | 40-100 |
| **Paddle** | Radial | High-solids, fibrous material | 1.2× | 20-60 |
| **Jet** | Hydraulic | Recirculation mixing | 1.5× | N/A (pump) |

#### Mixing Intensity

| Intensity | Specific Power [W/m³] | Mixing Time [min] | Application |
|-----------|----------------------|-------------------|-------------|
| **Low** | 3 | 15-30 | Liquid manure, low-solids substrates |
| **Medium** | 5 | 8-15 | Standard operation, energy crops |
| **High** | 8 | 3-8 | High-solids, fibrous substrates |

% TODO: add sources for those numbers

#### Sizing Example

```python
# For 2000 m³ digester with medium intensity
tank_volume = 2000  # m³
specific_power = 5  # W/m³ for medium intensity

P_required = (tank_volume * specific_power) / 1000  # kW
P_required *= 1.2  # Safety factor for fibrous material
# = 12 kW

mixer = Mixer(
    "mix1",
    tank_volume=2000,
    mixing_intensity="medium",
    power_installed=15  # Round up to standard size
)
```

#### Outputs

```python
{
    'P_consumed': 12.5,          # Current power [kW]
    'P_average': 3.1,            # Time-averaged [kW]
    'is_running': True,          # Operating state
    'mixing_quality': 0.85,      # Quality index (0-1)
    'reynolds_number': 15000,    # Flow regime indicator
    'power_number': 0.32,        # Dimensionless power
    'mixing_time': 8.5,          # Time to homogeneity [min]
    'shear_rate': 45.2,          # Average shear [1/s]
    'specific_power': 6.25,      # Power density [kW/m³]
    'tip_speed': 2.8             # Impeller tip speed [m/s]
}
```

#### Usage Example

```python
# Medium intensity propeller mixer
mixer = Mixer(
    component_id="mix1",
    mixer_type="propeller",
    tank_volume=2000,
    mixing_intensity="medium",
    intermittent=True,
    on_time_fraction=0.25  # 6 hours per day
)

mixer.initialize()

result = mixer.step(
    t=0,
    dt=1/24,
    inputs={
        'enable_mixing': True,
        'speed_setpoint': 1.0,
        'fluid_viscosity': 0.05  # Pa·s
    }
)

print(f"Power: {result['P_consumed']:.1f} kW")
print(f"Average power: {result['P_average']:.1f} kW")
print(f"Mixing quality: {result['mixing_quality']:.2f}")
print(f"Mixing time: {result['mixing_time']:.1f} min")
```

#### Intermittent Operation

Intermittent mixing reduces energy consumption:

```python
# Continuous vs intermittent comparison
# Continuous: 15 kW × 24 h = 360 kWh/day
# Intermittent (25%): 15 kW × 6 h = 90 kWh/day
# Savings: 270 kWh/day (75%)

mixer_continuous = Mixer(
    "mix_cont",
    tank_volume=2000,
    intermittent=False
)

mixer_intermittent = Mixer(
    "mix_int",
    tank_volume=2000,
    intermittent=True,
    on_time_fraction=0.25
)

# Both achieve similar mixing quality
```

**Recommended On-Time Fractions:**

| Substrate Type | On-Time | Total Hours/Day | Energy Savings |
|----------------|---------|-----------------|----------------|
| Liquid manure | 15-20% | 3.6-4.8 h | 80-85% |
| Energy crops | 20-30% | 4.8-7.2 h | 70-80% |
| High-solids mix | 25-35% | 6.0-8.4 h | 65-75% |
| Fibrous materials | 30-40% | 7.2-9.6 h | 60-70% |

% TODO: add sources for those numbers

## Feeding Components

### SubstrateStorage

Storage facilities for different substrate types with quality tracking.

#### Parameters

```python
from pyadm1.components.feeding import SubstrateStorage

storage = SubstrateStorage(
    component_id="silo1",
    storage_type="vertical_silo",    # See table below
    substrate_type="corn_silage",    # See table below
    capacity=1000.0,                 # Max capacity [t or m³]
    initial_level=800.0,             # Initial inventory
    degradation_rate=None,           # Auto-calculated
    temperature=288.15,              # Storage temp [K] (15°C)
    name="Corn Silage Silo"
)
```

#### Storage Types

| Type | Degradation [1/d] | Best For | Typical Size | Investment |
|------|------------------|----------|--------------|------------|
| **Vertical Silo** | 0.0005 | Corn/grass silage | 500-2000 t | High |
| **Horizontal Silo** | 0.0008 | Large operations | 1000-3000 t | Medium |
| **Bunker Silo** | 0.001 | Drive-over access | 1000-5000 t | Medium |
| **Clamp** | 0.0025 | Seasonal storage | 500-2000 t | Low |
| **Above-ground Tank** | 0.0002 | Liquid manure | 500-3000 m³ | High |
| **Below-ground Tank** | 0.0001 | Liquid storage | 1000-5000 m³ | Very High |

% TODO: add sources for those numbers

#### Substrate Types

| Substrate | Density [kg/m³] | DM [%] | VS [% of DM] | Typical Storage |
|-----------|----------------|--------|--------------|-----------------|
| Corn silage | 650 | 35 | 95 | Silo |
| Grass silage | 700 | 30 | 92 | Silo |
| Whole crop silage | 680 | 32 | 94 | Silo/Bunker |
| Liquid manure | 1020 | 8 | 80 | Tank |
| Solid manure | 850 | 25 | 75 | Clamp |
| Food waste | 1000 | 20 | 90 | Tank |

% TODO: add sources for those numbers

#### Quality Degradation

Storage quality degrades over time:

```python
# Quality factor at time t:
quality(t) = quality(0) × exp(-k × t)

# Where:
# k = degradation_rate [1/d]
# t = storage_time [days]

# Example: Corn silage in vertical silo
# After 30 days: quality = 1.0 × exp(-0.0005 × 30) = 0.985 (98.5%)
# After 90 days: quality = 1.0 × exp(-0.0005 × 90) = 0.956 (95.6%)
```

**Temperature Effect:**

Temperature affects degradation (Q10 = 2):

```python
# Degradation increases with temperature
T_ref = 288.15  # 15°C reference
k_ref = 0.0005  # Base rate

# At 20°C (293.15 K):
k_20C = k_ref × 2^((293.15-288.15)/10) = 0.0007

# At 10°C (283.15 K):
k_10C = k_ref × 2^((283.15-288.15)/10) = 0.0004
```

#### Outputs

```python
{
    'current_level': 750.0,      # Inventory [t or m³]
    'utilization': 0.75,         # Fill level (0-1)
    'quality_factor': 0.95,      # Quality (0-1)
    'available_mass': 712.5,     # Usable mass
    'degradation_rate': 0.0005,  # Current rate
    'losses_this_step': 0.4,     # Losses [t or m³]
    'withdrawn_this_step': 15.0, # Withdrawn [t or m³]
    'is_empty': False,
    'is_full': False,
    'storage_time': 25.5,        # Days stored
    'dry_matter': 35.0,          # DM [%]
    'vs_content': 95.0           # VS [% of DM]
}
```

#### Usage Example

```python
# Corn silage storage
storage = SubstrateStorage(
    component_id="silo1",
    storage_type="vertical_silo",
    substrate_type="corn_silage",
    capacity=1000,
    initial_level=800
)

storage.initialize()

# Daily operation
result = storage.step(
    t=10,
    dt=1,
    inputs={
        'withdrawal_rate': 15,    # m³/d or t/d
        'refill_amount': 0,
        'temperature': 288.15
    }
)

print(f"Level: {result['current_level']:.1f} t")
print(f"Quality: {result['quality_factor']:.3f}")
print(f"Available: {result['available_mass']:.1f} t")
print(f"Losses: {result['losses_this_step']:.2f} t")
```

#### Storage Management Strategy

```python
# Optimal refill timing
def should_refill(storage_result, safety_days=7):
    """Determine if refill is needed"""
    level = storage_result['current_level']
    daily_usage = 15  # t/d
    days_remaining = level / daily_usage

    return days_remaining < safety_days

# Quality-based rotation
def check_quality(storage_result, min_quality=0.90):
    """Alert if quality too low"""
    quality = storage_result['quality_factor']
    if quality < min_quality:
        print(f"Warning: Quality at {quality:.1%}")
        return False
    return True
```

### Feeder

Automated dosing systems for precise substrate feeding.

#### Parameters

```python
from pyadm1.components.feeding import Feeder

feeder = Feeder(
    component_id="feed1",
    feeder_type="screw",             # Auto-selected if None
    Q_max=20.0,                      # Max flow [m³/d or t/d]
    substrate_type="solid",          # "solid", "slurry", "liquid", "fibrous"
    dosing_accuracy=None,            # Auto-calculated
    power_installed=None,            # Auto-calculated
    enable_dosing_noise=True,        # Realistic variance
    name="Screw Feeder"
)
```

#### Feeder Types

| Type | Accuracy [±%] | Best For | Speed Control | Power [kW/m³/h] |
|------|--------------|----------|---------------|----------------|
| **Screw** | 5 | Solid substrates | Good | 0.8 |
| **Twin Screw** | 3 | Better control | Excellent | 1.0 |
| **Progressive Cavity** | 2 | Viscous slurries | Good | 1.2 |
| **Piston** | 1 | Precise dosing | Excellent | 1.5 |
| **Centrifugal Pump** | 8 | Low viscosity | Fair | 0.5 |
| **Mixer Wagon** | 10 | Batch feeding | N/A | 2.0 |

% TODO: add sources for those numbers

#### Dosing Accuracy

Real feeders have variance around setpoints:

```python
# With dosing_noise enabled:
# Actual flow = Setpoint + noise
# Where noise ~ Normal(0, accuracy × setpoint)

# Example: Screw feeder (5% accuracy) at 15 m³/d
# Typical range: 14.25 - 15.75 m³/d
# Occasional: 13.5 - 16.5 m³/d (±2σ)

feeder = Feeder("feed1", Q_max=20, dosing_accuracy=0.05)
```

#### Power Requirements

Power depends on substrate type:

| Substrate | Base Power [kW/m³/h] | Modifier | Total |
|-----------|---------------------|----------|-------|
| Liquid | 0.5 | ×0.7 | 0.35 |
| Slurry | 0.8 | ×1.0 | 0.80 |
| Solid | 0.8 | ×1.4 | 1.12 |
| Fibrous | 0.8 | ×1.8 | 1.44 |

% TODO: add sources for those numbers

```python
# Example: 15 m³/h screw feeder for fibrous substrate
Q_nom_h = 15 / 24  # = 0.625 m³/h
P = 0.8 * 0.625 * 1.8 * 1.3  # [base × Q × modifier × safety]
  = 1.17 kW
```

#### Outputs

```python
{
    'Q_actual': 14.8,            # Actual flow [m³/d]
    'is_running': True,
    'load_factor': 0.74,         # Load (0-1)
    'P_consumed': 2.5,           # Power [kW]
    'blockage_detected': False,  # Alarm
    'dosing_error': 1.3,         # Error [%]
    'speed_fraction': 0.95,      # Speed (0-1)
    'dosing_accuracy': 0.05,     # Accuracy
    'total_mass_fed': 1250.0     # Cumulative [t]
}
```

#### Usage Example

```python
# Screw feeder for solid substrates
feeder = Feeder(
    component_id="feed1",
    feeder_type="screw",
    Q_max=20.0,
    substrate_type="solid",
    enable_dosing_noise=True
)

feeder.initialize()

result = feeder.step(
    t=0,
    dt=1/24,
    inputs={
        'Q_setpoint': 15.0,
        'enable_feeding': True,
        'substrate_available': 500,
        'speed_setpoint': 1.0
    }
)

print(f"Target: 15.0 m³/d")
print(f"Actual: {result['Q_actual']:.2f} m³/d")
print(f"Error: {result['dosing_error']:.1f}%")
print(f"Power: {result['P_consumed']:.2f} kW")
```

#### Blockage Detection

Feeders can detect and handle blockages:

```python
# Automatic handling
if result['blockage_detected']:
    print("Blockage detected!")
    # Feeder automatically reduces flow to 10%
    # Continue monitoring

# Monitor cumulative blockages
print(f"Total blockages: {feeder.state['n_blockages']}")
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

# 3. Feed pump
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
print("\nFinal Results:")
print(f"Storage level: {final['components']['silo1']['current_level']:.1f} t")
print(f"Feeder throughput: {final['components']['feed1']['total_mass_fed']:.1f} t")
print(f"Pump energy: {final['components']['pump1']['energy_consumed']:.1f} kWh")
print(f"Mixer energy: {final['components']['mix1']['energy_consumed']:.1f} kWh")
print(f"Biogas: {final['components']['main_digester']['Q_gas']:.1f} m³/d")
```

### Energy Analysis

```python
# Calculate parasitic load
def calculate_parasitic_load(results):
    """Calculate total parasitic energy consumption"""
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
print(f"\nEnergy Analysis:")
print(f"CHP production: {analysis['chp_production']:.0f} kWh")
print(f"Pump consumption: {analysis['pump_energy']:.0f} kWh")
print(f"Mixer consumption: {analysis['mixer_energy']:.0f} kWh")
print(f"Parasitic load: {analysis['parasitic_fraction']:.1%}")
print(f"Net production: {analysis['net_energy']:.0f} kWh")
```

## Troubleshooting

### Common Issues

**Issue**: Pump not delivering flow

**Solution**: Check pressure head and speed settings
```python
result = pump.step(0, 1/24, {
    'Q_setpoint': 15.0,
    'enable_pump': True,
    'pressure_head': 50.0  # Ensure sufficient head
})

if result['Q_actual'] < 0.5 * result['Q_setpoint']:
    print("Check: pressure head, blockages, power supply")
```

**Issue**: Mixer consuming excessive energy

**Solution**: Use intermittent operation
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

**Issue**: Feeder dosing inaccuracy too high

**Solution**: Use more precise feeder type or disable noise
```python
# Less precise: Screw (±5%)
feeder_screw = Feeder("feed1", feeder_type="screw")

# More precise: Piston (±1%)
feeder_piston = Feeder("feed1", feeder_type="piston")

# Or disable realistic noise for idealized simulation
feeder_ideal = Feeder(
    "feed1",
    feeder_type="screw",
    enable_dosing_noise=False
)
```

**Issue**: Storage quality degrading too fast

**Solution**: Check temperature and storage type
```python
# Poor: Clamp storage at 20°C
storage_poor = SubstrateStorage(
    "clamp1",
    storage_type="clamp",        # High degradation
    temperature=293.15           # Warm
)
# Degradation: ~0.003/d → 91% quality after 30 days

# Better: Silo at 15°C
storage_good = SubstrateStorage(
    "silo1",
    storage_type="vertical_silo", # Low degradation
    temperature=288.15            # Cool
)
# Degradation: ~0.0005/d → 98.5% quality after 30 days
```

## Component Summary Table

| Component | Purpose | Key Parameters | Typical Power | Notes |
|-----------|---------|----------------|---------------|-------|
| **Pump** | Material transfer | Q_nom, pressure_head | 2-10 kW | Size for 80-90% max flow |
| **Mixer** | Homogenization | mixing_intensity, on_time | 5-20 kW | Use intermittent (25% on-time) |
| **Storage** | Substrate storage | capacity, storage_type | 0 kW | Monitor quality degradation |
| **Feeder** | Dosing | Q_max, feeder_type | 1-5 kW | Enable dosing noise for realism |

% TODO: add sources for those numbers

## Next Steps

- **Examples**: See [examples/two_stage_plant.md](..examples/two_stage_plant.md) for complete implementation
- **Optimization**: Use parameter sweeps to optimize component sizing
- **[API Reference](../api_reference/components.rst)**: See detailed class documentation for advanced features

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
