# Feeding Components

Components for substrate handling, storage, and dosing in biogas plants.

## SubstrateStorage

Storage tank for various substrate types with inventory tracking and quality management.

### Parameters

```python
from pyadm1.components.feeding import SubstrateStorage

storage = SubstrateStorage(
    component_id="silo1",
    storage_type="vertical_silo",    # See table below
    substrate_type="corn_silage",    # See table below
    capacity=1000.0,                 # Max. capacity [t or m³]
    initial_level=800.0,             # Initial inventory
    degradation_rate=None,           # Auto-computed
    temperature=288.15,              # Storage temperature [K] (15 °C)
    name="Corn-silage storage"
)
```

### Storage Types

| Type | Degradation [1/d] | Best for | Typical size | Investment |
|------|-------------------|----------|--------------|------------|
| **Vertical Silo** | 0.0005 | Corn/grass silage | 500–2000 t | High |
| **Horizontal Silo** | 0.0008 | Large operations | 1000–3000 t | Medium |
| **Bunker Silo** | 0.001 | Drive-over access | 1000–5000 t | Medium |
| **Clamp** | 0.0025 | Seasonal storage | 500–2000 t | Low |
| **Above-ground Tank** | 0.0002 | Liquid manure | 500–3000 m³ | High |
| **Below-ground Tank** | 0.0001 | Liquid storage | 1000–5000 m³ | Very high |

### Substrate Types

| Substrate | Density [kg/m³] | DM [%] | VS [% of DM] | Typical storage |
|-----------|-----------------|--------|--------------|-----------------|
| Corn silage | 650 | 35 | 95 | Silo |
| Grass silage | 700 | 30 | 92 | Silo |
| Whole-plant silage | 680 | 32 | 94 | Silo/bunker |
| Liquid manure | 1020 | 8 | 80 | Tank |
| Solid manure | 850 | 25 | 75 | Clamp |
| Biowaste | 1000 | 20 | 90 | Tank |

### Quality Degradation

Stored quality degrades over time:

```python
# Quality factor at time t:
quality(t) = quality(0) × exp(-k × t)

# where:
# k = degradation_rate [1/d]
# t = storage_time [days]

# Example: corn silage in a vertical silo
# After 30 days: quality = 1.0 × exp(-0.0005 × 30) = 0.985 (98.5%)
# After 90 days: quality = 1.0 × exp(-0.0005 × 90) = 0.956 (95.6%)
```

**Temperature effect:**

Temperature influences degradation (Q10 = 2):

```python
# Degradation increases with temperature
T_ref = 288.15  # 15 °C reference
k_ref = 0.0005  # baseline rate

# At 20 °C (293.15 K):
k_20C = k_ref × 2^((293.15-288.15)/10) = 0.0007

# At 10 °C (283.15 K):
k_10C = k_ref × 2^((283.15-288.15)/10) = 0.0004
```

### Outputs

```python
{
    'current_level': 750.0,      # Inventory [t or m³]
    'utilization': 0.75,         # Fill level (0–1)
    'quality_factor': 0.95,      # Quality (0–1)
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

### Usage Example

```python
# Corn-silage storage
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

print(f"Fill level: {result['current_level']:.1f} t")
print(f"Quality: {result['quality_factor']:.3f}")
print(f"Available: {result['available_mass']:.1f} t")
print(f"Losses: {result['losses_this_step']:.2f} t")
```

### Storage Management Strategy

```python
def should_refill(storage_result, safety_days=7):
    """Determine whether a refill is needed."""
    level = storage_result['current_level']
    daily_usage = 15  # t/d
    days_remaining = level / daily_usage

    return days_remaining < safety_days

def check_quality(storage_result, min_quality=0.90):
    """Alert when quality falls too low."""
    quality = storage_result['quality_factor']
    if quality < min_quality:
        print(f"Warning: quality at {quality:.1%}")
        return False
    return True
```

## Feeder

Automated dosing systems for precise substrate feeding.

### Parameters

```python
from pyadm1.components.feeding import Feeder

feeder = Feeder(
    component_id="feed1",
    feeder_type="screw",             # Auto-selected if None
    Q_max=20.0,                      # Max. flow [m³/d or t/d]
    substrate_type="solid",          # "solid", "slurry", "liquid", "fibrous"
    dosing_accuracy=None,            # Auto-computed
    power_installed=None,            # Auto-computed
    enable_dosing_noise=True,        # Realistic variance
    name="Screw feeder"
)
```

### Feeder Types

| Type | Accuracy [±%] | Best for | Speed control | Power [kW/m³/h] |
|------|---------------|----------|---------------|-----------------|
| **Screw** | 5 | Solid substrates | Good | 0.8 |
| **Twin Screw** | 3 | Better control | Excellent | 1.0 |
| **Progressive Cavity** | 2 | Viscous slurries | Good | 1.2 |
| **Piston** | 1 | Precise dosing | Excellent | 1.5 |
| **Centrifugal Pump** | 8 | Low viscosity | Mediocre | 0.5 |
| **Mixer Wagon** | 10 | Batch feeding | N/A | 2.0 |

### Dosing Accuracy

Real feeders have variance around the setpoint:

```python
# With dosing_noise enabled:
# Actual flow = setpoint + noise
# where noise ~ Normal(0, accuracy × setpoint)

# Example: screw feeder (5% accuracy) at 15 m³/d
# Typical range: 14.25 – 15.75 m³/d
# Occasionally: 13.5 – 16.5 m³/d (±2σ)

feeder = Feeder("feed1", Q_max=20, dosing_accuracy=0.05)
```

### Power Requirements

Power depends on substrate type:

| Substrate | Base power [kW/m³/h] | Modifier | Total |
|-----------|----------------------|----------|-------|
| Liquid | 0.5 | ×0.7 | 0.35 |
| Slurry | 0.8 | ×1.0 | 0.80 |
| Solid | 0.8 | ×1.4 | 1.12 |
| Fibrous | 0.8 | ×1.8 | 1.44 |

```python
# Example: 15 m³/h screw feeder for fibrous substrate
Q_nom_h = 15 / 24  # = 0.625 m³/h
P = 0.8 * 0.625 * 1.8 * 1.3  # [base × Q × modifier × safety]
  = 1.17 kW
```

### Outputs

```python
{
    'Q_actual': 14.8,            # Actual flow [m³/d]
    'is_running': True,
    'load_factor': 0.74,         # Load (0–1)
    'P_consumed': 2.5,           # Power [kW]
    'blockage_detected': False,  # Alarm
    'dosing_error': 1.3,         # Error [%]
    'speed_fraction': 0.95,      # Speed (0–1)
    'dosing_accuracy': 0.05,     # Accuracy
    'total_mass_fed': 1250.0     # Cumulative [t]
}
```

### Usage Example

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

### Blockage Detection

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

## Complete Feeding Chain

### Integrated System

```python
from pyadm1.configurator import BiogasPlant, PlantConfigurator
from pyadm1.components.feeding import SubstrateStorage, Feeder
from pyadm1.components.mechanical import Pump
from pyadm1.substrates import Feedstock

# Setup
feedstock = Feedstock(feeding_freq=48)
plant = BiogasPlant("Complete Feeding System")
config = PlantConfigurator(plant, feedstock)

# 1. Multiple substrate storage tanks
corn_storage = SubstrateStorage(
    "corn_silo",
    storage_type="vertical_silo",
    substrate_type="corn_silage",
    capacity=1000,
    initial_level=800
)
plant.add_component(corn_storage)

manure_storage = SubstrateStorage(
    "manure_tank",
    storage_type="above_ground_tank",
    substrate_type="manure_liquid",
    capacity=500,
    initial_level=400
)
plant.add_component(manure_storage)

# 2. Feeders for each substrate
corn_feeder = Feeder(
    "corn_feeder",
    feeder_type="screw",
    Q_max=15.0,
    substrate_type="solid"
)
plant.add_component(corn_feeder)

manure_feeder = Feeder(
    "manure_feeder",
    feeder_type="progressive_cavity",
    Q_max=10.0,
    substrate_type="slurry"
)
plant.add_component(manure_feeder)

# 3. Mixing pump
mix_pump = Pump(
    "mix_pump",
    pump_type="progressive_cavity",
    Q_nom=25.0,
    pressure_head=50.0
)
plant.add_component(mix_pump)

# 4. Digester
digester, storage = config.add_digester(
    "main_digester",
    V_liq=2000,
    Q_substrates=[15, 10, 0, 0, 0, 0, 0, 0, 0, 0]
)

# 5. Energy system
config.add_chp("chp1", P_el_nom=500)
config.add_heating("heat1", target_temperature=308.15)

# Connections
config.connect("corn_silo", "corn_feeder", "default")
config.connect("manure_tank", "manure_feeder", "default")
config.connect("corn_feeder", "mix_pump", "default")
config.connect("manure_feeder", "mix_pump", "default")
config.connect("mix_pump", "main_digester", "liquid")
config.auto_connect_digester_to_chp("main_digester", "chp1")
config.auto_connect_chp_to_heating("chp1", "heat1")

# Simulate
plant.initialize()
results = plant.simulate(duration=30, dt=1/24, save_interval=1.0)

# Feeding analysis
def feeding_system_analysis(results):
    """Analyze feeding-system performance."""
    final = results[-1]
    comp = final['components']

    # Inventory
    corn_level = comp['corn_silo']['current_level']
    manure_level = comp['manure_tank']['current_level']

    # Quality
    corn_quality = comp['corn_silo']['quality_factor']
    manure_quality = comp['manure_tank']['quality_factor']

    # Throughput
    corn_fed = comp['corn_feeder']['total_mass_fed']
    manure_fed = comp['manure_feeder']['total_mass_fed']

    # Energy consumption
    corn_feeder_energy = comp['corn_feeder'].get('energy_consumed', 0)
    manure_feeder_energy = comp['manure_feeder'].get('energy_consumed', 0)
    pump_energy = comp['mix_pump']['energy_consumed']

    total_feed_energy = corn_feeder_energy + manure_feeder_energy + pump_energy

    return {
        'corn_remaining': corn_level,
        'manure_remaining': manure_level,
        'corn_quality': corn_quality,
        'manure_quality': manure_quality,
        'total_corn_fed': corn_fed,
        'total_manure_fed': manure_fed,
        'feeding_energy': total_feed_energy
    }

analysis = feeding_system_analysis(results)
print("\nFeeding-system analysis:")
print(f"Corn remaining: {analysis['corn_remaining']:.0f} t (quality: {analysis['corn_quality']:.1%})")
print(f"Manure remaining: {analysis['manure_remaining']:.0f} m³ (quality: {analysis['manure_quality']:.1%})")
print(f"Total corn fed: {analysis['total_corn_fed']:.0f} t")
print(f"Total manure fed: {analysis['total_manure_fed']:.0f} m³")
print(f"Feeding energy: {analysis['feeding_energy']:.0f} kWh")
```

## Optimization Strategies

### 1. Substrate-Mix Optimization

```python
def optimize_substrate_mix(available_substrates, target_vs_loading):
    """Optimize substrate mix for target VS loading."""

    # Example substrates
    substrates = {
        'corn_silage': {'vs': 0.33, 'cost': 30},  # 33% VS, 30 €/t
        'manure': {'vs': 0.06, 'cost': 0},        # 6% VS, free
        'biowaste': {'vs': 0.17, 'cost': -20}     # 17% VS, gate fee
    }

    # Simple mix calculation (can be extended for optimization)
    corn_fraction = 0.60
    manure_fraction = 0.30
    biowaste_fraction = 0.10

    mix_vs = (corn_fraction * substrates['corn_silage']['vs'] +
              manure_fraction * substrates['manure']['vs'] +
              biowaste_fraction * substrates['biowaste']['vs'])

    mix_cost = (corn_fraction * substrates['corn_silage']['cost'] +
                manure_fraction * substrates['manure']['cost'] +
                biowaste_fraction * substrates['biowaste']['cost'])

    print(f"Optimized mix:")
    print(f"- Corn: {corn_fraction:.0%}")
    print(f"- Manure: {manure_fraction:.0%}")
    print(f"- Biowaste: {biowaste_fraction:.0%}")
    print(f"Resulting VS: {mix_vs:.1%}")
    print(f"Cost: {mix_cost:.1f} €/t")

    return {
        'corn': corn_fraction,
        'manure': manure_fraction,
        'biowaste': biowaste_fraction,
        'total_vs': mix_vs,
        'cost': mix_cost
    }

optimized_mix = optimize_substrate_mix({}, 0.20)
```

### 2. Inventory Management

```python
def manage_inventory(storage_results, forecast_days=30):
    """Manage inventory with forecasting."""

    for name, result in storage_results.items():
        level = result['current_level']
        capacity = result.get('capacity', 1000)
        daily_usage = 15  # example

        days_remaining = level / daily_usage

        print(f"\n{name}:")
        print(f"- Current inventory: {level:.0f} ({level/capacity:.1%} of capacity)")
        print(f"- Days remaining: {days_remaining:.1f}")

        if days_remaining < 7:
            print("- ACTION: refill urgently!")
            refill_amount = capacity * 0.8 - level
            print(f"- Recommended refill: {refill_amount:.0f}")
        elif days_remaining < 14:
            print("- WARNING: plan a refill")

        # Quality check
        quality = result['quality_factor']
        if quality < 0.90:
            print(f"- QUALITY: low ({quality:.1%}) - consider use order")

# Example usage
storage_results = {
    'corn_silo': final['components']['corn_silo'],
    'manure_tank': final['components']['manure_tank']
}
manage_inventory(storage_results)
```

### 3. Dosing-Accuracy Optimization

```python
def optimize_dosing_accuracy(substrate_value, process_sensitivity):
    """Pick a feeder type based on requirements."""

    # High-value substrates or sensitive processes need high accuracy
    if substrate_value > 40 or process_sensitivity == "high":
        recommended_type = "piston"
        accuracy = 0.01
    elif substrate_value > 20 or process_sensitivity == "medium":
        recommended_type = "progressive_cavity"
        accuracy = 0.02
    else:
        recommended_type = "screw"
        accuracy = 0.05

    print(f"Recommended feeder type: {recommended_type}")
    print(f"Expected accuracy: ±{accuracy:.0%}")

    return recommended_type, accuracy

# Example: high-value energy crops
feeder_type, accuracy = optimize_dosing_accuracy(substrate_value=35,
                                                  process_sensitivity="medium")
```

## Troubleshooting

### Problem: Rapid Quality Loss

**Diagnosis:**

```python
storage_result = storage.step(t, dt, inputs)

if storage_result['quality_factor'] < 0.95 and storage_result['storage_time'] < 30:
    print("Rapid quality loss detected:")
    print(f"- Quality: {storage_result['quality_factor']:.1%}")
    print(f"- Storage time: {storage_result['storage_time']:.1f} days")
    print(f"- Degradation rate: {storage_result['degradation_rate']:.4f} 1/d")
    print(f"- Temperature: {storage.temperature:.1f} K")
```

**Solutions:**

```python
# Option 1: Improve storage type
storage_improved = SubstrateStorage(
    "silo1",
    storage_type="vertical_silo",  # from "clamp"
    substrate_type="corn_silage",
    capacity=1000
)

# Option 2: Lower temperature
storage.temperature = 283.15  # 10 °C instead of 15 °C

# Option 3: Faster usage (reduce storage time)
increase_daily_usage = True
```

### Problem: Feeder Blockages

**Diagnosis:**

```python
if feeder.state['n_blockages'] > 5:
    print(f"Frequent blockages detected: {feeder.state['n_blockages']}")
    print("Possible causes:")
    print("- Fibrous substrate unsuitable for feeder type")
    print("- Foreign objects in substrate")
    print("- Wear or maintenance required")
```

**Solutions:**  
- Switch to a more robust feeder type (twin screw)  
- Improve substrate preparation  
- Implement a maintenance schedule  

### Problem: Inconsistent Dosing

**Diagnosis:**

```python
dosing_errors = [r['components']['feed1']['dosing_error']
                for r in results]
avg_error = sum(dosing_errors) / len(dosing_errors)

if avg_error > 10:
    print(f"High average dosing error: {avg_error:.1f}%")
    print("Recommendations:")
    print("- Consider a more precise feeder type")
    print("- Check calibration")
    print("- Disable dosing_noise for idealized simulation")
```

## Best Practices

1. **Implement quality monitoring**  
   - Track quality_factor over time  
   - Alert below 90% quality  
   - Plan FIFO rotation  

2. **Maintain inventory safety**  
   - 7–14 days minimum safety stock  
   - Schedule refills in advance  
   - Account for seasonal availability  

3. **Optimize energy use**  
   - Use VFDs for variable dosing  
   - Minimize idle times  
   - Right-size feeders  

4. **Optimize substrate mix**  
   - Balanced VS loading  
   - Cost optimization  
   - Nutrient balancing  

5. **Schedule maintenance**  
   - Monitor wear parts  
   - Preventive maintenance for feeders  
   - Storage-tank inspections  

## Next Steps

- [Biological Components](biological.md): Digester and process control  
- [Energy Components](energy.md): CHP and heating systems  
- [Mechanical Components](mechanical.md): Pumps and mixers  
- [API Reference](../../api/feeding.md): Detailed class documentation  
