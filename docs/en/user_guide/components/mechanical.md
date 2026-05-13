# Mechanical Components

Mechanical plant components for material handling and process control.

## Pump

Pumps for substrate transport, recirculation, and digestate processing in biogas plants.

### Parameters

```python
from pyadm1.components.mechanical import Pump

pump = Pump(
    component_id="pump1",
    pump_type="progressive_cavity",  # or "centrifugal", "piston"
    Q_nom=15.0,                      # Nominal flow rate [m³/h]
    pressure_head=50.0,              # Design pressure head [m]
    efficiency=None,                 # Auto-computed if None
    motor_efficiency=0.90,           # Motor efficiency (0–1)
    fluid_density=1020.0,            # Fluid density [kg/m³]
    speed_control=True,              # Variable-frequency drive
    name="Feed pump"
)
```

### Pump Type Comparison

| Type | Best for | Efficiency | Advantages | Disadvantages |
|------|----------|------------|------------|---------------|
| **Centrifugal** | Low-viscosity liquids | 65–75% | • High flow rates<br>• Robust<br>• Low maintenance | • Not self-priming<br>• Poor with high viscosity<br>• Efficiency drops with solids |
| **Progressive Cavity** | Viscous slurries | 50–70% | • Handles high solids<br>• Self-priming<br>• Gentle transport | • Lower efficiency<br>• Higher maintenance<br>• Speed-dependent pressure |
| **Piston** | High-pressure applications | 70–85% | • High-pressure capability<br>• Precise flow control<br>• Good efficiency | • Higher cost<br>• More complex<br>• Sensitive to solids |

### Sizing Guidelines

**Flow-rate selection:**

| Plant size | Substrate feed [m³/d] | Pump Q_nom [m³/h] | Typical pressure [m] |
|------------|-----------------------|-------------------|----------------------|
| Small | 10–25 | 5–15 | 30–50 |
| Medium | 25–75 | 15–40 | 40–60 |
| Large | 75–200 | 40–100 | 50–80 |

**Pressure-head considerations:**

```python
# Compute required pressure head
H_static = 5.0      # Vertical lift [m]
H_friction = 8.0    # Pipe friction losses [m]
H_process = 2.0     # Process pressure [m]
H_safety = 1.2      # Safety factor

H_required = (H_static + H_friction + H_process) * H_safety
# = 18.0 m

pump = Pump("pump1", Q_nom=15, pressure_head=H_required)
```

### Outputs

```python
{
    'P_consumed': 8.5,           # Power consumption [kW]
    'Q_actual': 10.0,            # Actual flow rate [m³/h]
    'is_running': True,          # Operating state
    'efficiency': 0.68,          # Current efficiency
    'pressure_actual': 48.5,     # Actual pressure [m]
    'speed_fraction': 1.0,       # Speed (0–1)
    'specific_energy': 0.85      # Energy per volume [kWh/m³]
}
```

### Usage Example

```python
# Progressive-cavity pump for substrate transport
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

### Power Consumption

Pumps compute power based on the hydraulic formula:

```text
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

**Typical power consumption:**

| Flow [m³/h] | Head [m] | Pump type | Power [kW] |
|-------------|----------|-----------|------------|
| 10 | 30 | Centrifugal | 2.5 |
| 10 | 50 | Progressive cavity | 4.5 |
| 15 | 50 | Progressive cavity | 6.8 |
| 20 | 60 | Piston | 10.5 |

## Mixer

Mixers and agitators that maintain homogeneity in anaerobic digesters.

### Parameters

```python
from pyadm1.components.mechanical import Mixer

mixer = Mixer(
    component_id="mix1",
    mixer_type="propeller",          # or "paddle", "jet"
    tank_volume=2000.0,              # Tank volume [m³]
    tank_diameter=None,              # Auto-computed if None
    mixing_intensity="medium",       # "low", "medium", "high"
    power_installed=None,            # Auto-computed if None
    intermittent=True,               # Intermittent operation
    on_time_fraction=0.25,           # 25% on-time
    name="Main mixer"
)
```

### Mixer Types

| Type | Flow pattern | Best for | Power factor | Typical speed [rpm] |
|------|--------------|----------|--------------|---------------------|
| **Propeller** | Axial | Large tanks, liquid substrates | 1.0× | 40–100 |
| **Paddle** | Radial | High solids, fibrous material | 1.2× | 20–60 |
| **Jet** | Hydraulic | Recirculation mixing | 1.5× | N/A (pump) |

### Mixing Intensity

| Intensity | Specific power [W/m³] | Mixing time [min] | Application |
|-----------|-----------------------|-------------------|-------------|
| **Low** | 3 | 15–30 | Liquid manure, low solids |
| **Medium** | 5 | 8–15 | Standard operation, energy crops |
| **High** | 8 | 3–8 | High solids, fibrous substrates |

### Sizing Example

```python
# For a 2000 m³ digester with medium intensity
tank_volume = 2000  # m³
specific_power = 5  # W/m³ for medium intensity

P_required = (tank_volume * specific_power) / 1000  # kW
P_required *= 1.2  # safety factor for fibrous material
# = 12 kW

mixer = Mixer(
    "mix1",
    tank_volume=2000,
    mixing_intensity="medium",
    power_installed=15  # round up to standard size
)
```

### Outputs

```python
{
    'P_consumed': 12.5,          # Current power [kW]
    'P_average': 3.1,            # Time-averaged [kW]
    'is_running': True,          # Operating state
    'mixing_quality': 0.85,      # Quality index (0–1)
    'reynolds_number': 15000,    # Flow regime indicator
    'power_number': 0.32,        # Dimensionless power number
    'mixing_time': 8.5,          # Time to homogeneity [min]
    'shear_rate': 45.2,          # Average shear rate [1/s]
    'specific_power': 6.25,      # Power density [kW/m³]
    'tip_speed': 2.8             # Impeller tip speed [m/s]
}
```

### Usage Example

```python
# Medium-intensity propeller mixer
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

### Intermittent Operation

Intermittent mixing reduces energy use:

```python
# Comparison continuous vs. intermittent
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

# Both reach similar mixing quality
```

**Recommended on-time fractions:**

| Substrate type | On-time | Total hours/day | Energy savings |
|----------------|---------|------------------|----------------|
| Liquid manure | 15–20% | 3.6–4.8 h | 80–85% |
| Energy crops | 20–30% | 4.8–7.2 h | 70–80% |
| High solids | 25–35% | 6.0–8.4 h | 65–75% |
| Fibrous material | 30–40% | 7.2–9.6 h | 60–70% |

### Mixing-Power Calculation

Mixers compute power based on the power-number correlation:

```python
# Mechanical power from power-number correlation
# P = Np * ρ * N³ * D⁵

N = operating_speed / 60.0  # Hz (revolutions/s)
D = impeller_diameter       # m

P_mech = power_number * fluid_density * N**3 * D**5 / 1000.0  # kW

# Account for motor efficiency (typically 85–95%)
motor_efficiency = 0.90
P_electrical = P_mech / motor_efficiency

# Limit to installed power
P_actual = min(P_electrical, power_installed)
```

**Reynolds number for mixing:**

```text
Re = ρ * N * D² / μ

where:
- ρ = fluid density [kg/m³]
- N = rotational speed [Hz]
- D = impeller diameter [m]
- μ = viscosity [Pa·s]
```

**Power number (depends on mixer type and Reynolds number):**

- **Propeller:**
  - Laminar (Re < 100): Np = 14.0 * Re^(-0.67)
  - Transition (100 < Re < 10000): Np = 1.2 * Re^(-0.15)
  - Turbulent (Re > 10000): Np = 0.32

- **Paddle:**
  - Laminar (Re < 10): Np = 300.0 / Re
  - Transition (10 < Re < 10000): Np = 8.0 * Re^(-0.25)
  - Turbulent (Re > 10000): Np = 5.0

### Mixing-Time Estimation

Based on the Nienow correlation:

```text
θ_mix = C * (D_T/D)^α * (H/D_T)^β / N

where:
- C, α, β = constants depending on mixer type
- D_T = tank diameter [m]
- D = impeller diameter [m]
- H = tank height [m]
- N = rotational speed [Hz]
```

**Typical constants:**

| Mixer type | C | α | β |
|------------|---|---|---|
| Propeller | 5.3 | 2.0 | 0.5 |
| Paddle | 6.5 | 2.5 | 0.7 |
| Jet | 4.0 | 1.5 | 0.3 |

## Integration Example

### Complete Pump and Mixing Chain

```python
from pyadm1.configurator import BiogasPlant, PlantConfigurator
from pyadm1.components.mechanical import Pump, Mixer
from pyadm1.substrates import Feedstock

# Setup
feedstock = Feedstock(feeding_freq=48)
plant = BiogasPlant("Mechanical System Plant")
config = PlantConfigurator(plant, feedstock)

# 1. Substrate feed pump
feed_pump = Pump(
    "feed_pump",
    pump_type="progressive_cavity",
    Q_nom=15.0,
    pressure_head=50.0,
    speed_control=True
)
plant.add_component(feed_pump)

# 2. Digester
digester, storage = config.add_digester(
    "main_digester",
    V_liq=2000,
    Q_substrates=[15, 10, 0, 0, 0, 0, 0, 0, 0, 0]
)

# 3. Main mixer
main_mixer = Mixer(
    "main_mixer",
    mixer_type="propeller",
    tank_volume=2000,
    mixing_intensity="medium",
    intermittent=True,
    on_time_fraction=0.25
)
plant.add_component(main_mixer)

# 4. Recirculation pump
recirc_pump = Pump(
    "recirc_pump",
    pump_type="centrifugal",
    Q_nom=50.0,  # Higher flow for recirculation
    pressure_head=10.0,  # Lower pressure
    speed_control=True
)
plant.add_component(recirc_pump)

# 5. Digestate pump
digestate_pump = Pump(
    "digestate_pump",
    pump_type="progressive_cavity",
    Q_nom=20.0,
    pressure_head=30.0
)
plant.add_component(digestate_pump)

# Energy system
config.add_chp("chp1", P_el_nom=500)
config.add_heating("heat1", target_temperature=308.15)

# Connections
config.connect("feed_pump", "main_digester", "liquid")
config.auto_connect_digester_to_chp("main_digester", "chp1")
config.auto_connect_chp_to_heating("chp1", "heat1")

# Simulate
plant.initialize()
results = plant.simulate(duration=30, dt=1/24, save_interval=1.0)

# Mechanical energy analysis
def mechanical_energy_analysis(results):
    """Analyze mechanical energy consumption."""
    final = results[-1]
    comp = final['components']

    # Pump energy
    feed_pump_energy = comp['feed_pump']['energy_consumed']
    recirc_pump_energy = comp['recirc_pump']['energy_consumed']
    digestate_pump_energy = comp['digestate_pump']['energy_consumed']
    total_pump_energy = feed_pump_energy + recirc_pump_energy + digestate_pump_energy

    # Mixer energy
    mixer_energy = comp['main_mixer']['energy_consumed']

    # Total mechanical energy
    total_mech_energy = total_pump_energy + mixer_energy

    # CHP production
    chp_energy = comp['chp1']['P_el'] * 30 * 24  # kWh

    # Parasitic load
    parasitic_fraction = total_mech_energy / chp_energy if chp_energy > 0 else 0

    return {
        'feed_pump': feed_pump_energy,
        'recirc_pump': recirc_pump_energy,
        'digestate_pump': digestate_pump_energy,
        'total_pump': total_pump_energy,
        'mixer': mixer_energy,
        'total_mechanical': total_mech_energy,
        'chp_production': chp_energy,
        'parasitic_fraction': parasitic_fraction,
        'net_energy': chp_energy - total_mech_energy
    }

analysis = mechanical_energy_analysis(results)
print("\nMechanical energy analysis:")
print(f"Feed pump: {analysis['feed_pump']:.0f} kWh")
print(f"Recirculation pump: {analysis['recirc_pump']:.0f} kWh")
print(f"Digestate pump: {analysis['digestate_pump']:.0f} kWh")
print(f"Total pumps: {analysis['total_pump']:.0f} kWh")
print(f"Mixer: {analysis['mixer']:.0f} kWh")
print(f"Total mechanical: {analysis['total_mechanical']:.0f} kWh")
print(f"Parasitic load: {analysis['parasitic_fraction']:.1%}")
print(f"Net energy production: {analysis['net_energy']:.0f} kWh")
```

## Optimization Strategies

### 1. Pump Optimization

```python
def optimize_pump_sizing(Q_required, H_required, pump_type="progressive_cavity"):
    """Optimize pump sizing for efficiency."""

    # Size for 80–90% of nominal load (highest efficiency)
    Q_nom = Q_required / 0.85

    # Add safety margin for pressure head
    H_nom = H_required * 1.2

    pump = Pump(
        "optimized_pump",
        pump_type=pump_type,
        Q_nom=Q_nom,
        pressure_head=H_nom,
        speed_control=True  # VFD for part-load optimization
    )

    return pump

# Example: 12 m³/h required at 40 m head
optimized = optimize_pump_sizing(12.0, 40.0)
print(f"Optimized pump size: {optimized.Q_nom:.1f} m³/h")
```

### 2. Mixing Optimization

```python
def optimize_mixing_strategy(substrate_type, tank_volume):
    """Pick the optimal mixing strategy based on substrate."""

    strategies = {
        'liquid_manure': {
            'intensity': 'low',
            'on_time_fraction': 0.20,
            'mixer_type': 'propeller'
        },
        'energy_crops': {
            'intensity': 'medium',
            'on_time_fraction': 0.25,
            'mixer_type': 'propeller'
        },
        'high_solids': {
            'intensity': 'medium',
            'on_time_fraction': 0.30,
            'mixer_type': 'paddle'
        },
        'fibrous': {
            'intensity': 'high',
            'on_time_fraction': 0.35,
            'mixer_type': 'paddle'
        }
    }

    strategy = strategies.get(substrate_type, strategies['energy_crops'])

    mixer = Mixer(
        "optimized_mixer",
        mixer_type=strategy['mixer_type'],
        tank_volume=tank_volume,
        mixing_intensity=strategy['intensity'],
        intermittent=True,
        on_time_fraction=strategy['on_time_fraction']
    )

    return mixer, strategy

# Example: fibrous substrates
mixer, strategy = optimize_mixing_strategy('fibrous', 2000)
print(f"Optimized mixing strategy:")
print(f"- Type: {strategy['mixer_type']}")
print(f"- Intensity: {strategy['intensity']}")
print(f"- On-time: {strategy['on_time_fraction']:.0%}")
```

### 3. Energy Minimization

```python
def minimize_mechanical_energy(plant_config):
    """Strategies to minimize mechanical energy use."""

    strategies = []

    # 1. Use intermittent mixing
    strategies.append({
        'name': 'Intermittent mixing',
        'saving': 0.70,  # 70% savings
        'implementation': 'on_time_fraction=0.25'
    })

    # 2. VFDs for pumps
    strategies.append({
        'name': 'VFD for part-load operation',
        'saving': 0.30,  # 30% savings at part load
        'implementation': 'speed_control=True'
    })

    # 3. Proper sizing
    strategies.append({
        'name': 'Optimal sizing',
        'saving': 0.15,  # 15% via efficiency optimization
        'implementation': 'Q_nom = Q_required / 0.85'
    })

    # 4. Lower mixing intensity where possible
    strategies.append({
        'name': 'Adjusted mixing intensity',
        'saving': 0.40,  # 40% via lower intensity
        'implementation': 'mixing_intensity="low" for liquid substrates'
    })

    total_potential = sum(s['saving'] for s in strategies)

    print("Energy-minimization strategies:")
    for s in strategies:
        print(f"- {s['name']}: {s['saving']:.0%} savings")
        print(f"  Implementation: {s['implementation']}")

    return strategies

strategies = minimize_mechanical_energy({})
```

## Troubleshooting

### Problem: Pump Delivers Insufficient Flow

**Diagnosis:**

```python
pump_result = pump.step(0, 1/24, {'Q_setpoint': 15, 'enable_pump': True})

if pump_result['Q_actual'] < 0.8 * pump_result.get('Q_setpoint', 15):
    print("Low pump flow - check:")
    print(f"- Current efficiency: {pump_result['efficiency']:.1%}")
    print(f"- Pressure head: {pump_result['pressure_actual']:.1f} m")
    print(f"- Is the pump correctly sized for the application?")

    # Check for overload
    if pump.speed_fraction > 1.0:
        print("- WARNING: pump overloaded!")
```

**Solutions:**
- Increase pump size if consistently overloaded
- Reduce friction losses in piping
- Check for blockages or wear

### Problem: Mixer Consumes Too Much Energy

**Diagnosis:**

```python
mixer_result = mixer.step(0, 1/24, {})

specific_power = mixer_result['P_consumed'] / mixer.tank_volume  # kW/m³

if specific_power > 6.0:  # upper bound for medium intensity
    print(f"High specific power: {specific_power:.1f} W/m³")
    print("Optimization options:")

    if not mixer.intermittent:
        print("- Enable intermittent operation (70% savings)")

    if mixer.mixing_intensity == "high":
        print("- Reduce to medium intensity if possible")
```

**Solutions:**

```python
# Implement intermittent operation
mixer_optimized = Mixer(
    "mix1",
    tank_volume=mixer.tank_volume,
    mixing_intensity="medium",
    intermittent=True,
    on_time_fraction=0.25  # 75% energy savings
)
```

### Problem: Poor Mixing Quality

**Diagnosis:**

```python
if mixer_result['mixing_quality'] < 0.7:
    print(f"Low mixing quality: {mixer_result['mixing_quality']:.2f}")
    print(f"Mixing time: {mixer_result['mixing_time']:.1f} min")
    print(f"Reynolds number: {mixer_result['reynolds_number']:.0f}")

    if mixer_result['reynolds_number'] < 1000:
        print("- Laminar flow - increase speed or impeller size")

    if mixer_result['mixing_time'] > 30:
        print("- Long mixing time - increase intensity or on-time")
```

**Solutions:**
- Increase mixing intensity for difficult substrates
- Extend on-time for intermittent operation
- Consider larger impeller or higher speed

## Best Practices

1. **Size pumps for optimal efficiency**
   - Operate at 80–90% of nominal load
   - Use VFDs for variable load demands

2. **Implement intermittent mixing**
   - 25% on-time for most applications
   - Adjust to substrate type

3. **Regular maintenance**
   - Monitor pump efficiency over time
   - Check mixer wear

4. **Optimize system design**
   - Minimize piping losses
   - Correct pump placement

5. **Monitor energy consumption**
   - Track parasitic load
   - Target: <10% of CHP production

## Next Steps

- [Biological Components](biological.md): Digester and process control
- [Energy Components](energy.md): CHP and heating systems
- [Feeding Components](feeding.md): Storage and dosing
- [API Reference](../../api/mechanical.md): Detailed class documentation
