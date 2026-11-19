# Energy Components

Components for energy generation, conversion, and storage in biogas plants.

## CHP (Combined Heat and Power)

Combined heat and power unit that converts biogas into electricity and heat.

### Parameters

```python
configurator.add_chp(
    chp_id="chp_main",
    P_el_nom=500.0,        # Nominal electrical power [kW]
    eta_el=0.40,           # Electrical efficiency (40%)
    eta_th=0.45,           # Thermal efficiency (45%)
    name="Main CHP"
)
```

### Typical CHP Specifications

| Type | Size [kW_el] | η_el | η_th | Gas demand [m³/d @ 60% CH4] |
|------|--------------|------|------|-----------------------------|
| Small | 100–250 | 0.38 | 0.48 | 600–1500 |
| Medium | 250–750 | 0.40 | 0.45 | 1500–4500 |
| Large | 750–2000 | 0.42 | 0.43 | 4500–12000 |

### Technology Options

```python
# Gas engine (most common)
chp_engine = configurator.add_chp(
    "chp1", P_el_nom=500, eta_el=0.40, eta_th=0.45
)

# Micro turbine (100–500 kW)
chp_turbine = configurator.add_chp(
    "chp2", P_el_nom=250, eta_el=0.30, eta_th=0.55
)

# High-efficiency (>1 MW)
chp_large = configurator.add_chp(
    "chp3", P_el_nom=1500, eta_el=0.42, eta_th=0.43
)
```

### Outputs

```python
{
    'P_el': 450.0,              # Electrical power [kW]
    'P_th': 506.3,              # Thermal power [kW]
    'Q_gas_consumed': 2700.0,   # Gas consumption [m³/d]
}
```

### Advanced CHP Control

```python
# Variable load control
inputs = {
    'Q_ch4': 800.0,           # Available methane [m³/d]
    'load_setpoint': 0.75     # Run at 75% capacity
}
result = chp.step(t, dt, inputs)
```

### Power Calculation

CHP units compute power based on available methane:

```python
# Methane energy content: ~10 kWh/m³
E_ch4 = 10.0  # kWh/m³

# Available power from methane
P_available = Q_ch4_available / 24.0 * E_ch4  # kW

# Electrical power
P_el = min(P_el_nom, P_available * eta_el)

# Thermal power
P_th = P_el * eta_th / eta_el

# Gas consumption
Q_ch4_consumed = P_el / eta_el * 24.0 / E_ch4  # m³/d
Q_gas_consumed = Q_ch4_consumed / CH4_fraction # m³/d biogas
```

#### Methane content of the gas

The CHP burns the gas with its actual methane content; lean gas means more
biogas for the same power. The content is taken from, in this order:

1. `Q_ch4_supplied_m3_per_day` / `Q_gas_supplied_m3_per_day` from the gas storage,  
2. a `CH4_fraction` input,  
3. `Q_ch4` / `Q_gas` of a directly connected digester,  
4. `ch4_content_default` (constructor argument, default 0.60) when the source  
   carries no composition at all.

`BiogasPlant.step` sizes the gas demand on each storage from the methane
fraction that storage currently holds. The fraction used is reported as
`CH4_fraction` in the CHP outputs.

The heating value of methane is the constructor argument `lhv_ch4`
(default 10.0 kWh/m³) of both `CHP` and `Boiler`.

The boiler and the flare burn the same gas:

- `Boiler` uses `lhv_ch4 * CH4_fraction` as the biogas heating value when its  
  gas supply reports a `CH4_fraction`; the flat `lhv_biogas` (6.0 kWh/m³ at
  60 % CH4) applies only without a composition. The value used is reported as
  `lhv_biogas_actual`.
- `Flare` takes `CH4_fraction` from its input, falling back to its  
  `ch4_fraction_default` (0.6), and reports the unburnt methane as
  `CH4_slip_m3` next to `CH4_destroyed_m3`.

## Heating System

Maintains digester temperature using CHP waste heat and auxiliary heating.

### Parameters

```python
configurator.add_heating(
    heating_id="heating_main",
    target_temperature=308.15,      # Target temperature [K]
    heat_loss_coefficient=None,     # UA [kW/K]; None = from the digester, else 0.5
    name="Main Digester Heating",
    digester_id="main_digester",    # Digester this system heats (optional)
)
```

`digester_id` connects the digester to the heating system. Only then does the
heating receive the digester temperature, the substrate feed (needed for the
process-heat term) and the digester's `heat_loss_coefficient`. Without the
connection only the flat heat-loss term is computed. The heat loss coefficient
is resolved in this order:

1. `heat_loss_coefficient` given to the heating system,  
2. the value the connected digester derives from its k-values,  
3. 0.5 kW/K.  

### Heat-Loss Coefficients

| Insulation | k [kW/K] | Description |
|------------|----------|-------------|
| Excellent | 0.3–0.4 | Modern, well insulated |
| Good | 0.4–0.6 | Standard insulation |
| Poor | 0.6–1.0 | Old or minimal insulation |

### Outputs

```python
{
    'Q_heat_supplied': 125.5,    # Total heat supplied [kW]
    'P_th_used': 110.0,          # CHP heat used [kW]
    'P_aux_heat': 15.5           # Auxiliary heating required [kW]
}
```

### Heat-Demand Calculation

```python
# Heat demand = heat loss + process heat
Q_loss = UA * (T_digester - T_ambient)                          # [kW]
Q_process = Σ_i Q_i * ρ_i * c_p,i * (T_target - T_substrate,i)  # [kW]
Q_total = Q_loss + Q_process                                    # [kW]

# Use CHP heat first, then auxiliary
if Q_total <= P_th_available:
    P_aux = 0
else:
    P_aux = Q_total - P_th_available
```

### Ambient Temperature

`T_ambient` is a plant-level setting. `BiogasPlant` hands it to every heating
system on each step. It enters the heat-loss term only.

```python
plant = BiogasPlant("My Plant", T_ambient=278.15)  # 5 °C; default 288.15 (15 °C)

plant.T_ambient = 268.15   # may be changed between steps, e.g. from weather data
```

A `HeatingSystem` stepped on its own, without a plant, still falls back to
15 °C unless `T_ambient` is passed in its step inputs.

### Ground Temperature

The floor slab loses heat to the soil, not to the outside air. `T_ground` is
the second plant-level temperature, with a default of 283.15 K (10 °C) —
roughly constant over the year below the slab.

```python
plant = BiogasPlant("My Plant", T_ambient=268.15, T_ground=281.15)
```

It applies only when the heat loss coefficient comes from the digester's
k-values, because only then is the floor share
(`digester.heat_loss_coefficient_ground`) known:

```python
Q_loss = (UA - UA_ground) * (T_digester - T_ambient) + UA_ground * (T_digester - T_ground)
```

A flat `heat_loss_coefficient` (given to the heating system, or the 0.5 kW/K
default) acts against `T_ambient` as a whole.

### Substrate Temperature

The process-heat term is independent of the ambient temperature: manure from
the barn or silage from the clamp does not follow the outside air. Each
substrate is heated from its own storage temperature, the `Temp` field (°C) of
its substrate definition; substrates without one start at 15 °C. Pass
`T_substrate` [K] in the step inputs to override the inlet temperature for all
substrates.

### Example: Multi-Stage Heating

```python
from pyadm1.configurator import BiogasPlant, PlantConfigurator
from pyadm1.substrates import Feedstock

feedstock = Feedstock(feeding_freq=48)
plant = BiogasPlant("Two-Stage Plant")
config = PlantConfigurator(plant, feedstock)

# Two digesters at different temperatures
config.add_digester("hydro", V_liq=500, T_ad=328.15)  # 55 °C
config.add_digester("main", V_liq=2000, T_ad=308.15)  # 35 °C

# One CHP
config.add_chp("chp1", P_el_nom=500)

# Separate heating for each digester
config.add_heating("heat_hydro", target_temperature=328.15, heat_loss_coefficient=0.3)
config.add_heating("heat_main", target_temperature=308.15, heat_loss_coefficient=0.5)

# Connect CHP to both heaters
config.auto_connect_chp_to_heating("chp1", "heat_hydro")
config.auto_connect_chp_to_heating("chp1", "heat_main")

# Simulate
plant.initialize()
results = plant.simulate(duration=30, dt=1/24, save_interval=1.0)

# Analyze heat distribution
final = results[-1]
print(f"Hydrolysis heater: {final['components']['heat_hydro']['Q_heat_supplied']:.1f} kW")
print(f"Main heater: {final['components']['heat_main']['Q_heat_supplied']:.1f} kW")
print(f"Auxiliary heat hydrolysis: {final['components']['heat_hydro']['P_aux_heat']:.1f} kW")
print(f"Auxiliary heat main: {final['components']['heat_main']['P_aux_heat']:.1f} kW")
```

## Gas Storage

Biogas storage with pressure management (created automatically per digester).

### Types

```python
from pyadm1.components.energy import GasStorage

# Low-pressure membrane storage (most common)
storage_membrane = GasStorage(
    component_id="storage1",
    storage_type="membrane",
    capacity_m3=1000.0,      # Capacity at STP [m³]
    p_min_bar=0.95,          # Min. pressure [bar]
    p_max_bar=1.05,          # Max. pressure [bar]
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

### Outputs

```python
{
    'stored_volume_m3': 450.0,       # Current stored volume [m³ STP]
    'pressure_bar': 1.01,            # Current pressure [bar]
    'utilization': 0.45,             # Fill level (0–1)
    'vented_volume_m3': 0.0,         # Gas flared this step [m³]
    'Q_gas_supplied_m3_per_day': 2700.0, # Gas supplied [m³/d]
    'Q_ch4_supplied_m3_per_day': 1458.0, # Methane within that supply [m³/d]
    'CH4_fraction': 0.54,            # CH4 fraction of the stored gas (0–1)
    'CO2_fraction': 0.44             # CO2 fraction of the stored gas (0–1)
}
```

### Gas Composition

The storage keeps the CH4 and CO2 volume fraction of its inventory; the
remainder is H2, water vapour and trace gases. The gas is treated as ideally
mixed: inflow is blended in first, and everything leaving in the same step
(supply, overflow, safety vent) carries the blended composition.

```python
storage = GasStorage("storage", capacity_m3=1000.0,
                     initial_ch4_fraction=0.60, initial_co2_fraction=0.40)

storage.step(t, dt, {
    "Q_gas_in_m3_per_day": 2500.0,
    "Q_ch4_in_m3_per_day": 1350.0,   # optional; without it the inflow
    "Q_co2_in_m3_per_day": 1100.0,   # is taken to match the stored gas
})
```

Inside a plant the digester's `Q_ch4` and `Q_co2` are routed in automatically,
and both the CHP and the biogas upgrading unit receive the actual methane
content instead of a fixed assumption.

### Pressure Model

The storage estimates pressure based on stored volume:

**Low pressure (membrane/dome):**

```text
p = p_atm + frac * (p_max - p_atm)
```

**High pressure (compressed):**

```text
p = p_min + frac^α * (p_max - p_min)  # α > 1 for non-linear rise
```

### Safety Venting

Inflow and demand of one step act on the storage together. Only what is left
above the capacity after the consumers were served is vented:

```python
stored = stored + inflow - supplied
vented = max(0.0, stored - capacity)
```

The vented volume therefore does not depend on the step length: a storage of
300 m³ fed with 2400 m³/d and drawn at 2000 m³/d vents the same amount with
hourly and with daily steps.

### Gas Balance in a Plant

`BiogasPlant.step()` steps every gas storage once per plant step:

- **Production** of all digesters connected by a gas connection flows in.  
- **Demand** of all CHPs and upgrading units on the storage is added up. If the storage cannot cover it, each consumer gets a share in proportion to its demand.  
- **Vented gas** goes to the flare of the storage. A flare connected to the storage itself takes precedence; otherwise the flare behind the first consumer is used (`add_chp()` and `add_bgaa()` create that one). Without a flare the gas is released unburnt.  
- A **boiler** connected to a storage burns gas left after the main consumers. It covers the auxiliary heat (`P_aux_heat`) of the heating systems connected to it and reports what it could not cover as `P_th_unmet`.  

Over any period: produced = burnt + vented + change of the stored volume.

## Flare

Safety system for combusting excess biogas.

```python
from pyadm1.components.energy import Flare

flare = Flare(
    component_id="flare1",
    destruction_efficiency=0.98,  # 98% CH4 destroyed
    name="Emergency Flare"
)
```

### Outputs

```python
{
    'vented_volume_m3': 0.0,         # Volume combusted this step [m³]
    'cumulative_vented_m3': 125.5,   # Cumulative combusted volume [m³]
    'CH4_destroyed_m3': 0.0,         # CH4 destroyed this step [m³]
    'CH4_slip_m3': 0.0               # CH4 passing the flame unburnt this step [m³]
}
```

### Flare Control

Inside a plant the flare receives the gas its storage vents and the gas an
upgrading unit cannot take. It burns whenever:
- The gas storage is full and production exceeds demand  
- The CHP consumes less gas than is being produced  

Standalone, the flare is driven through its inputs:

```python
# Gas from storage to flare
flare_inputs = {
    'Q_gas_in_m3_per_day': vented_gas,
    'CH4_fraction': 0.6  # 60% methane in biogas
}

result = flare.step(t, dt, flare_inputs)
print(f"CH4 destroyed: {result['CH4_destroyed_m3']:.2f} m³")
```

## Electricity Balance

Own consumption is the electricity the plant uses itself. Every consumer
reports it as `P_consumed` [kW]:

| Component         | Source of `P_consumed`                                          |
|-------------------|-----------------------------------------------------------------|
| `Mixer`, `Pump`   | Hydraulic / mixing power model                                  |
| `Feeder`          | `specific_energy` [kWh/t FM], else the `power_installed` curve  |
| `Separator`       | `specific_energy` [kWh/t FM]                                    |
| `BiogasUpgrading` | `specific_electricity` [kWh/m³ raw gas], default 0.25           |
| `CHP`             | `own_consumption_fraction` of the gross `P_el`, default 0.03    |

The CHP keeps `P_el` as gross output and adds `P_el_net`. The plant sums
everything:

```python
plant.step(dt)
balance = plant.get_electricity_balance()
# {'P_el_produced': 500.0, 'P_el_consumed': 41.3, 'P_el_net': 458.7,
#  'own_consumption_fraction': 0.083, 'consumers': {'chp': 15.0, 'mixer_main': 11.2, ...},
#  'energy_produced_kwh': ..., 'energy_consumed_kwh': ..., 'energy_net_kwh': ...}
```

The cumulative energies advance on every `step()` and are saved with the plant.
The defaults for the CHP auxiliaries and the upgrading unit are typical values;
set them from the data sheet of the actual unit.

## Complete Energy System

### Integrated Energy Chain

```python
from pyadm1.configurator import BiogasPlant, PlantConfigurator
from pyadm1.substrates import Feedstock

# Setup
feedstock = Feedstock(feeding_freq=48)
plant = BiogasPlant("Energy-Optimized Plant")
config = PlantConfigurator(plant, feedstock)

# Digester
digester, _ = config.add_digester(
    "main_digester",
    V_liq=2000,
    V_gas=300,
    Q_substrates=[15, 10, 0, 0, 0, 0, 0, 0, 0, 0]
)

# CHP
config.add_chp("chp1", P_el_nom=500, eta_el=0.40, eta_th=0.45)

# Heating
config.add_heating("heat1", target_temperature=308.15, heat_loss_coefficient=0.5)

# Auto-connections (also creates the flare automatically)
config.auto_connect_digester_to_chp("main_digester", "chp1")
config.auto_connect_chp_to_heating("chp1", "heat1")

# Simulate
plant.initialize()
results = plant.simulate(duration=30, dt=1/24, save_interval=1.0)

# Energy balance
def energy_balance(results):
    """Compute the plant energy balance."""
    final = results[-1]
    comp = final['components']

    # Gas production
    Q_gas = comp['main_digester']['Q_gas']  # m³/d
    Q_ch4 = comp['main_digester']['Q_ch4']  # m³/d
    E_gas = Q_ch4 * 10.0  # kWh/d (10 kWh/m³ CH4)

    # CHP output
    P_el = comp['chp1']['P_el']  # kW
    P_th = comp['chp1']['P_th']  # kW
    E_el = P_el * 24  # kWh/d
    E_th = P_th * 24  # kWh/d

    # Heat demand
    Q_heat = comp['heat1']['Q_heat_supplied']  # kW
    E_heat_needed = Q_heat * 24  # kWh/d

    # Efficiencies
    eta_el_actual = E_el / E_gas if E_gas > 0 else 0
    eta_th_actual = E_th / E_gas if E_gas > 0 else 0
    eta_total = (E_el + E_th) / E_gas if E_gas > 0 else 0

    # Heat utilization
    heat_utilization = E_heat_needed / E_th if E_th > 0 else 0

    return {
        'E_gas': E_gas,
        'E_el': E_el,
        'E_th': E_th,
        'E_heat_needed': E_heat_needed,
        'eta_el': eta_el_actual,
        'eta_th': eta_th_actual,
        'eta_total': eta_total,
        'heat_utilization': heat_utilization,
        'excess_heat': max(0, E_th - E_heat_needed)
    }

balance = energy_balance(results)
print("\nEnergy balance:")
print(f"Gas energy: {balance['E_gas']:.0f} kWh/d")
print(f"Electricity: {balance['E_el']:.0f} kWh/d (η={balance['eta_el']:.1%})")
print(f"Heat: {balance['E_th']:.0f} kWh/d (η={balance['eta_th']:.1%})")
print(f"Overall efficiency: {balance['eta_total']:.1%}")
print(f"Heat utilization: {balance['heat_utilization']:.1%}")
print(f"Excess heat: {balance['excess_heat']:.0f} kWh/d")
```

## Optimization Strategies

### 1. Heat-Utilization Optimization

```python
def optimize_heat_utilization(plant, results):
    """Optimize heat utilization through load management."""

    # Analyze excess heat
    excess_heat = []
    for result in results:
        P_th = result['components']['chp1']['P_th']
        Q_heat = result['components']['heat1']['Q_heat_supplied']
        excess_heat.append(max(0, P_th - Q_heat))

    avg_excess = sum(excess_heat) / len(excess_heat)

    if avg_excess > 50:  # kW
        print(f"Average excess heat: {avg_excess:.1f} kW")
        print("Optimization options:")
        print("- Reduce CHP size")
        print("- Add additional heat use (drying, etc.)")
        print("- Use heat storage")

    return avg_excess

optimize_heat_utilization(plant, results)
```

### 2. Load-Following Operation

```python
# CHP load control based on gas production
def load_following_control(Q_gas_available, P_el_nom):
    """Adjust CHP load to available gas."""

    # Minimum load: 40% for stable combustion
    min_load = 0.4

    # Compute optimal load
    E_gas = Q_gas_available / 24 * 10  # kW
    load = min(1.0, max(min_load, E_gas / (P_el_nom / 0.40)))

    return load

# Apply in simulation
load_setpoint = load_following_control(Q_gas_available, 500)
chp_inputs = {
    'Q_gas_supplied_m3_per_day': Q_gas_available,
    'load_setpoint': load_setpoint
}
```

### 3. Gas-Storage Management

```python
def manage_gas_storage(storage_state, chp_demand):
    """Optimize gas storage fill level."""

    utilization = storage_state['utilization']
    pressure = storage_state['pressure_bar']

    # Target: 30–70% fill for flexibility
    if utilization < 0.3:
        print("Low storage level - increase gas production or reduce CHP load")
        adjust_load = 0.8
    elif utilization > 0.7:
        print("High storage level - increase CHP load or prepare for venting")
        adjust_load = 1.2
    else:
        adjust_load = 1.0

    return adjust_load

# Apply in simulation
load_adjustment = manage_gas_storage(storage.outputs_data, chp_demand)
```

## Performance Metrics

### CHP Availability

```python
def calculate_chp_availability(results, P_el_nom):
    """Compute CHP availability and utilization."""

    total_hours = len(results) / 24  # days * 24 h
    running_hours = sum(1 for r in results if r['components']['chp1']['P_el'] > 0) / 24

    availability = running_hours / total_hours

    # Load factor: electrical power relative to the nominal power
    load_factors = [r['components']['chp1']['P_el'] / P_el_nom for r in results]
    avg_load = sum(load_factors) / len(load_factors)

    return {
        'availability': availability,
        'running_hours': running_hours,
        'avg_load': avg_load,
        'full_load_hours': running_hours * avg_load
    }

metrics = calculate_chp_availability(results, P_el_nom=500)
print(f"CHP availability: {metrics['availability']:.1%}")
print(f"Operating hours: {metrics['running_hours']:.0f} h")
print(f"Average load: {metrics['avg_load']:.1%}")
print(f"Full-load hours: {metrics['full_load_hours']:.0f} h")
```

## Troubleshooting

### Problem: CHP Not Running

**Diagnosis:**

```python
chp_outputs = chp.step(t, dt, inputs)

if chp_outputs['P_el'] == 0:
    print("CHP not running - check:")
    print(f"- Available gas: {inputs.get('Q_gas_supplied_m3_per_day', 0):.1f} m³/d")
    print(f"- Minimum gas demand: {P_el_nom / eta_el * 24 / 10:.1f} m³/d")
    print(f"- Storage pressure: {storage.outputs_data['pressure_bar']:.2f} bar")
```

### Problem: Excessive Venting

**Cause:** Gas production > CHP consumption

**Solution:**

```python
# Option 1: Increase CHP capacity
config.add_chp("chp1", P_el_nom=750)  # From 500 to 750 kW

# Option 2: Add a second CHP
config.add_chp("chp2", P_el_nom=250)

# Option 3: Enlarge gas storage
storage = GasStorage("storage1", capacity_m3=1500)  # From 1000 to 1500
```

### Problem: Insufficient Heat

**Diagnosis:**

```python
heat_outputs = heating.step(t, dt, inputs)

if heat_outputs['P_aux_heat'] > 50:  # kW auxiliary heat
    print("High auxiliary heat demand:")
    print(f"- CHP heat: {heat_outputs['P_th_used']:.1f} kW")
    print(f"- Auxiliary heat: {heat_outputs['P_aux_heat']:.1f} kW")
    print("Solutions:")
    print("- Improve insulation (reduce k)")
    print("- Increase CHP size")
    print("- Lower digester temperature")
```

## Next Steps

- [Biological Components](biological.md): Digester and process control  
- [Mechanical Components](mechanical.md): Pumps and mixers  
- [Feeding Components](feeding.md): Storage and dosing  
- [API Reference](../../api/energy.md): Detailed class documentation  
