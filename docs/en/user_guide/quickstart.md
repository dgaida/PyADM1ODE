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
from pyadm1.configurator.plant_builder import BiogasPlant
from pyadm1.substrates.feedstock import Feedstock
from pyadm1.configurator.plant_configurator import PlantConfigurator

# 1. Create feedstock manager (loads the bundled substrate library)
feedstock = Feedstock(feeding_freq=48)  # Time between feeding events [hours]

# 2. Define substrate feed rates [t FM/d], one entry per substrate slot
# Slot 0 is maize silage, slot 1 is cattle manure (see "Available Substrates")
Q_substrates = [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]

# 3. Create and configure plant
plant = BiogasPlant("My First Biogas Plant")
configurator = PlantConfigurator(plant, feedstock)

configurator.add_digester(
    digester_id="main_digester",
    V_liq=2000.0,        # Liquid volume [m³]
    V_gas=300.0,         # Gas volume [m³]
    T_ad=308.15,         # Temperature [K] = 35°C
    Q_substrates=Q_substrates
)

# 4. Initialize and simulate
plant.initialize()

results = plant.simulate(
    duration=10.0,       # Simulation time [days]
    dt=1.0/24.0,        # Time step [days] = 1 hour
    save_interval=1.0   # Save results daily
)

# 5. View results
for result in results:
    time = result["time"]
    digester = result["components"]["main_digester"]

    print(f"Day {time:.0f}:")
    print(f"  Biogas:  {digester['Q_gas']:.1f} Nm³/d")
    print(f"  Methane: {digester['Q_ch4']:.1f} Nm³/d")
    print(f"  pH:      {digester['pH']:.2f}")
    print(f"  VFA:     {digester['VFA']:.2f} g/L")
```

**Expected Output:**

```text
Simulated 100/240 steps
Simulated 200/240 steps
Day 0:
  Biogas:  4601.8 Nm³/d
  Methane: 992.4 Nm³/d
  pH:      7.16
  VFA:     0.05 g/L
Day 1:
  Biogas:  2559.6 Nm³/d
  Methane: 1247.4 Nm³/d
  pH:      7.34
  VFA:     0.15 g/L
...
Day 10:
  Biogas:  2561.9 Nm³/d
  Methane: 1349.9 Nm³/d
  pH:      7.37
  VFA:     0.20 g/L
```

The two `Simulated ...` lines are progress messages of `simulate()`. `Day 0` is
the result after the first hourly step, when the gas flow is still settling
from the automatically built initial state.

Feed rates are tonnes of fresh matter per day (t FM/d). Gas flows are dry gas
at standard conditions (Nm³/d at 0°C and 1.01325 bar), and the methane content
is that of the dry gas: 1349.9 / 2561.9 = 52.7% at day 10.

## Basic Single Digester

A complete script for a single digester is [`examples/01_basic_digester.py`](https://github.com/dgaida/PyADM1ODE/blob/master/examples/01_basic_digester.py); it uses its own digester size and substrate mix.

### System Architecture

The basic digester consists of:

```text
[Substrate Feed] → [Digester] → [Gas Storage] → [Biogas Output]
   15 t/d maize        2000 m³      300 m³         ~2560 Nm³/d
   10 t/d manure       @ 35°C       membrane       ~53% CH₄
```

### Key Features

- **Single-stage CSTR**: Continuously stirred tank reactor  
- **Automatic gas storage**: Created per digester (low-pressure membrane)  
- **Substrate mixing**: Corn silage + cattle manure  
- **Temperature control**: Mesophilic (35°C)  

### Plant Configuration Details

The digester call of the first simulation, on a new plant and with a readable name:

```python
plant = BiogasPlant("My First Biogas Plant")
configurator = PlantConfigurator(plant, feedstock)

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
    Q_substrates=[15, 10, 0, 0, 0, 0, 0, 0, 0, 0]
)

print(plant.get_summary())
```

Output:

```text
=== My First Biogas Plant ===
Simulation time: 0.00 days

Components (2):
  - Main Digester (digester)  
  - Main Digester Gas Storage (storage)  

Connections (1):
  - Main Digester -> Main Digester Gas Storage (gas)  
```

The digester has no internal gas storage: the storage is a separate component
with the ID `<digester_id>_storage`, here `main_digester_storage`.

The initial state of the digester is built automatically from the feedstock
and the feed rates. To start from your own state, pass a 41-element state
vector as `adm1_state`, for example one loaded with
`pyadm1.core.adm1.get_state_zero_from_csv`.

### Understanding the Output

**Gas Storage Behavior:**

```python
# results[-1]["components"]["main_digester_storage"] of the first simulation
# at day 10 (values rounded):
{
    'stored_volume_m3': 300.0,           # Current volume [m³ STP]
    'pressure_bar': 1.05,                # Current pressure [bar]
    'utilization': 1.0,                  # Fill level (1.0 = full)
    'vented_volume_m3': 106.7,           # Vented this step [m³]
    'cumulative_vented_m3': 25513.2,     # Vented since the start [m³]
    'Q_gas_supplied_m3_per_day': 0.0,    # Supplied to consumers [m³/d]
    'Q_ch4_supplied_m3_per_day': 0.0,    # Methane within that supply [m³/d]
    'CH4_fraction': 0.527,               # CH4 fraction of the stored gas
    'CO2_fraction': 0.473,               # CO2 fraction of the stored gas
    'pressure_setpoint_bar': None        # Pressure setpoint, if one was set
}
```

The first simulation has no gas consumer, so the storage is full within the
first day and all further production is vented:

- Pressure rises with the fill level from 1.013 bar (empty) to 1.05 bar (full)  
- Gas beyond the capacity is vented (`vented_volume_m3` per step, `cumulative_vented_m3` in total)  
- Without a flare the vented gas is released unburnt  
- With a CHP attached, the vented gas is burnt in the flare of the CHP (next section)  

### Process Stability Indicators

pH, VFA and TAC are part of the digester results. The limits in this check are
common rules of thumb, not values defined by PyADM1ODE; adjust them to your
plant.

```python
# Check process stability at the end of the first simulation
final = results[-1]["components"]["main_digester"]

# pH between 6.8 and 7.5
if 6.8 <= final['pH'] <= 7.5:
    print("✓ pH stable")
else:
    print(f"⚠ pH unstable: {final['pH']:.2f}")

# VFA below 5 g/L
if final['VFA'] < 5.0:
    print("✓ VFA acceptable")
else:
    print(f"⚠ High VFA: {final['VFA']:.2f} g/L")

# FOS/TAC ratio (VFA / TAC) below 0.3
if final['TAC'] > 0:
    fos_tac = final['VFA'] / final['TAC']
    if fos_tac < 0.3:
        print(f"✓ FOS/TAC stable: {fos_tac:.3f}")
    else:
        print(f"⚠ FOS/TAC high: {fos_tac:.3f}")
```

Output:

```text
✓ pH stable
✓ VFA acceptable
✓ FOS/TAC stable: 0.036
```

**Performance metrics of the first simulation (day 10):**

| Metric | Value |
|--------|-------|
| Biogas production | 2561.9 Nm³/d |
| Methane production | 1349.9 Nm³/d |
| Methane content | 52.7% |
| Specific gas yield | 102.5 Nm³/t FM |
| pH | 7.37 |
| VFA | 0.20 g/L |
| TAC | 5.57 g CaCO₃/L |
| FOS/TAC | 0.036 |

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
print(f"  Biogas:  {digester['Q_gas']:.1f} Nm³/d")
print(f"  Methane: {digester['Q_ch4']:.1f} Nm³/d")
print(f"  CH4 Content: {digester['Q_ch4']/digester['Q_gas']*100:.1f}%")
print(f"\nCHP Performance:")
print(f"  Electrical Power: {chp['P_el']:.1f} kW")
print(f"  Thermal Power:    {chp['P_th']:.1f} kW")
print(f"  Gas Consumed:     {chp['Q_gas_consumed']:.1f} Nm³/d")
print(f"\nHeating:")
print(f"  Heat Supplied:    {heating['Q_heat_supplied']:.1f} kW")
print(f"  Auxiliary Heat:   {heating['P_aux_heat']:.1f} kW")
```

**Expected Output:**

```text
Simulated 100/240 steps
Simulated 200/240 steps

Final Results (Day 10):

Digester Performance:
  Biogas:  2561.9 Nm³/d
  Methane: 1349.9 Nm³/d
  CH4 Content: 52.7%

CHP Performance:
  Electrical Power: 225.0 kW
  Thermal Power:    253.1 kW
  Gas Consumed:     2561.9 Nm³/d

Heating:
  Heat Supplied:    10.0 kW
  Auxiliary Heat:   0.0 kW
```

The CHP delivers 225.0 kW of its 500 kW rating: it burns all the gas the
digester produces, the storage is emptied in every step and nothing reaches
the flare.

The heat demand of 10.0 kW is the flat heat-loss term only (0.5 kW/K × 20 K
between 35°C and the default ambient temperature of 15°C). Pass
`digester_id="main_digester"` to `add_heating` to include the heat for warming
the substrate feed; the demand is then 36.1 kW.

**Automatic Component Creation:**

PlantConfigurator creates and connects:

- **Gas Storage**: One per digester (membrane, sized to V_gas)  
- **Flare**: One per CHP (safety combustion, 98% CH₄ destruction)  

**Connection Chain:**

```text
Digester → Gas Storage → CHP → Flare
                          ↓
                      Heating
```

## Two-Stage Digestion

A complete two-stage script is [`examples/02_two_stage_plant.py`](https://github.com/dgaida/PyADM1ODE/blob/master/examples/02_two_stage_plant.py); it uses its own tank sizes and substrate mix.

### System Architecture

```text
[Feed] → [Hydrolysis] → [Storage 1] ↘
         500 m³          100 m³       → [CHP] → [Flare]
         @ 45°C                      ↗  500 kW    98%
                                    ↓
[Effluent] → [Main] → [Storage 2] ↗  [Heat 1] + [Heat 2]
             2000 m³    300 m³        45°C       35°C
             @ 35°C
```

### Key Features

- **Temperature-phased**: 45°C in the first stage, 35°C in the second  
- **Two digesters in series**: The effluent of the first stage feeds the second  
- **One gas storage per digester**: Both storages supply the same CHP  
- **Mechanical components**: Pumps and mixers add their electricity demand  
- **Energy integration**: The CHP supplies electricity and the heat for both heating systems  

### Configuration

```python
plant = BiogasPlant("Two-Stage Plant")
configurator = PlantConfigurator(plant, feedstock)

# Stage 1: Hydrolysis tank (45°C)
configurator.add_digester(
    digester_id="hydrolysis_tank",
    V_liq=500.0,
    V_gas=100.0,
    T_ad=318.15,  # 45°C for faster hydrolysis
    Q_substrates=[15, 10, 0, 0, 0, 0, 0, 0, 0, 0]
)

# Stage 2: Main digester (mesophilic, 35°C)
configurator.add_digester(
    digester_id="main_digester",
    V_liq=2000.0,
    V_gas=300.0,
    T_ad=308.15,  # 35°C
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

# Initialize and simulate
plant.initialize()
results = plant.simulate(duration=10.0, dt=1.0/24.0, save_interval=1.0)
```

The pumps and mixers are added without connections: they run at their nominal
operating point and contribute only their electricity demand.
`configurator.add_mixer(mixer_id, digester_id)` instead installs a mixer in a
digester and takes the tank geometry from it (see
[Mechanical components](components/mechanical.md)).

### Gas Flow in Four Passes

Every plant step runs in four passes, so that each component is stepped once
with the gas it actually receives:

1. **Digesters** and all other components outside the gas train: the digesters produce gas  
2. **Gas storages**: each storage takes the production of its digester and the demand of its consumers at the same time; gas beyond its capacity is vented  
3. **Gas consumers**: the CHP runs on the gas actually supplied, then the heating systems use its heat  
4. **Flares**: burn the gas the storages vented  

The flows of the last step (day 10):

```python
final = results[-1]["components"]

for digester_id in ("hydrolysis_tank", "main_digester"):
    storage = final[f"{digester_id}_storage"]
    print(f"{digester_id}:")
    print(f"  Produced:         {final[digester_id]['Q_gas']:7.1f} Nm³/d")
    print(f"  Storage supplies: {storage['Q_gas_supplied_m3_per_day']:7.1f} Nm³/d")
    print(f"  Storage vents:    {storage['vented_volume_m3']:7.1f} m³")

print("chp_main:")
print(f"  Consumed:         {final['chp_main']['Q_gas_consumed']:7.1f} Nm³/d")
print(f"  Flared:           {final['chp_main_flare']['vented_volume_m3']:7.1f} m³")
```

Output:

```text
hydrolysis_tank:
  Produced:          2087.8 Nm³/d
  Storage supplies:  2087.8 Nm³/d
  Storage vents:        0.0 m³
main_digester:
  Produced:           397.1 Nm³/d
  Storage supplies:   397.1 Nm³/d
  Storage vents:        0.0 m³
chp_main:
  Consumed:          2484.9 Nm³/d
  Flared:               0.0 m³
```

### Comparison with the Single-Stage Plant

Both plants receive the same feed. Values at day 10:

| Aspect | Single-Stage | Two-Stage |
|--------|-------------|-----------|
| **Digester volume** | 2000 m³ | 500 m³ + 2000 m³ |
| **Biogas production** | 2561.9 Nm³/d | 2484.9 Nm³/d |
| **Methane production** | 1349.9 Nm³/d | 1315.6 Nm³/d |
| **CH₄ content** | 52.7% | 52.9% |
| **CHP electrical power** | 225.0 kW | 219.3 kW |

Ten days are short compared with the retention time (86 days in the
single-stage digester), so these values describe the start of operation, not a
steady state.

### Expected Results

**Energy Balance:**

```python
chp = final["chp_main"]
balance = plant.get_electricity_balance()

print("Energy Production:")
print(f"  Electrical (gross):  {balance['P_el_produced']:7.1f} kW")
print(f"  Thermal:             {chp['P_th']:7.1f} kW")

print("\nOwn Consumption:")
for component_id, power in balance["consumers"].items():
    print(f"  {component_id + ':':<20} {power:7.2f} kW")
print(f"  {'Total:':<20} {balance['P_el_consumed']:7.2f} kW")

print(f"\nNet Electrical Output: {balance['P_el_net']:7.1f} kW")

heat_demand = final["heating_1"]["Q_heat_supplied"] + final["heating_2"]["Q_heat_supplied"]
print("\nHeat Utilization:")
print(f"  Heating demand:      {heat_demand:7.1f} kW")
print(f"  CHP thermal supply:  {chp['P_th']:7.1f} kW")
```

Output:

```text
Energy Production:
  Electrical (gross):    219.3 kW
  Thermal:               246.7 kW

Own Consumption:
  chp_main:               6.58 kW
  feed_pump:              0.03 kW
  transfer_pump:          0.04 kW
  mixer_1:                3.75 kW
  mixer_2:                2.50 kW
  Total:                 12.90 kW

Net Electrical Output:   206.4 kW

Heat Utilization:
  Heating demand:         25.0 kW
  CHP thermal supply:    246.7 kW
```

## Working with Different Substrates

### Available Substrates

PyADM1ODE includes 13 pre-configured agricultural substrates. `Feedstock()`
without a substrate list loads all of them in this order; the slot is the
position in `Q_substrates`:

| Slot | Substrate ID | Substrate |
|------|--------------|-----------|
| 0 | `maize_silage_milk_ripeness` | Maize silage (milk ripeness) |
| 1 | `cattle_manure` | Cattle manure |
| 2 | `swine_manure` | Swine manure |
| 3 | `corn_cob_mix` | Corn-cob mix (CCM) |
| 4 | `grass_silage` | Grass silage |
| 5 | `green_rye_silage` | Green rye silage |
| 6 | `cereal_gps_silage` | Cereal whole-plant silage (GPS) |
| 7 | `onion_waste` | Onion waste |
| 8 | `maize_silage_gummersbach` | Maize silage (Gummersbach plant) |
| 9 | `cattle_manure_solid` | Cattle manure (solid) |
| 10 | `chicken_manure_dry` | Chicken manure dry (HTK) |
| 11 | `swine_manure_gummersbach` | Swine manure (Gummersbach plant) |
| 12 | `wheat_whole_plant_silage` | Wheat whole-plant silage |

`Q_substrates` may be shorter than the list; missing slots count as zero. To
fix the slots yourself, name the substrates:
`Feedstock(["maize_silage_milk_ripeness", "cattle_manure"])` with
`Q_substrates=[15, 10]` gives the same results as the first simulation. The
substrates are described in the [substrate library](../substrates/library.md).

### Substrate Feed Examples

```python
# High-energy mix (maize silage + cattle manure)
Q = [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]

# Grass-based (grass silage + cattle manure)
Q = [0, 5, 0, 0, 20, 0, 0, 0, 0, 0]

# Waste-based (manure + vegetables)
Q = [0, 10, 15, 0, 0, 0, 0, 5, 0, 0]

# Energy crop focus (maize silage + green rye silage + cattle manure)
Q = [20, 5, 0, 0, 0, 10, 0, 0, 0, 0]
```

These arrays only show how the slots are addressed. Whether a mix runs stably
depends on the plant, see [Troubleshooting](#troubleshooting).

### Substrate Information

Get detailed substrate properties:

```python
from pyadm1.substrates import SubstrateRegistry

# View substrate parameters
substrate = SubstrateRegistry().get("maize_silage_milk_ripeness")
print(f"Name: {substrate.name}")
print(f"pH value: {substrate.pH}")
print(f"Dry matter: {substrate.TS / 10:.1f} %FM")
print(f"Volatile solids content: {(1 - substrate.fRA) * 100:.1f} %TS")
print(f"Biochemical methane potential: {substrate.BMP:.0f} Nm³ CH4/t VS")
```

Output:

```text
Name: Maize silage (milk ripeness)
pH value: 3.9
Dry matter: 32.6 %FM
Volatile solids content: 94.0 %TS
Biochemical methane potential: 338 Nm³ CH4/t VS
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

The file holds the configuration and the current state of the plant, so the
loaded plant continues at the simulation time it was saved at.

## Understanding Results

### Key Output Variables

#### Digester Outputs

- `Q_gas` - Biogas production, dry gas [Nm³/d]  
- `Q_ch4` - Methane production [Nm³/d]  
- `Q_co2` - CO2 production [Nm³/d]  
- `pH` - pH value [-]  
- `VFA` - Volatile fatty acids [g HAceq/L]  
- `TAC` - Total alkalinity [g CaCO3/L]  

#### CHP Outputs

- `P_el` - Electrical power [kW]  
- `P_th` - Thermal power [kW]  
- `Q_gas_consumed` - Gas consumption [Nm³/d]  
- `CH4_fraction` - Methane fraction of the gas burnt [0-1]  
- `P_consumed` - Electricity used by the CHP's auxiliaries [kW]  

#### Heating Outputs

- `Q_heat_supplied` - Heat delivered [kW]  
- `P_th_used` - CHP heat used [kW]  
- `P_aux_heat` - Auxiliary heat needed [kW]  

#### Gas Storage Outputs

- `stored_volume_m3` - Current volume [m³ STP]  
- `pressure_bar` - Current pressure [bar]  
- `utilization` - Fill level [0-1]  
- `vented_volume_m3` - Gas vented in this step [m³]  
- `Q_gas_supplied_m3_per_day` - Gas supplied to consumers [m³/d]  

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
    configurator.add_digester("main_digester", V_liq=2000, Q_substrates=Q)

    plant.initialize()
    results = plant.simulate(duration=10.0, dt=1.0/24.0)

    final = results[-1]["components"]["main_digester"]
    results_all.append({
        'feed': feed,
        'biogas': final['Q_gas'],
        'methane': final['Q_ch4']
    })

for r in results_all:
    print(f"Feed {r['feed']} t/d → CH4 {r['methane']:.1f} Nm³/d")
```

Output (after the progress messages):

```text
...
Feed 10 t/d → CH4 989.4 Nm³/d
Feed 15 t/d → CH4 1376.3 Nm³/d
Feed 20 t/d → CH4 1750.4 Nm³/d
Feed 25 t/d → CH4 2113.1 Nm³/d
```

### Pattern 2: Time Series Analysis

Track evolution over time. `results` is the list returned by
`plant.simulate()`:

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
ax1.set_ylabel('Production [Nm³/d]')
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
   - [`examples/01_basic_digester.py`](https://github.com/dgaida/PyADM1ODE/blob/master/examples/01_basic_digester.py)  
   - [`examples/02_two_stage_plant.py`](https://github.com/dgaida/PyADM1ODE/blob/master/examples/02_two_stage_plant.py)  
   - [`examples/05_parallel_two_stage_simulation.py`](https://github.com/dgaida/PyADM1ODE/blob/master/examples/05_parallel_two_stage_simulation.py)  
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
configurator.add_digester(
    "main_digester", V_liq=2000.0, V_gas=300.0, T_ad=308.15,
    Q_substrates=[15, 10, 0, 0, 0, 0, 0, 0, 0, 0]
)
configurator.add_chp("chp_main", P_el_nom=500.0, eta_el=0.40, eta_th=0.45)
configurator.add_heating("heating_main", target_temperature=308.15)

# Connect
configurator.auto_connect_digester_to_chp("main_digester", "chp_main")
configurator.auto_connect_chp_to_heating("chp_main", "heating_main")
# General form: configurator.connect(from_id, to_id, connection_type)

# Simulate
plant.initialize()
results = plant.simulate(duration=10.0, dt=1.0/24.0, save_interval=1.0)

# Save/Load
plant.to_json("my_plant_config.json")
plant = BiogasPlant.from_json("my_plant_config.json", feedstock)
```

### Temperature Conversions

```python
# Common temperatures
T_mesophilic = 308.15  # 35°C
T_thermophilic = 328.15  # 55°C
T_psychrophilic = 298.15  # 25°C

# Convert °C to K
T_celsius = 35.0
T_K = T_celsius + 273.15  # 308.15
```

### Default Parameters

The values used when an argument is omitted:

```python
# configurator.add_digester
V_liq = 1050.0         # Liquid volume [m³]
V_gas = 150.0          # Gas volume [m³]
T_ad = 315.15          # Temperature [K] = 42°C

# configurator.add_chp
P_el_nom = 500.0       # Electrical power [kW]
eta_el = 0.40          # Electrical efficiency
eta_th = 0.45          # Thermal efficiency

# configurator.add_heating
target_temperature = 315.15    # Setpoint [K] = 42°C
heat_loss_coefficient = None   # [kW/K]; None = from the digester, else 0.5

# plant.simulate
dt = 1.0 / 24.0        # Time step [days] = 1 hour
save_interval = None   # None = save every step
```

## Troubleshooting

### Issue: Simulation unstable

**Symptoms**: pH drops, VFA rises, methane production decreases

**Solutions**:

- Reduce substrate feed rate  
- Increase retention time (larger V_liq)  
- Check substrate composition  

Example: with the grass-based mix from
[Substrate Feed Examples](#substrate-feed-examples) in the 2000 m³ digester at
35°C, VFA rise from 0.5 g/L at day 10 to 10.3 g/L at day 60. Half the feed or
twice the volume only delays this (3.7 g/L at day 60). The maize mix of the
first simulation stays at about 0.2 g/L over the same 60 days.

### Issue: Low gas production

**Solutions**:

- Increase substrate feed  
- Check substrate degradability  
- Check the digester temperature  

### Issue: Slow simulation

**Solutions**:

- Increase the time step `dt`  
- Increase `save_interval` to store fewer results  
- Use parallel simulation for parameter sweeps  

The digester equations are integrated with an adaptive solver inside each
step; `dt` sets how often the components exchange their values. The first
simulation gives the same biogas production at day 10 (2561.9 Nm³/d) with
`dt=1.0` as with hourly steps.

For more help, see the [Installation Guide](installation.md) or contact <daniel.gaida@th-koeln.de>.

## References

- **ADM1 Model**: Batstone et al. (2002). *Anaerobic Digestion Model No. 1*. IWA Publishing.  
- **Leitfaden Biogas**: FNR (2016). <https://mediathek.fnr.de/leitfaden-biogas.html>  
