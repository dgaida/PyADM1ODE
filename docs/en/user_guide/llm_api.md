# API for LLM

This page provides a structured reference of the methods and classes required to automatically create a PyADM1ODE simulation model. This documentation is optimized for reading by Large Language Models (LLMs) to generate biogas plant configurations.

!!! abstract "Skill for LLMs"
    You can download the full API documentation for LLMs as a "Skill" file here: [Skill.md](Skill.md) (dynamically generated during documentation build)

## Core Workflow

To create a simulation, follow this sequence:  
1. Create a `Feedstock` object.  
2. Create a `BiogasPlant` object.  
3. Use the `PlantConfigurator` to add and connect components.  
4. Initialize the plant and start the simulation.  

## 1. Substrate Configuration (Feedstock)

The `Feedstock` object defines which substrates are used in the plant.

```python
from pyadm1.substrates import Feedstock

# Create a feedstock with maize silage and swine manure
feedstock = Feedstock(
    substrates=["maize_silage_milk_ripeness", "swine_manure"],
    feeding_freq=24,   # Feeding frequency per day
    total_simtime=30   # Total simulation time in days
)
```

## 2. Plant Basis (BiogasPlant)

```python
from pyadm1.configurator import BiogasPlant

plant = BiogasPlant("Plant Name")
```

## 3. Plant Configurator (PlantConfigurator)

The `PlantConfigurator` is the primary tool for building the topology.

```python
from pyadm1.configurator import PlantConfigurator

configurator = PlantConfigurator(plant, feedstock)
```

### Adding Digesters

```python
# Adds a digester. Automatically creates a gas storage unit.
# Q_substrates specifies the amount of substrates defined in the feedstock in [m³/d].
digester, state_info = configurator.add_digester(
    digester_id="main_digester",
    V_liq=2000.0,              # Liquid volume [m³]
    V_gas=300.0,               # Gas headspace [m³]
    T_ad=308.15,               # Temperature [K] (35°C = 308.15K)
    Q_substrates=[15.0, 10.0]  # Amounts corresponding to the feedstock list
)
```

### Energy Components

```python
# Add Combined Heat and Power (CHP) unit
configurator.add_chp(
    chp_id="chp_1",
    P_el_nom=500.0  # Nominal electrical power [kW]
)

# Add heating system
configurator.add_heating(
    heating_id="heating_1",
    target_temperature=308.15  # Target temperature [K]
)
```

### Mechanical Components

Mechanical components must be added directly to the `plant`:

```python
from pyadm1.components.mechanical import Pump, Mixer
from pyadm1.components.feeding import SubstrateStorage, Feeder

# Pump
pump = Pump("pump1", pump_type="progressive_cavity", Q_nom=15.0)
plant.add_component(pump)

# Mixer
mixer = Mixer("mix1", tank_volume=2000.0, intermittent=True, on_time_fraction=0.25)
plant.add_component(mixer)

# Substrate Storage
storage = SubstrateStorage("silo1", storage_type="vertical_silo", capacity=1000.0)
plant.add_component(storage)

# Feeder
feeder = Feeder("feed1", feeder_type="screw", Q_max=20.0)
plant.add_component(feeder)
```

### Establishing Connections

```python
# Manual connection
configurator.connect("source_id", "target_id", connection_type="liquid") # or "gas", "heat", "default"

# Automatic connection helpers (recommended)
configurator.auto_connect_digester_to_chp("main_digester", "chp_1")
configurator.auto_connect_chp_to_heating("chp_1", "heating_1")
```

## 4. Running the Simulation

```python
plant.initialize()
results = plant.simulate(
    duration=30,      # Duration in days
    dt=1/24,          # Time step (e.g., hourly)
    save_interval=1.0 # Interval for result snapshots
)
```

## Summary for LLM Prompts

Use these classes for construction:  
- **Biological Stage**: `configurator.add_digester()`  
- **Energy**: `configurator.add_chp()`, `configurator.add_heating()`  
- **Mechanics**: `Pump`, `Mixer`  
- **Logistics**: `SubstrateStorage`, `Feeder`  
- **Infrastructure**: `PlantConfigurator.connect()`  
