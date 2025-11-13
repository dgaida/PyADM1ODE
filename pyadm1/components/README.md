## Plant Module - Modular Biogas Plant Architecture

The plant module provides a flexible, component-based architecture for building
complex biogas plant configurations.

### Key Features

- **Modular Components**: Digesters, CHP units, heating systems, and more
- **Flexible Connections**: Connect components in series, parallel, or custom configurations
- **JSON Configuration**: Save and load complete plant configurations
- **Component Library**: Extensible system for adding new component types

### Available Components

#### Digester
- Wraps PyADM1 model
- Supports multiple digesters in series/parallel
- Full ADM1 state tracking

#### CHP (Combined Heat and Power)
- Converts biogas to electricity and heat
- Configurable efficiency and capacity
- Integration with C# DLL calculations

#### Heating System
- Maintains digester temperature
- Uses CHP waste heat
- Calculates auxiliary heat demand

### Quick Start

```python
from pyadm1.plant import BiogasPlant, Digester, CHP, HeatingSystem, Connection
from pyadm1.substrates.feedstock import Feedstock

# Initialize feedstock
feedstock = Feedstock(feeding_freq=48)

# Create plant
plant = BiogasPlant("My Biogas Plant")

# Add components
digester1 = Digester("dig1", feedstock, V_liq=2000, name="Main Digester")
digester2 = Digester("dig2", feedstock, V_liq=1000, name="Post Digester")
chp = CHP("chp1", P_el_nom=500, name="CHP Unit")

plant.add_component(digester1)
plant.add_component(digester2)
plant.add_component(chp)

# Connect components
plant.add_connection(Connection("dig1", "dig2", "liquid"))
plant.add_connection(Connection("dig1", "chp1", "gas"))
plant.add_connection(Connection("dig2", "chp1", "gas"))

# Initialize and simulate
plant.initialize()
results = plant.simulate(duration=30, dt=1/24)  # 30 days, hourly steps

# Save configuration
plant.to_json("my_plant_config.json")

# Load configuration
plant2 = BiogasPlant.from_json("my_plant_config.json", feedstock)
```

### JSON Configuration Format

Plant configurations are stored in JSON format with the following structure:

```json
{
  "plant_name": "My Biogas Plant",
  "simulation_time": 0.0,
  "components": [
    {
      "component_id": "digester_1",
      "component_type": "digester",
      "name": "Main Digester",
      "V_liq": 1977.0,
      "V_gas": 304.0,
      "T_ad": 308.15,
      "state": {...}
    },
    ...
  ],
  "connections": [
    {
      "from": "digester_1",
      "to": "digester_2",
      "type": "liquid"
    },
    ...
  ]
}
```

### Adding Custom Components

Create new component types by inheriting from `Component`:

```python
from pyadm1.plant.component_base import Component, ComponentType

class MyComponent(Component):
    def __init__(self, component_id, name=None):
        super().__init__(component_id, ComponentType.CUSTOM, name)

    def initialize(self, initial_state=None):
        self.state = {...}

    def step(self, t, dt, inputs):
        # Implement component behavior
        return outputs

    def to_dict(self):
        return {...}

    @classmethod
    def from_dict(cls, config):
        return cls(...)
```

### Examples

See `examples/two_stage_digester.py` for a complete two-stage digester
configuration with CHP and heating systems.

### Integration with C# DLLs

Component calculations can leverage existing C# DLLs:
- `plant.dll`: Plant-level calculations
- `biogas.dll`: Biogas production and composition
- `physchem.dll`: Physical-chemical properties

Simply add `clr.AddReference()` calls in component methods to access DLL functions.
