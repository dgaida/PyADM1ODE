# Plant Components

Modular Plant Components

This module provides a comprehensive library of biogas plant components that can be
combined to build complex plant configurations. All components follow a common
interface and can be dynamically loaded and connected.

Modules:
    base: Base classes (Component, ComponentType) defining the interface that all
          components must implement, including step(), initialize(), serialization,
          and connection management.

    registry: Component registry system for dynamic component discovery, loading,
             and instantiation, enabling plugin-like extensibility and automated
             component selection by LLM agents.

Subpackages:
    biological: Components for biological processes including digesters (single/multi-stage),
               hydrolysis tanks for pre-treatment, and solid-liquid separators for
               digestate processing.

    mechanical: Mechanical components including pumps (centrifugal, positive displacement),
               mixers/agitators (various types), valves (control, safety), and heat
               exchangers for thermal management.

    energy: Energy conversion and storage components including CHP units (gas engines,
           micro-turbines), boilers (auxiliary heating), gas storage (low/high pressure),
           and flares for safety gas combustion.

    feeding: Substrate handling components including storage silos, automated dosing
            systems (screw feeders, pumps), and mixer wagons for substrate preparation.

    sensors: Measurement components including physical sensors (pH, temperature, pressure),
            chemical sensors (VFA, ammonia, COD), and gas analyzers (CH4, CO2, H2S, O2).

Example:
    >>> from pyadm1.components.biological import Digester
    >>> from pyadm1.components.energy import CHP
    >>> from pyadm1.components import ComponentRegistry
    >>>
    >>> # Direct instantiation
    >>> digester = Digester("dig1", feedstock, V_liq=2000)
    >>>
    >>> # Via registry
    >>> registry = ComponentRegistry()
    >>> component = registry.create("Digester", "dig1", feedstock=feedstock)

## Subpackages

### [biological](biological.md)

Biological Process Components

### [energy](energy.md)

Energy Conversion and Storage Components

### [feeding](feeding.md)

Substrate Feeding and Storage Components

### [mechanical](mechanical.md)

Mechanical Plant Components

### [sensors](sensors.md)

Measurement and Sensor Components

## Base Classes

### Component Base

```python
from pyadm1.components.base import Component
```

Abstract base class for all biogas plant components.

All components must implement the step() method which performs
one simulation time step.

**Signature:**

```python
Component(
    component_id,
    component_type,
    name=None
)
```

**Methods:**

#### `add_input()`

```python
add_input(component_id)
```

Add an input connection.

#### `add_output()`

```python
add_output(component_id)
```

Add an output connection.

#### `get_state()`

```python
get_state()
```

Get current component state.

#### `initialize()`

```python
initialize(initial_state=None)
```

Initialize component state.

Parameters
----------
initial_state : Optional[Dict[str, Any]]
    Initial state values

#### `set_state()`

```python
set_state(state)
```

Set component state.

#### `step()`

```python
step(t, dt, inputs)
```

Perform one simulation time step.

Parameters
----------
t : float
    Current simulation time [days]
dt : float
    Time step size [days]
inputs : Dict[str, Any]
    Input data from connected components

Returns
-------
Dict[str, Any]
    Output data to be passed to connected components

#### `to_dict()`

```python
to_dict()
```

Serialize component to dictionary for JSON export.

Returns
-------
Dict[str, Any]
    Component configuration as dictionary

