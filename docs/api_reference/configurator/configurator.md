# Plant Model Configurator and MCP Server

Plant Model Configurator and MCP Server

This module provides tools for building, validating, and automatically configuring
biogas plant models, including an MCP server for LLM-driven plant design.

Modules:

    plant_builder: BiogasPlant class for assembling components into complete plant
                  models, managing component lifecycle, connections, and providing
                  JSON serialization for model persistence and sharing.

    connection_manager: Connection handling between components including type checking
                       (liquid, gas, heat, power), flow validation, and dependency
                       resolution for correct simulation order.

    validation: Model validation checking for physical consistency (mass/energy balance),
               completeness (all required connections), and parameter validity with
               detailed error messages for debugging.

Subpackages:

    templates: Pre-defined plant configurations for common biogas plant layouts
              (single-stage, two-stage, plug-flow, CSTR) serving as starting
              points for customization or examples for learning.

Example:

```python
    >>> from pyadm1.configurator import BiogasPlant, ConnectionManager
    >>> from pyadm1.components.biological import Digester
    >>>
    >>> # Build plant programmatically
    >>> plant = BiogasPlant("My Plant")
    >>> plant.add_component(Digester("dig1", feedstock, V_liq=2000))
    >>> plant.validate()
    >>>
    >>> # Save configuration
    >>> plant.to_json("my_plant.json")
    >>>
    >>> # Load from template
    >>> from pyadm1.configurator.templates import TwoStageTemplate
    >>> plant = TwoStageTemplate.create(feedstock=feedstock)
```

## Subpackages

### [templates](templates.md)

Plant Configuration Templates

## Base Classes

- [BiogasPlant](#biogasplant)
- [Connection](#connection)
- [ConnectionManager](#connectionmanager)
- [ConnectionType](#connectiontype)

### BiogasPlant

```python
from pyadm1.configurator import BiogasPlant
```

Complete biogas plant model with multiple components.

Manages component lifecycle, connections, and simulation.
Supports JSON-based configuration.

Attributes:

    plant_name (str): Name of the biogas plant.
    components (Dict[str, Component]): Dictionary of all plant components.
    connections (List[Connection]): List of connections between components.
    simulation_time (float): Current simulation time in days.

Example:

```python
    >>> from pyadm1.substrates.feedstock import Feedstock
    >>> from pyadm1.configurator.plant_builder import BiogasPlant
    >>> from pyadm1.components.biological.digester import Digester
    >>>
    >>> feedstock = Feedstock(feeding_freq=48)
    >>> plant = BiogasPlant("My Plant")
    >>> digester = Digester("dig1", feedstock, V_liq=2000)
    >>> plant.add_component(digester)
    >>> plant.initialize()
```

**Signature:**

```python
BiogasPlant(
    plant_name='Biogas Plant'
)
```

**Methods:**

#### `add_component()`

```python
add_component(component)
```

Add a component to the plant.

Args:

    component (Component): Component to add to the plant.

Raises:

    ValueError: If component with same ID already exists.

#### `add_connection()`

```python
add_connection(connection)
```

Add a connection between components.

Args:

    connection (Connection): Connection to add.

Raises:

    ValueError: If source or target component not found.

#### `get_summary()`

```python
get_summary()
```

Get human-readable summary of plant configuration.

Returns:

    str: Summary text with components and connections.

#### `initialize()`

```python
initialize()
```

Initialize all components.

Note: Most components auto-initialize in their constructor.
This method is kept for compatibility and to ensure any
components that need explicit initialization are handled.

#### `simulate()`

```python
simulate(duration, dt=0.041666666666666664, save_interval=None)
```

Run simulation for specified duration.

Args:

    duration (float): Simulation duration in days.
    dt (float): Time step in days. Defaults to 1 hour (1/24 day).
    save_interval (Optional[float]): Interval for saving results in days.
        If None, saves every step.

Returns:

    List[Dict[str, Any]]: Simulation results at each saved time point.
        Each entry contains 'time' and 'components' with component results.

Example:

```python
    >>> results = plant.simulate(duration=30, dt=1/24, save_interval=1.0)
    >>> print(f"Simulated {len(results)} time points")
```

#### `step()`

```python
step(dt)
```

Perform one simulation time step for all components.

This uses a three-pass execution model:
1. Execute digesters to produce gas → storages
2. Execute CHPs to determine gas demand → storages
3. Execute storages to supply gas → CHPs (re-execute with actual supply)

Args:

    dt (float): Time step in days.

Returns:

    Dict[str, Dict[str, Any]]: Results from all components.

#### `to_json()`

```python
to_json(filepath)
```

Save plant configuration to JSON file.

Args:

    filepath (str): Path to JSON file.

**Attributes:**

- plant_name (str): Name of the biogas plant.
- components (Dict[str, Component]): Dictionary of all plant components.
- connections (List[Connection]): List of connections between components.
- simulation_time (float): Current simulation time in days.


### Connection

```python
from pyadm1.configurator import Connection
```

Represents a connection between two components.

A connection defines a directed link from one component to another,
specifying what type of flow or signal is being transferred.

Attributes:

    from_component (str): Source component ID.
    to_component (str): Target component ID.
    connection_type (str): Type of connection.

Example:

```python
    >>> conn = Connection("digester_1", "chp_1", "gas")
    >>> config = conn.to_dict()
```

**Signature:**

```python
Connection(
    from_component,
    to_component,
    connection_type='default'
)
```

**Methods:**

#### `to_dict()`

```python
to_dict()
```

Serialize to dictionary.

Returns:

    Dict[str, Any]: Dictionary representation of the connection.

**Attributes:**

- from_component (str): Source component ID.
- to_component (str): Target component ID.
- connection_type (str): Type of connection.


### ConnectionManager

```python
from pyadm1.configurator import ConnectionManager
```

Manages connections between components in a biogas plant.

The ConnectionManager handles connection validation, dependency resolution,
and provides utilities for analyzing component relationships.

Attributes:

    connections (List[Connection]): List of all connections.

Example:

```python
    >>> manager = ConnectionManager()
    >>> manager.add_connection(Connection("dig1", "chp1", "gas"))
    >>> deps = manager.get_dependencies("chp1")
    >>> print(deps)  # ['dig1']
```

**Methods:**

#### `add_connection()`

```python
add_connection(connection)
```

Add a connection to the manager.

Args:

    connection (Connection): Connection to add.

Raises:

    ValueError: If connection already exists.

#### `clear()`

```python
clear()
```

Remove all connections.

#### `get_all_connections()`

```python
get_all_connections()
```

Get all connections.

Returns:

    List[Connection]: List of all connections.

#### `get_connected_components()`

```python
get_connected_components(component_id)
```

Get all components connected to the given component (directly or indirectly).

Args:

    component_id (str): Starting component ID.

Returns:

    Set[str]: Set of connected component IDs.

#### `get_connections_from()`

```python
get_connections_from(component_id)
```

Get all connections originating from a component.

Args:

    component_id (str): Component ID.

Returns:

    List[Connection]: List of outgoing connections.

#### `get_connections_to()`

```python
get_connections_to(component_id)
```

Get all connections terminating at a component.

Args:

    component_id (str): Component ID.

Returns:

    List[Connection]: List of incoming connections.

#### `get_dependencies()`

```python
get_dependencies(component_id)
```

Get all components that the given component depends on.

Args:

    component_id (str): Component ID.

Returns:

    List[str]: List of component IDs that this component depends on.

#### `get_dependents()`

```python
get_dependents(component_id)
```

Get all components that depend on the given component.

Args:

    component_id (str): Component ID.

Returns:

    List[str]: List of component IDs that depend on this component.

#### `get_execution_order()`

```python
get_execution_order(component_ids)
```

Determine execution order based on dependencies (topological sort).

Args:

    component_ids (List[str]): List of all component IDs.

Returns:

    List[str]: Component IDs in execution order.

Raises:

    ValueError: If circular dependencies are detected.

#### `has_circular_dependency()`

```python
has_circular_dependency(component_ids)
```

Check if there are circular dependencies.

Args:

    component_ids (List[str]): List of component IDs to check.

Returns:

    bool: True if circular dependencies exist, False otherwise.

#### `remove_connection()`

```python
remove_connection(from_component, to_component, connection_type=None)
```

Remove a connection.

Args:

    from_component (str): Source component ID.
    to_component (str): Target component ID.
    connection_type (Optional[str]): Connection type. If None, removes
        all connections between the components.

Returns:

    bool: True if at least one connection was removed, False otherwise.

#### `to_dict()`

```python
to_dict()
```

Serialize all connections to dictionary.

Returns:

    Dict[str, Any]: Dictionary with connection data.

#### `validate_connections()`

```python
validate_connections(component_ids)
```

Validate all connections and return list of issues.

Args:

    component_ids (List[str]): List of valid component IDs.

Returns:

    List[str]: List of validation error messages. Empty if all valid.

**Attributes:**

- connections (List[Connection]): List of all connections.


### ConnectionType

```python
from pyadm1.configurator import ConnectionType
```

Enumeration of connection types between components.

Attributes:

    LIQUID: Liquid flow connection (substrate, digestate).
    GAS: Biogas flow connection.
    HEAT: Heat transfer connection.
    POWER: Electrical power connection.
    CONTROL: Control signal connection.
    DEFAULT: Generic connection type.

**Signature:**

```python
ConnectionType(
    args,
    kwds
)
```

**Attributes:**

- LIQUID: Liquid flow connection (substrate, digestate).
- GAS: Biogas flow connection.
- HEAT: Heat transfer connection.
- POWER: Electrical power connection.
- CONTROL: Control signal connection.
- DEFAULT: Generic connection type.


