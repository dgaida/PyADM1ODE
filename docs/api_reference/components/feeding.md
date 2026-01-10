# Feeding

Substrate Feeding and Storage Components

Components for substrate handling, storage, and dosing into digesters.

Modules:

    substrate_storage: Storage facilities for different substrate types (silos for
                      solid substrates, tanks for liquid substrates), with inventory
                      management, quality degradation over time, and capacity monitoring.

    feeder: Automated dosing systems including screw feeders for solid substrates,
           progressive cavity pumps for liquid/slurry substrates, and piston feeders
           for fibrous materials, with flow rate control and dosing accuracy.

    mixer_wagon: Mobile substrate preparation systems for mixing multiple substrates
                before feeding, with mixing efficiency, recipe management, and
                substrate homogenization.

Example:

```python
    >>> from pyadm1.components.feeding import SubstrateStorage, Feeder, MixerWagon
    >>>
    >>> # Corn silage storage
    >>> storage = SubstrateStorage("silo1", substrate_type="corn_silage",
    ...                           capacity=1000, current_level=800)
    >>>
    >>> # Screw feeder for solid substrates
    >>> feeder = Feeder("feed1", feeder_type="screw",
    ...                Q_max=20, substrate_type="solid")
    >>>
    >>> # Mixer wagon for substrate preparation
    >>> wagon = MixerWagon("wagon1", capacity=30, mixing_time=15)
```

## Classes

- [Feeder](#feeder)
- [SubstrateStorage](#substratestorage)

### Feeder

```python
from pyadm1.components.feeding import Feeder
```

Feeder component for automated substrate dosing.

Models feeding systems that transfer substrates from storage to digesters.
Includes realistic operational characteristics like dosing accuracy,
capacity limits, and power consumption.

Attributes:

    feeder_type: Type of feeding system
    Q_max: Maximum flow rate [m³/d or t/d]
    substrate_type: Physical category of substrate
    dosing_accuracy: Accuracy of flow control (std dev as fraction)
    power_installed: Installed motor power [kW]
    current_flow: Current actual flow rate [m³/d or t/d]
    is_running: Operating state

Example:

```python
    >>> feeder = Feeder(
    ...     "feed1",
    ...     feeder_type="screw",
    ...     Q_max=20,
    ...     substrate_type="solid"
    ... )
    >>> feeder.initialize()
    >>> result = feeder.step(0, 1/24, {'Q_setpoint': 15})
```

**Signature:**

```python
Feeder(
    component_id,
    feeder_type=None,
    Q_max=20.0,
    substrate_type='solid',
    dosing_accuracy=None,
    power_installed=None,
    enable_dosing_noise=True,
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

Initialize feeder state.

Args:

    initial_state: Optional initial state with keys:
        - 'is_running': Initial operating state
        - 'current_flow': Initial flow rate [m³/d or t/d]
        - 'operating_hours': Cumulative operating hours
        - 'energy_consumed': Cumulative energy [kWh]
        - 'total_mass_fed': Cumulative mass [t or m³]

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

Args:

    t: Current time [days]
    dt: Time step [days]
    inputs: Input data with optional keys:
        - 'Q_setpoint': Desired flow rate [m³/d or t/d]
        - 'enable_feeding': Enable/disable feeder
        - 'substrate_available': Amount available in storage [t or m³]
        - 'speed_setpoint': Desired speed fraction (0-1)

Returns:

    Dict with keys:
        - 'Q_actual': Actual flow rate [m³/d or t/d]
        - 'is_running': Current operating state
        - 'load_factor': Operating load (0-1)
        - 'P_consumed': Power consumption [kW]
        - 'blockage_detected': Blockage alarm
        - 'dosing_error': Deviation from setpoint [%]
        - 'speed_fraction': Current speed fraction

#### `to_dict()`

```python
to_dict()
```

Serialize feeder to dictionary.

**Attributes:**

- feeder_type: Type of feeding system
- Q_max: Maximum flow rate [m³/d or t/d]
- substrate_type: Physical category of substrate
- dosing_accuracy: Accuracy of flow control (std dev as fraction)
- power_installed: Installed motor power [kW]
- current_flow: Current actual flow rate [m³/d or t/d]
- is_running: Operating state


---------------------------------------
### SubstrateStorage

```python
from pyadm1.components.feeding import SubstrateStorage
```

Storage facility component for biogas plant substrates.

Models storage of different substrate types with inventory tracking,
quality degradation, and capacity management. Supports both solid
(silage, solid manure) and liquid (liquid manure, slurry) substrates.

Attributes:

    storage_type: Type of storage facility
    substrate_type: Category of substrate stored
    capacity: Maximum storage capacity [t or m³]
    current_level: Current inventory level [t or m³]
    quality_factor: Current quality relative to fresh (0-1)
    degradation_rate: Quality degradation rate [1/d]
    density: Substrate bulk density [kg/m³]
    dry_matter: Dry matter content [%]
    vs_content: Volatile solids [% of DM]

Example:

```python
    >>> storage = SubstrateStorage(
    ...     "silo1",
    ...     storage_type="vertical_silo",
    ...     substrate_type="corn_silage",
    ...     capacity=1000,
    ...     initial_level=600
    ... )
    >>> storage.initialize()
    >>> outputs = storage.step(0, 1, {'withdrawal_rate': 15})
```

**Signature:**

```python
SubstrateStorage(
    component_id,
    storage_type='vertical_silo',
    substrate_type='corn_silage',
    capacity=1000.0,
    initial_level=0.0,
    degradation_rate=None,
    temperature=288.15,
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

Initialize storage state.

Args:

    initial_state: Optional initial state with keys:
        - 'current_level': Inventory level [t or m³]
        - 'quality_factor': Quality factor (0-1)
        - 'storage_time': Time stored [days]
        - 'cumulative_losses': Total losses [t or m³]

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

Args:

    t: Current time [days]
    dt: Time step [days]
    inputs: Input data with optional keys:
        - 'withdrawal_rate': Withdrawal rate [t/d or m³/d]
        - 'refill_amount': Amount to add [t or m³]
        - 'refill_quality': Quality of refill (0-1)
        - 'temperature': Ambient/storage temperature [K]

Returns:

    Dict with keys:
        - 'current_level': Current inventory [t or m³]
        - 'utilization': Fill level (0-1)
        - 'quality_factor': Current quality (0-1)
        - 'available_mass': Usable inventory [t or m³]
        - 'degradation_rate': Current degradation rate [1/d]
        - 'losses_this_step': Mass lost this timestep [t or m³]
        - 'withdrawn_this_step': Mass withdrawn [t or m³]
        - 'is_empty': Storage empty flag
        - 'is_full': Storage full flag

#### `to_dict()`

```python
to_dict()
```

Serialize storage to dictionary.

**Attributes:**

- storage_type: Type of storage facility
- substrate_type: Category of substrate stored
- capacity: Maximum storage capacity [t or m³]
- current_level: Current inventory level [t or m³]
- quality_factor: Current quality relative to fresh (0-1)
- degradation_rate: Quality degradation rate [1/d]
- density: Substrate bulk density [kg/m³]
- dry_matter: Dry matter content [%]
- vs_content: Volatile solids [% of DM]


---------------------------------------
