# Mechanical

Mechanical Plant Components

Mechanical equipment for material handling and process control.

Modules:

    pump: Pump models including centrifugal and positive displacement types,
         with power consumption calculation, flow rate control, and characteristic
         curves for different operating points.

    mixer: Agitator and stirrer models for digester mixing, including different
          types (propeller, paddle, jet), with power consumption based on mixing
          intensity, viscosity, and tank geometry.

    valve: Control and safety valves for flow regulation, including proportional
          control valves, on/off valves, and pressure relief valves with configurable
          characteristics and response times.

    heat_exchanger: Heat transfer equipment for substrate pre-heating and digestate
                   cooling, supporting different types (plate, tube), with effectiveness
                   calculation and fouling factor consideration.

Example:

```python
    >>> from pyadm1.components.mechanical import Pump, Mixer, Valve, HeatExchanger
    >>>
    >>> # Substrate feeding pump
    >>> pump = Pump("pump1", pump_type="positive_displacement",
    ...            Q_nom=10, pressure_head=5)
    >>>
    >>> # Digester mixer
    >>> mixer = Mixer("mix1", mixer_type="propeller",
    ...              power=15, mixing_intensity="medium")
    >>>
    >>> # Heat exchanger for substrate pre-heating
    >>> hex = HeatExchanger("hex1", effectiveness=0.7, area=50)
```

## Classes

- [Mixer](#mixer)
- [Pump](#pump)

### Mixer

```python
from pyadm1.components.mechanical import Mixer
```

Mixer/agitator component for biogas digesters.
Models mechanical or hydraulic mixing systems that maintain homogeneity
in anaerobic digesters. Calculates power consumption based on mixer type,
operating conditions, and fluid properties.

Attributes:

    mixer_type: Type of mixer (propeller, paddle, jet)
    tank_volume: Tank volume [m³]
    tank_diameter: Tank diameter [m]
    tank_height: Tank height [m]
    mixing_intensity: Mixing intensity level
    power_installed: Installed mixer power [kW]
    impeller_diameter: Impeller diameter [m]
    operating_speed: Mixer rotational speed [rpm]
    intermittent: Intermittent operation mode
    on_time_fraction: Fraction of time mixer is on (0-1)

Example:

```python
    >>> mixer = Mixer(
    ...     "mix1",
    ...     mixer_type="propeller",
    ...     tank_volume=2000,
    ...     mixing_intensity="medium"
    ... )
    >>> mixer.initialize()
    >>> result = mixer.step(0, 1/24, {})
```

**Signature:**

```python
Mixer(
    component_id,
    mixer_type='propeller',
    tank_volume=2000.0,
    tank_diameter=None,
    tank_height=None,
    mixing_intensity='medium',
    power_installed=None,
    impeller_diameter=None,
    operating_speed=None,
    intermittent=True,
    on_time_fraction=0.25,
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

Initialize mixer state.

Args:

    initial_state: Optional initial state dictionary with keys:
        - 'is_running': Mixer running state
        - 'current_speed_fraction': Speed fraction (0-1)
        - 'operating_hours': Cumulative operating hours
        - 'energy_consumed': Cumulative energy [kWh]

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
        - 'speed_setpoint': Desired speed fraction (0-1)
        - 'enable_mixing': Enable/disable mixer
        - 'fluid_viscosity': Fluid viscosity [Pa·s]
        - 'temperature': Fluid temperature [K]

Returns:

    Dict with keys:
        - 'P_consumed': Power consumption [kW]
        - 'P_average': Time-averaged power [kW]
        - 'is_running': Current running state
        - 'mixing_quality': Mixing quality index (0-1)
        - 'reynolds_number': Reynolds number
        - 'power_number': Power number
        - 'mixing_time': Mixing time [min]
        - 'shear_rate': Average shear rate [1/s]

#### `to_dict()`

```python
to_dict()
```

Serialize mixer to dictionary.

Returns:

    Dictionary representation

**Attributes:**

- mixer_type: Type of mixer (propeller, paddle, jet)
- tank_volume: Tank volume [m³]
- tank_diameter: Tank diameter [m]
- tank_height: Tank height [m]
- mixing_intensity: Mixing intensity level
- power_installed: Installed mixer power [kW]
- impeller_diameter: Impeller diameter [m]
- operating_speed: Mixer rotational speed [rpm]
- intermittent: Intermittent operation mode
- on_time_fraction: Fraction of time mixer is on (0-1)


---------------------------------------
### Pump

```python
from pyadm1.components.mechanical import Pump
```

Pump component for material handling in biogas plants.

Models different pump types for substrate feeding, recirculation, and
digestate transfer. Calculates power consumption based on flow rate,
pressure head, and pump efficiency.

Attributes:

    pump_type: Type of pump (centrifugal, progressive_cavity, piston)
    Q_nom: Nominal flow rate [m³/h]
    pressure_head: Pressure head [m] or [bar]
    efficiency: Pump efficiency at nominal point (0-1)
    motor_efficiency: Motor efficiency (0-1)
    fluid_density: Fluid density [kg/m³]
    speed_control: Enable variable speed drive (VSD)
    current_flow: Current flow rate [m³/h]
    is_running: Pump operating state

Example:

```python
    >>> pump = Pump(
    ...     "feed_pump",
    ...     pump_type="progressive_cavity",
    ...     Q_nom=10.0,
    ...     pressure_head=50.0
    ... )
    >>> pump.initialize()
    >>> result = pump.step(0, 1/24, {'Q_setpoint': 8.0})
```

**Signature:**

```python
Pump(
    component_id,
    pump_type='progressive_cavity',
    Q_nom=10.0,
    pressure_head=50.0,
    efficiency=None,
    motor_efficiency=0.9,
    fluid_density=1020.0,
    speed_control=True,
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

Initialize pump state.

Args:

    initial_state: Optional initial state dictionary with keys:
        - 'is_running': Initial pump state
        - 'current_flow': Initial flow rate [m³/h]
        - 'operating_hours': Cumulative operating hours
        - 'energy_consumed': Cumulative energy [kWh]
        - 'total_volume_pumped': Cumulative volume [m³]

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
        - 'Q_setpoint': Desired flow rate [m³/h]
        - 'enable_pump': Enable/disable pump
        - 'fluid_density': Fluid density [kg/m³]
        - 'fluid_viscosity': Fluid viscosity [Pa·s]
        - 'pressure_head': Required pressure head [m]

Returns:

    Dict with keys:
        - 'P_consumed': Power consumption [kW]
        - 'Q_actual': Actual flow rate [m³/h]
        - 'is_running': Current running state
        - 'efficiency': Current operating efficiency
        - 'pressure_actual': Actual pressure head [m]
        - 'speed_fraction': Speed as fraction of nominal

#### `to_dict()`

```python
to_dict()
```

Serialize pump to dictionary.

Returns:

    Dictionary representation

**Attributes:**

- pump_type: Type of pump (centrifugal, progressive_cavity, piston)
- Q_nom: Nominal flow rate [m³/h]
- pressure_head: Pressure head [m] or [bar]
- efficiency: Pump efficiency at nominal point (0-1)
- motor_efficiency: Motor efficiency (0-1)
- fluid_density: Fluid density [kg/m³]
- speed_control: Enable variable speed drive (VSD)
- current_flow: Current flow rate [m³/h]
- is_running: Pump operating state


---------------------------------------
