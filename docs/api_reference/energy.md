# Energy Components

Energy Conversion and Storage Components

Components for energy generation, conversion, and storage in biogas plants.

Modules:
    chp: Combined Heat and Power (CHP) units including gas engines and micro-turbines,
        with electrical and thermal efficiency curves, part-load behavior, maintenance
        scheduling, and emissions calculation.

    boiler: Auxiliary heating systems (gas, oil, biomass boilers) for peak heat demand
           and backup heating, with efficiency curves, fuel consumption, and emission
           factors.

    gas_storage: Biogas storage systems including low-pressure (membrane, dome) and
                high-pressure (compressed gas) storage, with pressure control, safety
                management, and capacity utilization.

    flare: Safety gas combustion system for excess biogas or emergency situations,
          with destruction efficiency, emissions calculation, and automatic ignition
          control.

Example:

```python
    >>> from pyadm1.components.energy import CHP, Boiler, GasStorage, Flare
    >>>
    >>> # CHP unit with 500 kW electrical output
    >>> chp = CHP("chp1", P_el_nom=500, eta_el=0.40, eta_th=0.45,
    ...          type="gas_engine")
    >>>
    >>> # Low-pressure gas storage (membrane roof)
    >>> storage = GasStorage("storage1", volume=1000,
    ...                     storage_type="membrane", p_max=0.015)
    >>>
    >>> # Emergency flare
    >>> flare = Flare("flare1", capacity=500, destruction_efficiency=0.98)
```

## Classes

### Boiler

```python
from pyadm1.components.energy import Boiler
```

Boiler component (stub for future implementation).

**Signature:**

```python
Boiler(
    component_id,
    P_th_nom=200.0,
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

#### `set_state()`

```python
set_state(state)
```

Set component state.

#### `step()`

```python
step(t, dt, inputs)
```

#### `to_dict()`

```python
to_dict()
```


### CHP

```python
from pyadm1.components.energy import CHP
```

Combined Heat and Power unit.

Converts biogas to electricity and heat with configurable efficiency.

Attributes:
    P_el_nom (float): Nominal electrical power in kW.
    eta_el (float): Electrical efficiency (0-1).
    eta_th (float): Thermal efficiency (0-1).
    load_factor (float): Current operating point (0-1).

Example:

```python
    >>> chp = CHP("chp1", P_el_nom=500, eta_el=0.40, eta_th=0.45)
    >>> chp.initialize()
    >>> result = chp.step(t=0, dt=1/24, inputs={"Q_ch4": 1000})
```

**Signature:**

```python
CHP(
    component_id,
    P_el_nom=500.0,
    eta_el=0.4,
    eta_th=0.45,
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

Initialize CHP state.

Args:

    initial_state (Optional[Dict[str, Any]]): Initial state with keys:
        - 'load_factor': Initial load factor (0-1)
        If None, uses default initialization.

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

    t (float): Current time in days.
    dt (float): Time step in days.
    inputs (Dict[str, Any]): Input data with keys:
        - 'Q_ch4': Methane flow rate [m³/d] (direct input)
        - 'Q_gas_supplied_m3_per_day': Available biogas from storage [m³/d]
        - 'load_setpoint': Desired load factor [0-1] (optional)

Returns:

    Dict[str, Any]: Output data with keys:
        - 'P_el': Electrical power [kW]
        - 'P_th': Thermal power [kW]
        - 'Q_gas_consumed': Biogas consumption [m³/d]
        - 'Q_gas_out_m3_per_day': Gas demand for connected storages [m³/d]
        - 'Q_ch4_remaining': Remaining methane [m³/d]

#### `to_dict()`

```python
to_dict()
```

Serialize to dictionary.

Returns:

    Dict[str, Any]: Component configuration as dictionary.

**Attributes:**

- P_el_nom (float): Nominal electrical power in kW.
- eta_el (float): Electrical efficiency (0-1).
- eta_th (float): Thermal efficiency (0-1).
- load_factor (float): Current operating point (0-1).
- Example:
- >>> chp = CHP("chp1", P_el_nom=500, eta_el=0.40, eta_th=0.45)
- >>> chp.initialize()
- >>> result = chp.step(t=0, dt=1/24, inputs={"Q_ch4": 1000})


### Flare

```python
from pyadm1.components.energy import Flare
```

Flare component for combusting vented biogas.

    The flare accepts an input `Q_gas_in_m3_per_day` and will combust it.
    It reports `vented_volume_m3` for the current timestep and `cumulative_vented_m3`.

    Parameters
    ----------
    component_id : str
        Unique id for the flare component.
    destruction_efficiency : float
        Fraction of methane destroyed (0..1). Default 0.98.
    name : Optional[str]
        Human readable name.

**Signature:**

```python
Flare(
    component_id,
    destruction_efficiency=0.98,
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

Initialize flare internal state.

        Args:

            initial_state: optional dict with 'cumulative_vented_m3' to restore state.

#### `set_state()`

```python
set_state(state)
```

Set component state.

#### `step()`

```python
step(t, dt, inputs)
```

Process one timestep and combust incoming gas.

        Args:

            t: current simulation time [days]
            dt: timestep length [days]
            inputs: dictionary that may contain:
                - 'Q_gas_in_m3_per_day': inflow (m³/day)
                - 'CH4_fraction': methane fraction in the gas (0..1). Default 0.6

        Returns:

            outputs_data dict with keys:
                - 'vented_volume_m3' (this timestep)
                - 'cumulative_vented_m3'
                - 'CH4_destroyed_m3' (m³ of CH4 destroyed this step)

#### `to_dict()`

```python
to_dict()
```

Serialize flare configuration and state.


### GasStorage

```python
from pyadm1.components.energy import GasStorage
```

Gas storage component.

Initialization args:
    component_id: unique id
    storage_type: 'membrane' | 'dome' | 'compressed'
    capacity_m3: usable gas volume at STP (m^3)
    p_min_bar: minimum operating pressure (bar)
    p_max_bar: maximum safe pressure (bar)
    initial_fill_fraction: initial stored fraction of capacity (0-1)
    name: optional human-readable name

**Signature:**

```python
GasStorage(
    component_id,
    storage_type='membrane',
    capacity_m3=1000.0,
    p_min_bar=0.95,
    p_max_bar=1.05,
    initial_fill_fraction=0.1,
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

Initialize storage state; initial_state may contain stored_volume_m3, pressure_setpoint_bar.

#### `set_state()`

```python
set_state(state)
```

Set component state.

#### `step()`

```python
step(t, dt, inputs)
```

One simulation step.

Inputs dictionary may contain:
    - 'Q_gas_in_m3_per_day'   : gas inflow from digesters/other sources (m^3/day)
    - 'Q_gas_out_m3_per_day'  : requested gas outflow (demand) (m^3/day)
    - 'set_pressure'          : desired pressure setpoint (bar)  (optional)
    - 'vent_to_flare'         : bool, if True allow venting to flare when overpressure (default True)

Returns outputs_data with keys:
    - 'stored_volume_m3'
    - 'pressure_bar'
    - 'utilization' (0-1)
    - 'vented_volume_m3' (this timestep)
    - 'Q_gas_supplied_m3_per_day' (actual supply that was delivered)

#### `to_dict()`

```python
to_dict()
```

Serialize configuration + current state.


