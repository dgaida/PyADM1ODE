# Biological Components

Biological Process Components

Components for biological conversion processes in biogas plants.

Modules:

    digester: Main fermenter component implementing ADM1 model for anaerobic digestion,
             supporting single or multiple fermenters in series/parallel, with flexible
             volume, temperature, and retention time configuration.

    hydrolysis: Pre-treatment tank for hydrolysis-dominated processes, useful for
               substrates with high lignocellulosic content, can operate at different
               temperatures and retention times than main digester.

    separator: Solid-liquid separation component for digestate processing, models
              mechanical (screw press, centrifuge) or gravitational separation with
              configurable separation efficiency and dry matter content.

Example:

    >>> from pyadm1.components.biological import Digester, Hydrolysis, Separator
    >>> from pyadm1.substrates import Feedstock
    >>>
    >>> feedstock = Feedstock(feeding_freq=48)
    >>>
    >>> # Two-stage digestion with hydrolysis pre-treatment
    >>> hydrolysis = Hydrolysis("hydro1", feedstock, V_liq=500, T_ad=318.15)
    >>> digester = Digester("dig1", feedstock, V_liq=2000, T_ad=308.15)
    >>> separator = Separator("sep1", separation_efficiency=0.95)

## Classes

### Digester

```python
from pyadm1.components.biological import Digester
```

Digester component using ADM1 model.

This component wraps the ADM1 implementation and can be
connected to other digesters or components in series/parallel.

Attributes:

    feedstock (Feedstock): Feedstock object for substrate management.
    V_liq (float): Liquid volume in m³.
    V_gas (float): Gas volume in m³.
    T_ad (float): Operating temperature in K.
    adm1 (ADM1): ADM1 model instance.
    simulator (Simulator): Simulator for ADM1.
    adm1_state (List[float]): Current ADM1 state vector (37 dimensions).
    Q_substrates (List[float]): Substrate feed rates in m³/d.

Example:

    >>> feedstock = Feedstock(feeding_freq=48)
    >>> digester = Digester("dig1", feedstock, V_liq=2000, V_gas=300)
    >>> digester.initialize({"adm1_state": initial_state, "Q_substrates": [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]})

**Signature:**

```python
Digester(
    component_id,
    feedstock,
    V_liq=1977.0,
    V_gas=304.0,
    T_ad=308.15,
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

#### `apply_calibration_parameters()`

```python
apply_calibration_parameters(parameters)
```

Apply calibration parameters to this digester.

Stores parameters for use during simulation. These override the
substrate-dependent parameters calculated from feedstock.

Args:
    parameters: Parameter values as {param_name: value}.

Example:
    >>> digester.apply_calibration_parameters({
    ...     'k_dis': 0.55,
    ...     'Y_su': 0.105,
    ...     'k_hyd_ch': 11.0
    ... })

#### `clear_calibration_parameters()`

```python
clear_calibration_parameters()
```

Clear all calibration parameters and revert to default substrate-dependent values.

Example:
    >>> digester.clear_calibration_parameters()

#### `get_calibration_parameters()`

```python
get_calibration_parameters()
```

Get currently applied calibration parameters.

Returns:
    dict: Current calibration parameters as {param_name: value}.

Example:
    >>> params = digester.get_calibration_parameters()
    >>> print(params)
    {'k_dis': 0.55, 'Y_su': 0.105}

#### `get_state()`

```python
get_state()
```

Get current component state.

#### `initialize()`

```python
initialize(initial_state=None)
```

Initialize digester state.

Args:
    initial_state (Optional[Dict[str, Any]]): Initial state with keys:
        - 'adm1_state': ADM1 state vector (37 dims)
        - 'Q_substrates': Substrate feed rates
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
        - 'Q_substrates': Fresh substrate feed rates [m³/d]
        - 'Q_in': Influent from previous digester [m³/d]
        - 'state_in': ADM1 state from previous digester (if connected)

Returns:
    Dict[str, Any]: Output data with keys:
        - 'Q_out': Effluent flow rate [m³/d]
        - 'state_out': ADM1 state vector for next digester
        - 'Q_gas': Biogas production [m³/d]
        - 'Q_ch4': Methane production [m³/d]
        - 'Q_co2': CO2 production [m³/d]
        - 'pH': pH value
        - 'VFA': VFA concentration [g/L]
        - 'TAC': TAC concentration [g CaCO3/L]

#### `to_dict()`

```python
to_dict()
```

Serialize to dictionary.

Returns:
    Dict[str, Any]: Component configuration as dictionary.

**Attributes:**

- feedstock (Feedstock): Feedstock object for substrate management.
- V_liq (float): Liquid volume in m³.
- V_gas (float): Gas volume in m³.
- T_ad (float): Operating temperature in K.
- adm1 (ADM1): ADM1 model instance.
- simulator (Simulator): Simulator for ADM1.
- adm1_state (List[float]): Current ADM1 state vector (37 dimensions).
- Q_substrates (List[float]): Substrate feed rates in m³/d.
- Example:
- >>> feedstock = Feedstock(feeding_freq=48)
- >>> digester = Digester("dig1", feedstock, V_liq=2000, V_gas=300)
- >>> digester.initialize({"adm1_state": initial_state, "Q_substrates": [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]})


### Hydrolysis

```python
from pyadm1.components.biological import Hydrolysis
```

Hydrolysis tank component (stub for future implementation).

**Signature:**

```python
Hydrolysis(
    component_id,
    feedstock,
    V_liq=500.0,
    T_ad=318.15,
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


### Separator

```python
from pyadm1.components.biological import Separator
```

Separator component (stub for future implementation).

**Signature:**

```python
Separator(
    component_id,
    separation_efficiency=0.95,
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


