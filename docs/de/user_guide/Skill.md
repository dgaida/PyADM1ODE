# PyADM1ODE Simulation Model Creation Skill

This document provides the full API documentation for the classes and methods required to build a PyADM1ODE biogas plant simulation model.

## Feedstock

```python
Feedstock
```

Computes ADM1 influent concentrations from substrate characterization.

Accepts either a single substrate (``SubstrateParams``, XML path, or bare
XML stem ID) or a list of substrates for co-digestion.  In multi-substrate
mode the influent DataFrame is generated from a volumetric-flow-weighted
blend of per-substrate concentrations.

Usage (single substrate)
------------------------
>>> from pyadm1 import Feedstock
>>> fs = Feedstock("maize_silage_milk_ripeness", feeding_freq=48, total_simtime=60)
>>> df = fs.get_influent_dataframe(Q=15.0)

Usage (co-digestion, up to 10 substrates)
-----------------------------------------
>>> fs = Feedstock(
...     ["maize_silage_milk_ripeness", "swine_manure"],
...     feeding_freq=24,
...     total_simtime=160,
... )
>>> Q = [11.4, 6.1, 0, 0, 0, 0, 0, 0, 0, 0]  # m³/d, up to 10 slots
>>> df = fs.get_influent_dataframe(Q=Q)

### Methods for Feedstock

#### __init__

```python
__init__(self, substrates: Union[pyadm1.substrates.feedstock.SubstrateParams, str, pathlib.Path, Sequence[Union[pyadm1.substrates.feedstock.SubstrateParams, str, pathlib.Path]], NoneType] = None, feeding_freq: int = 48, total_simtime: int = 60, simba_q_convention: bool = True) -> None
```

Parameters
----------
substrates : SubstrateParams | str | Path | list of those, optional
    A single substrate object/file path/ID, or a list of substrates
    to co-digest.  When ``None`` (the default), every substrate file
    under ``data/substrates/`` is loaded, ordered by the canonical
    default in :data:`_DEFAULT_SUBSTRATE_ORDER` (frequently-used
    substrates first, then variants; unknown IDs are appended in
    alphabetical order). Handy for demos and tests where the exact
    mix doesn't matter; pass an explicit list whenever the index
    of ``Q`` must be stable across releases.
feeding_freq : int
    Time between feeding events [hours].
total_simtime : int
    Total simulation duration [days].
simba_q_convention : bool, default True
    How to interpret ``Q`` in ``get_influent_dataframe(Q=...)``.

    * ``True`` (default, ADM1da convention): each ``Q_i``  
      [m³/d] is interpreted as a mass-equivalent flow.  Internally
      ``Q_actual_i = Q_input_i · 1000 / ρ_FM_i``.  For liquid
      substrates (TS < 200) this is a no-op (ρ_FM = 1000 by
      convention).  For solid substrates (e.g. maize silage) this
      produces a slightly smaller actual liquid volume.  
    * ``False``: ``Q`` is taken literally as the actual liquid  
      volume added to the reactor [m³/d].

#### actual_Q

```python
actual_Q(self, Q: Union[float, Sequence[float]]) -> List[float]
```

Return per-substrate actual liquid volume flows [m³/d].

Applies the ADM1da mass-to-volume conversion when
``simba_q_convention=True``; otherwise returns *Q* unchanged.

#### blended_concentrations

```python
blended_concentrations(self, Q: Union[float, Sequence[float]]) -> dict
```

Volumetric-flow-weighted influent concentrations (no Q field).

#### blended_density

```python
blended_density(self, Q: Union[float, Sequence[float]]) -> float
```

Volumetric-flow-weighted fresh-matter density [kg/m³].

#### blended_vs_content

```python
blended_vs_content(self, Q: Union[float, Sequence[float]]) -> float
```

Volumetric-flow-weighted VS content [kg VS/m³].

#### bmp_theoretical

```python
bmp_theoretical(self, index: int = 0) -> float
```

Theoretical biomethane potential of the i-th substrate [Nm³ CH₄/t VS].

#### get_influent_dataframe

```python
get_influent_dataframe(self, Q: Union[float, Sequence[float]]) -> pandas.DataFrame
```

Generate an ADM1 influent DataFrame for the full simulation period.

Substrate concentrations are constant in time (steady-state feed
composition assumption).  Pass the result to
``ADM1.set_influent_dataframe()``.

#### header

```python
header(self) -> List[str]
```

Names of ADM1 input stream columns.

#### simtime

```python
simtime(self) -> numpy.ndarray
```

Simulation time array [days].

#### total_cod

```python
total_cod(self, index: int = 0) -> float
```

Total COD concentration of the i-th substrate [kg COD/m³].

#### vs_content

```python
vs_content(self, index: int = 0) -> float
```

Volatile-solids content of the i-th substrate [kg VS/m³].

## BiogasPlant

```python
BiogasPlant
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
    >>> from pyadm1 import Feedstock, BiogasPlant
    >>> from pyadm1.components.biological import Digester
    >>>
    >>> feedstock = Feedstock(["maize_silage_milk_ripeness", "swine_manure"], feeding_freq=24)
    >>> plant = BiogasPlant("My Plant")
    >>> digester = Digester("dig1", feedstock, V_liq=1200, V_gas=216, T_ad=315.15)
    >>> plant.add_component(digester)
    >>> plant.initialize()

### Methods for BiogasPlant

#### __init__

```python
__init__(self, plant_name: str = 'Biogas Plant')
```

Initialize biogas plant.

Args:
    plant_name (str): Name of the plant. Defaults to "Biogas Plant".

#### add_component

```python
add_component(self, component: pyadm1.components.base.Component) -> None
```

Add a component to the plant.

Args:
    component (Component): Component to add to the plant.

Raises:
    ValueError: If component with same ID already exists.

#### add_connection

```python
add_connection(self, connection: pyadm1.configurator.connection_manager.Connection) -> None
```

Add a connection between components.

Args:
    connection (Connection): Connection to add.

Raises:
    ValueError: If source or target component not found.

#### from_json

```python
from_json(filepath: str, feedstock: Optional[pyadm1.substrates.feedstock.Feedstock] = None) -> 'BiogasPlant'
```

Load plant configuration from JSON file.

Args:
    filepath (str): Path to JSON file.
    feedstock (Optional[Feedstock]): Feedstock object for digesters.
        Required if plant has digesters.

Returns:
    BiogasPlant: Loaded plant model.

Raises:
    ValueError: If feedstock is None but plant contains digesters.

#### get_summary

```python
get_summary(self) -> str
```

Get human-readable summary of plant configuration.

Returns:
    str: Summary text with components and connections.

#### initialize

```python
initialize(self) -> None
```

Initialize all components.

Note: Most components auto-initialize in their constructor.
This method is kept for compatibility and to ensure any
components that need explicit initialization are handled.

#### simulate

```python
simulate(self, duration: float, dt: float = 0.041666666666666664, save_interval: Optional[float] = None) -> List[Dict[str, Any]]
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
    >>> results = plant.simulate(duration=30, dt=1/24, save_interval=1.0)
    >>> print(f"Simulated {len(results)} time points")

#### step

```python
step(self, dt: float) -> Dict[str, Dict[str, Any]]
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

#### to_json

```python
to_json(self, filepath: str) -> None
```

Save plant configuration to JSON file.

Args:
    filepath (str): Path to JSON file.

## PlantConfigurator

```python
PlantConfigurator
```

High-level configurator for building biogas plants.

Provides convenient methods for adding components with sensible defaults
and automatic setup of common configurations (gas storage attached to
digesters, flare attached to CHP, etc.).

### Methods for PlantConfigurator

#### __init__

```python
__init__(self, plant: pyadm1.configurator.plant_builder.BiogasPlant, feedstock: pyadm1.substrates.feedstock.Feedstock)
```

Parameters
----------
plant : BiogasPlant
    Plant instance to configure.
feedstock : Feedstock
    Feedstock used by all digesters added through this configurator.

#### add_chp

```python
add_chp(self, chp_id: str, P_el_nom: float = 500.0, eta_el: float = 0.4, eta_th: float = 0.45, name: Optional[str] = None) -> pyadm1.components.energy.chp.CHP
```

Add a CHP unit to the plant.

Automatically creates and connects a safety flare downstream of the CHP.

#### add_digester

```python
add_digester(self, digester_id: str, V_liq: float = 1050.0, V_gas: float = 150.0, T_ad: float = 315.15, name: Optional[str] = None, Q_substrates: Optional[list] = None, k_L_a: Optional[float] = None, adm1_state: Optional[list] = None, dynamic_volume: bool = False, initial_fill_fraction: float = 1.0, outflow_time_constant: float = 1.0) -> 'tuple[Digester, str]'
```

Add an ADM1da digester to the plant.

The digester's influent DataFrame, density, and steady-state initial
state are wired automatically from the attached :class:`Feedstock`.
A gas storage is auto-created and connected.

Parameters
----------
digester_id : str
    Unique identifier for this digester.
V_liq : float
    Liquid volume [m³] (default 1050).
V_gas : float
    Gas headspace volume [m³] (default 150).
T_ad : float
    Operating temperature [K] (default 315.15 = 42 °C).
name : str, optional
Q_substrates : list of float, optional
    Substrate feed rates [m³/d], one entry per substrate slot
    (up to 10 slots).
k_L_a : float, optional
    Override of the gas–liquid mass-transfer coefficient [1/d].
adm1_state : list of float, optional
    41-element initial state vector.  When supplied, replaces the
    auto-built steady-state vector.
dynamic_volume : bool, default False
    Enable a dynamic sludge-volume balance
    ``dV/dt = Q_in − Q_out − q_S,loss``, with ``Q_out`` from an
    overflow weir at ``V_liq``. When False, sludge volume stays
    constant.
initial_fill_fraction : float, default 1.0
    Starting sludge fill as a fraction of ``V_liq``. Only used when
    ``dynamic_volume=True``. Set below 1.0 to simulate a partially-
    filled startup transient.
outflow_time_constant : float, default 1.0
    Overflow-weir time constant ``τ_out`` [d]. Only used when
    ``dynamic_volume=True``.

Returns
-------
(Digester, str)
    The created digester and a one-line description of how the
    initial state was determined.

#### add_heating

```python
add_heating(self, heating_id: str, target_temperature: float = 308.15, heat_loss_coefficient: float = 0.5, name: Optional[str] = None) -> pyadm1.components.energy.heating.HeatingSystem
```

Add a heating system to the plant.

#### auto_connect_chp_to_heating

```python
auto_connect_chp_to_heating(self, chp_id: str, heating_id: str) -> None
```

Connect CHP → heating with heat flow.

#### auto_connect_digester_to_chp

```python
auto_connect_digester_to_chp(self, digester_id: str, chp_id: str) -> None
```

Connect digester → gas_storage → chp.

#### connect

```python
connect(self, from_component: str, to_component: str, connection_type: str = 'default') -> pyadm1.configurator.connection_manager.Connection
```

Connect two components.

#### create_single_stage_plant

```python
create_single_stage_plant(self, digester_config: Optional[Dict[str, Any]] = None, chp_config: Optional[Dict[str, Any]] = None, heating_config: Optional[Dict[str, Any]] = None, auto_connect: bool = True) -> Dict[str, Any]
```

Create a complete single-stage plant configuration.

#### create_two_stage_plant

```python
create_two_stage_plant(self, hydrolysis_config: Optional[Dict[str, Any]] = None, digester_config: Optional[Dict[str, Any]] = None, chp_config: Optional[Dict[str, Any]] = None, heating_configs: Optional[list] = None, auto_connect: bool = True) -> Dict[str, Any]
```

Create a two-stage plant: hydrolysis pre-tank → main fermenter.

The hydrolysis stage is just another :class:`Digester` instance with
a higher temperature and shorter HRT — there is no separate
``Hydrolysis`` class.

## Pump

```python
Pump
```

Pump component for material handling in biogas plants.

Models different pump types for substrate feeding, recirculation, and
digestate transfer. Calculates power consumption based on flow rate,
pressure head, and pump efficiency.

Attributes:
    pump_type: Type of pump (centrifugal, progressive_cavity, piston)
    Q_nom: Nominal flow rate [m³/d]
    pressure_head: Pressure head [m] or [bar]
    efficiency: Pump efficiency at nominal point (0-1)
    motor_efficiency: Motor efficiency (0-1)
    fluid_density: Fluid density [kg/m³]
    speed_control: Enable variable speed drive (VSD)
    current_flow: Current flow rate [m³/d]
    is_running: Pump operating state

Example:
    >>> pump = Pump(
    ...     "feed_pump",
    ...     pump_type="progressive_cavity",
    ...     Q_nom=10.0,
    ...     pressure_head=50.0
    ... )
    >>> pump.initialize()
    >>> result = pump.step(0, 1/24, {'Q_setpoint': 8.0})

### Methods for Pump

#### __init__

```python
__init__(self, component_id: str, pump_type: str = 'progressive_cavity', Q_nom: float = 10.0, pressure_head: float = 50.0, efficiency: Optional[float] = None, motor_efficiency: float = 0.9, fluid_density: float = 1020.0, speed_control: bool = True, name: Optional[str] = None)
```

Initialize pump component.

Args:
    component_id: Unique identifier
    pump_type: Type of pump ("centrifugal", "progressive_cavity", "piston")
    Q_nom: Nominal flow rate [m³/d]
    pressure_head: Design pressure head [m]
    efficiency: Pump efficiency (0-1), calculated if None
    motor_efficiency: Motor efficiency (0-1)
    fluid_density: Fluid density [kg/m³]
    speed_control: Enable variable speed drive
    name: Human-readable name

#### add_input

```python
add_input(self, component_id: str) -> None
```

Add an input connection.

#### add_output

```python
add_output(self, component_id: str) -> None
```

Add an output connection.

#### from_dict

```python
from_dict(config: Dict[str, Any]) -> 'Pump'
```

Create pump from dictionary.

Args:
    config: Configuration dictionary

Returns:
    Pump instance

#### get_state

```python
get_state(self) -> Dict[str, Any]
```

Get current component state.

#### initialize

```python
initialize(self, initial_state: Optional[Dict[str, Any]] = None) -> None
```

Initialize pump state.

Args:
    initial_state: Optional initial state dictionary with keys:  
        - 'is_running': Initial pump state  
        - 'current_flow': Initial flow rate [m³/d]  
        - 'operating_hours': Cumulative operating hours  
        - 'energy_consumed': Cumulative energy [kWh]  
        - 'total_volume_pumped': Cumulative volume [m³]  

#### set_state

```python
set_state(self, state: Dict[str, Any]) -> None
```

Set component state.

#### step

```python
step(self, t: float, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]
```

Perform one simulation time step.

Args:
    t: Current time [days]
    dt: Time step [days]
    inputs: Input data with optional keys:  
        - 'Q_setpoint': Desired flow rate [m³/d]  
        - 'Q_actual': Actual flow from connected upstream component [m³/d].  
          If provided, overrides setpoint-based calculation and is used
          directly for power consumption.  
        - 'Q_out': Actual effluent flow from an upstream digester [m³/d].  
          Used when the pump is connected between digesters.  
        - 'enable_pump': Enable/disable pump  
        - 'fluid_density': Fluid density [kg/m³]  
        - 'fluid_viscosity': Fluid viscosity [Pa·s]  
        - 'pressure_head': Required pressure head [m]  

Returns:
    Dict with keys:  
        - 'P_consumed': Power consumption [kW]  
        - 'Q_actual': Actual flow rate [m³/d]  
        - 'is_running': Current running state  
        - 'efficiency': Current operating efficiency  
        - 'pressure_actual': Actual pressure head [m]  
        - 'speed_fraction': Speed as fraction of nominal  

#### to_dict

```python
to_dict(self) -> Dict[str, Any]
```

Serialize pump to dictionary.

Returns:
    Dictionary representation

## Mixer

```python
Mixer
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
    >>> mixer = Mixer(
    ...     "mix1",
    ...     mixer_type="propeller",
    ...     tank_volume=2000,
    ...     mixing_intensity="medium"
    ... )
    >>> mixer.initialize()
    >>> result = mixer.step(0, 1/24, {})

### Methods for Mixer

#### __init__

```python
__init__(self, component_id: str, mixer_type: str = 'propeller', tank_volume: float = 2000.0, tank_diameter: Optional[float] = None, tank_height: Optional[float] = None, mixing_intensity: str = 'medium', power_installed: Optional[float] = None, impeller_diameter: Optional[float] = None, operating_speed: Optional[float] = None, intermittent: bool = True, on_time_fraction: float = 0.25, name: Optional[str] = None)
```

Initialize mixer component.

Args:
    component_id: Unique identifier
    mixer_type: Type of mixer ("propeller", "paddle", "jet")
    tank_volume: Tank liquid volume [m³]
    tank_diameter: Tank diameter [m] (calculated if None)
    tank_height: Tank height [m] (calculated if None)
    mixing_intensity: Intensity level ("low", "medium", "high")
    power_installed: Installed power [kW] (calculated if None)
    impeller_diameter: Impeller diameter [m] (calculated if None)
    operating_speed: Rotational speed [rpm] (calculated if None)
    intermittent: Enable intermittent operation
    on_time_fraction: Fraction of time mixer is on (0-1)
    name: Human-readable name

#### add_input

```python
add_input(self, component_id: str) -> None
```

Add an input connection.

#### add_output

```python
add_output(self, component_id: str) -> None
```

Add an output connection.

#### from_dict

```python
from_dict(config: Dict[str, Any]) -> 'Mixer'
```

Create mixer from dictionary.

Args:
    config: Configuration dictionary

Returns:
    Mixer instance

#### get_state

```python
get_state(self) -> Dict[str, Any]
```

Get current component state.

#### initialize

```python
initialize(self, initial_state: Optional[Dict[str, Any]] = None) -> None
```

Initialize mixer state.

Args:
    initial_state: Optional initial state dictionary with keys:  
        - 'is_running': Mixer running state  
        - 'current_speed_fraction': Speed fraction (0-1)  
        - 'operating_hours': Cumulative operating hours  
        - 'energy_consumed': Cumulative energy [kWh]  

#### set_state

```python
set_state(self, state: Dict[str, Any]) -> None
```

Set component state.

#### step

```python
step(self, t: float, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]
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

#### to_dict

```python
to_dict(self) -> Dict[str, Any]
```

Serialize mixer to dictionary.

Returns:
    Dictionary representation

## SubstrateStorage

```python
SubstrateStorage
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
    >>> storage = SubstrateStorage(
    ...     "silo1",
    ...     storage_type="vertical_silo",
    ...     substrate_type="corn_silage",
    ...     capacity=1000,
    ...     initial_level=600
    ... )
    >>> storage.initialize()
    >>> outputs = storage.step(0, 1, {'withdrawal_rate': 15})

### Methods for SubstrateStorage

#### __init__

```python
__init__(self, component_id: str, storage_type: str = 'vertical_silo', substrate_type: str = 'corn_silage', capacity: float = 1000.0, initial_level: float = 0.0, degradation_rate: Optional[float] = None, temperature: float = 288.15, name: Optional[str] = None)
```

Initialize substrate storage component.

Args:
    component_id: Unique identifier
    storage_type: Type of storage ("vertical_silo", "tank", etc.)
    substrate_type: Substrate category ("corn_silage", "manure_liquid", etc.)
    capacity: Maximum capacity [t or m³]
    initial_level: Initial inventory [t or m³]
    degradation_rate: Quality degradation rate [1/d] (auto-calculated if None)
    temperature: Storage temperature [K]
    name: Human-readable name

#### add_input

```python
add_input(self, component_id: str) -> None
```

Add an input connection.

#### add_output

```python
add_output(self, component_id: str) -> None
```

Add an output connection.

#### from_dict

```python
from_dict(config: Dict[str, Any]) -> 'SubstrateStorage'
```

Create storage from dictionary.

#### get_state

```python
get_state(self) -> Dict[str, Any]
```

Get current component state.

#### initialize

```python
initialize(self, initial_state: Optional[Dict[str, Any]] = None) -> None
```

Initialize storage state.

Args:
    initial_state: Optional initial state with keys:  
        - 'current_level': Inventory level [t or m³]  
        - 'quality_factor': Quality factor (0-1)  
        - 'storage_time': Time stored [days]  
        - 'cumulative_losses': Total losses [t or m³]  

#### set_state

```python
set_state(self, state: Dict[str, Any]) -> None
```

Set component state.

#### step

```python
step(self, t: float, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]
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

#### to_dict

```python
to_dict(self) -> Dict[str, Any]
```

Serialize storage to dictionary.

## Feeder

```python
Feeder
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
    >>> feeder = Feeder(
    ...     "feed1",
    ...     feeder_type="screw",
    ...     Q_max=20,
    ...     substrate_type="solid"
    ... )
    >>> feeder.initialize()
    >>> result = feeder.step(0, 1/24, {'Q_setpoint': 15})

### Methods for Feeder

#### __init__

```python
__init__(self, component_id: str, feeder_type: Optional[str] = None, Q_max: float = 20.0, substrate_type: Optional[str] = None, dosing_accuracy: Optional[float] = None, power_installed: Optional[float] = None, enable_dosing_noise: bool = True, name: Optional[str] = None)
```

Initialize feeder component.

Args:
    component_id: Unique identifier
    feeder_type: Type of feeder ("screw", "progressive_cavity", etc.)
    Q_max: Maximum flow rate [m³/d or t/d]
    substrate_type: Substrate category ("solid", "slurry", "liquid", "fibrous")
    dosing_accuracy: Standard deviation of flow as fraction (auto if None)
    power_installed: Installed power [kW] (auto-calculated if None)
    enable_dosing_noise: Add realistic dosing variance
    name: Human-readable name

#### add_input

```python
add_input(self, component_id: str) -> None
```

Add an input connection.

#### add_output

```python
add_output(self, component_id: str) -> None
```

Add an output connection.

#### from_dict

```python
from_dict(config: Dict[str, Any]) -> 'Feeder'
```

Create feeder from dictionary.

#### get_state

```python
get_state(self) -> Dict[str, Any]
```

Get current component state.

#### initialize

```python
initialize(self, initial_state: Optional[Dict[str, Any]] = None) -> None
```

Initialize feeder state.

Args:
    initial_state: Optional initial state with keys:  
        - 'is_running': Initial operating state  
        - 'current_flow': Initial flow rate [m³/d or t/d]  
        - 'operating_hours': Cumulative operating hours  
        - 'energy_consumed': Cumulative energy [kWh]  
        - 'total_mass_fed': Cumulative mass [t or m³]  

#### set_state

```python
set_state(self, state: Dict[str, Any]) -> None
```

Set component state.

#### step

```python
step(self, t: float, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]
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

#### to_dict

```python
to_dict(self) -> Dict[str, Any]
```

Serialize feeder to dictionary.
