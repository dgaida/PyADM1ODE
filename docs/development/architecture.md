# PyADM1ODE Architecture

This document describes the architecture of PyADM1ODE, a modular framework for biogas plant simulation and optimization.

## Table of Contents

1. [Overview](#overview)
2. [Design Principles](#design-principles)
3. [Core Architecture](#core-architecture)
4. [Module Organization](#module-organization)
5. [Component System](#component-system)
6. [Simulation Engine](#simulation-engine)
7. [Data Flow](#data-flow)
8. [Extension Points](#extension-points)
9. [Integration with External Systems](#integration-with-external-systems)

## Overview

PyADM1ODE is designed as a modular, extensible framework following object-oriented principles. The architecture separates concerns into distinct layers:

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                     │
│  - CLI/API                                                  │
│  - MCP Server (LLM Integration)                            │
│  - Jupyter Notebooks                                        │
└─────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  Configuration Layer                        │
│  - PlantConfigurator (High-level API)                      │
│  - PlantBuilder (Component assembly)                       │
│  - ConnectionManager (Component connections)               │
│  - Templates (Pre-defined configurations)                  │
└─────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Component Layer                           │
│  - Biological (Digesters, Hydrolysis, Separators)         │
│  - Energy (CHP, Heating, Gas Storage, Flare)              │
│  - Mechanical (Pumps, Mixers)                              │
│  - Feeding (Storage, Feeders)                              │
│  - Sensors (Physical, Chemical, Gas)                       │
└─────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Simulation Layer                          │
│  - Simulator (Single run)                                  │
│  - ParallelSimulator (Multi-scenario)                      │
│  - Time series management                                  │
│  - Result collection and analysis                          │
└─────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      Core Layer                             │
│  - ADM1 (ODE system)                                       │
│  - ADMParams (Model parameters)                            │
│  - ODESolver (Numerical integration)                       │
│  - Substrate characterization                              │
└─────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   External Dependencies                     │
│  - scipy (ODE solver)                                      │
│  - numpy (Numerical operations)                            │
│  - pythonnet (C# interop)                                  │
│  - C# DLLs (Substrate database)                            │
└─────────────────────────────────────────────────────────────┘
```

## Design Principles

### 1. Modularity

Each component is self-contained and can be used independently:

```python
# Components work standalone
from pyadm1.components.biological import Digester
from pyadm1.substrates import Feedstock

feedstock = Feedstock(feeding_freq=48)
digester = Digester("dig1", feedstock, V_liq=2000)
digester.initialize()
result = digester.step(t=0, dt=1/24, inputs={})
```

### 2. Separation of Concerns

- **Core**: ADM1 model implementation (pure science/math)
- **Components**: Engineering equipment models (plant hardware)
- **Configurator**: Plant assembly and connections (system design)
- **Simulation**: Time integration and result management (execution)

### 3. Extensibility

New components can be added without modifying existing code:

```python
from pyadm1.components.base import Component, ComponentType

class CustomComponent(Component):
    """User-defined component."""

    def __init__(self, component_id, custom_param, name=None):
        super().__init__(component_id, ComponentType.CUSTOM, name)
        self.custom_param = custom_param

    def step(self, t, dt, inputs):
        """Implement custom behavior."""
        # Custom logic here
        return {'output': self.custom_param * dt}

    def initialize(self, initial_state=None):
        """Initialize component."""
        pass

    def to_dict(self):
        """Serialize to dictionary."""
        return {
            'component_id': self.component_id,
            'component_type': self.component_type.value,
            'custom_param': self.custom_param
        }

# Register and use
from pyadm1.components import ComponentRegistry
registry = ComponentRegistry()
registry.register("CustomComponent", CustomComponent)
```

### 4. Type Safety

Type hints throughout for better IDE support and error prevention:

```python
from typing import Dict, List, Optional, Any

def simulate(
    duration: float,
    dt: float = 1.0/24.0,
    save_interval: Optional[float] = None
) -> List[Dict[str, Any]]:
    """Type-safe simulation method."""
    pass
```

### 5. Configuration as Code

Plant configurations are serializable and version-controllable:

```python
# Configuration is pure data
config = {
    "plant_name": "My Plant",
    "components": [...],
    "connections": [...]
}

# Can be saved/loaded
plant.to_json("config.json")
plant = BiogasPlant.from_json("config.json", feedstock)
```

## Core Architecture

### ADM1 Core

The heart of PyADM1ODE is the ADM1 implementation:

```python
class ADM1:
    """Main ADM1 model class.

    Implements the complete ADM1 ODE system with 37 state variables.
    Pure ODE formulation (no DAEs) for numerical stability.

    Attributes:
        V_liq: Liquid volume [m³]
        V_gas: Gas volume [m³]
        T_ad: Operating temperature [K]
        feedstock: Feedstock object
        state: Current ADM1 state vector (37 elements)
        params: Model parameters
    """

    def __init__(self, feedstock, V_liq=1977.0, V_gas=304.0, T_ad=308.15):
        """Initialize ADM1 model."""
        self.feedstock = feedstock
        self.V_liq = V_liq
        self.V_gas = V_gas
        self.T_ad = T_ad

        # Get parameters
        self.params = ADMParams.get_all_params(
            R=0.08314,
            T_base=298.15,
            T_ad=T_ad
        )

        # Initialize state
        self.state = None

    def ADM1_ODE(self, t, state_zero):
        """Calculate derivatives dy/dt.

        This is the core ODE function called by the solver.

        Args:
            t: Current time [days] (not used, autonomous system)
            state_zero: Current state vector (37 elements)

        Returns:
            Tuple of 37 derivatives
        """
        # Unpack state variables
        S_su, S_aa, S_fa, S_va, S_bu, S_pro, S_ac, S_h2, S_ch4 = state_zero[0:9]
        S_IC, S_IN, S_I = state_zero[9:12]
        X_xc, X_ch, X_pr, X_li, X_su, X_aa = state_zero[12:18]
        X_fa, X_c4, X_pro, X_ac, X_h2, X_I = state_zero[18:24]
        S_cat, S_an, S_va_ion, S_bu_ion, S_pro_ion = state_zero[24:29]
        S_ac_ion, S_hco3_ion, S_nh3, S_nh4_ion, S_gas_h2 = state_zero[29:34]
        S_gas_ch4, S_gas_co2, S_H_ion = state_zero[34:37]

        # Calculate process rates
        rates = BiochemicalProcesses.calculate_process_rates(
            state_zero,
            self._inhibitions,
            self.params,
            self._substrate_params
        )

        # Calculate acid-base rates
        acid_base_rates = BiochemicalProcesses.calculate_acid_base_rates(
            state_zero,
            self.params
        )

        # Calculate gas transfer rates
        gas_transfer_rates = BiochemicalProcesses.calculate_gas_transfer_rates(
            state_zero,
            self.params,
            self.params['R'] * self.T_ad,
            self.V_liq,
            self.V_gas
        )

        # Assemble derivatives (simplified - actual implementation has all 37)
        dS_su = rates[0] - rates[4] + self.influent_rate * (S_su_in - S_su)
        # ... (all 37 derivatives)

        return (dS_su, dS_aa, dS_fa, ...)  # All 37 derivatives
```

**Key Design Decisions:**

1. **Pure ODE System**: No differential-algebraic equations (DAEs) for better numerical stability
2. **Modular Calculation**: Process rates, inhibitions, and gas transfer separated
3. **Parameter Management**: Static parameter class for easy modification
4. **Substrate Integration**: C# DLL interop for substrate characterization

### Component Base Class

All components inherit from a common base:

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from enum import Enum

class ComponentType(Enum):
    """Enumeration of component types."""
    BIOLOGICAL = "biological"
    MECHANICAL = "mechanical"
    ENERGY = "energy"
    FEEDING = "feeding"
    SENSOR = "sensor"
    CONTROL = "control"
    CUSTOM = "custom"

class Component(ABC):
    """Abstract base class for all biogas plant components.

    All components must implement:
    - step(): Execute one simulation time step
    - initialize(): Set up initial state
    - to_dict(): Serialize to dictionary

    Attributes:
        component_id: Unique identifier
        component_type: Type of component
        name: Human-readable name
        inputs: Connected input components
        outputs: Connected output components
        state: Current component state
        outputs_data: Latest output data
    """

    def __init__(self, component_id: str, component_type: ComponentType,
                 name: Optional[str] = None):
        """Initialize component.

        Args:
            component_id: Unique identifier
            component_type: Type of component
            name: Optional human-readable name
        """
        self.component_id = component_id
        self.component_type = component_type
        self.name = name or component_id
        self.inputs: List[str] = []
        self.outputs: List[str] = []
        self.state: Dict[str, Any] = {}
        self.outputs_data: Dict[str, Any] = {}

    @abstractmethod
    def step(self, t: float, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Perform one simulation time step.

        Args:
            t: Current simulation time [days]
            dt: Time step size [days]
            inputs: Input data from connected components

        Returns:
            Output data to be passed to connected components
        """
        pass

    @abstractmethod
    def initialize(self, initial_state: Optional[Dict[str, Any]] = None):
        """Initialize component state.

        Args:
            initial_state: Optional initial state values
        """
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize component to dictionary for JSON export.

        Returns:
            Component configuration as dictionary
        """
        pass

    def add_input(self, component_id: str):
        """Add an input connection."""
        if component_id not in self.inputs:
            self.inputs.append(component_id)

    def add_output(self, component_id: str):
        """Add an output connection."""
        if component_id not in self.outputs:
            self.outputs.append(component_id)

    def get_state(self) -> Dict[str, Any]:
        """Get current component state."""
        return self.state.copy()

    def set_state(self, state: Dict[str, Any]):
        """Set component state."""
        self.state = state.copy()
```

## Module Organization

### pyadm1/core/

**Purpose**: Core ADM1 model implementation

**Files**:
- `adm1.py`: Main ADM1 class with ODE system
- `adm_params.py`: Static parameter definitions
- `adm_equations.py`: Process rates, inhibitions, gas transfer
- `solver.py`: ODE solver wrapper

**Design Pattern**: Functional decomposition for mathematical clarity

```python
# Clear separation of concerns
from pyadm1.core.adm_equations import BiochemicalProcesses, InhibitionFunctions

# Calculate inhibitions
inhibitions = InhibitionFunctions.calculate_inhibition_factors(...)

# Calculate process rates
rates = BiochemicalProcesses.calculate_process_rates(state, inhibitions, ...)
```

### pyadm1/components/

**Purpose**: Modular plant components

**Structure**:
```
components/
├── __init__.py          # Base classes, registry
├── base.py              # Component base class
├── registry.py          # Dynamic component loading
├── biological/          # Digesters, hydrolysis, separators
├── mechanical/          # Pumps, mixers, valves
├── energy/              # CHP, heating, gas storage, flare
├── feeding/             # Storage, feeders, mixer wagons
└── sensors/             # Physical, chemical, gas sensors
```

**Design Pattern**: Plugin architecture with registry

```python
from pyadm1.components import ComponentRegistry

registry = ComponentRegistry()

# Components self-register on import
from pyadm1.components.biological import Digester
from pyadm1.components.energy import CHP

# Can create by name
component = registry.create("Digester", "dig1", feedstock=feedstock)
```

### pyadm1/configurator/

**Purpose**: Plant model building and configuration

**Files**:
- `plant_builder.py`: BiogasPlant class (low-level)
- `plant_configurator.py`: PlantConfigurator (high-level API)
- `connection_manager.py`: Component connections
- `validation.py`: Model validation
- `templates/`: Pre-defined plant layouts

**Design Pattern**: Builder pattern with fluent API

```python
from pyadm1.configurator import PlantConfigurator

configurator = PlantConfigurator(plant, feedstock)

# Fluent API for plant construction
configurator \
    .add_digester("dig1", V_liq=2000, Q_substrates=[15, 10, 0, 0, 0, 0, 0, 0, 0, 0]) \
    .add_chp("chp1", P_el_nom=500) \
    .add_heating("heat1", target_temperature=308.15) \
    .auto_connect_digester_to_chp("dig1", "chp1") \
    .auto_connect_chp_to_heating("chp1", "heat1")
```

### pyadm1/simulation/

**Purpose**: Simulation execution and result management

**Files**:
- `simulator.py`: Single simulation runs
- `parallel.py`: Parallel multi-scenario simulation
- `scenarios.py`: Scenario management
- `time_series.py`: Time series data handling
- `results.py`: Result analysis and export

**Design Pattern**: Strategy pattern for different simulation modes

```python
from pyadm1.simulation import Simulator, ParallelSimulator

# Single run
simulator = Simulator(adm1)
result = simulator.simulate_AD_plant([0, 30], initial_state)

# Parallel scenarios
parallel = ParallelSimulator(adm1, n_workers=4)
results = parallel.run_scenarios(scenarios, duration=30, initial_state=state)
```

### pyadm1/substrates/

**Purpose**: Substrate management and characterization

**Files**:
- `feedstock.py`: Main Feedstock class
- `substrate_db.py`: Substrate database interface
- `xml_loader.py`: XML substrate file parser
- `characterization.py`: Substrate characterization methods

**Design Pattern**: Facade pattern for C# DLL interaction

```python
class Feedstock:
    """Facade for C# substrate DLLs.

    Hides complexity of .NET interop behind simple Python API.
    """

    def __init__(self, feeding_freq, substrate_xml='substrate_gummersbach.xml'):
        """Initialize with C# DLL."""
        # Load C# assemblies
        clr.AddReference(str(dll_path / "substrates.dll"))
        from substrates import Substrates as CSharpSubstrates

        self._substrates = CSharpSubstrates(substrate_xml)

    def get_influent_dataframe(self, Q):
        """Get ADM1 input stream as DataFrame.

        Wraps C# method calls in Pythonic interface.
        """
        # Call C# methods
        influent = self._substrates.getInfluent(Q, ...)

        # Convert to pandas DataFrame
        return pd.DataFrame(influent, columns=self.header())
```

## Component System

### Component Lifecycle

```python
# 1. Creation
component = Digester("dig1", feedstock, V_liq=2000)

# 2. Configuration
component.Q_substrates = [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]

# 3. Connection
plant.add_component(component)
configurator.connect("dig1", "chp1", "gas")

# 4. Initialization
component.initialize()

# 5. Simulation loop
for t in time_steps:
    result = component.step(t, dt, inputs)

# 6. Serialization
config = component.to_dict()
```

### Component Connections

Connections are typed and validated:

```python
class Connection:
    """Directed connection between components.

    Attributes:
        from_component: Source component ID
        to_component: Target component ID
        connection_type: Type of flow (liquid, gas, heat, power)
    """

    def __init__(self, from_component: str, to_component: str,
                 connection_type: str = "default"):
        self.from_component = from_component
        self.to_component = to_component
        self.connection_type = connection_type

class ConnectionManager:
    """Manages connections between components."""

    def add_connection(self, connection: Connection):
        """Add and validate connection."""
        # Type checking
        if connection.connection_type not in valid_types:
            raise ValueError(f"Invalid connection type: {connection.connection_type}")

        # Store connection
        self.connections.append(connection)

    def get_execution_order(self, component_ids: List[str]) -> List[str]:
        """Determine execution order via topological sort."""
        # Build dependency graph
        graph = self._build_dependency_graph(component_ids)

        # Topological sort
        return self._topological_sort(graph)
```

### Three-Pass Execution Model

PyADM1ODE uses a three-pass execution for realistic gas flow:

```python
def step(self, dt: float) -> Dict[str, Dict[str, Any]]:
    """Execute one simulation time step with three-pass model.

    Pass 1: Digesters produce gas → Gas storages
    Pass 2: Gas storages update (no consumption yet)
    Pass 3: CHPs consume gas → Storages supply → Flares vent excess
    """
    results = {}

    # Pass 1: Execute digesters
    for digester_id in self.digesters:
        component = self.components[digester_id]
        inputs = self._get_inputs(digester_id, results)
        result = component.step(self.simulation_time, dt, inputs)
        results[digester_id] = result

        # Gas to storage
        storage_id = f"{digester_id}_storage"
        if storage_id in self.components:
            storage_inputs = {'Q_gas_in_m3_per_day': result['Q_gas']}
            storage_result = self.components[storage_id].step(
                self.simulation_time, dt, storage_inputs
            )
            results[storage_id] = storage_result

    # Pass 2: Execute CHPs (first pass - determine demand)
    for chp_id in self.chps:
        component = self.components[chp_id]
        inputs = self._get_inputs(chp_id, results)
        result = component.step(self.simulation_time, dt, inputs)
        results[chp_id] = result

    # Pass 3: Execute storages (supply to CHPs)
    for storage_id in self.storages:
        if storage_id in results:
            # Already executed in Pass 1
            continue

        # Standalone storage
        component = self.components[storage_id]
        inputs = self._get_inputs(storage_id, results)
        result = component.step(self.simulation_time, dt, inputs)
        results[storage_id] = result

    # Re-execute CHPs with actual gas supply
    for chp_id in self.chps:
        component = self.components[chp_id]
        inputs = self._get_inputs(chp_id, results)
        result = component.step(self.simulation_time, dt, inputs)
        results[chp_id] = result

    # Execute remaining components
    for component_id in self.execution_order:
        if component_id not in results:
            component = self.components[component_id]
            inputs = self._get_inputs(component_id, results)
            result = component.step(self.simulation_time, dt, inputs)
            results[component_id] = result

    return results
```

## Simulation Engine

### Single Simulation

```python
class Simulator:
    """Handles single ADM1 simulation runs.

    Uses ODESolver for numerical integration.

    Attributes:
        adm1: ADM1 model instance
        solver: ODE solver instance
    """

    def __init__(self, adm1: ADM1, solver: Optional[ODESolver] = None):
        """Initialize simulator."""
        self.adm1 = adm1
        self.solver = solver or ODESolver()

    def simulate_AD_plant(self, tstep: List[float],
                          state_zero: List[float]) -> List[float]:
        """Simulate ADM1 for time span.

        Args:
            tstep: [t_start, t_end] in days
            state_zero: Initial ADM1 state (37 elements)

        Returns:
            Final ADM1 state after simulation
        """
        # Create ODE function
        def ode_func(t, y):
            return self.adm1.ADM1_ODE(t, y)

        # Solve ODE system
        result = self.solver.solve(
            ode_func,
            t_span=tstep,
            y0=state_zero
        )

        # Track process values
        for i, t in enumerate(result.t):
            state = result.y[:, i]
            self.adm1.print_params_at_current_state(state)

        return result.y[:, -1]  # Final state
```

### Parallel Simulation

```python
from multiprocessing import Pool
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class ScenarioResult:
    """Result from a single scenario."""
    scenario_id: int
    parameters: Dict[str, Any]
    success: bool
    duration: float
    final_state: Optional[List[float]]
    metrics: Dict[str, float]
    error: Optional[str]
    execution_time: float

class ParallelSimulator:
    """Parallel execution of multiple scenarios.

    Uses multiprocessing for CPU-bound simulation tasks.

    Attributes:
        adm1: Base ADM1 model (copied per worker)
        n_workers: Number of parallel workers
    """

    def __init__(self, adm1: ADM1, n_workers: Optional[int] = None):
        """Initialize parallel simulator."""
        self.adm1 = adm1
        self.n_workers = n_workers or cpu_count()

    def run_scenarios(self, scenarios: List[Dict[str, Any]],
                      duration: float,
                      initial_state: List[float],
                      **kwargs) -> List[ScenarioResult]:
        """Run multiple scenarios in parallel.

        Args:
            scenarios: List of parameter dictionaries
            duration: Simulation duration [days]
            initial_state: Initial ADM1 state

        Returns:
            List of ScenarioResult objects
        """
        # Create work items
        work_items = [
            (i, scenario, duration, initial_state, self.adm1, kwargs)
            for i, scenario in enumerate(scenarios)
        ]

        # Execute in parallel
        with Pool(processes=self.n_workers) as pool:
            results = pool.starmap(self._run_single_scenario, work_items)

        return results

    @staticmethod
    def _run_single_scenario(scenario_id, parameters, duration,
                             initial_state, adm1_base, kwargs):
        """Run single scenario (called by worker process)."""
        import time
        start_time = time.time()

        try:
            # Create ADM1 copy for this worker
            adm1 = deepcopy(adm1_base)

            # Apply parameters
            for param, value in parameters.items():
                if param == 'Q':
                    adm1.create_influent(value, 0)
                else:
                    adm1.set_calibration_parameters({param: value})

            # Simulate
            simulator = Simulator(adm1)
            final_state = simulator.simulate_AD_plant([0, duration], initial_state)

            # Compute metrics
            metrics = self._compute_metrics(final_state, adm1)

            return ScenarioResult(
                scenario_id=scenario_id,
                parameters=parameters,
                success=True,
                duration=duration,
                final_state=final_state,
                metrics=metrics,
                error=None,
                execution_time=time.time() - start_time
            )

        except Exception as e:
            return ScenarioResult(
                scenario_id=scenario_id,
                parameters=parameters,
                success=False,
                duration=duration,
                final_state=None,
                metrics={},
                error=str(e),
                execution_time=time.time() - start_time
            )
```

## Data Flow

### Input Data Flow

```
User Input → PlantConfigurator → BiogasPlant → Components
    ↓
Substrate XML → Feedstock → C# DLLs → ADM1 Input Stream
    ↓
Initial State CSV → ADM1 State Vector
```

### Simulation Data Flow

```
t=0: Initialize all components
     ↓
Loop over time steps:
     ↓
1. Get component execution order (topological sort)
     ↓
2. For each component:
     - Gather inputs from connected components
     - Call component.step(t, dt, inputs)
     - Store outputs for downstream components
     ↓
3. Collect results
     ↓
Save/Export results
```

### Output Data Flow

```
Component Outputs → Results Dictionary → Analysis/Visualization
                                      ↓
                                   Export:
                                   - JSON (configuration)
                                   - CSV (time series)
                                   - Plots (matplotlib)
```

## Extension Points

PyADM1ODE provides multiple extension points for customization and enhancement without modifying core code.

### 1. Adding New Components

Implement the Component interface to create custom components:

```python
from pyadm1.components.base import Component, ComponentType
from typing import Dict, Any, Optional

class MyCustomComponent(Component):
    """Example custom component implementation.

    Custom components can model any plant equipment or process
    not included in the standard library.

    Args:
        component_id: Unique identifier
        custom_param: Component-specific parameter
        name: Optional human-readable name

    Example:
        >>> component = MyCustomComponent("custom1", custom_param=42.0)
        >>> component.initialize()
        >>> result = component.step(0, 1/24, {})
    """

    def __init__(self, component_id: str, custom_param: float,
                 name: Optional[str] = None):
        """Initialize custom component.

        Args:
            component_id: Unique identifier
            custom_param: Custom parameter value
            name: Optional human-readable name
        """
        super().__init__(component_id, ComponentType.CUSTOM, name)
        self.custom_param = custom_param
        self.state = {
            'value': 0.0,
            'accumulated': 0.0,
            'operating_hours': 0.0
        }

    def initialize(self, initial_state: Optional[Dict[str, Any]] = None):
        """Initialize component state.

        Args:
            initial_state: Optional initial state dictionary
        """
        if initial_state:
            self.state.update(initial_state)

    def step(self, t: float, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one simulation time step.

        Args:
            t: Current time [days]
            dt: Time step [days]
            inputs: Input data from connected components

        Returns:
            Output data dictionary
        """
        # Custom logic
        input_value = inputs.get('input_value', 0.0)

        # Update state
        self.state['value'] = input_value * self.custom_param
        self.state['accumulated'] += self.state['value'] * dt
        self.state['operating_hours'] += dt * 24  # Convert to hours

        # Prepare outputs
        outputs = {
            'output': self.state['value'],
            'accumulated': self.state['accumulated'],
            'status': 'ok',
            'is_running': True
        }

        # Store for access
        self.outputs_data = outputs

        return outputs

    def to_dict(self) -> Dict[str, Any]:
        """Serialize component to dictionary.

        Returns:
            Dictionary representation of component
        """
        return {
            'component_id': self.component_id,
            'component_type': self.component_type.value,
            'custom_param': self.custom_param,
            'name': self.name,
            'state': self.state
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> 'MyCustomComponent':
        """Create component from dictionary.

        Args:
            config: Component configuration dictionary

        Returns:
            MyCustomComponent instance
        """
        component = cls(
            component_id=config['component_id'],
            custom_param=config['custom_param'],
            name=config.get('name')
        )
        if 'state' in config:
            component.state = config['state']
        return component

# Register component for dynamic loading
from pyadm1.components import ComponentRegistry

registry = ComponentRegistry()
registry.register("MyCustomComponent", MyCustomComponent)

# Use in plant
from pyadm1.configurator import BiogasPlant

plant = BiogasPlant("Plant with Custom Component")
custom_comp = MyCustomComponent("custom1", custom_param=42.0)
plant.add_component(custom_comp)
```

### 2. Custom Simulation Algorithms

Extend the Simulator class for specialized simulation algorithms:

```python
from pyadm1.simulation import Simulator
from pyadm1.core import ADM1, ODESolver
import numpy as np
from typing import List, Tuple, Optional

class AdaptiveSimulator(Simulator):
    """Simulator with adaptive time stepping and steady-state detection.

    Extends base Simulator with:
    - Adaptive time step control
    - Automatic steady-state detection
    - Error estimation and control

    Example:
        >>> simulator = AdaptiveSimulator(adm1, tolerance=1e-6)
        >>> state, converged = simulator.simulate_to_steady_state(
        ...     initial_state, max_time=1000
        ... )
    """

    def __init__(self, adm1: ADM1, tolerance: float = 1e-6,
                 solver: Optional[ODESolver] = None):
        """Initialize adaptive simulator.

        Args:
            adm1: ADM1 model instance
            tolerance: Tolerance for adaptive control
            solver: Optional custom ODE solver
        """
        super().__init__(adm1, solver)
        self.tolerance = tolerance
        self.error_history = []

    def adaptive_time_step(self, current_state: List[float],
                          dt_current: float) -> float:
        """Calculate adaptive time step based on error estimate.

        Uses embedded Runge-Kutta method for error estimation.

        Args:
            current_state: Current ADM1 state
            dt_current: Current time step

        Returns:
            Adjusted time step
        """
        # Estimate local truncation error
        def ode_func(t, y):
            return self.adm1.ADM1_ODE(t, y)

        # Two half-steps
        k1 = ode_func(0, current_state)
        state_half = [s + dt_current/2 * k for s, k in zip(current_state, k1)]
        k2 = ode_func(dt_current/2, state_half)
        state_two_half = [s + dt_current/2 * k for s, k in zip(state_half, k2)]

        # One full step
        state_full = [s + dt_current * k for s, k in zip(current_state, k1)]

        # Error estimate
        error = np.max(np.abs(np.array(state_two_half) - np.array(state_full)))
        self.error_history.append(error)

        # Adjust time step
        if error < self.tolerance / 10:
            dt_new = min(dt_current * 1.5, 0.1)  # Increase, max 0.1 days
        elif error > self.tolerance:
            dt_new = max(dt_current * 0.5, 1e-6)  # Decrease, min 1e-6 days
        else:
            dt_new = dt_current

        return dt_new

    def detect_steady_state(self, state_history: List[List[float]],
                           threshold: float = 1e-5) -> bool:
        """Detect when steady state is reached.

        Compares recent states to detect convergence.

        Args:
            state_history: List of recent ADM1 states
            threshold: Convergence threshold

        Returns:
            True if steady state detected
        """
        if len(state_history) < 10:
            return False

        # Compare last 10 states
        recent_states = state_history[-10:]

        # Calculate maximum change
        max_change = 0.0
        for i in range(1, len(recent_states)):
            diff = np.abs(np.array(recent_states[i]) - np.array(recent_states[i-1]))
            max_change = max(max_change, np.max(diff))

        return max_change < threshold

    def simulate_to_steady_state(self, initial_state: List[float],
                                 max_time: float = 1000.0,
                                 check_interval: float = 10.0) -> Tuple[List[float], bool]:
        """Simulate until steady state or max time.

        Args:
            initial_state: Initial ADM1 state
            max_time: Maximum simulation time [days]
            check_interval: Interval for steady-state check [days]

        Returns:
            Tuple of (final_state, converged)
        """
        t = 0.0
        state = initial_state.copy()
        state_history = [state]
        dt = 0.01  # Initial time step

        print(f"Starting simulation to steady state (max {max_time} days)...")

        while t < max_time:
            # Adaptive time step
            dt = self.adaptive_time_step(state, dt)

            # Simulate one step
            state = self.simulate_AD_plant([t, t + dt], state)
            state_history.append(state)
            t += dt

            # Check for steady state periodically
            if len(state_history) % int(check_interval / dt) == 0:
                if self.detect_steady_state(state_history):
                    print(f"✓ Steady state reached at t={t:.1f} days")
                    return state, True
                else:
                    print(f"  t={t:.1f} days, dt={dt:.4f}, checking...")

        print(f"✗ Max time {max_time} days reached without convergence")
        return state, False

# Usage example
from pyadm1.substrates import Feedstock

feedstock = Feedstock(feeding_freq=48)
adm1 = ADM1(feedstock, V_liq=2000, T_ad=308.15)

# Create adaptive simulator
simulator = AdaptiveSimulator(adm1, tolerance=1e-6)

# Run to steady state
initial_state = [0.01] * 37
final_state, converged = simulator.simulate_to_steady_state(
    initial_state,
    max_time=1000
)

if converged:
    print("Steady state achieved!")
    # Use final_state as initial condition for production runs
```

### 3. Custom Parameter Calibration

Implement custom calibration algorithms (see [PyADM1ODE_calibration](https://github.com/dgaida/PyADM1ODE_calibration)):

```python
from pyadm1.core import ADM1
from pyadm1.simulation import Simulator
from typing import Dict, List, Callable
import numpy as np

class CustomCalibrator:
    """Custom parameter calibration implementation.

    Example implementation of Bayesian optimization for parameter estimation.

    Args:
        adm1: ADM1 model instance
        measurement_data: Observed data for calibration
        parameters_to_calibrate: List of parameter names

    Example:
        >>> calibrator = CustomCalibrator(adm1, measurements, ['k_dis', 'Y_su'])
        >>> best_params = calibrator.calibrate(n_iterations=100)
    """

    def __init__(self, adm1: ADM1, measurement_data: Dict[str, List[float]],
                 parameters_to_calibrate: List[str]):
        """Initialize calibrator.

        Args:
            adm1: ADM1 model instance
            measurement_data: Dictionary with 'time' and measured variables
            parameters_to_calibrate: Parameter names to optimize
        """
        self.adm1 = adm1
        self.measurement_data = measurement_data
        self.parameters = parameters_to_calibrate
        self.simulator = Simulator(adm1)

    def objective_function(self, param_values: List[float]) -> float:
        """Calculate objective function (RMSE).

        Args:
            param_values: Parameter values to test

        Returns:
            Root mean square error
        """
        # Set parameters
        param_dict = dict(zip(self.parameters, param_values))
        self.adm1.set_calibration_parameters(param_dict)

        # Simulate
        try:
            results = []
            for i, t in enumerate(self.measurement_data['time'][:-1]):
                dt = self.measurement_data['time'][i+1] - t
                state = self.simulator.simulate_AD_plant([t, t+dt], state)
                results.append(state)

            # Calculate RMSE for methane production
            simulated_ch4 = [self.adm1.calc_gas(...)[-2] for state in results]
            measured_ch4 = self.measurement_data['Q_ch4']

            rmse = np.sqrt(np.mean((np.array(simulated_ch4) - np.array(measured_ch4))**2))

        except Exception as e:
            # Penalize failed simulations
            rmse = 1e10

        return rmse

    def calibrate(self, n_iterations: int = 100,
                  bounds: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, float]:
        """Run calibration using custom optimization.

        Args:
            n_iterations: Number of optimization iterations
            bounds: Parameter bounds as {param: (min, max)}

        Returns:
            Best parameter values
        """
        # Default bounds
        if bounds is None:
            bounds = {
                'k_dis': (0.1, 1.0),
                'Y_su': (0.05, 0.15),
                'k_hyd_ch': (5.0, 20.0),
                # ... other parameters
            }

        # Extract bounds for optimizer
        param_bounds = [bounds[p] for p in self.parameters]

        # Simple random search (replace with sophisticated algorithm)
        best_params = None
        best_error = float('inf')

        print(f"Starting calibration with {n_iterations} iterations...")

        for i in range(n_iterations):
            # Random sample within bounds
            param_values = [
                np.random.uniform(b[0], b[1])
                for b in param_bounds
            ]

            # Evaluate
            error = self.objective_function(param_values)

            if error < best_error:
                best_error = error
                best_params = param_values.copy()
                print(f"Iteration {i+1}: New best RMSE = {best_error:.4f}")

        # Return as dictionary
        result = dict(zip(self.parameters, best_params))
        print(f"\nCalibration complete!")
        print(f"Best parameters: {result}")
        print(f"Best RMSE: {best_error:.4f}")

        return result

# Usage
measurement_data = {
    'time': [0, 1, 2, 3, 4, 5],  # days
    'Q_ch4': [700, 720, 735, 745, 750, 752]  # m³/d
}

calibrator = CustomCalibrator(
    adm1,
    measurement_data,
    parameters_to_calibrate=['k_dis', 'Y_su']
)

best_params = calibrator.calibrate(n_iterations=100)

# Apply to model
adm1.set_calibration_parameters(best_params)
```

### 4. Custom Substrates

Add new substrate definitions via XML:

```xml
<!-- data/substrates/custom_substrates.xml -->
<?xml version="1.0" encoding="utf-8"?>
<substrates>
    <substrate name="custom_substrate">
        <!-- Basic properties -->
        <pH>4.5</pH>
        <TS>25.0</TS>  <!-- Total solids [%] -->
        <VS>95.0</VS>  <!-- Volatile solids [% of TS] -->

        <!-- Weender analysis -->
        <crude_protein>15.0</crude_protein>  <!-- [% of TS] -->
        <crude_fat>4.0</crude_fat>
        <crude_fiber>20.0</crude_fiber>
        <nitrogen_free_extract>55.0</nitrogen_free_extract>
        <ash>6.0</ash>

        <!-- Van Soest fractions -->
        <NDF>45.0</NDF>  <!-- Neutral detergent fiber [% of TS] -->
        <ADF>25.0</ADF>  <!-- Acid detergent fiber [% of TS] -->
        <ADL>5.0</ADL>   <!-- Acid detergent lignin [% of TS] -->

        <!-- COD fractions (will be calculated if not provided) -->
        <COD_total>1.5</COD_total>  <!-- [kg COD/kg FM] -->

        <!-- Kinetic parameters (optional, will use defaults if not provided) -->
        <k_dis>0.5</k_dis>          <!-- Disintegration rate [1/d] -->
        <k_hyd_ch>10.0</k_hyd_ch>   <!-- Carbohydrate hydrolysis [1/d] -->
        <k_hyd_pr>10.0</k_hyd_pr>   <!-- Protein hydrolysis [1/d] -->
        <k_hyd_li>10.0</k_hyd_li>   <!-- Lipid hydrolysis [1/d] -->
    </substrate>
</substrates>
```

Load custom substrates:

```python
from pyadm1.substrates import Feedstock

# Load with custom substrate file
feedstock = Feedstock(
    feeding_freq=48,
    substrate_xml='custom_substrates.xml'
)

# Verify substrate loaded
substrates = feedstock.mySubstrates()
n_substrates = substrates.getNumSubstrates()
print(f"Loaded {n_substrates} substrates")

# Use in simulation
Q_substrates = [0, 0, 0, 0, 0, 0, 0, 0, 0, 20]  # 20 m³/d custom substrate
```

### 5. Custom Result Analyzers

Create specialized analysis tools:

```python
from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt

class EconomicAnalyzer:
    """Analyze economic performance of biogas plant.

    Calculates revenues, costs, and profitability metrics.

    Args:
        electricity_price: Price per kWh [€/kWh]
        feed_in_tariff: Feed-in tariff [€/kWh]
        substrate_costs: Cost per substrate [€/m³ or €/t]

    Example:
        >>> analyzer = EconomicAnalyzer(
        ...     electricity_price=0.30,
        ...     feed_in_tariff=0.15,
        ...     substrate_costs={'corn_silage': 35, 'manure': 0}
        ... )
        >>> economics = analyzer.analyze(results, Q_substrates)
    """

    def __init__(self, electricity_price: float = 0.30,
                 feed_in_tariff: float = 0.15,
                 substrate_costs: Dict[str, float] = None):
        """Initialize economic analyzer.

        Args:
            electricity_price: Electricity price [€/kWh]
            feed_in_tariff: Feed-in tariff [€/kWh]
            substrate_costs: Substrate costs dictionary
        """
        self.electricity_price = electricity_price
        self.feed_in_tariff = feed_in_tariff
        self.substrate_costs = substrate_costs or {}

    def analyze(self, results: List[Dict[str, Any]],
                Q_substrates: List[float]) -> Dict[str, float]:
        """Analyze economic performance.

        Args:
            results: Simulation results
            Q_substrates: Substrate feed rates [m³/d]

        Returns:
            Economic metrics dictionary
        """
        # Extract data
        times = [r['time'] for r in results]
        duration = times[-1] - times[0]

        # Energy production
        if 'chp1' in results[0]['components']:
            P_el = [r['components']['chp1']['P_el'] for r in results]
            avg_P_el = sum(P_el) / len(P_el)
            annual_production = avg_P_el * 24 * 365  # kWh/year
        else:
            annual_production = 0

        # Revenues
        revenue_electricity = annual_production * self.feed_in_tariff

        # Substrate costs
        substrate_names = ['corn_silage', 'manure', 'rye', 'grass',
                          'wheat', 'gps', 'ccm', 'lime', 'cow_manure', 'onions']
        annual_substrate_cost = 0
        for i, substrate in enumerate(substrate_names):
            if i < len(Q_substrates) and Q_substrates[i] > 0:
                cost_per_unit = self.substrate_costs.get(substrate, 0)
                annual_cost = Q_substrates[i] * 365 * cost_per_unit
                annual_substrate_cost += annual_cost

        # Operating costs (simplified)
        operating_cost = annual_production * 0.03  # 3 ct/kWh

        # Total costs
        total_costs = annual_substrate_cost + operating_cost

        # Profit
        annual_profit = revenue_electricity - total_costs

        return {
            'annual_production_kWh': annual_production,
            'revenue_electricity_€': revenue_electricity,
            'substrate_costs_€': annual_substrate_cost,
            'operating_costs_€': operating_cost,
            'total_costs_€': total_costs,
            'annual_profit_€': annual_profit,
            'profit_margin': annual_profit / revenue_electricity if revenue_electricity > 0 else 0,
            'specific_cost_€_per_kWh': total_costs / annual_production if annual_production > 0 else 0
        }

    def plot_economics(self, economics: Dict[str, float]):
        """Plot economic analysis.

        Args:
            economics: Economic metrics from analyze()
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Revenue vs Costs
        categories = ['Revenue', 'Substrate\nCosts', 'Operating\nCosts', 'Profit']
        values = [
            economics['revenue_electricity_€'],
            -economics['substrate_costs_€'],
            -economics['operating_costs_€'],
            economics['annual_profit_€']
        ]
        colors = ['green', 'red', 'red', 'blue']

        ax1.bar(categories, values, color=colors, alpha=0.7)
        ax1.set_ylabel('€/year')
        ax1.set_title('Annual Economics')
        ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax1.grid(True, alpha=0.3)

        # Cost breakdown pie chart
        cost_categories = ['Substrates', 'Operation']
        cost_values = [
            economics['substrate_costs_€'],
            economics['operating_costs_€']
        ]

        ax2.pie(cost_values, labels=cost_categories, autopct='%1.1f%%',
                colors=['#ff9999', '#66b3ff'])
        ax2.set_title('Cost Breakdown')

        plt.tight_layout()
        plt.show()

# Usage
analyzer = EconomicAnalyzer(
    electricity_price=0.30,
    feed_in_tariff=0.15,
    substrate_costs={
        'corn_silage': 35,  # €/t
        'manure': 0,        # Free
        'grass': 25
    }
)

economics = analyzer.analyze(results, Q_substrates=[15, 10, 0, 0, 0, 0, 0, 0, 0, 0])
print(f"Annual profit: {economics['annual_profit_€']:,.0f} €/year")
analyzer.plot_economics(economics)
```

## Integration with External Systems

PyADM1ODE can be integrated with various external systems for enhanced functionality.

### 1. MCP Server Integration (LLM-Driven Design)

The Model Context Protocol (MCP) server enables LLM-driven plant design via natural language.

**Repository**: [PyADM1ODE_mcp](https://github.com/dgaida/PyADM1ODE_mcp)

**Architecture**:

```
User (Natural Language)
         ↓
    LLM (Claude)
         ↓
    MCP Server
         ↓
PyADM1ODE PlantConfigurator
         ↓
    BiogasPlant
```

**Example Interaction**:

```
User: "Design a 500 kW biogas plant with corn silage and manure"

LLM → MCP Server: create_plant(name="500kW Plant")
MCP Server → PyADM1ODE: BiogasPlant("500kW Plant")

LLM → MCP Server: add_digester(V_liq=2000, substrates=["corn", "manure"])
MCP Server → PyADM1ODE: configurator.add_digester(...)

LLM → MCP Server: add_chp(P_el_nom=500)
MCP Server → PyADM1ODE: configurator.add_chp(...)

LLM → MCP Server: simulate(duration=30)
MCP Server → PyADM1ODE: plant.simulate(30)

MCP Server → LLM: Results summary
LLM → User: "Your plant produces 750 m³/d methane at 60% content..."
```

**Installation**:

```bash
git clone https://github.com/dgaida/PyADM1ODE_mcp.git
cd PyADM1ODE_mcp
pip install -e .
```

**Key Features**:
- Natural language plant design
- Automatic component selection
- Parameter optimization suggestions
- Interactive refinement

### 2. Parameter Calibration System

Automated calibration from measurement data.

**Repository**: [PyADM1ODE_calibration](https://github.com/dgaida/PyADM1ODE_calibration)

**Architecture**:

```
Measurement Database
         ↓
  Data Preprocessing
         ↓
Calibration Algorithm (DE/PSO/Nelder-Mead)
         ↓
    PyADM1ODE ADM1
         ↓
Calibrated Parameters → Model
```

**Usage Example**:

```python
from pyadm1_calibration import Calibrator
from pyadm1.core import ADM1
from pyadm1.substrates import Feedstock

# Load measurement data
measurements = pd.read_csv('plant_data.csv')

# Create calibrator
feedstock = Feedstock(feeding_freq=48)
adm1 = ADM1(feedstock, V_liq=2000, T_ad=308.15)

calibrator = Calibrator(
    adm1=adm1,
    measurement_data=measurements,
    parameters_to_calibrate=['k_dis', 'Y_su', 'k_hyd_ch'],
    algorithm='differential_evolution'
)

# Run calibration
best_params, history = calibrator.calibrate(
    n_iterations=100,
    population_size=20
)

# Apply to model
adm1.set_calibration_parameters(best_params)

# Validate
validation_rmse = calibrator.validate(best_params, validation_data)
print(f"Validation RMSE: {validation_rmse:.4f}")
```

**Key Features**:
- Multiple optimization algorithms
- Online re-calibration
- Comprehensive validation metrics
- Database integration

### 3. Database Integration

Connect to various databases for input/output management:

```python
import sqlalchemy
from sqlalchemy import create_engine
import pandas as pd

class DatabaseConnector:
    """Connect PyADM1ODE to external databases.

    Supports PostgreSQL, MySQL, SQLite for:
    - Loading measurement data
    - Storing simulation results
    - Managing plant configurations

    Example:
        >>> db = DatabaseConnector('postgresql://user:pass@localhost/biogas')
        >>> measurements = db.load_measurements('plant_001', start_date, end_date)
        >>> db.save_results(results, 'simulation_001')
    """

    def __init__(self, connection_string: str):
        """Initialize database connection.

        Args:
            connection_string: SQLAlchemy connection string
        """
        self.engine = create_engine(connection_string)

    def load_measurements(self, plant_id: str,
                         start_date: str, end_date: str) -> pd.DataFrame:
        """Load measurement data from database.

        Args:
            plant_id: Plant identifier
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with measurement data
        """
        query = f"""
        SELECT timestamp, Q_gas, Q_ch4, pH, VFA, temperature
        FROM measurements
        WHERE plant_id = '{plant_id}'
          AND timestamp BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY timestamp
        """

        return pd.read_sql(query, self.engine)

    def save_results(self, results: List[Dict[str, Any]],
                    simulation_id: str):
        """Save simulation results to database.

        Args:
            results: Simulation results
            simulation_id: Unique simulation identifier
        """
        # Convert to DataFrame
        data = []
        for r in results:
            row = {
                'simulation_id': simulation_id,
                'timestamp': r['time'],
                'Q_gas': r['components']['main_digester']['Q_gas'],
                'Q_ch4': r['components']['main_digester']['Q_ch4'],
                'pH': r['components']['main_digester']['pH'],
                'VFA': r['components']['main_digester']['VFA']
            }
            data.append(row)

        df = pd.DataFrame(data)
        df.to_sql('simulation_results', self.engine,
                 if_exists='append', index=False)

    def load_plant_config(self, config_id: str) -> Dict[str, Any]:
        """Load plant configuration from database.

        Args:
            config_id: Configuration identifier

        Returns:
            Configuration dictionary
        """
        query = f"""
        SELECT config_json
        FROM plant_configurations
        WHERE config_id = '{config_id}'
        """

        result = pd.read_sql(query, self.engine)
        return json.loads(result['config_json'][0])

# Usage
db = DatabaseConnector('postgresql://user:pass@localhost/biogas_db')

# Load measurements
measurements = db.load_measurements(
    'plant_001',
    '2024-01-01',
    '2024-12-31'
)

# Run simulation
plant.initialize()
results = plant.simulate(duration=30)

# Save results
db.save_results(results, 'sim_20240115_001')
```

### 4. SCADA/Process Control Systems

Integrate with industrial SCADA systems for real-time monitoring and control:

```python
from typing import Callable, Dict, Any
import time
import threading

class SCADAInterface:
    """Interface to SCADA/DCS systems.

    Provides bidirectional communication for:
    - Reading process values
    - Writing setpoints
    - Alarm management

    Supports protocols:
    - OPC UA
    - Modbus TCP
    - MQTT

    Example:
        >>> scada = SCADAInterface(protocol='opcua', endpoint='opc.tcp://localhost:4840')
        >>> scada.start()
        >>> value = scada.read_tag('Digester.Temperature')
        >>> scada.write_tag('Feeder.Setpoint', 15.0)
    """

    def __init__(self, protocol: str = 'opcua',
                 endpoint: str = 'opc.tcp://localhost:4840'):
        """Initialize SCADA interface.

        Args:
            protocol: Communication protocol
            endpoint: Connection endpoint
        """
        self.protocol = protocol
        self.endpoint = endpoint
        self.connected = False
        self.tag_cache = {}
        self.callbacks = {}

    def connect(self):
        """Establish connection to SCADA system."""
        if self.protocol == 'opcua':
            # Example using opcua library
            from opcua import Client
            self.client = Client(self.endpoint)
            self.client.connect()
            self.connected = True
            print(f"Connected to OPC UA server at {self.endpoint}")

        elif self.protocol == 'mqtt':
            # Example using paho-mqtt
            import paho.mqtt.client as mqtt
            self.client = mqtt.Client()
            self.client.on_connect = self._on_mqtt_connect
            self.client.on_message = self._on_mqtt_message
            self.client.connect(self.endpoint.split(':')[0],
                              int(self.endpoint.split(':')[1]))
            self.client.loop_start()
            self.connected = True
            print(f"Connected to MQTT broker at {self.endpoint}")

    def disconnect(self):
        """Close connection to SCADA system."""
        if self.connected:
            self.client.disconnect()
            self.connected = False

    def read_tag(self, tag_name: str) -> Any:
        """Read value from SCADA tag.

        Args:
            tag_name: Tag name/address

        Returns:
            Tag value
        """
        if not self.connected:
            raise RuntimeError("Not connected to SCADA system")

        if self.protocol == 'opcua':
            node = self.client.get_node(tag_name)
            value = node.get_value()
            self.tag_cache[tag_name] = value
            return value

    def write_tag(self, tag_name: str, value: Any):
        """Write value to SCADA tag.

        Args:
            tag_name: Tag name/address
            value: Value to write
        """
        if not self.connected:
            raise RuntimeError("Not connected to SCADA system")

        if self.protocol == 'opcua':
            node = self.client.get_node(tag_name)
            node.set_value(value)
            print(f"Wrote {value} to {tag_name}")

    def subscribe(self, tag_name: str, callback: Callable):
        """Subscribe to tag changes.

        Args:
            tag_name: Tag to monitor
            callback: Function to call on change
        """
        self.callbacks[tag_name] = callback

        if self.protocol == 'opcua':
            handler = self._create_subscription_handler(callback)
            node = self.client.get_node(tag_name)
            sub = self.client.create_subscription(1000, handler)
            sub.subscribe_data_change(node)

# Integration with PyADM1ODE
class RealTimePlantControl:
    """Real-time plant control using SCADA integration.

    Synchronizes PyADM1ODE simulation with real plant data.

    Example:
        >>> controller = RealTimePlantControl(plant, scada)
        >>> controller.start_control_loop()
    """

    def __init__(self, plant: BiogasPlant, scada: SCADAInterface):
        """Initialize controller.

        Args:
            plant: BiogasPlant instance
            scada: SCADA interface
        """
        self.plant = plant
        self.scada = scada
        self.running = False

    def read_process_values(self) -> Dict[str, float]:
        """Read current process values from SCADA.

        Returns:
            Dictionary of process values
        """
        return {
            'temperature': self.scada.read_tag('Digester.Temperature'),
            'pH': self.scada.read_tag('Digester.pH'),
            'VFA': self.scada.read_tag('Digester.VFA'),
            'Q_gas': self.scada.read_tag('GasMeter.Flow'),
            'feed_rate': self.scada.read_tag('Feeder.ActualFlow')
        }

    def write_setpoints(self, setpoints: Dict[str, float]):
        """Write setpoints to SCADA.

        Args:
            setpoints: Dictionary of setpoint values
        """
        for tag, value in setpoints.items():
            self.scada.write_tag(tag, value)

    def control_loop(self, dt: float = 60.0):
        """Main control loop.

        Args:
            dt: Loop interval [seconds]
        """
        self.running = True

        while self.running:
            # Read process values
            process_values = self.read_process_values()

            # Update plant model
            # (Simplified - real implementation would be more sophisticated)

            # Run predictive simulation
            results = self.plant.simulate(duration=1.0, dt=1/24)

            # Calculate optimal setpoints
            setpoints = self._calculate_setpoints(results, process_values)

            # Write to SCADA
            self.write_setpoints(setpoints)

            # Wait for next cycle
            time.sleep(dt)

    def _calculate_setpoints(self, simulation_results, process_values):
        """Calculate optimal setpoints based on model prediction."""
        # Implement model predictive control logic
        return {
            'Feeder.Setpoint': 15.0,
            'Mixer.Speed': 80.0
        }

    def start_control_loop(self):
        """Start control loop in background thread."""
        thread = threading.Thread(target=self.control_loop)
        thread.daemon = True
        thread.start()

    def stop_control_loop(self):
        """Stop control loop."""
        self.running = False

# Usage
scada = SCADAInterface(protocol='opcua', endpoint='opc.tcp://plc.local:4840')
scada.connect()

controller = RealTimePlantControl(plant, scada)
controller.start_control_loop()
```

### 5. Weather and Market Data Integration

Integrate external data sources for enhanced predictions:

```python
import requests
from datetime import datetime, timedelta
from typing import Dict, List

class ExternalDataIntegration:
    """Integrate weather and market data into simulations.

    Sources:
    - Weather forecasts (temperature, radiation)
    - Energy market prices
    - Agricultural commodity prices

    Example:
        >>> data_source = ExternalDataIntegration(api_key='...')
        >>> weather = data_source.get_weather_forecast(location='Berlin', days=7)
        >>> prices = data_source.get_electricity_prices(market='DE', date='2024-01-15')
    """

    def __init__(self, weather_api_key: str = None,
                 market_api_key: str = None):
        """Initialize data integration.

        Args:
            weather_api_key: API key for weather service
            market_api_key: API key for market data
        """
        self.weather_api_key = weather_api_key
        self.market_api_key = market_api_key

    def get_weather_forecast(self, location: str,
                            days: int = 7) -> pd.DataFrame:
        """Get weather forecast.

        Args:
            location: Location name or coordinates
            days: Forecast days

        Returns:
            DataFrame with weather data
        """
        # Example using OpenWeatherMap API
        url = f"https://api.openweathermap.org/data/2.5/forecast"
        params = {
            'q': location,
            'appid': self.weather_api_key,
            'units': 'metric',
            'cnt': days * 8  # 3-hour intervals
        }

        response = requests.get(url, params=params)
        data = response.json()

        # Parse response
        forecast = []
        for item in data['list']:
            forecast.append({
                'timestamp': datetime.fromtimestamp(item['dt']),
                'temperature': item['main']['temp'],
                'humidity': item['main']['humidity'],
                'wind_speed': item['wind']['speed']
            })

        return pd.DataFrame(forecast)

    def get_electricity_prices(self, market: str = 'DE',
                              start_date: str = None,
                              end_date: str = None) -> pd.DataFrame:
        """Get electricity market prices.

        Args:
            market: Market identifier (DE, FR, etc.)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with price data
        """
        # Example using ENTSO-E API
        url = "https://transparency.entsoe.eu/api"
        params = {
            'securityToken': self.market_api_key,
            'documentType': 'A44',  # Day-ahead prices
            'in_Domain': f'{market}',
            'out_Domain': f'{market}',
            'periodStart': start_date.replace('-', '') + '0000',
            'periodEnd': end_date.replace('-', '') + '0000'
        }

        response = requests.get(url, params=params)

        # Parse XML response (simplified)
        # Real implementation would parse ENTSO-E XML format

        return pd.DataFrame({
            'timestamp': pd.date_range(start_date, end_date, freq='H'),
            'price_€_per_MWh': [45.5] * 24  # Example data
        })

    def optimize_operation(self, weather_forecast: pd.DataFrame,
                          price_forecast: pd.DataFrame,
                          plant: BiogasPlant) -> Dict[str, Any]:
        """Optimize plant operation based on forecasts.

        Args:
            weather_forecast: Weather data
            price_forecast: Electricity prices
            plant: BiogasPlant instance

        Returns:
            Optimization recommendations
        """
        # Simple example: Adjust CHP operation to price peaks
        avg_price = price_forecast['price_€_per_MWh'].mean()
        peak_hours = price_forecast[
            price_forecast['price_€_per_MWh'] > avg_price * 1.2
        ]

        recommendations = {
            'strategy': 'price_optimized',
            'peak_hours': peak_hours['timestamp'].tolist(),
            'recommended_actions': [
                f"Run CHP at full load during {len(peak_hours)} peak hours",
                f"Store gas during {24-len(peak_hours)} low-price hours",
                f"Expected additional revenue: {(avg_price * 0.2 * len(peak_hours)):.0f} €/day"
            ]
        }

        return recommendations

# Usage
data_integration = ExternalDataIntegration(
    weather_api_key='your_openweather_key',
    market_api_key='your_entsoe_key'
)

# Get forecasts
weather = data_integration.get_weather_forecast('Berlin', days=7)
prices = data_integration.get_electricity_prices(
    market='DE',
    start_date='2024-01-15',
    end_date='2024-01-22'
)

# Optimize operation
optimization = data_integration.optimize_operation(weather, prices, plant)
print(optimization['recommended_actions'])
```

### 6. Cloud Deployment

Deploy PyADM1ODE as cloud service:

```python
from flask import Flask, request, jsonify
from pyadm1.configurator import BiogasPlant
from pyadm1.substrates import Feedstock
import json

app = Flask(__name__)

class CloudSimulationService:
    """RESTful API for cloud-based biogas simulation.

    Endpoints:
    - POST /api/simulate: Run simulation
    - GET /api/plant/{id}: Get plant configuration
    - POST /api/plant: Create new plant

    Example:
        >>> # Start service
        >>> service = CloudSimulationService()
        >>> service.run(host='0.0.0.0', port=5000)
        >>>
        >>> # Client request
        >>> import requests
        >>> response = requests.post(
        ...     'http://api.example.com/api/simulate',
        ...     json={'plant_id': 'plant_001', 'duration': 30}
        ... )
    """

    def __init__(self):
        """Initialize cloud service."""
        self.plants = {}
        self.feedstock = Feedstock(feeding_freq=48)

    @app.route('/api/simulate', methods=['POST'])
    def simulate(self):
        """Run simulation endpoint."""
        data = request.json
        plant_id = data.get('plant_id')
        duration = data.get('duration', 30)

        # Get or create plant
        if plant_id not in self.plants:
            return jsonify({'error': 'Plant not found'}), 404

        plant = self.plants[plant_id]

        # Run simulation
        try:
            plant.initialize()
            results = plant.simulate(duration=duration, dt=1/24)

            # Extract summary
            final = results[-1]
            summary = {
                'status': 'success',
                'duration': duration,
                'biogas_production': final['components']['main_digester']['Q_gas'],
                'methane_production': final['components']['main_digester']['Q_ch4'],
                'electrical_power': final['components'].get('chp1', {}).get('P_el', 0),
                'pH': final['components']['main_digester']['pH']
            }

            return jsonify(summary), 200

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/api/plant', methods=['POST'])
    def create_plant(self):
        """Create new plant endpoint."""
        config = request.json

        try:
            # Create plant from config
            plant = BiogasPlant.from_dict(config, self.feedstock)
            plant_id = config['plant_name']
            self.plants[plant_id] = plant

            return jsonify({
                'status': 'created',
                'plant_id': plant_id
            }), 201

        except Exception as e:
            return jsonify({'error': str(e)}), 400

    @app.route('/api/plant/', methods=['GET'])
    def get_plant(self, plant_id):
        """Get plant configuration endpoint."""
        if plant_id not in self.plants:
            return jsonify({'error': 'Plant not found'}), 404

        plant = self.plants[plant_id]
        config = plant.to_dict()

        return jsonify(config), 200

    def run(self, host='0.0.0.0', port=5000):
        """Start service."""
        app.run(host=host, port=port)

# Deploy
if __name__ == '__main__':
    service = CloudSimulationService()
    service.run()
```

**Docker Deployment**:

```dockerfile
# Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    mono-complete \
    && rm -rf /var/lib/apt/lists/*

# Install PyADM1ODE
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e .

# Expose port
EXPOSE 5000

# Run service
CMD ["python", "cloud_service.py"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  pyadm1-api:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
    volumes:
      - ./data:/app/data
    restart: unless-stopped

  postgres:
    image: postgres:14
    environment:
      POSTGRES_DB: biogas_db
      POSTGRES_USER: pyadm1
      POSTGRES_PASSWORD: secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  postgres_data:
```

### 7. Jupyter Notebook Integration

Interactive analysis and visualization:

```python
# biogas_analysis.ipynb

# Cell 1: Setup
from pyadm1.configurator import BiogasPlant, PlantConfigurator
from pyadm1.substrates import Feedstock
import matplotlib.pyplot as plt
import pandas as pd
import ipywidgets as widgets
from IPython.display import display

%matplotlib inline

# Cell 2: Interactive Plant Designer
def create_interactive_plant():
    """Interactive widget for plant design."""

    # Widgets
    v_liq_slider = widgets.FloatSlider(
        value=2000, min=500, max=5000, step=100,
        description='V_liq [m³]:'
    )

    p_el_slider = widgets.FloatSlider(
        value=500, min=100, max=2000, step=50,
        description='CHP [kW]:'
    )

    corn_slider = widgets.FloatSlider(
        value=15, min=0, max=50, step=1,
        description='Corn [m³/d]:'
    )

    manure_slider = widgets.FloatSlider(
        value=10, min=0, max=50, step=1,
        description='Manure [m³/d]:'
    )

    button = widgets.Button(description='Simulate')
    output = widgets.Output()

    def on_button_click(b):
        with output:
            output.clear_output()

            # Create plant
            feedstock = Feedstock(feeding_freq=48)
            plant = BiogasPlant("Interactive Plant")
            config = PlantConfigurator(plant, feedstock)

            Q = [corn_slider.value, manure_slider.value, 0, 0, 0, 0, 0, 0, 0, 0]

            config.add_digester("dig1", V_liq=v_liq_slider.value,
                               Q_substrates=Q)
            config.add_chp("chp1", P_el_nom=p_el_slider.value)
            config.auto_connect_digester_to_chp("dig1", "chp1")

            # Simulate
            plant.initialize()
            results = plant.simulate(duration=30, dt=1/24, save_interval=1.0)

            # Plot
            times = [r['time'] for r in results]
            biogas = [r['components']['dig1']['Q_gas'] for r in results]
            power = [r['components']['chp1']['P_el'] for r in results]

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

            ax1.plot(times, biogas)
            ax1.set_ylabel('Biogas [m³/d]')
            ax1.grid(True)

            ax2.plot(times, power)
            ax2.set_xlabel('Time [days]')
            ax2.set_ylabel('Power [kW]')
            ax2.grid(True)

            plt.tight_layout()
            plt.show()

            # Summary
            final = results[-1]
            print(f"\nFinal Results:")
            print(f"Biogas: {final['components']['dig1']['Q_gas']:.1f} m³/d")
            print(f"Power: {final['components']['chp1']['P_el']:.1f} kW")

    button.on_click(on_button_click)

    display(v_liq_slider, p_el_slider, corn_slider, manure_slider, button, output)

# Cell 3: Run interactive designer
create_interactive_plant()
```

## Summary

PyADM1ODE's architecture provides:

1. **Modularity**: Independent, reusable components
2. **Extensibility**: Multiple extension points for customization
3. **Integration**: Interfaces to external systems (MCP, databases, SCADA, cloud)
4. **Scalability**: From single simulations to large-scale parallel analyses
5. **Maintainability**: Clean separation of concerns, well-documented APIs

The design enables:
- Rapid prototyping of biogas plant concepts
- Integration with existing infrastructure
- Extension for research and commercial applications
- Deployment in various environments (local, cloud, embedded)

For detailed implementation examples, see:
- [Adding Components Guide](adding_components.md)
- [API Reference](../api_reference/index.md)
- [Examples](../../examples/) Custom Substrates
