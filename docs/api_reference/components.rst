Plant Components
================

The components module provides a comprehensive library of biogas plant components
that can be combined to build complex plant configurations. All components follow
a common interface and can be dynamically loaded and connected.

Base Classes
------------

Component Base
~~~~~~~~~~~~~~

.. autoclass:: pyadm1.components.base.Component
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: __weakref__

   Abstract base class for all biogas plant components. All components must implement:

   - ``step(t, dt, inputs)``: Execute one simulation time step
   - ``initialize(state)``: Set up initial state
   - ``to_dict()`` / ``from_dict()``: Serialization for JSON export/import

   **Component Lifecycle:**

   1. **Creation**: ``component = ComponentClass(component_id, **params)``
   2. **Initialization**: ``component.initialize(initial_state)``
   3. **Simulation Loop**: ``outputs = component.step(t, dt, inputs)``
   4. **Serialization**: ``config = component.to_dict()``

   **Connection Management:**

   Components track their inputs and outputs:

   - ``inputs``: List of component IDs providing input
   - ``outputs``: List of component IDs receiving output
   - ``outputs_data``: Dictionary of output data for connected components

Component Type
~~~~~~~~~~~~~~

.. autoclass:: pyadm1.components.base.ComponentType
   :members:
   :undoc-members:
   :show-inheritance:

   Enumeration of component types:

   - ``DIGESTER``: Biological digester (ADM1)
   - ``CHP``: Combined heat and power unit
   - ``HEATING``: Heating system
   - ``STORAGE``: Gas storage
   - ``SEPARATOR``: Solid-liquid separator
   - ``MIXER``: Mixing equipment

Component Registry
~~~~~~~~~~~~~~~~~~

.. autoclass:: pyadm1.components.registry.ComponentRegistry
   :members:
   :undoc-members:
   :show-inheritance:

   Registry for dynamic component discovery and instantiation. Enables plugin-like
   extensibility and automated component selection.

   **Example:**

   .. code-block:: python

      from pyadm1.components import ComponentRegistry

      registry = ComponentRegistry()

      # Create component via registry
      digester = registry.create(
          "Digester",
          component_id="dig1",
          feedstock=feedstock,
          V_liq=2000
      )

      # List available components
      print(registry.list_components())

Factory Functions
~~~~~~~~~~~~~~~~~

.. autofunction:: pyadm1.components.registry.get_registry

   Get the global component registry instance.

.. autofunction:: pyadm1.components.registry.register_component

   Register a custom component in the global registry.

Biological Components
---------------------

Digester
~~~~~~~~

.. autoclass:: pyadm1.components.biological.digester.Digester
   :members:
   :undoc-members:
   :show-inheritance:

   Main fermenter component implementing ADM1 anaerobic digestion model.

   **Key Features:**

   - Wraps ADM1 core implementation
   - Supports series and parallel configurations
   - Automatic gas storage creation
   - Full ADM1 state tracking

   **Parameters:**

   :param component_id: Unique identifier
   :type component_id: str
   :param feedstock: Feedstock object for substrate management
   :type feedstock: Feedstock
   :param V_liq: Liquid volume [m³], default 1977.0
   :type V_liq: float
   :param V_gas: Gas volume [m³], default 304.0
   :type V_gas: float
   :param T_ad: Operating temperature [K], default 308.15 (35°C)
   :type T_ad: float
   :param name: Human-readable name, optional
   :type name: str

   **Outputs:**

   - ``Q_out``: Effluent flow rate [m³/d]
   - ``state_out``: ADM1 state vector for next digester
   - ``Q_gas``: Biogas production [m³/d]
   - ``Q_ch4``: Methane production [m³/d]
   - ``Q_co2``: CO2 production [m³/d]
   - ``pH``: pH value
   - ``VFA``: VFA concentration [g/L]
   - ``TAC``: Total alkalinity [g CaCO3/L]
   - ``gas_storage``: Attached storage diagnostics

   **Example:**

   .. code-block:: python

      from pyadm1.components.biological import Digester
      from pyadm1.substrates import Feedstock

      feedstock = Feedstock(feeding_freq=48)
      digester = Digester(
          component_id="main_digester",
          feedstock=feedstock,
          V_liq=2000,
          V_gas=300,
          T_ad=308.15,
          name="Main Digester"
      )

      # Initialize with substrate feed
      digester.initialize({
          'adm1_state': initial_state,
          'Q_substrates': [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]
      })

      # Simulate one step
      outputs = digester.step(t=0, dt=0.04167, inputs={})
      print(f"Biogas: {outputs['Q_gas']:.1f} m³/d")

Hydrolysis Tank
~~~~~~~~~~~~~~~

.. autoclass:: pyadm1.components.biological.hydrolysis.Hydrolysis
   :members:
   :undoc-members:
   :show-inheritance:

   Pre-treatment tank for hydrolysis-dominated processes. Useful for substrates
   with high lignocellulosic content.

   **Typical Configuration:**

   - Temperature: 45-55°C (thermophilic)
   - Volume: 20-30% of total digester volume
   - HRT: 2-7 days

Separator
~~~~~~~~~

.. autoclass:: pyadm1.components.biological.separator.Separator
   :members:
   :undoc-members:
   :show-inheritance:

   Solid-liquid separation for digestate processing. Models mechanical
   (screw press, centrifuge) or gravitational separation.

Energy Components
-----------------

CHP Unit
~~~~~~~~

.. autoclass:: pyadm1.components.energy.chp.CHP
   :members:
   :undoc-members:
   :show-inheritance:

   Combined Heat and Power unit for biogas conversion to electricity and heat.

   **Parameters:**

   :param component_id: Unique identifier
   :type component_id: str
   :param P_el_nom: Nominal electrical power [kW], default 500.0
   :type P_el_nom: float
   :param eta_el: Electrical efficiency [0-1], default 0.40
   :type eta_el: float
   :param eta_th: Thermal efficiency [0-1], default 0.45
   :type eta_th: float
   :param name: Human-readable name, optional
   :type name: str

   **Outputs:**

   - ``P_el``: Electrical power [kW]
   - ``P_th``: Thermal power [kW]
   - ``Q_gas_consumed``: Gas consumption [m³/d]
   - ``load_factor``: Operating point [0-1]

   **Performance Characteristics:**

   .. list-table::
      :header-rows: 1

      * - Type
        - Size [kW_el]
        - η_el
        - η_th
        - η_total
      * - Gas Engine
        - 100-2000
        - 0.38-0.42
        - 0.40-0.48
        - 0.85-0.90
      * - Micro-turbine
        - 100-500
        - 0.25-0.30
        - 0.50-0.60
        - 0.80-0.85

   **Example:**

   .. code-block:: python

      from pyadm1.components.energy import CHP

      chp = CHP(
          component_id="chp_main",
          P_el_nom=500,
          eta_el=0.40,
          eta_th=0.45,
          name="Main CHP"
      )

      # Simulate with gas input
      outputs = chp.step(
          t=0,
          dt=0.04167,
          inputs={'Q_gas_supplied_m3_per_day': 2700}
      )
      print(f"Power: {outputs['P_el']:.1f} kW_el")

Heating System
~~~~~~~~~~~~~~

.. autoclass:: pyadm1.components.energy.heating.HeatingSystem
   :members:
   :undoc-members:
   :show-inheritance:

   Heating system for maintaining digester temperature using CHP waste heat
   and auxiliary heating.

   **Parameters:**

   :param component_id: Unique identifier
   :type component_id: str
   :param target_temperature: Target temperature [K], default 308.15
   :type target_temperature: float
   :param heat_loss_coefficient: Heat loss [kW/K], default 0.5
   :type heat_loss_coefficient: float
   :param name: Human-readable name, optional
   :type name: str

   **Heat Loss Coefficients:**

   .. list-table::
      :header-rows: 1

      * - Insulation Quality
        - k [kW/K]
        - Description
      * - Excellent
        - 0.3-0.4
        - Modern, well-insulated
      * - Good
        - 0.4-0.6
        - Standard insulation
      * - Poor
        - 0.6-1.0
        - Old or minimal insulation

   **Outputs:**

   - ``Q_heat_supplied``: Heat delivered [kW]
   - ``P_th_used``: CHP heat used [kW]
   - ``P_aux_heat``: Auxiliary heat needed [kW]

Gas Storage
~~~~~~~~~~~

.. autoclass:: pyadm1.components.energy.gas_storage.GasStorage
   :members:
   :undoc-members:
   :show-inheritance:

   Biogas storage with pressure management. Supports multiple storage types:

   **Storage Types:**

   1. **Membrane (Low-Pressure)**:
      - Pressure: 0.95-1.05 bar
      - Capacity: 500-2000 m³ STP
      - Most common for biogas plants

   2. **Dome (Fixed)**:
      - Pressure: 0.98-1.02 bar
      - Capacity: 200-1000 m³ STP
      - Simple, robust design

   3. **Compressed (High-Pressure)**:
      - Pressure: 10-200 bar
      - Capacity: 50-500 m³ STP
      - Grid injection applications

   **Parameters:**

   :param component_id: Unique identifier
   :type component_id: str
   :param storage_type: 'membrane', 'dome', or 'compressed'
   :type storage_type: str
   :param capacity_m3: Usable volume at STP [m³]
   :type capacity_m3: float
   :param p_min_bar: Minimum pressure [bar], default 0.95
   :type p_min_bar: float
   :param p_max_bar: Maximum pressure [bar], default 1.05
   :type p_max_bar: float
   :param initial_fill_fraction: Initial fill level [0-1], default 0.1
   :type initial_fill_fraction: float

   **Outputs:**

   - ``stored_volume_m3``: Current storage [m³ STP]
   - ``pressure_bar``: Current pressure [bar]
   - ``utilization``: Fill level [0-1]
   - ``vented_volume_m3``: Gas vented this step [m³]
   - ``Q_gas_supplied_m3_per_day``: Gas supplied [m³/d]

Flare
~~~~~

.. autoclass:: pyadm1.components.energy.flare.Flare
   :members:
   :undoc-members:
   :show-inheritance:

   Safety system for excess biogas combustion.

   **Parameters:**

   :param component_id: Unique identifier
   :type component_id: str
   :param destruction_efficiency: Methane destruction [0-1], default 0.98
   :type destruction_efficiency: float
   :param name: Human-readable name, optional
   :type name: str

   **Outputs:**

   - ``vented_volume_m3``: Volume combusted this step [m³]
   - ``cumulative_vented_m3``: Total vented [m³]
   - ``CH4_destroyed_m3``: Methane destroyed [m³]

Component Patterns
------------------

Single-Stage Plant
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pyadm1.configurator import BiogasPlant
   from pyadm1.components.biological import Digester
   from pyadm1.components.energy import CHP, HeatingSystem
   from pyadm1.substrates import Feedstock

   feedstock = Feedstock(feeding_freq=48)
   plant = BiogasPlant("Single-Stage Plant")

   # Add components
   plant.add_component(Digester("dig1", feedstock, V_liq=2000))
   plant.add_component(CHP("chp1", P_el_nom=500))
   plant.add_component(HeatingSystem("heat1", target_temperature=308.15))

   # Connections handled automatically by PlantConfigurator

Two-Stage Plant
~~~~~~~~~~~~~~~

.. code-block:: python

   # Hydrolysis stage (thermophilic)
   plant.add_component(Digester(
       "hydro", feedstock,
       V_liq=500, T_ad=318.15,
       name="Hydrolysis Tank"
   ))

   # Main digestion stage (mesophilic)
   plant.add_component(Digester(
       "main", feedstock,
       V_liq=2000, T_ad=308.15,
       name="Main Digester"
   ))

   # Connect in series
   plant.add_connection(Connection("hydro", "main", "liquid"))

Custom Components
-----------------

Creating Custom Components
~~~~~~~~~~~~~~~~~~~~~~~~~~

Extend the ``Component`` base class:

.. code-block:: python

   from pyadm1.components.base import Component, ComponentType

   class CustomMixer(Component):
       def __init__(self, component_id, mixing_time=10.0):
           super().__init__(component_id, ComponentType.MIXER,
                           name=f"Mixer {component_id}")
           self.mixing_time = mixing_time

       def initialize(self, initial_state=None):
           self.state = {'mixed': False, 'time_elapsed': 0.0}

       def step(self, t, dt, inputs):
           self.state['time_elapsed'] += dt

           if self.state['time_elapsed'] >= self.mixing_time:
               self.state['mixed'] = True

           return {
               'is_mixed': self.state['mixed'],
               'mixing_progress': min(1.0,
                   self.state['time_elapsed'] / self.mixing_time)
           }

       def to_dict(self):
           return {
               'component_id': self.component_id,
               'component_type': self.component_type.value,
               'mixing_time': self.mixing_time,
               'state': self.state
           }

       @classmethod
       def from_dict(cls, config):
           mixer = cls(config['component_id'],
                      config.get('mixing_time', 10.0))
           if 'state' in config:
               mixer.state = config['state']
           return mixer

Registering Custom Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pyadm1.components import register_component

   register_component("CustomMixer", CustomMixer)

   # Now usable via registry
   from pyadm1.components import get_registry
   registry = get_registry()
   mixer = registry.create("CustomMixer", "mix1", mixing_time=15.0)

See Also
--------

- :doc:`../user_guide/components` for component usage guide
- :doc:`configurator` for plant building tools
- :doc:`core` for ADM1 implementation details

Performance Notes
-----------------

**Memory Usage:**

- Digester: ~1 KB state + 300 bytes per timestep saved
- CHP/Heating: ~100 bytes state
- Storage: ~200 bytes state

**Computational Cost:**

- Digester: O(1) per timestep (ADM1 ODE evaluation)
- CHP/Heating: O(1) per timestep (algebraic)
- Storage: O(1) per timestep (mass balance)

**Optimization Tips:**

1. Use larger time steps (dt = 1 hour typical)
2. Save results at intervals (daily) not every step
3. Initialize from steady-state when possible
4. Use parallel simulation for parameter sweeps
