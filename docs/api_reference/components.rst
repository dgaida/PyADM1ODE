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

Mechanical Components
---------------------

Pump
~~~~

.. autoclass:: pyadm1.components.mechanical.pump.Pump
   :members:
   :undoc-members:
   :show-inheritance:

   Pump component for material handling in biogas plants.

   Models different pump types for substrate feeding, recirculation, and
   digestate transfer with realistic power consumption based on flow rate,
   pressure head, and pump efficiency.

   **Parameters:**

   :param component_id: Unique identifier
   :type component_id: str
   :param pump_type: Type of pump ("centrifugal", "progressive_cavity", "piston")
   :type pump_type: str
   :param Q_nom: Nominal flow rate [m³/h], default 10.0
   :type Q_nom: float
   :param pressure_head: Design pressure head [m], default 50.0
   :type pressure_head: float
   :param efficiency: Pump efficiency (0-1), calculated if None
   :type efficiency: Optional[float]
   :param motor_efficiency: Motor efficiency (0-1), default 0.90
   :type motor_efficiency: float
   :param fluid_density: Fluid density [kg/m³], default 1020.0
   :type fluid_density: float
   :param speed_control: Enable variable speed drive, default True
   :type speed_control: bool
   :param name: Human-readable name, optional
   :type name: str

   **Pump Types:**

   .. list-table::
      :header-rows: 1

      * - Type
        - Best For
        - Efficiency
        - Flow Range [m³/h]
      * - Centrifugal
        - Low viscosity liquids
        - 65-75%
        - 5-200
      * - Progressive Cavity
        - Viscous slurries
        - 50-70%
        - 1-100
      * - Piston
        - High pressure
        - 70-85%
        - 1-50

   **Outputs:**

   .. code-block:: python

      {
          'P_consumed': 8.5,           # Power consumption [kW]
          'Q_actual': 10.0,            # Actual flow rate [m³/h]
          'is_running': True,          # Current running state
          'efficiency': 0.68,          # Operating efficiency
          'pressure_actual': 48.5,     # Actual pressure head [m]
          'speed_fraction': 1.0,       # Speed as fraction of nominal
          'specific_energy': 0.85      # Energy per volume [kWh/m³]
      }

   **Example:**

   .. code-block:: python

      from pyadm1.components.mechanical import Pump

      # Progressive cavity pump for substrate feeding
      pump = Pump(
          component_id="feed_pump",
          pump_type="progressive_cavity",
          Q_nom=15.0,
          pressure_head=50.0
      )

      pump.initialize()

      # Simulate pumping at 80% capacity
      result = pump.step(
          t=0,
          dt=1/24,
          inputs={'Q_setpoint': 12.0, 'enable_pump': True}
      )

      print(f"Power: {result['P_consumed']:.1f} kW")
      print(f"Flow: {result['Q_actual']:.1f} m³/h")

Mixer
~~~~~

.. autoclass:: pyadm1.components.mechanical.mixer.Mixer
   :members:
   :undoc-members:
   :show-inheritance:

   Mixer/agitator component for biogas digesters.

   Models mechanical or hydraulic mixing systems that maintain homogeneity
   in anaerobic digesters with power consumption based on mixer type,
   operating conditions, and fluid properties.

   **Parameters:**

   :param component_id: Unique identifier
   :type component_id: str
   :param mixer_type: Type of mixer ("propeller", "paddle", "jet")
   :type mixer_type: str
   :param tank_volume: Tank liquid volume [m³], default 2000.0
   :type tank_volume: float
   :param tank_diameter: Tank diameter [m], calculated if None
   :type tank_diameter: Optional[float]
   :param tank_height: Tank height [m], calculated if None
   :type tank_height: Optional[float]
   :param mixing_intensity: Intensity level ("low", "medium", "high")
   :type mixing_intensity: str
   :param power_installed: Installed power [kW], calculated if None
   :type power_installed: Optional[float]
   :param intermittent: Enable intermittent operation, default True
   :type intermittent: bool
   :param on_time_fraction: Fraction of time mixer is on (0-1), default 0.25
   :type on_time_fraction: float
   :param name: Human-readable name, optional
   :type name: str

   **Mixing Intensity Guidelines:**

   .. list-table::
      :header-rows: 1

      * - Intensity
        - Specific Power [W/m³]
        - Mixing Time [min]
        - Application
      * - Low
        - 3
        - 15-30
        - Liquid substrates
      * - Medium
        - 5
        - 8-15
        - Standard operation
      * - High
        - 8
        - 3-8
        - High-solids substrates

   **Outputs:**

   .. code-block:: python

      {
          'P_consumed': 12.5,          # Power consumption [kW]
          'P_average': 3.1,            # Time-averaged power [kW]
          'is_running': True,          # Current running state
          'mixing_quality': 0.85,      # Mixing quality (0-1)
          'reynolds_number': 15000,    # Reynolds number
          'power_number': 0.32,        # Power number
          'mixing_time': 8.5,          # Mixing time [min]
          'shear_rate': 45.2,          # Average shear rate [1/s]
          'specific_power': 6.25,      # Power per volume [kW/m³]
          'tip_speed': 2.8             # Impeller tip speed [m/s]
      }

   **Example:**

   .. code-block:: python

      from pyadm1.components.mechanical import Mixer

      # Propeller mixer for 2000 m³ digester
      mixer = Mixer(
          component_id="mix1",
          mixer_type="propeller",
          tank_volume=2000,
          mixing_intensity="medium",
          power_installed=15.0,
          intermittent=True,
          on_time_fraction=0.25
      )

      mixer.initialize()

      result = mixer.step(
          t=0,
          dt=1/24,
          inputs={'enable_mixing': True, 'speed_setpoint': 1.0}
      )

      print(f"Power: {result['P_consumed']:.1f} kW")
      print(f"Mixing quality: {result['mixing_quality']:.2f}")
      print(f"Mixing time: {result['mixing_time']:.1f} min")

Feeding Components
------------------

SubstrateStorage
~~~~~~~~~~~~~~~~

.. autoclass:: pyadm1.components.feeding.substrate_storage.SubstrateStorage
   :members:
   :undoc-members:
   :show-inheritance:

   Storage facility component for biogas plant substrates.

   Models storage of different substrate types with inventory tracking,
   quality degradation, and capacity management for both solid (silage,
   solid manure) and liquid (liquid manure, slurry) substrates.

   **Parameters:**

   :param component_id: Unique identifier
   :type component_id: str
   :param storage_type: Type of storage ("vertical_silo", "horizontal_silo",
                       "bunker_silo", "above_ground_tank", "below_ground_tank",
                       "clamp", "pile")
   :type storage_type: str
   :param substrate_type: Substrate category ("corn_silage", "grass_silage",
                         "manure_liquid", etc.)
   :type substrate_type: str
   :param capacity: Maximum capacity [t or m³], default 1000.0
   :type capacity: float
   :param initial_level: Initial inventory [t or m³], default 0.0
   :type initial_level: float
   :param degradation_rate: Quality degradation rate [1/d], auto-calculated if None
   :type degradation_rate: Optional[float]
   :param temperature: Storage temperature [K], default 288.15 (15°C)
   :type temperature: float
   :param name: Human-readable name, optional
   :type name: str

   **Storage Types:**

   .. list-table::
      :header-rows: 1

      * - Type
        - Degradation [1/d]
        - Best For
        - Typical Capacity
      * - Vertical Silo
        - 0.0005
        - Corn/grass silage
        - 500-2000 t
      * - Bunker Silo
        - 0.001
        - Large-scale silage
        - 1000-5000 t
      * - Tank (above)
        - 0.0002
        - Liquid manure
        - 500-3000 m³
      * - Tank (below)
        - 0.0001
        - Liquid storage
        - 1000-5000 m³

   **Outputs:**

   .. code-block:: python

      {
          'current_level': 750.0,      # Current inventory [t or m³]
          'utilization': 0.75,         # Fill level (0-1)
          'quality_factor': 0.95,      # Current quality (0-1)
          'available_mass': 712.5,     # Usable inventory [t or m³]
          'degradation_rate': 0.0005,  # Current rate [1/d]
          'losses_this_step': 0.4,     # Mass lost [t or m³]
          'withdrawn_this_step': 15.0, # Mass withdrawn [t or m³]
          'is_empty': False,           # Storage empty flag
          'is_full': False,            # Storage full flag
          'storage_time': 25.5,        # Time stored [days]
          'dry_matter': 35.0,          # DM content [%]
          'vs_content': 95.0           # VS content [% of DM]
      }

   **Example:**

   .. code-block:: python

      from pyadm1.components.feeding import SubstrateStorage

      # Corn silage vertical silo
      storage = SubstrateStorage(
          component_id="silo1",
          storage_type="vertical_silo",
          substrate_type="corn_silage",
          capacity=1000,
          initial_level=800
      )

      storage.initialize()

      # Withdraw substrate
      result = storage.step(
          t=10,
          dt=1,
          inputs={'withdrawal_rate': 15}
      )

      print(f"Level: {result['current_level']:.1f} t")
      print(f"Quality: {result['quality_factor']:.3f}")
      print(f"Available: {result['available_mass']:.1f} t")

Feeder
~~~~~~

.. autoclass:: pyadm1.components.feeding.feeder.Feeder
   :members:
   :undoc-members:
   :show-inheritance:

   Feeder component for automated substrate dosing.

   Models feeding systems that transfer substrates from storage to digesters
   with realistic operational characteristics like dosing accuracy, capacity
   limits, and power consumption.

   **Parameters:**

   :param component_id: Unique identifier
   :type component_id: str
   :param feeder_type: Type of feeder ("screw", "twin_screw", "progressive_cavity",
                      "piston", "centrifugal_pump", "mixer_wagon")
   :type feeder_type: Optional[str]
   :param Q_max: Maximum flow rate [m³/d or t/d], default 20.0
   :type Q_max: float
   :param substrate_type: Substrate category ("solid", "slurry", "liquid", "fibrous")
   :type substrate_type: str
   :param dosing_accuracy: Standard deviation of flow as fraction, auto if None
   :type dosing_accuracy: Optional[float]
   :param power_installed: Installed power [kW], auto-calculated if None
   :type power_installed: Optional[float]
   :param enable_dosing_noise: Add realistic dosing variance, default True
   :type enable_dosing_noise: bool
   :param name: Human-readable name, optional
   :type name: str

   **Feeder Types:**

   .. list-table::
      :header-rows: 1

      * - Type
        - Accuracy [±%]
        - Best For
        - Power [kW]
      * - Screw
        - 5
        - Solid substrates
        - 0.8/m³/h
      * - Twin Screw
        - 3
        - Better control
        - 1.0/m³/h
      * - Progressive Cavity
        - 2
        - Slurries
        - 1.2/m³/h
      * - Piston
        - 1
        - Precise dosing
        - 1.5/m³/h

   **Outputs:**

   .. code-block:: python

      {
          'Q_actual': 14.8,            # Actual flow rate [m³/d or t/d]
          'is_running': True,          # Current operating state
          'load_factor': 0.74,         # Operating load (0-1)
          'P_consumed': 2.5,           # Power consumption [kW]
          'blockage_detected': False,  # Blockage alarm
          'dosing_error': 1.3,         # Deviation from setpoint [%]
          'speed_fraction': 0.95,      # Current speed fraction
          'dosing_accuracy': 0.05,     # Accuracy (std dev)
          'total_mass_fed': 1250.0     # Cumulative mass [t or m³]
      }

   **Example:**

   .. code-block:: python

      from pyadm1.components.feeding import Feeder

      # Screw feeder for corn silage
      feeder = Feeder(
          component_id="feed1",
          feeder_type="screw",
          Q_max=20.0,
          substrate_type="solid"
      )

      feeder.initialize()

      result = feeder.step(
          t=0,
          dt=1/24,
          inputs={
              'Q_setpoint': 15.0,
              'enable_feeding': True,
              'substrate_available': 500
          }
      )

      print(f"Flow: {result['Q_actual']:.2f} m³/d")
      print(f"Power: {result['P_consumed']:.2f} kW")
      print(f"Error: {result['dosing_error']:.1f}%")

Component Integration Examples
-------------------------------

Complete Feeding System
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pyadm1.configurator import BiogasPlant, PlantConfigurator
   from pyadm1.components.feeding import SubstrateStorage, Feeder
   from pyadm1.components.mechanical import Pump, Mixer
   from pyadm1.substrates import Feedstock

   # Setup
   feedstock = Feedstock(feeding_freq=48)
   plant = BiogasPlant("Complete Plant with Feeding")
   config = PlantConfigurator(plant, feedstock)

   # Substrate storage
   storage = SubstrateStorage(
       component_id="silo1",
       storage_type="vertical_silo",
       substrate_type="corn_silage",
       capacity=1000,
       initial_level=800
   )
   plant.add_component(storage)

   # Feeder
   feeder = Feeder(
       component_id="feed1",
       feeder_type="screw",
       Q_max=20.0
   )
   plant.add_component(feeder)

   # Feed pump
   pump = Pump(
       component_id="pump1",
       pump_type="progressive_cavity",
       Q_nom=15.0
   )
   plant.add_component(pump)

   # Digester with mixer
   digester, _ = config.add_digester(
       "main_digester",
       V_liq=2000,
       Q_substrates=[15, 10, 0, 0, 0, 0, 0, 0, 0, 0]
   )

   mixer = Mixer(
       component_id="mix1",
       mixer_type="propeller",
       tank_volume=2000,
       mixing_intensity="medium"
   )
   plant.add_component(mixer)

   # Connect: storage → feeder → pump → digester
   config.connect("silo1", "feed1", "default")
   config.connect("feed1", "pump1", "default")
   config.connect("pump1", "main_digester", "liquid")

   # Initialize and simulate
   plant.initialize()
   results = plant.simulate(duration=30, dt=1/24)

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Optimize mixer operation
   mixer_results = []

   for intensity in ["low", "medium", "high"]:
       for on_time in [0.2, 0.25, 0.3]:
           mixer = Mixer(
               f"mix_{intensity}_{int(on_time*100)}",
               mixer_type="propeller",
               tank_volume=2000,
               mixing_intensity=intensity,
               on_time_fraction=on_time
           )

           # Simulate
           result = mixer.step(0, 1/24, {'enable_mixing': True})

           mixer_results.append({
               'intensity': intensity,
               'on_time': on_time,
               'power': result['P_average'],
               'quality': result['mixing_quality'],
               'energy_per_quality': result['P_average'] / result['mixing_quality']
           })

   # Find optimal configuration
   optimal = min(mixer_results, key=lambda x: x['energy_per_quality'])
   print(f"Optimal: {optimal['intensity']} intensity, "
         f"{optimal['on_time']:.0%} on-time")

See Also
--------

- :doc:`../user_guide/components` for component usage guide
- :doc:`configurator` for plant building tools
- :doc:`../examples/two_stage_plant` for complete example

Performance Notes
-----------------

**Component Resource Usage:**

.. list-table::
   :header-rows: 1

   * - Component
     - Memory [bytes]
     - CPU per Step
     - Typical dt [days]
   * - Pump
     - ~200
     - O(1)
     - 1/24 (1 hour)
   * - Mixer
     - ~300
     - O(1)
     - 1/24 (1 hour)
   * - Storage
     - ~250
     - O(1)
     - 1 (1 day)
   * - Feeder
     - ~300
     - O(1)
     - 1/24 (1 hour)

**Optimization Tips:**

1. Use intermittent operation for mixers (25% on-time typical)
2. Size pumps for 80-90% of maximum expected flow
3. Monitor storage quality degradation for optimal scheduling
4. Enable dosing noise for realistic simulations

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
