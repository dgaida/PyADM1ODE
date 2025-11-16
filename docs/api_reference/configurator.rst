Plant Configuration
===================

The configurator module provides tools for building, validating, and managing
biogas plant models. It includes high-level builders, connection management,
and an MCP server for LLM-driven plant design.

Plant Builder
-------------

BiogasPlant
~~~~~~~~~~~

.. autoclass:: pyadm1.configurator.plant_builder.BiogasPlant
   :members:
   :undoc-members:
   :show-inheritance:

   Main class for complete biogas plant models. Manages component lifecycle,
   connections, and provides JSON-based configuration persistence.

   **Core Methods:**

   - ``add_component(component)``: Add component to plant
   - ``add_connection(connection)``: Define connection between components
   - ``initialize()``: Initialize all components for simulation
   - ``step(dt)``: Execute one simulation time step
   - ``simulate(duration, dt, save_interval)``: Run full simulation
   - ``to_json(filepath)``: Save configuration
   - ``from_json(filepath, feedstock)``: Load configuration

   **Example:**

   .. code-block:: python

      from pyadm1.configurator import BiogasPlant
      from pyadm1.components.biological import Digester
      from pyadm1.substrates import Feedstock

      # Create plant
      feedstock = Feedstock(feeding_freq=48)
      plant = BiogasPlant("My Plant")

      # Add components
      digester = Digester("dig1", feedstock, V_liq=2000)
      plant.add_component(digester)

      # Initialize and simulate
      plant.initialize()
      results = plant.simulate(duration=30, dt=1/24, save_interval=1.0)

      # Results structure
      for result in results:
          time = result['time']
          components = result['components']
          print(f"Day {time}: Biogas = {components['dig1']['Q_gas']:.1f} m³/d")

   **Three-Pass Execution Model:**

   The plant uses a sophisticated execution strategy to handle gas flow:

   1. **Pass 1**: Execute digesters to produce gas → storages
   2. **Pass 2**: Execute storages with production only (no demand)
   3. **Pass 3**: Execute CHPs to determine demand → re-execute storages

   This ensures proper gas supply-demand matching.

PlantConfigurator
~~~~~~~~~~~~~~~~~

.. autoclass:: pyadm1.configurator.plant_configurator.PlantConfigurator
   :members:
   :undoc-members:
   :show-inheritance:

   High-level configurator providing convenient methods for adding components
   with sensible defaults and automatic setup.

   **Key Features:**

   - Automatic gas storage creation per digester
   - Automatic flare creation per CHP
   - Helper methods for common configurations
   - Template-based plant creation

   **Example:**

   .. code-block:: python

      from pyadm1.configurator import BiogasPlant
      from pyadm1.configurator.plant_configurator import PlantConfigurator
      from pyadm1.substrates import Feedstock

      feedstock = Feedstock(feeding_freq=48)
      plant = BiogasPlant("My Plant")
      config = PlantConfigurator(plant, feedstock)

      # Add digester (automatically creates gas storage)
      config.add_digester(
          digester_id="main_digester",
          V_liq=2000,
          V_gas=300,
          T_ad=308.15,
          Q_substrates=[15, 10, 0, 0, 0, 0, 0, 0, 0, 0]
      )

      # Add CHP (automatically creates flare)
      config.add_chp(
          chp_id="chp_main",
          P_el_nom=500,
          eta_el=0.40,
          eta_th=0.45
      )

      # Add heating
      config.add_heating(
          heating_id="heating_main",
          target_temperature=308.15
      )

      # Auto-connect gas flow: digester → storage → CHP
      config.auto_connect_digester_to_chp("main_digester", "chp_main")

      # Auto-connect heat flow: CHP → heating
      config.auto_connect_chp_to_heating("chp_main", "heating_main")

   **Template Methods:**

   .. code-block:: python

      # Single-stage plant
      components = config.create_single_stage_plant(
          digester_config={'V_liq': 2000},
          chp_config={'P_el_nom': 500},
          heating_config={'target_temperature': 308.15},
          auto_connect=True
      )

      # Two-stage plant
      components = config.create_two_stage_plant(
          hydrolysis_config={'V_liq': 500, 'T_ad': 318.15},
          digester_config={'V_liq': 1500},
          chp_config={'P_el_nom': 500},
          auto_connect=True
      )

Connection Management
---------------------

Connection
~~~~~~~~~~

.. autoclass:: pyadm1.configurator.connection_manager.Connection
   :members:
   :undoc-members:
   :show-inheritance:

   Represents a directed connection between two components.

   **Example:**

   .. code-block:: python

      from pyadm1.configurator.connection_manager import Connection

      # Liquid flow connection
      conn = Connection("digester_1", "digester_2", "liquid")
      plant.add_connection(conn)

      # Gas flow connection
      conn = Connection("digester_storage", "chp_1", "gas")
      plant.add_connection(conn)

      # Heat flow connection
      conn = Connection("chp_1", "heating_1", "heat")
      plant.add_connection(conn)

ConnectionType
~~~~~~~~~~~~~~

.. autoclass:: pyadm1.configurator.connection_manager.ConnectionType
   :members:
   :undoc-members:
   :show-inheritance:

   Enumeration of connection types:

   - ``LIQUID``: Liquid flow (digestate)
   - ``GAS``: Biogas flow
   - ``HEAT``: Heat transfer
   - ``POWER``: Electrical power
   - ``CONTROL``: Control signals
   - ``DEFAULT``: Generic connection

ConnectionManager
~~~~~~~~~~~~~~~~~

.. autoclass:: pyadm1.configurator.connection_manager.ConnectionManager
   :members:
   :undoc-members:
   :show-inheritance:

   Manages connections with validation and dependency resolution.

   **Key Methods:**

   .. code-block:: python

      from pyadm1.configurator.connection_manager import ConnectionManager

      manager = ConnectionManager()

      # Add connections
      manager.add_connection(Connection("dig1", "chp1", "gas"))

      # Query connections
      outgoing = manager.get_connections_from("dig1")
      incoming = manager.get_connections_to("chp1")

      # Dependency analysis
      deps = manager.get_dependencies("chp1")  # ['dig1']
      order = manager.get_execution_order(['dig1', 'chp1', 'heat1'])

      # Validation
      errors = manager.validate_connections(['dig1', 'chp1'])
      if errors:
          print("Connection errors:", errors)

MCP Server (LLM Integration)
-----------------------------

Server
~~~~~~

.. automodule:: pyadm1.configurator.mcp.server
   :members:
   :undoc-members:
   :show-inheritance:

   FastMCP server providing tools for LLM-driven biogas plant design.

   **Available Tools:**

   1. **create_plant**: Create new biogas plant
   2. **add_digester_component**: Add digester with ADM1
   3. **add_chp_unit**: Add CHP for power generation
   4. **add_heating_system**: Add heating for temperature control
   5. **connect_components**: Define connections between components
   6. **initialize_biogas_plant**: Prepare plant for simulation
   7. **simulate_biogas_plant**: Run simulation
   8. **get_biogas_plant_status**: View plant details
   9. **export_biogas_plant_config**: Save configuration to JSON
   10. **list_biogas_plants**: List all plants in registry
   11. **delete_biogas_plant**: Remove plant from registry

   **Available Prompts:**

   1. **system_guidance**: System-level guidance for plant design
   2. **component_selection**: Component sizing guidelines
   3. **connection_guidelines**: Connection rules and patterns
   4. **parameter_guidelines**: Parameter ranges and defaults
   5. **substrate_guide**: Substrate-specific recommendations
   6. **design_best_practices**: Best practices for design and operation
   7. **troubleshooting**: Common issues and solutions
   8. **example_plants**: Example configurations with tool sequences

   **Starting the Server:**

   .. code-block:: bash

      # Start MCP server
      python -m pyadm1.configurator.mcp.server

      # Or programmatically
      from pyadm1.configurator.mcp.server import start_server
      start_server(host="127.0.0.1", port=8000)

   **Server Endpoints:**

   - Base URL: ``http://127.0.0.1:8000``
   - SSE endpoint: ``http://127.0.0.1:8000/sse``

   **Claude Desktop Integration:**

   Add to ``claude_desktop_config.json``:

   .. code-block:: json

      {
        "mcpServers": {
          "pyadm1": {
            "command": "python",
            "args": ["-m", "pyadm1.configurator.mcp.server"],
            "env": {}
          }
        }
      }

Tools
~~~~~

.. automodule:: pyadm1.configurator.mcp.tools
   :members:
   :undoc-members:

   Tool implementations for MCP server.

   **PlantRegistry:**

   The registry maintains stateful plant instances across multiple tool calls:

   .. code-block:: python

      from pyadm1.configurator.mcp.tools import get_registry

      registry = get_registry()

      # Create plant
      plant_id = registry.create_plant("MyPlant", feeding_freq=48)

      # Get plant instance
      plant = registry.get_plant("MyPlant")
      configurator = registry.get_configurator("MyPlant")

      # List all plants
      plant_ids = registry.list_plants()

Prompts
~~~~~~~

.. automodule:: pyadm1.configurator.mcp.prompts
   :members:
   :undoc-members:

   System prompts and templates for LLM guidance.

   **Key Prompts:**

   - ``SYSTEM_PROMPT``: Main system guidance
   - ``COMPONENT_SELECTION_GUIDE``: Sizing rules and selection criteria
   - ``CONNECTION_RULES``: Connection patterns and validation
   - ``PARAMETER_RANGES``: Parameter bounds and defaults
   - ``SUBSTRATE_RECOMMENDATIONS``: Substrate-specific design
   - ``BEST_PRACTICES``: Design and operational guidelines
   - ``TROUBLESHOOTING_GUIDE``: Common issues and solutions
   - ``EXAMPLE_CONFIGURATIONS``: Complete example plants

   **Helper Functions:**

   .. code-block:: python

      from pyadm1.configurator.mcp.prompts import (
          get_prompt_for_plant_type,
          get_substrate_guidance,
          get_parameter_recommendation
      )

      # Get plant type guidance
      guidance = get_prompt_for_plant_type("two_stage")

      # Get substrate guidance
      substrate_info = get_substrate_guidance("corn_silage")

      # Get parameter recommendations
      params = get_parameter_recommendation("V_liq", plant_size="medium")
      # Returns: {'default': 2000, 'min': 1500, 'max': 3000}

Schemas
~~~~~~~

.. automodule:: pyadm1.configurator.mcp.schemas
   :members:
   :undoc-members:

   Pydantic data schemas for request/response validation.

   **Request Schemas:**

   - ``PlantCreateRequest``: Create plant parameters
   - ``DigesterAddRequest``: Digester parameters with validation
   - ``CHPAddRequest``: CHP parameters with efficiency checks
   - ``HeatingAddRequest``: Heating system parameters
   - ``ConnectionAddRequest``: Connection parameters
   - ``SimulationRequest``: Simulation parameters
   - ``PlantExportRequest``: Export parameters

   **Response Schemas:**

   - ``PlantCreateResponse``: Plant creation confirmation
   - ``PlantStatusResponse``: Complete plant status
   - ``SimulationResponse``: Simulation results summary
   - ``ErrorResponse``: Standardized error information

   **Validation Functions:**

   .. code-block:: python

      from pyadm1.configurator.mcp.schemas import (
          validate_substrate_mix,
          calculate_hrt,
          calculate_olr
      )

      # Validate substrate mix
      Q = [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]
      is_valid = validate_substrate_mix(Q)  # Raises ValueError if invalid

      # Calculate HRT
      hrt = calculate_hrt(V_liq=2000, Q_total=25)  # 80 days

      # Calculate OLR
      olr = calculate_olr(
          Q=[15, 10, 0, 0, 0, 0, 0, 0, 0, 0],
          V_liq=2000,
          VS_content=0.08
      )  # kg VS/(m³·d)

MCP Client
~~~~~~~~~~

.. autoclass:: pyadm1.configurator.mcp.client.IntelligentBiogasClient
   :members:
   :undoc-members:
   :show-inheritance:

   Intelligent MCP client for building plants from natural language descriptions.

   **Example:**

   .. code-block:: python

      import asyncio
      from pyadm1.configurator.mcp.client import IntelligentBiogasClient

      async def main():
          client = IntelligentBiogasClient("http://127.0.0.1:8000")

          try:
              await client.connect()

              # Build plant from description
              description = (
                  "Create a MyFarm biogas plant with a single-stage "
                  "2000 m³ digester, 500 kW CHP unit, and heating system. "
                  "Simulate for 30 days."
              )

              result = await client.build_plant_from_description(description)
              print(result)

              # Export configuration
              export_result = await client.export_configuration()

          finally:
              await client.disconnect()

      asyncio.run(main())

Usage Examples
--------------

Programmatic Plant Building
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pyadm1.configurator import BiogasPlant
   from pyadm1.configurator.plant_configurator import PlantConfigurator
   from pyadm1.substrates import Feedstock

   # Setup
   feedstock = Feedstock(feeding_freq=48)
   plant = BiogasPlant("Complete Plant")
   config = PlantConfigurator(plant, feedstock)

   # Add components
   config.add_digester(
       "main_digester",
       V_liq=2000,
       V_gas=300,
       Q_substrates=[15, 10, 0, 0, 0, 0, 0, 0, 0, 0]
   )
   config.add_chp("chp_main", P_el_nom=500)
   config.add_heating("heating_main", target_temperature=308.15)

   # Connect
   config.auto_connect_digester_to_chp("main_digester", "chp_main")
   config.auto_connect_chp_to_heating("chp_main", "heating_main")

   # Initialize and simulate
   plant.initialize()
   results = plant.simulate(duration=30, dt=1/24, save_interval=1.0)

   # Save configuration
   plant.to_json("my_plant.json")

Loading and Modifying
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pyadm1.configurator import BiogasPlant
   from pyadm1.substrates import Feedstock

   # Load existing plant
   feedstock = Feedstock(feeding_freq=48)
   plant = BiogasPlant.from_json("my_plant.json", feedstock)

   # Modify configuration
   plant.components["main_digester"].V_liq = 2500

   # Re-initialize and simulate
   plant.initialize()
   results = plant.simulate(duration=30, dt=1/24)

LLM-Driven Plant Design
~~~~~~~~~~~~~~~~~~~~~~~~

Using Claude Desktop with MCP:

1. Start the MCP server
2. Configure Claude Desktop
3. Use natural language to design plants

Example prompts:

- "Create a single-stage biogas plant with 2000 m³ digester and 500 kW CHP"
- "Add a two-stage configuration with thermophilic hydrolysis"
- "Connect all components and run a 30-day simulation"
- "Export the configuration to a JSON file"

See Also
--------

- :doc:`../user_guide/quickstart` for practical tutorials
- :doc:`components` for component details
- :doc:`../examples/two_stage_simulation` for complete example

Best Practices
--------------

**Configuration:**

1. Use PlantConfigurator for high-level building
2. Let auto-connection methods handle gas/heat routing
3. Validate configuration before simulation
4. Save configurations to JSON for reproducibility

**Simulation:**

1. Initialize from steady-state when possible
2. Use appropriate time steps (1 hour typical)
3. Save results at intervals (daily) not every step
4. Monitor process stability (pH, VFA/TAC)

**MCP Usage:**

1. Start server before using LLM integration
2. Use clear, detailed plant descriptions
3. Review generated configurations before simulation
4. Export important configurations for later use

**Performance:**

1. Reuse plant instances when possible
2. Avoid unnecessary re-initialization
3. Use parallel simulation for parameter studies
4. Profile long simulations to identify bottlenecks
