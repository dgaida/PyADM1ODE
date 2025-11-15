# pyadm1/configurator/mcp/server.py
"""
PyADM1 MCP Server with SSE Transport

FastMCP server providing tools for biogas plant modeling with stateful plant management.
This server exposes tools for LLM-driven plant design, configuration, and simulation.

The server handles:
- Tool registration and serving via FastMCP
- SSE (Server-Sent Events) transport for real-time communication
- System prompts for LLM guidance
- Error handling and logging

All business logic is delegated to the tools module, keeping this file focused
on server configuration and protocol handling.

Usage:
    # Start server
    python -m pyadm1.configurator.mcp.server

    # Or programmatically
    from pyadm1.configurator.mcp.server import start_server
    start_server(host="127.0.0.1", port=8000)

MCP Client Configuration (Claude Desktop):
    Add to claude_desktop_config.json:
    {
      "mcpServers": {
        "pyadm1": {
          "command": "python",
          "args": ["-m", "pyadm1.configurator.mcp.server"],
          "env": {}
        }
      }
    }
"""

from typing import Optional
from fastmcp import FastMCP

# Import all tools from the tools module
from pyadm1.configurator.mcp.tools import (
    create_biogas_plant,
    add_digester,
    add_chp,
    add_heating,
    add_connection,
    initialize_plant,
    simulate_plant,
    get_plant_status,
    export_plant_config,
    list_plants,
    delete_plant,
)

# Import prompts for LLM guidance
from pyadm1.configurator.mcp.prompts import (
    SYSTEM_PROMPT,
    COMPONENT_SELECTION_GUIDE,
    CONNECTION_RULES,
    PARAMETER_RANGES,
    SUBSTRATE_RECOMMENDATIONS,
    BEST_PRACTICES,
    TROUBLESHOOTING_GUIDE,
    EXAMPLE_CONFIGURATIONS,
)

# Import schemas for documentation (FastMCP can use them for auto-documentation)

# ==============================================================================
# Initialize FastMCP Server
# ==============================================================================

mcp = FastMCP(
    name="PyADM1 Biogas Plant Server",
    version="1.0.0",
    description=(
        "MCP server for designing, configuring, and simulating biogas plants "
        "using the PyADM1 framework. Provides tools for component-based plant "
        "modeling with ADM1 simulation capabilities."
    ),
)


# ==============================================================================
# Register System Prompts
# ==============================================================================


@mcp.prompt()
def system_guidance() -> str:
    """
    Get system-level guidance for biogas plant design.

    Returns comprehensive information about:
    - Role and capabilities
    - Design philosophy
    - Best practices
    - Parameter guidelines

    Returns:
        str: Complete system prompt for LLM guidance
    """
    return SYSTEM_PROMPT


@mcp.prompt()
def component_selection() -> str:
    """
    Get detailed guidelines for component selection and sizing.

    Returns:
        str: Component selection guide with sizing rules
    """
    return COMPONENT_SELECTION_GUIDE


@mcp.prompt()
def connection_guidelines() -> str:
    """
    Get rules and patterns for connecting components.

    Returns:
        str: Connection rules and validation guidelines
    """
    return CONNECTION_RULES


@mcp.prompt()
def parameter_guidelines() -> str:
    """
    Get parameter ranges and default values for all components.

    Returns:
        str: Parameter ranges and recommendations
    """
    return PARAMETER_RANGES


@mcp.prompt()
def substrate_guide() -> str:
    """
    Get substrate-specific design recommendations.

    Returns:
        str: Substrate recommendations and feeding strategies
    """
    return SUBSTRATE_RECOMMENDATIONS


@mcp.prompt()
def design_best_practices() -> str:
    """
    Get best practices for plant design and operation.

    Returns:
        str: Best practices guide
    """
    return BEST_PRACTICES


@mcp.prompt()
def troubleshooting() -> str:
    """
    Get troubleshooting guide for common issues.

    Returns:
        str: Troubleshooting guide
    """
    return TROUBLESHOOTING_GUIDE


@mcp.prompt()
def example_plants() -> str:
    """
    Get example plant configurations with complete tool sequences.

    Returns:
        str: Example configurations
    """
    return EXAMPLE_CONFIGURATIONS


# ==============================================================================
# Register Tools
# ==============================================================================


@mcp.tool()
def create_plant(plant_id: str, description: Optional[str] = None, feeding_freq: int = 48) -> str:
    """
    Create a new biogas plant model.

    This is the first step in building a plant. After creating the plant,
    you can add components (digesters, CHP units, etc.) and connect them.

    Args:
        plant_id: Unique identifier for the plant (e.g., "FarmAB_Plant")
        description: Optional description of the plant
        feeding_freq: Feeding frequency in hours (default: 48)

    Returns:
        Detailed status message including next steps
    """
    return create_biogas_plant(plant_id, description, feeding_freq)


@mcp.tool()
def add_digester_component(
    plant_id: str,
    digester_id: str,
    V_liq: float = 1977.0,
    V_gas: float = 304.0,
    T_ad: float = 308.15,
    name: Optional[str] = None,
    load_initial_state: bool = True,
) -> str:
    """
    Add a digester component to the plant.

    A digester is the main biological reactor where anaerobic digestion occurs.
    Multiple digesters can be added and connected in series or parallel.

    Args:
        plant_id: Plant identifier
        digester_id: Unique ID for this digester (e.g., "main_digester")
        V_liq: Liquid volume in m³ (default: 1977.0)
        V_gas: Gas volume in m³ (default: 304.0)
        T_ad: Operating temperature in K (default: 308.15 = 35°C)
        name: Human-readable name (optional)
        load_initial_state: Load default initial state (default: True)

    Returns:
        Detailed status with component information and next steps
    """
    return add_digester(plant_id, digester_id, V_liq, V_gas, T_ad, name, load_initial_state)


@mcp.tool()
def add_chp_unit(
    plant_id: str, chp_id: str, P_el_nom: float = 500.0, eta_el: float = 0.40, eta_th: float = 0.45, name: Optional[str] = None
) -> str:
    """
    Add a Combined Heat and Power (CHP) unit to the plant.

    The CHP converts biogas to electricity and heat. It should be connected
    to digesters via gas connections.

    Args:
        plant_id: Plant identifier
        chp_id: Unique ID for this CHP unit (e.g., "chp_main")
        P_el_nom: Nominal electrical power in kW (default: 500.0)
        eta_el: Electrical efficiency 0-1 (default: 0.40 = 40%)
        eta_th: Thermal efficiency 0-1 (default: 0.45 = 45%)
        name: Human-readable name (optional)

    Returns:
        Detailed status with CHP specifications and next steps
    """
    return add_chp(plant_id, chp_id, P_el_nom, eta_el, eta_th, name)


@mcp.tool()
def add_heating_system(
    plant_id: str,
    heating_id: str,
    target_temperature: float = 308.15,
    heat_loss_coefficient: float = 0.5,
    name: Optional[str] = None,
) -> str:
    """
    Add a heating system to maintain digester temperature.

    The heating system uses waste heat from CHP and/or auxiliary heating
    to maintain the target digester temperature.

    Args:
        plant_id: Plant identifier
        heating_id: Unique ID for heating system (e.g., "heating_main")
        target_temperature: Target temperature in K (default: 308.15 = 35°C)
        heat_loss_coefficient: Heat loss in kW/K (default: 0.5)
        name: Human-readable name (optional)

    Returns:
        Detailed status with heating specifications
    """
    return add_heating(plant_id, heating_id, target_temperature, heat_loss_coefficient, name)


@mcp.tool()
def connect_components(plant_id: str, from_component: str, to_component: str, connection_type: str = "default") -> str:
    """
    Connect two components in the plant.

    Creates a directed connection from one component to another, defining
    how material or energy flows between them.

    Args:
        plant_id: Plant identifier
        from_component: Source component ID
        to_component: Target component ID
        connection_type: Type of connection:
            - 'liquid': Liquid flow (digestate between digesters)
            - 'gas': Biogas flow (digester to CHP)
            - 'heat': Heat flow (CHP to heating)
            - 'power': Electrical power
            - 'default': Generic connection

    Returns:
        Detailed connection information and topology overview
    """
    return add_connection(plant_id, from_component, to_component, connection_type)


@mcp.tool()
def initialize_biogas_plant(plant_id: str) -> str:
    """
    Initialize the plant before simulation.

    This prepares all components for simulation and validates the configuration.
    Must be called before running any simulations.

    Args:
        plant_id: Plant identifier

    Returns:
        Initialization status and validation results
    """
    return initialize_plant(plant_id)


@mcp.tool()
def simulate_biogas_plant(plant_id: str, duration: float = 10.0, dt: float = 0.04167, save_interval: float = 1.0) -> str:
    """
    Run a simulation of the biogas plant.

    Simulates the plant operation for the specified duration and returns
    key performance indicators.

    Args:
        plant_id: Plant identifier
        duration: Simulation duration in days (default: 10.0)
        dt: Time step in days (default: 0.04167 = 1 hour)
        save_interval: How often to save results in days (default: 1.0 = daily)

    Returns:
        Detailed simulation results and performance metrics
    """
    return simulate_plant(plant_id, duration, dt, save_interval)


@mcp.tool()
def get_biogas_plant_status(plant_id: str) -> str:
    """
    Get comprehensive status of a biogas plant.

    Returns detailed information about the plant configuration, components,
    connections, and simulation history.

    Args:
        plant_id: Plant identifier

    Returns:
        Complete plant status report
    """
    return get_plant_status(plant_id)


@mcp.tool()
def export_biogas_plant_config(plant_id: str, filepath: Optional[str] = None) -> str:
    """
    Export plant configuration to JSON file.

    Saves the complete plant configuration including all components and
    connections for later reuse or documentation.

    Args:
        plant_id: Plant identifier
        filepath: Output file path (default: auto-generated)

    Returns:
        Export status and file location
    """
    return export_plant_config(plant_id, filepath)


@mcp.tool()
def list_biogas_plants() -> str:
    """
    List all biogas plants in the registry.

    Returns:
        Summary of all available plants
    """
    return list_plants()


@mcp.tool()
def delete_biogas_plant(plant_id: str) -> str:
    """
    Delete a biogas plant from the registry.

    Args:
        plant_id: Plant identifier to delete

    Returns:
        Deletion confirmation
    """
    return delete_plant(plant_id)


# ==============================================================================
# Server Entry Point
# ==============================================================================


def start_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """
    Start the FastMCP server with SSE transport.

    This function initializes and runs the MCP server, making all tools
    available to MCP clients (like Claude Desktop or custom applications).

    Args:
        host: Host address (default: 127.0.0.1)
        port: Port number (default: 8000)

    Example:
        >>> from pyadm1.configurator.mcp.server import start_server
        >>> start_server(host="0.0.0.0", port=8000)
    """
    print("=" * 70)
    print("PyADM1 MCP Server - Biogas Plant Modeling")
    print("=" * 70)
    print(f"\nServer starting at: http://{host}:{port}")
    print(f"SSE endpoint: http://{host}:{port}/sse")
    print("\nAvailable Tools:")
    print("  Plant Management:")
    print("    - create_plant: Create new biogas plant")
    print("    - list_biogas_plants: List all plants")
    print("    - get_biogas_plant_status: View plant details")
    print("    - delete_biogas_plant: Remove plant")
    print("    - export_biogas_plant_config: Export to JSON")
    print("\n  Component Addition:")
    print("    - add_digester_component: Add digester")
    print("    - add_chp_unit: Add CHP unit")
    print("    - add_heating_system: Add heating system")
    print("\n  Plant Configuration:")
    print("    - connect_components: Connect components")
    print("    - initialize_biogas_plant: Initialize for simulation")
    print("\n  Simulation:")
    print("    - simulate_biogas_plant: Run simulation")
    print("\nAvailable Prompts:")
    print("  - system_guidance: System-level guidance")
    print("  - component_selection: Component sizing guidelines")
    print("  - connection_guidelines: Connection rules")
    print("  - parameter_guidelines: Parameter ranges")
    print("  - substrate_guide: Substrate recommendations")
    print("  - design_best_practices: Best practices")
    print("  - troubleshooting: Troubleshooting guide")
    print("  - example_plants: Example configurations")
    print("\nPress Ctrl+C to stop")
    print("=" * 70 + "\n")

    try:
        # Run the FastMCP server with SSE transport
        mcp.run(transport="sse", host=host, port=port)
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
    except Exception as e:
        print(f"\n\nError starting server: {e}")
        raise


# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    """
    Main entry point when running as a module.

    Usage:
        python -m pyadm1.configurator.mcp.server
    """
    start_server()
