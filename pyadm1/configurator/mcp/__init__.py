"""
MCP Server for LLM-Driven Plant Configuration

Model Context Protocol (MCP) server implementation using fastmcp for automated
biogas plant model creation through LLM interaction.

Modules:
    server: FastMCP server main implementation handling initialization, tool registration,
           connection management, and serving the MCP protocol to LLM clients like
           Claude Desktop or custom applications.

    tools: MCP tools exposed to LLM including create_plant, add_component,
          connect_components, simulate_plant, calibrate_model, get_plant_status,
          and export_configuration with detailed parameter schemas.

    prompts: System prompts and templates for guiding LLM in plant design including
            component selection criteria, connection rules, parameter ranges, and
            best practices for different plant types and substrates.

    schemas: Pydantic data schemas for request/response validation, ensuring type
            safety and providing clear documentation of available parameters and
            expected formats for LLM function calling.

Example:
    >>> from pyadm1.configurator.mcp import MCPServer
    >>>
    >>> # Start MCP server
    >>> server = MCPServer(host="localhost", port=3000)
    >>> server.start()
    >>>
    >>> # Server exposes tools that LLM can call:
    >>> # - create_plant(name, description)
    >>> # - add_component(plant_id, component_type, params)
    >>> # - connect_components(plant_id, from_id, to_id, connection_type)
    >>> # - simulate_plant(plant_id, duration, dt)
    >>> # - calibrate_model(plant_id, measurements, parameters)
    >>> # - get_plant_status(plant_id)
    >>> # - export_configuration(plant_id, filepath)

Integration with Claude Desktop:
    Add to claude_desktop_config.json:
    {
      "mcpServers": {
        "pyadm1": {
          "command": "python",
          "args": ["-m", "pyadm1.configurator.mcp"],
          "env": {}
        }
      }
    }

Usage in prompts:
    "Create a two-stage biogas plant with 2000 m³ main digester,
    500 m³ hydrolysis tank, and 500 kW CHP unit for corn silage
    and cattle manure co-digestion."
"""

from pyadm1.configurator.mcp.server import start_server

# from pyadm1.configurator.mcp.tools import (
#     create_plant,
#     add_component,
#     connect_components,
#     simulate_plant,
#     calibrate_model,
#     get_plant_status,
#     export_configuration,
# )
# from pyadm1.configurator.mcp.schemas import (
#     PlantCreateRequest,
#     ComponentAddRequest,
#     ConnectionRequest,
#     SimulationRequest,
#     CalibrationRequest,
# )

__all__ = [
    "start_server",
    # "create_plant",
    # "add_component",
    # "connect_components",
    # "simulate_plant",
    # "calibrate_model",
    # "get_plant_status",
    # "export_configuration",
    # "PlantCreateRequest",
    # "ComponentAddRequest",
    # "ConnectionRequest",
    # "SimulationRequest",
    # "CalibrationRequest",
]
