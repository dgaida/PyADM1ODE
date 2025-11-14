"""
Plant Model Configurator and MCP Server

This module provides tools for building, validating, and automatically configuring
biogas plant models, including an MCP server for LLM-driven plant design.

Modules:
    plant_builder: BiogasPlant class for assembling components into complete plant
                  models, managing component lifecycle, connections, and providing
                  JSON serialization for model persistence and sharing.

    connection_manager: Connection handling between components including type checking
                       (liquid, gas, heat, power), flow validation, and dependency
                       resolution for correct simulation order.

    validation: Model validation checking for physical consistency (mass/energy balance),
               completeness (all required connections), and parameter validity with
               detailed error messages for debugging.

Subpackages:
    templates: Pre-defined plant configurations for common biogas plant layouts
              (single-stage, two-stage, plug-flow, CSTR) serving as starting
              points for customization or examples for learning.

    mcp: MCP (Model Context Protocol) server implementation using fastmcp for
        LLM integration, providing tools for automated plant design from natural
        language descriptions and integration with intelligent virtual advisors.

Example:
    >>> from pyadm1.configurator import BiogasPlant, ConnectionManager
    >>> from pyadm1.components.biological import Digester
    >>>
    >>> # Build plant programmatically
    >>> plant = BiogasPlant("My Plant")
    >>> plant.add_component(Digester("dig1", feedstock, V_liq=2000))
    >>> plant.validate()
    >>>
    >>> # Save configuration
    >>> plant.to_json("my_plant.json")
    >>>
    >>> # Load from template
    >>> from pyadm1.configurator.templates import TwoStageTemplate
    >>> plant = TwoStageTemplate.create(feedstock=feedstock)
"""

from pyadm1.configurator.plant_builder import BiogasPlant
from pyadm1.configurator.connection_manager import (
    ConnectionManager,
    Connection,
    ConnectionType,
)

# from pyadm1.configurator.validation import PlantValidator

# Import templates
from pyadm1.configurator import templates

# Import MCP server (optional dependency)
try:
    from pyadm1.configurator import mcp

    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    mcp = None

__all__ = [
    "BiogasPlant",
    "ConnectionManager",
    "Connection",
    "ConnectionType",
    # "PlantValidator",
    "templates",
    "mcp",
    "MCP_AVAILABLE",
]
