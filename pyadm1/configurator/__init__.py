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

Subpackages:
    templates: Placeholder for ready-made layouts; the layouts that exist are the
              ``create_*_plant`` methods of ``PlantConfigurator``.

Example:
    >>> from pyadm1 import Feedstock
    >>> from pyadm1.configurator import BiogasPlant, PlantConfigurator
    >>>
    >>> feedstock = Feedstock(["maize_silage_milk_ripeness", "cattle_manure"])
    >>> plant = BiogasPlant("My Plant")
    >>> configurator = PlantConfigurator(plant, feedstock)
    >>> configurator.add_digester("dig1", V_liq=2000, Q_substrates=[15, 10])
    >>>
    >>> # Save configuration
    >>> plant.to_json("my_plant.json")
"""

# Import templates
from . import templates
from .connection_manager import (
    Connection,
    ConnectionManager,
    ConnectionType,
)
from .plant_builder import BiogasPlant
from .plant_configurator import PlantConfigurator

__all__ = [
    "BiogasPlant",
    "Connection",
    "ConnectionManager",
    "ConnectionType",
    "PlantConfigurator",
    # "PlantValidator",
    "templates",
]
