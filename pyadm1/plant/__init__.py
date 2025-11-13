# ============================================================================
# pyadm1/plant/__init__.py
# ============================================================================
"""
Biogas plant component system.

This module provides a flexible architecture for building biogas plants
from modular components (digesters, CHP units, heating systems, etc.).
"""

from pyadm1.plant.component_base import Component, ComponentType
from pyadm1.plant.digester import Digester
from pyadm1.plant.chp import CHP
from pyadm1.plant.heating import HeatingSystem
from pyadm1.plant.plant_model import BiogasPlant
from pyadm1.plant.connection import Connection

__all__ = [
    "Component",
    "ComponentType",
    "Digester",
    "CHP",
    "HeatingSystem",
    "BiogasPlant",
    "Connection",
]
