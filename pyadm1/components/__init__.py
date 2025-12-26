"""
Modular Plant Components

This module provides a comprehensive library of biogas plant components that can be
combined to build complex plant configurations. All components follow a common
interface and can be dynamically loaded and connected.

Modules:
    base: Base classes (Component, ComponentType) defining the interface that all
          components must implement, including step(), initialize(), serialization,
          and connection management.

    registry: Component registry system for dynamic component discovery, loading,
             and instantiation, enabling plugin-like extensibility and automated
             component selection by LLM agents.

Subpackages:
    biological: Components for biological processes including digesters (single/multi-stage),
               hydrolysis tanks for pre-treatment, and solid-liquid separators for
               digestate processing.

    mechanical: Mechanical components including pumps (centrifugal, positive displacement),
               mixers/agitators (various types), valves (control, safety), and heat
               exchangers for thermal management.

    energy: Energy conversion and storage components including CHP units (gas engines,
           micro-turbines), boilers (auxiliary heating), gas storage (low/high pressure),
           and flares for safety gas combustion.

    feeding: Substrate handling components including storage silos, automated dosing
            systems (screw feeders, pumps), and mixer wagons for substrate preparation.

    sensors: Measurement components including physical sensors (pH, temperature, pressure),
            chemical sensors (VFA, ammonia, COD), and gas analyzers (CH4, CO2, H2S, O2).

Example:
    >>> from pyadm1.components.biological import Digester
    >>> from pyadm1.components.energy import CHP
    >>> from pyadm1.components import ComponentRegistry
    >>>
    >>> # Direct instantiation
    >>> digester = Digester("dig1", feedstock, V_liq=2000)
    >>>
    >>> # Via registry
    >>> registry = ComponentRegistry()
    >>> component = registry.create("Digester", "dig1", feedstock=feedstock)
"""

from .base import Component, ComponentType
from .registry import ComponentRegistry

# Import all component types for registry
from . import biological, mechanical, energy, feeding, sensors

__all__ = [
    "Component",
    "ComponentType",
    "ComponentRegistry",
    "biological",
    "mechanical",
    "energy",
    "feeding",
    "sensors",
]
