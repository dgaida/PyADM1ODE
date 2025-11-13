"""
PyADM1 - Advanced Biogas Plant Simulation Framework

A comprehensive Python framework for modeling, simulating, and optimizing
agricultural biogas plants based on the Anaerobic Digestion Model No. 1 (ADM1).

Main modules:
    core: Core ADM1 implementation with ODE system, parameters, and solver
    components: Modular plant components (biological, mechanical, energy, feeding, sensors)
    substrates: Substrate management and characterization
    configurator: Plant model builder and MCP server for LLM integration
    simulation: Simulation engine with parallel execution capabilities
    calibration: Parameter calibration and online re-calibration
    io: Data import/export (JSON, CSV, database, measurements)
    utils: Utility functions for math, units, logging, validation

Example:
    >>> from pyadm1 import BiogasPlant
    >>> from pyadm1.components.biological import Digester
    >>> from pyadm1.substrates import Feedstock
    >>>
    >>> feedstock = Feedstock(feeding_freq=48)
    >>> plant = BiogasPlant("My Plant")
    >>> plant.add_component(Digester("dig1", feedstock, V_liq=2000))
    >>> plant.initialize()
    >>> results = plant.simulate(duration=30, dt=1/24)
"""

from pyadm1.__version__ import __version__

# Core imports
from pyadm1.configurator.plant_builder import BiogasPlant
from pyadm1.substrates.feedstock import Feedstock
from pyadm1.simulation.simulator import Simulator

# Component base classes
from pyadm1.components.base import Component, ComponentType

__all__ = [
    "__version__",
    "BiogasPlant",
    "Feedstock",
    "Simulator",
    "Component",
    "ComponentType",
]

__author__ = "Daniel Gaida"
__email__ = "daniel.gaida@th-koeln.de"
__license__ = "MIT"
