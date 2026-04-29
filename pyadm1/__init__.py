"""
PyADM1 - Biogas Plant Simulation Framework (ADM1da).

A pure-Python framework for modeling, simulating, and optimizing agricultural
biogas plants based on **ADM1da** (Schlattmann 2011), an agricultural-biogas
extension of ADM1 (Batstone et al. 2002).  The model is a 41-state ODE system
with sub-fraction disintegration, temperature-dependent kinetics, and modified
inhibition; no .NET / DLL dependency.

This is an independent open-source implementation; not affiliated with the
ifak e.V. Magdeburg or the commercial SIMBA# biogas product.  The SIMBA#
biogas Tutorial 4.2 is referenced where appropriate as a published source for
parameter values and process structure.

Main modules:
    core: ADM1 ODE system, parameters, and solver.
    substrates: Substrate characterization and influent generation (Feedstock).
    components: Modular plant components (biological, mechanical, energy,
                feeding, sensors).
    configurator: Plant model builder and MCP server for LLM integration.
    simulation: Single-plant Simulator and ParallelSimulator (parameter sweeps,
                Monte Carlo).

Example:
    >>> from pyadm1 import BiogasPlant, Feedstock
    >>> from pyadm1.components.biological import Digester
    >>>
    >>> fs = Feedstock(["maize_silage_milk_ripeness", "swine_manure"], feeding_freq=24)
    >>> plant = BiogasPlant("My Plant")
    >>> dig = Digester("dig1", fs, V_liq=1200, V_gas=216, T_ad=315.15)
    >>> dig.initialize({"Q_substrates": [11.4, 6.1, 0, 0, 0, 0, 0, 0, 0, 0]})
    >>> plant.add_component(dig)
    >>> plant.initialize()
    >>> results = plant.simulate(duration=30, dt=1.0)
"""

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:  # pragma: no cover
    # For Python < 3.8
    from importlib_metadata import version, PackageNotFoundError  # type: ignore

try:
    __version__ = version("pyadm1")
except PackageNotFoundError:  # pragma: no cover
    # package is not installed
    __version__ = "unknown"

# Core imports
from .configurator import BiogasPlant
from .substrates import Feedstock
from .simulation import Simulator

# Component base classes
from .components import Component, ComponentType

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
