"""
Mechanical Plant Components

Mechanical equipment for material handling and process control.

Modules:
    pump: Pump models including centrifugal and positive displacement types,
         with power consumption calculation, flow rate control, and characteristic
         curves for different operating points.

    mixer: Agitator and stirrer models for digester mixing, including different
          types (propeller, paddle, jet), with power consumption based on mixing
          intensity, viscosity, and tank geometry.

    valve: Control and safety valves for flow regulation, including proportional
          control valves, on/off valves, and pressure relief valves with configurable
          characteristics and response times.

    heat_exchanger: Heat transfer equipment for substrate pre-heating and digestate
                   cooling, supporting different types (plate, tube), with effectiveness
                   calculation and fouling factor consideration.

Example:
    >>> from pyadm1.components.mechanical import Pump, Mixer, Valve, HeatExchanger
    >>>
    >>> # Substrate feeding pump
    >>> pump = Pump("pump1", pump_type="positive_displacement",
    ...            Q_nom=10, pressure_head=5)
    >>>
    >>> # Digester mixer
    >>> mixer = Mixer("mix1", mixer_type="propeller",
    ...              power=15, mixing_intensity="medium")
    >>>
    >>> # Heat exchanger for substrate pre-heating
    >>> hex = HeatExchanger("hex1", effectiveness=0.7, area=50)
"""

from .pump import Pump
from .mixer import Mixer

# from pyadm1.components.mechanical.valve import Valve
# from pyadm1.components.mechanical.heat_exchanger import HeatExchanger
#
__all__ = [
    "Pump",
    "Mixer",
    #     "Valve",
    #     "HeatExchanger",
]
