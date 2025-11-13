"""
Substrate Feeding and Storage Components

Components for substrate handling, storage, and dosing into digesters.

Modules:
    substrate_storage: Storage facilities for different substrate types (silos for
                      solid substrates, tanks for liquid substrates), with inventory
                      management, quality degradation over time, and capacity monitoring.

    feeder: Automated dosing systems including screw feeders for solid substrates,
           progressive cavity pumps for liquid/slurry substrates, and piston feeders
           for fibrous materials, with flow rate control and dosing accuracy.

    mixer_wagon: Mobile substrate preparation systems for mixing multiple substrates
                before feeding, with mixing efficiency, recipe management, and
                substrate homogenization.

Example:
    >>> from pyadm1.components.feeding import SubstrateStorage, Feeder, MixerWagon
    >>>
    >>> # Corn silage storage
    >>> storage = SubstrateStorage("silo1", substrate_type="corn_silage",
    ...                           capacity=1000, current_level=800)
    >>>
    >>> # Screw feeder for solid substrates
    >>> feeder = Feeder("feed1", feeder_type="screw",
    ...                Q_max=20, substrate_type="solid")
    >>>
    >>> # Mixer wagon for substrate preparation
    >>> wagon = MixerWagon("wagon1", capacity=30, mixing_time=15)
"""

from pyadm1.components.feeding.substrate_storage import SubstrateStorage
from pyadm1.components.feeding.feeder import Feeder
from pyadm1.components.feeding.mixer_wagon import MixerWagon

__all__ = [
    "SubstrateStorage",
    "Feeder",
    "MixerWagon",
]
