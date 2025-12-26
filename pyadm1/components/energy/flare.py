# ============================================================================
# pyadm1/components/energy/flare.py
# =============================================================================
"""Flare component.

The Flare is a gas sink that combusts (destroys) vented/excess biogas safely.
It keeps track of vented volume and combustion emissions (simplified).
"""
from typing import Dict, Any, Optional

from ..base import Component, ComponentType


class Flare(Component):
    """Flare component for combusting vented biogas.

    The flare accepts an input `Q_gas_in_m3_per_day` and will combust it.
    It reports `vented_volume_m3` for the current timestep and `cumulative_vented_m3`.

    Parameters
    ----------
    component_id : str
        Unique id for the flare component.
    destruction_efficiency : float
        Fraction of methane destroyed (0..1). Default 0.98.
    name : Optional[str]
        Human readable name.
    """

    def __init__(self, component_id: str, destruction_efficiency: float = 0.98, name: Optional[str] = None):
        super().__init__(component_id, ComponentType.STORAGE, name)
        self.destruction_efficiency: float = float(destruction_efficiency)
        self._cum_vented_m3: float = 0.0
        self.initialize()

    def initialize(self, initial_state: Optional[Dict[str, Any]] = None) -> None:
        """Initialize flare internal state.

        Args:
            initial_state: optional dict with 'cumulative_vented_m3' to restore state.
        """
        if initial_state and "cumulative_vented_m3" in initial_state:
            try:
                self._cum_vented_m3 = float(initial_state["cumulative_vented_m3"])
            except Exception:
                self._cum_vented_m3 = 0.0
        else:
            self._cum_vented_m3 = 0.0

        self.outputs_data = {
            "vented_volume_m3": 0.0,
            "cumulative_vented_m3": self._cum_vented_m3,
            "CH4_destroyed_m3": 0.0,
        }
        self._initialized = True

    def step(self, t: float, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process one timestep and combust incoming gas.

        Args:
            t: current simulation time [days]
            dt: timestep length [days]
            inputs: dictionary that may contain:
                - 'Q_gas_in_m3_per_day': inflow (m³/day)
                - 'CH4_fraction': methane fraction in the gas (0..1). Default 0.6

        Returns:
            outputs_data dict with keys:
                - 'vented_volume_m3' (this timestep)
                - 'cumulative_vented_m3'
                - 'CH4_destroyed_m3' (m³ of CH4 destroyed this step)
        """
        Q_in = float(inputs.get("Q_gas_in_m3_per_day", 0.0))
        ch4_frac = float(inputs.get("CH4_fraction", 0.6))

        # convert to volume in this timestep
        vol_in = Q_in * dt

        # combustion: a flare destroys a fraction of incoming CH4
        ch4_volume = vol_in * ch4_frac
        ch4_destroyed = ch4_volume * self.destruction_efficiency

        # record vented (treated) volume
        self._cum_vented_m3 += vol_in

        self.outputs_data = {
            "vented_volume_m3": float(vol_in),
            "cumulative_vented_m3": float(self._cum_vented_m3),
            "CH4_destroyed_m3": float(ch4_destroyed),
        }

        return self.outputs_data

    def to_dict(self) -> Dict[str, Any]:
        """Serialize flare configuration and state."""
        return {
            "component_id": self.component_id,
            "component_type": self.component_type.value,
            "name": self.name,
            "destruction_efficiency": self.destruction_efficiency,
            "cumulative_vented_m3": self._cum_vented_m3,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "Flare":
        """Instantiate Flare from dict created by `to_dict`."""
        flare = cls(
            component_id=config["component_id"],
            destruction_efficiency=config.get("destruction_efficiency", 0.98),
            name=config.get("name"),
        )
        flare.initialize({"cumulative_vented_m3": config.get("cumulative_vented_m3", 0.0)})
        return flare
