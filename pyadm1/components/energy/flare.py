# ============================================================================
# pyadm1/components/energy/flare.py
# =============================================================================
"""Flare component.

The Flare is a gas sink that combusts (destroys) vented/excess biogas safely.
It keeps track of vented volume and combustion emissions (simplified).
"""

from __future__ import annotations

from typing import Any

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
    ch4_fraction_default : float
        Methane fraction (0..1) assumed only when the incoming gas carries no
        ``CH4_fraction``. Default 0.6.
    """

    def __init__(
        self,
        component_id: str,
        destruction_efficiency: float = 0.98,
        name: str | None = None,
        ch4_fraction_default: float = 0.6,
    ):
        super().__init__(component_id, ComponentType.FLARE, name)
        self.destruction_efficiency: float = float(destruction_efficiency)
        self.ch4_fraction_default: float = float(ch4_fraction_default)
        self._cum_vented_m3: float = 0.0
        self.initialize()

    def initialize(self, initial_state: dict[str, Any] | None = None) -> None:
        """Initialize flare internal state.

        Args:
            initial_state: optional dict with 'cumulative_vented_m3' to restore state.
        """
        if initial_state and "cumulative_vented_m3" in initial_state:
            try:
                self._cum_vented_m3 = float(initial_state["cumulative_vented_m3"])
            except (TypeError, ValueError):
                self._cum_vented_m3 = 0.0
        else:
            self._cum_vented_m3 = 0.0

        self.outputs_data = {
            "vented_volume_m3": 0.0,
            "cumulative_vented_m3": self._cum_vented_m3,
            "CH4_destroyed_m3": 0.0,
            "CH4_slip_m3": 0.0,
        }
        self.state = {"cumulative_vented_m3": self._cum_vented_m3}
        self._initialized = True

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore a snapshot taken with :meth:`get_state`."""
        self.initialize(state)

    def step(self, t: float, dt: float, inputs: dict[str, Any]) -> dict[str, Any]:
        """Process one timestep and combust incoming gas.

        Args:
            t: current simulation time [days]
            dt: timestep length [days]
            inputs: dictionary that may contain:
                - 'Q_gas_in_m3_per_day': inflow (m³/day)
                - 'CH4_fraction': methane fraction in the gas (0..1). Defaults to
                  ``ch4_fraction_default`` when the source reports none.

        Returns:
            outputs_data dict with keys:
                - 'vented_volume_m3' (this timestep)
                - 'cumulative_vented_m3'
                - 'CH4_destroyed_m3' (m³ of CH4 destroyed this step)
                - 'CH4_slip_m3' (m³ of CH4 passing the flame unburnt this step)
        """
        Q_in = float(inputs.get("Q_gas_in_m3_per_day", 0.0))
        ch4_frac = float(inputs.get("CH4_fraction", self.ch4_fraction_default))

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
            "CH4_slip_m3": float(ch4_volume - ch4_destroyed),
        }
        self.state = {"cumulative_vented_m3": float(self._cum_vented_m3)}

        return self.outputs_data

    def to_dict(self) -> dict[str, Any]:
        """Serialize flare configuration and state."""
        return {
            "component_id": self.component_id,
            "component_type": self.component_type.value,
            "name": self.name,
            "destruction_efficiency": self.destruction_efficiency,
            "ch4_fraction_default": self.ch4_fraction_default,
            "cumulative_vented_m3": self._cum_vented_m3,
            "inputs": self.inputs,
            "outputs": self.outputs,
        }

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> Flare:
        """Instantiate Flare from dict created by `to_dict`."""
        flare = cls(
            component_id=config["component_id"],
            destruction_efficiency=config.get("destruction_efficiency", 0.98),
            name=config.get("name"),
            ch4_fraction_default=config.get("ch4_fraction_default", 0.6),
        )
        flare.inputs = config.get("inputs", [])
        flare.outputs = config.get("outputs", [])
        flare.initialize({"cumulative_vented_m3": config.get("cumulative_vented_m3", 0.0)})
        return flare
