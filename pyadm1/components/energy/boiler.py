# ============================================================================
# pyadm1/components/energy/boiler.py
# ============================================================================
"""Auxiliary boiler component (stub implementation)."""

from typing import Dict, Any, Optional
from ..base import Component, ComponentType


class Boiler(Component):
    """
    Auxiliary boiler component (stub for future implementation).

    Models a boiler that provides heat to the system when CHP heat is
    insufficient.

    Attributes:
        component_id: Unique identifier.
        P_th_nom: Nominal thermal power [kW].
        efficiency: Thermal efficiency (0-1).
        name: Optional name.
    """

    def __init__(
        self,
        component_id: str,
        P_th_nom: float = 500.0,
        efficiency: float = 0.9,
        name: Optional[str] = None,
    ):
        """
        Initialize the Boiler.

        Args:
            component_id: Unique identifier.
            P_th_nom: Nominal power [kW].
            efficiency: Boiler efficiency.
            name: Optional name.
        """
        super().__init__(component_id, ComponentType.BOILER, name)
        self.P_th_nom = P_th_nom
        self.efficiency = efficiency

    def initialize(self, initial_state: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize component state.

        Args:
            initial_state: Optional initial state.
        """
        self.state = {}

    def step(self, t: float, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute one simulation step.

        Args:
            t: Current time [days].
            dt: Time step [days].
            inputs: Input dictionary.

        Returns:
            Output dictionary.
        """
        return {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dictionary.

        Returns:
            Configuration dictionary.
        """
        return {
            "component_id": self.component_id,
            "component_type": self.component_type.value,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "Boiler":
        """
        Create instance from dictionary.

        Args:
            config: Configuration dictionary.

        Returns:
            New Boiler instance.
        """
        return cls(config["component_id"])
