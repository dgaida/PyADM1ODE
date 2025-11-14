# ============================================================================
# pyadm1/components/energy/gas_storage.py
# ============================================================================
"""Gas storage component (stub implementation)."""

from typing import Dict, Any, Optional
from pyadm1.components.base import Component, ComponentType


class GasStorage(Component):
    """Gas storage component (stub for future implementation)."""

    def __init__(self, component_id: str, volume: float = 1000.0, name: Optional[str] = None):
        super().__init__(component_id, ComponentType.STORAGE, name)

    def initialize(self, initial_state: Optional[Dict[str, Any]] = None) -> None:
        self.state = {}

    def step(self, t: float, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return {}

    def to_dict(self) -> Dict[str, Any]:
        return {"component_id": self.component_id, "component_type": self.component_type.value}

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "GasStorage":
        return cls(config["component_id"])
