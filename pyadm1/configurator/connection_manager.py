# ============================================================================
# pyadm1/plant/connection.py
# ============================================================================
"""
Connection class for linking components.
"""

from typing import Any, Dict


class Connection:
    """
    Represents a connection between two components.
    """

    def __init__(self, from_component: str, to_component: str, connection_type: str = "default"):
        """
        Initialize connection.

        Parameters
        ----------
        from_component : str
            Source component ID
        to_component : str
            Target component ID
        connection_type : str
            Type of connection (e.g., 'liquid', 'gas', 'heat', 'power')
        """
        self.from_component = from_component
        self.to_component = to_component
        self.connection_type = connection_type

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "from": self.from_component,
            "to": self.to_component,
            "type": self.connection_type,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "Connection":
        """Create from dictionary."""
        return cls(
            from_component=config["from"],
            to_component=config["to"],
            connection_type=config.get("type", "default"),
        )
