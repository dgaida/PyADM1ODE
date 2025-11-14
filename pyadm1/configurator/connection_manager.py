# ============================================================================
# pyadm1/configurator/connection_manager.py
# ============================================================================
"""
Connection management for biogas plant components.

This module provides classes for defining and managing connections between
components in a biogas plant configuration.
"""

from typing import Any, Dict, List, Set, Optional
from enum import Enum


class ConnectionType(Enum):
    """
    Enumeration of connection types between components.

    Attributes:
        LIQUID: Liquid flow connection (substrate, digestate).
        GAS: Biogas flow connection.
        HEAT: Heat transfer connection.
        POWER: Electrical power connection.
        CONTROL: Control signal connection.
        DEFAULT: Generic connection type.
    """

    LIQUID = "liquid"
    GAS = "gas"
    HEAT = "heat"
    POWER = "power"
    CONTROL = "control"
    DEFAULT = "default"


class Connection:
    """
    Represents a connection between two components.

    A connection defines a directed link from one component to another,
    specifying what type of flow or signal is being transferred.

    Attributes:
        from_component (str): Source component ID.
        to_component (str): Target component ID.
        connection_type (str): Type of connection.

    Example:
        >>> conn = Connection("digester_1", "chp_1", "gas")
        >>> config = conn.to_dict()
    """

    def __init__(self, from_component: str, to_component: str, connection_type: str = "default"):
        """
        Initialize connection.

        Args:
            from_component (str): Source component ID.
            to_component (str): Target component ID.
            connection_type (str): Type of connection (e.g., 'liquid', 'gas',
                'heat', 'power'). Defaults to "default".
        """
        self.from_component = from_component
        self.to_component = to_component
        self.connection_type = connection_type

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dictionary.

        Returns:
            Dict[str, Any]: Dictionary representation of the connection.
        """
        return {
            "from": self.from_component,
            "to": self.to_component,
            "type": self.connection_type,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "Connection":
        """
        Create from dictionary.

        Args:
            config (Dict[str, Any]): Dictionary with 'from', 'to', and
                optional 'type' keys.

        Returns:
            Connection: New connection instance.
        """
        return cls(
            from_component=config["from"],
            to_component=config["to"],
            connection_type=config.get("type", "default"),
        )

    def __repr__(self) -> str:
        """String representation of connection."""
        return f"Connection(from='{self.from_component}', " f"to='{self.to_component}', type='{self.connection_type}')"


class ConnectionManager:
    """
    Manages connections between components in a biogas plant.

    The ConnectionManager handles connection validation, dependency resolution,
    and provides utilities for analyzing component relationships.

    Attributes:
        connections (List[Connection]): List of all connections.

    Example:
        >>> manager = ConnectionManager()
        >>> manager.add_connection(Connection("dig1", "chp1", "gas"))
        >>> deps = manager.get_dependencies("chp1")
        >>> print(deps)  # ['dig1']
    """

    def __init__(self):
        """Initialize an empty connection manager."""
        self.connections: List[Connection] = []

    def add_connection(self, connection: Connection) -> None:
        """
        Add a connection to the manager.

        Args:
            connection (Connection): Connection to add.

        Raises:
            ValueError: If connection already exists.
        """
        # Check for duplicate
        for conn in self.connections:
            if (
                conn.from_component == connection.from_component
                and conn.to_component == connection.to_component
                and conn.connection_type == connection.connection_type
            ):
                raise ValueError(
                    f"Connection already exists: {connection.from_component} -> "
                    f"{connection.to_component} ({connection.connection_type})"
                )

        self.connections.append(connection)

    def remove_connection(self, from_component: str, to_component: str, connection_type: Optional[str] = None) -> bool:
        """
        Remove a connection.

        Args:
            from_component (str): Source component ID.
            to_component (str): Target component ID.
            connection_type (Optional[str]): Connection type. If None, removes
                all connections between the components.

        Returns:
            bool: True if at least one connection was removed, False otherwise.
        """
        removed = False
        self.connections = [
            conn
            for conn in self.connections
            if not (
                conn.from_component == from_component
                and conn.to_component == to_component
                and (connection_type is None or conn.connection_type == connection_type)
            )
            or not (removed := True)
        ]
        return removed

    def get_connections_from(self, component_id: str) -> List[Connection]:
        """
        Get all connections originating from a component.

        Args:
            component_id (str): Component ID.

        Returns:
            List[Connection]: List of outgoing connections.
        """
        return [conn for conn in self.connections if conn.from_component == component_id]

    def get_connections_to(self, component_id: str) -> List[Connection]:
        """
        Get all connections terminating at a component.

        Args:
            component_id (str): Component ID.

        Returns:
            List[Connection]: List of incoming connections.
        """
        return [conn for conn in self.connections if conn.to_component == component_id]

    def get_dependencies(self, component_id: str) -> List[str]:
        """
        Get all components that the given component depends on.

        Args:
            component_id (str): Component ID.

        Returns:
            List[str]: List of component IDs that this component depends on.
        """
        return [conn.from_component for conn in self.get_connections_to(component_id)]

    def get_dependents(self, component_id: str) -> List[str]:
        """
        Get all components that depend on the given component.

        Args:
            component_id (str): Component ID.

        Returns:
            List[str]: List of component IDs that depend on this component.
        """
        return [conn.to_component for conn in self.get_connections_from(component_id)]

    def get_execution_order(self, component_ids: List[str]) -> List[str]:
        """
        Determine execution order based on dependencies (topological sort).

        Args:
            component_ids (List[str]): List of all component IDs.

        Returns:
            List[str]: Component IDs in execution order.

        Raises:
            ValueError: If circular dependencies are detected.
        """
        # Build adjacency list
        in_degree = {comp_id: 0 for comp_id in component_ids}
        adjacency = {comp_id: [] for comp_id in component_ids}

        for conn in self.connections:
            if conn.from_component in component_ids and conn.to_component in component_ids:
                adjacency[conn.from_component].append(conn.to_component)
                in_degree[conn.to_component] += 1

        # Kahn's algorithm for topological sort
        queue = [comp_id for comp_id in component_ids if in_degree[comp_id] == 0]
        result = []

        while queue:
            current = queue.pop(0)
            result.append(current)

            for neighbor in adjacency[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Check for cycles
        if len(result) != len(component_ids):
            raise ValueError("Circular dependency detected in component connections")

        return result

    def has_circular_dependency(self, component_ids: List[str]) -> bool:
        """
        Check if there are circular dependencies.

        Args:
            component_ids (List[str]): List of component IDs to check.

        Returns:
            bool: True if circular dependencies exist, False otherwise.
        """
        try:
            self.get_execution_order(component_ids)
            return False
        except ValueError:
            return True

    def get_connected_components(self, component_id: str) -> Set[str]:
        """
        Get all components connected to the given component (directly or indirectly).

        Args:
            component_id (str): Starting component ID.

        Returns:
            Set[str]: Set of connected component IDs.
        """
        visited = set()
        queue = [component_id]

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue

            visited.add(current)

            # Add all connected components
            for conn in self.connections:
                if conn.from_component == current and conn.to_component not in visited:
                    queue.append(conn.to_component)
                elif conn.to_component == current and conn.from_component not in visited:
                    queue.append(conn.from_component)

        visited.discard(component_id)  # Remove the starting component
        return visited

    def validate_connections(self, component_ids: List[str]) -> List[str]:
        """
        Validate all connections and return list of issues.

        Args:
            component_ids (List[str]): List of valid component IDs.

        Returns:
            List[str]: List of validation error messages. Empty if all valid.
        """
        errors = []

        # Check for invalid component references
        for conn in self.connections:
            if conn.from_component not in component_ids:
                errors.append(f"Connection references non-existent source component: " f"{conn.from_component}")
            if conn.to_component not in component_ids:
                errors.append(f"Connection references non-existent target component: " f"{conn.to_component}")

        # Check for circular dependencies
        if self.has_circular_dependency(component_ids):
            errors.append("Circular dependency detected in connections")

        return errors

    def get_all_connections(self) -> List[Connection]:
        """
        Get all connections.

        Returns:
            List[Connection]: List of all connections.
        """
        return self.connections.copy()

    def clear(self) -> None:
        """Remove all connections."""
        self.connections.clear()

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize all connections to dictionary.

        Returns:
            Dict[str, Any]: Dictionary with connection data.
        """
        return {"connections": [conn.to_dict() for conn in self.connections]}

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "ConnectionManager":
        """
        Create ConnectionManager from dictionary.

        Args:
            config (Dict[str, Any]): Dictionary with 'connections' key.

        Returns:
            ConnectionManager: New manager with loaded connections.
        """
        manager = cls()
        for conn_config in config.get("connections", []):
            manager.add_connection(Connection.from_dict(conn_config))
        return manager

    def __len__(self) -> int:
        """Return number of connections."""
        return len(self.connections)

    def __repr__(self) -> str:
        """String representation of manager."""
        return f"ConnectionManager(connections={len(self.connections)})"
