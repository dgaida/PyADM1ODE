# -*- coding: utf-8 -*-
"""Unit tests for connection management utilities."""

import pytest

from pyadm1.configurator.connection_manager import (
    Connection,
    ConnectionManager,
    ConnectionType,
)


class TestConnection:
    """Tests for the Connection value object."""

    def test_connection_to_dict_from_dict_and_repr(self) -> None:
        conn = Connection("dig1", "chp1", "gas")

        data = conn.to_dict()
        recreated = Connection.from_dict(data)

        assert data == {"from": "dig1", "to": "chp1", "type": "gas"}
        assert recreated.from_component == "dig1"
        assert recreated.to_component == "chp1"
        assert recreated.connection_type == "gas"
        assert "Connection(from='dig1'" in repr(conn)

    def test_connection_from_dict_defaults_type(self) -> None:
        conn = Connection.from_dict({"from": "a", "to": "b"})
        assert conn.connection_type == "default"

    def test_connection_type_enum_values(self) -> None:
        assert ConnectionType.LIQUID.value == "liquid"
        assert ConnectionType.GAS.value == "gas"
        assert ConnectionType.HEAT.value == "heat"
        assert ConnectionType.POWER.value == "power"
        assert ConnectionType.CONTROL.value == "control"
        assert ConnectionType.DEFAULT.value == "default"


class TestConnectionManager:
    """Tests for ConnectionManager behavior and branches."""

    def _sample_manager(self) -> ConnectionManager:
        manager = ConnectionManager()
        manager.add_connection(Connection("a", "b", "liquid"))
        manager.add_connection(Connection("b", "c", "gas"))
        manager.add_connection(Connection("c", "d", "heat"))
        return manager

    def test_add_connection_duplicate_raises_and_len_repr(self) -> None:
        manager = ConnectionManager()
        conn = Connection("dig1", "chp1", "gas")

        manager.add_connection(conn)
        assert len(manager) == 1
        assert "connections=1" in repr(manager)

        with pytest.raises(ValueError, match="already exists"):
            manager.add_connection(Connection("dig1", "chp1", "gas"))

    def test_remove_connection_with_type_filter_and_without_type(self) -> None:
        manager = ConnectionManager()
        manager.add_connection(Connection("x", "y", "gas"))
        manager.add_connection(Connection("x", "y", "heat"))
        manager.add_connection(Connection("y", "z", "power"))

        removed_one = manager.remove_connection("x", "y", connection_type="gas")
        assert removed_one is True
        assert len(manager.connections) == 2
        assert all(
            not (c.from_component == "x" and c.to_component == "y" and c.connection_type == "gas") for c in manager.connections
        )

        removed_none = manager.remove_connection("x", "y", connection_type="liquid")
        assert removed_none is False

        removed_all_between = manager.remove_connection("x", "y")
        assert removed_all_between is True
        assert all(not (c.from_component == "x" and c.to_component == "y") for c in manager.connections)

    def test_getters_dependencies_dependents_and_copy_clear(self) -> None:
        manager = self._sample_manager()

        outgoing = manager.get_connections_from("b")
        incoming = manager.get_connections_to("c")
        deps = manager.get_dependencies("c")
        dependents = manager.get_dependents("b")
        all_connections = manager.get_all_connections()

        assert len(outgoing) == 1 and outgoing[0].to_component == "c"
        assert len(incoming) == 1 and incoming[0].from_component == "b"
        assert deps == ["b"]
        assert dependents == ["c"]
        assert len(all_connections) == 3

        all_connections.clear()
        assert len(manager.connections) == 3  # returned list is a copy

        manager.clear()
        assert len(manager.connections) == 0

    def test_get_execution_order_and_cycle_detection(self) -> None:
        manager = ConnectionManager()
        manager.add_connection(Connection("a", "b", "liquid"))
        manager.add_connection(Connection("b", "c", "liquid"))
        manager.add_connection(Connection("outside1", "outside2", "gas"))  # ignored by requested node set

        order = manager.get_execution_order(["a", "b", "c", "d"])

        assert order.index("a") < order.index("b") < order.index("c")
        assert set(order) == {"a", "b", "c", "d"}
        assert manager.has_circular_dependency(["a", "b", "c", "d"]) is False

        manager.add_connection(Connection("c", "a", "liquid"))
        with pytest.raises(ValueError, match="Circular dependency"):
            manager.get_execution_order(["a", "b", "c"])
        assert manager.has_circular_dependency(["a", "b", "c"]) is True

    def test_get_connected_components_traverses_both_directions_and_avoids_revisit(
        self,
    ) -> None:
        manager = ConnectionManager()
        manager.add_connection(Connection("a", "b", "liquid"))
        manager.add_connection(Connection("c", "b", "gas"))  # reverse-direction discovery
        manager.add_connection(Connection("c", "d", "heat"))  # forward-direction discovery
        manager.add_connection(Connection("d", "a", "control"))  # cycle exercises visited-skip branch
        manager.add_connection(Connection("x", "y", "default"))  # disconnected subgraph

        connected = manager.get_connected_components("a")

        assert connected == {"b", "c", "d"}

    def test_validate_connections_reports_invalid_refs_and_cycle(self) -> None:
        manager = ConnectionManager()
        manager.add_connection(Connection("a", "b", "liquid"))
        manager.add_connection(Connection("b", "a", "liquid"))  # cycle in valid subset
        manager.add_connection(Connection("missing_src", "a", "gas"))
        manager.add_connection(Connection("a", "missing_tgt", "heat"))

        errors = manager.validate_connections(["a", "b"])

        assert any("non-existent source component: missing_src" in e for e in errors)
        assert any("non-existent target component: missing_tgt" in e for e in errors)
        assert any("Circular dependency detected in connections" == e for e in errors)

    def test_to_dict_and_from_dict_roundtrip_and_empty_config(self) -> None:
        manager = ConnectionManager()
        manager.add_connection(Connection("dig1", "store1", "gas"))
        manager.add_connection(Connection("chp1", "heat1", "heat"))

        data = manager.to_dict()
        recreated = ConnectionManager.from_dict(data)
        empty = ConnectionManager.from_dict({})

        assert "connections" in data
        assert len(data["connections"]) == 2
        assert len(recreated.connections) == 2
        assert recreated.connections[0].to_dict() == data["connections"][0]
        assert len(empty.connections) == 0
