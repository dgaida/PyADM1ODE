# -*- coding: utf-8 -*-
"""Unit tests for the BiogasPlant class (plant_builder)."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pyadm1 import Feedstock
from pyadm1.components.biological import Digester
from pyadm1.configurator.connection_manager import Connection
from pyadm1.configurator.plant_builder import BiogasPlant


@pytest.fixture
def feedstock() -> Feedstock:
    return Feedstock(
        ["maize_silage_milk_ripeness", "swine_manure"],
        feeding_freq=24,
        total_simtime=5,
    )


class _StubComponent:
    """Minimal test double with the attributes BiogasPlant.step needs."""

    def __init__(self, component_id: str, component_type: str, **attrs):
        self.component_id = component_id
        self.name = component_id
        self.component_type = SimpleNamespace(value=component_type)
        self.inputs = []
        self.outputs = []
        self.outputs_data = {}
        self._initialized = True
        self.step_calls = []
        for key, value in attrs.items():
            setattr(self, key, value)

    def add_input(self, component_id: str) -> None:
        if component_id not in self.inputs:
            self.inputs.append(component_id)

    def add_output(self, component_id: str) -> None:
        if component_id not in self.outputs:
            self.outputs.append(component_id)

    def initialize(self) -> None:
        self._initialized = True

    def step(self, t, dt, inputs):
        self.step_calls.append({"t": t, "dt": dt, "inputs": dict(inputs)})
        if hasattr(self, "_step_impl"):
            return self._step_impl(t, dt, inputs)
        return self.outputs_data

    def to_dict(self):
        return {
            "component_id": self.component_id,
            "component_type": self.component_type.value,
            "name": self.name,
        }


class TestBiogasPlantInitialization:
    def test_default_name(self) -> None:
        plant = BiogasPlant()
        assert plant.plant_name == "Biogas Plant"

    def test_custom_name(self) -> None:
        plant = BiogasPlant("My Plant")
        assert plant.plant_name == "My Plant"

    def test_starts_with_no_components(self) -> None:
        plant = BiogasPlant()
        assert plant.components == {}
        assert plant.connections == []
        assert plant.simulation_time == 0.0


class TestComponentManagement:
    def test_add_component_stores_by_id(self, feedstock: Feedstock) -> None:
        plant = BiogasPlant()
        d = Digester("d1", feedstock)
        plant.add_component(d)

        assert plant.components["d1"] is d

    def test_add_component_duplicate_raises(self, feedstock: Feedstock) -> None:
        plant = BiogasPlant()
        plant.add_component(Digester("d1", feedstock))
        with pytest.raises(ValueError, match="already exists"):
            plant.add_component(Digester("d1", feedstock))

    def test_add_connection_validates_endpoints(self, feedstock: Feedstock) -> None:
        plant = BiogasPlant()
        plant.add_component(Digester("d1", feedstock))

        with pytest.raises(ValueError, match="Source component"):
            plant.add_connection(Connection("missing", "d1", "liquid"))
        with pytest.raises(ValueError, match="Target component"):
            plant.add_connection(Connection("d1", "missing", "liquid"))

    def test_add_connection_links_components(self, feedstock: Feedstock) -> None:
        plant = BiogasPlant()
        plant.add_component(Digester("d1", feedstock))
        plant.add_component(Digester("d2", feedstock))

        plant.add_connection(Connection("d1", "d2", "liquid"))

        assert "d2" in plant.components["d1"].outputs
        assert "d1" in plant.components["d2"].inputs


class TestInitializeAndStep:
    def test_initialize_calls_uninitialised_components(self) -> None:
        plant = BiogasPlant()
        a = _StubComponent("a", "digester")
        a._initialized = False
        plant.components["a"] = a

        plant.initialize()
        assert a._initialized is True

    def test_step_executes_topological_order(self) -> None:
        plant = BiogasPlant()

        # Build a small chain: source → mid → sink
        order = []

        def make(component_id, component_type):
            comp = _StubComponent(component_id, component_type)

            def impl(t, dt, inputs):
                order.append(component_id)
                return {component_id: True}

            comp._step_impl = impl
            return comp

        source = make("source", "digester")
        mid = make("mid", "digester")
        sink = make("sink", "digester")
        plant.components.update({"source": source, "mid": mid, "sink": sink})
        sink.inputs = ["mid"]
        mid.inputs = ["source"]

        plant.step(dt=1.0)

        assert order == ["source", "mid", "sink"]


class TestSerialization:
    def test_to_json_writes_components_and_connections(self, tmp_path: Path) -> None:
        plant = BiogasPlant("My Plant")
        a = _StubComponent("a", "digester")
        b = _StubComponent("b", "chp")
        plant.components.update({"a": a, "b": b})
        plant.connections.append(Connection("a", "b", "gas"))

        path = tmp_path / "plant.json"
        plant.to_json(str(path))

        cfg = json.loads(path.read_text())
        assert cfg["plant_name"] == "My Plant"
        assert {c["component_id"] for c in cfg["components"]} == {"a", "b"}
        # Connection.to_dict serialises as {"from", "to", "type"}
        assert cfg["connections"][0]["from"] == "a"
        assert cfg["connections"][0]["to"] == "b"
        assert cfg["connections"][0]["type"] == "gas"


class TestSummary:
    def test_summary_lists_components_and_connections(self) -> None:
        plant = BiogasPlant("My Plant")
        a = _StubComponent("a", "digester")
        b = _StubComponent("b", "chp")
        plant.components.update({"a": a, "b": b})
        plant.connections.append(Connection("a", "b", "gas"))

        summary = plant.get_summary()

        assert "My Plant" in summary
        assert "a" in summary and "b" in summary
        assert "gas" in summary
