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
        # Scalar attrs are echoed by to_dict so they reach Graph node params.
        self._params = {k: v for k, v in attrs.items() if isinstance(v, (int, float, str, bool))}

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
            **self._params,
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


class TestStepInputRouting:
    def test_step_ignores_inputs_referring_to_unknown_components(self) -> None:
        plant = BiogasPlant()
        comp = _StubComponent("a", "digester")
        comp.inputs = ["ghost"]  # dangling reference, e.g. after a component was dropped
        plant.components["a"] = comp

        plant.step(dt=1.0)

        assert comp.step_calls[0]["inputs"] == {}


class TestHeaterExecution:
    """Heaters are deferred past Pass 3 so they see the CHP's actual heat output."""

    @staticmethod
    def _chp_plant() -> BiogasPlant:
        plant = BiogasPlant()
        digester = _StubComponent("d1", "digester")
        digester.outputs_data = {"T_digester": 311.15}
        storage = _StubComponent("s1", "storage")
        storage.outputs_data = {"Q_gas_supplied_m3_per_day": 4000.0, "vented_volume_m3": 0.0}
        chp = _StubComponent("chp1", "chp", P_el_nom=500.0, eta_el=0.4)
        chp.outputs_data = {"P_th": 550.0}
        heater = _StubComponent("h1", "heating")
        plant.components.update({"d1": digester, "s1": storage, "chp1": chp, "h1": heater})
        plant.add_connection(Connection("d1", "s1", "gas"))
        plant.add_connection(Connection("s1", "chp1", "gas"))
        plant.add_connection(Connection("chp1", "h1", "heat"))
        plant.add_connection(Connection("d1", "h1", "liquid"))
        return plant

    def test_heater_receives_chp_heat_and_upstream_process_values(self) -> None:
        plant = self._chp_plant()

        plant.step(dt=1.0)

        heater_inputs = plant.components["h1"].step_calls[-1]["inputs"]
        assert heater_inputs["P_th_available"] == pytest.approx(550.0)
        # Non-heat upstream data (here the digester temperature) is forwarded too.
        assert heater_inputs["T_digester"] == pytest.approx(311.15)

    def test_chp_heat_is_split_evenly_between_two_heaters(self) -> None:
        plant = self._chp_plant()
        plant.components["h2"] = _StubComponent("h2", "heating")
        plant.add_connection(Connection("chp1", "h2", "heat"))

        plant.step(dt=1.0)

        for hid in ("h1", "h2"):
            assert plant.components[hid].step_calls[-1]["inputs"]["P_th_available"] == pytest.approx(275.0)

    def test_heater_without_a_chp_still_runs_with_zero_available_heat(self) -> None:
        plant = BiogasPlant()
        digester = _StubComponent("d1", "digester")
        digester.outputs_data = {"T_digester": 308.15}
        heater = _StubComponent("h1", "heating")
        plant.components.update({"d1": digester, "h1": heater})
        plant.add_connection(Connection("d1", "h1", "liquid"))

        results = plant.step(dt=1.0)

        assert "h1" in results
        heater_inputs = plant.components["h1"].step_calls[-1]["inputs"]
        assert heater_inputs["P_th_available"] == 0.0
        assert heater_inputs["T_digester"] == pytest.approx(308.15)


def _showcase_plant() -> BiogasPlant:
    """Plant carrying one component of every drawable type, plus all edge media."""
    plant = BiogasPlant("Showcase Plant")
    plant.components.update(
        {
            "d1": _StubComponent("d1", "digester", V_liq=1200.0, V_gas=200.0, T_ad=315.15),
            "s1": _StubComponent("s1", "storage", capacity_m3=500.0),
            "chp1": _StubComponent("chp1", "chp", P_el_nom=500.0, eta_el=0.42),
            "bg1": _StubComponent("bg1", "upgrading", capacity_m3h=300.0),
            "fl1": _StubComponent("fl1", "flare", destruction_efficiency=0.98),
            "sep1": _StubComponent("sep1", "separator", separation_efficiency=0.25),
            "ht1": _StubComponent("ht1", "heating", target_temperature=311.15),  # Kelvin
            "ht2": _StubComponent("ht2", "heating", target_temperature=38.0),  # Celsius
            "mx1": _StubComponent("mx1", "mixer"),  # no sizing params -> empty info line
        }
    )
    for src, dst, etype in [
        ("d1", "s1", "gas"),
        ("s1", "chp1", "gas"),
        ("s1", "bg1", "gas"),
        ("bg1", "fl1", "gas"),
        ("d1", "chp1", "gas"),  # spans two columns -> wider bow
        ("d1", "sep1", "liquid"),  # adjacent liquid -> straight
        ("sep1", "mx1", "liquid"),
        ("chp1", "ht1", "heat"),
        ("chp1", "ht2", "heat"),
    ]:
        plant.add_connection(Connection(src, dst, etype))
    return plant


class TestToGraph:
    def test_graph_mirrors_components_and_connections(self) -> None:
        plant = _showcase_plant()

        graph = plant.to_graph()

        assert set(graph.nodes) == set(plant.components)
        assert graph.nodes["d1"].ctype == "digester"
        assert graph.nodes["d1"].params["V_liq"] == 1200.0
        assert len(graph.edges) == len(plant.connections)
        assert {e.etype for e in graph.edges} == {"gas", "liquid", "heat"}

    def test_auto_created_component_types_are_flagged(self) -> None:
        graph = _showcase_plant().to_graph()

        assert graph.nodes["s1"].auto is True  # storages are auto-attached to digesters
        assert graph.nodes["fl1"].auto is True
        assert graph.nodes["d1"].auto is False

    def test_edge_lookup_helpers_filter_by_medium(self) -> None:
        graph = _showcase_plant().to_graph()

        assert {e.dst for e in graph.out_edges("s1")} == {"chp1", "bg1"}
        assert {e.dst for e in graph.out_edges("chp1", "heat")} == {"ht1", "ht2"}
        assert graph.out_edges("chp1", "liquid") == []
        assert {e.src for e in graph.in_edges("chp1", "gas")} == {"s1", "d1"}


class TestVisualizeGraph:
    """Smoke tests for the PNG renderer -- it must run and write a file."""

    def test_renders_every_component_type_and_edge_medium(self, tmp_path: Path) -> None:
        pytest.importorskip("networkx")
        out = tmp_path / "showcase.png"

        written = _showcase_plant().visualize_graph(output_path=str(out), title="Showcase", dpi=60)

        assert written == str(out)
        assert out.stat().st_size > 0

    def test_handles_a_cyclic_plant_without_a_topological_order(self, tmp_path: Path) -> None:
        pytest.importorskip("networkx")
        plant = BiogasPlant("Cyclic Plant")
        plant.components.update(
            {
                "d1": _StubComponent("d1", "digester", V_liq=1000.0),
                "d2": _StubComponent("d2", "digester", V_liq=800.0),
            }
        )
        plant.add_connection(Connection("d1", "d2", "liquid"))
        plant.add_connection(Connection("d2", "d1", "liquid"))
        plant.add_connection(Connection("d1", "d1", "liquid"))  # recirculation onto itself
        out = tmp_path / "cyclic.png"

        # Recycle loops must not raise; the layout falls back to a single column.
        assert plant.visualize_graph(output_path=str(out), dpi=60) == str(out)
        assert out.stat().st_size > 0

    def test_edges_pointing_outside_the_graph_are_skipped(self, tmp_path: Path) -> None:
        pytest.importorskip("networkx")
        from pyadm1.configurator.graph import Edge

        plant = _showcase_plant()
        graph = plant.to_graph()
        graph.edges.append(Edge("d1", "not_a_component", "gas"))
        plant.to_graph = lambda: graph  # type: ignore[method-assign]
        out = tmp_path / "dangling.png"

        assert plant.visualize_graph(output_path=str(out), dpi=60) == str(out)

    def test_default_output_path_is_derived_from_the_plant_name(self) -> None:
        pytest.importorskip("networkx")
        plant = _showcase_plant()
        plant.plant_name = "Graph Default Path Test"

        written = Path(plant.visualize_graph(dpi=60))
        try:
            assert written.name == "Graph_Default_Path_Test_graph.png"
            assert written.parent.name == "output"
            assert written.stat().st_size > 0
        finally:
            written.unlink(missing_ok=True)


class TestFromJson:
    def test_reports_a_component_type_without_a_registered_class(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pyadm1.components.registry import get_registry

        path = tmp_path / "plant.json"
        path.write_text(
            json.dumps(
                {
                    "plant_name": "Broken",
                    "components": [{"component_id": "chp1", "component_type": "chp", "name": "chp1"}],
                    "connections": [],
                }
            )
        )
        monkeypatch.setattr(get_registry(), "get_registered_components", dict)

        with pytest.raises(ValueError, match="No registered class for component type"):
            BiogasPlant.from_json(str(path))


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
