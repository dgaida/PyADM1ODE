# -*- coding: utf-8 -*-
"""Unit tests for the component registry."""

from types import ModuleType

import pytest

import pyadm1.components.registry as registry_module
from pyadm1.components.base import Component, ComponentType
from pyadm1.components.registry import ComponentRegistry


class DummyComponent(Component):
    """Minimal concrete component for registry tests."""

    def __init__(self, component_id: str, foo=None, name=None):  # noqa: ANN001
        super().__init__(component_id, ComponentType.MIXER, name)
        self.foo = foo

    def step(self, t, dt, inputs):  # noqa: ANN001
        return {}

    def initialize(self, initial_state=None):  # noqa: ANN001
        self.state = {}

    def to_dict(self):
        return {"component_id": self.component_id}

    @classmethod
    def from_dict(cls, config):
        return cls(config["component_id"])


class TestComponentRegistryInitialization:
    """Constructor and auto-registration behavior."""

    def test_init_creates_registry_and_calls_auto_register(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = []

        def fake_auto_register(self):  # noqa: ANN001
            calls.append(self)

        monkeypatch.setattr(ComponentRegistry, "_auto_register_components", fake_auto_register)
        registry = ComponentRegistry()

        assert registry._registry == {}
        assert calls == [registry]

    def test_auto_register_components_registers_all_when_imports_succeed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins

        target_imports = {
            "pyadm1.components.biological.digester": (
                "Digester",
                type("Digester", (), {}),
            ),
            "pyadm1.components.biological.hydrolysis": (
                "Hydrolysis",
                type("Hydrolysis", (), {}),
            ),
            "pyadm1.components.biological.separator": (
                "Separator",
                type("Separator", (), {}),
            ),
            "pyadm1.components.energy.chp": ("CHP", type("CHP", (), {})),
            "pyadm1.components.energy.heating": (
                "HeatingSystem",
                type("HeatingSystem", (), {}),
            ),
            "pyadm1.components.energy.boiler": ("Boiler", type("Boiler", (), {})),
            "pyadm1.components.energy.gas_storage": (
                "GasStorage",
                type("GasStorage", (), {}),
            ),
            "pyadm1.components.energy.flare": ("Flare", type("Flare", (), {})),
        }
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: ANN001, A002
            if name in target_imports:
                attr_name, cls_obj = target_imports[name]
                module = ModuleType(name)
                setattr(module, attr_name, cls_obj)
                return module
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        registry = ComponentRegistry.__new__(ComponentRegistry)
        registry._registry = {}

        registry._auto_register_components()

        assert set(registry._registry.keys()) == {
            "Digester",
            "Hydrolysis",
            "Separator",
            "CHP",
            "HeatingSystem",
            "Boiler",
            "GasStorage",
            "Flare",
        }

    def test_auto_register_components_skips_import_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins

        prefixes = (
            "pyadm1.components.biological.digester",
            "pyadm1.components.biological.hydrolysis",
            "pyadm1.components.biological.separator",
            "pyadm1.components.energy.chp",
            "pyadm1.components.energy.heating",
            "pyadm1.components.energy.boiler",
            "pyadm1.components.energy.gas_storage",
            "pyadm1.components.energy.flare",
        )
        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: ANN001, A002
            if name in prefixes:
                raise ImportError(f"blocked import: {name}")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        registry = ComponentRegistry.__new__(ComponentRegistry)
        registry._registry = {}

        registry._auto_register_components()

        assert registry._registry == {}


class TestComponentRegistryOperations:
    """Register/create/query/unregister methods."""

    @staticmethod
    def _empty_registry() -> ComponentRegistry:
        registry = ComponentRegistry.__new__(ComponentRegistry)
        registry._registry = {}
        return registry

    def test_register_and_duplicate_register_error(self) -> None:
        registry = self._empty_registry()

        registry.register("Dummy", DummyComponent)
        assert registry._registry["Dummy"] is DummyComponent

        with pytest.raises(ValueError, match="already registered"):
            registry.register("Dummy", DummyComponent)

    def test_unregister_success_and_missing_raises(self) -> None:
        registry = self._empty_registry()
        registry._registry["Dummy"] = DummyComponent

        registry.unregister("Dummy")
        assert "Dummy" not in registry._registry

        with pytest.raises(KeyError, match="not registered"):
            registry.unregister("Dummy")

    def test_create_success_and_missing_component_error(self) -> None:
        registry = self._empty_registry()
        registry._registry["Dummy"] = DummyComponent

        instance = registry.create("Dummy", "comp_1", foo=123)
        assert isinstance(instance, DummyComponent)
        assert instance.component_id == "comp_1"
        assert instance.foo == 123
        assert instance.name == "comp_1"

        with pytest.raises(KeyError, match="Available components"):
            registry.create("Missing", "x1")

    def test_registry_query_methods_return_expected_values_and_copies(self) -> None:
        registry = self._empty_registry()
        registry._registry["Dummy"] = DummyComponent

        registered = registry.get_registered_components()
        assert registered == {"Dummy": DummyComponent}
        assert registry.is_registered("Dummy") is True
        assert registry.is_registered("Nope") is False
        assert registry.list_components() == ["Dummy"]

        registered["Injected"] = object
        assert "Injected" not in registry._registry


class TestGlobalRegistryHelpers:
    """Global singleton registry helper functions."""

    def test_get_registry_creates_and_caches_singleton(self, monkeypatch: pytest.MonkeyPatch) -> None:
        created = []

        class FakeRegistry:
            def __init__(self):
                created.append(self)

        monkeypatch.setattr(registry_module, "_global_registry", None)
        monkeypatch.setattr(registry_module, "ComponentRegistry", FakeRegistry)

        reg1 = registry_module.get_registry()
        reg2 = registry_module.get_registry()

        assert reg1 is reg2
        assert len(created) == 1

    def test_register_component_uses_global_registry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = []

        class FakeRegistry:
            def register(self, name, component_class):  # noqa: ANN001
                calls.append((name, component_class))

        monkeypatch.setattr(registry_module, "get_registry", lambda: FakeRegistry())

        registry_module.register_component("Dummy", DummyComponent)

        assert calls == [("Dummy", DummyComponent)]
