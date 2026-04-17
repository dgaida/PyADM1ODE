# -*- coding: utf-8 -*-
"""
Unit tests for the HeatingSystem component and heating DLL helper functions.
"""

import math
import sys
from types import SimpleNamespace

import pyadm1.components.energy.heating as heating_module
from pyadm1.components.energy.heating import HeatingSystem


def _reset_heating_globals() -> None:
    heating_module._PHYSVALUE = None
    heating_module._BIOGAS = None
    heating_module._SUBSTRATES_FACTORY = None
    heating_module._SUBSTRATES_INSTANCE = None
    heating_module._DLL_INIT_DONE = False
    heating_module._HEAT_CALC_MODE = None


class TestHeatingInitialization:
    """Test suite for HeatingSystem initialization."""

    def test_heating_initialization_sets_defaults(self) -> None:
        heating = HeatingSystem("heat_1")
        assert heating.target_temperature == 308.15
        assert heating.heat_loss_coefficient == 0.5

    def test_initialize_sets_state_and_outputs(self) -> None:
        heating = HeatingSystem("heat_1")
        heating.initialize()

        assert "Q_heat_demand" in heating.state
        assert "Q_heat_supplied" in heating.state
        assert "energy_consumed" in heating.state
        assert "Q_heat_supplied" in heating.outputs_data
        assert "P_th_used" in heating.outputs_data
        assert "P_aux_heat" in heating.outputs_data


class TestHeatingStep:
    """Test suite for HeatingSystem step method."""

    def test_step_basic_heat_balance(self, monkeypatch) -> None:
        monkeypatch.setattr(heating_module, "_calc_process_heat_kw", lambda *_: 0.0)
        heating = HeatingSystem("heat_1", heat_loss_coefficient=0.5)
        heating.initialize()

        inputs = {
            "T_digester": 308.15,
            "T_ambient": 288.15,
            "P_th_available": 6.0,
        }

        result = heating.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert math.isclose(result["Q_heat_supplied"], 10.0, rel_tol=1e-6)
        assert math.isclose(result["P_th_used"], 6.0, rel_tol=1e-6)
        assert math.isclose(result["P_aux_heat"], 4.0, rel_tol=1e-6)
        assert math.isclose(heating.state["energy_consumed"], 4.0, rel_tol=1e-6)

    def test_step_zero_temperature_difference(self, monkeypatch) -> None:
        monkeypatch.setattr(heating_module, "_calc_process_heat_kw", lambda *_: 0.0)
        heating = HeatingSystem("heat_1", heat_loss_coefficient=0.5)
        heating.initialize()

        inputs = {
            "T_digester": 300.0,
            "T_ambient": 300.0,
            "P_th_available": 50.0,
        }

        result = heating.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert math.isclose(result["Q_heat_supplied"], 0.0, rel_tol=1e-6)
        assert math.isclose(result["P_th_used"], 0.0, rel_tol=1e-6)
        assert math.isclose(result["P_aux_heat"], 0.0, rel_tol=1e-6)

    def test_step_chp_covers_demand(self, monkeypatch) -> None:
        monkeypatch.setattr(heating_module, "_calc_process_heat_kw", lambda *_: 0.0)
        heating = HeatingSystem("heat_1", heat_loss_coefficient=0.5)
        heating.initialize()

        inputs = {
            "T_digester": 308.15,
            "T_ambient": 288.15,
            "P_th_available": 20.0,
        }

        result = heating.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)

        assert math.isclose(result["P_th_used"], 10.0, rel_tol=1e-6)
        assert math.isclose(result["P_aux_heat"], 0.0, rel_tol=1e-6)

    def test_step_includes_process_heat(self, monkeypatch) -> None:
        monkeypatch.setattr(heating_module, "_calc_process_heat_kw", lambda *_: 3.5)
        heating = HeatingSystem("heat_1", target_temperature=308.15, heat_loss_coefficient=0.5)
        heating.initialize()

        inputs = {
            "T_digester": 308.15,
            "T_ambient": 288.15,
            "P_th_available": 0.0,
            "Q_substrates": [10.0] + [0.0] * 9,
        }

        result = heating.step(t=0.0, dt=1.0 / 24.0, inputs=inputs)
        # Q_loss=10.0 and Q_process=3.5
        assert math.isclose(heating.state["Q_heat_demand"], 13.5, rel_tol=1e-6)
        assert math.isclose(result["P_aux_heat"], 13.5, rel_tol=1e-6)

    def test_step_defaults_used_when_inputs_missing(self, monkeypatch) -> None:
        monkeypatch.setattr(heating_module, "_calc_process_heat_kw", lambda *_: 0.0)
        heating = HeatingSystem("heat_1", target_temperature=300.0, heat_loss_coefficient=0.5)
        heating.initialize()

        # T_digester defaults to target_temperature, T_ambient defaults to 288.15
        result = heating.step(t=0.0, dt=1.0 / 24.0, inputs={})
        expected_q = 0.5 * (300.0 - 288.15)

        assert math.isclose(heating.state["Q_heat_demand"], expected_q, rel_tol=1e-6)
        assert math.isclose(result["P_th_used"], 0.0, rel_tol=1e-6)
        assert math.isclose(result["P_aux_heat"], expected_q, rel_tol=1e-6)


class TestHeatingSerialization:
    """Test serialization and deserialization paths."""

    def test_to_dict_contains_expected_fields(self) -> None:
        heating = HeatingSystem("heat_1", target_temperature=310.0, heat_loss_coefficient=0.7, name="H")
        d = heating.to_dict()

        assert d["component_id"] == "heat_1"
        assert d["name"] == "H"
        assert d["target_temperature"] == 310.0
        assert d["heat_loss_coefficient"] == 0.7
        assert "inputs" in d
        assert "outputs" in d
        assert "state" in d

    def test_from_dict_with_state_calls_initialize_with_state(self) -> None:
        cfg = {
            "component_id": "heat_1",
            "target_temperature": 311.0,
            "heat_loss_coefficient": 0.9,
            "name": "Heat",
            "inputs": ["a"],
            "outputs": ["b"],
            "state": {"Q_heat_demand": 1.0},
        }

        called_with = []
        original_initialize = HeatingSystem.initialize

        def fake_initialize(self, initial_state=None):  # noqa: ANN001
            called_with.append(initial_state)
            original_initialize(self, initial_state)

        try:
            HeatingSystem.initialize = fake_initialize
            heating = HeatingSystem.from_dict(cfg)
        finally:
            HeatingSystem.initialize = original_initialize

        assert heating.target_temperature == 311.0
        assert heating.heat_loss_coefficient == 0.9
        assert heating.inputs == ["a"]
        assert heating.outputs == ["b"]
        assert {"Q_heat_demand": 1.0} in called_with

    def test_from_dict_without_state_calls_default_initialize(self) -> None:
        cfg = {
            "component_id": "heat_1",
            "target_temperature": 311.0,
            "heat_loss_coefficient": 0.9,
        }
        heating = HeatingSystem.from_dict(cfg)
        assert "Q_heat_demand" in heating.state
        assert heating.state["Q_heat_demand"] == 0.0


class TestHeatingDllHelpers:
    """Test suite for DLL helper functions with mocked interop."""

    def test_init_heating_dll_skips_on_darwin(self, monkeypatch) -> None:
        _reset_heating_globals()
        monkeypatch.setattr(heating_module.platform, "system", lambda: "Darwin")
        heating_module._init_heating_dll()
        assert heating_module._DLL_INIT_DONE is True
        assert heating_module._PHYSVALUE is None

    def test_init_heating_dll_returns_if_clr_fails(self, monkeypatch) -> None:
        _reset_heating_globals()
        monkeypatch.setattr(heating_module.platform, "system", lambda: "Windows")

        import builtins

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):  # noqa: ANN001
            if name == "clr":
                raise ImportError("no clr")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        heating_module._init_heating_dll()
        assert heating_module._DLL_INIT_DONE is True
        assert heating_module._SUBSTRATES_FACTORY is None

    def test_init_heating_dll_falls_back_to_physvalue_capitalized_name(self, monkeypatch) -> None:
        _reset_heating_globals()
        monkeypatch.setattr(heating_module.platform, "system", lambda: "Windows")

        import builtins

        real_import = builtins.__import__
        fake_clr = SimpleNamespace(AddReference=lambda *_: None)
        fake_biogas = SimpleNamespace(substrates=lambda _: "factory_result")
        fake_physchem = SimpleNamespace(PhysValue=lambda value, unit: ("PhysValue", value, unit))

        monkeypatch.setitem(sys.modules, "clr", fake_clr)

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: ANN001, A002
            if name == "biogas":
                return fake_biogas
            if name == "physchem":
                if "physValue" in fromlist:
                    raise ImportError("physValue missing")
                return fake_physchem
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        heating_module._init_heating_dll()

        assert heating_module._BIOGAS is fake_biogas
        assert heating_module._SUBSTRATES_FACTORY is fake_biogas.substrates
        assert heating_module._PHYSVALUE is fake_physchem.PhysValue

    def test_init_heating_dll_returns_when_physchem_imports_both_fail(self, monkeypatch) -> None:
        _reset_heating_globals()
        monkeypatch.setattr(heating_module.platform, "system", lambda: "Windows")

        import builtins

        real_import = builtins.__import__
        fake_clr = SimpleNamespace(AddReference=lambda *_: None)
        fake_biogas = SimpleNamespace(substrates=lambda _: "factory_result")

        monkeypatch.setitem(sys.modules, "clr", fake_clr)

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: ANN001, A002
            if name == "biogas":
                return fake_biogas
            if name == "physchem":
                raise ImportError("physchem import failed")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        heating_module._init_heating_dll()

        assert heating_module._BIOGAS is None
        assert heating_module._SUBSTRATES_FACTORY is None
        assert heating_module._PHYSVALUE is None

    def test_get_substrates_instance_caches_factory_result(self, monkeypatch) -> None:
        _reset_heating_globals()
        calls = []

        def factory(path):  # noqa: ANN001
            calls.append(path)
            return {"path": path}

        heating_module._SUBSTRATES_FACTORY = factory
        heating_module._DLL_INIT_DONE = True
        one = heating_module._get_substrates_instance()
        two = heating_module._get_substrates_instance()

        assert one == two
        assert len(calls) == 1
        assert "substrate_gummersbach.xml" in calls[0]

    def test_get_substrates_instance_handles_factory_failure(self) -> None:
        _reset_heating_globals()

        def factory(_):  # noqa: ANN001
            raise RuntimeError("fail")

        heating_module._SUBSTRATES_FACTORY = factory
        heating_module._DLL_INIT_DONE = True
        assert heating_module._get_substrates_instance() is None

    def test_get_substrates_instance_returns_none_when_factory_unavailable_after_init(self, monkeypatch) -> None:
        _reset_heating_globals()
        monkeypatch.setattr(heating_module, "_init_heating_dll", lambda: None)

        assert heating_module._get_substrates_instance() is None

    def test_calc_process_heat_kw_returns_zero_for_empty_q(self) -> None:
        _reset_heating_globals()
        assert heating_module._calc_process_heat_kw([], 308.15) == 0.0

    def test_calc_process_heat_kw_returns_zero_when_substrates_missing(self, monkeypatch) -> None:
        _reset_heating_globals()
        monkeypatch.setattr(heating_module, "_get_substrates_instance", lambda: None)
        assert heating_module._calc_process_heat_kw([1.0], 308.15) == 0.0

    def test_calc_process_heat_kw_uses_substrates_calc_heat_power(self, monkeypatch) -> None:
        _reset_heating_globals()
        heating_module._PHYSVALUE = lambda value, unit: ("PV", value, unit)

        class Sub:
            @staticmethod
            def calcHeatPower(q, t):  # noqa: ANN001
                assert q == [1.0]
                assert t == ("PV", 308.15, "K")
                return SimpleNamespace(Value=12.3)

        monkeypatch.setattr(heating_module, "_get_substrates_instance", lambda: Sub())
        out = heating_module._calc_process_heat_kw([1.0], 308.15)
        assert math.isclose(out, 12.3, rel_tol=1e-9)
        assert heating_module._HEAT_CALC_MODE == "substrates_calcHeatPower"

    def test_calc_process_heat_kw_uses_admstate_calc_heat_power(self, monkeypatch) -> None:
        _reset_heating_globals()
        heating_module._PHYSVALUE = lambda value, unit: ("PV", value, unit)

        class Sub:
            pass

        heating_module._BIOGAS = SimpleNamespace(
            ADMstate=SimpleNamespace(
                calcHeatPower=lambda substrates, q, t: SimpleNamespace(Value=9.1),
            )
        )
        monkeypatch.setattr(heating_module, "_get_substrates_instance", lambda: Sub())
        out = heating_module._calc_process_heat_kw([2.0], 308.15)
        assert math.isclose(out, 9.1, rel_tol=1e-9)
        assert heating_module._HEAT_CALC_MODE == "admstate_calcHeatPower"

    def test_calc_process_heat_kw_uses_daily_heat_fallback(self, monkeypatch) -> None:
        _reset_heating_globals()
        heating_module._PHYSVALUE = lambda value, unit: ("PV", value, unit)

        class Sub:
            @staticmethod
            def calcSumQuantityOfHeatPerDay(q, t):  # noqa: ANN001
                return SimpleNamespace(Value=240.0)

        monkeypatch.setattr(heating_module, "_get_substrates_instance", lambda: Sub())
        out = heating_module._calc_process_heat_kw([2.0], 308.15)
        assert math.isclose(out, 10.0, rel_tol=1e-9)
        assert heating_module._HEAT_CALC_MODE == "substrates_calcSumQuantityOfHeatPerDay"

    def test_calc_process_heat_kw_handles_exception(self, monkeypatch) -> None:
        _reset_heating_globals()
        heating_module._PHYSVALUE = lambda value, unit: ("PV", value, unit)

        class Sub:
            @staticmethod
            def calcHeatPower(q, t):  # noqa: ANN001
                raise RuntimeError("boom")

        monkeypatch.setattr(heating_module, "_get_substrates_instance", lambda: Sub())
        out = heating_module._calc_process_heat_kw([1.0], 308.15)
        assert out == 0.0

    def test_calc_process_heat_kw_returns_zero_when_no_supported_heat_methods(self, monkeypatch) -> None:
        _reset_heating_globals()
        heating_module._PHYSVALUE = lambda value, unit: ("PV", value, unit)
        heating_module._BIOGAS = SimpleNamespace(ADMstate=SimpleNamespace())

        class Sub:
            pass

        monkeypatch.setattr(heating_module, "_get_substrates_instance", lambda: Sub())
        out = heating_module._calc_process_heat_kw([1.0], 308.15)
        assert out == 0.0
        assert heating_module._HEAT_CALC_MODE == "none"
