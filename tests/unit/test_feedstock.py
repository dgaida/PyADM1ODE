import sys
import types

import numpy as np
import pytest

import pyadm1.substrates.feedstock as feedstock_mod
from pyadm1.substrates.feedstock import Feedstock


class _FakePhysValue:
    def __init__(self, value, text=None):
        self.Value = value
        self._text = text if text is not None else str(value)

    def printValue(self):
        return self._text


class _FakeSubstrate:
    def calcXc(self):
        return _FakePhysValue(1.23, "XcVal")

    def calcCOD_SX(self):
        return _FakePhysValue(2.34, "CODSXVal")

    def calcTOC(self):
        return _FakePhysValue(3.45, "TOCVal")

    def calcCtoNratio(self):
        return 12.345


class _FakeSubstrates:
    def __init__(self):
        self._sub = _FakeSubstrate()

    def getNumSubstrates(self):
        return 2

    def getID(self, i):
        return f"s{i}"

    def get(self, substrate_id):
        return self._sub

    def get_param_of(self, substrate_id, key):
        values = {"pH": 7.2, "TS": 25.0, "VS": 80.0, "BMP": 0.3339, "TKN": 1.234}
        return values[key]


def test_get_influent_dataframe_raises_when_row_header_lengths_mismatch(monkeypatch):
    fs = Feedstock(feeding_freq=24, total_simtime=2)
    monkeypatch.setattr(fs, "_mixADMstreams", lambda Q: [1.0, 2.0])  # too short for header

    with pytest.raises(ValueError, match="Data rows do not match the header length"):
        fs.get_influent_dataframe([0.0] * 10)


def test_get_substrate_feed_mixtures_covers_adjustments_and_random_branch(monkeypatch):
    vals = iter([0.0, 1.0, 0.5, 0.25])  # deterministic for two random scenarios
    monkeypatch.setattr(feedstock_mod.np.random, "uniform", lambda: next(vals))

    q = [10.0, 20.0, 0.0]
    mixed = Feedstock.get_substrate_feed_mixtures(q, n=5)

    assert len(mixed) == 5
    assert mixed[0][:2] == [10.0, 20.0]
    assert mixed[1][:2] == [11.5, 21.5]
    assert mixed[2][:2] == [8.5, 18.5]
    assert mixed[3][0] == 8.5  # 10 + 0*3 - 1.5
    assert mixed[3][1] == 21.5  # 20 + 1*3 - 1.5
    assert mixed[4][0] == 10.0  # 10 + 0.5*3 - 1.5
    assert mixed[4][1] == 19.25  # 20 + 0.25*3 - 1.5


def test_calc_olr_from_toc_uses_all_substrates(monkeypatch):
    monkeypatch.setattr(Feedstock, "_mySubstrates", _FakeSubstrates())
    monkeypatch.setattr(Feedstock, "_get_TOC", classmethod(lambda cls, sid: _FakePhysValue(2.0 if sid == "s1" else 3.0)))

    olr = Feedstock.calc_OLR_fromTOC([10.0, 20.0], V_liq=10.0)

    assert olr == pytest.approx((2.0 * 10.0 + 3.0 * 20.0) / 10.0)


def test_get_substrate_params_string_formats_expected_fields(monkeypatch):
    monkeypatch.setattr(Feedstock, "_mySubstrates", _FakeSubstrates())
    monkeypatch.setattr(Feedstock, "_get_TOC", classmethod(lambda cls, sid: _FakePhysValue(3.45, "TOCVal")))

    text = Feedstock.get_substrate_params_string("s1")

    assert "pH value: 7.2" in text
    assert "Particulate chemical oxygen demand: XcVal" in text
    assert "Particulate disintegrated chemical oxygen demand: CODSXVal" in text
    assert "Total organic carbon: TOCVal" in text
    assert "Carbon-to-Nitrogen ratio: 12.34" in text


def test_get_toc_returns_calc_toc_result(monkeypatch):
    monkeypatch.setattr(Feedstock, "_mySubstrates", _FakeSubstrates())

    toc = Feedstock._get_TOC("s1")

    assert isinstance(toc, _FakePhysValue)
    assert toc.Value == 3.45


def test_mix_admstreams_falls_back_to_numpy_2d(monkeypatch):
    class FakeSubstratesForMix:
        def getNumSubstrates(self):
            return 2

        def get(self, i):
            return f"sub{i}"

    class FakeADMState:
        @staticmethod
        def calcADMstream(substrate, q):
            return [q, q + 1]

        @staticmethod
        def mixADMstreams(data):
            arr = np.asarray(data)
            if arr.ndim == 2:
                return [100.0, 200.0]
            raise RuntimeError("unexpected")

    system_mod = types.ModuleType("System")

    class _Array:
        @staticmethod
        def CreateInstance(*args, **kwargs):
            raise RuntimeError("force first path failure")

    system_mod.Array = _Array
    system_mod.Double = float

    monkeypatch.setattr(Feedstock, "_mySubstrates", FakeSubstratesForMix())
    monkeypatch.setattr(feedstock_mod, "ADMstate", FakeADMState)
    monkeypatch.setitem(sys.modules, "System", system_mod)

    result = Feedstock._mixADMstreams([1.0, 2.0])

    assert result == [100.0, 200.0]


def test_mix_admstreams_falls_back_to_flattened_1d(monkeypatch):
    class FakeSubstratesForMix:
        def getNumSubstrates(self):
            return 2

        def get(self, i):
            return f"sub{i}"

    class FakeADMState:
        @staticmethod
        def calcADMstream(substrate, q):
            return [q, q + 1]

        @staticmethod
        def mixADMstreams(data):
            arr = np.asarray(data)
            if arr.ndim == 2:
                raise RuntimeError("numpy 2d failed")
            if arr.ndim == 1:
                return [7.0, 8.0]
            raise RuntimeError("unexpected")

    system_mod = types.ModuleType("System")

    class _Array:
        @staticmethod
        def CreateInstance(*args, **kwargs):
            raise RuntimeError("force first path failure")

    system_mod.Array = _Array
    system_mod.Double = float

    monkeypatch.setattr(Feedstock, "_mySubstrates", FakeSubstratesForMix())
    monkeypatch.setattr(feedstock_mod, "ADMstate", FakeADMState)
    monkeypatch.setitem(sys.modules, "System", system_mod)

    result = Feedstock._mixADMstreams([1.0, 2.0])

    assert result == [7.0, 8.0]


def test_mix_admstreams_raises_type_error_when_all_paths_fail(monkeypatch):
    class FakeSubstratesForMix:
        def getNumSubstrates(self):
            return 1

        def get(self, i):
            return "sub1"

    class FakeADMState:
        @staticmethod
        def calcADMstream(substrate, q):
            return [q]

        @staticmethod
        def mixADMstreams(data):
            raise RuntimeError("mix failed")

    system_mod = types.ModuleType("System")

    class _Array:
        @staticmethod
        def CreateInstance(*args, **kwargs):
            raise RuntimeError("force first path failure")

    system_mod.Array = _Array
    system_mod.Double = float

    monkeypatch.setattr(Feedstock, "_mySubstrates", FakeSubstratesForMix())
    monkeypatch.setattr(feedstock_mod, "ADMstate", FakeADMState)
    monkeypatch.setitem(sys.modules, "System", system_mod)

    with pytest.raises(TypeError, match="Failed to mix ADM streams"):
        Feedstock._mixADMstreams([1.0])


def test_header_and_simtime_methods_return_internal_values():
    fs = Feedstock(feeding_freq=24, total_simtime=3)

    assert fs.header() == Feedstock._header
    np.testing.assert_allclose(fs.simtime(), np.array([0.0, 1.0, 2.0]))
