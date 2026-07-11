# -*- coding: utf-8 -*-
"""Digester-level tests for the switchable ADM1 right-hand-side backend.

The torch backend must be a drop-in for the numpy one: same construction API,
same simulation trajectory, selectable per digester, via config, or via a
process-wide default.
"""

import numpy as np
import pytest

from pyadm1 import Feedstock, get_default_adm1_backend, set_default_adm1_backend
from pyadm1.components.biological import Digester

pytest.importorskip("torch")

_Q = [11.4, 6.1, 0, 0, 0, 0, 0, 0, 0, 0]


@pytest.fixture
def feedstock() -> Feedstock:
    return Feedstock(
        ["maize_silage_milk_ripeness", "swine_manure"],
        feeding_freq=24,
        total_simtime=10,
    )


def _make(feedstock: Feedstock, backend) -> Digester:
    d = Digester("dig_1", feedstock, V_liq=1200.0, V_gas=216.0, T_ad=315.15, backend=backend)
    d.initialize({"Q_substrates": _Q})
    return d


def test_backend_defaults_to_numpy(feedstock: Feedstock) -> None:
    assert _make(feedstock, None).adm1.backend == "numpy"


def test_backend_explicit_torch(feedstock: Feedstock) -> None:
    assert _make(feedstock, "torch").adm1.backend == "torch"


def test_step_equivalence_numpy_vs_torch(feedstock: Feedstock) -> None:
    """A multi-day run must produce the same trajectory and outputs."""
    d_np = _make(feedstock, "numpy")
    d_pt = _make(feedstock, "torch")

    # Same starting state (pre-inoculation is deterministic, but be explicit).
    d_pt.adm1_state = list(d_np.adm1_state)

    for day in range(4):
        out_np = d_np.step(t=float(day), dt=1.0, inputs={"Q_substrates": _Q})
        out_pt = d_pt.step(t=float(day), dt=1.0, inputs={"Q_substrates": _Q})

        np.testing.assert_allclose(out_pt["state_out"], out_np["state_out"], rtol=1e-5, atol=1e-8)
        for key in ("Q_gas", "Q_ch4", "Q_co2", "pH", "VFA", "TAC"):
            assert out_pt[key] == pytest.approx(out_np[key], rel=1e-5, abs=1e-6)


def test_dynamic_volume_equivalence(feedstock: Feedstock) -> None:
    """The torch adapter must also drive the dynamic-volume balance identically."""
    kwargs = dict(V_liq=1200.0, V_gas=216.0, T_ad=315.15, dynamic_volume=True, initial_fill_fraction=0.8)
    d_np = Digester("dig_1", feedstock, backend="numpy", **kwargs)
    d_pt = Digester("dig_1", feedstock, backend="torch", **kwargs)
    d_np.initialize({"Q_substrates": _Q})
    d_pt.initialize({"Q_substrates": _Q})
    d_pt.adm1_state = list(d_np.adm1_state)

    for day in range(4):
        d_np.step(t=float(day), dt=1.0, inputs={"Q_substrates": _Q})
        d_pt.step(t=float(day), dt=1.0, inputs={"Q_substrates": _Q})
        assert d_pt.V_liq == pytest.approx(d_np.V_liq, rel=1e-6)


def test_global_default_backend(feedstock: Feedstock) -> None:
    """set_default_adm1_backend routes new digesters without an explicit arg."""
    assert get_default_adm1_backend() == "numpy"
    try:
        set_default_adm1_backend("torch")
        d = _make(feedstock, None)
        assert d.adm1.backend == "torch"
        # An explicit argument still wins over the global default.
        assert _make(feedstock, "numpy").adm1.backend == "numpy"
    finally:
        set_default_adm1_backend("numpy")
    assert get_default_adm1_backend() == "numpy"


def test_config_roundtrip_preserves_backend(feedstock: Feedstock) -> None:
    d = _make(feedstock, "torch")
    cfg = d.to_dict()
    assert cfg["adm1_backend"] == "torch"
    rebuilt = Digester.from_dict(cfg, feedstock=feedstock)
    assert rebuilt.adm1.backend == "torch"


def test_configurator_passes_backend(feedstock: Feedstock) -> None:
    from pyadm1 import BiogasPlant
    from pyadm1.configurator.plant_configurator import PlantConfigurator

    plant = BiogasPlant("t")
    cfg = PlantConfigurator(plant, feedstock)
    digester, _ = cfg.add_digester("dig_1", backend="torch", Q_substrates=_Q)
    assert digester.adm1.backend == "torch"
