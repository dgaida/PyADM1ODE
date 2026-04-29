# -*- coding: utf-8 -*-
"""Unit tests for the ADM1 Digester component (SIMBA# biogas, 41-state)."""

import pytest

from pyadm1 import Feedstock
from pyadm1.components.biological import Digester
from pyadm1.components.energy.gas_storage import GasStorage
from pyadm1.core.adm1 import STATE_SIZE


@pytest.fixture
def feedstock() -> Feedstock:
    """Real feedstock built from the bundled XML library."""
    return Feedstock(
        ["maize_silage_milk_ripeness", "swine_manure"],
        feeding_freq=24,
        total_simtime=10,
    )


class TestDigesterInitialization:
    """Constructor behaviour."""

    def test_sets_component_id_and_volumes(self, feedstock: Feedstock) -> None:
        d = Digester("dig_1", feedstock, V_liq=1200.0, V_gas=216.0, T_ad=315.15)

        assert d.component_id == "dig_1"
        assert d.V_liq == 1200.0
        assert d.V_gas == 216.0
        assert d.T_ad == 315.15
        assert d.component_type.value == "digester"

    def test_creates_attached_gas_storage(self, feedstock: Feedstock) -> None:
        d = Digester("dig_1", feedstock, V_gas=216.0)

        assert isinstance(d.gas_storage, GasStorage)
        assert d.gas_storage.component_id == "dig_1_storage"

    def test_default_name_falls_back_to_component_id(self, feedstock: Feedstock) -> None:
        d = Digester("dig_1", feedstock)
        assert d.name == "dig_1"

    def test_initialize_with_no_substrate_uses_default_state(self, feedstock: Feedstock) -> None:
        d = Digester("dig_1", feedstock)
        d.initialize()

        assert len(d.adm1_state) == STATE_SIZE
        assert d._initialized is True
        assert d.state["pH"] == 7.0
        assert d.state["Q_gas"] == 0.0

    def test_initialize_with_q_substrates_builds_pre_inoculated_state(self, feedstock: Feedstock) -> None:
        d = Digester("dig_1", feedstock, V_liq=1200.0, V_gas=216.0, T_ad=315.15)
        d.initialize({"Q_substrates": [11.4, 6.1, 0, 0, 0, 0, 0, 0, 0, 0]})

        assert len(d.adm1_state) == STATE_SIZE
        # Pre-inoculated state seeds biomass above the washout threshold
        assert d.adm1_state[27] > 0.5  # X_ac
        assert d.state["Q_substrates"][0] == 11.4

    def test_initialize_with_user_supplied_state_uses_it(self, feedstock: Feedstock) -> None:
        custom = [0.001 * (i + 1) for i in range(STATE_SIZE)]
        d = Digester("dig_1", feedstock)
        d.initialize({"adm1_state": custom, "Q_substrates": [11.4, 6.1] + [0.0] * 8})

        assert d.adm1_state == custom


class TestDigesterStep:
    """Single-step integration behaviour."""

    def test_step_advances_state_and_returns_outputs(self, feedstock: Feedstock) -> None:
        d = Digester("dig_1", feedstock, V_liq=1200.0, V_gas=216.0, T_ad=315.15)
        d.initialize({"Q_substrates": [11.4, 6.1, 0, 0, 0, 0, 0, 0, 0, 0]})

        out = d.step(t=0.0, dt=1.0, inputs={"Q_substrates": [11.4, 6.1, 0, 0, 0, 0, 0, 0, 0, 0]})

        assert out["Q_gas"] >= 0.0
        assert out["Q_ch4"] >= 0.0
        assert out["Q_co2"] >= 0.0
        assert 4.5 < out["pH"] < 9.0
        assert "gas_storage" in out
        assert "stored_volume_m3" in out["gas_storage"]
        assert len(out["state_out"]) == STATE_SIZE

    def test_step_after_warmup_produces_realistic_methane(self, feedstock: Feedstock) -> None:
        d = Digester("dig_1", feedstock, V_liq=1200.0, V_gas=216.0, T_ad=315.15)
        d.initialize({"Q_substrates": [11.4, 6.1, 0, 0, 0, 0, 0, 0, 0, 0]})

        # Warm-up
        for day in range(3):
            d.step(t=float(day), dt=1.0, inputs={"Q_substrates": [11.4, 6.1, 0, 0, 0, 0, 0, 0, 0, 0]})

        out = d.step(t=3.0, dt=1.0, inputs={"Q_substrates": [11.4, 6.1, 0, 0, 0, 0, 0, 0, 0, 0]})

        # Sanity check: significant methane production within plausible range.
        assert 200.0 < out["Q_ch4"] < 5000.0
        assert 6.5 < out["pH"] < 8.5


class TestDigesterSerialization:
    """to_dict / from_dict round-trip."""

    def test_to_dict_returns_config(self, feedstock: Feedstock) -> None:
        d = Digester("dig_1", feedstock, V_liq=1200.0, V_gas=216.0, T_ad=315.15)
        cfg = d.to_dict()

        assert cfg["component_id"] == "dig_1"
        assert cfg["component_type"] == "digester"
        assert cfg["V_liq"] == 1200.0
        assert cfg["V_gas"] == 216.0
        assert cfg["T_ad"] == 315.15

    def test_from_dict_recreates_instance_with_defaults(self, feedstock: Feedstock) -> None:
        d = Digester.from_dict({"component_id": "dig_from_cfg"}, feedstock)

        assert isinstance(d, Digester)
        assert d.component_id == "dig_from_cfg"
        assert d.feedstock is feedstock
