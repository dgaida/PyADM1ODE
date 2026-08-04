"""Unit tests for the ADM1 Digester component (SIMBA# biogas, 41-state)."""

from types import SimpleNamespace

import numpy as np
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
            d.step(
                t=float(day),
                dt=1.0,
                inputs={"Q_substrates": [11.4, 6.1, 0, 0, 0, 0, 0, 0, 0, 0]},
            )

        out = d.step(t=3.0, dt=1.0, inputs={"Q_substrates": [11.4, 6.1, 0, 0, 0, 0, 0, 0, 0, 0]})

        # Sanity check: significant methane production within plausible range.
        assert 200.0 < out["Q_ch4"] < 5000.0
        assert 6.5 < out["pH"] < 8.5


class TestDigesterRobustness:
    """Degraded inputs must not take the whole simulation down."""

    def test_a_corrupt_gas_storage_state_falls_back_to_a_fresh_storage(self, feedstock: Feedstock) -> None:
        d = Digester("dig_1", feedstock, V_liq=1200.0, V_gas=216.0)

        d.initialize({"gas_storage": {"stored_volume_m3": object()}})

        assert d._initialized
        assert d.gas_storage.stored_volume_m3 >= 0.0
        assert d.gas_storage.outputs_data["stored_volume_m3"] >= 0.0

    def test_a_valid_gas_storage_state_is_restored(self, feedstock: Feedstock) -> None:
        d = Digester("dig_1", feedstock, V_liq=1200.0, V_gas=216.0)

        d.initialize({"gas_storage": {"stored_volume_m3": 42.0}})

        assert d.gas_storage.stored_volume_m3 == pytest.approx(42.0)

    def test_a_failing_ph_solve_falls_back_to_neutral(self, feedstock: Feedstock, monkeypatch) -> None:
        """The acid-base Newton solve is numeric; a blow-up must not abort the step."""
        d = Digester("dig_1", feedstock, V_liq=1200.0, V_gas=216.0)
        d.initialize({"Q_substrates": [11.4, 6.1, 0, 0, 0, 0, 0, 0, 0, 0]})

        def exploding_ph(*args, **kwargs):
            raise FloatingPointError("charge balance did not converge")

        monkeypatch.setattr(type(d.adm1), "_calc_ph", staticmethod(exploding_ph))

        indicators = d._compute_indicators()

        assert indicators["pH"] == 7.0
        # The remaining indicators are still computed from the state.
        assert indicators["VFA"] >= 0.0
        assert indicators["TS"] >= 0.0

    def test_a_failed_integration_names_the_component(self, feedstock: Feedstock, monkeypatch) -> None:
        import pyadm1.components.biological.digester as digester_mod

        d = Digester("dig_broken", feedstock, V_liq=1200.0, V_gas=216.0)
        d.initialize({"Q_substrates": [11.4, 6.1, 0, 0, 0, 0, 0, 0, 0, 0]})

        def failing_solve_ivp(**kwargs):
            return SimpleNamespace(success=False, message="step size underflow", y=None)

        monkeypatch.setattr(digester_mod, "solve_ivp", failing_solve_ivp)

        with pytest.raises(RuntimeError, match=r"ADM1 integration failed in 'dig_broken'.*step size underflow"):
            d.step(t=0.0, dt=1.0, inputs={"Q_substrates": [11.4, 6.1, 0, 0, 0, 0, 0, 0, 0, 0]})


class TestDigesterHydraulicRetentionTime:
    """HRT follows a first-order lag toward V/Q, and simply ages without feed."""

    def test_hrt_grows_with_elapsed_time_when_the_digester_is_not_fed(self, feedstock: Feedstock) -> None:
        d = Digester("dig_1", feedstock, V_liq=1200.0, V_gas=216.0)
        d.initialize({"Q_substrates": [0.0] * 10})

        d.step(t=0.0, dt=1.0, inputs={"Q_substrates": [0.0] * 10})
        after_one_day = d.state["HRT"]
        d.step(t=1.0, dt=2.0, inputs={"Q_substrates": [0.0] * 10})

        # Nothing flows through, so the sludge just keeps ageing: HRT += dt.
        assert after_one_day == pytest.approx(1.0)
        assert d.state["HRT"] == pytest.approx(3.0)

    def test_hrt_relaxes_towards_the_volume_over_flow_ratio_when_fed(self, feedstock: Feedstock) -> None:
        d = Digester("dig_1", feedstock, V_liq=1200.0, V_gas=216.0)
        Q = [11.4, 6.1, 0, 0, 0, 0, 0, 0, 0, 0]
        d.initialize({"Q_substrates": Q})

        for day in range(30):
            d.step(t=float(day), dt=1.0, inputs={"Q_substrates": Q})

        # dHRT/dt = 1 - (Q_in/V)*HRT with HRT(0) = 0 solves to (V/Q)(1 - exp(-Q t / V)).
        Q_in = float(np.sum(d.adm1._Q))
        a = Q_in / 1200.0
        expected = (1.0 / a) * (1.0 - np.exp(-a * 30.0))

        assert d.state["HRT"] == pytest.approx(expected, rel=1e-9)
        assert d.state["HRT"] < 1200.0 / Q_in  # still approaching the steady state from below


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

    def test_dynamic_volume_round_trip_restores_the_live_sludge_volume(self, feedstock: Feedstock) -> None:
        """A partially filled dynamic-volume digester must not snap back to its nominal fill."""
        d = Digester(
            "dig_dyn",
            feedstock,
            V_liq=1200.0,
            V_gas=216.0,
            dynamic_volume=True,
            initial_fill_fraction=1.0,
        )
        d.initialize({"Q_substrates": [11.4, 6.1, 0, 0, 0, 0, 0, 0, 0, 0]})
        d.V_liq = 900.0  # e.g. drawn down during operation
        d.adm1.V_liq = 900.0
        d.state["V_liq"] = 900.0

        restored = Digester.from_dict(d.to_dict(), feedstock)

        assert restored._dynamic_volume is True
        assert restored.V_liq == pytest.approx(900.0)
        assert restored.adm1.V_liq == pytest.approx(900.0)
        assert restored.state["V_liq"] == pytest.approx(900.0)
        # The nominal maximum is kept separately, so the weir still knows its target.
        assert restored._V_liq_max == pytest.approx(1200.0)

    def test_a_fixed_volume_digester_ignores_a_stored_volume_entry(self, feedstock: Feedstock) -> None:
        d = Digester("dig_fixed", feedstock, V_liq=1200.0, V_gas=216.0, dynamic_volume=False)
        d.initialize({"Q_substrates": [0.0] * 10})
        config = d.to_dict()
        config["state"] = {**config["state"], "V_liq": 900.0}

        restored = Digester.from_dict(config, feedstock)

        assert restored._dynamic_volume is False
        assert restored.V_liq == pytest.approx(1200.0)

    def test_round_trip_preserves_the_adm1_backend(self, feedstock: Feedstock) -> None:
        d = Digester("dig_np", feedstock, V_liq=1200.0, V_gas=216.0)

        restored = Digester.from_dict(d.to_dict(), feedstock)

        assert restored.adm1.backend == d.adm1.backend
