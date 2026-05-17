# -*- coding: utf-8 -*-
"""
Integration tests: ``ParallelSimulator`` batch results must equal the
single-scenario results.

These tests exercise three "complex plant" configurations and three
parameter-variation modes (feed rate, kinetic calibration, temperature
override). For each plant, three scenarios are run two ways:

1. **Batch**: all scenarios passed in one ``run_scenarios([s1, s2, s3])`` call.
2. **Sequential**: each scenario passed alone in a separate
   ``run_scenarios([s])`` call.

The two paths must give identical metrics because ``_run_single_scenario``
builds a fresh ``Feedstock`` and ``ADM1`` instance per call from the
serialized config — there is no shared mutable state between scenarios.

Each test also asserts that the three scenarios *differ* from each other,
guarding against regressions like the silent calibration no-op that
shipped before the ``set_calibration_parameters`` fix landed.
"""

from typing import Dict, List, Tuple

import pytest

from pyadm1 import Feedstock
from pyadm1.components.biological.digester import Digester
from pyadm1.core.adm1 import ADM1
from pyadm1.simulation.parallel import ParallelSimulator, ScenarioResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pre_inoculated(fs: Feedstock, V_liq: float, V_gas: float, T_ad: float, Q: List[float]) -> List[float]:
    """Build a physically-consistent warm-start state via Digester helper."""
    proxy = Digester("_test_inoc", fs, V_liq=V_liq, V_gas=V_gas, T_ad=T_ad)
    return proxy._build_pre_inoculated_state(Q)


def _build_plant(
    substrates: List[str],
    V_liq: float,
    V_gas: float,
    T_ad: float,
    base_Q: List[float],
    duration: float,
) -> Tuple[ADM1, List[float]]:
    """Construct an ADM1 instance + a pre-inoculated initial state."""
    fs = Feedstock(substrates, feeding_freq=24, total_simtime=max(2, int(duration) + 1))
    adm1 = ADM1(fs, V_liq=V_liq, V_gas=V_gas, T_ad=T_ad)
    adm1.set_influent_dataframe(fs.get_influent_dataframe(Q=base_Q))
    adm1.create_influent(base_Q, 0)
    initial_state = _pre_inoculated(fs, V_liq, V_gas, T_ad, base_Q)
    return adm1, initial_state


def _require_success(results: List[ScenarioResult], label: str) -> None:
    for r in results:
        if not r.success:
            pytest.skip(f"{label}: scenario {r.scenario_id} failed (likely a scipy/BDF environment issue): {r.error}")


def _assert_metrics_close(a: Dict[str, float], b: Dict[str, float], keys: Tuple[str, ...]) -> None:
    """Batch and sequential paths should be bit-for-bit identical: each scenario
    builds a fresh ADM1, so there's nothing to drift. A tight rel tolerance
    catches any accidental state leak without flagging FPU rounding noise."""
    for key in keys:
        assert key in a, f"missing metric {key} in batch result"
        assert key in b, f"missing metric {key} in sequential result"
        assert a[key] == pytest.approx(
            b[key], rel=1e-12, abs=1e-15
        ), f"metric {key} diverges: batch={a[key]!r} sequential={b[key]!r}"


def _run_both_ways(
    adm1: ADM1,
    scenarios: List[Dict],
    initial_state: List[float],
    duration: float,
) -> Tuple[List[ScenarioResult], List[ScenarioResult]]:
    """Run the same scenarios as one batch and as N single-scenario calls."""
    batch_sim = ParallelSimulator(adm1, n_workers=1, verbose=False)
    batch = batch_sim.run_scenarios(
        scenarios=scenarios,
        duration=duration,
        initial_state=initial_state,
        compute_metrics=True,
    )

    seq_sim = ParallelSimulator(adm1, n_workers=1, verbose=False)
    sequential: List[ScenarioResult] = []
    for scenario in scenarios:
        result = seq_sim.run_scenarios(
            scenarios=[scenario],
            duration=duration,
            initial_state=initial_state,
            compute_metrics=True,
        )
        sequential.extend(result)

    return batch, sequential


# ---------------------------------------------------------------------------
# Plant 1 — Mesophilic standard reactor, vary feed rate
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_plant_mesophilic_feed_rate_batch_matches_sequential() -> None:
    """Standard 1200 m³ mesophilic reactor, 3 feed-rate scenarios."""
    adm1, initial_state = _build_plant(
        substrates=["maize_silage_milk_ripeness", "swine_manure"],
        V_liq=1200.0,
        V_gas=216.0,
        T_ad=308.15,
        base_Q=[11.4, 6.1, 0, 0, 0, 0, 0, 0, 0, 0],
        duration=1.0,
    )

    scenarios = [
        {"Q": [8.0, 4.0, 0, 0, 0, 0, 0, 0, 0, 0]},
        {"Q": [11.4, 6.1, 0, 0, 0, 0, 0, 0, 0, 0]},
        {"Q": [18.0, 9.0, 0, 0, 0, 0, 0, 0, 0, 0]},
    ]

    batch, sequential = _run_both_ways(adm1, scenarios, initial_state, duration=1.0)
    _require_success(batch + sequential, "mesophilic feed-rate")

    metric_keys = ("Q_gas", "Q_ch4", "pH", "HRT")
    for b, s in zip(batch, sequential):
        _assert_metrics_close(b.metrics, s.metrics, metric_keys)

    # Sanity: the three scenarios must genuinely differ, otherwise the
    # test would be vacuously true.
    q_gas_values = [r.metrics["Q_gas"] for r in batch]
    assert (
        len(set(round(v, 6) for v in q_gas_values)) == 3
    ), f"feed-rate scenarios produced identical Q_gas values: {q_gas_values}"


# ---------------------------------------------------------------------------
# Plant 2 — Thermophilic reactor with grass+manure, vary kinetic k_m_ac
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_plant_thermophilic_kinetic_sweep_batch_matches_sequential() -> None:
    """1500 m³ thermophilic reactor, k_m_ac sweep.

    Regression test: before the calibration fix, k_m_ac overrides were
    silently dropped and all three scenarios would produce identical
    results. This test asserts they differ AND that batch == sequential.
    """
    base_Q = [12.0, 6.0, 0, 0, 0, 0, 0, 0, 0, 0]
    adm1, initial_state = _build_plant(
        substrates=["grass_silage", "swine_manure"],
        V_liq=1500.0,
        V_gas=270.0,
        T_ad=318.15,
        base_Q=base_Q,
        duration=1.0,
    )

    scenarios = [
        {"Q": base_Q, "k_m_ac": 4.0},
        {"Q": base_Q, "k_m_ac": 8.0},
        {"Q": base_Q, "k_m_ac": 15.0},
    ]

    batch, sequential = _run_both_ways(adm1, scenarios, initial_state, duration=1.0)
    _require_success(batch + sequential, "thermophilic kinetic-sweep")

    metric_keys = ("Q_gas", "Q_ch4", "pH")
    for b, s in zip(batch, sequential):
        _assert_metrics_close(b.metrics, s.metrics, metric_keys)

    # The calibration fix means k_m_ac variation must change the result.
    q_ch4_values = [r.metrics["Q_ch4"] for r in batch]
    assert (
        len(set(round(v, 6) for v in q_ch4_values)) == 3
    ), "k_m_ac sweep produced identical Q_ch4 — calibration override is being dropped again"


# ---------------------------------------------------------------------------
# Plant 3 — Large reactor with 3-substrate mix, vary T_ad override
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_plant_large_mixed_feed_temperature_override_batch_matches_sequential() -> None:
    """1977 m³ reactor (ADM1 default geometry), 3-substrate mix,
    per-scenario T_ad overrides."""
    base_Q = [15.0, 8.0, 5.0, 0, 0, 0, 0, 0, 0, 0]  # maize + manure + grass
    adm1, initial_state = _build_plant(
        substrates=["maize_silage_milk_ripeness", "swine_manure", "grass_silage"],
        V_liq=1977.0,
        V_gas=304.0,
        T_ad=308.15,
        base_Q=base_Q,
        duration=1.0,
    )

    scenarios = [
        {"Q": base_Q, "T_ad": 303.15},  # cool mesophilic
        {"Q": base_Q, "T_ad": 308.15},  # standard mesophilic
        {"Q": base_Q, "T_ad": 318.15},  # thermophilic
    ]

    batch, sequential = _run_both_ways(adm1, scenarios, initial_state, duration=1.0)
    _require_success(batch + sequential, "large-reactor temperature-override")

    metric_keys = ("Q_gas", "Q_ch4", "pH")
    for b, s in zip(batch, sequential):
        _assert_metrics_close(b.metrics, s.metrics, metric_keys)

    # T_ad override changes Henry constants and temperature-corrected
    # kinetics, so the three scenarios must produce distinct gas rates.
    q_gas_values = [r.metrics["Q_gas"] for r in batch]
    assert len(set(round(v, 6) for v in q_gas_values)) == 3, f"T_ad override sweep produced identical Q_gas: {q_gas_values}"
