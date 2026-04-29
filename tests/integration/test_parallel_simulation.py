# -*- coding: utf-8 -*-
"""Integration test: ParallelSimulator end-to-end (sequential mode)."""

import pytest

from pyadm1 import Feedstock
from pyadm1.core.adm1 import ADM1, STATE_SIZE
from pyadm1.simulation.parallel import ParallelSimulator


@pytest.mark.integration
def test_run_two_scenarios_sequentially() -> None:
    feedstock = Feedstock(
        ["maize_silage_milk_ripeness", "swine_manure"],
        feeding_freq=24,
        total_simtime=3,
    )
    adm1 = ADM1(feedstock, V_liq=1200.0, V_gas=216.0, T_ad=315.15)
    adm1.set_influent_dataframe(feedstock.get_influent_dataframe(Q=[11.4, 6.1]))
    adm1.create_influent([11.4, 6.1], 0)

    initial_state = [0.01] * STATE_SIZE
    initial_state[37:41] = [1.0e-5, 0.65, 0.33, 0.65 + 0.33 + 1.0e-5]

    parallel = ParallelSimulator(adm1, n_workers=1, verbose=False)
    scenarios = [
        {"Q": [10.0, 5.0, 0, 0, 0, 0, 0, 0, 0, 0]},
        {"Q": [12.0, 6.0, 0, 0, 0, 0, 0, 0, 0, 0]},
    ]

    results = parallel.run_scenarios(
        scenarios=scenarios,
        duration=2.0,
        initial_state=initial_state,
        compute_metrics=True,
    )

    assert len(results) == 2
    for r in results:
        if not r.success:
            pytest.fail(f"scenario {r.scenario_id} failed: {r.error}")
        assert "Q_gas" in r.metrics
        assert "Q_ch4" in r.metrics
        assert r.metrics["Q_gas"] >= 0.0
