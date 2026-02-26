import pytest

from pyadm1.simulation.simulator import Simulator


class _FakeADM1:
    def __init__(self):
        self.feedstock = object()


class _FakeSolver:
    pass


def test_determine_best_feed_by_n_sims_raises_for_n_less_than_3():
    sim = Simulator(_FakeADM1(), solver=_FakeSolver())

    with pytest.raises(ValueError, match="n must be at least 3"):
        sim.determine_best_feed_by_n_sims(
            state_zero=[0.0] * 37,
            Q=[0.0] * 10,
            Qch4sp=1.0,
            feeding_freq=48,
            n=2,
        )


def test_simulator_properties_return_injected_instances():
    adm1 = _FakeADM1()
    solver = _FakeSolver()
    sim = Simulator(adm1, solver=solver)

    assert sim.adm1 is adm1
    assert sim.solver is solver
