"""
Liquid routing between components: mixing several inflows and splitting one
outflow.

Both behaviours were missing before: a post-digester fed by two fermenters saw
only one of them (the last one processed silently overwrote the other), and a
separator could not send part of its press water back to a digester because a
connection always carried the full flow.
"""

from __future__ import annotations

import pytest

from pyadm1.components.biological.separator import Separator
from pyadm1.configurator.connection_manager import Connection


class _StubComponent:
    """Minimal stand-in for a component with a liquid effluent."""

    def __init__(self, component_id: str, q_out: float, state: list[float] | None = None):
        self.component_id = component_id
        self.outputs_data: dict = {"Q_out": q_out}
        if state is not None:
            self.outputs_data["state_out"] = state


def _route(targets_inputs, components, connections):
    """Reproduce the liquid-routing block of ``plant_builder`` for one component.

    Kept in the test so the expected contract is spelled out explicitly:
    accumulate the flow, weight the state by flow, honour ``split_fraction``.
    """
    inputs: dict = {}
    q_liquid = 0.0
    state_load: list[float] | None = None
    for input_id in targets_inputs:
        out = components[input_id].outputs_data
        conn = next(c for c in connections if c.from_component == input_id)
        if conn.connection_type != "liquid":
            inputs.update(out)
            continue
        q = float(out.get("Q_out", 0.0)) * conn.split_fraction
        if q <= 0.0:
            continue
        q_liquid += q
        state = out.get("state_out")
        if state is not None:
            if state_load is None:
                state_load = [0.0] * len(state)
            for idx, conc in enumerate(state):
                state_load[idx] += q * float(conc)
    if q_liquid > 0.0:
        inputs["Q_in"] = q_liquid
        if state_load is not None:
            inputs["state_in"] = [load / q_liquid for load in state_load]
    return inputs


class TestConnectionSplitFraction:
    def test_defaults_to_full_flow(self):
        conn = Connection("a", "b", "liquid")
        assert conn.split_fraction == 1.0

    def test_full_flow_is_not_serialized(self):
        assert "split_fraction" not in Connection("a", "b", "liquid").to_dict()

    def test_partial_flow_round_trips(self):
        data = Connection("sep", "F1", "liquid", split_fraction=0.55).to_dict()
        assert data["split_fraction"] == pytest.approx(0.55)
        assert Connection.from_dict(data).split_fraction == pytest.approx(0.55)


class TestLiquidMixing:
    def test_two_inflows_are_summed(self):
        components = {"F1": _StubComponent("F1", 64.0), "F2": _StubComponent("F2", 48.0)}
        connections = [Connection("F1", "N1", "liquid"), Connection("F2", "N1", "liquid")]
        assert _route(["F1", "F2"], components, connections)["Q_in"] == pytest.approx(112.0)

    def test_state_is_flow_weighted(self):
        components = {
            "F1": _StubComponent("F1", 30.0, [10.0, 0.0]),
            "F2": _StubComponent("F2", 10.0, [2.0, 4.0]),
        }
        connections = [Connection("F1", "N1", "liquid"), Connection("F2", "N1", "liquid")]
        mixed = _route(["F1", "F2"], components, connections)["state_in"]
        # (30*10 + 10*2)/40 = 8.0 ; (30*0 + 10*4)/40 = 1.0
        assert mixed == pytest.approx([8.0, 1.0])

    def test_split_fraction_scales_the_flow(self):
        components = {"sep": _StubComponent("sep", 20.0, [5.0])}
        connections = [Connection("sep", "F1", "liquid", split_fraction=0.55)]
        routed = _route(["sep"], components, connections)
        assert routed["Q_in"] == pytest.approx(11.0)
        # concentrations are unaffected by the split — only the flow is
        assert routed["state_in"] == pytest.approx([5.0])


class TestSeparatorLiquidStream:
    """The separator has to hand the liquid phase on as a real ADM1 stream."""

    @staticmethod
    def _state(particulate: float = 4.0, soluble: float = 0.5) -> list[float]:
        state = [soluble] * 37
        for idx in range(12, 25):
            state[idx] = particulate
        return state

    def test_emits_cascade_keys(self):
        sep = Separator("sep", separator_type="screw_press")
        sep.initialize()
        out = sep.step(t=0.0, dt=1.0, inputs={"Q_in": 40.0, "state_in": self._state()})
        assert out["Q_out"] == pytest.approx(out["Q_liquid"])
        assert out["state_out"] is not None
        assert len(out["state_out"]) == 37

    def test_particulates_are_depleted_solubles_are_not(self):
        sep = Separator("sep", separator_type="screw_press")
        sep.initialize()
        q_in = 40.0
        out = sep.step(t=0.0, dt=1.0, inputs={"Q_in": q_in, "state_in": self._state()})
        keep = (1.0 - sep.separation_efficiency) * q_in / out["Q_liquid"]
        assert out["state_out"][12] == pytest.approx(4.0 * keep)
        assert out["state_out"][0] == pytest.approx(0.5)

    def test_particulate_load_is_conserved(self):
        sep = Separator("sep", separator_type="screw_press")
        sep.initialize()
        q_in = 40.0
        out = sep.step(t=0.0, dt=1.0, inputs={"Q_in": q_in, "state_in": self._state()})
        load_in = 4.0 * q_in
        load_liquid = out["state_out"][12] * out["Q_liquid"]
        assert load_liquid == pytest.approx(load_in * (1.0 - sep.separation_efficiency))

    def test_ts_is_taken_from_the_routed_state_key(self):
        """``plant_builder`` passes ``state_in``; the old code only read ``state_out``."""
        sep = Separator("sep", separator_type="screw_press")
        sep.initialize()
        via_in = sep.step(t=0.0, dt=1.0, inputs={"Q_in": 40.0, "state_in": self._state()})
        solids_from_state = via_in["Q_solid"]

        sep2 = Separator("sep2", separator_type="screw_press")
        sep2.initialize()
        without_state = sep2.step(t=0.0, dt=1.0, inputs={"Q_in": 40.0})  # 40 kg/m3 fallback
        assert solids_from_state != pytest.approx(without_state["Q_solid"])

    def test_without_upstream_state_it_stays_a_reporting_component(self):
        sep = Separator("sep", separator_type="screw_press")
        sep.initialize()
        out = sep.step(t=0.0, dt=1.0, inputs={"Q_in": 40.0, "TS_in": 60.0})
        assert out["state_out"] is None
        assert out["Q_out"] == pytest.approx(out["Q_liquid"])
