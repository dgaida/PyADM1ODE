# ============================================================================
# pyadm1/components/energy/gas_storage.py
# ====================================================================
# Gas storage component implementation
# ====================================================================
"""
Gas storage component.

Supports:
- low-pressure storage: 'membrane' or 'dome' (small overpressure, linear head model)
- high-pressure storage: 'compressed' (pressure increases strongly with stored fraction)
Features:
- capacity accounting (m^3 at STP)
- gas composition (CH4 / CO2 fraction) of the stored gas, ideally mixed
- simple pressure model (bar)
- pressure setpoint handling via inputs['set_pressure'] (bar)
- safety venting if pressure > p_max (vented volume sent to flare)
- outputs_data contains stored_volume, pressure, utilization, vented_volume, Q_gas_supplied,
  CH4_fraction, CO2_fraction, Q_ch4_supplied
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..base import Component, ComponentType


@dataclass
class GasStorageConfig:
    """Configuration parameters for a GasStorage component (type, capacity, pressure limits, initial fill)."""

    storage_type: str = "membrane"  # 'membrane' | 'dome' | 'compressed'
    capacity_m3: float = 1000.0  # usable volume at STP [m^3]
    p_atm_bar: float = 1.01325  # atmospheric pressure [bar]
    p_min_bar: float = 0.9  # minimum allowed pressure for supply [bar]
    p_max_bar: float = 1.05  # maximum allowed pressure (low-pressure overpressure) or safety setpoint [bar]
    # For compressed storage, p_max_bar should be high (e.g. 200.0 bar) and p_min_bar > p_atm_bar
    initial_fill_fraction: float = 0.1  # initial stored / capacity


class GasStorage(Component):
    """
    Gas storage component.

    Arguments:
        component_id: unique id
        storage_type: 'membrane' | 'dome' | 'compressed'
        capacity_m3: usable gas volume at STP (m^3)
        p_min_bar: minimum operating pressure (bar)
        p_max_bar: maximum safe pressure (bar)
        name: optional human-readable name
        initial_ch4_fraction: CH4 volume fraction of the initial fill (0-1)
        initial_co2_fraction: CO2 volume fraction of the initial fill (0-1)

    The stored gas is treated as ideally mixed: inflow is blended into the
    inventory first, and everything leaving in the same step (supply, overflow,
    safety vent) carries the blended composition. The remainder
    ``1 - CH4 - CO2`` is H2, water vapour and trace gases.
    """

    def __init__(
        self,
        component_id: str,
        storage_type: str = "membrane",
        capacity_m3: float = 1000.0,
        p_min_bar: float = 0.95,
        p_max_bar: float = 1.05,
        initial_fill_fraction: float = 0.1,
        name: str | None = None,
        initial_ch4_fraction: float = 0.60,
        initial_co2_fraction: float = 0.40,
    ):
        """

        Args:
            component_id: unique id
            storage_type: 'membrane' | 'dome' | 'compressed'
            capacity_m3: usable gas volume at STP (m^3)
            p_min_bar: minimum operating pressure (bar)
            p_max_bar: maximum safe pressure (bar)
            initial_fill_fraction: initial stored fraction of capacity (0-1)
            name: optional human-readable name
            initial_ch4_fraction: CH4 volume fraction of the initial fill (0-1)
            initial_co2_fraction: CO2 volume fraction of the initial fill (0-1)
        """
        super().__init__(component_id, ComponentType.STORAGE, name)

        # Config
        self.storage_type = storage_type.lower()
        if self.storage_type not in ("membrane", "dome", "compressed"):
            raise ValueError("storage_type must be 'membrane', 'dome' or 'compressed'")

        self.capacity_m3 = float(capacity_m3)
        self.p_atm_bar = 1.01325
        self.p_min_bar = float(p_min_bar)
        self.p_max_bar = float(p_max_bar)

        # State variables
        self.stored_volume_m3 = float(self.capacity_m3) * float(initial_fill_fraction)
        # cumulative vented (for logging)
        self._cum_vented_m3 = 0.0

        # Composition of the stored gas (volume fractions, ideally mixed)
        self.ch4_fraction, self.co2_fraction = self._checked_fractions(initial_ch4_fraction, initial_co2_fraction)

        # control setpoint (pressure in bar) - None means no active setpoint
        self.pressure_setpoint_bar: float | None = None

        # initialize default state
        self.initialize()

    @staticmethod
    def _checked_fractions(ch4_fraction: float, co2_fraction: float) -> tuple[float, float]:
        """Validate a CH4/CO2 fraction pair: each in [0, 1], together at most 1."""
        ch4, co2 = float(ch4_fraction), float(co2_fraction)
        if not (0.0 <= ch4 <= 1.0 and 0.0 <= co2 <= 1.0) or ch4 + co2 > 1.0 + 1e-9:
            raise ValueError(f"CH4/CO2 fractions must lie in [0, 1] and sum to at most 1, got {ch4} and {co2}")
        return ch4, co2

    def initialize(self, initial_state: dict[str, Any] | None = None) -> None:
        """
        Initialize storage state; initial_state may contain stored_volume_m3,
        pressure_setpoint_bar, ch4_fraction, co2_fraction, cumulative_vented_m3.
        """
        cum_vented = 0.0
        if initial_state is not None:
            if "ch4_fraction" in initial_state or "co2_fraction" in initial_state:
                self.ch4_fraction, self.co2_fraction = self._checked_fractions(
                    initial_state.get("ch4_fraction", self.ch4_fraction),
                    initial_state.get("co2_fraction", self.co2_fraction),
                )
            if "stored_volume_m3" in initial_state:
                self.stored_volume_m3 = float(initial_state["stored_volume_m3"])
            if "pressure_setpoint_bar" in initial_state:
                self.pressure_setpoint_bar = initial_state["pressure_setpoint_bar"]
            cum_vented = float(initial_state.get("cumulative_vented_m3", 0.0))

        # clamp stored volume
        self.stored_volume_m3 = max(0.0, min(self.stored_volume_m3, self.capacity_m3))

        self._cum_vented_m3 = cum_vented

        # outputs_data template
        self.outputs_data = {
            "stored_volume_m3": self.stored_volume_m3,
            "pressure_bar": self._estimate_pressure_bar(),
            "utilization": self.stored_volume_m3 / max(1e-9, self.capacity_m3),
            "vented_volume_m3": 0.0,
            "Q_gas_supplied_m3_per_day": 0.0,
            "Q_ch4_supplied_m3_per_day": 0.0,
            "CH4_fraction": self.ch4_fraction,
            "CO2_fraction": self.co2_fraction,
            "cumulative_vented_m3": self._cum_vented_m3,
        }
        self._sync_state()

        self._initialized = True

    def _sync_state(self) -> None:
        """Mirror everything that carries over between steps into ``state``."""
        self.state = {
            "stored_volume_m3": float(self.stored_volume_m3),
            "ch4_fraction": float(self.ch4_fraction),
            "co2_fraction": float(self.co2_fraction),
            "pressure_setpoint_bar": self.pressure_setpoint_bar,
            "cumulative_vented_m3": float(self._cum_vented_m3),
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore a snapshot taken with :meth:`get_state`."""
        self.initialize(state)

    # ------------------------------
    # Internal helpers
    # ------------------------------
    def _estimate_pressure_bar(self) -> float:
        """
        Estimate pressure [bar] from stored volume and storage type.

        Models:
        - low-pressure (membrane/dome): small overpressure scales with stored fraction
            p = p_atm + frac * (p_max_bar - p_atm)
        - compressed: pressure scales from p_min to p_max with stored fraction
            p = p_min + frac^alpha * (p_max - p_min)  (alpha > 1 to reflect nonlinear increase)
        """
        frac = 0.0
        if self.capacity_m3 > 0:
            frac = max(0.0, min(self.stored_volume_m3 / self.capacity_m3, 1.0))

        if self.storage_type in ("membrane", "dome"):
            # low-pressure head/membrane; small overpressure
            p = self.p_atm_bar + frac * (self.p_max_bar - self.p_atm_bar)
        else:  # compressed
            # use nonlinear exponent to mimic gas compression behaviour (very simplified)
            alpha = 2.0
            p = self.p_min_bar + (frac**alpha) * (self.p_max_bar - self.p_min_bar)

        return float(p)

    # ------------------------------
    # Main simulation step
    # ------------------------------
    def step(self, t: float, dt: float, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        One simulation step.

        Args:
            t:
            dt:
            inputs: Inputs dictionary may contain:
                - 'Q_gas_in_m3_per_day'   : gas inflow from digesters/other sources (m^3/day)
                - 'Q_ch4_in_m3_per_day'   : methane within that inflow (m^3/day) (optional)
                - 'Q_co2_in_m3_per_day'   : CO2 within that inflow (m^3/day) (optional)
                  Inflow without a composition is taken to match the stored gas.
                - 'Q_gas_out_m3_per_day'  : requested gas outflow (demand) (m^3/day)
                - 'set_pressure'          : desired pressure setpoint (bar)  (optional)
                - 'vent_to_flare'         : bool, if True allow venting to flare when overpressure (default True)

        Returns:
            object: Returns outputs_data with keys:
                - 'stored_volume_m3'
                - 'pressure_bar'
                - 'utilization' (0-1)
                - 'vented_volume_m3' (this timestep)
                - 'Q_gas_supplied_m3_per_day' (actual supply that was delivered)
                - 'Q_ch4_supplied_m3_per_day' (methane within that supply)
                - 'CH4_fraction', 'CO2_fraction' (composition of the stored gas)
        """
        # get flows (units: m^3/day) -> convert to volume for this timestep: m^3 = Q * dt (dt in days)
        Q_in = float(inputs.get("Q_gas_in_m3_per_day", 0.0))
        Q_out_req = float(inputs.get("Q_gas_out_m3_per_day", 0.0))

        # pressure setpoint handling (bar) - if provided, store it for internal control
        if "set_pressure" in inputs:
            sp = inputs["set_pressure"]
            try:
                self.pressure_setpoint_bar = None if sp is None else float(sp)
            except (TypeError, ValueError) as e:
                # ignore invalid setpoint
                print(e)

        # convert to volumes for this dt
        vol_in = Q_in * dt
        vol_out_req = Q_out_req * dt

        vented_this_step = 0.0
        # supplied_this_step = 0.0

        # Composition: blend the inflow into the inventory before any gas
        # leaves. Supply, overflow and safety vent below then all carry this
        # blended composition, so the fractions need no further update.
        blended_m3 = self.stored_volume_m3 + vol_in
        if vol_in > 0.0 and blended_m3 > 0.0:
            ch4_in = inputs.get("Q_ch4_in_m3_per_day")
            co2_in = inputs.get("Q_co2_in_m3_per_day")
            ch4_in_frac = self.ch4_fraction if ch4_in is None else min(max(float(ch4_in) / Q_in, 0.0), 1.0)
            co2_in_frac = self.co2_fraction if co2_in is None else min(max(float(co2_in) / Q_in, 0.0), 1.0 - ch4_in_frac)
            self.ch4_fraction = (self.stored_volume_m3 * self.ch4_fraction + vol_in * ch4_in_frac) / blended_m3
            self.co2_fraction = (self.stored_volume_m3 * self.co2_fraction + vol_in * co2_in_frac) / blended_m3

        # Inflow and outflow of one step happen at the same time, so the
        # inflow is added first and only what is left after the supply can
        # overflow. Venting before the supply would make the vented volume
        # depend on the step length (a daily step would vent a full day's
        # production that the consumers burn in the same step).
        self.stored_volume_m3 += vol_in

        # Supply (outflow): cannot supply more than stored
        # If pressure below p_min, limit supply (simulate pressure control)
        current_pressure = self._estimate_pressure_bar()

        # If a pressure setpoint is requested and higher than current,
        # we might choose to deny outflow to increase pressure
        restrict_factor = 1.0
        # If a pressure setpoint above the current pressure is requested, restrict
        # outflow to prioritize charging: reduce allowed outflow proportionally,
        # factor between 0..1 where 0 blocks outflow and 1 allows full outflow.
        if self.pressure_setpoint_bar is not None and self.pressure_setpoint_bar > current_pressure:
            gap = self.pressure_setpoint_bar - current_pressure
            span = max(1e-6, self.p_max_bar - self.p_atm_bar)
            restrict_factor = max(0.0, 1.0 - gap / span)

        allowed_out_vol = self.stored_volume_m3 * restrict_factor  # simple limit
        desired_out_vol = min(vol_out_req, allowed_out_vol)

        # Also ensure we don't drop below zero or below some minimum reserve needed to maintain p_min:
        # approximate required volume to keep p >= p_min: invert pressure estimate crudely by fraction
        if self.storage_type in ("membrane", "dome"):
            # frac required to maintain p_min: (p_min - p_atm)/(p_max - p_atm)
            if (self.p_max_bar - self.p_atm_bar) > 1e-9:
                frac_needed = max(
                    0.0,
                    (self.p_min_bar - self.p_atm_bar) / (self.p_max_bar - self.p_atm_bar),
                )
            else:
                frac_needed = 0.0
        else:  # compressed
            # invert nonlinear mapping p = p_min + frac^alpha*(p_max-p_min)
            if (self.p_max_bar - self.p_min_bar) > 1e-9:
                alpha = 2.0
                frac_needed = max(
                    0.0,
                    ((self.p_min_bar - self.p_min_bar) / (self.p_max_bar - self.p_min_bar)) ** (1.0 / alpha),
                )
                # above formula trivial -> 0, but keep general structure; use small reserve fraction
                frac_needed = 0.01
            else:
                frac_needed = 0.0

        reserve_volume = frac_needed * self.capacity_m3
        # ensure not to supply below reserve
        max_out_vol_after_reserve = max(0.0, self.stored_volume_m3 - reserve_volume)
        desired_out_vol = min(desired_out_vol, max_out_vol_after_reserve)

        # perform outflow
        self.stored_volume_m3 -= desired_out_vol
        supplied_this_step_m3_per_day = desired_out_vol / max(dt, 1e-12)  # convert back to m3/day

        # Overflow: gas beyond the capacity leaves through the safety vent.
        # ``vent_to_flare`` only says where it goes; the storage cannot hold
        # it either way, so it is always counted as vented.
        overflow = self.stored_volume_m3 - self.capacity_m3
        if overflow > 1e-12:
            vented_this_step += overflow
            self._cum_vented_m3 += overflow
            self.stored_volume_m3 = self.capacity_m3

        current_pressure = self._estimate_pressure_bar()

        # Clamp stored volume
        self.stored_volume_m3 = max(0.0, min(self.stored_volume_m3, self.capacity_m3))

        # Update outputs_data
        self.outputs_data = {
            "stored_volume_m3": float(self.stored_volume_m3),
            "pressure_bar": float(current_pressure),
            "utilization": float(self.stored_volume_m3 / max(1e-9, self.capacity_m3)),
            "vented_volume_m3": float(vented_this_step),
            "Q_gas_supplied_m3_per_day": float(supplied_this_step_m3_per_day),
            "Q_ch4_supplied_m3_per_day": float(supplied_this_step_m3_per_day * self.ch4_fraction),
            "CH4_fraction": float(self.ch4_fraction),
            "CO2_fraction": float(self.co2_fraction),
            "cumulative_vented_m3": float(self._cum_vented_m3),
            "pressure_setpoint_bar": self.pressure_setpoint_bar,
        }
        self._sync_state()

        return self.outputs_data

    def withdraw(self, volume_m3: float) -> float:
        """
        Take gas out of the storage outside of :meth:`step`.

        Used by the plant for consumers whose demand is only known after the
        storage was stepped (the auxiliary boiler). The supply reported for
        the step is not changed.

        Args:
            volume_m3 (float): Volume asked for [m³].

        Returns:
            float: Volume actually taken [m³], limited to what is stored.
        """
        taken = min(max(float(volume_m3), 0.0), self.stored_volume_m3)
        self.stored_volume_m3 -= taken
        self.outputs_data["stored_volume_m3"] = float(self.stored_volume_m3)
        self.outputs_data["pressure_bar"] = self._estimate_pressure_bar()
        self.outputs_data["utilization"] = float(self.stored_volume_m3 / max(1e-9, self.capacity_m3))
        self._sync_state()
        return taken

    # ------------------------------
    # Serialization
    # ------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration + current state."""
        return {
            "component_id": self.component_id,
            "component_type": self.component_type.value,
            "name": self.name,
            "storage_type": self.storage_type,
            "capacity_m3": self.capacity_m3,
            "p_atm_bar": self.p_atm_bar,
            "p_min_bar": self.p_min_bar,
            "p_max_bar": self.p_max_bar,
            "stored_volume_m3": self.stored_volume_m3,
            "ch4_fraction": self.ch4_fraction,
            "co2_fraction": self.co2_fraction,
            "pressure_setpoint_bar": self.pressure_setpoint_bar,
            "cumulative_vented_m3": self._cum_vented_m3,
            "outputs_data": self.outputs_data,
            "inputs": self.inputs,
            "outputs": self.outputs,
        }

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> GasStorage:
        """Create GasStorage from dict produced by to_dict."""
        gs = cls(
            component_id=config["component_id"],
            storage_type=config.get("storage_type", "membrane"),
            capacity_m3=config.get("capacity_m3", 1000.0),
            p_min_bar=config.get("p_min_bar", 0.95),
            p_max_bar=config.get("p_max_bar", 1.05),
            initial_fill_fraction=0.0,
            name=config.get("name"),
            initial_ch4_fraction=config.get("ch4_fraction", 0.60),
            initial_co2_fraction=config.get("co2_fraction", 0.40),
        )
        # restore state if present
        if "stored_volume_m3" in config:
            try:
                gs.stored_volume_m3 = float(config["stored_volume_m3"])
            except (TypeError, ValueError) as e:
                print(e)

        if "pressure_setpoint_bar" in config:
            gs.pressure_setpoint_bar = config.get("pressure_setpoint_bar")

        gs.initialize(
            {
                "stored_volume_m3": gs.stored_volume_m3,
                "pressure_setpoint_bar": gs.pressure_setpoint_bar,
                "cumulative_vented_m3": config.get("cumulative_vented_m3", 0.0),
            }
        )
        gs.inputs = config.get("inputs", [])
        gs.outputs = config.get("outputs", [])

        return gs
