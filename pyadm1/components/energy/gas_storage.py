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
- simple pressure model (bar)
- pressure setpoint handling via inputs['set_pressure'] (bar)
- safety venting if pressure > p_max (vented volume sent to flare)
- outputs_data contains stored_volume, pressure, utilization, vented_volume, Q_gas_supplied
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass

from ..base import Component, ComponentType


@dataclass
class GasStorageConfig:
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
    """

    def __init__(
        self,
        component_id: str,
        storage_type: str = "membrane",
        capacity_m3: float = 1000.0,
        p_min_bar: float = 0.95,
        p_max_bar: float = 1.05,
        initial_fill_fraction: float = 0.1,
        name: Optional[str] = None,
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

        # control setpoint (pressure in bar) - None means no active setpoint
        self.pressure_setpoint_bar: Optional[float] = None

        # initialize default state
        self.initialize()

    def initialize(self, initial_state: Optional[Dict[str, Any]] = None) -> None:
        """Initialize storage state; initial_state may contain stored_volume_m3, pressure_setpoint_bar."""
        if initial_state is not None:
            if "stored_volume_m3" in initial_state:
                self.stored_volume_m3 = float(initial_state["stored_volume_m3"])
            if "pressure_setpoint_bar" in initial_state:
                self.pressure_setpoint_bar = initial_state["pressure_setpoint_bar"]

        # clamp stored volume
        self.stored_volume_m3 = max(0.0, min(self.stored_volume_m3, self.capacity_m3))

        self._cum_vented_m3 = 0.0

        # outputs_data template
        self.outputs_data = {
            "stored_volume_m3": self.stored_volume_m3,
            "pressure_bar": self._estimate_pressure_bar(),
            "utilization": self.stored_volume_m3 / max(1e-9, self.capacity_m3),
            "vented_volume_m3": 0.0,
            "Q_gas_supplied_m3_per_day": 0.0,
        }

        self._initialized = True

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
    def step(self, t: float, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        One simulation step.

        Args:
            t:
            dt:
            inputs: Inputs dictionary may contain:
                - 'Q_gas_in_m3_per_day'   : gas inflow from digesters/other sources (m^3/day)
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
        """
        # get flows (units: m^3/day) -> convert to volume for this timestep: m^3 = Q * dt (dt in days)
        Q_in = float(inputs.get("Q_gas_in_m3_per_day", 0.0))
        Q_out_req = float(inputs.get("Q_gas_out_m3_per_day", 0.0))
        allow_vent = bool(inputs.get("vent_to_flare", True))

        print(f"gas storage Q_in {Q_in}, Q_out: {Q_out_req}")

        # pressure setpoint handling (bar) - if provided, store it for internal control
        if "set_pressure" in inputs:
            sp = inputs["set_pressure"]
            try:
                self.pressure_setpoint_bar = None if sp is None else float(sp)
            except Exception as e:
                # ignore invalid setpoint
                print(e)

        # convert to volumes for this dt
        vol_in = Q_in * dt
        vol_out_req = Q_out_req * dt

        vented_this_step = 0.0
        # supplied_this_step = 0.0

        # Inflow: attempt to store incoming gas
        free_capacity_m3 = self.capacity_m3 - self.stored_volume_m3
        if vol_in <= free_capacity_m3 + 1e-12:
            # all incoming stored
            self.stored_volume_m3 += vol_in
            overflow = 0.0
        else:
            # store what fits, overflow is vented (or forwarded externally)
            stored = max(0.0, free_capacity_m3)
            overflow = vol_in - stored
            self.stored_volume_m3 += stored
            if allow_vent:
                vented_this_step += overflow
                self._cum_vented_m3 += overflow
            else:
                # if venting not allowed, assume overflow is rejected (lost) (count as vented for safety)
                vented_this_step += overflow
                self._cum_vented_m3 += overflow

        # Supply (outflow): cannot supply more than stored
        # If pressure below p_min, limit supply (simulate pressure control)
        current_pressure = self._estimate_pressure_bar()

        # If a pressure setpoint is requested and higher than current,
        # we might choose to deny outflow to increase pressure
        restrict_factor = 1.0
        if self.pressure_setpoint_bar is not None:
            # If setpoint is higher than current pressure, prioritize charging (i.e., restrict outflow)
            if self.pressure_setpoint_bar > current_pressure:
                # reduce allowed outflow proportionally
                # factor between 0..1 -> 0 means block outflow, 1 means full
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
                frac_needed = max(0.0, (self.p_min_bar - self.p_atm_bar) / (self.p_max_bar - self.p_atm_bar))
            else:
                frac_needed = 0.0
        else:  # compressed
            # invert nonlinear mapping p = p_min + frac^alpha*(p_max-p_min)
            if (self.p_max_bar - self.p_min_bar) > 1e-9:
                alpha = 2.0
                frac_needed = max(
                    0.0, ((self.p_min_bar - self.p_min_bar) / (self.p_max_bar - self.p_min_bar)) ** (1.0 / alpha)
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

        print("storage tank:", supplied_this_step_m3_per_day, self.stored_volume_m3)

        # After flows, update pressure and check safety
        current_pressure = self._estimate_pressure_bar()

        # Safety: overpressure -> vent to flare
        if current_pressure > self.p_max_bar:
            # compute how much volume must be removed to reach p_max
            # invert our pressure models approximately by reducing stored fraction until p <= p_max
            # perform simple linear backoff: remove fraction proportional to pressure exceedance
            if self.storage_type in ("membrane", "dome"):
                # p = p_atm + frac*(p_max - p_atm), want frac_target = (p_max - p_atm)/(p_max - p_atm) = 1.0
                # but if p_max_bar is absolute safety we want to bring pressure to p_max_bar.
                # compute fraction that yields p_max_bar
                target_frac = max(
                    0.0, min(1.0, (self.p_max_bar - self.p_atm_bar) / max(1e-9, (self.p_max_bar - self.p_atm_bar)))
                )
                # if current fraction > 1.0 (theory), set to 1.0
                target_frac = min(target_frac, 1.0)
                target_volume = target_frac * self.capacity_m3
                if self.stored_volume_m3 > target_volume:
                    vent = self.stored_volume_m3 - target_volume
                else:
                    vent = 0.0
            else:  # compressed
                # reduce to fraction that yields p_max_bar approx equal to 1.0 fraction
                # simpler: vent until stored <= capacity * 0.999 (small safety)
                target_volume = min(self.capacity_m3, self.capacity_m3 * 0.999)
                vent = max(0.0, self.stored_volume_m3 - target_volume)

            # Vent if allowed (or always vent for safety)
            if allow_vent and vent > 0.0:
                vented_this_step += vent
                self.stored_volume_m3 -= vent
                self._cum_vented_m3 += vent

            # recompute pressure
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
            "cumulative_vented_m3": float(self._cum_vented_m3),
            "pressure_setpoint_bar": self.pressure_setpoint_bar,
        }

        return self.outputs_data

    # ------------------------------
    # Serialization
    # ------------------------------
    def to_dict(self) -> Dict[str, Any]:
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
            "pressure_setpoint_bar": self.pressure_setpoint_bar,
            "outputs_data": self.outputs_data,
            "inputs": self.inputs,
            "outputs": self.outputs,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "GasStorage":
        """Create GasStorage from dict produced by to_dict."""
        gs = cls(
            component_id=config["component_id"],
            storage_type=config.get("storage_type", "membrane"),
            capacity_m3=config.get("capacity_m3", 1000.0),
            p_min_bar=config.get("p_min_bar", 0.95),
            p_max_bar=config.get("p_max_bar", 1.05),
            initial_fill_fraction=0.0,
            name=config.get("name"),
        )
        # restore state if present
        if "stored_volume_m3" in config:
            try:
                gs.stored_volume_m3 = float(config["stored_volume_m3"])
            except Exception as e:
                print(e)

        if "pressure_setpoint_bar" in config:
            gs.pressure_setpoint_bar = config.get("pressure_setpoint_bar")

        gs.initialize({"stored_volume_m3": gs.stored_volume_m3, "pressure_setpoint_bar": gs.pressure_setpoint_bar})

        return gs
