# ============================================================================
# pyadm1/components/biological/adm1da_digester.py
# ============================================================================
"""
ADM1da-based digester component.

Mirrors the public interface of :class:`pyadm1.components.biological.Digester`
but wraps :class:`pyadm1.core.adm1da.ADM1da` (41-state SIMBA# biogas model)
instead of the legacy 37-state ADM1.  Interchangeable with the ADM1 Digester
at the component level, so plants can switch biological back-ends without
restructuring the topology.

Key differences from the ADM1 Digester
--------------------------------------
* State vector has 41 elements (vs 37) — indices 0-36 are liquid/charge-
  balance, 37-40 are gas partial pressures.
* Flow Q is at ``_state_input[37]`` (vs index 33 in ADM1).
* The ODE method is ``ADM_ODE`` (vs ``ADM1_ODE`` in ADM1); called directly
  via :func:`scipy.integrate.solve_ivp`, bypassing the ADM1-specific
  :class:`Simulator` wrapper.
* pH, VFA and TAC are computed natively from the ADM1da state vector (no
  C# DLL dependency).

Example
-------
    >>> from pyadm1.components.biological import ADM1daDigester
    >>> from pyadm1.substrates.adm1da_feedstock import ADM1daFeedstock
    >>> fs = ADM1daFeedstock(...)
    >>> dig = ADM1daDigester("dig1", fs, V_liq=1200, V_gas=216, T_ad=315.15)
    >>> dig.initialize()
"""

from typing import Any, Dict, Optional

import numpy as np
from scipy.integrate import solve_ivp

from .digester_base import DigesterBase
from ...core.adm1da import (
    ADM1da,
    STATE_SIZE as _ADM1DA_STATE_SIZE,
    _IDX_P_H2,
    _IDX_P_CH4,
    _IDX_P_CO2,
    _IDX_P_TOTAL,
    _IDX_S_NH4,
    _IDX_S_NH3,
    _IDX_S_HCO3,
    _IDX_S_AC_ION,
    _IDX_S_PRO_ION,
    _IDX_S_BU_ION,
    _IDX_S_VA_ION,
    _IDX_S_CATION,
    _IDX_S_ANION,
    _IDX_S_VA,
    _IDX_S_BU,
    _IDX_S_PRO,
    _IDX_S_AC,
)
from ...core.adm1da_params import ADM1daParams

# Q is the last column of the ADM1da influent vector (index 37)
_Q_IDX_ADM1DA = 37


class ADM1daDigester(DigesterBase):
    """
    Digester component wrapping the ADM1da (SIMBA# biogas) model.

    Public API is intentionally identical to
    :class:`pyadm1.components.biological.Digester` so the two classes are
    drop-in interchangeable at the plant-composition level.  Both derive
    from :class:`DigesterBase`, which supplies the shared geometry/storage
    scaffolding.

    Parameters
    ----------
    component_id : str
    feedstock : ADM1daFeedstock or Feedstock or None
        Feedstock used to derive the influent when ``create_influent`` is
        called without a pre-set DataFrame.  Pass ``None`` if you intend to
        drive the digester exclusively via ``set_influent_dataframe``.
    V_liq, V_gas, T_ad : float
        Reactor liquid volume [m³], gas headspace [m³] and temperature [K].
    name : str, optional
    """

    def __init__(
        self,
        component_id: str,
        feedstock,
        V_liq: float = 1977.0,
        V_gas: float = 304.0,
        T_ad: float = 308.15,
        name: Optional[str] = None,
    ):
        super().__init__(component_id, feedstock, V_liq, V_gas, T_ad, name)

        self.adm1 = ADM1da(feedstock=feedstock, V_liq=V_liq, V_gas=V_gas, T_ad=T_ad)

    # ------------------------------------------------------------------
    # Component lifecycle
    # ------------------------------------------------------------------

    def initialize(self, initial_state: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the digester state and attached gas storage.

        Keys accepted in *initial_state* mirror :class:`Digester`:

        * ``adm1_state`` : list of 41 floats (ADM1da state vector)
        * ``Q_substrates`` : list of substrate feed rates [m³/d]
        * ``gas_storage`` : dict forwarded to the attached gas storage
        """
        if initial_state is None:
            self.adm1_state = [0.01] * _ADM1DA_STATE_SIZE
        else:
            if "adm1_state" in initial_state:
                self.adm1_state = list(initial_state["adm1_state"])
            else:
                self.adm1_state = [0.01] * _ADM1DA_STATE_SIZE
            if "Q_substrates" in initial_state:
                self.Q_substrates = list(initial_state["Q_substrates"])

        self.state = {
            "adm1_state": self.adm1_state,
            "Q_substrates": self.Q_substrates,
            "Q_gas": 0.0,
            "Q_ch4": 0.0,
            "Q_co2": 0.0,
            "pH": 7.0,
            "VFA": 0.0,
            "TAC": 0.0,
        }

        gs_state = initial_state.get("gas_storage") if initial_state else None
        try:
            self.gas_storage.initialize(gs_state)
        except Exception:
            self.gas_storage.initialize()

        self._initialized = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _has_valid_state_input(self) -> bool:
        state_input = getattr(self.adm1, "_state_input", None)
        if state_input is None:
            return False
        try:
            float(state_input[_Q_IDX_ADM1DA])
        except (TypeError, ValueError, IndexError):
            return False
        return True

    def _compute_indicators(self) -> Dict[str, float]:
        """Compute pH and VFA directly from the ADM1da state (no DLL)."""
        st = self.adm1_state
        inhib = ADM1daParams.get_inhibition_params(self.adm1._R, self.adm1._T_base, self.adm1._T_ad)
        try:
            S_H = ADM1da._calc_ph(
                st[_IDX_S_NH4],
                st[_IDX_S_NH3],
                st[_IDX_S_HCO3],
                st[_IDX_S_AC_ION],
                st[_IDX_S_PRO_ION],
                st[_IDX_S_BU_ION],
                st[_IDX_S_VA_ION],
                st[_IDX_S_CATION],
                st[_IDX_S_ANION],
                inhib["K_w"],
            )
            pH = -np.log10(max(float(S_H), 1.0e-14))
        except Exception:
            pH = 7.0

        # VFA as total volatile fatty acids expressed in g HAc-eq/L
        # (sum of COD-based acids divided by their COD factors, then × M_HAc)
        # Simple COD-weighted approximation: just report sum(S_va..S_ac) in g COD/L
        vfa = float(st[_IDX_S_VA] + st[_IDX_S_BU] + st[_IDX_S_PRO] + st[_IDX_S_AC])

        # TAC not natively available in ADM1da; report HCO3⁻ as carbonate alk proxy
        # (kmol/m³ → g CaCO3/L: ×100 g/mol → ×100 kg/m³ → ×0.1 g/L equivalent)
        tac = float(st[_IDX_S_HCO3]) * 100.0  # rough proxy, g CaCO3/L
        return {"pH": pH, "VFA": vfa, "TAC": tac}

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, t: float, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Integrate the ADM1da ODE by *dt* days and return outputs.

        Accepts the same inputs as :class:`Digester.step` and returns the
        same output schema (``Q_out``, ``state_out``, ``Q_gas``, ``Q_ch4``,
        ``Q_co2``, ``pH``, ``VFA``, ``TAC``, ``gas_storage``).
        """
        if "Q_substrates" in inputs:
            self.Q_substrates = inputs["Q_substrates"]

        if "state_in" in inputs and "Q_in" in inputs:
            Q_in = float(inputs["Q_in"])
            state_in = inputs["state_in"]

            self.adm1.create_influent(self.Q_substrates, int(t / dt))
            Q_sub = float(self.adm1._state_input[_Q_IDX_ADM1DA])
            Q_total = Q_in + Q_sub

            if Q_total > 0 and len(state_in) >= _Q_IDX_ADM1DA:
                for idx in range(_Q_IDX_ADM1DA):
                    c_sub = float(self.adm1._state_input[idx])
                    c_in = float(state_in[idx])
                    self.adm1._state_input[idx] = (c_sub * Q_sub + c_in * Q_in) / Q_total
                self.adm1._state_input[_Q_IDX_ADM1DA] = Q_total
        else:
            self.adm1.create_influent(self.Q_substrates, int(t / dt))

        # Integrate over [t, t+dt]
        result = solve_ivp(
            fun=self.adm1.ADM_ODE,
            t_span=(t, t + dt),
            y0=self.adm1_state,
            method="BDF",
            rtol=1.0e-6,
            atol=1.0e-8,
            max_step=max(0.1 * dt, 1.0e-3),
        )
        if not result.success:
            raise RuntimeError(f"ADM1da integration failed in '{self.component_id}': " f"{result.message}")
        self.adm1_state = list(result.y[:, -1])

        # Gas production from partial pressures at the tail of the state vector
        q_gas, q_ch4, q_co2, _, _ = self.adm1.calc_gas(
            self.adm1_state[_IDX_P_H2],
            self.adm1_state[_IDX_P_CH4],
            self.adm1_state[_IDX_P_CO2],
            self.adm1_state[_IDX_P_TOTAL],
        )

        indicators = self._compute_indicators()
        self.state["adm1_state"] = self.adm1_state
        self.state["Q_gas"] = q_gas
        self.state["Q_ch4"] = q_ch4
        self.state["Q_co2"] = q_co2
        self.state.update(indicators)

        Q_out = (
            float(self.adm1._state_input[_Q_IDX_ADM1DA]) if self._has_valid_state_input() else float(np.sum(self.Q_substrates))
        )

        gs_outputs = self.gas_storage.step(
            t=t,
            dt=dt,
            inputs={
                "Q_gas_in_m3_per_day": float(q_gas),
                "Q_gas_out_m3_per_day": 0.0,
                "vent_to_flare": True,
            },
        )
        self.gas_storage.outputs_data["Q_gas_in_m3_per_day"] = float(q_gas)

        self.outputs_data = {
            "Q_out": Q_out,
            "state_out": self.adm1_state,
            "Q_gas": float(q_gas),
            "Q_ch4": float(q_ch4),
            "Q_co2": float(q_co2),
            "pH": indicators["pH"],
            "VFA": indicators["VFA"],
            "TAC": indicators["TAC"],
            "Q_gas_to_storage_m3_per_day": float(q_gas),
            "gas_storage": {
                "component_id": self.gas_storage.component_id,
                "stored_volume_m3": gs_outputs["stored_volume_m3"],
                "pressure_bar": gs_outputs["pressure_bar"],
                "vented_volume_m3": gs_outputs["vented_volume_m3"],
                "Q_gas_supplied_m3_per_day": gs_outputs["Q_gas_supplied_m3_per_day"],
            },
        }
        return self.outputs_data

    @classmethod
    def from_dict(cls, config: Dict[str, Any], feedstock) -> "ADM1daDigester":
        digester = cls(
            component_id=config["component_id"],
            feedstock=feedstock,
            V_liq=config.get("V_liq", 1977.0),
            V_gas=config.get("V_gas", 304.0),
            T_ad=config.get("T_ad", 308.15),
            name=config.get("name"),
        )
        digester.inputs = config.get("inputs", [])
        digester.outputs = config.get("outputs", [])
        if "state" in config:
            digester.initialize(config["state"])
        return digester
