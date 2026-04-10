# ============================================================================
# pyadm1/components/biological/separator.py
# ============================================================================
"""
Solid-liquid separator component for digestate processing.

Models mechanical separation of digestate into a solid (press cake) and
liquid fraction using a steady-state mass balance.  Four separator types
are supported:

    screw_press       - most common in agricultural biogas plants
    decanter          - higher separation efficiency, higher energy demand
    belt_press        - medium efficiency, low energy
    vibrating_screen  - coarse screening, low TS in solid fraction

Mass balance (per unit time):
    m_TS_in  = Q_in * TS_in                        [kg/d]
    m_TS_sol = m_TS_in * separation_efficiency      [kg/d]
    Q_solid  = m_TS_sol / (TS_solid_target * rho_solid)  [m3/d]
    Q_liquid = Q_in - Q_solid                       [m3/d]

Nutrient partitioning (fractions to solid phase, literature values):
    Nitrogen (N):    20-50 % depending on type
    Phosphorus (P):  60-90 % (P is mainly particulate)

If an ADM1 state vector (state_out from Digester) is provided as input,
total solids are estimated from particulate COD components (indices 12-24):
    TS [kg/m3] = sum(X_i [kg COD/m3]) / COD_VS_ratio / VS_TS_ratio

References:
    - KTBL (2013): Faustzahlen Biogas, 3rd ed., KTBL-Heft 469
    - Hjorth et al. (2010): Solid-liquid separation of animal slurry,
      Bioresource Technology 101, pp. 10–23
    - Möller & Müller (2012): Substrates for anaerobic digestion,
      Bioresource Technology 113, pp. 78–89
    - Battista et al. (2015): Digestate management strategies,
      Renewable and Sustainable Energy Reviews

Example:
    >>> from pyadm1.components.biological import Separator
    >>>
    >>> sep = Separator("sep1", separator_type="screw_press",
    ...                 separation_efficiency=0.60)
    >>> sep.initialize()
    >>> result = sep.step(t=0, dt=1.0,
    ...     inputs={"Q_in": 25.0, "TS_in": 40.0})
    >>> print(f"Liquid: {result['Q_liquid']:.1f} m3/d, "
    ...       f"Solid: {result['Q_solid']:.2f} m3/d")
"""

from typing import Dict, Any, Optional
from enum import Enum

from ..base import Component, ComponentType


class SeparatorType(str, Enum):
    """Enumeration of separator types."""

    SCREW_PRESS = "screw_press"
    DECANTER = "decanter"
    BELT_PRESS = "belt_press"
    VIBRATING_SCREEN = "vibrating_screen"


# ---------------------------------------------------------------------------
# Type-specific default parameters
# Source: KTBL (2013), Hjorth et al. (2010)
# ---------------------------------------------------------------------------
_SEPARATOR_DEFAULTS: Dict[str, Dict[str, float]] = {
    # separation_efficiency: fraction of total solids captured in solid phase
    # ts_solid_target:       target TS content of solid fraction [kg/m3]
    #                        (= TS% * density_solid / 100, density ~900 kg/m3)
    # n_to_solid:            fraction of total nitrogen going to solid phase
    # p_to_solid:            fraction of total phosphorus going to solid phase
    # specific_energy:       energy demand [kWh/t fresh mass input]  KTBL 2013
    "screw_press": {
        "separation_efficiency": 0.60,
        "ts_solid_target": 230.0,  # ~25% TS at 920 kg/m3 solid density
        "n_to_solid": 0.35,
        "p_to_solid": 0.75,
        "specific_energy": 2.0,  # kWh/t FM
    },
    "decanter": {
        "separation_efficiency": 0.75,
        "ts_solid_target": 290.0,  # ~32% TS
        "n_to_solid": 0.45,
        "p_to_solid": 0.85,
        "specific_energy": 5.0,  # kWh/t FM
    },
    "belt_press": {
        "separation_efficiency": 0.50,
        "ts_solid_target": 180.0,  # ~20% TS
        "n_to_solid": 0.25,
        "p_to_solid": 0.65,
        "specific_energy": 3.0,  # kWh/t FM
    },
    "vibrating_screen": {
        "separation_efficiency": 0.35,
        "ts_solid_target": 110.0,  # ~12% TS
        "n_to_solid": 0.15,
        "p_to_solid": 0.55,
        "specific_energy": 1.0,  # kWh/t FM
    },
}

# ADM1 state indices of particulate (non-soluble) components [kg COD/m3]
# Indices 12-24 of the 37-element ADM1 state vector (Batstone et al. 2002)
_ADM1_PARTICULATE_INDICES = list(range(12, 25))

# Conversion: kg COD -> kg TS
# VS/COD = 1/1.42 (standard ADM1 biomass factor)
# TS/VS  = 1/0.80 (typical digestate, Möller & Müller 2012)
_COD_TO_TS = 1.0 / 1.42 / 0.80  # kg TS per kg COD

# Default digestate fluid density [kg/m3]
_DIGESTATE_DENSITY = 1020.0


class Separator(Component):
    """
    Solid-liquid separator for digestate processing.

    Splits the effluent flow from a digester into a solid fraction (press
    cake) and a liquid fraction (separated effluent) using a mass-balance
    model.  Nutrient (N, P) partitioning between fractions is included.

    Attributes:
        separator_type:       Mechanical type (SeparatorType enum).
        separation_efficiency: Fraction of total solids going to solid phase (0-1).
        ts_solid_target:      Target TS concentration in solid fraction [kg/m3].
        n_to_solid:           Fraction of input nitrogen transferred to solid (0-1).
        p_to_solid:           Fraction of input phosphorus transferred to solid (0-1).
        specific_energy:      Power demand per tonne of fresh-mass input [kWh/t FM].
        fluid_density:        Digestate density [kg/m3].
    """

    def __init__(
        self,
        component_id: str,
        separator_type: str = "screw_press",
        separation_efficiency: Optional[float] = None,
        ts_solid_target: Optional[float] = None,
        n_to_solid: Optional[float] = None,
        p_to_solid: Optional[float] = None,
        specific_energy: Optional[float] = None,
        fluid_density: float = _DIGESTATE_DENSITY,
        name: Optional[str] = None,
    ):
        """
        Initialize separator component.

        Args:
            component_id:         Unique identifier.
            separator_type:       One of "screw_press", "decanter",
                                  "belt_press", "vibrating_screen".
            separation_efficiency: Override default solid-capture efficiency (0-1).
            ts_solid_target:      Override default solid-fraction TS [kg/m3].
            n_to_solid:           Override default N fraction to solid (0-1).
            p_to_solid:           Override default P fraction to solid (0-1).
            specific_energy:      Override default power demand [kWh/t FM].
            fluid_density:        Digestate density [kg/m3]. Default 1020.
            name:                 Human-readable display name.
        """
        super().__init__(component_id, ComponentType.SEPARATOR, name)

        self.separator_type = SeparatorType(separator_type.lower())
        defaults = _SEPARATOR_DEFAULTS[self.separator_type.value]

        self.separation_efficiency = (
            separation_efficiency if separation_efficiency is not None else defaults["separation_efficiency"]
        )
        self.ts_solid_target = ts_solid_target if ts_solid_target is not None else defaults["ts_solid_target"]
        self.n_to_solid = n_to_solid if n_to_solid is not None else defaults["n_to_solid"]
        self.p_to_solid = p_to_solid if p_to_solid is not None else defaults["p_to_solid"]
        self.specific_energy = specific_energy if specific_energy is not None else defaults["specific_energy"]
        self.fluid_density = float(fluid_density)

        # Cumulative tracking
        self.total_solid_mass = 0.0  # kg
        self.total_liquid_vol = 0.0  # m3
        self.energy_consumed = 0.0  # kWh

        self.initialize()

    # ------------------------------------------------------------------
    # Component interface
    # ------------------------------------------------------------------

    def initialize(self, initial_state: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize separator state.

        Args:
            initial_state: Optional dict with keys:
                - 'total_solid_mass': cumulative solid output [kg]
                - 'total_liquid_vol': cumulative liquid output [m3]
                - 'energy_consumed':  cumulative energy use [kWh]
        """
        if initial_state:
            self.total_solid_mass = float(initial_state.get("total_solid_mass", 0.0))
            self.total_liquid_vol = float(initial_state.get("total_liquid_vol", 0.0))
            self.energy_consumed = float(initial_state.get("energy_consumed", 0.0))

        self.state = {
            "total_solid_mass": self.total_solid_mass,
            "total_liquid_vol": self.total_liquid_vol,
            "energy_consumed": self.energy_consumed,
        }

        self.outputs_data = {
            "Q_liquid": 0.0,
            "Q_solid": 0.0,
            "TS_liquid": 0.0,
            "TS_solid": self.ts_solid_target,
            "VS_liquid": 0.0,
            "VS_solid": 0.0,
            "TAN_liquid": 0.0,
            "TAN_solid": 0.0,
            "TP_liquid": 0.0,
            "TP_solid": 0.0,
            "P_consumed": 0.0,
            "separation_efficiency": self.separation_efficiency,
        }

        self._initialized = True

    def step(self, t: float, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform one simulation time step.

        Args:
            t:   Current simulation time [days].
            dt:  Time step [days].
            inputs: Dict with keys:
                - 'Q_in'    [m3/d]    Influent flow (required, or from Q_out
                                       of connected Digester).
                - 'Q_out'   [m3/d]    Alias for Q_in (Digester output key).
                - 'TS_in'   [kg/m3]   Total solids concentration in influent.
                                       If omitted, estimated from ADM1 state.
                - 'VS_in'   [kg/m3]   Volatile solids (optional, default 0.85*TS).
                - 'TAN_in'  [kg/m3]   Total ammonium nitrogen (optional).
                - 'TP_in'   [kg/m3]   Total phosphorus (optional).
                - 'state_out' [list]  ADM1 state vector from upstream Digester.
                                       Used to estimate TS_in if not given.

        Returns:
            Dict with keys:
                - 'Q_liquid'   [m3/d]  Liquid fraction flow rate.
                - 'Q_solid'    [m3/d]  Solid fraction flow rate.
                - 'TS_liquid'  [kg/m3] TS in liquid fraction.
                - 'TS_solid'   [kg/m3] TS in solid fraction (= ts_solid_target).
                - 'VS_liquid'  [kg/m3] VS in liquid fraction.
                - 'VS_solid'   [kg/m3] VS in solid fraction.
                - 'TAN_liquid' [kg/m3] TAN in liquid fraction.
                - 'TAN_solid'  [kg/m3] TAN in solid fraction.
                - 'TP_liquid'  [kg/m3] Total P in liquid fraction.
                - 'TP_solid'   [kg/m3] Total P in solid fraction.
                - 'P_consumed' [kW]    Electrical power draw.
                - 'separation_efficiency' [-] Active efficiency value.
        """
        # --- resolve influent flow -----------------------------------------
        Q_in = float(inputs.get("Q_in", inputs.get("Q_out", 0.0)))

        if Q_in <= 0.0:
            return self.outputs_data  # no flow, no separation

        # --- resolve total solids ------------------------------------------
        TS_in = float(inputs.get("TS_in", 0.0))

        if TS_in <= 0.0:
            # Estimate from ADM1 state vector if available
            adm1_state = inputs.get("state_out")
            if adm1_state is not None:
                TS_in = self._estimate_ts_from_adm1(adm1_state)
            else:
                # Fallback: typical mesophilic digestate TS ~40 kg/m3 (4%)
                TS_in = 40.0

        VS_in = float(inputs.get("VS_in", TS_in * 0.75))  # ~75% VS/TS
        TAN_in = float(inputs.get("TAN_in", 0.0))
        TP_in = float(inputs.get("TP_in", 0.0))

        # --- mass balance --------------------------------------------------
        m_TS_total = Q_in * TS_in  # kg/d total solids in
        m_VS_total = Q_in * VS_in  # kg/d VS in

        # VS separation efficiency slightly higher than TS (VS is more
        # particulate than mineral ash fraction); use 1.1x factor, capped at 0.95
        vs_sep_eff = min(self.separation_efficiency * 1.1, 0.95)

        m_TS_solid = m_TS_total * self.separation_efficiency  # kg/d to solid
        m_VS_solid = m_VS_total * vs_sep_eff  # kg/d VS to solid

        m_TS_liquid = m_TS_total - m_TS_solid  # kg/d to liquid
        m_VS_liquid = m_VS_total - m_VS_solid  # kg/d VS to liquid

        # Volume of solid fraction: m_TS_solid = Q_solid * ts_solid_target
        Q_solid = m_TS_solid / max(self.ts_solid_target, 1.0)  # m3/d
        Q_liquid = max(Q_in - Q_solid, 0.0)  # m3/d

        # Concentrations in each fraction
        TS_liquid = m_TS_liquid / max(Q_liquid, 1e-9)  # kg/m3
        VS_liquid = m_VS_liquid / max(Q_liquid, 1e-9)  # kg/m3
        VS_solid = m_VS_solid / max(Q_solid, 1e-9)  # kg/m3

        # Nutrient partitioning
        TAN_solid_total = TAN_in * Q_in * self.n_to_solid  # kg/d
        TAN_liquid_total = TAN_in * Q_in * (1.0 - self.n_to_solid)

        TP_solid_total = TP_in * Q_in * self.p_to_solid  # kg/d
        TP_liquid_total = TP_in * Q_in * (1.0 - self.p_to_solid)

        TAN_solid_conc = TAN_solid_total / max(Q_solid, 1e-9)  # kg/m3
        TAN_liquid_conc = TAN_liquid_total / max(Q_liquid, 1e-9)  # kg/m3
        TP_solid_conc = TP_solid_total / max(Q_solid, 1e-9)  # kg/m3
        TP_liquid_conc = TP_liquid_total / max(Q_liquid, 1e-9)  # kg/m3

        # --- power consumption --------------------------------------------
        # P [kW] = specific_energy [kWh/t] * FM_rate [t/d] / 24 [h/d]
        FM_rate_t_per_day = Q_in * self.fluid_density / 1000.0  # t FM/d
        P_consumed = self.specific_energy * FM_rate_t_per_day / 24.0  # kW

        # --- cumulative accounting ----------------------------------------
        dt_hours = dt * 24.0
        self.total_solid_mass += m_TS_solid * dt  # kg
        self.total_liquid_vol += Q_liquid * dt  # m3
        self.energy_consumed += P_consumed * dt_hours  # kWh

        # --- update state and outputs ------------------------------------
        self.state.update(
            {
                "total_solid_mass": self.total_solid_mass,
                "total_liquid_vol": self.total_liquid_vol,
                "energy_consumed": self.energy_consumed,
            }
        )

        self.outputs_data = {
            "Q_liquid": float(Q_liquid),
            "Q_solid": float(Q_solid),
            "TS_liquid": float(TS_liquid),
            "TS_solid": float(self.ts_solid_target),
            "VS_liquid": float(VS_liquid),
            "VS_solid": float(VS_solid),
            "TAN_liquid": float(TAN_liquid_conc),
            "TAN_solid": float(TAN_solid_conc),
            "TP_liquid": float(TP_liquid_conc),
            "TP_solid": float(TP_solid_conc),
            "P_consumed": float(P_consumed),
            "separation_efficiency": float(self.separation_efficiency),
            # Convenience summary
            "solid_fraction_ts_pct": float(self.ts_solid_target / (self.fluid_density * 10.0)),  # % TS
            "recovery_solid_pct": float(self.separation_efficiency * 100.0),
        }

        return self.outputs_data

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize separator configuration to dictionary."""
        return {
            "component_id": self.component_id,
            "component_type": self.component_type.value,
            "name": self.name,
            "separator_type": self.separator_type.value,
            "separation_efficiency": self.separation_efficiency,
            "ts_solid_target": self.ts_solid_target,
            "n_to_solid": self.n_to_solid,
            "p_to_solid": self.p_to_solid,
            "specific_energy": self.specific_energy,
            "fluid_density": self.fluid_density,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "Separator":
        """
        Create separator from dictionary.

        Args:
            config: Configuration dictionary (from to_dict()).

        Returns:
            Initialized Separator instance.
        """
        sep = cls(
            component_id=config["component_id"],
            separator_type=config.get("separator_type", "screw_press"),
            separation_efficiency=config.get("separation_efficiency"),
            ts_solid_target=config.get("ts_solid_target"),
            n_to_solid=config.get("n_to_solid"),
            p_to_solid=config.get("p_to_solid"),
            specific_energy=config.get("specific_energy"),
            fluid_density=config.get("fluid_density", _DIGESTATE_DENSITY),
            name=config.get("name"),
        )

        sep.inputs = config.get("inputs", [])
        sep.outputs = config.get("outputs", [])

        if "state" in config:
            sep.initialize(config["state"])

        return sep

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_ts_from_adm1(state: Any) -> float:
        """
        Estimate digestate TS from ADM1 state vector.

        Sums particulate COD components (indices 12-24, units kg COD/m3)
        and converts to kg TS/m3 using standard stoichiometric factors.

        Args:
            state: Sequence of 37 ADM1 state values [kg COD/m3 or kmol/m3].

        Returns:
            Estimated TS concentration [kg/m3].
        """
        try:
            particulate_cod = sum(float(state[i]) for i in _ADM1_PARTICULATE_INDICES if i < len(state))
            # Convert kg COD/m3 -> kg TS/m3
            return max(particulate_cod * _COD_TO_TS, 0.0)
        except (TypeError, IndexError):
            return 40.0  # fallback: typical digestate 4% TS
