# ============================================================================
# pyadm1/components/biological/hydrolysis.py
# ============================================================================
"""
Hydrolysis pre-treatment tank component for two-stage biogas plants.

The hydrolysis stage is the first reactor in a two-stage anaerobic digestion
system.  It operates at thermophilic conditions (default 55 °C) with a short
hydraulic retention time (HRT ≈ 2–5 days).  The primary processes are:

    1. Disintegration and hydrolysis of complex organic polymers (Xc, Xch, Xpr, Xli)
    2. Acidogenesis: conversion of monomers to short-chain VFAs, H₂ and CO₂
    3. Some acetogenesis; methanogenesis is kinetically suppressed at short HRT

The full ADM1 ODE system is used, identical to the Digester component.
The thermophilic temperature and short HRT naturally suppress methanogens
because their maximum growth rate is slower than the wash-out rate.

Output stream keys are compatible with the Digester component so that the
effluent can be passed directly via `state_in` / `Q_in` inputs to a
downstream Digester.

References:
    - Batstone et al. (2002): ADM1, IWA Publishing
    - Karakashev et al. (2005): Influence of temperature on methanogenesis,
      Appl Environ Microbiol 71(12), pp. 7171–7177
    - Boone & Mah (1987): Methanobacterium, Bergey's Manual

Example:
    >>> from pyadm1.components.biological import Hydrolysis, Digester
    >>> from pyadm1.substrates import Feedstock
    >>>
    >>> feedstock = Feedstock(feeding_freq=48)
    >>>
    >>> # Stage 1: hydrolysis at 55 °C, 3-day HRT
    >>> hydro = Hydrolysis("hydro1", feedstock, V_liq=500, V_gas=50, T_ad=328.15)
    >>> hydro.initialize({"adm1_state": initial_state,
    ...                   "Q_substrates": [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]})
    >>>
    >>> # Stage 2: methanogenesis at 35 °C
    >>> digester = Digester("dig1", feedstock, V_liq=2000, V_gas=304, T_ad=308.15)
    >>> digester.initialize({"adm1_state": initial_state_2})
    >>>
    >>> # Simulation step: chain stages
    >>> hydro_out = hydro.step(t, dt, inputs={})
    >>> dig_out   = digester.step(t, dt, inputs={
    ...     "Q_in":    hydro_out["Q_out"],
    ...     "state_in": hydro_out["state_out"],
    ... })
"""


def _try_load_clr():
    import platform

    if platform.system() == "Darwin":
        return None
    try:
        import clr

        return clr
    except Exception as e:
        print(e)
        return None


clr = _try_load_clr()

import os  # noqa: E402
from typing import Dict, Any, List, Optional  # noqa: E402
import numpy as np  # noqa: E402

from ..base import Component, ComponentType  # noqa: E402
from ...core import ADM1  # noqa: E402
from ...substrates import Feedstock  # noqa: E402
from ...simulation import Simulator  # noqa: E402
from ..energy import GasStorage  # noqa: E402

if clr is None:
    raise RuntimeError("CLR features unavailable on this platform")
else:
    dll_path = os.path.join(os.path.dirname(__file__), "..", "..", "dlls")
    clr.AddReference(os.path.join(dll_path, "plant"))
    from biogas import ADMstate  # noqa: E402  # type: ignore


class Hydrolysis(Component):
    """
    Hydrolysis pre-treatment tank for two-stage anaerobic digestion.

    Wraps the full ADM1 model and operates at thermophilic conditions with a
    short HRT so that hydrolysis and acidogenesis dominate over methanogenesis.
    The component is connectable in series with a downstream Digester via the
    standard ``state_in`` / ``Q_in`` input keys.

    Attributes:
        feedstock (Feedstock): Substrate management and ADM1 input stream.
        V_liq (float):         Liquid volume [m³].
        V_gas (float):         Gas headspace volume [m³].
        T_ad (float):          Operating temperature [K].
        adm1 (ADM1):           ADM1 model instance.
        simulator (Simulator): ODE integrator for ADM1.
        adm1_state (List[float]): Current ADM1 state vector (37 elements).
        Q_substrates (List[float]): Substrate feed rates [m³/d].
        gas_storage (GasStorage): Attached low-pressure gas storage.

    Example:
        >>> feedstock = Feedstock(feeding_freq=48)
        >>> hydro = Hydrolysis("hydro1", feedstock, V_liq=500, T_ad=328.15)
        >>> hydro.initialize({"adm1_state": state0,
        ...                   "Q_substrates": [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]})
    """

    def __init__(
        self,
        component_id: str,
        feedstock: Feedstock,
        V_liq: float = 500.0,
        V_gas: float = 75.0,
        T_ad: float = 328.15,  # 55 °C — typical thermophilic hydrolysis
        name: Optional[str] = None,
    ):
        """
        Initialize hydrolysis tank.

        Args:
            component_id: Unique identifier.
            feedstock:    Feedstock object for substrate management.
            V_liq:        Liquid volume [m³]. Default 500.
            V_gas:        Gas headspace volume [m³]. Default 75
                          (smaller than a methanogenic digester because less
                          biogas is expected from the hydrolysis stage).
            T_ad:         Operating temperature [K]. Default 328.15 (55 °C).
            name:         Optional human-readable name.
        """
        super().__init__(component_id, ComponentType.DIGESTER, name)

        self.feedstock = feedstock
        self.V_liq = V_liq
        self.V_gas = V_gas
        self.T_ad = T_ad

        # ADM1 model
        self.adm1 = ADM1(feedstock)
        self.adm1.V_liq = V_liq
        self.adm1._V_gas = V_gas
        self.adm1._T_ad = T_ad

        self.simulator = Simulator(self.adm1)

        # Low-pressure gas storage sized to the gas headspace
        storage_id = f"{self.component_id}_storage"
        self.gas_storage: GasStorage = GasStorage(
            component_id=storage_id,
            storage_type="membrane",
            capacity_m3=max(50.0, float(self.V_gas)),
            p_min_bar=0.95,
            p_max_bar=1.05,
            initial_fill_fraction=0.1,
            name=f"{self.name} Gas Storage",
        )

        # ADM1 state vector (37 dimensions)
        self.adm1_state: List[float] = []

        # Substrate feed rates [m³/d] — 10 slots matching substrate XML
        self.Q_substrates: List[float] = [0] * 10

    # ------------------------------------------------------------------
    # Component interface
    # ------------------------------------------------------------------

    def initialize(self, initial_state: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize hydrolysis tank state.

        Args:
            initial_state: Optional dict with keys:
                - ``adm1_state``:   ADM1 state vector (37 elements).
                - ``Q_substrates``: Substrate feed rates [m³/d] (10 elements).
                - ``gas_storage``:  Dict to restore gas storage state.
        """
        if initial_state is None:
            self.adm1_state = [0.01] * 37
        else:
            if "adm1_state" in initial_state:
                self.adm1_state = initial_state["adm1_state"]
            if "Q_substrates" in initial_state:
                self.Q_substrates = initial_state["Q_substrates"]

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

        try:
            gs_state = None
            if initial_state and "gas_storage" in initial_state:
                gs_state = initial_state["gas_storage"]
            self.gas_storage.initialize(gs_state)
        except Exception as e:
            print(e)
            self.gas_storage.initialize()

        self._initialized = True

    def step(self, t: float, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform one simulation time step.

        Input handling is identical to ``Digester.step()`` so the two
        components can be chained transparently.

        Args:
            t:      Current simulation time [days].
            dt:     Time step [days].
            inputs: Dict with optional keys:
                - ``Q_substrates``: Fresh substrate feed rates [m³/d].
                - ``Q_in``:         Effluent flow from an upstream component [m³/d].
                - ``state_in``:     ADM1 state vector from upstream component.

        Returns:
            Dict with keys:
                - ``Q_out``:                   Effluent flow [m³/d].
                - ``state_out``:               ADM1 state vector (37 elements).
                - ``Q_gas``:                   Biogas production [m³/d].
                - ``Q_ch4``:                   Methane fraction [m³/d].
                - ``Q_co2``:                   CO₂ fraction [m³/d].
                - ``pH``:                      pH of effluent.
                - ``VFA``:                     VFA concentration [g HAc eq/L].
                - ``TAC``:                     Total alkalinity [g CaCO₃ eq/L].
                - ``Q_gas_to_storage_m3_per_day``: Biogas sent to storage [m³/d].
                - ``gas_storage``:             Dict with gas storage diagnostics.
        """
        # Override substrate feed rates if provided
        if "Q_substrates" in inputs:
            self.Q_substrates = inputs["Q_substrates"]

        # Build ADM1 influent — merge with upstream effluent when connected
        if "state_in" in inputs and "Q_in" in inputs:
            Q_in = float(inputs["Q_in"])
            state_in = inputs["state_in"]

            # Fresh substrate influent (populates adm1._state_input)
            self.adm1.create_influent(self.Q_substrates, int(t / dt))
            Q_sub = float(self.adm1._state_input[33])
            Q_total = Q_in + Q_sub

            if Q_total > 0:
                # Flow-weighted mixing of liquid-phase components (indices 0–32)
                for idx in range(33):
                    c_sub = self.adm1._state_input[idx]
                    c_in = float(state_in[idx])
                    self.adm1._state_input[idx] = (c_sub * Q_sub + c_in * Q_in) / Q_total

                self.adm1._state_input[33] = Q_total
        else:
            # No upstream stage: standard fresh substrate influent
            self.adm1.create_influent(self.Q_substrates, int(t / dt))

        # Integrate ADM1 ODE
        t_span = [t, t + dt]
        self.adm1_state = self.simulator.simulate_AD_plant(t_span, self.adm1_state)

        # Gas production
        q_gas, q_ch4, q_co2, _, _ = self.adm1.calc_gas(
            self.adm1_state[33],  # pi_Sh2
            self.adm1_state[34],  # pi_Sch4
            self.adm1_state[35],  # pi_Sco2
            self.adm1_state[36],  # pTOTAL
        )

        # Update internal state
        self.state["adm1_state"] = self.adm1_state
        self.state["Q_gas"] = q_gas
        self.state["Q_ch4"] = q_ch4
        self.state["Q_co2"] = q_co2

        # Process indicators via DLL
        try:
            self.state["pH"] = ADMstate.calcPHOfADMstate(self.adm1_state)
            self.state["VFA"] = ADMstate.calcVFAOfADMstate(self.adm1_state, "gHAceq/l").Value
            self.state["TAC"] = ADMstate.calcTACOfADMstate(self.adm1_state, "gCaCO3eq/l").Value
        except Exception as e:
            print(f"Warning: Could not calculate process indicators: {e}")

        # Total effluent flow
        Q_out = float(self.adm1._state_input[33]) if self.adm1._state_input is not None else np.sum(self.Q_substrates)

        # Route gas production to attached storage
        gs_outputs = self.gas_storage.step(
            t=t,
            dt=dt,
            inputs={
                "Q_gas_in_m3_per_day": q_gas,
                "Q_gas_out_m3_per_day": 0.0,
                "vent_to_flare": True,
            },
        )
        self.gas_storage.outputs_data["Q_gas_in_m3_per_day"] = q_gas

        self.outputs_data = {
            "Q_out": Q_out,
            "state_out": self.adm1_state,
            "Q_gas": q_gas,
            "Q_ch4": q_ch4,
            "Q_co2": q_co2,
            "pH": self.state.get("pH", 7.0),
            "VFA": self.state.get("VFA", 0.0),
            "TAC": self.state.get("TAC", 0.0),
            "Q_gas_to_storage_m3_per_day": q_gas,
            "gas_storage": {
                "component_id": self.gas_storage.component_id,
                "stored_volume_m3": gs_outputs["stored_volume_m3"],
                "pressure_bar": gs_outputs["pressure_bar"],
                "vented_volume_m3": gs_outputs["vented_volume_m3"],
                "Q_gas_supplied_m3_per_day": gs_outputs["Q_gas_supplied_m3_per_day"],
            },
        }

        return self.outputs_data

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration and current state to dictionary."""
        return {
            "component_id": self.component_id,
            "component_type": self.component_type.value,
            "name": self.name,
            "V_liq": self.V_liq,
            "V_gas": self.V_gas,
            "T_ad": self.T_ad,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "state": self.state,
        }

    @classmethod
    def from_dict(cls, config: Dict[str, Any], feedstock: Feedstock) -> "Hydrolysis":
        """
        Create Hydrolysis from dictionary.

        Args:
            config:    Configuration dict (produced by ``to_dict()``).
            feedstock: Feedstock object.

        Returns:
            Initialized Hydrolysis instance.
        """
        hydro = cls(
            component_id=config["component_id"],
            feedstock=feedstock,
            V_liq=config.get("V_liq", 500.0),
            V_gas=config.get("V_gas", 75.0),
            T_ad=config.get("T_ad", 328.15),
            name=config.get("name"),
        )

        hydro.inputs = config.get("inputs", [])
        hydro.outputs = config.get("outputs", [])

        if "state" in config:
            hydro.initialize(config["state"])

        return hydro
