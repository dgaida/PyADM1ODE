# ============================================================================
# pyadm1/components/biological/digester.py
# ============================================================================
"""
Digester component wrapping PyADM1 model.

This module provides the Digester class which encapsulates the ADM1 model
for anaerobic digestion in a component-based framework.
"""


def try_load_clr():
    import platform

    if platform.system() == "Darwin":
        return None
    try:
        import clr

        return clr
    except Exception as e:
        print(e)
        return None


clr = try_load_clr()


import os  # noqa: E402  # type: ignore

from typing import Dict, Any, List, Optional  # noqa: E402  # type: ignore
import numpy as np  # noqa: E402  # type: ignore

from ..base import Component, ComponentType  # noqa: E402  # type: ignore
from ...core import ADM1  # noqa: E402  # type: ignore
from ...substrates import Feedstock  # noqa: E402  # type: ignore
from ...simulation import Simulator  # noqa: E402  # type: ignore
from ..energy import GasStorage  # noqa: E402  # type: ignore

if clr is None:
    raise RuntimeError("CLR features unavailable on this platform")
else:
    # CLR reference must be added before importing from DLL
    dll_path = os.path.join(os.path.dirname(__file__), "..", "..", "dlls")
    clr.AddReference(os.path.join(dll_path, "plant"))
    from biogas import ADMstate  # noqa: E402  # type: ignore


class Digester(Component):
    """
    Digester component using ADM1 model.

    This component wraps the ADM1 implementation and can be
    connected to other digesters or components in series/parallel.

    Attributes:

        feedstock (Feedstock): Feedstock object for substrate management.
        V_liq (float): Liquid volume in m³.
        V_gas (float): Gas volume in m³.
        T_ad (float): Operating temperature in K.
        adm1 (ADM1): ADM1 model instance.
        simulator (Simulator): Simulator for ADM1.
        adm1_state (List[float]): Current ADM1 state vector (37 dimensions).
        Q_substrates (List[float]): Substrate feed rates in m³/d.

    Example:

        >>> feedstock = Feedstock(feeding_freq=48)
        >>> digester = Digester("dig1", feedstock, V_liq=2000, V_gas=300)
        >>> digester.initialize({"adm1_state": initial_state, "Q_substrates": [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]})
    """

    def __init__(
        self,
        component_id: str,
        feedstock: Feedstock,
        V_liq: float = 1977.0,
        V_gas: float = 304.0,
        T_ad: float = 308.15,
        name: Optional[str] = None,
    ):
        """
        Initialize digester component.

        Args:
            component_id (str): Unique identifier.
            feedstock (Feedstock): Feedstock object for substrate management.
            V_liq (float): Liquid volume in m³. Defaults to 1977.0.
            V_gas (float): Gas volume in m³. Defaults to 304.0.
            T_ad (float): Operating temperature in K. Defaults to 308.15 (35°C).
            name (Optional[str]): Human-readable name. Defaults to component_id.
        """
        super().__init__(component_id, ComponentType.DIGESTER, name)

        self.feedstock = feedstock
        self.V_liq = V_liq
        self.V_gas = V_gas
        self.T_ad = T_ad

        # Initialize ADM1
        self.adm1 = ADM1(feedstock)
        self.adm1.V_liq = V_liq
        self.adm1._V_gas = V_gas
        self.adm1._T_ad = T_ad

        self.simulator = Simulator(self.adm1)

        # Gas storage attached to this digester (created per-digester)
        storage_id = f"{self.component_id}_storage"
        # create a low-pressure membrane storage sized roughly proportional to V_gas
        # capacity_m3 is in m³ STP; use V_gas as a baseline
        self.gas_storage: GasStorage = GasStorage(
            component_id=storage_id,
            storage_type="membrane",
            capacity_m3=max(50.0, float(self.V_gas)),  # sensible minimum
            p_min_bar=0.95,
            p_max_bar=1.05,
            initial_fill_fraction=0.1,
            name=f"{self.name} Gas Storage",
        )

        # ADM1 state vector (37 dimensions)
        self.adm1_state: List[float] = []

        # Substrate feed rates [m³/d]
        self.Q_substrates: List[float] = [0] * 10

    def initialize(self, initial_state: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize digester state.

        Args:
            initial_state (Optional[Dict[str, Any]]): Initial state with keys:
                - 'adm1_state': ADM1 state vector (37 dims)
                - 'Q_substrates': Substrate feed rates
                If None, uses default initialization.
        """
        if initial_state is None:
            # Default initialization
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

        # initialize gas storage (keep separate state namespace)
        try:
            gs_state = None
            if initial_state and "gas_storage" in initial_state:
                gs_state = initial_state["gas_storage"]
            self.gas_storage.initialize(gs_state)
        except Exception as e:
            print(e)
            self.gas_storage.initialize()

        # Mark as initialized
        self._initialized = True

    def step(self, t: float, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform one simulation time step.

        Args:
            t (float): Current time in days.
            dt (float): Time step in days.
            inputs (Dict[str, Any]): Input data with keys:
                - 'Q_substrates': Fresh substrate feed rates [m³/d]
                - 'Q_in': Influent from previous digester [m³/d]
                - 'state_in': ADM1 state from previous digester (if connected)

        Returns:
            Dict[str, Any]: Output data with keys:
                - 'Q_out': Effluent flow rate [m³/d]
                - 'state_out': ADM1 state vector for next digester
                - 'Q_gas': Biogas production [m³/d]
                - 'Q_ch4': Methane production [m³/d]
                - 'Q_co2': CO2 production [m³/d]
                - 'pH': pH value
                - 'VFA': VFA concentration [g/L]
                - 'TAC': TAC concentration [g CaCO3/L]
        """
        # Get substrate feed or influent from previous stage
        if "Q_substrates" in inputs:
            self.Q_substrates = inputs["Q_substrates"]

        # Create influent stream
        # If connected to previous digester, mix with its effluent
        if "state_in" in inputs and "Q_in" in inputs:
            # TODO: Implement mixing of substrate feed with previous stage effluent
            # For now, just use substrate feed
            pass

        # Update ADM1 influent
        self.adm1.create_influent(self.Q_substrates, int(t / dt))

        # Simulate ADM1
        t_span = [t, t + dt]
        self.adm1_state = self.simulator.simulate_AD_plant(t_span, self.adm1_state)

        # Calculate gas production
        q_gas, q_ch4, q_co2, p_gas = self.adm1.calc_gas(
            self.adm1_state[33],  # pi_Sh2
            self.adm1_state[34],  # pi_Sch4
            self.adm1_state[35],  # pi_Sco2
            self.adm1_state[36],  # pTOTAL
        )

        # Update state
        self.state["adm1_state"] = self.adm1_state
        self.state["Q_gas"] = q_gas
        self.state["Q_ch4"] = q_ch4
        self.state["Q_co2"] = q_co2

        # Calculate process indicators (if available via DLL)
        try:
            self.state["pH"] = ADMstate.calcPHOfADMstate(self.adm1_state)
            self.state["VFA"] = ADMstate.calcVFAOfADMstate(self.adm1_state, "gHAceq/l").Value
            self.state["TAC"] = ADMstate.calcTACOfADMstate(self.adm1_state, "gCaCO3eq/l").Value
        except Exception as e:
            # Fallback if DLL not available
            print(f"Warning: Could not calculate process indicators: {e}")

        # Prepare output
        Q_out = np.sum(self.Q_substrates)

        # --- integrate with attached gas storage ---
        gs_inputs = {
            "Q_gas_in_m3_per_day": q_gas,  # ✓ This is correct - daily gas production
            "Q_gas_out_m3_per_day": 0.0,  # Storage not drained by digester directly
            "vent_to_flare": True,
        }

        gs_outputs = self.gas_storage.step(t=t, dt=dt, inputs=gs_inputs)

        # Store the gas input for potential second-pass access
        self.gas_storage.outputs_data["Q_gas_in_m3_per_day"] = q_gas

        # --- digester outputs ---
        self.outputs_data = {
            "Q_out": Q_out,
            "state_out": self.adm1_state,
            "Q_gas": q_gas,
            "Q_ch4": q_ch4,
            "Q_co2": q_co2,
            "pH": self.state.get("pH", 7.0),
            "VFA": self.state.get("VFA", 0.0),
            "TAC": self.state.get("TAC", 0.0),
            # gas sent to storage
            "Q_gas_to_storage_m3_per_day": q_gas,
            # inline storage diagnostics
            "gas_storage": {
                "component_id": self.gas_storage.component_id,
                "stored_volume_m3": gs_outputs["stored_volume_m3"],
                "pressure_bar": gs_outputs["pressure_bar"],
                "vented_volume_m3": gs_outputs["vented_volume_m3"],
                "Q_gas_supplied_m3_per_day": gs_outputs["Q_gas_supplied_m3_per_day"],
            },
        }

        return self.outputs_data

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dictionary.

        Returns:
            Dict[str, Any]: Component configuration as dictionary.
        """
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
    def from_dict(cls, config: Dict[str, Any], feedstock: Feedstock) -> "Digester":
        """
        Create from dictionary.

        Args:
            config (Dict[str, Any]): Component configuration.
            feedstock (Feedstock): Feedstock object.

        Returns:
            Digester: Initialized digester component.
        """
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

    def apply_calibration_parameters(self, parameters: dict) -> None:
        """
        Apply calibration parameters to this digester.

        Stores parameters for use during simulation. These override the
        substrate-dependent parameters calculated from feedstock.

        Args:
            parameters: Parameter values as {param_name: value}.

        Example:
            >>> digester.apply_calibration_parameters({
            ...     'k_dis': 0.55,
            ...     'Y_su': 0.105,
            ...     'k_hyd_ch': 11.0
            ... })
        """
        if not hasattr(self, "_calibration_params"):
            self._calibration_params = {}

        self._calibration_params.update(parameters)

        # Also store in ADM1 instance for access during simulation
        self.adm1._calibration_params = self._calibration_params.copy()

        if hasattr(self, "_verbose") and self._verbose:
            print(f"Applied {len(parameters)} calibration parameters to digester '{self.component_id}'")

    def get_calibration_parameters(self) -> dict:
        """
        Get currently applied calibration parameters.

        Returns:
            dict: Current calibration parameters as {param_name: value}.

        Example:
            >>> params = digester.get_calibration_parameters()
            >>> print(params)
            {'k_dis': 0.55, 'Y_su': 0.105}
        """
        if hasattr(self, "_calibration_params"):
            return self._calibration_params.copy()
        return {}

    def clear_calibration_parameters(self) -> None:
        """
        Clear all calibration parameters and revert to default substrate-dependent values.

        Example:
            >>> digester.clear_calibration_parameters()
        """
        if hasattr(self, "_calibration_params"):
            del self._calibration_params

        if hasattr(self.adm1, "_calibration_params"):
            del self.adm1._calibration_params

        if hasattr(self, "_verbose") and self._verbose:
            print(f"Cleared calibration parameters from digester '{self.component_id}'")
