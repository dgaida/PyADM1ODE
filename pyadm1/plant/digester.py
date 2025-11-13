# ============================================================================
# pyadm1/plant/digester.py
# ============================================================================
"""
Digester component wrapping PyADM1 model.
"""

import clr
import os

from typing import Dict, Any, List, Optional
import numpy as np

from pyadm1.plant.component_base import Component, ComponentType
from pyadm1.core.pyadm1 import PyADM1
from pyadm1.substrates.feedstock import Feedstock
from pyadm1.core.simulator import Simulator

# CLR reference must be added before importing from DLL
dll_path = os.path.join(os.path.dirname(__file__), "..", "dlls")
clr.AddReference(os.path.join(dll_path, "plant"))
from biogas import ADMstate  # noqa: E402  # type: ignore


class Digester(Component):
    """
    Digester component using ADM1 model.

    This component wraps the PyADM1 implementation and can be
    connected to other digesters or components in series/parallel.
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

        Parameters
        ----------
        component_id : str
            Unique identifier
        feedstock : Feedstock
            Feedstock object for substrate management
        V_liq : float
            Liquid volume [m³]
        V_gas : float
            Gas volume [m³]
        T_ad : float
            Operating temperature [K]
        name : Optional[str]
            Human-readable name
        """
        super().__init__(component_id, ComponentType.DIGESTER, name)

        self.feedstock = feedstock
        self.V_liq = V_liq
        self.V_gas = V_gas
        self.T_ad = T_ad

        # Initialize ADM1
        self.adm1 = PyADM1(feedstock)
        self.adm1.V_liq = V_liq
        self.adm1._V_gas = V_gas
        self.adm1._T_ad = T_ad

        self.simulator = Simulator(self.adm1)

        # ADM1 state vector (37 dimensions)
        self.adm1_state: List[float] = []

        # Substrate feed rates [m³/d]
        self.Q_substrates: List[float] = [0] * 10

    def initialize(self, initial_state: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize digester state.

        Parameters
        ----------
        initial_state : Optional[Dict[str, Any]]
            Initial state with keys:
            - 'adm1_state': ADM1 state vector (37 dims)
            - 'Q_substrates': Substrate feed rates
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

    def step(self, t: float, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform one simulation time step.

        Parameters
        ----------
        t : float
            Current time [days]
        dt : float
            Time step [days]
        inputs : Dict[str, Any]
            Input data with keys:
            - 'Q_substrates': Fresh substrate feed rates [m³/d]
            - 'Q_in': Influent from previous digester [m³/d]
            - 'state_in': ADM1 state from previous digester (if connected)

        Returns
        -------
        Dict[str, Any]
            Output data with keys:
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
        self.adm1.createInfluent(self.Q_substrates, int(t / dt))

        # Simulate ADM1
        t_span = [t, t + dt]
        self.adm1_state = self.simulator.simulateADplant(t_span, self.adm1_state)

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
            print(e)

        # Prepare output
        Q_out = np.sum(self.Q_substrates)

        self.outputs_data = {
            "Q_out": Q_out,
            "state_out": self.adm1_state,
            "Q_gas": q_gas,
            "Q_ch4": q_ch4,
            "Q_co2": q_co2,
            "pH": self.state.get("pH", 7.0),
            "VFA": self.state.get("VFA", 0.0),
            "TAC": self.state.get("TAC", 0.0),
        }

        return self.outputs_data

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
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
        """Create from dictionary."""
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
