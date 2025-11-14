# ============================================================================
# pyadm1/plant/component_base.py
# ============================================================================
"""
Base classes for biogas plant components.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional


class ComponentType(Enum):
    """Enumeration of component types."""

    DIGESTER = "digester"
    CHP = "chp"
    HEATING = "heating"
    STORAGE = "storage"
    SEPARATOR = "separator"
    MIXER = "mixer"


class Component(ABC):
    """
    Abstract base class for all biogas plant components.

    All components must implement the step() method which performs
    one simulation time step.
    """

    def __init__(self, component_id: str, component_type: ComponentType, name: Optional[str] = None):
        """
        Initialize a component.

        Parameters
        ----------
        component_id : str
            Unique identifier for this component
        component_type : ComponentType
            Type of component
        name : Optional[str]
            Human-readable name (defaults to component_id)
        """
        self.component_id = component_id
        self.component_type = component_type
        self.name = name or component_id

        # Connections to other components
        self.inputs: List[str] = []  # IDs of input components
        self.outputs: List[str] = []  # IDs of output components

        # State variables
        self.state: Dict[str, Any] = {}

        # Output variables (what this component provides to others)
        self.outputs_data: Dict[str, Any] = {}

        # Flag to track if component has been initialized
        self._initialized: bool = False

    @abstractmethod
    def step(self, t: float, dt: float, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform one simulation time step.

        Parameters
        ----------
        t : float
            Current simulation time [days]
        dt : float
            Time step size [days]
        inputs : Dict[str, Any]
            Input data from connected components

        Returns
        -------
        Dict[str, Any]
            Output data to be passed to connected components
        """
        pass

    @abstractmethod
    def initialize(self, initial_state: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize component state.

        Parameters
        ----------
        initial_state : Optional[Dict[str, Any]]
            Initial state values
        """
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize component to dictionary for JSON export.

        Returns
        -------
        Dict[str, Any]
            Component configuration as dictionary
        """
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, config: Dict[str, Any]) -> "Component":
        """
        Create component from dictionary configuration.

        Parameters
        ----------
        config : Dict[str, Any]
            Component configuration

        Returns
        -------
        Component
            Initialized component
        """
        pass

    def get_state(self) -> Dict[str, Any]:
        """Get current component state."""
        return self.state.copy()

    def set_state(self, state: Dict[str, Any]) -> None:
        """Set component state."""
        self.state = state.copy()

    def add_input(self, component_id: str) -> None:
        """Add an input connection."""
        if component_id not in self.inputs:
            self.inputs.append(component_id)

    def add_output(self, component_id: str) -> None:
        """Add an output connection."""
        if component_id not in self.outputs:
            self.outputs.append(component_id)
