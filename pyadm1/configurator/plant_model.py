# ============================================================================
# pyadm1/plant/plant_model.py
# ============================================================================
"""
Main biogas plant model with component management and JSON I/O.
"""

import json
from typing import Dict, Any, List, Optional

from pyadm1.plant.component_base import Component, ComponentType
from pyadm1.plant.digester import Digester
from pyadm1.plant.chp import CHP
from pyadm1.plant.heating import HeatingSystem
from pyadm1.plant.connection import Connection
from pyadm1.substrates.feedstock import Feedstock


class BiogasPlant:
    """
    Complete biogas plant model with multiple components.

    Manages component lifecycle, connections, and simulation.
    Supports JSON-based configuration.
    """

    def __init__(self, plant_name: str = "Biogas Plant"):
        """
        Initialize biogas plant.

        Parameters
        ----------
        plant_name : str
            Name of the plant
        """
        self.plant_name = plant_name
        self.components: Dict[str, Component] = {}
        self.connections: List[Connection] = []
        self.simulation_time = 0.0

    def add_component(self, component: Component) -> None:
        """
        Add a component to the plant.

        Parameters
        ----------
        component : Component
            Component to add
        """
        if component.component_id in self.components:
            raise ValueError(f"Component with ID '{component.component_id}' already exists")
        self.components[component.component_id] = component

    def add_connection(self, connection: Connection) -> None:
        """
        Add a connection between components.

        Parameters
        ----------
        connection : Connection
            Connection to add
        """
        # Verify components exist
        if connection.from_component not in self.components:
            raise ValueError(f"Source component '{connection.from_component}' not found")
        if connection.to_component not in self.components:
            raise ValueError(f"Target component '{connection.to_component}' not found")

        # Update component connections
        from_comp = self.components[connection.from_component]
        to_comp = self.components[connection.to_component]

        from_comp.add_output(connection.to_component)
        to_comp.add_input(connection.from_component)

        self.connections.append(connection)

    def initialize(self) -> None:
        """Initialize all components."""
        for component in self.components.values():
            component.initialize()

    def step(self, dt: float) -> Dict[str, Dict[str, Any]]:
        """
        Perform one simulation time step for all components.

        Parameters
        ----------
        dt : float
            Time step [days]

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Results from all components
        """
        results = {}

        # Build dependency graph and execute in order
        execution_order = self._get_execution_order()

        for component_id in execution_order:
            component = self.components[component_id]

            # Gather inputs from connected components
            inputs = {}
            for input_id in component.inputs:
                if input_id in self.components:
                    input_comp = self.components[input_id]
                    inputs.update(input_comp.outputs_data)

            # Execute component
            output = component.step(self.simulation_time, dt, inputs)
            results[component_id] = output

        self.simulation_time += dt

        return results

    def simulate(self, duration: float, dt: float = 1.0 / 24.0, save_interval: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Run simulation for specified duration.

        Parameters
        ----------
        duration : float
            Simulation duration [days]
        dt : float
            Time step [days]
        save_interval : Optional[float]
            Interval for saving results [days]. If None, saves every step.

        Returns
        -------
        List[Dict[str, Any]]
            Simulation results at each saved time point
        """
        if save_interval is None:
            save_interval = dt

        results = []
        next_save_time = self.simulation_time

        n_steps = int(duration / dt)

        for i in range(n_steps):
            step_result = self.step(dt)

            # Save results at specified interval
            if self.simulation_time >= next_save_time:
                results.append(
                    {
                        "time": self.simulation_time,
                        "components": step_result,
                    }
                )
                next_save_time += save_interval

            if (i + 1) % 100 == 0:
                print(f"Simulated {i + 1}/{n_steps} steps")

        return results

    def _get_execution_order(self) -> List[str]:
        """
        Determine execution order based on component dependencies.

        Returns topologically sorted list of component IDs.
        """
        # Simple topological sort
        visited = set()
        order = []

        def visit(comp_id: str):
            if comp_id in visited:
                return
            visited.add(comp_id)

            component = self.components[comp_id]
            for input_id in component.inputs:
                if input_id in self.components:
                    visit(input_id)

            order.append(comp_id)

        for comp_id in self.components:
            visit(comp_id)

        return order

    def to_json(self, filepath: str) -> None:
        """
        Save plant configuration to JSON file.

        Parameters
        ----------
        filepath : str
            Path to JSON file
        """
        config = {
            "plant_name": self.plant_name,
            "simulation_time": self.simulation_time,
            "components": [comp.to_dict() for comp in self.components.values()],
            "connections": [conn.to_dict() for conn in self.connections],
        }

        with open(filepath, "w") as f:
            json.dump(config, f, indent=2)

        print(f"Plant configuration saved to {filepath}")

    @classmethod
    def from_json(cls, filepath: str, feedstock: Optional[Feedstock] = None) -> "BiogasPlant":
        """
        Load plant configuration from JSON file.

        Parameters
        ----------
        filepath : str
            Path to JSON file
        feedstock : Optional[Feedstock]
            Feedstock object for digesters (required if plant has digesters)

        Returns
        -------
        BiogasPlant
            Loaded plant model
        """
        with open(filepath, "r") as f:
            config = json.load(f)

        plant = cls(config.get("plant_name", "Biogas Plant"))
        plant.simulation_time = config.get("simulation_time", 0.0)

        # Create components
        for comp_config in config.get("components", []):
            comp_type = ComponentType(comp_config["component_type"])

            if comp_type == ComponentType.DIGESTER:
                if feedstock is None:
                    raise ValueError("Feedstock required for loading plant with digesters")
                component = Digester.from_dict(comp_config, feedstock)
            elif comp_type == ComponentType.CHP:
                component = CHP.from_dict(comp_config)
            elif comp_type == ComponentType.HEATING:
                component = HeatingSystem.from_dict(comp_config)
            else:
                raise ValueError(f"Unknown component type: {comp_type}")

            plant.add_component(component)

        # Create connections
        for conn_config in config.get("connections", []):
            connection = Connection.from_dict(conn_config)
            plant.add_connection(connection)

        print(f"Plant configuration loaded from {filepath}")

        return plant

    def get_summary(self) -> str:
        """
        Get human-readable summary of plant configuration.

        Returns
        -------
        str
            Summary text
        """
        lines = [
            f"=== {self.plant_name} ===",
            f"Simulation time: {self.simulation_time:.2f} days",
            f"\nComponents ({len(self.components)}):",
        ]

        for comp in self.components.values():
            lines.append(f"  - {comp.name} ({comp.component_type.value})")

        lines.append(f"\nConnections ({len(self.connections)}):")
        for conn in self.connections:
            from_name = self.components[conn.from_component].name
            to_name = self.components[conn.to_component].name
            lines.append(f"  - {from_name} -> {to_name} ({conn.connection_type})")

        return "\n".join(lines)
