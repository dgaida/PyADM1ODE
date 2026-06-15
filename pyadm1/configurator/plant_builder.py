# ============================================================================
# pyadm1/configurator/plant_builder.py
# ============================================================================
"""
Main biogas plant model with component management and JSON I/O.

This module provides the BiogasPlant class which manages multiple components
and their connections to build complete biogas plant configurations.
"""

import json
from typing import Dict, Any, List, Optional

from pyadm1.components.base import Component, ComponentType
from pyadm1.components.biological.digester import Digester
from pyadm1.configurator.connection_manager import Connection
from pyadm1.substrates.feedstock import Feedstock


class BiogasPlant:
    """
    Complete biogas plant model with multiple components.

    Manages component lifecycle, connections, and simulation.
    Supports JSON-based configuration.

    Attributes:
        plant_name (str): Name of the biogas plant.
        components (Dict[str, Component]): Dictionary of all plant components.
        connections (List[Connection]): List of connections between components.
        simulation_time (float): Current simulation time in days.

    Example:
        >>> from pyadm1 import Feedstock, BiogasPlant
        >>> from pyadm1.components.biological import Digester
        >>>
        >>> feedstock = Feedstock(["maize_silage_milk_ripeness", "swine_manure"], feeding_freq=24)
        >>> plant = BiogasPlant("My Plant")
        >>> digester = Digester("dig1", feedstock, V_liq=1200, V_gas=216, T_ad=315.15)
        >>> plant.add_component(digester)
        >>> plant.initialize()
    """

    def __init__(self, plant_name: str = "Biogas Plant"):
        """
        Initialize biogas plant.

        Args:
            plant_name (str): Name of the plant. Defaults to "Biogas Plant".
        """
        self.plant_name = plant_name
        self.components: Dict[str, Component] = {}
        self.connections: List[Connection] = []
        self.simulation_time = 0.0

    def add_component(self, component: Component) -> None:
        """
        Add a component to the plant.

        Args:
            component (Component): Component to add to the plant.

        Raises:
            ValueError: If component with same ID already exists.
        """
        if component.component_id in self.components:
            raise ValueError(f"Component with ID '{component.component_id}' already exists")
        self.components[component.component_id] = component

    def add_connection(self, connection: Connection) -> None:
        """
        Add a connection between components.

        Args:
            connection (Connection): Connection to add.

        Raises:
            ValueError: If source or target component not found.
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
        """
        Initialize all components.

        Note: Most components auto-initialize in their constructor.
        This method is kept for compatibility and to ensure any
        components that need explicit initialization are handled.
        """
        for component in self.components.values():
            if not component._initialized:
                component.initialize()

    def step(self, dt: float) -> Dict[str, Dict[str, Any]]:
        """
        Perform one simulation time step for all components.

        This uses a multi-pass execution model:
        1. Execute digesters to produce gas → storages
        2. Execute CHPs to determine gas demand → storages
        3. Execute storages to supply gas → CHPs (re-execute with actual supply)
        4. Execute BiogasUpgrading units: re-execute storages with BGAA demand,
           then re-execute BGAA with actual supply, then re-execute its flare.

        Args:
            dt (float): Time step in days.

        Returns:
            Dict[str, Dict[str, Any]]: Results from all components.
        """
        results = {}

        # Build dependency graph
        execution_order = self._get_execution_order()

        # ========================================================================
        # PASS 1: Execute all non-storage, non-heating components
        # ========================================================================
        # Heaters are also deferred (until after Pass 3) because their CHP
        # input ``P_th_available`` is only meaningful once the CHP has run
        # in Pass 3 with the actual gas supply. Stepping a heater here would
        # see a stale/zero ``P_th`` and (a) charge the auxiliary boiler for
        # heat the CHP would otherwise have supplied and (b) double-count the
        # heater's ``energy_consumed`` if we then re-stepped it in Pass 3.
        for component_id in execution_order:
            component = self.components[component_id]

            # Skip storages in first pass
            if component.component_type.value == "storage":
                continue
            # Skip heaters in first pass (see comment above)
            if component.component_type.value == "heating":
                continue

            # Gather inputs from connected components. For "liquid"
            # cascade connections (e.g. primary digester -> post-digester),
            # the upstream component writes its effluent as Q_out /
            # state_out but the downstream digester reads its inflow as
            # Q_in / state_in -- remap on the fly so the cascade actually
            # carries flow. Other connection types (gas, heat, ...) keep
            # their keys verbatim.
            inputs = {}
            for input_id in component.inputs:
                if input_id not in self.components:
                    continue
                input_comp = self.components[input_id]
                out = input_comp.outputs_data
                conn_type = None
                for conn in self.connections:
                    if conn.from_component == input_id and conn.to_component == component_id:
                        conn_type = conn.connection_type
                        break
                if conn_type == "liquid":
                    if "Q_out" in out:
                        inputs["Q_in"] = out["Q_out"]
                    if "state_out" in out:
                        inputs["state_in"] = out["state_out"]
                else:
                    inputs.update(out)

            # Execute component
            output = component.step(self.simulation_time, dt, inputs)
            results[component_id] = output

        # ========================================================================
        # PASS 2: Execute gas storages with gas production from digesters
        # ========================================================================
        for component_id in execution_order:
            component = self.components[component_id]

            if component.component_type.value != "storage":
                continue

            # Get gas input from connected digesters
            gas_input = 0.0
            for conn in self.connections:
                if conn.to_component == component_id and conn.connection_type == "gas":
                    source_comp = self.components.get(conn.from_component)
                    if source_comp and source_comp.component_type.value == "digester":
                        gas_input += source_comp.outputs_data.get("Q_gas", 0.0)

            # Execute storage with production only (no demand yet)
            storage_inputs = {
                "Q_gas_in_m3_per_day": gas_input,
                "Q_gas_out_m3_per_day": 0.0,  # No demand yet
                "vent_to_flare": True,
            }

            output = component.step(self.simulation_time, dt, storage_inputs)
            results[component_id] = output

        # ========================================================================
        # PASS 3: Handle gas demand from CHPs
        # ========================================================================
        for component_id, component in self.components.items():
            if component.component_type.value != "chp":
                continue

            # Calculate CHP gas demand based on current operating point
            P_el_nom = component.P_el_nom
            eta_el = component.eta_el
            load_setpoint = 1.0  # Full load by default

            # Calculate required gas (m³/d biogas at 60% CH4)
            E_ch4 = 10.0  # kWh/m³ CH4
            CH4_content = 0.60
            P_required = load_setpoint * P_el_nom  # kW
            Q_ch4_required = (P_required / eta_el) * 24.0 / E_ch4  # m³/d CH4
            Q_gas_required = Q_ch4_required / CH4_content  # m³/d biogas

            # Find all gas storages connected to this CHP
            connected_storages = []
            for conn in self.connections:
                if conn.to_component == component_id and conn.connection_type == "gas":
                    storage_id = conn.from_component
                    if storage_id in self.components:
                        storage_comp = self.components[storage_id]
                        if storage_comp.component_type.value == "storage":
                            connected_storages.append(storage_id)

            if not connected_storages:
                continue

            # Distribute demand equally among storages
            demand_per_storage = Q_gas_required / len(connected_storages)

            total_supplied = 0.0

            # Re-execute storages with gas demand.
            #
            # IMPORTANT: do not pass ``Q_gas_in_m3_per_day`` again here.
            # The digester's production for this step was already routed
            # into the storage in Pass 2 and is now sitting in
            # ``storage.stored_volume_m3``. The storage's ``step()`` adds
            # ``Q_in * dt`` to ``stored_volume_m3`` on every call, so
            # re-passing the production would double-count it -- inflating
            # both inventory growth and (when the storage is saturated)
            # vented gas, producing the "vented > produced" symptom in
            # whole-run mass balances.
            #
            # We also need to preserve any vent that happened in Pass 2,
            # because the storage resets ``vented_volume_m3`` to 0 at the
            # start of every ``step()`` call. After this Pass-3 call, the
            # output dict's ``vented_volume_m3`` would otherwise only
            # contain Pass 3's vent (typically 0; Pass 2 carries the
            # overflow vents). Carry the Pass-2 value through manually.
            for storage_id in connected_storages:
                storage = self.components[storage_id]

                pass2_vented = float(results.get(storage_id, {}).get("vented_volume_m3", 0.0))

                storage_inputs = {
                    "Q_gas_in_m3_per_day": 0.0,
                    "Q_gas_out_m3_per_day": demand_per_storage,
                    "vent_to_flare": True,
                }

                storage_output = storage.step(self.simulation_time, dt, storage_inputs)

                # Merge Pass-2 vent volume into the final per-step report.
                # ``cumulative_vented_m3`` already tracks all vents across
                # both passes inside the storage component itself, so we
                # only need to repair the per-step field here.
                storage_output["vented_volume_m3"] = float(storage_output.get("vented_volume_m3", 0.0)) + pass2_vented
                results[storage_id] = storage_output

                # Accumulate supplied gas
                supplied = storage_output.get("Q_gas_supplied_m3_per_day", 0.0)
                total_supplied += supplied

            # Re-execute CHP with actual gas supply
            chp_inputs = {
                "Q_gas_supplied_m3_per_day": total_supplied,
                "load_setpoint": load_setpoint,
            }

            chp_output = component.step(self.simulation_time, dt, chp_inputs)
            results[component_id] = chp_output

            # ============================================================
            # PASS 3b: Re-execute heaters connected to this CHP with the
            # CHP's now-current thermal output. Without this, heaters
            # always see ``P_th_available = 0`` from the CHP's
            # initialised/stale outputs and charge all heat demand to the
            # auxiliary boiler -- even when the CHP has plenty of free
            # thermal power.
            #
            # When multiple heaters share one CHP, split the available
            # thermal power equally. This is a defensible-but-simple
            # allocation that preserves energy conservation
            # (``sum(P_th_used) <= chp.P_th``). A demand-proportional split
            # would require a separate iteration.
            # ============================================================
            connected_heaters = [
                conn.to_component
                for conn in self.connections
                if conn.from_component == component_id
                and conn.connection_type == "heat"
                and conn.to_component in self.components
                and self.components[conn.to_component].component_type.value == "heating"
            ]

            if connected_heaters:
                P_th_per_heater = chp_output.get("P_th", 0.0) / len(connected_heaters)
                for h_id in connected_heaters:
                    heater = self.components[h_id]
                    heater_inputs: Dict[str, Any] = {"P_th_available": P_th_per_heater}
                    # Forward any non-heat upstream inputs the heater may
                    # need (e.g. T_digester from a connected digester).
                    for conn in self.connections:
                        if conn.to_component == h_id and conn.connection_type != "heat":
                            source = self.components.get(conn.from_component)
                            if source is not None:
                                heater_inputs.update(source.outputs_data)
                    results[h_id] = heater.step(self.simulation_time, dt, heater_inputs)

        # ============================================================
        # Catch any heaters NOT connected to a CHP. They were skipped in
        # Pass 1 to avoid the stale-CHP-output problem, so they need to
        # run here with their full upstream inputs but no CHP heat.
        # ============================================================
        stepped_heaters = {cid for cid in results if cid in self.components}
        for component_id in execution_order:
            component = self.components[component_id]
            if component.component_type.value != "heating":
                continue
            if component_id in stepped_heaters:
                continue
            heater_inputs = {"P_th_available": 0.0}
            for conn in self.connections:
                if conn.to_component == component_id:
                    source = self.components.get(conn.from_component)
                    if source is not None:
                        heater_inputs.update(source.outputs_data)
            results[component_id] = component.step(self.simulation_time, dt, heater_inputs)

        # ========================================================================
        # PASS 4: Handle gas demand from BiogasUpgrading units
        #
        # Mirror of the CHP pass: re-execute connected storages with BGAA's
        # capacity as demand, then re-execute the BGAA with the actually supplied
        # volume, then update the downstream flare with the capacity overflow.
        # Without this pass, storages never supply gas to BGAA and fill until the
        # storage's internal overflow vent fires — causing the flare to activate
        # spuriously at every timestep once the storage is full.
        # ========================================================================
        for component_id, component in self.components.items():
            if component.component_type.value != "upgrading":
                continue

            # BGAA requests its full nominal capacity; supply is capped by storage level
            Q_gas_demand = component.capacity_m3_per_day

            connected_storages = [
                conn.from_component
                for conn in self.connections
                if conn.to_component == component_id
                and conn.connection_type == "gas"
                and conn.from_component in self.components
                and self.components[conn.from_component].component_type.value == "storage"
            ]

            if not connected_storages:
                continue

            demand_per_storage = Q_gas_demand / len(connected_storages)
            total_supplied = 0.0

            for storage_id in connected_storages:
                storage = self.components[storage_id]

                # Preserve the per-step vent logged in Pass 2 (same reason as CHP pass)
                pass2_vented = float(results.get(storage_id, {}).get("vented_volume_m3", 0.0))

                storage_inputs = {
                    "Q_gas_in_m3_per_day": 0.0,  # already added in Pass 2; avoid double-count
                    "Q_gas_out_m3_per_day": demand_per_storage,
                    "vent_to_flare": True,
                }
                storage_output = storage.step(self.simulation_time, dt, storage_inputs)
                storage_output["vented_volume_m3"] = float(storage_output.get("vented_volume_m3", 0.0)) + pass2_vented
                results[storage_id] = storage_output
                total_supplied += storage_output.get("Q_gas_supplied_m3_per_day", 0.0)

            # Re-execute BGAA with actual gas supply
            bgaa_output = component.step(self.simulation_time, dt, {"Q_gas_in_m3_per_day": total_supplied})
            results[component_id] = bgaa_output

            # Re-execute downstream flare with BGAA's capacity overflow
            for conn in self.connections:
                if conn.from_component != component_id or conn.connection_type != "gas":
                    continue
                flare_id = conn.to_component
                if flare_id in self.components and self.components[flare_id].component_type.value == "flare":
                    flare_inputs = {"Q_gas_in_m3_per_day": bgaa_output.get("Q_gas_out_m3_per_day", 0.0)}
                    results[flare_id] = self.components[flare_id].step(self.simulation_time, dt, flare_inputs)

        self.simulation_time += dt

        return results

    def simulate(
        self,
        duration: float,
        dt: float = 1.0 / 24.0,
        save_interval: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run simulation for specified duration.

        Args:
            duration (float): Simulation duration in days.
            dt (float): Time step in days. Defaults to 1 hour (1/24 day).
            save_interval (Optional[float]): Interval for saving results in days.
                If None, saves every step.

        Returns:
            List[Dict[str, Any]]: Simulation results at each saved time point.
                Each entry contains 'time' and 'components' with component results.

        Example:
            >>> results = plant.simulate(duration=30, dt=1/24, save_interval=1.0)
            >>> print(f"Simulated {len(results)} time points")
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

        Returns:
            List[str]: Topologically sorted list of component IDs.
        """
        # Simple topological sort
        visited = set()
        order = []

        def visit(comp_id: str) -> None:
            """Recursive DFS helper: visit dependencies first, then append this component."""
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

        Args:
            filepath (str): Path to JSON file.
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

        Args:
            filepath (str): Path to JSON file.
            feedstock (Optional[Feedstock]): Feedstock object for digesters.
                Required if plant has digesters.

        Returns:
            BiogasPlant: Loaded plant model.

        Raises:
            ValueError: If feedstock is None but plant contains digesters.
        """
        with open(filepath, "r") as f:
            config = json.load(f)

        plant = cls(config.get("plant_name", "Biogas Plant"))
        plant.simulation_time = config.get("simulation_time", 0.0)

        # Create components.  Digesters need the supplied feedstock; for every
        # other component type we delegate to the class's ``from_dict`` via the
        # global component registry so all auto-attached children (gas storage,
        # flares, mixers, sensors, …) round-trip without bespoke branches.
        from pyadm1.components.registry import get_registry

        registry = get_registry()
        type_to_registry_key = {
            ComponentType.DIGESTER: "Digester",
            ComponentType.CHP: "CHP",
            ComponentType.HEATING: "HeatingSystem",
            ComponentType.STORAGE: "GasStorage",
            ComponentType.FLARE: "Flare",
            ComponentType.BOILER: "Boiler",
            ComponentType.SEPARATOR: "Separator",
            ComponentType.UPGRADING: "BiogasUpgrading",
        }

        for comp_config in config.get("components", []):
            comp_type = ComponentType(comp_config["component_type"])

            if comp_type == ComponentType.DIGESTER:
                if feedstock is None:
                    raise ValueError("Feedstock required for loading plant with digesters")
                component = Digester.from_dict(comp_config, feedstock)
            elif comp_type in type_to_registry_key:
                cls_obj = registry.get_registered_components().get(type_to_registry_key[comp_type])
                if cls_obj is None:
                    raise ValueError(f"No registered class for component type: {comp_type}")
                component = cls_obj.from_dict(comp_config)
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

        Returns:
            str: Summary text with components and connections.
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
