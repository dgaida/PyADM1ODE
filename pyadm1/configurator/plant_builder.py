# ============================================================================
# pyadm1/configurator/plant_builder.py
# ============================================================================
"""
Main biogas plant model with component management and JSON I/O.

This module provides the BiogasPlant class which manages multiple components
and their connections to build complete biogas plant configurations.
"""

from __future__ import annotations

import json
import warnings
from typing import Any

from pyadm1.components.base import Component, ComponentType
from pyadm1.components.biological.digester import Digester
from pyadm1.configurator.connection_manager import Connection
from pyadm1.configurator.graph import Graph, normalize_candidate
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

    def __init__(self, plant_name: str = "Biogas Plant", T_ambient: float = 288.15, T_ground: float = 283.15):
        """
        Initialize biogas plant.

        Args:
            plant_name (str): Name of the plant. Defaults to "Biogas Plant".
            T_ambient (float): Ambient temperature in K, handed to every
                heating system on each step. Defaults to 288.15 (15 °C). The
                attribute may be reassigned between steps, e.g. to follow a
                weather time series.
            T_ground (float): Soil temperature in K, handed to every heating
                system alongside ``T_ambient``. The floor slab of a digester
                with k-values loses heat against it. Defaults to 283.15
                (10 °C), roughly constant over the year below the slab.
        """
        self.plant_name = plant_name
        self.T_ambient = float(T_ambient)
        self.T_ground = float(T_ground)
        # Cumulative electricity balance [kWh], advanced on every step.
        self.energy_produced_kwh = 0.0
        self.energy_consumed_kwh = 0.0
        # Cumulative substrate use, advanced on every step from the feed the
        # digesters actually received: {substrate_id: {"mass_t", "cost"}}.
        self.substrate_use: dict[str, dict[str, float]] = {}
        self.components: dict[str, Component] = {}
        self.connections: list[Connection] = []
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

        A connection without a type (``"default"``) between two components
        whose link can only mean one thing gets that type, e.g. digester to
        digester becomes ``"liquid"``. A connection that already exists is
        skipped with a warning, because a second identical edge would count
        its flow twice.

        Raises:
            ValueError: If source or target component not found.
        """
        # Verify components exist
        if connection.from_component not in self.components:
            raise ValueError(f"Source component '{connection.from_component}' not found")
        if connection.to_component not in self.components:
            raise ValueError(f"Target component '{connection.to_component}' not found")

        if connection.connection_type == "default":
            connection.connection_type = self._infer_connection_type(connection)

        for existing in self.connections:
            if (
                existing.from_component == connection.from_component
                and existing.to_component == connection.to_component
                and existing.connection_type == connection.connection_type
            ):
                warnings.warn(
                    f"Connection {connection.from_component} -> {connection.to_component} "
                    f"({connection.connection_type}) already exists and is not added again.",
                    UserWarning,
                    stacklevel=2,
                )
                return

        # Update component connections
        from_comp = self.components[connection.from_component]
        to_comp = self.components[connection.to_component]

        from_comp.add_output(connection.to_component)
        to_comp.add_input(connection.from_component)

        self.connections.append(connection)

    def _infer_connection_type(self, connection: Connection) -> str:
        """
        Type of an untyped connection, derived from the two component types.

        Args:
            connection (Connection): Connection whose type is ``"default"``.

        Returns:
            str: ``"liquid"``, ``"gas"`` or ``"heat"`` where the pair is
                unambiguous, otherwise ``"default"``.
        """
        source = self.components[connection.from_component].component_type.value
        target = self.components[connection.to_component].component_type.value
        liquid = {"digester", "separator", "pump"}
        if source in liquid and target in liquid:
            return "liquid"
        if (source == "digester" and target == "storage") or (
            source in {"storage", "chp", "upgrading"} and target in {"chp", "upgrading", "flare", "boiler"}
        ):
            return "gas"
        if source in {"chp", "boiler"} and target == "heating":
            return "heat"
        return "default"

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

    def step(self, dt: float) -> dict[str, dict[str, Any]]:
        """
        Perform one simulation time step for all components.

        This uses a multi-pass execution model:
        1. Execute everything that is not part of the gas train (digesters,
           feed chain, mixers, pumps, separators, sensors, ...).
        2. Execute every gas storage once, with the production of its
           digesters and the demand of all its consumers (CHPs, upgrading
           units) known up front.
        3. Execute the consumers with the gas they were actually supplied,
           then the heating systems with the CHP heat now available.
        4. Execute the flares with the gas the storages vented and the
           upgrading units could not take.

        Every component is stepped exactly once per plant step.

        Args:
            dt (float): Time step in days.

        Returns:
            Dict[str, Dict[str, Any]]: Results from all components.
        """
        results = {}

        # Build dependency graph
        execution_order = self._get_execution_order()
        consumers_of = {
            component_id: self._gas_consumers(component_id)
            for component_id, component in self.components.items()
            if component.component_type.value == "storage"
        }
        storage_fed = {consumer_id for consumers in consumers_of.values() for consumer_id in consumers}

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
            # Flares and storage-fed gas consumers run in the gas passes, so
            # that they are stepped once, with the gas they really receive.
            if component.component_type.value in ("flare", "boiler") or component_id in storage_fed:
                continue

            # Gather inputs from connected components. For "liquid"
            # cascade connections (e.g. primary digester -> post-digester),
            # the upstream component writes its effluent as Q_out /
            # state_out but the downstream digester reads its inflow as
            # Q_in / state_in -- remap on the fly so the cascade actually
            # carries flow. Other connection types (gas, heat, ...) keep
            # their keys verbatim.
            inputs = {}
            # Liquid inflows are ACCUMULATED, not overwritten: a post-digester
            # fed by two fermenters receives the sum of both flows, and the
            # state handed downstream is their flow-weighted mixture. Before
            # this, the last input processed silently won and half the flow
            # disappeared. ``split_fraction`` on the connection lets one source
            # feed several targets with a share each (separator recirculation).
            q_liquid = 0.0
            state_load: list[float] | None = None
            feed_rates: dict[int, float] = {}
            feed_storages: list[Component] = []
            for input_id in component.inputs:
                if input_id not in self.components:
                    continue
                input_comp = self.components[input_id]
                out = input_comp.outputs_data

                # Feed chain: substrate storage -> feeder -> digester. The
                # feeder sees the stock, the digester the flow actually dosed.
                source_type = input_comp.component_type.value
                if source_type == "substrate_storage" and component.component_type.value == "feeder":
                    factor = float(getattr(component, "flow_to_mass_factor", 1.0))
                    inputs["substrate_available"] = inputs.get("substrate_available", 0.0) + (
                        input_comp.available_stock / factor
                    )
                    feed_storages.append(input_comp)
                    continue
                if source_type == "feeder" and component.component_type.value == "digester":
                    index = out.get("substrate_index")
                    if index is not None:
                        feed_rates[int(index)] = feed_rates.get(int(index), 0.0) + float(out.get("Q_actual", 0.0))
                    continue
                conn_type = None
                split = 1.0
                for conn in self.connections:
                    if conn.from_component == input_id and conn.to_component == component_id:
                        conn_type = conn.connection_type
                        split = float(getattr(conn, "split_fraction", 1.0))
                        break
                if conn_type == "liquid":
                    q = float(out.get("Q_out", 0.0)) * split
                    if q <= 0.0:
                        continue
                    q_liquid += q
                    state = out.get("state_out")
                    if state is not None:
                        if state_load is None:
                            state_load = [0.0] * len(state)
                        for idx, conc in enumerate(state):
                            if idx < len(state_load):
                                state_load[idx] += q * float(conc)
                else:
                    inputs.update(out)
            if q_liquid > 0.0:
                inputs["Q_in"] = q_liquid
                if state_load is not None:
                    inputs["state_in"] = [load / q_liquid for load in state_load]

            # Slots served by a feeder follow its dosed flow (noise, blockage,
            # empty storage); all other slots keep the configured rate.
            if feed_rates:
                Q_substrates = list(component.Q_substrates)
                for index, rate in feed_rates.items():
                    if 0 <= index < len(Q_substrates):
                        Q_substrates[index] = rate
                inputs["Q_substrates"] = Q_substrates

            # Execute component
            output = component.step(self.simulation_time, dt, inputs)
            results[component_id] = output

            # Book what the feeder actually drew. The storage was stepped
            # before the feeder (losses, refill), so this must not step it again.
            if feed_storages:
                remaining = float(output.get("Q_actual", 0.0)) * dt * float(getattr(component, "flow_to_mass_factor", 1.0))
                for storage in feed_storages:
                    remaining -= storage.withdraw(remaining)
                    results[storage.component_id] = storage.outputs_data

        # ========================================================================
        # PASS 2: Execute every gas storage once
        #
        # Production and demand of a step act on the storage at the same time.
        # Stepping the storage with the production first and the demand later
        # vented a whole step's production before any consumer could draw it,
        # which made the gas burnt depend on step length and storage size.
        # ========================================================================
        supply: dict[str, dict[str, float]] = {cid: {"gas": 0.0, "ch4": 0.0} for cid in storage_fed}
        flare_load: dict[str, dict[str, float]] = {}
        n_storages_of = {
            consumer_id: sum(1 for consumers in consumers_of.values() if consumer_id in consumers)
            for consumer_id in storage_fed
        }
        for component_id in execution_order:
            if component_id not in consumers_of:
                continue
            storage = self.components[component_id]

            gas_input = 0.0
            ch4_input = 0.0
            co2_input = 0.0
            for source_id in self._connected(component_id, "gas", incoming=True, component_type="digester"):
                out = self.components[source_id].outputs_data
                gas_input += float(out.get("Q_gas", 0.0))
                ch4_input += float(out.get("Q_ch4", 0.0))
                co2_input += float(out.get("Q_co2", 0.0))

            # Methane content the consumers will draw: the inventory blended
            # with this step's inflow. A storage that holds (almost) no
            # methane is asked at the consumer's reference content instead of
            # for an unbounded volume.
            stored = float(getattr(storage, "stored_volume_m3", 0.0))
            blended = stored + gas_input * dt
            ch4_fraction = float(getattr(storage, "ch4_fraction", 0.0))
            if blended > 0.0:
                ch4_fraction = (stored * ch4_fraction + ch4_input * dt) / blended

            demands: dict[str, float] = {}
            for consumer_id in consumers_of[component_id]:
                consumer = self.components[consumer_id]
                if consumer.component_type.value == "chp":
                    fraction = ch4_fraction if ch4_fraction >= 0.01 else float(getattr(consumer, "ch4_content_default", 0.60))
                    # Methane the CHP needs at full load [m³/d]
                    q_ch4 = (consumer.P_el_nom / consumer.eta_el) * 24.0 / float(getattr(consumer, "lhv_ch4", 10.0))
                    demands[consumer_id] = q_ch4 / fraction / n_storages_of[consumer_id]
                else:  # upgrading: asks for its nominal raw-gas capacity
                    demands[consumer_id] = float(consumer.capacity_m3_per_day) / n_storages_of[consumer_id]
            total_demand = sum(demands.values())

            output = storage.step(
                self.simulation_time,
                dt,
                {
                    "Q_gas_in_m3_per_day": gas_input,
                    "Q_ch4_in_m3_per_day": ch4_input,
                    "Q_co2_in_m3_per_day": co2_input,
                    "Q_gas_out_m3_per_day": total_demand,
                    "vent_to_flare": True,
                },
            )
            results[component_id] = output

            # Several consumers share what the storage delivered in
            # proportion to what they asked for.
            supplied = float(output.get("Q_gas_supplied_m3_per_day", 0.0))
            supplied_ch4 = float(output.get("Q_ch4_supplied_m3_per_day", 0.0))
            for consumer_id, demand in demands.items():
                share = demand / total_demand if total_demand > 0.0 else 0.0
                supply[consumer_id]["gas"] += supplied * share
                supply[consumer_id]["ch4"] += supplied_ch4 * share

            # Vented gas goes to the flare of this storage, if it has one.
            vented = float(output.get("vented_volume_m3", 0.0))
            flare_id = self._flare_of(component_id, consumers_of[component_id])
            if flare_id is not None and vented > 0.0:
                load = flare_load.setdefault(flare_id, {"gas": 0.0, "ch4": 0.0})
                load["gas"] += vented
                load["ch4"] += vented * float(output.get("CH4_fraction", 0.0))

        # ========================================================================
        # PASS 3: Execute the gas consumers with the gas actually supplied
        # ========================================================================
        for component_id in execution_order:
            if component_id not in storage_fed:
                continue
            component = self.components[component_id]
            got = supply[component_id]

            if component.component_type.value == "chp":
                chp_inputs = {
                    "Q_gas_supplied_m3_per_day": got["gas"],
                    "Q_ch4_supplied_m3_per_day": got["ch4"],
                    "load_setpoint": 1.0,
                }
                results[component_id] = component.step(self.simulation_time, dt, chp_inputs)
                continue

            # Biogas upgrading: gas beyond its capacity goes to its flare.
            bgaa_inputs = {"Q_gas_in_m3_per_day": got["gas"]}
            if got["gas"] > 0.0:
                bgaa_inputs["CH4_fraction"] = got["ch4"] / got["gas"]
            bgaa_output = component.step(self.simulation_time, dt, bgaa_inputs)
            results[component_id] = bgaa_output
            overflow = float(bgaa_output.get("Q_gas_out_m3_per_day", 0.0)) * dt
            flare_ids = self._connected(component_id, "gas", incoming=False, component_type="flare")
            if flare_ids and overflow > 0.0:
                load = flare_load.setdefault(flare_ids[0], {"gas": 0.0, "ch4": 0.0})
                load["gas"] += overflow
                load["ch4"] += overflow * float(bgaa_inputs.get("CH4_fraction", 0.0))

        # ========================================================================
        # PASS 3b: Execute the heating systems. They were skipped in Pass 1
        # because the thermal output of a storage-fed CHP is only known now.
        #
        # When multiple heaters share one CHP, the available thermal power is
        # split equally. This is a defensible-but-simple allocation that
        # preserves energy conservation (``sum(P_th_used) <= chp.P_th``).
        # ========================================================================
        for component_id in execution_order:
            component = self.components[component_id]
            if component.component_type.value != "heating":
                continue
            heater_inputs: dict[str, Any] = {"T_ambient": self.T_ambient, "T_ground": self.T_ground}
            P_th_available = 0.0
            for conn in self.connections:
                source = self.components.get(conn.from_component)
                if conn.to_component != component_id or source is None:
                    continue
                if conn.connection_type == "heat" and source.component_type.value == "chp":
                    n_heaters = len(self._connected(conn.from_component, "heat", incoming=False, component_type="heating"))
                    P_th_available += float(source.outputs_data.get("P_th", 0.0)) / max(n_heaters, 1)
                else:
                    # Non-heat upstream signals (e.g. T_digester, feed).
                    heater_inputs.update(source.outputs_data)
            heater_inputs["P_th_available"] = P_th_available
            results[component_id] = component.step(self.simulation_time, dt, heater_inputs)

        # ========================================================================
        # PASS 3c: Execute the boilers. A boiler covers the auxiliary heat its
        # heating systems still need after the CHP heat, and burns biogas left
        # in its storages after the main consumers were served.
        # ========================================================================
        for component_id in execution_order:
            component = self.components[component_id]
            if component.component_type.value != "boiler":
                continue
            heater_ids = {
                conn.from_component if conn.to_component == component_id else conn.to_component
                for conn in self.connections
                if component_id in (conn.from_component, conn.to_component)
            }
            demand = sum(
                float(self.components[h_id].outputs_data.get("P_aux_heat", 0.0))
                for h_id in heater_ids
                if h_id in self.components and self.components[h_id].component_type.value == "heating"
            )
            boiler_inputs: dict[str, Any] = {"P_th_demand": demand}
            storages = [
                self.components[s_id] for s_id in self._connected(component_id, "gas", incoming=True, component_type="storage")
            ]
            if storages:
                available = sum(float(st.stored_volume_m3) for st in storages)
                boiler_inputs["Q_gas_available_m3_per_day"] = available / dt if dt > 0.0 else 0.0
                if available > 0.0:
                    boiler_inputs["CH4_fraction"] = (
                        sum(float(st.stored_volume_m3) * float(st.ch4_fraction) for st in storages) / available
                    )
            boiler_output = component.step(self.simulation_time, dt, boiler_inputs)
            results[component_id] = boiler_output
            remaining = float(boiler_output.get("Q_gas_consumed_m3_per_day", 0.0)) * dt
            for st in storages:
                remaining -= st.withdraw(remaining)
                results[st.component_id] = st.outputs_data

        # ========================================================================
        # PASS 4: Execute the flares with the gas routed to them
        # ========================================================================
        for component_id in execution_order:
            component = self.components[component_id]
            if component.component_type.value != "flare":
                continue
            load = flare_load.get(component_id, {"gas": 0.0, "ch4": 0.0})
            flare_inputs: dict[str, Any] = {"Q_gas_in_m3_per_day": load["gas"] / dt if dt > 0.0 else 0.0}
            if load["gas"] > 0.0:
                flare_inputs["CH4_fraction"] = load["ch4"] / load["gas"]
            results[component_id] = component.step(self.simulation_time, dt, flare_inputs)

        self.simulation_time += dt

        balance = self.get_electricity_balance()
        self.energy_produced_kwh += balance["P_el_produced"] * dt * 24.0
        self.energy_consumed_kwh += balance["P_el_consumed"] * dt * 24.0

        # Substrate use: what each digester was really fed in this step
        # (after any feeder has overridden the configured rate).
        for component in self.components.values():
            feedstock = getattr(component, "feedstock", None)
            if component.component_type.value != "digester" or not hasattr(feedstock, "calc_substrate_costs"):
                continue
            step_costs = feedstock.calc_substrate_costs(component.Q_substrates, dt)
            for substrate_id, mass in step_costs["mass_t"].items():
                entry = self.substrate_use.setdefault(substrate_id, {"mass_t": 0.0, "cost": 0.0})
                entry["mass_t"] += mass
                entry["cost"] += step_costs["by_substrate"][substrate_id]

        return results

    def _connected(self, component_id: str, connection_type: str, incoming: bool, component_type: str) -> list[str]:
        """
        IDs of the components linked to *component_id* by one kind of connection.

        Args:
            component_id (str): Component whose neighbours are wanted.
            connection_type (str): Connection type to follow, e.g. ``"gas"``.
            incoming (bool): True for sources feeding the component, False
                for targets it feeds.
            component_type (str): ``ComponentType`` value the neighbour must have.

        Returns:
            List[str]: Neighbour IDs in connection order, without duplicates.
        """
        found: list[str] = []
        for conn in self.connections:
            if conn.connection_type != connection_type:
                continue
            here, other = (conn.to_component, conn.from_component) if incoming else (conn.from_component, conn.to_component)
            if here != component_id or other in found:
                continue
            neighbour = self.components.get(other)
            if neighbour is not None and neighbour.component_type.value == component_type:
                found.append(other)
        return found

    def _gas_consumers(self, storage_id: str) -> list[str]:
        """IDs of the CHPs and upgrading units that draw gas from a storage."""
        return self._connected(storage_id, "gas", incoming=False, component_type="chp") + self._connected(
            storage_id, "gas", incoming=False, component_type="upgrading"
        )

    def _flare_of(self, storage_id: str, consumer_ids: list[str]) -> str | None:
        """
        Flare that burns what a storage vents.

        A flare connected to the storage itself takes precedence; otherwise
        the flare downstream of the first consumer that has one is used (the
        configurator attaches the safety flare to the CHP or upgrading unit).

        Args:
            storage_id (str): Gas storage component ID.
            consumer_ids (List[str]): Gas consumers of that storage.

        Returns:
            Optional[str]: Flare component ID, or None when the vented gas has
                no flare to go to (it is then released unburnt).
        """
        for source_id in [storage_id, *consumer_ids]:
            flares = self._connected(source_id, "gas", incoming=False, component_type="flare")
            if flares:
                return flares[0]
        return None

    def get_substrate_costs(self) -> dict[str, Any]:
        """
        Substrate costs of the operation so far, from the simulated feed.

        Every ``step()`` adds what the digesters actually received, priced with
        the ``cost`` [EUR/t FM] of each substrate. For costs from a measured or
        estimated feed instead, use :meth:`Feedstock.calc_substrate_costs`.

        Returns:
            Dict[str, Any]: ``total`` [EUR], ``by_substrate`` ({id: EUR}),
                ``mass_t`` ({id: t FM}) and ``specific_cost`` [EUR/t FM]
                (total cost per tonne fed, 0 when nothing was fed).
        """
        by_substrate = {sid: entry["cost"] for sid, entry in self.substrate_use.items()}
        mass = {sid: entry["mass_t"] for sid, entry in self.substrate_use.items()}
        total = sum(by_substrate.values())
        total_mass = sum(mass.values())
        return {
            "total": total,
            "by_substrate": by_substrate,
            "mass_t": mass,
            "specific_cost": total / total_mass if total_mass > 0.0 else 0.0,
        }

    def get_electricity_balance(self) -> dict[str, Any]:
        """
        Electricity produced and consumed by the plant at the current step.

        Production is the gross ``P_el`` of every CHP. Consumption is the
        ``P_consumed`` every component reports: mixers, pumps, feeders,
        separators, biogas upgrading and the CHP's own auxiliaries.

        Returns:
            Dict[str, Any]: ``P_el_produced``, ``P_el_consumed``, ``P_el_net``
                [kW], ``own_consumption_fraction`` (consumed / produced, 0 when
                nothing is produced), ``consumers`` ({component_id: kW}) and the
                cumulative ``energy_produced_kwh`` / ``energy_consumed_kwh`` /
                ``energy_net_kwh``.
        """
        produced = 0.0
        consumers: dict[str, float] = {}
        for component_id, component in self.components.items():
            out = component.outputs_data
            if component.component_type.value == "chp":
                produced += float(out.get("P_el", 0.0))
            P_consumed = float(out.get("P_consumed", 0.0))
            if P_consumed > 0.0:
                consumers[component_id] = P_consumed
        consumed = sum(consumers.values())

        return {
            "P_el_produced": produced,
            "P_el_consumed": consumed,
            "P_el_net": produced - consumed,
            "own_consumption_fraction": consumed / produced if produced > 0.0 else 0.0,
            "consumers": consumers,
            "energy_produced_kwh": self.energy_produced_kwh,
            "energy_consumed_kwh": self.energy_consumed_kwh,
            "energy_net_kwh": self.energy_produced_kwh - self.energy_consumed_kwh,
        }

    def simulate(
        self,
        duration: float,
        dt: float = 1.0 / 24.0,
        save_interval: float | None = None,
    ) -> list[dict[str, Any]]:
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

        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")

        results = []
        next_save_time = self.simulation_time

        # ``duration / dt`` is rarely an exact integer in floating point
        # (0.3 / 0.1 = 2.999...), so truncating it would drop the last step.
        # A duration that is not a multiple of dt ends with one shorter step.
        tolerance = 1.0e-9
        n_full = int(duration / dt + tolerance)
        remainder = duration - n_full * dt
        step_lengths = [dt] * n_full
        if remainder > tolerance * max(dt, 1.0):
            step_lengths.append(remainder)
        n_steps = len(step_lengths)

        for i, step_dt in enumerate(step_lengths):
            step_result = self.step(step_dt)

            # Save results at specified interval
            if self.simulation_time >= next_save_time - tolerance:
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

    def _get_execution_order(self) -> list[str]:
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

    def to_graph(self) -> Graph:
        """
        Build the typed component/connection graph for this plant.

        Serializes the plant exactly like the LLM benchmark does (the
        ``components`` / ``connections`` dict produced by ``to_dict``) and
        normalizes it into the benchmark ``Graph`` data model. The returned
        graph has one ``Node`` per component (keyed by component id, carrying
        its ``component_type`` and serialized scalar parameters) and one
        ``Edge`` per connection (``liquid`` / ``gas`` / ``heat``).

        Returns:
            Graph: Graph with ``nodes`` (Dict[str, Node]) and ``edges``
                (List[Edge]), identical to what ``normalize_candidate`` builds
                in the benchmark.
        """
        candidate = {
            "components": [comp.to_dict() for comp in self.components.values()],
            "connections": [conn.to_dict() for conn in self.connections],
        }
        return normalize_candidate(candidate)

    def visualize_graph(
        self,
        output_path: str | None = None,
        dpi: int = 150,
        title: str | None = None,
    ) -> str:
        """
        Render the plant graph (see :meth:`to_graph`) to a PNG file.

        Mirrors the Mermaid ``flowchart LR`` style used for the benchmark
        dataset: components are drawn as rounded boxes laid out left-to-right by
        process depth, liquid connections as solid labelled arrows and gas
        connections as dotted labelled arrows (heat as dashed). Unlike the
        dataset diagrams, no dataset-specific annotations (obligations,
        confidences, reference ids) are shown -- only the plant's own
        components and connections.

        Args:
            output_path (Optional[str]): Target PNG path. Defaults to
                ``output/<plant_name>_graph.png`` in the current working
                directory.
            dpi (int): Resolution of the saved figure.
            title (Optional[str]): Figure title. ``None`` (default) shows no
                title.

        Returns:
            str: Path to the written PNG file.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import networkx as nx
        from matplotlib.lines import Line2D
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

        graph = self.to_graph()

        # Serialized component_type -> proper PyADM1ODE class name for display.
        class_names = {
            "digester": "Digester",
            "storage": "GasStorage",
            "chp": "CHP",
            "flare": "Flare",
            "heating": "HeatingSystem",
            "boiler": "Boiler",
            "separator": "Separator",
            "upgrading": "BiogasUpgrading",
            "mixer": "Mixer",
            "pump": "Pump",
            "feeder": "Feeder",
            "substrate_storage": "SubstrateStorage",
            "sensor": "Sensor",
        }

        # Edge style per transport medium, matching the Mermaid dataset look:
        # liquid = solid, gas = dotted, heat = dashed.
        edge_styles = {
            "liquid": {"linestyle": "solid", "color": "#333333", "label": "liquid"},
            "gas": {"linestyle": "dotted", "color": "#C0703C", "label": "gas"},
            "heat": {"linestyle": "dashed", "color": "#C44E52", "label": "heat"},
        }

        # --- Layered left-to-right layout (process depth = longest path) ------
        dg = nx.DiGraph()
        dg.add_nodes_from(graph.nodes)
        for edge in graph.edges:
            dg.add_edge(edge.src, edge.dst)

        # Longest-path depth from any source; robust against cycles.
        depth: dict[str, int] = {n: 0 for n in dg.nodes}
        try:
            for n in nx.topological_sort(dg):
                for succ in dg.successors(n):
                    depth[succ] = max(depth[succ], depth[n] + 1)
        except nx.NetworkXUnfeasible:
            depth = {n: 0 for n in dg.nodes}  # cyclic: fall back to single column

        # Group nodes by depth (column) and assign vertical slots.
        columns: dict[int, list[str]] = {}
        for nid in graph.nodes:
            columns.setdefault(depth[nid], []).append(nid)

        def _key_info(ctype: str, p: dict[str, Any]) -> list[str]:
            """Most relevant sizing/rating values per component type."""

            def num(key, fmt="{:g}"):
                v = p.get(key)
                return fmt.format(v) if isinstance(v, (int, float)) else None

            lines: list[str] = []
            if ctype == "digester":
                if num("V_liq"):
                    lines.append(f"V_liq {num('V_liq')} m³")
                if num("V_gas"):
                    lines.append(f"V_gas {num('V_gas')} m³")
                if isinstance(p.get("T_ad"), (int, float)):
                    lines.append(f"T {p['T_ad'] - 273.15:.0f} °C")
            elif ctype == "storage":
                if num("capacity_m3"):
                    lines.append(f"V_gas {num('capacity_m3')} m³")
            elif ctype == "chp":
                if num("P_el_nom"):
                    lines.append(f"P_el {num('P_el_nom')} kW")
                if num("eta_el"):
                    lines.append(f"η_el {float(p['eta_el']) * 100:.0f} %")
            elif ctype == "upgrading":
                if num("capacity_m3h"):
                    lines.append(f"{num('capacity_m3h')} m³/h")
            elif ctype == "separator":
                if num("separation_efficiency"):
                    lines.append(f"η {float(p['separation_efficiency']) * 100:.0f} %")
            elif ctype == "flare":
                if num("destruction_efficiency"):
                    lines.append(f"η {float(p['destruction_efficiency']) * 100:.0f} %")
            elif ctype == "heating":
                if isinstance(p.get("target_temperature"), (int, float)):
                    t = p["target_temperature"]
                    t = t - 273.15 if t > 200 else t
                    lines.append(f"T {t:.0f} °C")
            return lines

        # Per-node info line; box width adapts to the longest text shown.
        infos: dict[str, str] = {
            nid: "  ·  ".join(_key_info(graph.nodes[nid].ctype, graph.nodes[nid].params)) for nid in graph.nodes
        }
        longest = max([len(infos[n]) for n in infos] + [len(n) for n in graph.nodes] + [10])
        box_h = 1.3
        box_w = max(3.2, 0.095 * longest + 1.0)

        # --- Layered left-to-right layout (columns by process depth) ----------
        x_gap, y_gap = box_w + 1.8, 1.9
        pos: dict[str, tuple] = {}
        max_rows = max((len(c) for c in columns.values()), default=1)
        for col, nids in sorted(columns.items()):
            n_rows = len(nids)
            offset = (max_rows - n_rows) / 2.0
            for row, nid in enumerate(sorted(nids)):
                pos[nid] = (col * x_gap, -(row + offset) * y_gap)

        n_cols = max(columns) + 1 if columns else 1
        fig, ax = plt.subplots(figsize=(max(7, n_cols * (box_w + 1.8)), max(4, max_rows * 1.9)))

        for nid, (x, y) in pos.items():
            node = graph.nodes[nid]
            ctype = node.ctype
            box = FancyBboxPatch(
                (x - box_w / 2, y - box_h / 2),
                box_w,
                box_h,
                boxstyle="round,pad=0.03,rounding_size=0.15",
                linewidth=1.4,
                edgecolor="#34495E",
                facecolor="#ECF3FB",
                zorder=2,
            )
            ax.add_patch(box)
            ax.text(x, y + box_h * 0.22, nid, ha="center", va="center", fontsize=9, fontweight="bold", zorder=3)
            ax.text(
                x,
                y + box_h * 0.02,
                class_names.get(ctype, ctype),
                ha="center",
                va="center",
                fontsize=7.5,
                color="#555555",
                zorder=3,
            )
            info = infos[nid]
            if info:
                ax.text(
                    x,
                    y - box_h * 0.28,
                    info,
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="#1A5276",
                    zorder=3,
                )

        def _border(cx, cy, tx, ty):
            """Point on the box border of (cx, cy) facing (tx, ty)."""
            dx, dy = tx - cx, ty - cy
            if dx == 0 and dy == 0:
                return cx, cy
            sx = (box_w / 2 + 0.06) / abs(dx) if dx else float("inf")
            sy = (box_h / 2 + 0.06) / abs(dy) if dy else float("inf")
            t = min(sx, sy)
            return cx + dx * t, cy + dy * t

        used_styles: dict[str, dict] = {}
        # Bow gas edges up and heat edges down so parallel liquid/gas/heat
        # connections separate; edges spanning intermediate columns bow more
        # (like Mermaid) to clear the boxes in between.
        bow_dir = {"liquid": 0.0, "gas": 1.0, "heat": -1.0}
        for edge in graph.edges:
            if edge.src not in pos or edge.dst not in pos:
                continue
            style = edge_styles.get(edge.etype, edge_styles["liquid"])
            used_styles[edge.etype] = style
            cx0, cy0 = pos[edge.src]
            cx1, cy1 = pos[edge.dst]
            # Start/end on the box borders so the arrow head stays visible.
            x0, y0 = _border(cx0, cy0, cx1, cy1)
            x1, y1 = _border(cx1, cy1, cx0, cy0)
            span = abs(depth[edge.dst] - depth[edge.src])
            direction = bow_dir.get(edge.etype, 0.0)
            chord_x = abs(cx1 - cx0)
            if chord_x < 0.1:  # same column (e.g. cyclic) -> sidestep
                rad = 0.3
            elif direction == 0.0 and span <= 1:  # adjacent liquid -> straight
                rad = 0.0
            else:
                # Fixed apex offset (data units) regardless of span, so long
                # multi-column edges bow gently instead of dipping far down.
                apex = 0.8 if span <= 1 else 1.2
                rad = direction * 2 * apex / chord_x
            conn = f"arc3,rad={rad}"
            # Body line in the medium's linestyle, but without a head, so a
            # dotted gas line does not produce a dotted (broken) arrow head.
            ax.add_patch(
                FancyArrowPatch(
                    (x0, y0),
                    (x1, y1),
                    connectionstyle=conn,
                    arrowstyle="-",
                    linewidth=1.6,
                    linestyle=style["linestyle"],
                    color=style["color"],
                    shrinkA=1,
                    shrinkB=2,
                    zorder=1,
                )
            )
            # Solid filled arrow head only (linewidth 0 -> no body line drawn).
            ax.add_patch(
                FancyArrowPatch(
                    (x0, y0),
                    (x1, y1),
                    connectionstyle=conn,
                    arrowstyle="-|>,head_length=5,head_width=3",
                    mutation_scale=2.4,
                    linewidth=0,
                    color=style["color"],
                    shrinkA=1,
                    shrinkB=2,
                    zorder=1,
                )
            )

        legend = [
            Line2D([0], [0], color=s["color"], linestyle=s["linestyle"], lw=1.6, label=s["label"])
            for et, s in edge_styles.items()
            if et in used_styles
        ]
        if legend:
            ax.legend(handles=legend, loc="lower right", fontsize=8, title="connection", framealpha=0.9)
        if title:
            ax.set_title(title)
        ax.axis("off")

        # Tight content bounding box: boxes plus room for the edge bows
        # (gas bows below, heat above). Sizing the figure to this box keeps the
        # image height matched to the graph instead of leaving large gaps.
        xs = [x for x, _ in pos.values()]
        ys = [y for _, y in pos.values()]
        pad = 0.5
        x_lo, x_hi = min(xs) - box_w / 2 - pad, max(xs) + box_w / 2 + pad
        y_lo = min(ys) - box_h / 2 - (1.4 if "gas" in used_styles else 0.0) - pad
        y_hi = max(ys) + box_h / 2 + (1.4 if "heat" in used_styles else 0.0) + pad
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(y_lo, y_hi)
        ax.set_aspect("equal", adjustable="box")

        scale = 0.55
        fig.set_size_inches((x_hi - x_lo) * scale, (y_hi - y_lo) * scale)
        fig.tight_layout()

        if output_path is None:
            from pathlib import Path

            # The working directory, not the package location: an installed
            # package lives in site-packages, which is no place for results.
            out_dir = Path.cwd() / "output"
            out_dir.mkdir(exist_ok=True)
            safe_name = self.plant_name.replace(" ", "_")
            output_path = str(out_dir / f"{safe_name}_graph.png")

        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return output_path

    def to_json(self, filepath: str) -> None:
        """
        Save plant configuration to JSON file.

        Args:
            filepath (str): Path to JSON file.
        """
        config = {
            "plant_name": self.plant_name,
            "T_ambient": self.T_ambient,
            "T_ground": self.T_ground,
            "energy_produced_kwh": self.energy_produced_kwh,
            "energy_consumed_kwh": self.energy_consumed_kwh,
            "substrate_use": self.substrate_use,
            "simulation_time": self.simulation_time,
            # ``component_class`` names the concrete class. The component type
            # alone cannot: all three sensor classes share ``sensor``.
            "components": [{**comp.to_dict(), "component_class": type(comp).__name__} for comp in self.components.values()],
            "connections": [conn.to_dict() for conn in self.connections],
        }

        with open(filepath, "w") as f:
            json.dump(config, f, indent=2)

        print(f"Plant configuration saved to {filepath}")

    @classmethod
    def from_json(cls, filepath: str, feedstock: Feedstock | None = None) -> BiogasPlant:
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
        with open(filepath) as f:
            config = json.load(f)

        plant = cls(
            config.get("plant_name", "Biogas Plant"),
            T_ambient=config.get("T_ambient", 288.15),
            T_ground=config.get("T_ground", 283.15),
        )
        plant.simulation_time = config.get("simulation_time", 0.0)
        plant.energy_produced_kwh = float(config.get("energy_produced_kwh", 0.0))
        plant.energy_consumed_kwh = float(config.get("energy_consumed_kwh", 0.0))
        plant.substrate_use = {
            sid: {"mass_t": float(entry["mass_t"]), "cost": float(entry["cost"])}
            for sid, entry in config.get("substrate_use", {}).items()
        }

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
            ComponentType.MIXER: "Mixer",
            ComponentType.PUMP: "Pump",
            ComponentType.FEEDER: "Feeder",
            ComponentType.SUBSTRATE_STORAGE: "SubstrateStorage",
        }
        registered = registry.get_registered_components()

        for comp_config in config.get("components", []):
            comp_type = ComponentType(comp_config["component_type"])
            # Files written by this version name the class; older files carry
            # only the type, which is unambiguous for everything but sensors.
            class_name = comp_config.get("component_class") or type_to_registry_key.get(comp_type)

            if comp_type == ComponentType.DIGESTER:
                if feedstock is None:
                    raise ValueError("Feedstock required for loading plant with digesters")
                component = Digester.from_dict(comp_config, feedstock)
            elif class_name is None:
                raise ValueError(
                    f"Cannot load component '{comp_config.get('component_id')}' of type '{comp_type.value}': "
                    "the file names no component_class and the type alone does not identify the class."
                )
            elif class_name not in registered:
                raise ValueError(f"No registered class '{class_name}' for component type: {comp_type}")
            else:
                component = registered[class_name].from_dict(comp_config)
                # The feedstock is not serializable; a heating system needs it
                # for the process-heat term, so hand it back in like a digester.
                if comp_type == ComponentType.HEATING and feedstock is not None:
                    component.feedstock = feedstock

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
