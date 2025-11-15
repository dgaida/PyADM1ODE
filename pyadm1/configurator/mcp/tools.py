# pyadm1/configurator/mcp/tools.py
"""
MCP Tools for Biogas Plant Configuration

This module provides the core tool functions exposed through the FastMCP server
for LLM-driven biogas plant modeling. Each tool corresponds to a specific action
in the plant building workflow.

Tools are organized in a logical sequence:
1. Plant creation (create_biogas_plant)
2. Component addition (add_digester, add_chp, add_heating)
3. Component connection (add_connection)
4. Plant initialization (initialize_plant)
5. Simulation (simulate_plant)
6. Status and export (get_plant_status, export_plant_config)
7. Management (list_plants, delete_plant)
"""

import traceback
from typing import Dict, Any, List, Optional
from pathlib import Path

from pyadm1.configurator.plant_builder import BiogasPlant
from pyadm1.components.energy.flare import Flare
from pyadm1.configurator.connection_manager import Connection
from pyadm1.substrates.feedstock import Feedstock
from pyadm1.configurator.plant_configurator import PlantConfigurator


class PlantRegistry:
    """
    Registry for managing multiple biogas plant instances.

    This class maintains a collection of plant instances along with their
    associated feedstocks, configurators, and metadata, enabling stateful
    plant modeling across multiple tool calls.

    Attributes:
        plants: Dictionary mapping plant IDs to BiogasPlant instances
        feedstocks: Dictionary mapping plant IDs to Feedstock instances
        configurators: Dictionary mapping plant IDs to PlantConfigurator instances
        metadata: Dictionary mapping plant IDs to metadata dictionaries
    """

    def __init__(self) -> None:
        """Initialize empty registries for plants, feedstocks, configurators, and metadata."""
        self.plants: Dict[str, BiogasPlant] = {}
        self.feedstocks: Dict[str, Feedstock] = {}
        self.configurators: Dict[str, PlantConfigurator] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}

    def create_plant(self, plant_id: str, **kwargs: Any) -> str:
        """
        Create a new plant instance and register it.

        Args:
            plant_id: Unique identifier for the plant
            **kwargs: Additional configuration parameters
                feeding_freq: Feeding frequency in hours (default: 48)

        Returns:
            The plant_id that was created

        Raises:
            ValueError: If plant_id already exists

        Example:
            >>> registry = PlantRegistry()
            >>> plant_id = registry.create_plant("MyPlant", feeding_freq=48)
        """
        if plant_id in self.plants:
            raise ValueError(f"Plant '{plant_id}' already exists")

        plant = BiogasPlant(plant_id)
        feedstock = Feedstock(feeding_freq=kwargs.get("feeding_freq", 48))
        configurator = PlantConfigurator(plant, feedstock)

        self.plants[plant_id] = plant
        self.feedstocks[plant_id] = feedstock
        self.configurators[plant_id] = configurator
        self.metadata[plant_id] = {
            "created": True,
            "initialized": False,
            "components": {},
            "connections": [],
            "simulation_runs": 0,
        }

        return plant_id

    def get_plant(self, plant_id: str) -> BiogasPlant:
        """
        Retrieve a plant instance by ID.

        Args:
            plant_id: Plant identifier

        Returns:
            BiogasPlant instance

        Raises:
            ValueError: If plant_id not found
        """
        if plant_id not in self.plants:
            raise ValueError(f"Plant '{plant_id}' not found")
        return self.plants[plant_id]

    def get_configurator(self, plant_id: str) -> PlantConfigurator:
        """
        Retrieve the configurator for a plant.

        Args:
            plant_id: Plant identifier

        Returns:
            PlantConfigurator instance

        Raises:
            ValueError: If plant_id not found
        """
        if plant_id not in self.configurators:
            raise ValueError(f"Configurator for plant '{plant_id}' not found")
        return self.configurators[plant_id]

    def get_feedstock(self, plant_id: str) -> Feedstock:
        """
        Retrieve the feedstock associated with a plant.

        Args:
            plant_id: Plant identifier

        Returns:
            Feedstock instance

        Raises:
            ValueError: If plant_id not found
        """
        if plant_id not in self.feedstocks:
            raise ValueError(f"Feedstock for plant '{plant_id}' not found")
        return self.feedstocks[plant_id]

    def get_metadata(self, plant_id: str) -> Dict[str, Any]:
        """
        Retrieve metadata for a plant.

        Args:
            plant_id: Plant identifier

        Returns:
            Dictionary containing plant metadata

        Raises:
            ValueError: If plant_id not found
        """
        if plant_id not in self.metadata:
            raise ValueError(f"Metadata for plant '{plant_id}' not found")
        return self.metadata[plant_id]

    def list_plants(self) -> List[str]:
        """
        Get list of all registered plant IDs.

        Returns:
            List of plant identifier strings
        """
        return list(self.plants.keys())

    def delete_plant(self, plant_id: str) -> None:
        """
        Remove a plant and its associated data from the registry.

        Args:
            plant_id: Plant identifier to delete
        """
        if plant_id in self.plants:
            del self.plants[plant_id]
        if plant_id in self.feedstocks:
            del self.feedstocks[plant_id]
        if plant_id in self.configurators:
            del self.configurators[plant_id]
        if plant_id in self.metadata:
            del self.metadata[plant_id]


# Global registry instance shared across all tool calls
_registry = PlantRegistry()


def get_registry() -> PlantRegistry:
    """
    Get the global plant registry instance.

    Returns:
        The singleton PlantRegistry instance
    """
    return _registry


def create_biogas_plant(plant_id: str, description: Optional[str] = None, feeding_freq: int = 48) -> str:
    """
    Create a new biogas plant model.

    This is the first step in building a plant. After creating the plant,
    you can add components (digesters, CHP units, etc.) and connect them.

    Args:
        plant_id: Unique identifier for the plant (e.g., "FarmAB_Plant")
        description: Optional description of the plant
        feeding_freq: Feeding frequency in hours (default: 48)

    Returns:
        Detailed status message including next steps

    Example:
        >>> result = create_biogas_plant("MyFarm", "Two-stage digestion with CHP", 48)
    """
    try:
        registry = get_registry()
        registry.create_plant(plant_id, feeding_freq=feeding_freq)

        metadata = registry.get_metadata(plant_id)
        if description:
            metadata["description"] = description

        return (
            f"✓ Successfully created biogas plant '{plant_id}'\n\n"
            f"Plant Status:\n"
            f"  - ID: {plant_id}\n"
            f"  - Description: {description or 'No description provided'}\n"
            f"  - Feeding frequency: {feeding_freq} hours\n"
            f"  - Components: 0\n"
            f"  - Connections: 0\n\n"
            f"Next steps:\n"
            f"  1. Add components using add_digester(), add_chp(), or add_heating()\n"
            f"  2. Connect components using add_connection()\n"
            f"  3. Initialize plant using initialize_plant()\n"
            f"  4. Run simulation using simulate_plant()"
        )
    except Exception as e:
        return f"✗ Error creating plant: {str(e)}\n{traceback.format_exc()}"


def add_digester(
    plant_id: str,
    digester_id: str,
    V_liq: float = 1977.0,
    V_gas: float = 304.0,
    T_ad: float = 308.15,
    name: Optional[str] = None,
    load_initial_state: bool = True,
) -> str:
    """
    Add a digester component to the plant.

    A digester is the main biological reactor where anaerobic digestion occurs.
    Multiple digesters can be added and connected in series or parallel.

    Args:
        plant_id: Plant identifier
        digester_id: Unique ID for this digester (e.g., "main_digester")
        V_liq: Liquid volume in m³ (default: 1977.0)
        V_gas: Gas volume in m³ (default: 304.0)
        T_ad: Operating temperature in K (default: 308.15 = 35°C)
        name: Human-readable name (optional)
        load_initial_state: Load default initial state (default: True)

    Returns:
        Detailed status with component information and next steps
    """
    try:
        registry = get_registry()
        metadata = registry.get_metadata(plant_id)
        configurator = registry.get_configurator(plant_id)

        # Create digester
        _, state_info = configurator.add_digester(
            digester_id=digester_id, V_liq=V_liq, V_gas=V_gas, T_ad=T_ad, name=name or digester_id
        )

        storage_id = f"{digester_id}_storage"

        # Add to metadata
        metadata["components"][storage_id] = {
            "type": "gas_storage",
            "capacity_m3": max(50.0, V_gas),
            "name": f"{digester_id}_storage",
        }

        # Update metadata
        metadata["components"][digester_id] = {
            "type": "digester",
            "V_liq": V_liq,
            "V_gas": V_gas,
            "T_ad": T_ad,
            "name": name or digester_id,
        }

        # Calculate retention time
        HRT = V_liq / 25.0  # Assuming default flow of 25 m³/d

        component_count = len(metadata["components"])

        return (
            f"✓ Successfully added digester '{digester_id}' to plant '{plant_id}'\n\n"
            f"Digester Configuration:\n"
            f"  - ID: {digester_id}\n"
            f"  - Name: {name or digester_id}\n"
            f"  - Liquid volume: {V_liq:.1f} m³\n"
            f"  - Gas volume: {V_gas:.1f} m³\n"
            f"  - Total volume: {V_liq + V_gas:.1f} m³\n"
            f"  - Temperature: {T_ad:.2f} K ({T_ad - 273.15:.1f}°C)\n"
            f"  - Hydraulic retention time: ~{HRT:.1f} days (at 25 m³/d)\n"
            f"{state_info}\n"
            f"Plant Status:\n"
            f"  - Total components: {component_count}\n"
            f"  - Digesters: {sum(1 for c in metadata['components'].values() if c['type'] == 'digester')}\n\n"
            f"Next steps:\n"
            f"  - Add more digesters for multi-stage configuration\n"
            f"  - Add CHP unit with add_chp()\n"
            f"  - Add heating system with add_heating()\n"
            f"  - Connect components with add_connection()"
        )
    except Exception as e:
        return f"✗ Error adding digester: {str(e)}\n{traceback.format_exc()}"


def add_chp(
    plant_id: str, chp_id: str, P_el_nom: float = 500.0, eta_el: float = 0.40, eta_th: float = 0.45, name: Optional[str] = None
) -> str:
    """
    Add a Combined Heat and Power (CHP) unit to the plant.

    The CHP converts biogas to electricity and heat. It should be connected
    to digesters via gas connections.

    Args:
        plant_id: Plant identifier
        chp_id: Unique ID for this CHP unit (e.g., "chp_main")
        P_el_nom: Nominal electrical power in kW (default: 500.0)
        eta_el: Electrical efficiency 0-1 (default: 0.40 = 40%)
        eta_th: Thermal efficiency 0-1 (default: 0.45 = 45%)
        name: Human-readable name (optional)

    Returns:
        Detailed status with CHP specifications and next steps
    """
    try:
        registry = get_registry()
        metadata = registry.get_metadata(plant_id)
        configurator = registry.get_configurator(plant_id)

        configurator.add_chp(chp_id=chp_id, P_el_nom=P_el_nom, eta_el=eta_el, eta_th=eta_th, name=name or chp_id)

        flare_id = f"{chp_id}_flare"

        metadata["components"][flare_id] = {
            "type": "flare",
            "name": f"{chp_id}_flare",
        }

        # Update metadata
        metadata["components"][chp_id] = {
            "type": "chp",
            "P_el_nom": P_el_nom,
            "eta_el": eta_el,
            "eta_th": eta_th,
            "name": name or chp_id,
        }

        # Calculate performance metrics
        P_th_nom = P_el_nom * eta_th / eta_el
        P_total = P_el_nom / eta_el
        eta_total = eta_el + eta_th

        # Estimate biogas demand (assuming 10 kWh/m³ CH4, 60% CH4 content)
        biogas_demand = P_total * 24 / (10 * 0.6)  # m³/d at full load

        component_count = len(metadata["components"])

        return (
            f"✓ Successfully added CHP unit '{chp_id}' to plant '{plant_id}'\n\n"
            f"CHP Configuration:\n"
            f"  - ID: {chp_id}\n"
            f"  - Name: {name or chp_id}\n"
            f"  - Electrical power: {P_el_nom:.1f} kW ({P_el_nom * 24:.0f} kWh/day)\n"
            f"  - Thermal power: {P_th_nom:.1f} kW ({P_th_nom * 24:.0f} kWh/day)\n"
            f"  - Total input: {P_total:.1f} kW\n"
            f"  - Electrical efficiency: {eta_el * 100:.1f}%\n"
            f"  - Thermal efficiency: {eta_th * 100:.1f}%\n"
            f"  - Total efficiency: {eta_total * 100:.1f}%\n"
            f"  - Biogas demand (full load): ~{biogas_demand:.1f} m³/day\n\n"
            f"Plant Status:\n"
            f"  - Total components: {component_count}\n"
            f"  - CHP units: {sum(1 for c in metadata['components'].values() if c['type'] == 'chp')}\n\n"
            f"Next steps:\n"
            f"  - Connect digesters to CHP with add_connection(from='digester_id', to='{chp_id}', type='gas')\n"
            f"  - Add heating system to use waste heat\n"
            f"  - Connect CHP to heating with add_connection(from='{chp_id}', to='heating_id', type='heat')"
        )
    except Exception as e:
        return f"✗ Error adding CHP: {str(e)}\n{traceback.format_exc()}"


def add_heating(
    plant_id: str,
    heating_id: str,
    target_temperature: float = 308.15,
    heat_loss_coefficient: float = 0.5,
    name: Optional[str] = None,
) -> str:
    """
    Add a heating system to maintain digester temperature.

    The heating system uses waste heat from CHP and/or auxiliary heating
    to maintain the target digester temperature.

    Args:
        plant_id: Plant identifier
        heating_id: Unique ID for heating system (e.g., "heating_main")
        target_temperature: Target temperature in K (default: 308.15 = 35°C)
        heat_loss_coefficient: Heat loss in kW/K (default: 0.5)
        name: Human-readable name (optional)

    Returns:
        Detailed status with heating specifications
    """
    try:
        registry = get_registry()
        metadata = registry.get_metadata(plant_id)
        configurator = registry.get_configurator(plant_id)

        configurator.add_heating(
            heating_id=heating_id,
            target_temperature=target_temperature,
            heat_loss_coefficient=heat_loss_coefficient,
            name=name or heating_id,
        )

        # Update metadata
        metadata["components"][heating_id] = {
            "type": "heating",
            "target_temperature": target_temperature,
            "heat_loss_coefficient": heat_loss_coefficient,
            "name": name or heating_id,
        }

        # Calculate heat demand (assuming 15°C ambient)
        T_ambient = 288.15
        heat_demand = heat_loss_coefficient * (target_temperature - T_ambient)

        component_count = len(metadata["components"])

        return (
            f"✓ Successfully added heating system '{heating_id}' to plant '{plant_id}'\n\n"
            f"Heating Configuration:\n"
            f"  - ID: {heating_id}\n"
            f"  - Name: {name or heating_id}\n"
            f"  - Target temperature: {target_temperature:.2f} K ({target_temperature - 273.15:.1f}°C)\n"
            f"  - Heat loss coefficient: {heat_loss_coefficient:.2f} kW/K\n"
            f"  - Estimated heat demand: ~{heat_demand:.1f} kW (at 15°C ambient)\n"
            f"  - Daily heat demand: ~{heat_demand * 24:.0f} kWh/day\n\n"
            f"Plant Status:\n"
            f"  - Total components: {component_count}\n"
            f"  - Heating systems: {sum(1 for c in metadata['components'].values() if c['type'] == 'heating')}\n\n"
            f"Next steps:\n"
            f"  - Connect CHP to heating with add_connection(from='chp_id', to='{heating_id}', type='heat')\n"
            f"  - Initialize and simulate plant"
        )
    except Exception as e:
        return f"✗ Error adding heating: {str(e)}\n{traceback.format_exc()}"


def add_connection(plant_id: str, from_component: str, to_component: str, connection_type: str = "default") -> str:
    """
    Connect two components in the plant.

    Creates a directed connection from one component to another, defining
    how material or energy flows between them.

    Args:
        plant_id: Plant identifier
        from_component: Source component ID
        to_component: Target component ID
        connection_type: Type of connection:
            - 'liquid': Liquid flow (digestate between digesters)
            - 'gas': Biogas flow (digester to CHP)
            - 'heat': Heat flow (CHP to heating)
            - 'power': Electrical power
            - 'default': Generic connection

    Returns:
        Detailed connection information and topology overview
    """
    try:
        registry = get_registry()
        plant = registry.get_plant(plant_id)
        metadata = registry.get_metadata(plant_id)

        # Verify components exist
        if from_component not in metadata["components"]:
            return f"✗ Error: Source component '{from_component}' not found in plant"
        if to_component not in metadata["components"]:
            return f"✗ Error: Target component '{to_component}' not found in plant"

        connection = Connection(from_component, to_component, connection_type)
        plant.add_connection(connection)

        # Update metadata
        metadata["connections"].append({"from": from_component, "to": to_component, "type": connection_type})

        # Build topology view
        from_comp = metadata["components"][from_component]
        to_comp = metadata["components"][to_component]

        # Count connections by type
        connection_types: Dict[str, int] = {}
        for conn in metadata["connections"]:
            conn_type = conn["type"]
            connection_types[conn_type] = connection_types.get(conn_type, 0) + 1

        topology_view = _build_topology_view(metadata)

        return (
            f"✓ Successfully connected '{from_component}' → '{to_component}'\n\n"
            f"Connection Details:\n"
            f"  - From: {from_comp['name']} ({from_comp['type']})\n"
            f"  - To: {to_comp['name']} ({to_comp['type']})\n"
            f"  - Type: {connection_type}\n\n"
            f"Plant Topology:\n"
            f"  - Total connections: {len(metadata['connections'])}\n"
            f"  - By type: {', '.join(f'{k}: {v}' for k, v in connection_types.items())}\n\n"
            f"Connection Flow:\n"
            f"{topology_view}\n"
            f"Next steps:\n"
            f"  - Add more connections to complete the plant topology\n"
            f"  - Use get_plant_status() to review the complete plant\n"
            f"  - Initialize plant with initialize_plant()\n"
            f"  - Run simulation with simulate_plant()"
        )
    except Exception as e:
        return f"✗ Error adding connection: {str(e)}\n{traceback.format_exc()}"


def initialize_plant(plant_id: str) -> str:
    """
    Initialize the plant before simulation.

    This prepares all components for simulation and validates the configuration.
    Must be called before running any simulations.

    Args:
        plant_id: Plant identifier

    Returns:
        Initialization status and validation results
    """
    try:
        registry = get_registry()
        plant = registry.get_plant(plant_id)
        metadata = registry.get_metadata(plant_id)

        # Initialize plant
        plant.initialize()

        metadata["initialized"] = True

        # Validate configuration
        warnings: List[str] = []

        # ==========================================================
        # AUTOMATIC GAS INFRASTRUCTURE COMPLETION
        # ==========================================================
        # digesters = [cid for cid, cfg in metadata["components"].items() if cfg["type"] == "digester"]
        storages = [cid for cid, cfg in metadata["components"].items() if cfg["type"] == "gas_storage"]
        chps = [cid for cid, cfg in metadata["components"].items() if cfg["type"] == "chp"]

        # ----------------------------------------------------------
        # If no CHP exists → add plant-level flare
        # ----------------------------------------------------------
        if len(chps) == 0:
            plant_flare_id = f"{plant_id}_flare"
            if plant_flare_id not in metadata["components"]:
                flare = Flare(component_id=plant_flare_id, name=f"{plant_id}_flare")
                plant.add_component(flare)
                metadata["components"][plant_flare_id] = {
                    "type": "flare",
                    "name": f"{plant_id}_flare",
                }
                chp_flares = [plant_flare_id]
            else:
                chp_flares = [plant_flare_id]
        else:
            # Flares already created in add_chp()
            chp_flares = [f"{cid}_flare" for cid in chps]

        # ----------------------------------------------------------
        # Connect all gas storages → CHP or → global flare
        # ----------------------------------------------------------
        for storage in storages:
            if len(chps) > 0:
                # TODO: all gas_storages are connected to all chps. if one storage is connected
                #  to 2 chps, there is no 50/50 split, but the amount of gas is doubled.
                for chp in chps:
                    conn = Connection(storage, chp, "gas")
                    plant.add_connection(conn)
                    metadata["connections"].append({"from": storage, "to": chp, "type": "gas"})
            else:
                # connect storage → plant flare
                for flare_id in chp_flares:
                    conn = Connection(storage, flare_id, "gas")
                    plant.add_connection(conn)
                    metadata["connections"].append({"from": storage, "to": flare_id, "type": "gas"})

        # Check for digesters
        digester_count = sum(1 for c in metadata["components"].values() if c["type"] == "digester")
        if digester_count == 0:
            warnings.append("No digesters found - plant needs at least one digester")

        # Check for gas connections to CHP
        chp_count = sum(1 for c in metadata["components"].values() if c["type"] == "chp")
        if chp_count > 0:
            gas_to_chp = sum(
                1
                for c in metadata["connections"]
                if c["type"] == "gas" and metadata["components"].get(c["to"], {}).get("type") == "chp"
            )
            if gas_to_chp == 0:
                warnings.append("CHP units present but no gas connections from digesters")

        warning_text = ""
        if warnings:
            warning_text = "\n⚠ Warnings:\n" + "\n".join(f"  - {w}" for w in warnings) + "\n"

        return (
            f"✓ Successfully initialized plant '{plant_id}'\n\n"
            f"Plant Configuration:\n"
            f"  - Components: {len(metadata['components'])}\n"
            f"    • Digesters: {digester_count}\n"
            f"    • CHP units: {chp_count}\n"
            f"    • Heating systems: {sum(1 for c in metadata['components'].values() if c['type'] == 'heating')}\n"
            f"  - Connections: {len(metadata['connections'])}\n"
            f"  - Status: Ready for simulation\n"
            f"{warning_text}\n"
            f"Next steps:\n"
            f"  - Run simulation with simulate_plant()\n"
            f"  - View plant summary with get_plant_status()"
        )
    except Exception as e:
        return f"✗ Error initializing plant: {str(e)}\n{traceback.format_exc()}"


def simulate_plant(plant_id: str, duration: float = 10.0, dt: float = 0.04167, save_interval: float = 1.0) -> str:
    """
    Run a simulation of the biogas plant.

    Simulates the plant operation for the specified duration and returns
    key performance indicators.

    Args:
        plant_id: Plant identifier
        duration: Simulation duration in days (default: 10.0)
        dt: Time step in days (default: 0.04167 = 1 hour)
        save_interval: How often to save results in days (default: 1.0 = daily)

    Returns:
        Detailed simulation results and performance metrics
    """
    try:
        registry = get_registry()
        plant = registry.get_plant(plant_id)
        metadata = registry.get_metadata(plant_id)

        if not metadata["initialized"]:
            return (
                f"✗ Error: Plant '{plant_id}' must be initialized before simulation\n"
                f"Please call initialize_plant('{plant_id}') first"
            )

        # Run simulation
        results = plant.simulate(duration=duration, dt=dt, save_interval=save_interval)

        metadata["simulation_runs"] += 1
        metadata["last_simulation"] = {
            "duration": duration,
            "dt": dt,
            "save_interval": save_interval,
            "result_count": len(results),
        }

        # Build results summary
        if results:
            final_result = results[-1]
            summary_lines = [
                f"✓ Successfully simulated plant '{plant_id}' for {duration:.1f} days\n",
                "Simulation Configuration:",
                f"  - Duration: {duration:.1f} days",
                f"  - Time step: {dt * 24:.1f} hours",
                f"  - Save interval: {save_interval:.1f} days",
                f"  - Result snapshots: {len(results)}",
                f"  - Simulation time: {final_result['time']:.2f} days\n",
                "Performance Metrics:\n",
            ]

            # Process each component
            for comp_id, comp_results in final_result["components"].items():
                comp_meta = metadata["components"].get(comp_id, {})
                comp_type = comp_meta.get("type", "unknown")
                comp_name = comp_meta.get("name", comp_id)

                summary_lines.append(f"  {comp_name} ({comp_type}):")

                if comp_type == "digester":
                    q_gas = comp_results.get("Q_gas", 0)
                    q_ch4 = comp_results.get("Q_ch4", 0)
                    pH = comp_results.get("pH", 0)
                    vfa = comp_results.get("VFA", 0)
                    tac = comp_results.get("TAC", 0)

                    ch4_percent = (q_ch4 / q_gas * 100) if q_gas > 0 else 0

                    summary_lines.extend(
                        [
                            f"    - Biogas production: {q_gas:.1f} m³/day",
                            f"    - Methane production: {q_ch4:.1f} m³/day ({ch4_percent:.1f}% CH4)",
                            f"    - pH: {pH:.2f}",
                            f"    - VFA: {vfa:.2f} g/L",
                            f"    - TAC: {tac:.2f} g/L",
                            f"    - FOS/TAC: {(vfa / tac if tac > 0 else 0):.3f}",
                        ]
                    )

                elif comp_type == "chp":
                    p_el = comp_results.get("P_el", 0)
                    p_th = comp_results.get("P_th", 0)
                    q_gas_consumed = comp_results.get("Q_gas_consumed", 0)

                    summary_lines.extend(
                        [
                            f"    - Electrical power: {p_el:.1f} kW ({p_el * 24:.0f} kWh/day)",
                            f"    - Thermal power: {p_th:.1f} kW ({p_th * 24:.0f} kWh/day)",
                            f"    - Biogas consumption: {q_gas_consumed:.1f} m³/day",
                            f"    - Load factor: {(p_el / comp_meta.get('P_el_nom', 1) * 100):.1f}%",
                        ]
                    )

                elif comp_type == "heating":
                    q_heat = comp_results.get("Q_heat_supplied", 0)
                    p_th_used = comp_results.get("P_th_used", 0)
                    p_aux = comp_results.get("P_aux_heat", 0)

                    summary_lines.extend(
                        [
                            f"    - Heat supplied: {q_heat:.1f} kW",
                            f"    - CHP heat used: {p_th_used:.1f} kW",
                            f"    - Auxiliary heat: {p_aux:.1f} kW",
                            f"    - CHP coverage: {(p_th_used / q_heat * 100 if q_heat > 0 else 0):.1f}%",
                        ]
                    )

                summary_lines.append("")

            return "\n".join(summary_lines) + (
                "\nNext steps:\n"
                "  - Review results with get_plant_status()\n"
                "  - Export configuration with export_plant_config()\n"
                "  - Run longer simulation or adjust parameters"
            )
        else:
            return "✗ Simulation completed but no results generated"

    except Exception as e:
        return f"✗ Error during simulation: {str(e)}\n{traceback.format_exc()}"


def get_plant_status(plant_id: str) -> str:
    """
    Get comprehensive status of a biogas plant.

    Returns detailed information about the plant configuration, components,
    connections, and simulation history.

    Args:
        plant_id: Plant identifier

    Returns:
        Complete plant status report
    """
    try:
        registry = get_registry()
        metadata = registry.get_metadata(plant_id)

        lines = [
            "═══════════════════════════════════════════════════════",
            f"Biogas Plant Status: {plant_id}",
            "═══════════════════════════════════════════════════════\n",
        ]

        # Description
        if "description" in metadata:
            lines.append(f"Description: {metadata['description']}\n")

        # Overall status
        lines.extend(
            [
                "Status:",
                f"  - Initialized: {'Yes' if metadata['initialized'] else 'No'}",
                f"  - Simulation runs: {metadata['simulation_runs']}",
                f"  - Components: {len(metadata['components'])}",
                f"  - Connections: {len(metadata['connections'])}\n",
            ]
        )

        # Components detail
        if metadata["components"]:
            lines.append("Components:")
            for comp_id, comp_info in metadata["components"].items():
                comp_type = comp_info["type"]
                comp_name = comp_info["name"]

                lines.append(f"  • {comp_name} ({comp_id})")
                lines.append(f"    Type: {comp_type}")

                if comp_type == "digester":
                    lines.extend(
                        [
                            f"    Liquid volume: {comp_info['V_liq']:.1f} m³",
                            f"    Gas volume: {comp_info['V_gas']:.1f} m³",
                            f"    Temperature: {comp_info['T_ad'] - 273.15:.1f}°C",
                        ]
                    )
                elif comp_type == "chp":
                    lines.extend(
                        [
                            f"    Electrical power: {comp_info['P_el_nom']:.1f} kW",
                            f"    Electrical efficiency: {comp_info['eta_el'] * 100:.1f}%",
                            f"    Thermal efficiency: {comp_info['eta_th'] * 100:.1f}%",
                        ]
                    )
                elif comp_type == "heating":
                    lines.extend(
                        [
                            f"    Target temperature: {comp_info['target_temperature'] - 273.15:.1f}°C",
                            f"    Heat loss coefficient: {comp_info['heat_loss_coefficient']:.2f} kW/K",
                        ]
                    )
                lines.append("")

        # Connections
        if metadata["connections"]:
            lines.append("Connections:")
            for conn in metadata["connections"]:
                from_name = metadata["components"][conn["from"]]["name"]
                to_name = metadata["components"][conn["to"]]["name"]
                conn_type = conn["type"]
                lines.append(f"  {from_name} --[{conn_type}]--> {to_name}")
            lines.append("")

        # Last simulation
        if "last_simulation" in metadata:
            sim = metadata["last_simulation"]
            lines.extend(
                [
                    "Last Simulation:",
                    f"  - Duration: {sim['duration']:.1f} days",
                    f"  - Time step: {sim['dt'] * 24:.1f} hours",
                    f"  - Results saved: {sim['result_count']} snapshots\n",
                ]
            )

        # Recommendations
        lines.append("Recommendations:")
        if not metadata["initialized"]:
            lines.append("  ⚠ Initialize plant before simulation")
        if len(metadata["components"]) == 0:
            lines.append("  ⚠ Add components to the plant")
        if len(metadata["connections"]) == 0 and len(metadata["components"]) > 1:
            lines.append("  ⚠ Connect components to define process flow")
        if metadata["simulation_runs"] == 0 and metadata["initialized"]:
            lines.append("  → Ready to run simulation")
        if metadata["simulation_runs"] > 0:
            lines.append("  ✓ Plant has been simulated successfully")

        lines.append(f"\n{'═' * 55}")

        return "\n".join(lines)

    except Exception as e:
        return f"✗ Error getting plant status: {str(e)}\n{traceback.format_exc()}"


def export_plant_config(plant_id: str, filepath: Optional[str] = None) -> str:
    """
    Export plant configuration to JSON file.

    Saves the complete plant configuration including all components and
    connections for later reuse or documentation.

    Args:
        plant_id: Plant identifier
        filepath: Output file path (default: auto-generated)

    Returns:
        Export status and file location
    """
    try:
        registry = get_registry()
        plant = registry.get_plant(plant_id)

        if filepath is None:
            filepath = f"plant_config_{plant_id}.json"

        plant.to_json(filepath)

        # Get file size
        file_path = Path(filepath)
        file_size = file_path.stat().st_size if file_path.exists() else 0

        return (
            f"✓ Successfully exported plant configuration\n\n"
            f"Export Details:\n"
            f"  - Plant ID: {plant_id}\n"
            f"  - File: {filepath}\n"
            f"  - Size: {file_size:,} bytes\n\n"
            f"The configuration can be loaded later using load_plant_config()"
        )

    except Exception as e:
        return f"✗ Error exporting plant: {str(e)}\n{traceback.format_exc()}"


def list_plants() -> str:
    """
    List all biogas plants in the registry.

    Returns:
        Summary of all available plants
    """
    try:
        registry = get_registry()
        plant_ids = registry.list_plants()

        if not plant_ids:
            return "No plants found in registry.\n\n" "Create a new plant with create_biogas_plant()"

        lines = [f"Available Biogas Plants ({len(plant_ids)}):", "=" * 50]

        for plant_id in plant_ids:
            metadata = registry.get_metadata(plant_id)
            comp_count = len(metadata["components"])
            sim_runs = metadata["simulation_runs"]
            initialized = "✓" if metadata["initialized"] else "○"

            lines.append(f"\n{initialized} {plant_id}:" f"\n  - Components: {comp_count}" f"\n  - Simulations: {sim_runs}")

            if "description" in metadata:
                lines.append(f"  - Description: {metadata['description']}")

        lines.append("\n" + "=" * 50)

        return "\n".join(lines)

    except Exception as e:
        return f"✗ Error listing plants: {str(e)}"


def delete_plant(plant_id: str) -> str:
    """
    Delete a biogas plant from the registry.

    Args:
        plant_id: Plant identifier to delete

    Returns:
        Deletion confirmation
    """
    try:
        registry = get_registry()

        if plant_id not in registry.list_plants():
            return f"✗ Plant '{plant_id}' not found"

        registry.delete_plant(plant_id)

        return f"✓ Successfully deleted plant '{plant_id}'\n\n" f"Remaining plants: {len(registry.list_plants())}"

    except Exception as e:
        return f"✗ Error deleting plant: {str(e)}"


def _build_topology_view(metadata: Dict[str, Any]) -> str:
    """
    Build ASCII topology visualization of plant connections.

    Args:
        metadata: Plant metadata dictionary containing components and connections

    Returns:
        Formatted string showing plant topology
    """
    lines: List[str] = []
    components = metadata["components"]
    connections = metadata["connections"]

    # Group by component type
    digesters = [k for k, v in components.items() if v["type"] == "digester"]
    chps = [k for k, v in components.items() if v["type"] == "chp"]
    heating = [k for k, v in components.items() if v["type"] == "heating"]

    if digesters:
        lines.append("  Digesters:")
        for d in digesters:
            outgoing = [c for c in connections if c["from"] == d]
            if outgoing:
                for conn in outgoing:
                    lines.append(f"    {d} --[{conn['type']}]--> {conn['to']}")
            else:
                lines.append(f"    {d} (not connected)")

    if chps:
        lines.append("  CHP Units:")
        for c in chps:
            incoming = [cn for cn in connections if cn["to"] == c]
            outgoing = [cn for cn in connections if cn["from"] == c]
            if incoming or outgoing:
                lines.append(f"    {c}")
                for conn in incoming:
                    lines.append(f"      ← [{conn['type']}] {conn['from']}")
                for conn in outgoing:
                    lines.append(f"      → [{conn['type']}] {conn['to']}")

    if heating:
        lines.append("  Heating Systems:")
        for h in heating:
            incoming = [cn for cn in connections if cn["to"] == h]
            if incoming:
                for conn in incoming:
                    lines.append(f"    {h} ← [{conn['type']}] {conn['from']}")

    return "\n".join(lines) if lines else "  (No connections yet)"
