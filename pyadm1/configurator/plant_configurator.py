# pyadm1/configurator/plant_configurator.py
"""
High-level plant configuration helpers.

This module provides convenient methods for adding components to biogas plants
with sensible defaults and validation. These methods are used both by direct
API users and by the MCP server tools.
"""

from typing import Optional, Dict, Any
from pathlib import Path

from pyadm1.configurator.plant_builder import BiogasPlant
from pyadm1.components.biological.digester import Digester
from pyadm1.components.energy.chp import CHP
from pyadm1.components.energy.heating import HeatingSystem
from pyadm1.components.energy.gas_storage import GasStorage
from pyadm1.components.energy.flare import Flare
from pyadm1.configurator.connection_manager import Connection
from pyadm1.substrates.feedstock import Feedstock
from pyadm1.core.adm1 import get_state_zero_from_initial_state


class PlantConfigurator:
    """
    High-level configurator for building biogas plants.

    This class provides convenient methods for adding components with
    sensible defaults and automatic setup of common configurations.
    """

    def __init__(self, plant: BiogasPlant, feedstock: Feedstock):
        """
        Initialize configurator.

        Args:
            plant: BiogasPlant instance to configure
            feedstock: Feedstock instance for digesters
        """
        self.plant = plant
        self.feedstock = feedstock

    def add_digester(
        self,
        digester_id: str,
        V_liq: float = 1977.0,
        V_gas: float = 304.0,
        T_ad: float = 308.15,
        name: Optional[str] = None,
        load_initial_state: bool = True,
        initial_state_file: Optional[str] = None,
        Q_substrates: Optional[list] = None,
    ) -> (Digester, str):
        """
        Add a digester component to the plant.

        Automatically creates and connects a gas storage for the digester.

        Args:
            digester_id: Unique identifier for this digester
            V_liq: Liquid volume in m³ (default: 1977.0)
            V_gas: Gas volume in m³ (default: 304.0)
            T_ad: Operating temperature in K (default: 308.15 = 35°C)
            name: Human-readable name (optional)
            load_initial_state: Load default initial state (default: True)
            initial_state_file: Path to custom initial state CSV (optional)
            Q_substrates: Initial substrate feed rates [m³/d] (optional)

        Returns:
            Created Digester component

        Example:
            >>> config = PlantConfigurator(plant, feedstock)
            >>> digester = config.add_digester(
            ...     "main_digester",
            ...     V_liq=2000,
            ...     Q_substrates=[15, 10, 0, 0, 0, 0, 0, 0, 0, 0]
            ... )
        """
        # Create digester
        digester = Digester(
            component_id=digester_id, feedstock=self.feedstock, V_liq=V_liq, V_gas=V_gas, T_ad=T_ad, name=name or digester_id
        )

        # Initialize with state
        if load_initial_state:
            if initial_state_file:
                # Load from custom file
                adm1_state = get_state_zero_from_initial_state(initial_state_file)
                state_info = f"  - Initial state: Loaded from {initial_state_file}\n"
            else:
                # Load from default file
                try:
                    data_path = Path(__file__).parent.parent.parent / "data" / "initial_states"
                    default_file = data_path / "digester_initial8.csv"

                    if default_file.exists():
                        adm1_state = get_state_zero_from_initial_state(str(default_file))
                        state_info = f"  - Initial state: Loaded from {default_file.name}\n"
                    else:
                        adm1_state = None
                        state_info = "  - Initial state: Default initialization\n"
                except Exception as e:
                    adm1_state = None
                    state_info = f"  - Initial state: Default (error loading file: {str(e)})\n"

            # Set substrate feeds
            if Q_substrates is None:
                Q_substrates = [0] * 10

            digester.initialize({"adm1_state": adm1_state, "Q_substrates": Q_substrates})
        else:
            digester.initialize()
            state_info = "  - Initial state: Not initialized\n"

        # Add to plant
        self.plant.add_component(digester)

        # ----------------------------------------------------------
        # AUTOMATIC GAS STORAGE FOR DIGESTER
        # ----------------------------------------------------------

        storage_id = f"{digester_id}_storage"
        storage = GasStorage(
            component_id=storage_id,
            storage_type="membrane",
            capacity_m3=max(50.0, V_gas),
            name=f"{name or digester_id} Gas Storage",
        )
        self.plant.add_component(storage)

        # Connect: digester → storage
        self.connect(digester_id, storage_id, "gas")

        return digester, state_info

    def add_chp(
        self, chp_id: str, P_el_nom: float = 500.0, eta_el: float = 0.40, eta_th: float = 0.45, name: Optional[str] = None
    ) -> CHP:
        """
        Add a CHP unit to the plant.

        Args:
            chp_id: Unique identifier for this CHP unit
            P_el_nom: Nominal electrical power in kW (default: 500.0)
            eta_el: Electrical efficiency 0-1 (default: 0.40)
            eta_th: Thermal efficiency 0-1 (default: 0.45)
            name: Human-readable name (optional)

        Returns:
            Created CHP component

        Example:
            >>> chp = config.add_chp("chp_main", P_el_nom=500)
        """
        chp = CHP(component_id=chp_id, P_el_nom=P_el_nom, eta_el=eta_el, eta_th=eta_th, name=name or chp_id)

        self.plant.add_component(chp)

        # ----------------------------------------------------------
        # AUTOMATIC CHP FLARE
        # ----------------------------------------------------------
        flare_id = f"{chp_id}_flare"
        flare = Flare(component_id=flare_id, name=f"{chp_id}_flare")
        self.plant.add_component(flare)

        # Connect: CHP → flare
        self.connect(chp_id, flare_id, "gas")

        return chp

    def add_heating(
        self,
        heating_id: str,
        target_temperature: float = 308.15,
        heat_loss_coefficient: float = 0.5,
        name: Optional[str] = None,
    ) -> HeatingSystem:
        """
        Add a heating system to the plant.

        Args:
            heating_id: Unique identifier for heating system
            target_temperature: Target temperature in K (default: 308.15 = 35°C)
            heat_loss_coefficient: Heat loss in kW/K (default: 0.5)
            name: Human-readable name (optional)

        Returns:
            Created HeatingSystem component

        Example:
            >>> heating = config.add_heating("heating_main")
        """
        heating = HeatingSystem(
            component_id=heating_id,
            target_temperature=target_temperature,
            heat_loss_coefficient=heat_loss_coefficient,
            name=name or heating_id,
        )

        self.plant.add_component(heating)

        return heating

    def connect(self, from_component: str, to_component: str, connection_type: str = "default") -> Connection:
        """
        Connect two components.

        Args:
            from_component: Source component ID
            to_component: Target component ID
            connection_type: Type of connection ('liquid', 'gas', 'heat', etc.)

        Returns:
            Created Connection

        Example:
            >>> config.connect("digester_1", "chp_1", "gas")
        """
        connection = Connection(from_component, to_component, connection_type)
        self.plant.add_connection(connection)
        return connection

    def auto_connect_digester_to_chp(self, digester_id: str, chp_id: str) -> None:
        """
        Automatically connect digester to CHP through gas storage.

        Creates the connection chain: digester -> gas_storage -> chp

        Args:
            digester_id: Digester component ID
            chp_id: CHP component ID

        Raises:
            ValueError: If gas storage for digester is not found
        """
        # Gas storage is created with pattern: {digester_id}_storage
        storage_id = f"{digester_id}_storage"

        # Verify the storage exists
        if storage_id not in self.plant.components:
            raise ValueError(
                f"Gas storage '{storage_id}' not found for digester '{digester_id}'. "
                f"Ensure digester was added via PlantConfigurator.add_digester()"
            )

        # Connect: digester -> storage (already done in add_digester)
        # Connect: storage -> chp
        self.connect(storage_id, chp_id, "gas")

    def auto_connect_chp_to_heating(self, chp_id: str, heating_id: str) -> None:
        """
        Automatically connect CHP to heating with heat flow.

        Args:
            chp_id: CHP component ID
            heating_id: Heating component ID
        """
        self.connect(chp_id, heating_id, "heat")

    def create_single_stage_plant(
        self,
        digester_config: Optional[Dict[str, Any]] = None,
        chp_config: Optional[Dict[str, Any]] = None,
        heating_config: Optional[Dict[str, Any]] = None,
        auto_connect: bool = True,
    ) -> Dict[str, Any]:
        """
        Create a complete single-stage plant configuration.

        Args:
            digester_config: Configuration for digester (optional)
            chp_config: Configuration for CHP (optional)
            heating_config: Configuration for heating (optional)
            auto_connect: Automatically connect components (default: True)

        Returns:
            Dictionary with created component IDs

        Example:
            >>> components = config.create_single_stage_plant(
            ...     digester_config={'V_liq': 2000},
            ...     chp_config={'P_el_nom': 500}
            ... )
        """
        # Default configs
        digester_config = digester_config or {}
        chp_config = chp_config or {}
        heating_config = heating_config or {}

        # Set default IDs if not provided
        digester_config.setdefault("digester_id", "main_digester")
        chp_config.setdefault("chp_id", "chp_main")
        heating_config.setdefault("heating_id", "heating_main")

        # Create components
        digester = self.add_digester(**digester_config)

        components = {"digester": digester.component_id}

        if chp_config:
            chp = self.add_chp(**chp_config)
            components["chp"] = chp.component_id

            if auto_connect:
                self.auto_connect_digester_to_chp(digester.component_id, chp.component_id)

        if heating_config:
            heating = self.add_heating(**heating_config)
            components["heating"] = heating.component_id

            if auto_connect and "chp" in components:
                self.auto_connect_chp_to_heating(chp.component_id, heating.component_id)

        return components

    def create_two_stage_plant(
        self,
        hydrolysis_config: Optional[Dict[str, Any]] = None,
        digester_config: Optional[Dict[str, Any]] = None,
        chp_config: Optional[Dict[str, Any]] = None,
        heating_configs: Optional[list] = None,
        auto_connect: bool = True,
    ) -> Dict[str, Any]:
        """
        Create a complete two-stage plant configuration.

        Args:
            hydrolysis_config: Configuration for hydrolysis tank (optional)
            digester_config: Configuration for main digester (optional)
            chp_config: Configuration for CHP (optional)
            heating_configs: List of heating configurations (optional)
            auto_connect: Automatically connect components (default: True)

        Returns:
            Dictionary with created component IDs

        Example:
            >>> components = config.create_two_stage_plant(
            ...     hydrolysis_config={'V_liq': 500, 'T_ad': 318.15},
            ...     digester_config={'V_liq': 1500}
            ... )
        """
        # Default configs
        hydrolysis_config = hydrolysis_config or {}
        digester_config = digester_config or {}
        chp_config = chp_config or {}

        # Set default IDs
        hydrolysis_config.setdefault("digester_id", "hydrolysis_tank")
        hydrolysis_config.setdefault("name", "Hydrolysis Tank")
        hydrolysis_config.setdefault("T_ad", 318.15)  # 45°C for thermophilic

        digester_config.setdefault("digester_id", "main_digester")
        digester_config.setdefault("name", "Main Digester")

        chp_config.setdefault("chp_id", "chp_main")

        # Create components
        hydrolysis = self.add_digester(**hydrolysis_config)
        digester = self.add_digester(**digester_config)

        components = {"hydrolysis": hydrolysis.component_id, "digester": digester.component_id}

        # Connect digesters in series
        if auto_connect:
            self.connect(hydrolysis.component_id, digester.component_id, "liquid")

        # Add CHP
        if chp_config:
            chp = self.add_chp(**chp_config)
            components["chp"] = chp.component_id

            if auto_connect:
                # Connect both digesters to CHP
                self.auto_connect_digester_to_chp(hydrolysis.component_id, chp.component_id)
                self.auto_connect_digester_to_chp(digester.component_id, chp.component_id)

        # Add heating systems
        if heating_configs:
            components["heating"] = []
            for i, heating_cfg in enumerate(heating_configs):
                heating_cfg.setdefault("heating_id", f"heating_{i+1}")
                heating = self.add_heating(**heating_cfg)
                components["heating"].append(heating.component_id)

                if auto_connect and "chp" in components:
                    self.auto_connect_chp_to_heating(chp.component_id, heating.component_id)

        return components
