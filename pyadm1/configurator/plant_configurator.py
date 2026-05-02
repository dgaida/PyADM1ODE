# pyadm1/configurator/plant_configurator.py
"""
High-level plant configuration helpers.

Convenience methods for adding components to biogas plants with sensible
defaults and validation.  Used both by direct API users and by the MCP server
tools.
"""

from typing import Optional, Dict, Any

from pyadm1.configurator.plant_builder import BiogasPlant
from pyadm1.components.biological.digester import Digester
from pyadm1.components.energy.chp import CHP
from pyadm1.components.energy.heating import HeatingSystem
from pyadm1.components.energy.gas_storage import GasStorage
from pyadm1.components.energy.flare import Flare
from pyadm1.configurator.connection_manager import Connection
from pyadm1.substrates.feedstock import Feedstock


class PlantConfigurator:
    """
    High-level configurator for building biogas plants.

    Provides convenient methods for adding components with sensible defaults
    and automatic setup of common configurations (gas storage attached to
    digesters, flare attached to CHP, etc.).
    """

    def __init__(self, plant: BiogasPlant, feedstock: Feedstock):
        """
        Parameters
        ----------
        plant : BiogasPlant
            Plant instance to configure.
        feedstock : Feedstock
            Feedstock used by all digesters added through this configurator.
        """
        self.plant = plant
        self.feedstock = feedstock

    def add_digester(
        self,
        digester_id: str,
        V_liq: float = 1050.0,
        V_gas: float = 150.0,
        T_ad: float = 315.15,
        name: Optional[str] = None,
        Q_substrates: Optional[list] = None,
        k_L_a: Optional[float] = None,
        adm1_state: Optional[list] = None,
    ) -> "tuple[Digester, str]":
        """
        Add an ADM1da digester to the plant.

        The digester's influent DataFrame, density, and steady-state initial
        state are wired automatically from the attached :class:`Feedstock`.
        A gas storage is auto-created and connected.

        Parameters
        ----------
        digester_id : str
            Unique identifier for this digester.
        V_liq : float
            Liquid volume [m³] (default 1050).
        V_gas : float
            Gas headspace volume [m³] (default 150).
        T_ad : float
            Operating temperature [K] (default 315.15 = 42 °C).
        name : str, optional
        Q_substrates : list of float, optional
            Substrate feed rates [m³/d], one entry per substrate slot
            (up to 10 slots).
        k_L_a : float, optional
            Override of the gas–liquid mass-transfer coefficient [1/d].
        adm1_state : list of float, optional
            41-element initial state vector.  When supplied, replaces the
            auto-built steady-state vector.

        Returns
        -------
        (Digester, str)
            The created digester and a one-line description of how the
            initial state was determined.
        """
        digester = Digester(
            component_id=digester_id,
            feedstock=self.feedstock,
            V_liq=V_liq,
            V_gas=V_gas,
            T_ad=T_ad,
            name=name or digester_id,
        )

        if k_L_a is not None:
            digester.adm1.set_calibration_parameters({"k_L_a": float(k_L_a)})

        if Q_substrates is None:
            Q_substrates = [0.0] * 10

        init_kwargs: Dict[str, Any] = {"Q_substrates": Q_substrates}
        if adm1_state is not None:
            init_kwargs["adm1_state"] = list(adm1_state)
            state_info = "  - Initial state: User-supplied 41-element ADM1 vector\n"
        else:
            state_info = "  - Initial state: Auto-built steady-state from feedstock\n"
        digester.initialize(init_kwargs)

        self.plant.add_component(digester)

        # Automatic gas storage + connection
        storage_id = f"{digester_id}_storage"
        storage = GasStorage(
            component_id=storage_id,
            storage_type="membrane",
            capacity_m3=max(50.0, V_gas),
            name=f"{name or digester_id} Gas Storage",
        )
        self.plant.add_component(storage)
        self.connect(digester_id, storage_id, "gas")

        return digester, state_info

    def add_chp(
        self,
        chp_id: str,
        P_el_nom: float = 500.0,
        eta_el: float = 0.40,
        eta_th: float = 0.45,
        name: Optional[str] = None,
    ) -> CHP:
        """
        Add a CHP unit to the plant.

        Automatically creates and connects a safety flare downstream of the CHP.
        """
        chp = CHP(
            component_id=chp_id,
            P_el_nom=P_el_nom,
            eta_el=eta_el,
            eta_th=eta_th,
            name=name or chp_id,
        )

        self.plant.add_component(chp)

        flare_id = f"{chp_id}_flare"
        flare = Flare(component_id=flare_id, name=f"{chp_id}_flare")
        self.plant.add_component(flare)
        self.connect(chp_id, flare_id, "gas")

        return chp

    def add_heating(
        self,
        heating_id: str,
        target_temperature: float = 308.15,
        heat_loss_coefficient: float = 0.5,
        name: Optional[str] = None,
    ) -> HeatingSystem:
        """Add a heating system to the plant."""
        heating = HeatingSystem(
            component_id=heating_id,
            target_temperature=target_temperature,
            heat_loss_coefficient=heat_loss_coefficient,
            name=name or heating_id,
            feedstock=self.feedstock,
        )

        self.plant.add_component(heating)

        return heating

    def connect(self, from_component: str, to_component: str, connection_type: str = "default") -> Connection:
        """Connect two components."""
        connection = Connection(from_component, to_component, connection_type)
        self.plant.add_connection(connection)
        return connection

    def auto_connect_digester_to_chp(self, digester_id: str, chp_id: str) -> None:
        """Connect digester → gas_storage → chp."""
        storage_id = f"{digester_id}_storage"

        if storage_id not in self.plant.components:
            raise ValueError(
                f"Gas storage '{storage_id}' not found for digester '{digester_id}'. "
                f"Ensure the digester was added via PlantConfigurator.add_digester()."
            )

        self.connect(storage_id, chp_id, "gas")

    def auto_connect_chp_to_heating(self, chp_id: str, heating_id: str) -> None:
        """Connect CHP → heating with heat flow."""
        self.connect(chp_id, heating_id, "heat")

    def create_single_stage_plant(
        self,
        digester_config: Optional[Dict[str, Any]] = None,
        chp_config: Optional[Dict[str, Any]] = None,
        heating_config: Optional[Dict[str, Any]] = None,
        auto_connect: bool = True,
    ) -> Dict[str, Any]:
        """Create a complete single-stage plant configuration."""
        digester_config = digester_config or {}
        chp_config = chp_config or {}
        heating_config = heating_config or {}

        digester_config.setdefault("digester_id", "main_digester")
        chp_config.setdefault("chp_id", "chp_main")
        heating_config.setdefault("heating_id", "heating_main")

        digester, _ = self.add_digester(**digester_config)

        components = {
            "digester": digester.component_id,
            "storage": f"{digester.component_id}_storage",
        }

        if chp_config:
            chp = self.add_chp(**chp_config)
            components["chp"] = chp.component_id
            components["flare"] = f"{chp.component_id}_flare"

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
        Create a two-stage plant: hydrolysis pre-tank → main fermenter.

        The hydrolysis stage is just another :class:`Digester` instance with
        a higher temperature and shorter HRT — there is no separate
        ``Hydrolysis`` class.
        """
        hydrolysis_config = hydrolysis_config or {}
        digester_config = digester_config or {}
        chp_config = chp_config or {}

        hydrolysis_config.setdefault("digester_id", "hydrolysis_tank")
        hydrolysis_config.setdefault("name", "Hydrolysis Tank")
        hydrolysis_config.setdefault("T_ad", 318.15)  # 45 °C
        hydrolysis_config.setdefault("V_liq", 500.0)
        hydrolysis_config.setdefault("V_gas", 75.0)

        digester_config.setdefault("digester_id", "main_digester")
        digester_config.setdefault("name", "Main Digester")

        chp_config.setdefault("chp_id", "chp_main")

        hydrolysis, _ = self.add_digester(**hydrolysis_config)
        digester, _ = self.add_digester(**digester_config)

        components = {
            "hydrolysis": hydrolysis.component_id,
            "hydrolysis_storage": f"{hydrolysis.component_id}_storage",
            "digester": digester.component_id,
            "digester_storage": f"{digester.component_id}_storage",
        }

        if auto_connect:
            self.connect(hydrolysis.component_id, digester.component_id, "liquid")

        if chp_config:
            chp = self.add_chp(**chp_config)
            components["chp"] = chp.component_id
            components["flare"] = f"{chp.component_id}_flare"

            if auto_connect:
                self.auto_connect_digester_to_chp(hydrolysis.component_id, chp.component_id)
                self.auto_connect_digester_to_chp(digester.component_id, chp.component_id)

        if heating_configs:
            components["heating"] = []
            for i, heating_cfg in enumerate(heating_configs):
                heating_cfg.setdefault("heating_id", f"heating_{i + 1}")
                heating = self.add_heating(**heating_cfg)
                components["heating"].append(heating.component_id)

                if auto_connect and "chp" in components:
                    self.auto_connect_chp_to_heating(chp.component_id, heating.component_id)

        return components
