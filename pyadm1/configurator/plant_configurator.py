# pyadm1/configurator/plant_configurator.py
"""
High-level plant configuration helpers.

Convenience methods for adding components to biogas plants with sensible
defaults and validation.  Used both by direct API users and by the MCP server
tools.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyadm1.components.energy.biogas_upgrading import BiogasUpgrading

from pyadm1.components.biological.digester import Digester
from pyadm1.components.energy.chp import CHP
from pyadm1.components.energy.flare import Flare
from pyadm1.components.energy.gas_storage import GasStorage
from pyadm1.components.energy.heating import HeatingSystem
from pyadm1.components.feeding.feeder import Feeder
from pyadm1.components.feeding.substrate_storage import SubstrateStorage
from pyadm1.components.mechanical.mixer import Mixer
from pyadm1.configurator.connection_manager import Connection
from pyadm1.configurator.plant_builder import BiogasPlant
from pyadm1.substrates.feedstock import Feedstock


class PlantConfigurator:
    """
    High-level configurator for building biogas plants.

    Provides convenient methods for adding components with sensible defaults
    and automatic setup of common configurations (gas storage attached to
    digesters, flare attached to CHP, etc.).
    """

    def __init__(self, plant: BiogasPlant, feedstock: Feedstock | None = None):
        """
        Parameters
        ----------
        plant : BiogasPlant
            Plant instance to configure.
        feedstock : Feedstock, optional
            Feedstock used by all digesters added through this configurator.
            May be omitted to lay out the plant first and supply the substrates
            later via :meth:`set_feedstock` -- useful when the structure is
            known before the feed is. Digesters built without a feedstock can
            be created and initialized, but not simulated until one is attached.
        """
        self.plant = plant
        self.feedstock = feedstock

    def set_feedstock(
        self,
        feedstock: Feedstock,
        digester_ids: list[str] | None = None,
        Q_substrates: dict[str, list] | None = None,
        rebuild_state: bool = True,
    ) -> list[str]:
        """
        Attach a feedstock to this configurator and wire it into digesters.

        Complements the optional ``feedstock`` constructor argument: build the
        plant structure first, add the substrates once they are known.

        Parameters
        ----------
        feedstock : Feedstock
            Feedstock to attach. Also becomes the default for digesters added
            afterwards.
        digester_ids : list of str, optional
            Restrict the update to these digesters. Defaults to every digester
            in the plant.
        Q_substrates : dict, optional
            Per-digester feed rates ``{digester_id: [m³/d, ...]}``. Digesters
            not listed keep their current rates.
        rebuild_state : bool, default True
            Re-derive each digester's pre-inoculated steady state from the new
            blend.

        Returns
        -------
        list of str
            IDs of the digesters that were updated.
        """
        self.feedstock = feedstock
        updated: list[str] = []
        for comp_id, component in self.plant.components.items():
            if not isinstance(component, Digester):
                continue
            if digester_ids is not None and comp_id not in digester_ids:
                continue
            component.set_feedstock(
                feedstock,
                Q_substrates=(Q_substrates or {}).get(comp_id),
                rebuild_state=rebuild_state,
            )
            updated.append(comp_id)
        return updated

    def add_digester(
        self,
        digester_id: str,
        V_liq: float = 1050.0,
        V_gas: float = 150.0,
        T_ad: float = 315.15,
        name: str | None = None,
        Q_substrates: list | None = None,
        k_L_a: float | None = None,
        adm1_state: list | None = None,
        dynamic_volume: bool = False,
        initial_fill_fraction: float = 1.0,
        outflow_time_constant: float = 1.0,
        backend: str | None = None,
        feedstock: Feedstock | None = None,
        diameter: float | None = None,
        roof_type: str = "membrane",
        k_wall: float | None = None,
        k_roof: float | None = None,
        k_ground: float | None = None,
    ) -> tuple[Digester, str]:
        """
        Add an ADM1da digester to the plant.

        The digester's steady-state initial state is derived from the attached
        :class:`Feedstock`; the influent is blended from the feed rates of every step.
        A gas storage is auto-created and connected.

        When no feedstock is available (neither here nor on the configurator),
        the digester is still created and initialized -- attach the substrates
        later with :meth:`set_feedstock`. ``T_ad`` can be changed at any time
        via :meth:`Digester.set_temperature`.

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
            Substrate feed rates, one entry per substrate of the feedstock.
            With the feedstock's default ``simba_q_convention=True`` the
            values are tonnes of fresh matter per day [t FM/d]; otherwise
            they are volume flows [m³/d].
        k_L_a : float, optional
            Override of the gas–liquid mass-transfer coefficient [1/d].
        adm1_state : list of float, optional
            41-element initial state vector.  When supplied, replaces the
            auto-built steady-state vector.
        dynamic_volume : bool, default False
            Enable a dynamic sludge-volume balance
            ``dV/dt = Q_in − Q_out − q_S,loss``, with ``Q_out`` from an
            overflow weir at ``V_liq``. When False, sludge volume stays
            constant.
        initial_fill_fraction : float, default 1.0
            Starting sludge fill as a fraction of ``V_liq``. Only used when
            ``dynamic_volume=True``. Set below 1.0 to simulate a partially-
            filled startup transient.
        outflow_time_constant : float, default 1.0
            Overflow-weir time constant ``τ_out`` [d]. Only used when
            ``dynamic_volume=True``.
        backend : str, optional
            ADM1 right-hand-side backend, ``"numpy"`` (default) or ``"torch"``
            (differentiable, same values). ``None`` uses the process-wide
            default (see :func:`pyadm1.set_default_adm1_backend`).
        feedstock : Feedstock, optional
            Per-digester feedstock. Falls back to the configurator's feedstock,
            which may itself be ``None``.
        diameter : float, optional
            Inner tank diameter [m]. Estimated from ``V_liq + V_gas`` when
            omitted; the tank heights are always derived from it.
        roof_type : str, default "membrane"
            Roof construction: ``"membrane"``, ``"concrete"``, ``"steel"`` or
            ``"open"``.
        k_wall, k_roof, k_ground : float, optional
            U-values of wall, roof and floor slab [W/(m²·K)]. When set, the
            digester derives its heat loss coefficient from them; a heating
            system attached via ``add_heating(..., digester_id=...)`` uses it.

        Returns
        -------
        (Digester, str)
            The created digester and a one-line description of how the
            initial state was determined.
        """
        fs = feedstock if feedstock is not None else self.feedstock

        digester = Digester(
            component_id=digester_id,
            feedstock=fs,
            V_liq=V_liq,
            V_gas=V_gas,
            T_ad=T_ad,
            name=name or digester_id,
            dynamic_volume=dynamic_volume,
            initial_fill_fraction=initial_fill_fraction,
            outflow_time_constant=outflow_time_constant,
            backend=backend,
            diameter=diameter,
            roof_type=roof_type,
            k_wall=k_wall,
            k_roof=k_roof,
            k_ground=k_ground,
        )

        if k_L_a is not None:
            digester.adm1.set_calibration_parameters({"k_L_a": float(k_L_a)})

        if Q_substrates is None:
            Q_substrates = [0.0] * 10

        init_kwargs: dict[str, Any] = {"Q_substrates": Q_substrates}
        if adm1_state is not None:
            init_kwargs["adm1_state"] = list(adm1_state)
            state_info = "  - Initial state: User-supplied 41-element ADM1 vector\n"
        elif fs is None:
            state_info = "  - Initial state: no feedstock yet — call set_feedstock() before simulating\n"
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
        name: str | None = None,
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
        target_temperature: float = 315.15,
        heat_loss_coefficient: float | None = None,
        name: str | None = None,
        digester_id: str | None = None,
    ) -> HeatingSystem:
        """
        Add a heating system to the plant.

        Parameters
        ----------
        heating_id : str
            Unique identifier for this heating system.
        target_temperature : float
            Setpoint [K] (default 315.15 = 42 °C, matching the ``T_ad`` default
            of :meth:`add_digester`). Pass it explicitly whenever the digester
            runs at a different temperature -- the two are not linked.
        heat_loss_coefficient : float, optional
            Heat loss coefficient (UA) [kW/K]. An explicit value always wins.
            When omitted, the value the connected digester derives from its
            k-values is used, else 0.5 kW/K.
        name : str, optional
        digester_id : str, optional
            Digester this system heats. Connects the two so the heating
            receives the digester's temperature, substrate feed (for the
            process-heat term) and heat loss coefficient. Without it only
            the flat heat-loss term is computed.
        """
        heating = HeatingSystem(
            component_id=heating_id,
            target_temperature=target_temperature,
            heat_loss_coefficient=heat_loss_coefficient,
            name=name or heating_id,
            feedstock=self.feedstock,
        )

        self.plant.add_component(heating)

        if digester_id is not None:
            self.auto_connect_digester_to_heating(digester_id, heating_id)

        return heating

    def _substrate_slot(self, substrate: int | str, feedstock: Feedstock | None) -> tuple[int, Feedstock]:
        """Resolve a substrate given by slot index or by ID against the feedstock."""
        fs = feedstock if feedstock is not None else self.feedstock
        if fs is None:
            raise ValueError("A feedstock is required to resolve the substrate; pass one or set it on the configurator.")
        ids = list(fs._substrate_ids)
        if isinstance(substrate, str):
            if substrate not in ids:
                raise ValueError(f"Substrate '{substrate}' is not part of the feedstock {ids}.")
            return ids.index(substrate), fs
        if not 0 <= int(substrate) < len(ids):
            raise ValueError(f"Substrate index {substrate} is outside the feedstock's {len(ids)} substrates.")
        return int(substrate), fs

    def add_substrate_storage(
        self,
        storage_id: str,
        substrate: int | str,
        capacity: float = 1000.0,
        initial_level: float | None = None,
        storage_type: str = "bunker_silo",
        substrate_type: str = "corn_silage",
        name: str | None = None,
        feedstock: Feedstock | None = None,
        **storage_kwargs: Any,
    ) -> SubstrateStorage:
        """
        Add a storage for one substrate of the feedstock.

        Density, dry matter and volatile solids come from the substrate
        definition instead of the storage's typical-value table. The inventory
        is kept in tonnes of fresh matter.

        Parameters
        ----------
        storage_id : str
            Unique identifier for this storage.
        substrate : int or str
            Slot index or substrate ID within the feedstock.
        capacity : float
            Capacity [t FM] (default 1000).
        initial_level : float, optional
            Starting stock [t FM]. When omitted the stock is unlimited: the
            feeder always gets what it asks for and no refills are needed.
            Give a number to track the inventory.
        storage_type : str
            ``"bunker_silo"``, ``"vertical_silo"``, ``"clamp"``,
            ``"above_ground_tank"``, ...
        substrate_type : str
            Category of the storage's loss model (``"corn_silage"``,
            ``"manure_liquid"``, ...); it no longer decides density or solids.
        name : str, optional
        feedstock : Feedstock, optional
            Falls back to the configurator's feedstock.
        **storage_kwargs
            Further :class:`SubstrateStorage` arguments (``degradation_rate``,
            ``temperature``).
        """
        index, fs = self._substrate_slot(substrate, feedstock)
        params = fs._subs[index]

        storage = SubstrateStorage(
            component_id=storage_id,
            storage_type=storage_type,
            substrate_type=substrate_type,
            capacity=capacity,
            initial_level=initial_level,
            name=name or storage_id,
            density=fs._densities[index],
            dry_matter=params.TS / 10.0,  # kg/t FM -> % FM
            vs_content=(1.0 - params.fRA) * 100.0,  # % of TS
            **storage_kwargs,
        )
        self.plant.add_component(storage)
        return storage

    def add_feeder(
        self,
        feeder_id: str,
        digester_id: str,
        substrate: int | str,
        storage_id: str | None = None,
        Q_setpoint: float | None = None,
        name: str | None = None,
        feedstock: Feedstock | None = None,
        **feeder_kwargs: Any,
    ) -> Feeder:
        """
        Add a feeder that doses one substrate into a digester.

        Closes the chain storage -> feeder -> digester: the feeder is limited
        by the stock of its storage, the storage is drawn down by what the
        feeder actually doses, and the digester's substrate slot follows that
        dosed flow (including dosing noise and blockages) instead of the
        configured rate.

        Parameters
        ----------
        feeder_id : str
            Unique identifier for this feeder.
        digester_id : str
            Digester that is fed.
        substrate : int or str
            Slot index or substrate ID within the feedstock.
        storage_id : str, optional
            Substrate storage the feeder draws from. Without one the supply
            is unlimited.
        Q_setpoint : float, optional
            Feed rate, in the unit of the digester's ``Q_substrates``.
            Defaults to the rate configured for that slot on the digester.
        name : str, optional
        feedstock : Feedstock, optional
            Falls back to the digester's, then the configurator's feedstock.
        **feeder_kwargs
            Further :class:`Feeder` arguments (``feeder_type``, ``Q_max``,
            ``specific_energy``, ``enable_dosing_noise``, ...).
        """
        digester = self.plant.components.get(digester_id)
        if not isinstance(digester, Digester):
            raise ValueError(f"Digester '{digester_id}' not found; add it before its feeder.")
        if storage_id is not None and not isinstance(self.plant.components.get(storage_id), SubstrateStorage):
            raise ValueError(f"Substrate storage '{storage_id}' not found; add it before its feeder.")

        index, fs = self._substrate_slot(substrate, feedstock if feedstock is not None else digester.feedstock)
        if Q_setpoint is None:
            Q_setpoint = float(digester.Q_substrates[index]) if index < len(digester.Q_substrates) else 0.0
        feeder_kwargs.setdefault("Q_max", max(20.0, 1.5 * Q_setpoint))

        feeder = Feeder(
            component_id=feeder_id,
            name=name or feeder_id,
            Q_setpoint=Q_setpoint,
            substrate_index=index,
            # Tonnes per unit of Q: 1 with the ADM1da mass convention, rho/1000 for m³.
            flow_to_mass_factor=fs.q_conversion_factors[index] * fs._densities[index] / 1000.0,
            **feeder_kwargs,
        )
        self.plant.add_component(feeder)

        if storage_id is not None:
            self.connect(storage_id, feeder_id, "feed")
        self.connect(feeder_id, digester_id, "feed")
        return feeder

    def add_mixer(
        self,
        mixer_id: str,
        digester_id: str,
        mixer_type: str = "propeller",
        mixing_intensity: str = "medium",
        name: str | None = None,
        **mixer_kwargs: Any,
    ) -> Mixer:
        """
        Add a mixer to a digester.

        The tank geometry is taken from the digester instead of being
        estimated a second time: ``tank_volume`` is its liquid volume,
        ``tank_diameter`` its diameter and ``tank_height`` its sludge level.
        The two are connected, so the mixer follows the digester's
        temperature and total solids on every step (both change the
        viscosity, and with it the mixing power).

        Parameters
        ----------
        mixer_id : str
            Unique identifier for this mixer.
        digester_id : str
            Digester the mixer is installed in.
        mixer_type : str
            ``"propeller"``, ``"paddle"`` or ``"jet"``.
        mixing_intensity : str
            ``"low"``, ``"medium"`` or ``"high"``.
        name : str, optional
        **mixer_kwargs
            Further :class:`Mixer` arguments (``power_installed``,
            ``impeller_diameter``, ``on_time_fraction``, ``fluid_viscosity``,
            ...). The three tank arguments may not be overridden.
        """
        digester = self.plant.components.get(digester_id)
        if not isinstance(digester, Digester):
            raise ValueError(f"Digester '{digester_id}' not found; add it before its mixer.")
        overridden = {"tank_volume", "tank_diameter", "tank_height"} & set(mixer_kwargs)
        if overridden:
            raise ValueError(f"{sorted(overridden)} come from digester '{digester_id}' and cannot be set here.")

        mixer = Mixer(
            component_id=mixer_id,
            mixer_type=mixer_type,
            tank_volume=digester._V_liq_max,
            tank_diameter=digester.diameter,
            tank_height=digester.liquid_height,
            mixing_intensity=mixing_intensity,
            name=name or mixer_id,
            **mixer_kwargs,
        )
        self.plant.add_component(mixer)
        self.connect(digester_id, mixer_id, "control")
        return mixer

    def connect(
        self,
        from_component: str,
        to_component: str,
        connection_type: str = "default",
        split_fraction: float = 1.0,
    ) -> Connection:
        """Connect two components.

        ``split_fraction`` (0..1) sends only part of the source's effluent along
        this edge — e.g. a separator that recirculates 60 % of its press water
        into the digester and passes the rest to the store. The default 1.0
        keeps the previous behaviour: every downstream component sees the full
        flow.
        """
        connection = Connection(from_component, to_component, connection_type, split_fraction)
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

    def auto_connect_digester_to_heating(self, digester_id: str, heating_id: str) -> None:
        """Connect digester → heating so temperature, feed and UA reach the heating."""
        self.connect(digester_id, heating_id, "control")

    def auto_connect_chp_to_heating(self, chp_id: str, heating_id: str) -> None:
        """Connect CHP → heating with heat flow."""
        self.connect(chp_id, heating_id, "heat")

    def add_bgaa(
        self,
        bgaa_id: str,
        capacity_m3h: float = 500.0,
        ch4_recovery: float = 0.98,
        ch4_content_in: float = 0.55,
        ch4_content_out: float = 0.97,
        name: str | None = None,
    ) -> BiogasUpgrading:
        """
        Add a biogas upgrading unit (Biogasaufbereitungsanlage) to the plant.

        Automatically creates and connects an emergency flare downstream of
        the BGAA for capacity overflow (``{bgaa_id}_flare``).
        """
        from pyadm1.components.energy.biogas_upgrading import BiogasUpgrading

        bgaa = BiogasUpgrading(
            component_id=bgaa_id,
            capacity_m3h=capacity_m3h,
            ch4_recovery=ch4_recovery,
            ch4_content_in=ch4_content_in,
            ch4_content_out=ch4_content_out,
            name=name or bgaa_id,
        )
        self.plant.add_component(bgaa)

        flare_id = f"{bgaa_id}_flare"
        flare = Flare(component_id=flare_id, name=f"{bgaa_id}_flare")
        self.plant.add_component(flare)
        self.connect(bgaa_id, flare_id, "gas")

        return bgaa

    def auto_connect_digester_to_bgaa(self, digester_id: str, bgaa_id: str) -> None:
        """Connect digester gas storage → BGAA."""
        storage_id = f"{digester_id}_storage"
        if storage_id not in self.plant.components:
            raise ValueError(
                f"Gas storage '{storage_id}' not found for digester '{digester_id}'. "
                f"Ensure the digester was added via PlantConfigurator.add_digester()."
            )
        self.connect(storage_id, bgaa_id, "gas")

    def create_single_stage_plant(
        self,
        digester_config: dict[str, Any] | None = None,
        chp_config: dict[str, Any] | None = None,
        heating_config: dict[str, Any] | None = None,
        auto_connect: bool = True,
    ) -> dict[str, Any]:
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
        hydrolysis_config: dict[str, Any] | None = None,
        digester_config: dict[str, Any] | None = None,
        chp_config: dict[str, Any] | None = None,
        heating_configs: list | None = None,
        auto_connect: bool = True,
    ) -> dict[str, Any]:
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
