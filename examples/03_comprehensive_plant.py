#!/usr/bin/env python3
# ============================================================================
# examples/03_comprehensive_plant.py
# ============================================================================
"""
Comprehensive Biogas Plant Example

Demonstrates all currently functional PyADM1ODE components in a realistic
two-stage biogas plant simulation with documented benchmark values.

Components used (17):
    Substrate management:
        - SubstrateStorage  (corn silage silo, 500 t)
        - SubstrateStorage  (manure tank, 200 m3)
        - Feeder            (screw feeder for silage)
        - Feeder            (progressive cavity pump for manure)
        - Pump              (substrate transfer pump)
    Biology:
        - Hydrolysis        (thermophilic pre-treatment, 1200 m3, 55 C)
        - Digester          (main methanogenesis, 2000 m3, 35 C)
        - Separator         (screw press for digestate solid/liquid split)
    Mechanical:
        - Mixer             (agitator hydrolysis stage)
        - Mixer             (agitator main digester)
    Energy:
        - GasStorage        (auto, one per fermentation stage)
        - CHP               (500 kW_el)
        - Flare             (auto, safety flare)
        - HeatingSystem     (heating hydrolysis stage)
        - HeatingSystem     (heating main digester)
        - Boiler            (dual-fuel backup boiler, 150 kW_th)

Components NOT yet implemented (stubs or missing):
    1. H2S removal         - biological/chemical desulfurization    (MISSING - should be added)
    2. Sensors             - pH probes, gas meters, H2S analysis    (empty folder)
    3. Gas upgrading       - biomethane production (membrane, PSA)
    4. Seasonal temperature - time-variable ambient temperature
    5. Digestate post-treatment - composting, NH3 stripping
    6. Substrate pre-treatment  - shredder, conditioner, pre-heating
    7. SCADA / control     - PID controllers, alarm system
    8. Emergency concept   - bypass lines, emergency flare sizing

Benchmark reference values (Source: FNR Leitfaden Biogas, KTBL):
    OLR:          2-4 kg VS/(m3*d)  optimal, > 5 critical
    HRT:          20-40 days
    pH:           7.0-7.8 optimal, < 6.8 or > 8.2 critical
    FOS/TAC:      < 0.3 stable, 0.3-0.4 observe, > 0.4 critical
    CH4 fraction: 52-58% (maize/manure), 62-68% (lipid-rich substrates)
    Biogas yield: ~160 Nm3/t FM (mix maize+manure, KTBL)
                  ~200-220 Nm3/t FM (pure maize silage)
                  ~ 20-35  Nm3/t FM (pure swine manure)
                  Note: BMP values refer to fresh mass (FM), not VS
    CHP eta_el:   38-42 %,  eta_th: 42-47 %
    Own consumption: 8-12 % of gross power

Pump power references:
    KTBL (2013): Faustzahlen Biogas, 3rd ed., KTBL-Heft 469
        Specific energy demand substrate pumps:
            Manure/slurry:   0.5-1.0 kWh/m3  (short pipes, H < 20 m)
            Silage/solids:   1.5-3.0 kWh/m3  (screw conveyor)
    Hydraulic power formula (DIN 24260):
        P_hydraulic = rho * g * Q * H  [W]
        P_shaft     = P_hydraulic / eta_pump
        P_electric  = P_shaft / eta_motor
        Typical pressure heads for biogas plants:
            Near feeding  (< 50 m pipe, H_geo < 5 m):  H = 15-25 m
            Remote feeding (> 100 m pipe):              H = 30-50 m

Separator references:
    KTBL (2013): Faustzahlen Biogas, 3rd ed., KTBL-Heft 469
    Hjorth et al. (2010): Solid-liquid separation of animal slurry,
        Bioresource Technology 101, pp. 10-23

Boiler references:
    KTBL (2013): Faustzahlen Biogas, 3rd ed., pp. 160-165
    DVGW G 260 (2021): Natural gas quality, LHV H-gas = 10.0 kWh/m3
    VDI 4631 (2012): Part-load efficiency of heating systems

Usage:
    python examples/03_comprehensive_plant.py
"""

from pathlib import Path

# ============================================================================
# Configuration constants
# ============================================================================

# --- Fermenters ---
# Hydrolysis stage design:
#   oTS/d = 15 m3/d * 200 kg/m3 + 10 m3/d * 41 kg/m3 = 3410 kg oTS/d
#   Target OLR = 2.8 kg oTS/(m3*d)  ->  V = 3410 / 2.8 = 1218 m3  -> 1200 m3
#   HRT = 1200 / 25 = 48 days
V_LIQ_HYDRO = 1200.0  # m3  hydrolysis stage  (OLR ~2.8 kg VS/(m3*d), HRT ~48 d)
V_GAS_HYDRO = 185.0  # m3  (~15 % of liquid volume)
T_AD_HYDRO = 328.15  # K   55 C thermophilic

V_LIQ_2 = 1800.0  # m3  main methanogenesis digester
V_GAS_2 = 310.0  # m3
T_AD_2 = 308.15  # K   35 C mesophilic

# --- Substrate inputs (m3/d, indices 0-9 per substrate_gummersbach.xml) ---
# Index 0: corn silage,  Index 1: swine manure
Q_MAIZE = 15.0  # m3/d corn silage
Q_MANURE = 10.0  # m3/d swine manure
Q_SUB = [Q_MAIZE, Q_MANURE, 0, 0, 0, 0, 0, 0, 0, 0]

# --- CHP ---
P_EL_NOM = 500.0  # kW  nominal electrical power
ETA_EL = 0.40  # -   electrical efficiency
ETA_TH = 0.45  # -   thermal efficiency

# --- Substrate parameters (KTBL/FNR literature values) ---
# Bulk density x TS x VS = VS content [kg VS/m3 FM]
#   Corn silage:  700 kg/m3 x 30% TS x 95% VS = 200 kg VS/m3
#   Swine manure: 1020 kg/m3 x 5% TS x 80% VS =  41 kg VS/m3
DENSITY_MAIZE = 700.0  # kg FM/m3
DENSITY_MANURE = 1020.0  # kg FM/m3
OTS_MAIZE = 200.0  # kg VS/m3 FM
OTS_MANURE = 41.0  # kg VS/m3 FM

# --- Simulation ---
DURATION = 30.0  # days
DT = 1.0 / 24.0  # 1-hour time step
SAVE_INTERVAL = 1.0  # daily snapshots
T_AMBIENT = 283.15  # K   10 C ambient temperature (autumn/winter)


def main():
    """Simulate comprehensive biogas plant with all functional components."""
    from pyadm1.configurator.plant_builder import BiogasPlant
    from pyadm1.substrates.feedstock import Feedstock
    from pyadm1.core.adm1 import get_state_zero_from_initial_state
    from pyadm1.configurator.plant_configurator import PlantConfigurator
    from pyadm1.components.biological import Hydrolysis, Separator
    from pyadm1.components.energy import Boiler, GasStorage
    from pyadm1.components.mechanical.mixer import Mixer
    from pyadm1.components.mechanical.pump import Pump
    from pyadm1.components.feeding.substrate_storage import SubstrateStorage
    from pyadm1.components.feeding.feeder import Feeder

    _hdr = "=" * 72
    print(_hdr)
    print("  PyADM1ODE - Comprehensive Biogas Plant Example (03)")
    print(_hdr)

    # ------------------------------------------------------------------
    # 1. Feedstock and initial ADM1 state
    # ------------------------------------------------------------------
    feedstock = Feedstock(feeding_freq=48)

    data_path = Path(__file__).parent.parent / "data" / "initial_states"
    initial_state_file = data_path / "digester_initial8.csv"
    if initial_state_file.exists():
        print(f"\nInitial state loaded: {initial_state_file.name}")
        adm1_state = get_state_zero_from_initial_state(str(initial_state_file))
    else:
        print("  Warning: initial state file not found - using defaults.")
        adm1_state = None

    # ------------------------------------------------------------------
    # 2. Substrate storage and feeding  (standalone, outside plant)
    # ------------------------------------------------------------------
    print("\n" + _hdr)
    print("  SUBSTRATE MANAGEMENT")
    print(_hdr)

    # Corn silage silo (500 t capacity, 400 t filled)
    silo = SubstrateStorage(
        component_id="silo_maize",
        storage_type="vertical_silo",
        substrate_type="corn_silage",
        capacity=500.0,
        initial_level=400.0,
        name="Corn Silage Silo",
    )
    print(
        f"  Corn silage silo:  {silo.current_level:.0f} t / {silo.capacity:.0f} t  "
        f"(TS={silo.dry_matter:.1f}%, VS={silo.vs_content:.1f}%)"
    )

    # Manure storage tank (200 m3, liquid)
    tank = SubstrateStorage(
        component_id="tank_manure",
        storage_type="above_ground_tank",
        substrate_type="manure_liquid",
        capacity=200.0,
        initial_level=150.0,
        name="Manure Storage Tank",
    )
    print(f"  Manure tank:       {tank.current_level:.0f} m3 / {tank.capacity:.0f} m3  " f"(TS={tank.dry_matter:.1f}%)")

    # Screw feeder for corn silage (solid)
    feeder_maize = Feeder(
        component_id="feeder_maize",
        feeder_type="screw",
        Q_max=20.0,
        substrate_type="solid",
        name="Screw Feeder Maize",
    )
    print(f"  Screw feeder:      {feeder_maize.power_installed:.1f} kW  " f"(max {feeder_maize.Q_max:.0f} m3/d)")

    # Progressive cavity pump for manure (slurry)
    feeder_manure = Feeder(
        component_id="feeder_manure",
        feeder_type="progressive_cavity",
        Q_max=15.0,
        substrate_type="slurry",
        name="PC Pump Manure",
    )
    print(f"  PC pump manure:    {feeder_manure.power_installed:.1f} kW  " f"(max {feeder_manure.Q_max:.0f} m3/d)")

    # Transfer pump: reception pit -> hydrolysis stage
    # Sizing: 25 m3/d total, near feeding (<50 m pipe)
    # Pressure head 20 m (H_geo~3 m + pipe losses ~17 m, KTBL reference)
    pump_feed = Pump(
        component_id="pump_feed",
        pump_type="progressive_cavity",
        Q_nom=25.0,  # m3/d  total substrate (maize + manure)
        pressure_head=20.0,  # m     KTBL near-feeding reference
        name="Substrate Transfer Pump",
    )
    print(f"  Transfer pump:     {pump_feed.Q_nom:.0f} m3/d  H={pump_feed.pressure_head:.0f} m")

    # Screw press separator for digestate (standalone, stepped in simulation loop)
    separator = Separator(
        component_id="separator_1",
        separator_type="screw_press",
        name="Screw Press Separator",
    )
    separator.initialize()
    print(
        f"\n  Screw press separator: "
        f"separation_eff={separator.separation_efficiency*100:.0f}%  "
        f"TS_solid={separator.ts_solid_target:.0f} kg/m3 (~25%)  "
        f"spec_energy={separator.specific_energy:.1f} kWh/t FM"
    )

    # Auxiliary boiler: dual-fuel backup heating (biogas first, natural gas supplement)
    # Sizing: covers peak heat demand when CHP waste heat is insufficient.
    # P_th_nom = 150 kW  (typical for plants up to 500 kW_el, KTBL 2013 p.162)
    boiler = Boiler(
        component_id="boiler_1",
        P_th_nom=150.0,  # kW  nominal thermal output
        efficiency=0.92,  # EN 303-1: condensing gas boiler
        fuel_type="dual",  # biogas first, natural gas as backup
        name="Auxiliary Boiler",
    )
    boiler.initialize()
    print(
        f"  Auxiliary boiler:      P_th_nom={boiler.P_th_nom:.0f} kW  "
        f"eta={boiler.efficiency*100:.0f}%  fuel={boiler.fuel_type}"
    )

    # ------------------------------------------------------------------
    # 3. Plant configuration
    # ------------------------------------------------------------------
    print("\n" + _hdr)
    print("  PLANT CONFIGURATION")
    print(_hdr)

    plant = BiogasPlant("Comprehensive Biogas Plant Demo")
    configurator = PlantConfigurator(plant, feedstock)

    # --- Stage 1: Hydrolysis pre-treatment (thermophilic, 55 C) ---
    print(f"\n  Hydrolysis stage: {V_LIQ_HYDRO:.0f} m3, " f"{T_AD_HYDRO - 273.15:.0f} C, OLR ~2.8 kg VS/(m3*d)")
    hydrolysis_1 = Hydrolysis(
        component_id="hydrolysis_1",
        feedstock=feedstock,
        V_liq=V_LIQ_HYDRO,
        V_gas=V_GAS_HYDRO,
        T_ad=T_AD_HYDRO,
        name="Hydrolysis Pre-treatment",
    )
    hydrolysis_1.initialize(
        {
            "adm1_state": adm1_state,
            "Q_substrates": Q_SUB,
        }
    )
    plant.add_component(hydrolysis_1)

    # External gas storage for hydrolysis stage (required for CHP gas supply logic)
    storage_hydro = GasStorage(
        component_id="hydrolysis_1_storage",
        storage_type="membrane",
        capacity_m3=max(50.0, V_GAS_HYDRO),
        name="Hydrolysis Gas Storage",
    )
    plant.add_component(storage_hydro)
    configurator.connect("hydrolysis_1", "hydrolysis_1_storage", "gas")

    # --- Stage 2: Main methanogenesis digester (mesophilic, 35 C) ---
    print(f"  Main digester:     {V_LIQ_2:.0f} m3, {T_AD_2 - 273.15:.0f} C mesophilic")
    configurator.add_digester(
        digester_id="digester_2",
        V_liq=V_LIQ_2,
        V_gas=V_GAS_2,
        T_ad=T_AD_2,
        name="Main Methanogenesis Digester",
        load_initial_state=adm1_state is not None,
        initial_state_file=str(initial_state_file) if adm1_state else None,
        Q_substrates=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # receives effluent from hydrolysis only
    )

    # --- Mixers ---
    print("  Mixer 1 (hydrolysis): propeller, high intensity, 30% duty cycle")
    mixer_1 = Mixer(
        component_id="mixer_1",
        mixer_type="propeller",
        tank_volume=V_LIQ_HYDRO,
        mixing_intensity="high",
        power_installed=7.5,
        intermittent=True,
        on_time_fraction=0.30,
        name="Hydrolysis Agitator",
    )
    plant.add_component(mixer_1)

    print("  Mixer 2 (digester):   propeller, medium, 25% duty cycle")
    mixer_2 = Mixer(
        component_id="mixer_2",
        mixer_type="propeller",
        tank_volume=V_LIQ_2,
        mixing_intensity="medium",
        power_installed=22.0,
        intermittent=True,
        on_time_fraction=0.25,
        name="Main Digester Agitator",
    )
    plant.add_component(mixer_2)

    # --- CHP (500 kW_el) ---
    print("  CHP: 500 kW_el, eta_el=40%, eta_th=45%")
    configurator.add_chp(
        chp_id="chp_1",
        P_el_nom=P_EL_NOM,
        eta_el=ETA_EL,
        eta_th=ETA_TH,
        name="CHP Engine",
    )

    # --- Heating systems ---
    print(f"  Heating H1: UA=0.6 kW/K, setpoint {T_AD_HYDRO - 273.15:.0f} C")
    configurator.add_heating(
        heating_id="heating_1",
        target_temperature=T_AD_HYDRO,
        heat_loss_coefficient=0.6,
        name="Hydrolysis Heating",
    )
    print(f"  Heating H2: UA=0.8 kW/K, setpoint {T_AD_2 - 273.15:.0f} C")
    configurator.add_heating(
        heating_id="heating_2",
        target_temperature=T_AD_2,
        heat_loss_coefficient=0.8,
        name="Main Digester Heating",
    )

    # ------------------------------------------------------------------
    # 4. Connections
    # ------------------------------------------------------------------
    print("\n  Connections:")
    print("    Liquid: Hydrolysis -> Main Digester (cascade)")
    print("    Gas:    Hydrolysis -> GasStorage_H -> CHP")
    print("    Gas:    Digester2  -> GasStorage_D -> CHP")
    print("    Heat:   CHP -> Heating H1, Heating H2")

    configurator.connect("hydrolysis_1", "digester_2", "liquid")
    configurator.auto_connect_digester_to_chp("hydrolysis_1", "chp_1")
    configurator.auto_connect_digester_to_chp("digester_2", "chp_1")
    configurator.auto_connect_chp_to_heating("chp_1", "heating_1")
    configurator.auto_connect_chp_to_heating("chp_1", "heating_2")

    # ------------------------------------------------------------------
    # 5. Initialize
    # ------------------------------------------------------------------
    plant.initialize()

    # Plant component summary
    print("\n" + _hdr)
    print(f"  PLANT SUMMARY: {plant.plant_name}")
    print(f"  {'ID':<30s}  {'Type':<12s}  Name")
    print("  " + "-" * 67)
    for _cid, _comp in sorted(
        plant.components.items(),
        key=lambda x: (x[1].component_type.value, x[0]),
    ):
        print(f"  {_cid:<30s}  {_comp.component_type.value:<12s}  {_comp.name}")
    print(f"\n  Plant components:   {len(plant.components)}")
    print("  Standalone:          7  " "(silo, manure tank, screw feeder, PC pump, transfer pump, separator, boiler)")
    print(_hdr)

    # ------------------------------------------------------------------
    # 6. Simulation loop  (custom loop for step-by-step live output)
    # ------------------------------------------------------------------
    print(f"\n  Starting simulation: {DURATION:.0f} days, dt={DT*24:.0f} h")
    print(f"  Ambient temperature: {T_AMBIENT - 273.15:.1f} C\n")

    _live_hdr = (
        f"  {'Day':>4}  {'Q_gas':>8}  {'Q_CH4':>8}  {'CH4%':>5}  "
        f"{'pH H1':>6}  {'pH D2':>6}  {'FOS/TAC1':>9}  "
        f"{'P_el kW':>8}  {'Silo t':>7}  {'Tank m3':>8}"
    )
    print(_live_hdr)
    print("  " + "-" * (len(_live_hdr) - 2))

    n_steps = int(DURATION / DT)
    save_every = max(1, int(SAVE_INTERVAL / DT))
    all_results = []

    for step_idx in range(n_steps):
        t = step_idx * DT

        # Substrate management
        silo_out = silo.step(
            t,
            DT,
            {
                "withdrawal_rate": Q_MAIZE * DENSITY_MAIZE / 1000.0,  # t/d
            },
        )
        tank_out = tank.step(
            t,
            DT,
            {
                "withdrawal_rate": Q_MANURE,
                "refill_amount": 20.0 if tank.current_level < 0.30 * tank.capacity else 0.0,
            },
        )
        feeder_maize_out = feeder_maize.step(
            t,
            DT,
            {
                "Q_setpoint": Q_MAIZE,
                "substrate_available": silo_out["current_level"],
            },
        )
        feeder_manure_out = feeder_manure.step(
            t,
            DT,
            {
                "Q_setpoint": Q_MANURE,
                "substrate_available": tank_out["current_level"],
            },
        )
        pump_feed_out = pump_feed.step(t, DT, {"Q_setpoint": Q_MAIZE + Q_MANURE})

        # Main plant step: hydrolysis, digester, gas storages, CHP, heating
        # plant.step() increments simulation_time internally
        component_outputs = plant.step(DT)

        # Separator: solid/liquid split of main digester effluent
        sep_out = separator.step(
            t,
            DT,
            {
                "Q_in": component_outputs.get("digester_2", {}).get("Q_out", 0.0),
                "state_out": component_outputs.get("digester_2", {}).get("state_out"),
            },
        )

        # Boiler: cover residual heat demand not met by CHP waste heat.
        # Gas available = total produced minus CHP consumption (approximate).
        _h1_out = component_outputs.get("heating_1", {})
        _h2_out = component_outputs.get("heating_2", {})
        _chp_out = component_outputs.get("chp_1", {})
        _p_aux_total = _h1_out.get("P_aux_heat", 0.0) + _h2_out.get("P_aux_heat", 0.0)
        _q_gas_produced = component_outputs.get("hydrolysis_1", {}).get("Q_gas", 0.0) + component_outputs.get(
            "digester_2", {}
        ).get("Q_gas", 0.0)
        _q_gas_remaining = max(0.0, _q_gas_produced - _chp_out.get("Q_gas_consumed", 0.0))
        boiler_out = boiler.step(
            t,
            DT,
            {
                "P_th_demand": _p_aux_total,
                "Q_gas_available_m3_per_day": _q_gas_remaining,
            },
        )

        # Collect daily snapshots
        if step_idx % save_every == 0:
            component_outputs["silo_maize"] = silo_out
            component_outputs["tank_manure"] = tank_out
            component_outputs["feeder_maize"] = feeder_maize_out
            component_outputs["feeder_manure"] = feeder_manure_out
            component_outputs["pump_feed"] = pump_feed_out
            component_outputs["separator_1"] = sep_out
            component_outputs["boiler_1"] = boiler_out
            all_results.append({"time": t, "components": component_outputs})

            # Live output row
            _h1 = component_outputs.get("hydrolysis_1", {})
            _d2 = component_outputs.get("digester_2", {})
            _chp = component_outputs.get("chp_1", {})
            _qg = _h1.get("Q_gas", 0) + _d2.get("Q_gas", 0)
            _qc = _h1.get("Q_ch4", 0) + _d2.get("Q_ch4", 0)
            _pct = 100 * _qc / _qg if _qg > 0 else 0.0
            _fos = _h1.get("VFA", 0) / _h1.get("TAC", 1) if _h1.get("TAC", 0) > 0 else 0.0
            print(
                f"  {t:>4.0f}  {_qg:>8.1f}  {_qc:>8.1f}  {_pct:>4.1f}%  "
                f"{_h1.get('pH', 0):>6.2f}  {_d2.get('pH', 0):>6.2f}  "
                f"{_fos:>9.3f}  "
                f"{_chp.get('P_el', 0):>8.1f}  "
                f"{silo_out.get('current_level', 0):>7.1f}  "
                f"{tank_out.get('current_level', 0):>8.1f}"
            )

    print(f"\n  Simulation complete. {len(all_results)} daily snapshots.")

    # ------------------------------------------------------------------
    # 7. Results analysis
    # ------------------------------------------------------------------
    final = all_results[-1]["components"]

    def _status(value, ok_low, ok_high, warn_low=None, warn_high=None):
        """Return STABLE / OBSERVE / CRITICAL based on threshold ranges."""
        if warn_low is None:
            warn_low = ok_low - (ok_high - ok_low) * 0.5
        if warn_high is None:
            warn_high = ok_high + (ok_high - ok_low) * 0.5
        if ok_low <= value <= ok_high:
            return "STABLE    "
        if warn_low <= value <= warn_high:
            return "OBSERVE   "
        return "CRITICAL  "

    # ----------------------------------------------------------------
    print("\n" + _hdr)
    print("  A) SUBSTRATE MANAGEMENT  (final values after 30 days)")
    print(_hdr)
    s = final.get("silo_maize", {})
    m = final.get("tank_manure", {})
    fm = final.get("feeder_maize", {})
    fn = final.get("feeder_manure", {})
    pp = final.get("pump_feed", {})

    q_silo_withdrawn = Q_MAIZE * DENSITY_MAIZE / 1000.0 * DURATION  # t FM
    q_tank_withdrawn = Q_MANURE * DURATION  # m3

    print("\n  Corn silage silo:")
    print(f"    Level:             {s.get('current_level', 0):>8.1f} t    " f"({s.get('utilization', 0)*100:.1f} %)")
    print(f"    Quality factor:    {s.get('quality_factor', 1.0):>8.3f}       (target: > 0.95)")
    print(f"    Total withdrawal: ~{q_silo_withdrawn:>8.1f} t    in {DURATION:.0f} days")
    print("\n  Manure tank:")
    print(f"    Level:             {m.get('current_level', 0):>8.1f} m3   " f"({m.get('utilization', 0)*100:.1f} %)")
    print(f"    Quality factor:    {m.get('quality_factor', 1.0):>8.3f}")
    print(f"    Total withdrawal: ~{q_tank_withdrawn:>8.1f} m3   in {DURATION:.0f} days")
    print("\n  Feeding equipment (last time step):")
    print(
        f"    Screw feeder:    {fm.get('Q_actual', 0.0):>6.2f} m3/d  "
        f"P={fm.get('P_consumed', 0.0):.2f} kW  "
        f"running: {'Yes' if fm.get('is_running') else 'No'}"
    )
    print(f"    PC pump manure:  {fn.get('Q_actual', 0.0):>6.2f} m3/d  " f"P={fn.get('P_consumed', 0.0):.2f} kW")
    print(
        f"    Transfer pump:   {pp.get('Q_actual', 0.0):>6.1f} m3/d  "
        f"P={pp.get('P_consumed', 0.0):.2f} kW  "
        f"eta={pp.get('efficiency', 0.0)*100:.1f}%  "
        f"spec={pp.get('specific_energy', 0.0):.2f} kWh/m3"
    )

    # ----------------------------------------------------------------
    print("\n" + _hdr)
    print("  B) BIOLOGICAL PROCESS")
    print(_hdr)
    h1 = final.get("hydrolysis_1", {})
    d2 = final.get("digester_2", {})

    ph_h1 = h1.get("pH", 0.0)
    vfa_h1 = h1.get("VFA", 0.0)
    tac_h1 = h1.get("TAC", 0.0)
    fos_h1 = vfa_h1 / tac_h1 if tac_h1 > 0 else 0.0

    ph_d2 = d2.get("pH", 0.0)
    vfa_d2 = d2.get("VFA", 0.0)
    tac_d2 = d2.get("TAC", 0.0)
    fos_d2 = vfa_d2 / tac_d2 if tac_d2 > 0 else 0.0

    olr_h1 = (Q_MAIZE * OTS_MAIZE + Q_MANURE * OTS_MANURE) / V_LIQ_HYDRO
    olr_d2 = (Q_MAIZE * OTS_MAIZE + Q_MANURE * OTS_MANURE) / V_LIQ_2
    hrt_h1 = V_LIQ_HYDRO / (Q_MAIZE + Q_MANURE)
    hrt_d2 = V_LIQ_2 / (Q_MAIZE + Q_MANURE)

    print(f"\n  --- Hydrolysis stage ({T_AD_HYDRO - 273.15:.0f} C thermophilic) ---")
    print(f"  OLR:      {olr_h1:>7.2f} kg VS/(m3*d)  [Benchmark: 2-4, Critical: >5]")
    print(f"  HRT:      {hrt_h1:>7.1f} d             [Benchmark: 20-40 d]")
    print(f"  pH:       {ph_h1:>7.2f}               [{_status(ph_h1, 7.0, 7.8, 6.8, 8.2)}]  " f"Benchmark: 7.0-7.8")
    print(f"  VFA:      {vfa_h1:>7.2f} g/L            [Benchmark: < 3 g/L]")
    print(f"  TAC:      {tac_h1:>7.2f} g CaCO3/L")
    print(
        f"  FOS/TAC:  {fos_h1:>7.3f}               [{_status(fos_h1, 0.0, 0.3, 0.0, 0.4)}]  "
        f"Benchmark: <0.3 stable, >0.4 critical"
    )

    print(f"\n  --- Main digester ({T_AD_2 - 273.15:.0f} C mesophilic) ---")
    print(f"  OLR:      {olr_d2:>7.2f} kg VS/(m3*d)  [Benchmark: 2-4, Critical: >5]")
    print(f"  HRT:      {hrt_d2:>7.1f} d             [Benchmark: 20-40 d]")
    print(f"  pH:       {ph_d2:>7.2f}               [{_status(ph_d2, 7.0, 7.8, 6.8, 8.2)}]")
    print(f"  VFA:      {vfa_d2:>7.2f} g/L")
    print(f"  TAC:      {tac_d2:>7.2f} g CaCO3/L")
    print(f"  FOS/TAC:  {fos_d2:>7.3f}               [{_status(fos_d2, 0.0, 0.3, 0.0, 0.4)}]")

    # ----------------------------------------------------------------
    print("\n" + _hdr)
    print("  C) GAS PRODUCTION & CHP")
    print(_hdr)
    chp = final.get("chp_1", {})
    gs1 = final.get("hydrolysis_1_storage", {})
    gs2 = final.get("digester_2_storage", {})
    fla = final.get("chp_1_flare", {})

    q_gas_h1 = h1.get("Q_gas", 0.0)
    q_ch4_h1 = h1.get("Q_ch4", 0.0)
    q_gas_d2 = d2.get("Q_gas", 0.0)
    q_ch4_d2 = d2.get("Q_ch4", 0.0)
    q_gas_tot = q_gas_h1 + q_gas_d2
    q_ch4_tot = q_ch4_h1 + q_ch4_d2
    ch4_pct = 100.0 * q_ch4_tot / q_gas_tot if q_gas_tot > 0 else 0.0

    fm_per_day = (Q_MAIZE * DENSITY_MAIZE + Q_MANURE * DENSITY_MANURE) / 1000.0  # t FM/d
    biogas_yield = q_gas_tot / fm_per_day if fm_per_day > 0 else 0.0  # Nm3/t FM

    print(f"\n  Hydrolysis stage: Q_gas={q_gas_h1:7.1f} m3/d  Q_CH4={q_ch4_h1:7.1f} m3/d")
    print(f"  Main digester:    Q_gas={q_gas_d2:7.1f} m3/d  Q_CH4={q_ch4_d2:7.1f} m3/d")
    print(f"  Total:            Q_gas={q_gas_tot:7.1f} m3/d  Q_CH4={q_ch4_tot:7.1f} m3/d")
    print(f"  CH4 fraction:     {ch4_pct:7.1f} %          [Benchmark: 52-58 %]")
    print(f"  Biogas yield:     {biogas_yield:7.0f} Nm3/t FM  [Benchmark: ~160 Nm3/t FM (mix)]")

    print(
        f"\n  Gas storage H1:  {gs1.get('stored_volume_m3', 0.0):6.1f} m3  "
        f"({gs1.get('utilization', 0.0)*100:.1f}%)  "
        f"P={gs1.get('pressure_bar', 0.0):.3f} bar"
    )
    print(
        f"  Gas storage D2:  {gs2.get('stored_volume_m3', 0.0):6.1f} m3  "
        f"({gs2.get('utilization', 0.0)*100:.1f}%)  "
        f"P={gs2.get('pressure_bar', 0.0):.3f} bar"
    )

    fla_vent = fla.get("vented_volume_m3", 0.0) if fla else 0.0
    print(f"  Safety flare:    {fla_vent:6.3f} m3/d")

    p_el = chp.get("P_el", 0.0)
    p_th = chp.get("P_th", 0.0)
    q_gas_chp = chp.get("Q_gas_consumed", 0.0)

    blr = final.get("boiler_1", {})
    q_gas_blr = blr.get("Q_gas_consumed_m3_per_day", 0.0)
    q_ng_blr = blr.get("Q_natural_gas_m3_per_day", 0.0)
    q_gas_total_consumed = q_gas_chp + q_gas_blr
    q_gas_surplus = max(0.0, q_gas_tot - q_gas_total_consumed)

    print("\n  CHP:")
    print(f"    Gas consumption: {q_gas_chp:7.1f} m3/d")
    print(f"    P_el (gross):    {p_el:7.1f} kW   [Nominal: {P_EL_NOM:.0f} kW]")
    print(f"    P_th:            {p_th:7.1f} kW")
    print("\n  Auxiliary boiler:")
    print(f"    Biogas consumed: {q_gas_blr:7.1f} m3/d")
    print(f"    Natural gas:     {q_ng_blr:7.1f} m3/d  (grid backup)")
    print(
        f"    P_th_supplied:   {blr.get('P_th_supplied', 0.0):7.1f} kW  "
        f"(load={blr.get('load_fraction', 0.0)*100:.0f}%,  "
        f"eta={blr.get('efficiency_actual', 0.0)*100:.1f}%)"
    )
    print("\n  Gas balance:")
    print(f"    Produced:        {q_gas_tot:7.1f} m3/d")
    print(f"    CHP + Boiler:    {q_gas_total_consumed:7.1f} m3/d")
    print(f"    Surplus/storage: {q_gas_surplus:7.1f} m3/d")

    # ----------------------------------------------------------------
    print("\n" + _hdr)
    print("  D) ENERGY BALANCE")
    print(_hdr)
    h1_heat = final.get("heating_1", {})
    h2_heat = final.get("heating_2", {})
    m1 = final.get("mixer_1", {})
    m2 = final.get("mixer_2", {})
    sep_f = final.get("separator_1", {})

    p_mix_1 = m1.get("P_average", 0.0)
    p_mix_2 = m2.get("P_average", 0.0)
    p_feeder = feeder_maize.outputs_data.get("P_consumed", 0.0) + feeder_manure.outputs_data.get("P_consumed", 0.0)
    p_pump = pump_feed.outputs_data.get("P_consumed", 0.0)
    p_sep = sep_f.get("P_consumed", 0.0)
    p_own_use = p_mix_1 + p_mix_2 + p_feeder + p_pump + p_sep
    p_net = p_el - p_own_use
    own_use_pct = 100.0 * p_own_use / p_el if p_el > 0 else 0.0

    blr_f = final.get("boiler_1", {})
    p_boiler = blr_f.get("P_th_supplied", 0.0)

    q_heat_1 = h1_heat.get("Q_heat_supplied", 0.0)
    q_heat_2 = h2_heat.get("Q_heat_supplied", 0.0)
    q_heat = q_heat_1 + q_heat_2
    p_th_used = h1_heat.get("P_th_used", 0.0) + h2_heat.get("P_th_used", 0.0)
    # Remaining demand covered by boiler; recalculate coverage including boiler
    p_heat_covered = p_th_used + p_boiler
    heat_cov = 100.0 * p_heat_covered / q_heat if q_heat > 0 else 0.0
    chp_cov = 100.0 * p_th_used / q_heat if q_heat > 0 else 0.0

    print(f"""
  Power generation:
    Gross power (CHP):           {p_el:>8.1f} kW
    - Agitator hydrolysis:       {p_mix_1:>8.2f} kW   [30% duty cycle]
    - Agitator main digester:    {p_mix_2:>8.2f} kW   [25% duty cycle]
    - Feeding equipment:         {p_feeder:>8.2f} kW
    - Transfer pump:             {p_pump:>8.2f} kW
    - Screw press separator:     {p_sep:>8.2f} kW
    Total own consumption:       {p_own_use:>8.2f} kW   ({own_use_pct:.1f}%)  [Benchmark: 8-12%]
    -------------------------------------------------------
    Net power:                   {p_net:>8.1f} kW

  Heat balance:
    Total heat demand:           {q_heat:>8.1f} kW
    CHP waste heat used:         {p_th_used:>8.1f} kW   ({chp_cov:.1f}% coverage)
    Auxiliary boiler:            {p_boiler:>8.1f} kW   (biogas+natgas backup)
    Total heat covered:          {p_heat_covered:>8.1f} kW   ({heat_cov:.1f}%)  [Target: 100%]""")

    # ----------------------------------------------------------------
    print("\n" + _hdr)
    print("  E) SEPARATOR / DIGESTATE TREATMENT  (final values)")
    print(_hdr)
    sep_f = final.get("separator_1", {})
    q_in_sep = d2.get("Q_out", 0.0)

    print(f"\n  Input:   Q_in = {q_in_sep:>7.1f} m3/d  (effluent from main digester)")
    print("\n  Liquid fraction (separated effluent):")
    print(f"    Q_liquid:        {sep_f.get('Q_liquid', 0.0):>7.2f} m3/d")
    ts_liq = sep_f.get("TS_liquid", 0.0)
    print(f"    TS_liquid:       {ts_liq:>7.1f} kg/m3  (~{ts_liq / 10.0:.1f}% TS)")
    print(f"    TAN_liquid:      {sep_f.get('TAN_liquid', 0.0):>7.3f} kg/m3  (plant-available N)")
    print("\n  Solid fraction (press cake):")
    print(f"    Q_solid:         {sep_f.get('Q_solid', 0.0):>7.3f} m3/d")
    ts_sol = sep_f.get("TS_solid", 0.0)
    print(f"    TS_solid:        {ts_sol:>7.0f} kg/m3  (~{ts_sol / 9.0:.1f}% TS)")
    print(f"    Solid recovery:  {sep_f.get('recovery_solid_pct', 0.0):>7.1f}%  of input TS captured")
    print(
        f"\n  Power:   {sep_f.get('P_consumed', 0.0):.2f} kW  "
        f"(spec. {separator.specific_energy:.1f} kWh/t FM input, "
        f"KTBL 2013)"
    )

    # ----------------------------------------------------------------
    print("\n" + _hdr)
    print("  F) PROCESS STABILITY ASSESSMENT  (final values)")
    print(_hdr)
    checks = [
        ("pH Hydrolysis", ph_h1, 7.0, 7.8, 6.8, 8.2),
        ("pH Main digester", ph_d2, 7.0, 7.8, 6.8, 8.2),
        ("FOS/TAC Hydrolysis", fos_h1, 0.0, 0.3, 0.0, 0.4),
        ("FOS/TAC Main digester", fos_d2, 0.0, 0.3, 0.0, 0.4),
        ("VFA Hydrolysis [g/L]", vfa_h1, 0.0, 3.0, 0.0, 5.0),
        ("VFA Main digester [g/L]", vfa_d2, 0.0, 3.0, 0.0, 5.0),
        ("OLR Hydrolysis", olr_h1, 1.0, 4.0, 0.5, 5.5),
        ("OLR Main digester", olr_d2, 1.0, 4.0, 0.5, 5.5),
        ("CH4 fraction [%]", ch4_pct, 50.0, 70.0, 45.0, 75.0),
        ("Net power [kW]", p_net, 0.0, 9999, 0.0, 9999),
        ("Total heat coverage [%]", heat_cov, 100.0, 100.0, 90.0, 100.0),
        ("CHP heat share [%]", chp_cov, 80.0, 100.0, 50.0, 100.0),
        ("Own consumption [%]", own_use_pct, 0.0, 12.0, 0.0, 15.0),
    ]

    print()
    for label, val, ok_lo, ok_hi, warn_lo, warn_hi in checks:
        status = _status(val, ok_lo, ok_hi, warn_lo, warn_hi)
        print(f"  {label:<30s}  {val:>8.2f}   [{status}]")

    print("\n" + _hdr)
    print("  Simulation completed successfully.")
    print(_hdr + "\n")

    return (
        plant,
        all_results,
        {
            "silo": silo,
            "tank": tank,
            "feeder_maize": feeder_maize,
            "feeder_manure": feeder_manure,
            "pump_feed": pump_feed,
            "separator": separator,
            "boiler": boiler,
        },
    )


if __name__ == "__main__":
    plant, results, standalone = main()
