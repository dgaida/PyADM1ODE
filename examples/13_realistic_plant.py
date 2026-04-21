#!/usr/bin/env python3
# ============================================================================
# examples/13_realistic_plant.py
# ============================================================================
"""

Model selector
--------------
Set ``MODEL_TYPE = "adm1"`` (legacy Digester/Hydrolysis back-end) or
``MODEL_TYPE = "adm1da"`` (SIMBA# ADM1da extension, 41-state).  The fermenter
factory ``_make_fermenter()`` returns the appropriate component class so the
plant topology stays identical across back-ends.  Both values are wired
end-to-end in the Component framework.
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ============================================================================
# Model selector  — toggle ADM1 vs ADM1da here
# ============================================================================
MODEL_TYPE: str = "adm1da"  # "adm1" or "adm1da"


# ============================================================================
# Plant geometry  (from R&I 0199_BGA_Perl-Borg_A101b + plant data sheet)
# ============================================================================
# Tank volumes and feed rates (as per owner's data sheet):
#   Dosierung (solid-feed throughput via D1)   40   m³/d   (daily dose)
#   Vorgrube V (reception pit)                200   m³     gross
#   Fermenter F1                             1416   m³     gross
#   Nachgärer N                              1416   m³     gross
#   Gärproduktlager G                        3971   m³     gross
# Liquid/gas split follows ÖKOBiT standard: gas dome ≈ 15 % of gross for
# heated reactors, ≈ 12 % for the larger cold digestate-storage dome.

# ------- Feststoffdosierung D1  (daily solid-feed throughput) ---------------
Q_DOSING = 40.0  # m³/d  nominal solid-feed dose rate via D1 screw feeder

# ------- Vorgrube V  (reception pit, Øi = 8.00 m, h ≈ 4.00 m, ~200 m³) ------
V_LIQ_V = 180.0  # m³  liquid  (free-board ~10 %)
V_GAS_V = 20.0  # m³  vent volume (covered pit)
T_AD_V = 293.15  # K   20 °C — unheated reception pit

# ------- Fermenter F1  (BA19, h = 5.80 m, V_gross = 1416 m³) ----------------
V_LIQ_F1 = 1200.0  # m³  liquid  (85 % fill)
V_GAS_F1 = 216.0  # m³  zeltdach dome (15 %)
T_AD_F1 = 315.15  # K   42 °C

# ------- Nachgärer N  (identical to F1, V_gross = 1416 m³) ------------------
V_LIQ_N = 1200.0  # m³
V_GAS_N = 216.0  # m³
T_AD_N = 315.15  # K   42 °C

# ------- Gärproduktlager G  (gas-tight storage, V_gross = 3971 m³) ----------
V_LIQ_G = 3495.0  # m³  liquid  (88 % fill)
V_GAS_G = 476.0  # m³  dome (12 %)
T_AD_G = 308.15  # K   35 °C — reduced heating, residual methanogenesis

# ------- CHP (BHKW) ---------------------------------------------------------
P_EL_NOM = 500.0  # kW_el  500 kW class
ETA_EL = 0.40  # [-]
ETA_TH = 0.45  # [-]

# ------- Auxiliary boiler ---------------------------------------------------
P_BOILER_NOM = 150.0  # kW_th   EN 303-1 condensing gas boiler


# ============================================================================
# Fermenter factory  — routes ADM1 / ADM1da to the correct component
# ============================================================================


def _make_fermenter(
    kind: str,  # "hydrolysis" | "digester"
    component_id: str,
    feedstock,
    V_liq: float,
    V_gas: float,
    T_ad: float,
    name: str,
):
    """
    Build a biological reactor component that respects the active MODEL_TYPE.

    Parameters
    ----------
    kind : {"hydrolysis", "digester"}
        Role of the reactor in the plant train.
    component_id, feedstock, V_liq, V_gas, T_ad, name
        Passed through to the underlying component constructor.

    Returns
    -------
    Component
        Either a ``Hydrolysis`` or ``Digester`` instance.

    Notes
    -----
    ``MODEL_TYPE == "adm1"`` returns the legacy ``Hydrolysis`` / ``Digester``
    wrapper around :class:`pyadm1.core.adm1.ADM1` (37 states).
    ``MODEL_TYPE == "adm1da"`` returns :class:`ADM1daDigester`, which wraps
    :class:`pyadm1.core.adm1da.ADM1da` (41 states, SIMBA# biogas extension).
    The Vorgrube (``kind="hydrolysis"``) uses the same ADM1daDigester class
    under the ADM1da back-end — the low operating temperature and short HRT
    naturally suppress methanogenesis, so no separate pre-treatment class
    is required for the ADM1da topology.
    """
    from pyadm1.components.biological import (
        Hydrolysis,
        Digester,
        ADM1daDigester,
    )

    if MODEL_TYPE == "adm1":
        cls = Hydrolysis if kind == "hydrolysis" else Digester
        return cls(
            component_id=component_id,
            feedstock=feedstock,
            V_liq=V_liq,
            V_gas=V_gas,
            T_ad=T_ad,
            name=name,
        )

    if MODEL_TYPE == "adm1da":
        return ADM1daDigester(
            component_id=component_id,
            feedstock=feedstock,
            V_liq=V_liq,
            V_gas=V_gas,
            T_ad=T_ad,
            name=name,
        )

    raise ValueError(f"Unknown MODEL_TYPE='{MODEL_TYPE}'. Expected 'adm1' or 'adm1da'.")


# ============================================================================
# Plant builder
# ============================================================================


def build_plant():
    """
    Build the Perl-Borg plant topology and return the configured BiogasPlant.

    Does *not* initialise biological state, run the simulation, or define
    substrate flow rates.  Those must be supplied by the caller once the
    structure has been reviewed.
    """
    from pyadm1.configurator.plant_builder import BiogasPlant
    from pyadm1.configurator.plant_configurator import PlantConfigurator
    from pyadm1.components.energy import Boiler, GasStorage
    from pyadm1.components.mechanical.mixer import Mixer
    from pyadm1.components.mechanical.pump import Pump
    from pyadm1.components.feeding.substrate_storage import SubstrateStorage
    from pyadm1.components.feeding.feeder import Feeder
    from pyadm1.substrates.feedstock import Feedstock

    # ------------------------------------------------------------------
    # Feedstock (ADM1-compatible — substrate flows left at 0 for now)
    # ------------------------------------------------------------------
    feedstock = Feedstock(feeding_freq=24)

    plant = BiogasPlant("Biogasanlage Perl-Borg")
    cfg = PlantConfigurator(plant, feedstock)

    # ------------------------------------------------------------------
    # Substrate management — silo (maize silage) + manure reception tank
    # ------------------------------------------------------------------
    silo_maize = SubstrateStorage(
        component_id="silo_maize",
        storage_type="vertical_silo",
        substrate_type="corn_silage",
        capacity=500.0,  # t FM  (placeholder)
        initial_level=0.0,
        name="Maissilage-Silo",
    )
    plant.add_component(silo_maize)

    tank_manure = SubstrateStorage(
        component_id="tank_manure",
        storage_type="above_ground_tank",
        substrate_type="manure_liquid",
        capacity=200.0,  # m³  (placeholder)
        initial_level=0.0,
        name="Güllevorlage",
    )
    plant.add_component(tank_manure)

    # ------------------------------------------------------------------
    # Feststoffeintrag D1  (VBM05G-WC-XP, 22 kW screw, 5 auger screws)
    # Nominal daily solid-feed dose = Q_DOSING (40 m³/d) into F1.
    # ------------------------------------------------------------------
    feeder_d1 = Feeder(
        component_id="feeder_d1",
        feeder_type="screw",
        Q_max=Q_DOSING,  # m³/d  nominal dose rate (= Dosierung)
        substrate_type="solid",
        name="Feststoffeintrag D1 (VBM05G-WC-XP)",
    )
    plant.add_component(feeder_d1)

    # ------------------------------------------------------------------
    # Vorgrube V  — reception pit with paddle mixer, unheated
    # ------------------------------------------------------------------
    vorgrube = _make_fermenter(
        kind="hydrolysis",
        component_id="vorgrube_v",
        feedstock=feedstock,
        V_liq=V_LIQ_V,
        V_gas=V_GAS_V,
        T_ad=T_AD_V,
        name="Vorgrube V",
    )
    plant.add_component(vorgrube)

    mixer_v = Mixer(
        component_id="mixer_v",
        mixer_type="paddle",
        tank_volume=V_LIQ_V,
        mixing_intensity="medium",
        power_installed=15.0,  # kW  — 2G Suma 150-380
        intermittent=True,
        on_time_fraction=0.30,
        name="Paddelrührwerk V (2G Suma 15 kW)",
    )
    plant.add_component(mixer_v)

    # ------------------------------------------------------------------
    # Fermenter F1  — main methanogenesis stage
    # ------------------------------------------------------------------
    fermenter_f1 = _make_fermenter(
        kind="digester",
        component_id="fermenter_f1",
        feedstock=feedstock,
        V_liq=V_LIQ_F1,
        V_gas=V_GAS_F1,
        T_ad=T_AD_F1,
        name="Fermenter F1 (BA19)",
    )
    plant.add_component(fermenter_f1)
    storage_f1 = GasStorage(
        component_id="fermenter_f1_storage",
        storage_type="membrane",
        capacity_m3=max(50.0, V_GAS_F1),
        name="Gasspeicher F1",
    )
    plant.add_component(storage_f1)
    cfg.connect("fermenter_f1", "fermenter_f1_storage", "gas")

    mixer_f1 = Mixer(
        component_id="mixer_f1",
        mixer_type="paddle",
        tank_volume=V_LIQ_F1,
        mixing_intensity="medium",
        power_installed=15.0,
        intermittent=True,
        on_time_fraction=0.30,
        name="Paddelrührwerk F1 (RSF01F1-SP)",
    )
    plant.add_component(mixer_f1)

    # ------------------------------------------------------------------
    # Nachgärer N  — second stage
    # ------------------------------------------------------------------
    nachgaerer_n = _make_fermenter(
        kind="digester",
        component_id="nachgaerer_n",
        feedstock=feedstock,
        V_liq=V_LIQ_N,
        V_gas=V_GAS_N,
        T_ad=T_AD_N,
        name="Nachgärer N",
    )
    plant.add_component(nachgaerer_n)
    storage_n = GasStorage(
        component_id="nachgaerer_n_storage",
        storage_type="membrane",
        capacity_m3=max(50.0, V_GAS_N),
        name="Gasspeicher N",
    )
    plant.add_component(storage_n)
    cfg.connect("nachgaerer_n", "nachgaerer_n_storage", "gas")

    mixer_n = Mixer(
        component_id="mixer_n",
        mixer_type="paddle",
        tank_volume=V_LIQ_N,
        mixing_intensity="medium",
        power_installed=15.0,
        intermittent=True,
        on_time_fraction=0.25,
        name="Paddelrührwerk N (RSF02N-SP)",
    )
    plant.add_component(mixer_n)

    # ------------------------------------------------------------------
    # Gärrestlager G  — digestate storage (gas-tight, paddle mixer)
    # ------------------------------------------------------------------
    gaerrest_g = _make_fermenter(
        kind="digester",
        component_id="gaerrest_g",
        feedstock=feedstock,
        V_liq=V_LIQ_G,
        V_gas=V_GAS_G,
        T_ad=T_AD_G,
        name="Gärrestlager G",
    )
    plant.add_component(gaerrest_g)
    storage_g = GasStorage(
        component_id="gaerrest_g_storage",
        storage_type="membrane",
        capacity_m3=max(50.0, V_GAS_G),
        name="Gasspeicher G",
    )
    plant.add_component(storage_g)
    cfg.connect("gaerrest_g", "gaerrest_g_storage", "gas")

    mixer_g = Mixer(
        component_id="mixer_g",
        mixer_type="paddle",
        tank_volume=V_LIQ_G,
        mixing_intensity="low",
        power_installed=15.0,
        intermittent=True,
        on_time_fraction=0.15,
        name="Paddelrührwerk G (VSM01G-SP-XB)",
    )
    plant.add_component(mixer_g)

    # ------------------------------------------------------------------
    # Transfer pumps (Technikgebäude I)
    # ------------------------------------------------------------------
    pump_v_f1 = Pump(
        component_id="pump_v_f1",
        pump_type="progressive_cavity",
        Q_nom=30.0,  # m³/d  placeholder
        pressure_head=20.0,
        name="Übergabepumpe V -> F1 (MPT01I-SP-XB)",
    )
    plant.add_component(pump_v_f1)

    pump_f1_n = Pump(
        component_id="pump_f1_n",
        pump_type="progressive_cavity",
        Q_nom=30.0,
        pressure_head=15.0,
        name="Übergabepumpe F1 -> N (MPT02I-SP-XB)",
    )
    plant.add_component(pump_f1_n)

    pump_n_g = Pump(
        component_id="pump_n_g",
        pump_type="progressive_cavity",
        Q_nom=30.0,
        pressure_head=15.0,
        name="Übergabepumpe N -> G (MPT03I-SP-XB)",
    )
    plant.add_component(pump_n_g)

    # ------------------------------------------------------------------
    # CHP (BHKW) + automatic safety flare
    # ------------------------------------------------------------------
    cfg.add_chp(
        chp_id="bhkw_1",
        P_el_nom=P_EL_NOM,
        eta_el=ETA_EL,
        eta_th=ETA_TH,
        name="BHKW 500 kW",
    )

    # ------------------------------------------------------------------
    # Heating systems (one per heated fermenter) — hot-water header XB
    # ------------------------------------------------------------------
    cfg.add_heating(
        heating_id="heating_f1",
        target_temperature=T_AD_F1,
        heat_loss_coefficient=1.0,
        name="Heizkreis F1 (MTT01B-WH)",
    )
    cfg.add_heating(
        heating_id="heating_n",
        target_temperature=T_AD_N,
        heat_loss_coefficient=1.0,
        name="Heizkreis N (MTT02B-WH)",
    )
    cfg.add_heating(
        heating_id="heating_g",
        target_temperature=T_AD_G,
        heat_loss_coefficient=0.8,
        name="Heizkreis G",
    )

    # ------------------------------------------------------------------
    # Auxiliary boiler (dual-fuel backup)
    # ------------------------------------------------------------------
    boiler = Boiler(
        component_id="boiler_1",
        P_th_nom=P_BOILER_NOM,
        efficiency=0.92,
        fuel_type="dual",
        name="Heizkessel (Backup)",
    )
    plant.add_component(boiler)

    # ------------------------------------------------------------------
    # External hydrolysis gas storage  (V also vents gas)
    # The add_digester helper only attaches storage to components added
    # via that helper; V is built manually above, so add storage here.
    # ------------------------------------------------------------------
    storage_v = GasStorage(
        component_id="vorgrube_v_storage",
        storage_type="membrane",
        capacity_m3=max(50.0, V_GAS_V),
        name="Gasspeicher Vorgrube V",
    )
    plant.add_component(storage_v)
    cfg.connect("vorgrube_v", "vorgrube_v_storage", "gas")

    # ------------------------------------------------------------------
    # Connections (liquid cascade, gas to CHP, heat to fermenters)
    # ------------------------------------------------------------------
    # Liquid: V -> F1 -> N -> G
    cfg.connect("vorgrube_v", "fermenter_f1", "liquid")
    cfg.connect("fermenter_f1", "nachgaerer_n", "liquid")
    cfg.connect("nachgaerer_n", "gaerrest_g", "liquid")

    # Gas: V, F1, N, G -> (storage) -> CHP
    cfg.connect("vorgrube_v_storage", "bhkw_1", "gas")
    cfg.auto_connect_digester_to_chp("fermenter_f1", "bhkw_1")
    cfg.auto_connect_digester_to_chp("nachgaerer_n", "bhkw_1")
    cfg.auto_connect_digester_to_chp("gaerrest_g", "bhkw_1")

    # Heat: CHP -> heating circuits
    cfg.auto_connect_chp_to_heating("bhkw_1", "heating_f1")
    cfg.auto_connect_chp_to_heating("bhkw_1", "heating_n")
    cfg.auto_connect_chp_to_heating("bhkw_1", "heating_g")

    return plant, {
        "feedstock": feedstock,
        "silo_maize": silo_maize,
        "tank_manure": tank_manure,
        "feeder_d1": feeder_d1,
        "pump_v_f1": pump_v_f1,
        "pump_f1_n": pump_f1_n,
        "pump_n_g": pump_n_g,
        "boiler": boiler,
    }


# ============================================================================
# Main — print the assembled plant topology
# ============================================================================


def main():
    hdr = "=" * 74
    print(hdr)
    print("  PyADM1ODE — Biogasanlage Perl-Borg (R&I 0199, A101b)")
    print(f"  Model back-end: {MODEL_TYPE}")
    print(hdr)

    plant, standalone = build_plant()

    print(f"\n  Plant name            : {plant.plant_name}")
    print(f"  Managed components    : {len(plant.components)}")
    print(f"  Standalone components : {len(standalone)}")
    print(f"  Connections           : {len(plant.connections)}")

    print(f"\n  Components ({len(plant.components)}):")
    print(f"  {'ID':<28s}  {'Type':<14s}  Name")
    print("  " + "-" * 71)
    for cid, comp in sorted(
        plant.components.items(),
        key=lambda x: (x[1].component_type.value, x[0]),
    ):
        print(f"  {cid:<28s}  {comp.component_type.value:<14s}  {comp.name}")

    print(f"\n  Connections ({len(plant.connections)}):")
    for conn in plant.connections:
        print(
            f"    {conn.from_component:<26s} --[{conn.connection_type:<6s}]-->  " f"{conn.to_component}"
        )  # ASCII-only arrow for Windows cp1252

    print("\n  NOTE: Substrate feed rates and simulation duration are not set.")
    print("        Define Q_substrates and call plant.initialize() + step loop")
    print("        once the feeding scenario is specified.")
    print(hdr + "\n")

    return plant, standalone


if __name__ == "__main__":
    plant, standalone = main()
