"""Shared helpers for sensor demonstration examples."""

from __future__ import annotations

from typing import Any, Dict, Tuple


def build_single_digester_plant() -> Tuple[Any, Any, str, list[float]]:
    """
    Build a simple one-fermenter biogas plant with maize silage + swine manure.

    Returns:
        Tuple of (plant, digester, storage_id, q_substrates)
    """
    from pyadm1.configurator.plant_builder import BiogasPlant
    from pyadm1.configurator.plant_configurator import PlantConfigurator
    from pyadm1.substrates.feedstock import Feedstock

    feedstock = Feedstock(
        ["maize_silage_milk_ripeness", "swine_manure"],
        feeding_freq=48,
        total_simtime=60,
    )
    q_substrates = [15.0, 10.0]

    plant = BiogasPlant("Single Digester Sensor Demo")
    configurator = PlantConfigurator(plant, feedstock)
    digester, _ = configurator.add_digester(
        digester_id="main_digester",
        V_liq=2000.0,
        V_gas=300.0,
        T_ad=308.15,
        name="Main Digester",
        Q_substrates=q_substrates,
    )
    configurator.add_chp(
        chp_id="chp_1",
        P_el_nom=500.0,
        eta_el=0.40,
        eta_th=0.45,
        name="Sensor Demo BHKW",
    )
    configurator.auto_connect_digester_to_chp("main_digester", "chp_1")
    plant.initialize()

    return plant, digester, "main_digester_storage", q_substrates


def substrate_feed_profile(t_days: float) -> list[float]:
    """
    Demo substrate feed profile [m3/d].

    Maize silage stays constant at 15 m3/d.
    Swine manure varies smoothly between 8 and 10 m3/d.
    """
    import numpy as np

    swine_manure = 9.0 + 1.0 * np.sin(2.0 * np.pi * t_days / 2.0)
    return [15.0, float(swine_manure)]


def apply_substrate_feed(digester: Any, q_substrates: list[float]) -> None:
    """Update the current digester substrate feed profile."""
    digester.Q_substrates = list(q_substrates)
    if getattr(digester, "state", None) is not None:
        digester.state["Q_substrates"] = list(q_substrates)


def digester_temperature_profile(t_days: float) -> float:
    """
    Smooth operating-temperature profile for the demo digester [K].

    The profile oscillates around 35°C with a slow and a fast component so the
    temperature sensor and the process both see a changing signal.
    """
    import numpy as np

    return float(308.15 + 1.8 * np.sin(2.0 * np.pi * t_days / 3.5) + 0.4 * np.sin(2.0 * np.pi * t_days))


def apply_digester_temperature(digester: Any, temperature_k: float) -> None:
    """Push a new operating temperature into the digester and wrapped ADM1 model."""
    digester.T_ad = float(temperature_k)
    if hasattr(digester, "adm1"):
        digester.adm1._T_ad = float(temperature_k)


def extract_physical_signals(digester: Any, digester_out: Dict[str, Any], storage_out: Dict[str, Any]) -> Dict[str, float]:
    """Build physical sensor signals from plant outputs."""
    return {
        "current_level": float(storage_out.get("stored_volume_m3", 0.0)),
        "Q_actual": float(digester_out.get("Q_out", 0.0)),
        "pressure_bar": float(storage_out.get("pressure_bar", 0.0)),
        "temperature": float(getattr(digester, "T_ad", 308.15)),
        "pH": float(digester_out.get("pH", 7.0)),
    }


def extract_chemical_signals(digester_out: Dict[str, Any]) -> Dict[str, float]:
    """
    Build chemical sensor signals from digester outputs and ADM1 effluent state.

    Notes:
    - NH3 is derived from ADM1 free ammonia state S_nh3.
    - COD is approximated from COD-based soluble and particulate ADM1 states.
    - Phosphate is approximated from the generic ADM1 anion pool plus particulate
      organic matter because phosphate is not explicitly tracked as a dedicated
      state in this model.
    """
    state = digester_out.get("state_out", [])
    if not state:
        return {"VFA": 0.0, "NH3": 0.0, "COD": 0.0, "nitrogen": 0.0, "phosphate": 0.0}

    cod_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 27, 28, 29, 30]
    cod_g_l = max(0.0, sum(float(state[idx]) for idx in cod_indices))

    free_ammonia_g_l = max(0.0, float(state[32]) * 14.0)
    total_nitrogen_mg_l = max(0.0, float(state[10] + state[32]) * 14000.0)

    phosphate_from_anions = max(0.0, float(state[26]) * 31000.0 * 0.12)
    phosphate_from_organics = max(0.0, 35.0 * float(state[14]) + 20.0 * float(state[24]))
    phosphate_mg_l = max(10.0, phosphate_from_anions + phosphate_from_organics)

    return {
        "VFA": float(digester_out.get("VFA", 0.0)),
        "NH3": free_ammonia_g_l,
        "COD": cod_g_l,
        "nitrogen": total_nitrogen_mg_l,
        "phosphate": phosphate_mg_l,
    }


def extract_gas_signals(digester_out: Dict[str, Any], q_substrates: list[float]) -> Dict[str, float]:
    """
    Build gas sensor signals from digester gas production and process state.

    Notes:
    - CH4 and CO2 are derived directly from modelled gas flows.
    - H2S, O2, and trace gas values are plant-linked approximations because
      these species are not explicitly output by the current digester model.
    """
    q_gas = max(1.0e-9, float(digester_out.get("Q_gas", 0.0)))
    q_ch4 = max(0.0, float(digester_out.get("Q_ch4", 0.0)))
    q_co2 = max(0.0, float(digester_out.get("Q_co2", 0.0)))
    vfa = max(0.0, float(digester_out.get("VFA", 0.0)))
    ph = float(digester_out.get("pH", 7.0))
    manure_feed = float(q_substrates[1]) if len(q_substrates) > 1 else 0.0

    ch4_pct = max(0.0, min(100.0, 100.0 * q_ch4 / q_gas))
    co2_pct = max(0.0, min(100.0, 100.0 * q_co2 / q_gas))

    h2s_ppm = max(50.0, 180.0 + 22.0 * manure_feed + 28.0 * vfa + 120.0 * max(0.0, 7.15 - ph))
    o2_pct = max(0.02, min(1.0, 0.06 + 4.0 / max(q_gas, 80.0)))
    trace_ppm = max(1.0, 12.0 + 0.035 * h2s_ppm + 0.015 * q_gas)

    return {
        "CH4": ch4_pct,
        "CO2": co2_pct,
        "H2S": h2s_ppm,
        "O2": o2_pct,
        "trace_gas": trace_ppm,
    }
