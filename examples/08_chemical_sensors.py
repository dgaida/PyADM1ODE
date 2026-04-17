#!/usr/bin/env python3
# ============================================================================
# examples/08_chemical_sensors.py
# ============================================================================
"""
Example: chemical sensors on a simple one-fermenter biogas plant.

This example shows how to:
- build a single-digester plant with maize silage and swine manure feed
- attach chemical sensors to digester-liquid signals
- collect measured values for later analysis
- plot measured values against the real process values at the end

The example uses five chemical analyzers:
- VFA analyzer
- ammonia analyzer
- COD analyzer
- total nitrogen nutrient analyzer
- phosphate nutrient analyzer

Usage:
    py examples/08_chemical_sensors.py
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    """Run the chemical sensor example."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for this example: pip install matplotlib") from exc

    from pyadm1.components.sensors import ChemicalSensor
    from examples._sensor_demo_helpers import (
        apply_substrate_feed,
        build_single_digester_plant,
        extract_chemical_signals,
        substrate_feed_profile,
    )

    duration_days = 10.0
    dt = 1.0 / 24.0
    n_steps = int(duration_days / dt)

    plant, digester, _, _ = build_single_digester_plant()

    vfa_sensor = ChemicalSensor(
        component_id="vfa_sensor",
        sensor_type="VFA",
        analyzer_method="online_titration",
        signal_key="VFA",
        measurement_range=(0.0, 15.0),
        measurement_noise=0.06,
        accuracy=0.04,
        drift_rate=0.01,
        sample_interval=20.0 / 1440,
        measurement_delay=10.0 / 1440,
        response_time=1.0 / 1440,
        detection_limit=0.01,
        unit="g/L",
        name="VFA Analyzer",
    )
    vfa_sensor.add_input(digester.component_id)

    ammonia_sensor = ChemicalSensor(
        component_id="ammonia_sensor",
        sensor_type="ammonia",
        analyzer_method="ion_selective",
        signal_key="NH3",
        measurement_range=(0.0, 6.0),
        measurement_noise=0.03,
        accuracy=0.01,
        drift_rate=0.002,
        sample_interval=15.0 / 1440,
        measurement_delay=10.0 / 1440,
        response_time=3.0 / 1440,
        detection_limit=0.01,
        unit="g/L",
        name="Ammonia Analyzer",
    )
    ammonia_sensor.add_input(digester.component_id)

    cod_sensor = ChemicalSensor(
        component_id="cod_sensor",
        sensor_type="COD",
        analyzer_method="spectroscopy",
        signal_key="COD",
        measurement_range=(0.0, 120.0),
        measurement_noise=2.0,
        accuracy=0.5,
        drift_rate=0.03,
        sample_interval=5.0 / 1440,
        measurement_delay=3.0 / 1440,
        response_time=0.0,
        detection_limit=0.10,
        unit="g/L",
        name="COD Analyzer",
    )
    cod_sensor.add_input(digester.component_id)

    nitrogen_sensor = ChemicalSensor(
        component_id="nitrogen_sensor",
        sensor_type="nutrients",
        analyzer_method="colorimetric",
        signal_key="nitrogen",
        measurement_range=(0.0, 10000.0),
        measurement_noise=3.0,
        accuracy=2.0,
        drift_rate=0.5,
        sample_interval=60.0 / 1440,
        measurement_delay=30.0 / 1440,
        response_time=1.0 / 1440,
        detection_limit=1.0,
        unit="mg/L",
        name="Total Nitrogen Analyzer",
    )
    nitrogen_sensor.add_input(digester.component_id)

    phosphate_sensor = ChemicalSensor(
        component_id="phosphate_sensor",
        sensor_type="nutrients",
        analyzer_method="colorimetric",
        signal_key="phosphate",
        measurement_range=(0.0, 4000.0),
        measurement_noise=3.0,
        accuracy=2.0,
        drift_rate=0.5,
        sample_interval=60.0 / 1440,
        measurement_delay=30.0 / 1440,
        response_time=1.0 / 1440,
        detection_limit=1.0,
        unit="mg/L",
        name="Phosphate Analyzer",
    )
    phosphate_sensor.add_input(digester.component_id)

    history = {
        "time_days": [],
        "vfa_true": [],
        "vfa_measured": [],
        "ammonia_true": [],
        "ammonia_measured": [],
        "cod_true": [],
        "cod_measured": [],
        "nitrogen_true": [],
        "nitrogen_measured": [],
        "phosphate_true": [],
        "phosphate_measured": [],
    }

    print("=" * 76)
    print("PyADM1ODE Chemical Sensor Example")
    print("=" * 76)
    print("Single fermenter with maize silage + swine manure feed over 10 days")
    print()
    header = (
        f"{'Time [d]':>8} | "
        f"{'VFA r/m [g/L]':>17} | "
        f"{'NH3 r/m [g/L]':>17} | "
        f"{'COD r/m [g/L]':>17} | "
        f"{'N r/m [mg/L]':>17} | "
        f"{'P r/m [mg/L]':>17}"
    )
    print(header)
    print("-" * (len(header) + 2))

    for step_idx in range(n_steps):
        apply_substrate_feed(digester, substrate_feed_profile(plant.simulation_time))
        results = plant.step(dt)
        t = plant.simulation_time
        digester_out = results[digester.component_id]
        liquid_state = extract_chemical_signals(digester_out)

        vfa_out = vfa_sensor.step(t=t, dt=dt, inputs={"VFA": liquid_state["VFA"]})
        ammonia_out = ammonia_sensor.step(t=t, dt=dt, inputs={"NH3": liquid_state["NH3"]})
        cod_out = cod_sensor.step(t=t, dt=dt, inputs={"COD": liquid_state["COD"]})
        nitrogen_out = nitrogen_sensor.step(t=t, dt=dt, inputs={"nitrogen": liquid_state["nitrogen"]})
        phosphate_out = phosphate_sensor.step(t=t, dt=dt, inputs={"phosphate": liquid_state["phosphate"]})

        history["time_days"].append(t)
        history["vfa_true"].append(liquid_state["VFA"])
        history["vfa_measured"].append(vfa_out["measurement"])
        history["ammonia_true"].append(liquid_state["NH3"])
        history["ammonia_measured"].append(ammonia_out["measurement"])
        history["cod_true"].append(liquid_state["COD"])
        history["cod_measured"].append(cod_out["measurement"])
        history["nitrogen_true"].append(liquid_state["nitrogen"])
        history["nitrogen_measured"].append(nitrogen_out["measurement"])
        history["phosphate_true"].append(liquid_state["phosphate"])
        history["phosphate_measured"].append(phosphate_out["measurement"])

        if step_idx % 6 == 0:
            print(
                f"{t:8.3f} | "
                f"{liquid_state['VFA']:8.2f}/{vfa_out['measurement']:8.2f} | "
                f"{liquid_state['NH3']:8.2f}/{ammonia_out['measurement']:8.2f} | "
                f"{liquid_state['COD']:8.2f}/{cod_out['measurement']:8.2f} | "
                f"{liquid_state['nitrogen']:8.0f}/{nitrogen_out['measurement']:8.0f} | "
                f"{liquid_state['phosphate']:8.0f}/{phosphate_out['measurement']:8.0f}"
            )

    fig, axes = plt.subplots(5, 1, figsize=(11, 15), sharex=True)

    axes[0].plot(history["time_days"], history["vfa_true"], label="Real VFA", linewidth=2.0)
    axes[0].plot(history["time_days"], history["vfa_measured"], label="Measured VFA", linewidth=1.5, alpha=0.85)
    axes[0].set_ylabel(f"VFA [{vfa_sensor.unit}]")
    axes[0].set_title("VFA Analyzer")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(history["time_days"], history["ammonia_true"], label="Real NH3", linewidth=2.0)
    axes[1].plot(history["time_days"], history["ammonia_measured"], label="Measured NH3", linewidth=1.5, alpha=0.85)
    axes[1].set_ylabel(f"NH3 [{ammonia_sensor.unit}]")
    axes[1].set_title("Ammonia Analyzer")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(history["time_days"], history["cod_true"], label="Real COD", linewidth=2.0)
    axes[2].plot(history["time_days"], history["cod_measured"], label="Measured COD", linewidth=1.5, alpha=0.85)
    axes[2].set_ylabel(f"COD [{cod_sensor.unit}]")
    axes[2].set_title("COD Analyzer")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    axes[3].plot(history["time_days"], history["nitrogen_true"], label="Real total N", linewidth=2.0)
    axes[3].plot(history["time_days"], history["nitrogen_measured"], label="Measured total N", linewidth=1.5, alpha=0.85)
    axes[3].set_ylabel(f"N [{nitrogen_sensor.unit}]")
    axes[3].set_title("Total Nitrogen Analyzer")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    axes[4].plot(history["time_days"], history["phosphate_true"], label="Real phosphate", linewidth=2.0)
    axes[4].plot(history["time_days"], history["phosphate_measured"], label="Measured phosphate", linewidth=1.5, alpha=0.85)
    axes[4].set_xlabel("Time [days]")
    axes[4].set_ylabel(f"P [{phosphate_sensor.unit}]")
    axes[4].set_title("Phosphate Analyzer")
    axes[4].grid(True, alpha=0.3)
    axes[4].legend()

    fig.suptitle("Chemical Sensors on a One-Fermenter Biogas Plant")
    fig.tight_layout()

    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / "chemical_sensor_example.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {plot_path}")

    if "agg" not in plt.get_backend().lower():
        plt.show()


if __name__ == "__main__":
    main()
