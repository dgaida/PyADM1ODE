#!/usr/bin/env python3
# ============================================================================
# examples/09_gas_sensors.py
# ============================================================================
"""
Example: gas sensors on a simple one-fermenter biogas plant.

This example shows how to:
- build a single-digester plant with maize silage and swine manure feed
- attach gas sensors to the digester gas stream
- collect measured values for later analysis
- plot measured values against the real gas values at the end

The example uses five gas analyzers:
- methane analyzer
- carbon dioxide analyzer
- hydrogen sulfide analyzer
- oxygen analyzer
- trace gas analyzer

Usage:
    py examples/09_gas_sensors.py
"""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    """Run the gas sensor example."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for this example: pip install matplotlib") from exc

    from pyadm1.components.sensors import GasSensor
    from examples._sensor_demo_helpers import (
        apply_substrate_feed,
        build_single_digester_plant,
        extract_gas_signals,
        substrate_feed_profile,
    )

    duration_days = 10.0
    dt = 1.0 / 24.0
    n_steps = int(duration_days / dt)

    plant, digester, _, q_substrates = build_single_digester_plant()

    ch4_sensor = GasSensor(
        component_id="ch4_sensor",
        sensor_type="CH4",
        analyzer_method="infrared",
        signal_key="CH4",
        measurement_range=(0.0, 100.0),
        measurement_noise=0.12,
        accuracy=0.08,
        drift_rate=0.01,
        response_time=1.0 / 24.0,
        sample_interval=1.0 / 24.0,
        detection_limit=0.1,
        unit="%",
        name="Methane Analyzer",
    )
    ch4_sensor.add_input(digester.component_id)

    co2_sensor = GasSensor(
        component_id="co2_sensor",
        sensor_type="CO2",
        analyzer_method="infrared",
        signal_key="CO2",
        measurement_range=(0.0, 100.0),
        measurement_noise=0.12,
        accuracy=0.08,
        drift_rate=0.008,
        response_time=1.0 / 24.0,
        sample_interval=1.0 / 24.0,
        detection_limit=0.1,
        unit="%",
        name="CO2 Analyzer",
    )
    co2_sensor.add_input(digester.component_id)

    h2s_sensor = GasSensor(
        component_id="h2s_sensor",
        sensor_type="H2S",
        analyzer_method="electrochemical",
        signal_key="H2S",
        measurement_range=(0.0, 5000.0),
        measurement_noise=10.0,
        accuracy=4.0,
        drift_rate=3.0,
        response_time=2.0 / 24.0,
        sample_interval=2.0 / 24.0,
        detection_limit=1.0,
        unit="ppm",
        name="H2S Analyzer",
    )
    h2s_sensor.add_input(digester.component_id)

    o2_sensor = GasSensor(
        component_id="o2_sensor",
        sensor_type="O2",
        analyzer_method="paramagnetic",
        signal_key="O2",
        measurement_range=(0.0, 25.0),
        measurement_noise=0.01,
        accuracy=0.008,
        drift_rate=0.002,
        response_time=1.0 / 24.0,
        sample_interval=1.0 / 24.0,
        detection_limit=0.01,
        unit="%",
        name="O2 Analyzer",
    )
    o2_sensor.add_input(digester.component_id)

    trace_sensor = GasSensor(
        component_id="trace_sensor",
        sensor_type="trace_gas",
        analyzer_method="photoionization",
        signal_key="trace_gas",
        measurement_range=(0.0, 2000.0),
        measurement_noise=3.0,
        accuracy=1.5,
        drift_rate=1.0,
        response_time=2.0 / 24.0,
        sample_interval=2.0 / 24.0,
        detection_limit=0.5,
        unit="ppm",
        name="Trace Gas Analyzer",
    )
    trace_sensor.add_input(digester.component_id)

    history = {
        "time_days": [],
        "ch4_true": [],
        "ch4_measured": [],
        "co2_true": [],
        "co2_measured": [],
        "h2s_true": [],
        "h2s_measured": [],
        "o2_true": [],
        "o2_measured": [],
        "trace_true": [],
        "trace_measured": [],
    }

    print("=" * 76)
    print("PyADM1ODE Gas Sensor Example")
    print("=" * 76)
    print("Single fermenter with maize silage + swine manure feed over 10 days")
    print()
    header = (
        f"{'Time [d]':>8} | "
        f"{'CH4 r/m [%]':>15} | "
        f"{'CO2 r/m [%]':>15} | "
        f"{'H2S r/m [ppm]':>17} | "
        f"{'O2 r/m [%]':>14} | "
        f"{'Trace r/m [ppm]':>19}"
    )
    print(header)
    print("-" * (len(header) + 2))

    for step_idx in range(n_steps):
        current_q_substrates = substrate_feed_profile(plant.simulation_time)
        apply_substrate_feed(digester, current_q_substrates)
        results = plant.step(dt)
        t = plant.simulation_time
        digester_out = results[digester.component_id]
        gas_state = extract_gas_signals(digester_out, current_q_substrates)

        ch4_out = ch4_sensor.step(t=t, dt=dt, inputs={"CH4": gas_state["CH4"]})
        co2_out = co2_sensor.step(t=t, dt=dt, inputs={"CO2": gas_state["CO2"]})
        h2s_out = h2s_sensor.step(t=t, dt=dt, inputs={"H2S": gas_state["H2S"]})
        o2_out = o2_sensor.step(t=t, dt=dt, inputs={"O2": gas_state["O2"]})
        trace_out = trace_sensor.step(t=t, dt=dt, inputs={"trace_gas": gas_state["trace_gas"]})

        history["time_days"].append(t)
        history["ch4_true"].append(gas_state["CH4"])
        history["ch4_measured"].append(ch4_out["measurement"])
        history["co2_true"].append(gas_state["CO2"])
        history["co2_measured"].append(co2_out["measurement"])
        history["h2s_true"].append(gas_state["H2S"])
        history["h2s_measured"].append(h2s_out["measurement"])
        history["o2_true"].append(gas_state["O2"])
        history["o2_measured"].append(o2_out["measurement"])
        history["trace_true"].append(gas_state["trace_gas"])
        history["trace_measured"].append(trace_out["measurement"])

        if step_idx % 6 == 0:
            print(
                f"{t:8.3f} | "
                f"{gas_state['CH4']:7.2f}/{ch4_out['measurement']:7.2f} | "
                f"{gas_state['CO2']:7.2f}/{co2_out['measurement']:7.2f} | "
                f"{gas_state['H2S']:8.0f}/{h2s_out['measurement']:8.0f} | "
                f"{gas_state['O2']:6.3f}/{o2_out['measurement']:6.3f} | "
                f"{gas_state['trace_gas']:9.1f}/{trace_out['measurement']:9.1f}"
            )

    fig, axes = plt.subplots(5, 1, figsize=(11, 15), sharex=True)

    axes[0].plot(history["time_days"], history["ch4_true"], label="Real CH4", linewidth=2.0)
    axes[0].plot(history["time_days"], history["ch4_measured"], label="Measured CH4", linewidth=1.5, alpha=0.85)
    axes[0].set_ylabel(f"CH4 [{ch4_sensor.unit}]")
    axes[0].set_title("Methane Analyzer")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(history["time_days"], history["co2_true"], label="Real CO2", linewidth=2.0)
    axes[1].plot(history["time_days"], history["co2_measured"], label="Measured CO2", linewidth=1.5, alpha=0.85)
    axes[1].set_ylabel(f"CO2 [{co2_sensor.unit}]")
    axes[1].set_title("CO2 Analyzer")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(history["time_days"], history["h2s_true"], label="Real H2S", linewidth=2.0)
    axes[2].plot(history["time_days"], history["h2s_measured"], label="Measured H2S", linewidth=1.5, alpha=0.85)
    axes[2].set_ylabel(f"H2S [{h2s_sensor.unit}]")
    axes[2].set_title("H2S Analyzer")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    axes[3].plot(history["time_days"], history["o2_true"], label="Real O2", linewidth=2.0)
    axes[3].plot(history["time_days"], history["o2_measured"], label="Measured O2", linewidth=1.5, alpha=0.85)
    axes[3].set_ylabel(f"O2 [{o2_sensor.unit}]")
    axes[3].set_title("O2 Analyzer")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    axes[4].plot(history["time_days"], history["trace_true"], label="Real trace gas", linewidth=2.0)
    axes[4].plot(history["time_days"], history["trace_measured"], label="Measured trace gas", linewidth=1.5, alpha=0.85)
    axes[4].set_xlabel("Time [days]")
    axes[4].set_ylabel(f"Trace [{trace_sensor.unit}]")
    axes[4].set_title("Trace Gas Analyzer")
    axes[4].grid(True, alpha=0.3)
    axes[4].legend()

    fig.suptitle("Gas Sensors on a One-Fermenter Biogas Plant")
    fig.tight_layout()

    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / "gas_sensor_example.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {plot_path}")

    if "agg" not in plt.get_backend().lower():
        plt.show()


if __name__ == "__main__":
    main()
