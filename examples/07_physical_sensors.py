#!/usr/bin/env python3
# ============================================================================
# examples/07_physical_sensors.py
# ============================================================================
"""
Example: physical sensors on a simple one-fermenter biogas plant.

This example shows how to:
- build a single-digester plant with maize silage and swine manure feed
- attach physical sensors to plant-side process signals
- collect measured values for later analysis
- plot measured values against the real process values at the end

The example uses all five physical sensor types:
- level sensor on an external gas storage tank
- flow sensor on the digester effluent flow
- pressure sensor on the external gas storage tank
- temperature sensor on the digester temperature
- pH sensor on the digester liquid

Usage:
    py examples/07_physical_sensors.py
"""

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    """Run the physical sensor example."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("matplotlib is required for this example: pip install matplotlib") from exc

    from pyadm1.components.energy.gas_storage import GasStorage
    from pyadm1.components.sensors import PhysicalSensor
    from examples._sensor_demo_helpers import (
        apply_substrate_feed,
        apply_digester_temperature,
        build_single_digester_plant,
        digester_temperature_profile,
        extract_physical_signals,
        substrate_feed_profile,
    )

    duration_days = 10.0
    dt = 1.0 / 24.0
    n_steps = int(duration_days / dt)

    plant, digester, _, _ = build_single_digester_plant()
    gas_tank = GasStorage(
        component_id="sensor_demo_gas_tank",
        storage_type="membrane",
        capacity_m3=2000.0,
        p_min_bar=0.98,
        p_max_bar=1.08,
        initial_fill_fraction=0.25,
        name="Sensor Demo Gas Tank",
    )

    level_sensor = PhysicalSensor(
        component_id="gas_storage_level_sensor",
        sensor_type="level",
        signal_key="current_level",
        measurement_range=(0.0, gas_tank.capacity_m3),
        measurement_noise=1.2,
        accuracy=0.4,
        drift_rate=0.02,
        response_time=2.0 / 24.0,
        sample_interval=1.0 / 24.0,
        unit="m3",
        name="Gas Storage Level Sensor",
    )
    level_sensor.add_input(gas_tank.component_id)

    flow_sensor = PhysicalSensor(
        component_id="digester_flow_sensor",
        sensor_type="flow",
        signal_key="Q_actual",
        measurement_noise=0.10,
        accuracy=0.05,
        drift_rate=0.01,
        response_time=1.0 / 24.0,
        sample_interval=1.0 / 24.0,
        unit="m3/d",
        name="Digester Effluent Flow Sensor",
    )
    flow_sensor.add_input(digester.component_id)

    pressure_sensor = PhysicalSensor(
        component_id="gas_pressure_sensor",
        sensor_type="pressure",
        signal_key="pressure_bar",
        measurement_noise=0.001,
        accuracy=0.001,
        drift_rate=0.0001,
        response_time=1.0 / 24.0,
        sample_interval=1.0 / 24.0,
        unit="bar",
        name="Gas Pressure Sensor",
    )
    pressure_sensor.add_input(gas_tank.component_id)

    temperature_sensor = PhysicalSensor(
        component_id="digester_temperature_sensor",
        sensor_type="temperature",
        signal_key="temperature",
        measurement_noise=0.08,
        accuracy=0.03,
        drift_rate=0.01,
        response_time=2.0 / 24.0,
        sample_interval=1.0 / 24.0,
        unit="K",
        name="Digester Temperature Sensor",
    )
    temperature_sensor.add_input(digester.component_id)

    ph_sensor = PhysicalSensor(
        component_id="digester_ph_sensor",
        sensor_type="pH",
        signal_key="pH",
        measurement_noise=0.02,
        accuracy=0.01,
        drift_rate=0.005,
        response_time=2.0 / 24.0,
        sample_interval=1.0 / 24.0,
        unit="pH",
        name="Digester pH Sensor",
    )
    ph_sensor.add_input(digester.component_id)

    history = {
        "time_days": [],
        "level_true": [],
        "level_measured": [],
        "flow_true": [],
        "flow_measured": [],
        "pressure_true": [],
        "pressure_measured": [],
        "temperature_true": [],
        "temperature_measured": [],
        "ph_true": [],
        "ph_measured": [],
    }

    print("=" * 76)
    print("PyADM1ODE Physical Sensor Example")
    print("=" * 76)
    print("Single fermenter with maize silage + swine manure feed over 10 days")
    print("External gas tank added for level and pressure sensing; digester temperature varies over time")
    print("BHKW size increased to 500 kW_el; tank inflow now uses gas remaining after CHP consumption")
    print()
    header = (
        f"{'Time [d]':>8} | "
        f"{'Level r/m [m3]':>17} | "
        f"{'Flow r/m [m3/d]':>17} | "
        f"{'Press r/m [bar]':>17} | "
        f"{'Temp r/m [K]':>17} | "
        f"{'pH r/m':>10}"
    )
    print(header)
    print("-" * (len(header) + 2))

    for step_idx in range(n_steps):
        apply_substrate_feed(digester, substrate_feed_profile(plant.simulation_time))
        apply_digester_temperature(digester, digester_temperature_profile(plant.simulation_time))
        results = plant.step(dt)
        t = plant.simulation_time

        digester_out = results[digester.component_id]
        chp_out = results["chp_1"]
        net_gas_after_chp = max(
            0.0,
            float(digester_out.get("Q_gas", 0.0)) - float(chp_out.get("Q_gas_consumed", 0.0)),
        )
        gas_demand = max(
            0.0,
            0.68 * net_gas_after_chp + 25.0 * (1.0 + np.sin(2.0 * np.pi * t / 1.5)),
        )
        storage_out = gas_tank.step(
            t=t,
            dt=dt,
            inputs={
                "Q_gas_in_m3_per_day": net_gas_after_chp,
                "Q_gas_out_m3_per_day": gas_demand,
                "vent_to_flare": True,
            },
        )
        physical_state = extract_physical_signals(digester, digester_out, storage_out)

        level_out = level_sensor.step(t=t, dt=dt, inputs={"current_level": physical_state["current_level"]})
        flow_out = flow_sensor.step(t=t, dt=dt, inputs={"Q_actual": physical_state["Q_actual"]})
        pressure_out = pressure_sensor.step(t=t, dt=dt, inputs={"pressure_bar": physical_state["pressure_bar"]})
        temperature_out = temperature_sensor.step(t=t, dt=dt, inputs={"temperature": physical_state["temperature"]})
        ph_out = ph_sensor.step(t=t, dt=dt, inputs={"pH": physical_state["pH"]})

        history["time_days"].append(t)
        history["level_true"].append(physical_state["current_level"])
        history["level_measured"].append(level_out["measurement"])
        history["flow_true"].append(physical_state["Q_actual"])
        history["flow_measured"].append(flow_out["measurement"])
        history["pressure_true"].append(physical_state["pressure_bar"])
        history["pressure_measured"].append(pressure_out["measurement"])
        history["temperature_true"].append(physical_state["temperature"])
        history["temperature_measured"].append(temperature_out["measurement"])
        history["ph_true"].append(physical_state["pH"])
        history["ph_measured"].append(ph_out["measurement"])

        if step_idx % 6 == 0:
            print(
                f"{t:8.3f} | "
                f"{physical_state['current_level']:8.2f}/{level_out['measurement']:8.2f} | "
                f"{physical_state['Q_actual']:8.2f}/{flow_out['measurement']:8.2f} | "
                f"{physical_state['pressure_bar']:8.4f}/{pressure_out['measurement']:8.4f} | "
                f"{physical_state['temperature']:8.2f}/{temperature_out['measurement']:8.2f} | "
                f"{physical_state['pH']:5.2f}/{ph_out['measurement']:5.2f}"
            )

    fig, axes = plt.subplots(5, 1, figsize=(11, 15), sharex=True)

    axes[0].plot(history["time_days"], history["level_true"], label="Real level", linewidth=2.0)
    axes[0].plot(history["time_days"], history["level_measured"], label="Measured level", linewidth=1.5, alpha=0.85)
    axes[0].set_ylabel(f"Level [{level_sensor.unit}]")
    axes[0].set_title("Gas Storage Level Sensor")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(history["time_days"], history["flow_true"], label="Real flow", linewidth=2.0)
    axes[1].plot(history["time_days"], history["flow_measured"], label="Measured flow", linewidth=1.5, alpha=0.85)
    axes[1].set_ylabel(f"Flow [{flow_sensor.unit}]")
    axes[1].set_title("Digester Effluent Flow Sensor")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(history["time_days"], history["pressure_true"], label="Real pressure", linewidth=2.0)
    axes[2].plot(history["time_days"], history["pressure_measured"], label="Measured pressure", linewidth=1.5, alpha=0.85)
    axes[2].set_ylabel(f"Pressure [{pressure_sensor.unit}]")
    axes[2].set_title("Gas Pressure Sensor")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    axes[3].plot(history["time_days"], history["temperature_true"], label="Real temperature", linewidth=2.0)
    axes[3].plot(
        history["time_days"], history["temperature_measured"], label="Measured temperature", linewidth=1.5, alpha=0.85
    )
    axes[3].set_ylabel(f"Temperature [{temperature_sensor.unit}]")
    axes[3].set_title("Digester Temperature Sensor")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    axes[4].plot(history["time_days"], history["ph_true"], label="Real pH", linewidth=2.0)
    axes[4].plot(history["time_days"], history["ph_measured"], label="Measured pH", linewidth=1.5, alpha=0.85)
    axes[4].set_xlabel("Time [days]")
    axes[4].set_ylabel(f"pH [{ph_sensor.unit}]")
    axes[4].set_title("Digester pH Sensor")
    axes[4].grid(True, alpha=0.3)
    axes[4].legend()

    fig.suptitle("Physical Sensors on a One-Fermenter Biogas Plant")
    fig.tight_layout()

    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / "physical_sensor_example.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {plot_path}")

    if "agg" not in plt.get_backend().lower():
        plt.show()


if __name__ == "__main__":
    main()
