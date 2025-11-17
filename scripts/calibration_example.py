# ============================================================================
# examples/calibration_workflow_complete.py
# ============================================================================
"""
Complete Calibration Workflow Example

This example demonstrates the full calibration workflow including:
1. Plant setup
2. Measurement data preparation
3. Parameter calibration
4. Sensitivity analysis
5. Identifiability assessment
6. Validation
7. Application of calibrated parameters

Author: PyADM1 Team
Date: 2024
"""

import numpy as np
import pandas as pd
from pathlib import Path

# PyADM1 imports
from pyadm1.configurator import BiogasPlant
from pyadm1.substrates import Feedstock
from pyadm1.components.biological import Digester
from pyadm1.components.energy import CHP
from pyadm1.calibration import InitialCalibrator
from pyadm1.io import MeasurementData
from pyadm1.core.adm1 import get_state_zero_from_initial_state


def create_example_plant(feedstock: Feedstock) -> BiogasPlant:
    """
    Create an example biogas plant for calibration.

    Args:
        feedstock: Feedstock object for substrate management.

    Returns:
        BiogasPlant: Configured plant ready for calibration.
    """
    print("Creating example biogas plant...")

    plant = BiogasPlant("Calibration Example Plant")

    # Add main digester
    digester = Digester(
        component_id="main_digester",
        feedstock=feedstock,
        V_liq=2000.0,  # 2000 m³
        V_gas=300.0,  # 300 m³
        T_ad=308.15,  # 35°C
        name="Main Fermenter",
    )

    # Load initial state
    data_path = Path("data/initial_states")
    initial_state_file = data_path / "digester_initial8.csv"

    if initial_state_file.exists():
        adm1_state = get_state_zero_from_initial_state(str(initial_state_file))
        Q_substrates = [15.0, 10.0, 0, 0, 0, 0, 0, 0, 0, 0]
        digester.initialize({"adm1_state": adm1_state, "Q_substrates": Q_substrates})
    else:
        print("Warning: Initial state file not found, using defaults")
        digester.initialize()

    plant.add_component(digester)

    # Add CHP unit
    chp = CHP(component_id="chp_main", P_el_nom=500.0, eta_el=0.40, eta_th=0.45, name="CHP Unit")  # 500 kW
    plant.add_component(chp)

    # Connect components
    from pyadm1.configurator.connection_manager import Connection

    plant.add_connection(Connection("main_digester", "chp_main", "gas"))

    # Initialize plant
    plant.initialize()

    print(f"Plant created with {len(plant.components)} components")
    return plant


def create_synthetic_measurements(duration_days: int = 30) -> pd.DataFrame:
    """
    Create synthetic measurement data for testing calibration.

    In practice, this would be replaced with actual plant measurements.

    Args:
        duration_days: Duration of measurement period in days.

    Returns:
        pd.DataFrame: Synthetic measurement data.
    """
    print(f"Creating synthetic measurements for {duration_days} days...")

    # Create hourly timestamps
    n_hours = duration_days * 24
    timestamps = pd.date_range(start="2024-01-01", periods=n_hours, freq="H")

    # Generate synthetic data with realistic noise
    np.random.seed(42)

    data = pd.DataFrame(
        {
            "timestamp": timestamps,
            # Substrate feeds (m³/d)
            "Q_sub1": 15.0 + np.random.normal(0, 0.5, n_hours),  # Corn silage
            "Q_sub2": 10.0 + np.random.normal(0, 0.3, n_hours),  # Cattle manure
            # Measured outputs
            "Q_ch4": 750 + np.random.normal(0, 20, n_hours) + 30 * np.sin(np.arange(n_hours) * 2 * np.pi / 24),
            "Q_gas": 1250 + np.random.normal(0, 30, n_hours) + 40 * np.sin(np.arange(n_hours) * 2 * np.pi / 24),
            "pH": 7.2 + np.random.normal(0, 0.05, n_hours),
            "VFA": 2.5 + np.random.normal(0, 0.15, n_hours) + 0.3 * np.sin(np.arange(n_hours) * 2 * np.pi / 168),
            "TAC": 15.0 + np.random.normal(0, 0.5, n_hours),
            "T_digester": 308.15 + np.random.normal(0, 0.5, n_hours),
        }
    )

    # Ensure non-negative values
    for col in ["Q_sub1", "Q_sub2", "Q_ch4", "Q_gas", "VFA", "TAC"]:
        data[col] = data[col].clip(lower=0)

    # Ensure pH in reasonable range
    data["pH"] = data["pH"].clip(6.5, 8.0)

    print(f"Created {len(data)} measurement points")
    return data


def main():
    """Main calibration workflow."""

    print("=" * 70)
    print("PyADM1 Calibration Workflow Example")
    print("=" * 70)

    # ========================================================================
    # 1. Setup
    # ========================================================================
    print("\n" + "=" * 70)
    print("1. SETUP")
    print("=" * 70)

    # Create feedstock
    feedstock = Feedstock(feeding_freq=48)  # Feed every 48 hours

    # Create plant
    plant = create_example_plant(feedstock)

    # Create/load measurement data
    measurements_df = create_synthetic_measurements(duration_days=30)

    # Save to CSV for reference
    measurements_df.to_csv("calibration_measurements.csv", index=False)
    print("Saved synthetic measurements to 'calibration_measurements.csv'")

    # ========================================================================
    # 2. Load and Validate Measurement Data
    # ========================================================================
    print("\n" + "=" * 70)
    print("2. LOAD AND VALIDATE MEASUREMENT DATA")
    print("=" * 70)

    measurements = MeasurementData.from_csv("calibration_measurements.csv", timestamp_column="timestamp", resample="1H")

    # Validate data quality
    validation = measurements.validate(
        expected_ranges={
            "pH": (6.0, 8.5),
            "VFA": (0.0, 10.0),
            "Q_ch4": (0.0, 2000.0),
        }
    )

    print("\nData validation:")
    print(f"  Valid: {validation.is_valid}")
    print(f"  Quality score: {validation.quality_score:.2f}")
    print(f"  Number of samples: {validation.statistics['n_rows']}")
    print(f"  Missing data: {validation.statistics['pct_missing']:.1f}%")

    if not validation.is_valid:
        validation.print_report()

    # Clean data
    print("\nCleaning measurement data...")
    n_outliers = measurements.remove_outliers(method="zscore", threshold=3.0)
    print(f"  Removed {n_outliers} outliers")

    measurements.fill_gaps(method="interpolate", limit=3)
    print("  Filled gaps with interpolation")

    # ========================================================================
    # 3. Parameter Calibration
    # ========================================================================
    print("\n" + "=" * 70)
    print("3. PARAMETER CALIBRATION")
    print("=" * 70)

    # Create calibrator
    calibrator = InitialCalibrator(plant, verbose=True)

    # Define parameters to calibrate
    parameters_to_calibrate = [
        "k_dis",  # Disintegration rate
        "k_hyd_ch",  # Carbohydrate hydrolysis rate
        "Y_su",  # Sugar uptake yield
    ]

    # Define custom bounds (optional)
    custom_bounds = {
        "k_dis": (0.3, 0.8),
        "Y_su": (0.05, 0.15),
    }

    # Run calibration
    print("\nStarting calibration...")
    result = calibrator.calibrate(
        measurements=measurements,
        parameters=parameters_to_calibrate,
        bounds=custom_bounds,
        objectives=["Q_ch4", "pH"],
        weights={"Q_ch4": 0.8, "pH": 0.2},
        method="differential_evolution",
        validation_split=0.2,
        max_iterations=50,  # Reduced for example
        population_size=10,
        sensitivity_analysis=True,
    )

    # ========================================================================
    # 4. Analyze Results
    # ========================================================================
    print("\n" + "=" * 70)
    print("4. CALIBRATION RESULTS")
    print("=" * 70)

    if result.success:
        print("\n✓ Calibration successful!")
        print(f"\nObjective value: {result.objective_value:.6f}")
        print(f"Number of iterations: {result.n_iterations}")
        print(f"Execution time: {result.execution_time:.1f} seconds")

        print("\n" + "-" * 70)
        print("Calibrated Parameters:")
        print("-" * 70)
        for param, value in result.parameters.items():
            initial = result.initial_parameters[param]
            change = ((value - initial) / initial * 100) if initial != 0 else 0
            print(f"  {param:15s}: {initial:8.4f} → {value:8.4f}  ({change:+6.1f}%)")

        if result.validation_metrics:
            print("\n" + "-" * 70)
            print("Validation Metrics:")
            print("-" * 70)
            for metric, value in result.validation_metrics.items():
                print(f"  {metric:20s}: {value:8.4f}")

        if result.sensitivity:
            print("\n" + "-" * 70)
            print("Parameter Sensitivities:")
            print("-" * 70)
            for param, sensitivity in result.sensitivity.items():
                print(f"  {param:15s}: {sensitivity:8.4e}")

        # Save results
        result.to_json("calibration_result.json")
        print("\n✓ Results saved to 'calibration_result.json'")

    else:
        print(f"\n✗ Calibration failed: {result.message}")
        return

    # ========================================================================
    # 5. Sensitivity Analysis
    # ========================================================================
    print("\n" + "=" * 70)
    print("5. DETAILED SENSITIVITY ANALYSIS")
    print("=" * 70)

    sensitivity_results = calibrator.sensitivity_analysis(
        parameters=result.parameters, measurements=measurements, objectives=["Q_ch4", "pH", "VFA"]
    )

    print("\nParameter Sensitivity Indices:")
    print("-" * 70)
    for param, sens_result in sensitivity_results.items():
        print(f"\n{param} (value: {sens_result.base_value:.4f}):")
        print("  Sensitivity indices:")
        for obj, sens in sens_result.sensitivity_indices.items():
            print(f"    {obj:8s}: {sens:10.4e}")
        print(f"  Variance contribution: {sens_result.variance_contribution:.4e}")

    # ========================================================================
    # 6. Identifiability Analysis
    # ========================================================================
    print("\n" + "=" * 70)
    print("6. PARAMETER IDENTIFIABILITY ANALYSIS")
    print("=" * 70)

    identifiability_results = calibrator.identifiability_analysis(
        parameters=result.parameters, measurements=measurements, correlation_threshold=0.8
    )

    print("\nParameter Identifiability:")
    print("-" * 70)
    for param, ident_result in identifiability_results.items():
        status = "✓ Identifiable" if ident_result.is_identifiable else "✗ Not identifiable"
        print(f"\n{param}: {status}")
        print(f"  Reason: {ident_result.reason}")
        print(
            f"  Confidence interval: [{ident_result.confidence_interval[0]:.4f}, "
            f"{ident_result.confidence_interval[1]:.4f}]"
        )
        print(f"  Objective sensitivity: {ident_result.objective_sensitivity:.4e}")

        if ident_result.correlation_with:
            print("  Correlations:")
            for other_param, corr in ident_result.correlation_with.items():
                if abs(corr) > 0.5:
                    print(f"    {other_param}: {corr:6.3f}")

    # ========================================================================
    # 7. Apply Calibrated Parameters
    # ========================================================================
    print("\n" + "=" * 70)
    print("7. APPLY CALIBRATED PARAMETERS")
    print("=" * 70)

    # Apply to plant
    digester = plant.components["main_digester"]
    digester.apply_calibration_parameters(result.parameters)

    print("\n✓ Applied calibrated parameters to plant")
    print(f"  Parameters applied: {list(result.parameters.keys())}")

    # Verify application
    applied_params = digester.get_calibration_parameters()
    print("\n  Verification:")
    for param, value in applied_params.items():
        print(f"    {param}: {value:.4f}")

    # ========================================================================
    # 8. Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("8. SUMMARY")
    print("=" * 70)

    print("\nCalibration workflow completed successfully!")
    print("\nKey achievements:")
    print(f"  • Calibrated {len(result.parameters)} parameters")
    print(f"  • Achieved objective value: {result.objective_value:.6f}")
    print(f"  • Validation R² (Q_ch4): {result.validation_metrics.get('Q_ch4_r2', 0):.3f}")
    print(f"  • All parameters identifiable: {all(r.is_identifiable for r in identifiability_results.values())}")
    print("  • Parameters applied to plant model")

    print("\nOutput files generated:")
    print("  • calibration_measurements.csv - Measurement data")
    print("  • calibration_result.json - Calibration results")

    print("\nNext steps:")
    print("  1. Validate calibrated model with independent data")
    print("  2. Use calibrated plant for process optimization")
    print("  3. Monitor parameter drift over time")
    print("  4. Re-calibrate periodically with new measurements")

    print("\n" + "=" * 70)
    print("Calibration workflow complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
