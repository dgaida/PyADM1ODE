#!/usr/bin/env python3
"""
Example: Using Parameter Bounds and Measurement Data for Calibration

This script demonstrates how to use the parameter bounds and measurement data
modules for ADM1 model calibration.

Demonstrates complete workflow:

1. **Parameter Bounds Setup**
   - Load default bounds
   - Validate parameters
   - Calculate penalties

2. **Measurement Data Loading**
   - Load CSV data
   - Validate quality
   - Clean outliers
   - Fill gaps

3. **Calibration Workflow**
   - Prepare bounds
   - Load and clean data
   - Split calibration/validation
   - Setup optimization
   - Post-calibration validation

Usage:
    python scripts/calibration_example.py
"""

import sys
from pathlib import Path

# Add pyadm1 to path if needed
# sys.path.insert(0, str(Path(__file__).parent.parent))

from pyadm1.calibration.parameter_bounds import (
    create_default_bounds,
)
from pyadm1.io.measurement_data import (
    MeasurementData,
)


def example_parameter_bounds():
    """Demonstrate parameter bounds usage."""
    print("=" * 70)
    print("PARAMETER BOUNDS EXAMPLE")
    print("=" * 70)

    # Create default bounds
    bounds = create_default_bounds()

    print("\n1. Get bounds for specific parameters:")
    print("-" * 50)

    for param_name in ["k_dis", "Y_su", "k_m_ac"]:
        bound = bounds.get_bounds(param_name)
        if bound:
            print(f"\n{param_name}:")
            print(f"  Range: [{bound.lower:.4f}, {bound.upper:.4f}]")
            print(f"  Default: {bound.default:.4f}")
            print(f"  Type: {bound.bound_type.value}")
            print(f"  Unit: {bound.unit}")
            print(f"  Description: {bound.description}")

    # Check if values are within bounds
    print("\n2. Validate parameter values:")
    print("-" * 50)

    test_values = {
        "k_dis": 0.5,
        "Y_su": 0.10,
        "k_m_ac": 8.0,
    }

    for param, value in test_values.items():
        is_valid = bounds.is_within_bounds(param, value)
        print(f"{param} = {value:.4f}: {'✓ Valid' if is_valid else '✗ Invalid'}")

    # Test out-of-bounds values
    print("\n3. Test out-of-bounds values:")
    print("-" * 50)

    out_of_bounds = {
        "k_dis": 1.2,  # Too high
        "Y_su": 0.02,  # Too low
    }

    for param, value in out_of_bounds.items():
        is_valid = bounds.is_within_bounds(param, value)
        penalty = bounds.calculate_penalty(param, value)
        clipped = bounds.clip_to_bounds(param, value)

        print(f"\n{param} = {value:.4f}:")
        print(f"  Valid: {is_valid}")
        print(f"  Penalty: {penalty:.4f}")
        print(f"  Clipped value: {clipped:.4f}")

    # Validate complete parameter set
    print("\n4. Validate complete parameter set:")
    print("-" * 50)

    parameters = {
        "k_dis": 0.5,
        "k_hyd_ch": 10.0,
        "Y_su": 0.10,
        "Y_aa": 0.08,
        "k_m_ac": 8.0,
    }

    is_valid, errors = bounds.validate_parameters(parameters)
    print(f"Valid: {is_valid}")
    if errors:
        print("Errors:")
        for error in errors:
            print(f"  - {error}")

    # Calculate total penalty
    total_penalty = bounds.calculate_total_penalty(parameters)
    print(f"Total penalty: {total_penalty:.4f}")

    # Scale/unscale parameters
    print("\n5. Parameter scaling to unit interval:")
    print("-" * 50)

    param = "k_dis"
    value = 0.6
    scaled = bounds.scale_to_unit_interval(param, value)
    unscaled = bounds.unscale_from_unit_interval(param, scaled)

    print(f"{param} = {value:.4f}")
    print(f"  Scaled to [0,1]: {scaled:.4f}")
    print(f"  Unscaled back: {unscaled:.4f}")


def example_measurement_data():
    """Demonstrate measurement data usage."""
    print("\n" + "=" * 70)
    print("MEASUREMENT DATA EXAMPLE")
    print("=" * 70)

    # First, generate synthetic data if file doesn't exist
    csv_file = "plant_measurements.csv"

    if not Path(csv_file).exists():
        print("\nGenerating synthetic measurement data...")
        from generate_measurement_data import generate_measurement_data

        df = generate_measurement_data(n_days=7)
        df.to_csv(csv_file, index=False)
        print(f"  Saved to: {csv_file}")

    # Load measurement data
    print("\n1. Load measurement data:")
    print("-" * 50)

    data = MeasurementData.from_csv(csv_file, timestamp_column="timestamp", resample="1H")

    print(data)

    # Validate data
    print("\n2. Validate data quality:")
    print("-" * 50)

    validation = data.validate()
    validation.print_report()

    # Get summary statistics
    print("\n3. Summary statistics:")
    print("-" * 50)
    print(data.summary()[["pH", "VFA", "Q_gas", "Q_ch4"]])

    # Detect and remove outliers
    print("\n4. Outlier detection:")
    print("-" * 50)

    # Create a copy for outlier removal
    data_clean = MeasurementData(data.data.copy())

    n_outliers = data_clean.remove_outliers(columns=["pH", "VFA", "Q_gas", "Q_ch4"], method="zscore", threshold=3.0)

    print(f"Removed {n_outliers} outliers")

    # Fill missing data
    print("\n5. Fill missing data:")
    print("-" * 50)

    missing_before = data_clean.data.isna().sum().sum()
    print(f"Missing values before filling: {missing_before}")

    data_clean.fill_gaps(columns=["pH", "VFA", "TAC", "Q_gas", "Q_ch4"], method="interpolate", limit=3)

    missing_after = data_clean.data.isna().sum().sum()
    print(f"Missing values after filling: {missing_after}")
    print(f"Filled: {missing_before - missing_after} values")

    # Get specific measurements
    print("\n6. Extract measurements:")
    print("-" * 50)

    pH = data_clean.get_measurement("pH")
    print(f"pH measurements: {len(pH)} values")
    print(f"  Mean: {pH.mean():.2f}")
    print(f"  Std: {pH.std():.2f}")
    print(f"  Range: [{pH.min():.2f}, {pH.max():.2f}]")

    VFA = data_clean.get_measurement("VFA")
    print(f"\nVFA measurements: {len(VFA)} values")
    print(f"  Mean: {VFA.mean():.2f} g/L")
    print(f"  Std: {VFA.std():.2f} g/L")

    # Get substrate feeds
    print("\n7. Extract substrate feeds:")
    print("-" * 50)

    Q = data_clean.get_substrate_feeds(substrate_columns=["Q_sub_maize", "Q_sub_manure", "Q_sub_grass"])

    print(f"Substrate feed array shape: {Q.shape}")
    print("Mean feed rates (m³/d):")
    print(f"  Maize: {Q[:, 0].mean():.1f}")
    print(f"  Manure: {Q[:, 1].mean():.1f}")
    print(f"  Grass: {Q[:, 2].mean():.1f}")

    # Time window selection
    print("\n8. Select time window:")
    print("-" * 50)

    start_time = data_clean.data.index[24]  # After 24 hours
    end_time = data_clean.data.index[96]  # After 96 hours

    window_data = data_clean.get_time_window(start_time, end_time)
    print(f"Selected window: {len(window_data)} samples")
    print(f"  From: {window_data.data.index[0]}")
    print(f"  To: {window_data.data.index[-1]}")


def example_calibration_workflow():
    """Demonstrate complete calibration workflow."""
    print("\n" + "=" * 70)
    print("COMPLETE CALIBRATION WORKFLOW")
    print("=" * 70)

    # 1. Setup parameter bounds
    print("\n1. Setup parameter bounds:")
    print("-" * 50)

    bounds = create_default_bounds()

    # Parameters to calibrate
    calib_params = ["k_dis", "k_hyd_ch", "Y_su", "k_m_ac"]

    print("Parameters for calibration:")
    for param in calib_params:
        bound = bounds.get_bounds(param)
        if bound:
            print(f"  {param}: [{bound.lower:.4f}, {bound.upper:.4f}]")

    # Get default initial values
    initial_values = bounds.get_default_values(calib_params)
    print("\nInitial parameter values:")
    for param, value in initial_values.items():
        print(f"  {param}: {value:.4f}")

    # 2. Load and preprocess measurement data
    print("\n2. Load and preprocess measurement data:")
    print("-" * 50)

    csv_file = "plant_measurements.csv"

    if not Path(csv_file).exists():
        print("Generating synthetic data...")
        from generate_measurement_data import generate_measurement_data

        df = generate_measurement_data(n_days=7)
        df.to_csv(csv_file, index=False)

    data = MeasurementData.from_csv(csv_file, resample="1H")

    # Validate
    validation = data.validate()
    print(f"Data quality score: {validation.quality_score:.2f}")

    # Clean data
    n_outliers = data.remove_outliers(method="zscore", threshold=3.0)
    data.fill_gaps(method="interpolate", limit=3)
    print(f"Removed {n_outliers} outliers and filled gaps")

    # 3. Prepare calibration data
    print("\n3. Prepare calibration data:")
    print("-" * 50)

    # Split into calibration and validation sets
    n_total = len(data)
    n_calib = int(n_total * 0.8)

    calib_data = MeasurementData(data.data.iloc[:n_calib].copy())
    valid_data = MeasurementData(data.data.iloc[n_calib:].copy())

    print(f"Calibration samples: {len(calib_data)}")
    print(f"Validation samples: {len(valid_data)}")

    # Extract objectives
    objectives = {
        "Q_ch4": calib_data.get_measurement("Q_ch4"),
        "pH": calib_data.get_measurement("pH"),
        "VFA": calib_data.get_measurement("VFA"),
    }

    print("\nObjectives for calibration:")
    for obj_name, obj_series in objectives.items():
        print(f"  {obj_name}: {len(obj_series)} values, " f"mean={obj_series.mean():.2f}, std={obj_series.std():.2f}")

    # 4. Optimization setup (pseudo-code)
    print("\n4. Optimization setup:")
    print("-" * 50)

    print("Would perform calibration with:")
    print(f"  Parameters: {calib_params}")
    print(f"  Objectives: {list(objectives.keys())}")
    print("  Method: differential_evolution")
    print("  Population size: 15")
    print("  Max iterations: 100")

    # Example objective function (pseudo-code)
    def objective_function(param_values):
        """
        Objective function for calibration.

        Would:
        1. Create parameter dict from values
        2. Validate parameters against bounds
        3. Run simulation with parameters
        4. Calculate error vs measurements
        5. Add penalty for out-of-bounds values
        """
        params = dict(zip(calib_params, param_values))

        # Validate bounds
        is_valid, _ = bounds.validate_parameters(params)
        if not is_valid:
            penalty = bounds.calculate_total_penalty(params)
            return 1e6 + penalty

        # Simulate (pseudo-code)
        # simulated = simulate_adm1(params, calib_data)

        # Calculate error (pseudo-code)
        # error = calculate_rmse(simulated, objectives)

        # return error

        return 0.0  # Placeholder

    print("\nObjective function defined (pseudo-code)")

    # 5. Post-calibration validation
    print("\n5. Post-calibration validation:")
    print("-" * 50)

    # Example calibrated parameters
    calibrated_params = {
        "k_dis": 0.52,
        "k_hyd_ch": 10.5,
        "Y_su": 0.098,
        "k_m_ac": 7.8,
    }

    print("Calibrated parameters:")
    for param, value in calibrated_params.items():
        bound = bounds.get_bounds(param)
        initial = initial_values[param]
        change_pct = ((value - initial) / initial) * 100

        print(f"  {param}: {initial:.4f} → {value:.4f} ({change_pct:+.1f}%)")

        # Check if within bounds
        is_valid = bounds.is_within_bounds(param, value)
        if not is_valid:
            print("    ⚠ WARNING: Outside bounds!")

    # Validate on validation set
    print("\nValidation on hold-out data:")
    print("  (Would simulate with calibrated params and compare to validation_data)")

    print("\n" + "=" * 70)
    print("CALIBRATION WORKFLOW COMPLETE")
    print("=" * 70)


def main():
    """Run all examples."""
    try:
        # Example 1: Parameter bounds
        example_parameter_bounds()

        # Example 2: Measurement data
        example_measurement_data()

        # Example 3: Complete workflow
        example_calibration_workflow()

        print("\n✓ All examples completed successfully!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
