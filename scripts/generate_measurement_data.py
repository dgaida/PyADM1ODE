#!/usr/bin/env python3
"""
Generate synthetic measurement data for biogas plant calibration.

This script creates a realistic CSV file with biogas plant measurements
including substrate feeds, process variables, and gas production data.

Generates realistic synthetic biogas plant data with:
- Configurable duration and sampling rate
- Realistic process dynamics
- Time-delayed gas production
- Feed rate variations
- Measurement noise
- Outliers (1% of data)
- Missing data gaps (2-5%)

Usage:
    python scripts/generate_measurement_data.py --output data/plant_measurements.csv --days 30

    # Generate clean data (no noise/outliers)
    python scripts/generate_measurement_data.py --no-noise --no-outliers --no-missing
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import argparse


def generate_measurement_data(
    start_date: str = "2024-01-01",
    n_days: int = 30,
    sampling_interval_hours: int = 1,
    add_noise: bool = True,
    add_outliers: bool = True,
    add_missing: bool = True,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic biogas plant measurement data.

    Args:
        start_date: Start date (ISO format)
        n_days: Number of days to simulate
        sampling_interval_hours: Sampling interval in hours
        add_noise: Add measurement noise
        add_outliers: Add occasional outliers
        add_missing: Add missing data gaps
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame with measurement data
    """
    np.random.seed(random_seed)

    # Create timestamp index
    start = pd.to_datetime(start_date)
    n_samples = n_days * 24 // sampling_interval_hours
    timestamps = [start + timedelta(hours=i * sampling_interval_hours) for i in range(n_samples)]

    # Initialize data dictionary
    data = {"timestamp": timestamps}

    # =======================================================================
    # SUBSTRATE FEEDS
    # =======================================================================
    # Maize silage: base 15 m³/d with some variation
    Q_maize_base = 15.0
    Q_maize = Q_maize_base + 2.0 * np.sin(np.linspace(0, 4 * np.pi, n_samples))

    # Add feed changes (step increases/decreases)
    feed_changes = [n_samples // 4, n_samples // 2, 3 * n_samples // 4]
    for change_idx in feed_changes:
        if change_idx < n_samples:
            Q_maize[change_idx:] += np.random.choice([-1.5, 0, 1.5])

    data["Q_sub_maize"] = Q_maize

    # Swine manure: base 10 m³/d with less variation
    Q_manure_base = 10.0
    Q_manure = Q_manure_base + 1.0 * np.sin(np.linspace(0, 3 * np.pi, n_samples))
    data["Q_sub_manure"] = Q_manure

    # Grass silage: occasional feeding
    Q_grass = np.zeros(n_samples)
    grass_periods = [
        (n_samples // 3, n_samples // 3 + n_samples // 10),
        (2 * n_samples // 3, 2 * n_samples // 3 + n_samples // 15),
    ]
    for start_idx, end_idx in grass_periods:
        Q_grass[start_idx:end_idx] = 5.0
    data["Q_sub_grass"] = Q_grass

    # =======================================================================
    # PROCESS VARIABLES
    # =======================================================================
    # pH: stable around 7.2-7.3, decreases slightly with overfeeding
    total_feed = Q_maize + Q_manure + Q_grass
    pH_base = 7.25
    pH = pH_base - 0.02 * (total_feed - 25) / 5  # Decreases with OLR
    pH = np.clip(pH, 6.8, 7.6)
    data["pH"] = pH

    # VFA: inversely related to pH, increases with overfeeding
    VFA_base = 2.0
    VFA = VFA_base + 1.5 * (7.3 - pH)  # Increases when pH drops
    VFA = np.clip(VFA, 0.5, 5.0)
    data["VFA"] = VFA

    # TAC: relatively stable, slight variations
    TAC_base = 12.5
    TAC = TAC_base + 0.5 * np.sin(np.linspace(0, 2 * np.pi, n_samples))
    TAC = np.clip(TAC, 10.0, 15.0)
    data["TAC"] = TAC

    # FOS/TAC ratio
    data["FOS_TAC"] = VFA / TAC

    # Temperature: mesophilic digester (35°C = 308.15 K)
    T_base = 308.15
    T_digester = T_base + 0.5 * np.sin(np.linspace(0, np.pi, n_samples))
    data["T_digester"] = T_digester

    # =======================================================================
    # GAS PRODUCTION
    # =======================================================================
    # Biogas production: related to feed rate with time delay
    # Simple model: Q_gas ≈ 34 * Q_total (m³ biogas per m³ substrate)
    Q_gas_base = 34 * total_feed

    # Add time delay (1-2 day lag for gas production)
    delay_hours = 36
    delay_samples = delay_hours // sampling_interval_hours
    Q_gas = np.roll(Q_gas_base, delay_samples)

    # Add dynamics (exponential smoothing)
    alpha = 0.3  # Smoothing factor
    Q_gas_smooth = np.zeros_like(Q_gas)
    Q_gas_smooth[0] = Q_gas[0]
    for i in range(1, len(Q_gas)):
        Q_gas_smooth[i] = alpha * Q_gas[i] + (1 - alpha) * Q_gas_smooth[i - 1]

    data["Q_gas"] = Q_gas_smooth

    # Methane content: typically 55-60%
    CH4_content_base = 58.0  # percent
    CH4_content = CH4_content_base + 2.0 * np.sin(np.linspace(0, 5 * np.pi, n_samples))
    CH4_content = np.clip(CH4_content, 52.0, 63.0)
    data["CH4_content"] = CH4_content

    # Methane production
    data["Q_ch4"] = Q_gas_smooth * (CH4_content / 100)

    # CO2 production (remainder of biogas)
    data["Q_co2"] = Q_gas_smooth * (1 - CH4_content / 100)

    # Gas pressure: slight overpressure
    P_gas_base = 1.015
    P_gas = P_gas_base + 0.005 * (Q_gas_smooth / 850 - 1)
    data["P_gas"] = P_gas

    # =======================================================================
    # CHP PERFORMANCE
    # =======================================================================
    # Electrical power: ~40% of methane energy (10 kWh/m³ CH4)
    eta_el = 0.40
    E_ch4 = 10.0  # kWh/m³
    P_el = data["Q_ch4"] / 24 * E_ch4 * eta_el  # kW
    data["P_el"] = P_el

    # Thermal power: ~45% of methane energy
    eta_th = 0.45
    P_th = data["Q_ch4"] / 24 * E_ch4 * eta_th  # kW
    data["P_th"] = P_th

    # =======================================================================
    # ADD REALISTIC NOISE, OUTLIERS, AND MISSING DATA
    # =======================================================================
    df = pd.DataFrame(data)

    if add_noise:
        # Add measurement noise
        noise_levels = {
            "Q_sub_maize": 0.3,
            "Q_sub_manure": 0.2,
            "Q_sub_grass": 0.1,
            "pH": 0.05,
            "VFA": 0.15,
            "TAC": 0.3,
            "T_digester": 0.2,
            "Q_gas": 10.0,
            "Q_ch4": 6.0,
            "Q_co2": 4.0,
            "CH4_content": 0.5,
            "P_gas": 0.002,
            "P_el": 5.0,
            "P_th": 5.0,
        }

        for col, noise_std in noise_levels.items():
            if col in df.columns:
                noise = np.random.normal(0, noise_std, len(df))
                df[col] = df[col] + noise

    if add_outliers:
        # Add occasional outliers (1% of data)
        outlier_prob = 0.01
        outlier_columns = ["pH", "VFA", "Q_gas", "Q_ch4"]

        for col in outlier_columns:
            if col in df.columns:
                n_outliers = int(len(df) * outlier_prob)
                outlier_indices = np.random.choice(len(df), n_outliers, replace=False)

                # Outliers are 3-5 sigma away from mean
                col_mean = df[col].mean()
                col_std = df[col].std()
                outlier_values = (
                    col_mean + np.random.choice([-1, 1], n_outliers) * np.random.uniform(3, 5, n_outliers) * col_std
                )

                df.loc[outlier_indices, col] = outlier_values

    if add_missing:
        # Add missing data (2-5% randomly distributed)
        missing_prob = 0.03
        missing_columns = ["VFA", "TAC", "Q_gas", "Q_ch4"]

        for col in missing_columns:
            if col in df.columns:
                n_missing = int(len(df) * missing_prob)
                missing_indices = np.random.choice(len(df), n_missing, replace=False)
                df.loc[missing_indices, col] = np.nan

        # Add some longer gaps (sensor maintenance)
        # 2-3 gaps of 3-6 hours each
        n_gaps = np.random.randint(2, 4)
        gap_columns = ["pH", "VFA", "TAC"]

        for _ in range(n_gaps):
            gap_start = np.random.randint(0, len(df) - 6)
            gap_length = np.random.randint(3, 7)
            gap_col = np.random.choice(gap_columns)

            df.loc[gap_start : gap_start + gap_length, gap_col] = np.nan

    # Round values to realistic precision
    df["Q_sub_maize"] = df["Q_sub_maize"].round(1)
    df["Q_sub_manure"] = df["Q_sub_manure"].round(1)
    df["Q_sub_grass"] = df["Q_sub_grass"].round(1)
    df["pH"] = df["pH"].round(2)
    df["VFA"] = df["VFA"].round(2)
    df["TAC"] = df["TAC"].round(1)
    df["FOS_TAC"] = df["FOS_TAC"].round(3)
    df["T_digester"] = df["T_digester"].round(2)
    df["Q_gas"] = df["Q_gas"].round(1)
    df["Q_ch4"] = df["Q_ch4"].round(1)
    df["Q_co2"] = df["Q_co2"].round(1)
    df["CH4_content"] = df["CH4_content"].round(1)
    df["P_gas"] = df["P_gas"].round(4)
    df["P_el"] = df["P_el"].round(1)
    df["P_th"] = df["P_th"].round(1)

    return df


def main():
    """Main function to generate and save measurement data."""
    parser = argparse.ArgumentParser(description="Generate synthetic biogas plant measurement data")
    parser.add_argument("--output", type=str, default="plant_measurements.csv", help="Output CSV file path")
    parser.add_argument("--days", type=int, default=30, help="Number of days to simulate")
    parser.add_argument("--interval", type=int, default=1, help="Sampling interval in hours")
    parser.add_argument("--no-noise", action="store_true", help="Disable measurement noise")
    parser.add_argument("--no-outliers", action="store_true", help="Disable outliers")
    parser.add_argument("--no-missing", action="store_true", help="Disable missing data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    print("Generating measurement data...")
    print(f"  Duration: {args.days} days")
    print(f"  Sampling interval: {args.interval} hour(s)")
    print(f"  Noise: {not args.no_noise}")
    print(f"  Outliers: {not args.no_outliers}")
    print(f"  Missing data: {not args.no_missing}")

    # Generate data
    df = generate_measurement_data(
        n_days=args.days,
        sampling_interval_hours=args.interval,
        add_noise=not args.no_noise,
        add_outliers=not args.no_outliers,
        add_missing=not args.no_missing,
        random_seed=args.seed,
    )

    # Save to CSV
    df.to_csv(args.output, index=False)

    print(f"\nSaved {len(df)} samples to: {args.output}")
    print("\nData summary:")
    print(df.describe())

    # Print data quality info
    print("\nMissing data:")
    missing = df.isna().sum()
    for col, count in missing.items():
        if count > 0:
            pct = (count / len(df)) * 100
            print(f"  {col}: {count} ({pct:.1f}%)")

    print("\nFirst few rows:")
    print(df.head())


if __name__ == "__main__":
    main()
