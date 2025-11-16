# Example Measurement Data CSV Template

## File: `plant_measurements.csv`

This CSV file contains typical biogas plant measurement data for calibration.

### Column Descriptions:

1. **timestamp** - Date and time of measurement (ISO 8601 format)
2. **Q_sub_maize** - Maize silage feed rate [m³/d]
3. **Q_sub_manure** - Swine manure feed rate [m³/d]
4. **Q_sub_grass** - Grass silage feed rate [m³/d]
5. **pH** - pH value in digester [-]
6. **VFA** - Volatile fatty acids concentration [g HAc eq/L]
7. **TAC** - Total alkalinity [g CaCO3 eq/L]
8. **FOS_TAC** - Ratio of VFA to TAC [-]
9. **T_digester** - Digester temperature [K]
10. **Q_gas** - Total biogas production [m³/d]
11. **Q_ch4** - Methane production [m³/d]
12. **Q_co2** - CO2 production [m³/d]
13. **CH4_content** - Methane content in biogas [%]
14. **P_gas** - Gas pressure [bar]
15. **P_el** - Electrical power output from CHP [kW]
16. **P_th** - Thermal power output from CHP [kW]

### Example Data (first 10 rows):

```csv
timestamp,Q_sub_maize,Q_sub_manure,Q_sub_grass,pH,VFA,TAC,FOS_TAC,T_digester,Q_gas,Q_ch4,Q_co2,CH4_content,P_gas,P_el,P_th
2024-01-01 00:00:00,15.0,10.0,0.0,7.25,2.10,12.5,0.168,308.15,850.5,510.3,340.2,60.0,1.015,420.5,472.6
2024-01-01 01:00:00,15.0,10.0,0.0,7.24,2.15,12.4,0.173,308.20,852.1,511.3,340.8,60.0,1.016,421.2,473.4
2024-01-01 02:00:00,15.0,10.0,0.0,7.23,2.12,12.5,0.170,308.18,851.3,510.8,340.5,60.0,1.015,420.8,473.0
2024-01-01 03:00:00,15.0,10.0,0.0,7.25,2.08,12.6,0.165,308.15,850.0,510.0,340.0,60.0,1.014,420.2,472.2
2024-01-01 04:00:00,15.0,10.0,0.0,7.26,2.05,12.7,0.161,308.10,849.2,509.5,339.7,60.0,1.014,419.8,471.8
2024-01-01 05:00:00,15.0,10.0,0.0,7.27,2.02,12.8,0.158,308.12,848.5,509.1,339.4,60.0,1.013,419.5,471.4
2024-01-01 06:00:00,15.5,10.0,0.0,7.26,2.06,12.7,0.162,308.15,855.0,513.0,342.0,60.0,1.015,422.0,474.3
2024-01-01 07:00:00,15.5,10.0,0.0,7.25,2.10,12.6,0.167,308.18,856.2,513.7,342.5,60.0,1.016,422.6,474.9
2024-01-01 08:00:00,15.5,10.0,0.0,7.24,2.14,12.5,0.171,308.20,857.0,514.2,342.8,60.0,1.016,423.0,475.4
2024-01-01 09:00:00,15.5,10.0,0.0,7.23,2.18,12.4,0.176,308.22,857.8,514.7,343.1,60.0,1.017,423.4,475.8
```

## Complete Template CSV File

### For 30 days of hourly measurements:

```csv
timestamp,Q_sub_maize,Q_sub_manure,Q_sub_grass,pH,VFA,TAC,FOS_TAC,T_digester,Q_gas,Q_ch4,Q_co2,CH4_content,P_gas,P_el,P_th
# ... (720 rows total for 30 days x 24 hours)
```

### Typical Value Ranges for Agricultural Biogas Plants:

- **Q_sub_maize**: 10-25 m³/d (corn silage)
- **Q_sub_manure**: 5-15 m³/d (swine manure)
- **Q_sub_grass**: 0-10 m³/d (grass silage)
- **pH**: 6.8-7.8 (optimal 7.0-7.5)
- **VFA**: 0.5-4.0 g/L (stable operation < 3.0)
- **TAC**: 8-20 g CaCO3/L
- **FOS_TAC**: 0.1-0.4 (warning > 0.3, critical > 0.4)
- **T_digester**: 306-310 K (33-37°C for mesophilic)
- **Q_gas**: 500-1500 m³/d (depends on plant size)
- **Q_ch4**: 300-900 m³/d (50-60% of biogas)
- **CH4_content**: 50-65% (optimal 55-60%)
- **P_gas**: 1.01-1.03 bar (slight overpressure)
- **P_el**: 300-600 kW (for 500 kW CHP)
- **P_th**: 340-680 kW (thermal efficiency ~45%)

## Data Quality Considerations:

1. **Sampling Frequency**:
   - Online sensors: 1-15 minute intervals
   - Lab measurements: Daily to weekly
   - Recommended for calibration: Hourly aggregated data

2. **Missing Data**:
   - Sensor failures and maintenance windows are common
   - Fill gaps < 2 hours with interpolation
   - Mark longer gaps for special handling

3. **Outliers**:
   - Sensor drift and calibration issues
   - Process upsets (overfeeding, pump failures)
   - Use robust outlier detection before calibration

4. **Measurement Uncertainty**:
   - pH: ±0.1 units
   - VFA: ±10% of reading
   - Gas flow: ±2-5% of reading
   - Temperature: ±0.5 K

## Loading Example in Python:

```python
from pyadm1.io.measurement_data import MeasurementData

# Load and preprocess data
data = MeasurementData.from_csv(
    "plant_measurements.csv",
    timestamp_column="timestamp",
    resample="1H"  # Hourly resampling
)

# Validate data quality
validation = data.validate()
validation.print_report()

# Remove outliers
n_outliers = data.remove_outliers(method="zscore", threshold=3.0)
print(f"Removed {n_outliers} outliers")

# Fill gaps
data.fill_gaps(method="interpolate", limit=2)

# Get specific measurements
pH = data.get_measurement("pH")
Q_gas = data.get_measurement("Q_gas")

# Get substrate feeds for simulation
Q = data.get_substrate_feeds(
    substrate_columns=["Q_sub_maize", "Q_sub_manure", "Q_sub_grass"]
)
```

## Notes for Calibration:

1. **Initial Calibration** requires at least 7-30 days of stable operation data
2. **Online Calibration** uses rolling windows of 3-7 days
3. **Steady-state periods** are most valuable for calibration
4. **Dynamic periods** (feed changes, disturbances) help identify kinetic parameters
5. **Multiple operating conditions** improve parameter identifiability
