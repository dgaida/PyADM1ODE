# Model Calibration Guide

This guide covers parameter calibration in PyADM1ODE, including initial calibration from historical data and online re-calibration during plant operation.

## Overview

Model calibration is essential for accurate biogas plant simulation. PyADM1ODE provides:

- **Initial Calibration**: Batch optimization from historical measurement data
- **Online Re-calibration**: Adaptive parameter adjustment during operation
- **Sensitivity Analysis**: Identification of influential parameters
- **Identifiability Assessment**: Detection of over-parameterization
- **Validation Tools**: Goodness-of-fit metrics and residual analysis

## Quick Start

```python
from pyadm1.calibration import Calibrator
from pyadm1.io import MeasurementData
from pyadm1.configurator import BiogasPlant
from pyadm1.substrates import Feedstock

# Load plant and measurements
feedstock = Feedstock(feeding_freq=48)
plant = BiogasPlant.from_json("plant.json", feedstock)
measurements = MeasurementData.from_csv("plant_data.csv")

# Create calibrator
calibrator = Calibrator(plant)

# Calibrate parameters
result = calibrator.calibrate_initial(
    measurements=measurements,
    parameters=["k_dis", "k_hyd_ch", "Y_su"],
    objectives=["Q_ch4", "pH"],
    weights={"Q_ch4": 0.8, "pH": 0.2}
)

# Apply calibrated parameters
if result.success:
    calibrator.apply_calibration(result)
```

## Initial Calibration

### When to Use Initial Calibration

Use initial calibration when you have:
- Historical measurement data (≥2 weeks recommended)
- Stable plant operation during measurement period
- Reliable measurements of key outputs (gas production, pH, VFA)
- Known substrate feed rates and composition

### Parameter Selection

Choose parameters based on:

**High Priority** (most sensitive to calibration):
- `k_dis`: Disintegration rate [1/d]
- `k_hyd_ch`, `k_hyd_pr`, `k_hyd_li`: Hydrolysis rates [1/d]
- `Y_su`, `Y_aa`: Yield coefficients [kg COD/kg COD]

**Medium Priority**:
- `k_m_c4`, `k_m_pro`, `k_m_ac`, `k_m_h2`: Maximum uptake rates [1/d]
- `K_S_su`, `K_S_aa`: Half-saturation constants [kg COD/m³]

**Low Priority** (usually well-defined):
- Stoichiometric coefficients (C, N content)
- Physical-chemical constants (K_a, K_H)

TODO: add publications that confirm which parameters have which priority

### Example: Single-Parameter Calibration

```python
# Calibrate disintegration rate only
result = calibrator.calibrate_initial(
    measurements=measurements,
    parameters=["k_dis"],
    bounds={"k_dis": (0.3, 0.8)},
    objectives=["Q_ch4"],
    max_iterations=50
)

print(f"Optimal k_dis: {result.parameters['k_dis']:.3f}")
print(f"Objective value: {result.objective_value:.4f}")
```

### Example: Multi-Parameter Calibration

```python
# Calibrate multiple substrate-dependent parameters
result = calibrator.calibrate_initial(
    measurements=measurements,
    parameters=["k_dis", "k_hyd_ch", "k_hyd_pr", "Y_su"],
    bounds={
        "k_dis": (0.3, 0.8),
        "k_hyd_ch": (5.0, 15.0),
        "k_hyd_pr": (5.0, 15.0),
        "Y_su": (0.05, 0.15)
    },
    objectives=["Q_ch4", "pH", "VFA"],
    weights={"Q_ch4": 0.6, "pH": 0.2, "VFA": 0.2},
    method="differential_evolution",
    validation_split=0.2,
    max_iterations=100,
    population_size=15
)
```

### Optimization Methods

#### Differential Evolution (Default, Recommended)

Best for: Global optimization, multiple local minima

```python
result = calibrator.calibrate_initial(
    measurements=measurements,
    parameters=["k_dis", "Y_su"],
    method="differential_evolution",
    max_iterations=100,
    population_size=15,  # 15 × n_parameters recommended
    tolerance=1e-4
)
```

**Advantages**:
- Global optimization (avoids local minima)
- Robust to parameter scaling
- No gradient calculation needed

**Disadvantages**:
- Slower than local methods
- Requires more function evaluations

% TODO: add publication to DE

#### Nelder-Mead (Local)

Best for: Fast refinement near known optimum

```python
result = calibrator.calibrate_initial(
    measurements=measurements,
    parameters=["k_dis", "Y_su"],
    method="nelder_mead",
    max_iterations=50,
    tolerance=1e-4
)
```

**Advantages**:
- Fast convergence
- Simple implementation
- No gradient required

**Disadvantages**:
- Local optimization only
- May get stuck in local minima
- Sensitive to initial guess

#### L-BFGS-B (Gradient-Based)

Best for: When gradients are available, large-scale problems

```python
result = calibrator.calibrate_initial(
    measurements=measurements,
    parameters=["k_dis", "Y_su", "k_hyd_ch"],
    method="lbfgsb",
    max_iterations=50,
    tolerance=1e-6
)
```

**Advantages**:
- Very fast convergence
- Handles box constraints
- Memory efficient

**Disadvantages**:
- Local optimization
- Requires numerical gradients (slower in practice)

% TODO: add publication to L-BFGS-B

### Multi-Objective Calibration

Balance multiple objectives with weighted sum:

```python
result = calibrator.calibrate_initial(
    measurements=measurements,
    parameters=["k_dis", "Y_su", "k_hyd_ch"],
    objectives=["Q_ch4", "pH", "VFA"],
    weights={
        "Q_ch4": 0.6,  # Most important
        "pH": 0.2,     # Secondary
        "VFA": 0.2     # Secondary
    },
    method="differential_evolution"
)
```

**Weight Selection Guidelines**:
- Sum to 1.0 (automatically normalized)
- Higher weight = more influence on optimization
- Gas production typically most important (0.6-0.8)
- Process stability (pH, VFA) secondary (0.2-0.4)

### Validation Split

Reserve data for validation:

```python
result = calibrator.calibrate_initial(
    measurements=measurements,
    parameters=["k_dis", "Y_su"],
    validation_split=0.2,  # 20% for validation
    objectives=["Q_ch4", "pH"]
)

# Check validation metrics
print(f"Training RMSE: {result.objective_value:.2f}")
print(f"Validation R²: {result.validation_metrics['Q_ch4_r2']:.3f}")
```

**Recommendations**:
- Use 20-30% for validation
- Avoid over-fitting with small datasets
- Check validation metrics match training metrics

## Sensitivity Analysis

Identify influential parameters:

```python
# After calibration
sensitivity = calibrator.sensitivity_analysis(
    parameters=result.parameters,
    measurements=measurements,
    objectives=["Q_ch4", "pH", "VFA"],
    perturbation=0.01  # 1% parameter perturbation
)

# Analyze results
for param, sens_result in sensitivity.items():
    print(f"\n{param}:")
    print(f"  Base value: {sens_result.base_value:.4f}")
    print("  Sensitivity indices:")
    for obj, index in sens_result.sensitivity_indices.items():
        print(f"    {obj}: {index:+.4e}")
```

**Interpretation**:
- **High sensitivity** (|S| > 0.1): Parameter strongly affects output
- **Low sensitivity** (|S| < 0.01): Parameter has minimal effect
- **Negative sensitivity**: Inverse relationship

## Identifiability Analysis

Check if parameters can be reliably estimated:

```python
identifiability = calibrator.identifiability_analysis(
    parameters=result.parameters,
    measurements=measurements,
    confidence_level=0.95,
    correlation_threshold=0.8
)

# Review identifiability
for param, ident in identifiability.items():
    if not ident.is_identifiable:
        print(f"⚠ {param}: {ident.reason}")

        # Check correlations
        for other, corr in ident.correlation_with.items():
            if abs(corr) > 0.8:
                print(f"  High correlation with {other}: {corr:.3f}")
```

**Common Issues**:

1. **Low Sensitivity**: Parameter doesn't affect measured outputs
   - **Solution**: Remove from calibration or get better measurements

2. **High Correlation**: Two parameters compensate for each other
   - **Solution**: Fix one parameter or add discriminating measurements

3. **Wide Confidence Interval**: Insufficient data to estimate parameter
   - **Solution**: Collect more data or use prior knowledge

## Online Re-Calibration

Adapt parameters during plant operation:

```python
# Configure trigger conditions
calibrator.set_trigger(
    variance_threshold=0.15,      # Trigger when variance > 15%
    time_threshold=24.0,          # Minimum 24h between calibrations
    consecutive_violations=3      # Require 3 consecutive violations
)

# Check if re-calibration needed
should_cal, reason = calibrator.should_recalibrate(recent_data)

if should_cal:
    print(f"Re-calibration triggered: {reason}")

    # Re-calibrate with bounded changes
    result = calibrator.calibrate_online(
        measurements=recent_data,
        parameters=["k_dis", "Y_su"],
        max_parameter_change=0.10,  # Max 10% change
        time_window=7,               # Use last 7 days
        method="nelder_mead"         # Fast local optimization
    )

    if result.success:
        calibrator.apply_calibration(result)
```

### When to Use Online Re-Calibration

Appropriate for:
- Long-term operation (months)
- Gradual substrate property changes
- Seasonal variations
- Equipment drift

**Not appropriate for**:
- Short-term disturbances (handle with control)
- Sensor failures (fix measurements first)
- Major process changes (use initial calibration)

### Parameter Change Limits

Prevent unrealistic drift:

```python
result = calibrator.calibrate_online(
    measurements=recent_data,
    parameters=["k_dis", "Y_su"],
    max_parameter_change=0.10,  # ±10% maximum
    time_window=7
)

# Check parameter changes
for param, new_val in result.parameters.items():
    old_val = result.initial_parameters[param]
    change_pct = ((new_val - old_val) / old_val) * 100
    print(f"{param}: {old_val:.3f} → {new_val:.3f} ({change_pct:+.1f}%)")
```

**Recommended Limits**:
- **Conservative**: 5-10% (most common)
- **Moderate**: 10-20% (seasonal changes)
- **Aggressive**: 20-30% (major changes, use carefully)

## Validation and Quality Checks

### Goodness-of-Fit Metrics

```python
from pyadm1.calibration.validation import CalibrationValidator

validator = CalibrationValidator(plant)

metrics = validator.validate(
    parameters=result.parameters,
    measurements=validation_data,
    objectives=["Q_ch4", "pH", "VFA"]
)

# Review metrics
for obj, obj_metrics in metrics.items():
    print(f"\n{obj}:")
    print(f"  RMSE: {obj_metrics.rmse:.2f}")
    print(f"  R²:   {obj_metrics.r2:.3f}")
    print(f"  NSE:  {obj_metrics.nse:.3f}")
    print(f"  PBIAS: {obj_metrics.pbias:.1f}%")
```

**Metric Interpretation**:

| Metric                          | Excellent | Good | Fair | Poor |
|---------------------------------|-----------|------|------|------|
| R²                              | > 0.90 | 0.75-0.90 | 0.50-0.75 | < 0.50 |
| NSE (Nash-Sutcliffe Efficiency) | > 0.90 | 0.70-0.90 | 0.50-0.70 | < 0.50 |
| PBIAS (Percent Bias)            | < ±5% | ±5-±10% | ±10-±25% | > ±25% |

### Residual Analysis

```python
# Analyze residuals
residuals = validator.analyze_residuals(
    measurements=validation_data,
    simulated=simulated_outputs,
    objectives=["Q_ch4", "pH"]
)

for obj, res_analysis in residuals.items():
    print(f"\n{obj}:")

    # Check normality
    if not res_analysis.is_normally_distributed():
        print("  ⚠ Residuals not normally distributed")

    # Check autocorrelation
    if res_analysis.has_autocorrelation():
        print(f"  ⚠ Significant autocorrelation: {res_analysis.autocorrelation:.3f}")

    # Check heteroscedasticity
    if res_analysis.has_heteroscedasticity():
        print("  ⚠ Heteroscedasticity detected")

    # Report outliers
    n_outliers = len(res_analysis.outlier_indices)
    if n_outliers > 0:
        pct = n_outliers / len(res_analysis.residuals) * 100
        print(f"  ⚠ {n_outliers} outliers ({pct:.1f}%)")
```

### Cross-Validation

Test generalization with k-fold cross-validation:

```python
cv_results = validator.cross_validate(
    parameters=result.parameters,
    measurements=full_dataset,
    n_folds=5,
    objectives=["Q_ch4", "pH"]
)

# Calculate mean metrics
for obj in ["Q_ch4", "pH"]:
    r2_values = [m.r2 for m in cv_results[obj]]
    rmse_values = [m.rmse for m in cv_results[obj]]

    print(f"\n{obj} (5-fold CV):")
    print(f"  R²:   {np.mean(r2_values):.3f} ± {np.std(r2_values):.3f}")
    print(f"  RMSE: {np.mean(rmse_values):.2f} ± {np.std(rmse_values):.2f}")
```

## Best Practices

### 1. Measurement Data Quality

**Essential Requirements**:
- ✓ Hourly or daily measurements
- ✓ Minimum 2 weeks of data
- ✓ Stable operation period
- ✓ Accurate substrate feed records
- ✓ < 10% missing data

**Data Preparation**:
```python
# Load and validate
measurements = MeasurementData.from_csv("plant_data.csv")
validation = measurements.validate()

if validation.quality_score < 0.7:
    print("⚠ Data quality issues - see report")
    validation.print_report()

# Clean data
measurements.remove_outliers(method="zscore", threshold=3.0)
measurements.fill_gaps(method="interpolate", limit=3)
```

### 2. Parameter Selection Strategy

**Start Small**:
1. Begin with 1-2 most sensitive parameters
2. Add parameters incrementally
3. Check identifiability at each step
4. Limit to 3-5 parameters for typical datasets

**Example Progression**:
```python
# Step 1: Most sensitive parameter
result_1 = calibrator.calibrate_initial(
    measurements=measurements,
    parameters=["k_dis"],
    objectives=["Q_ch4"]
)

# Step 2: Add hydrolysis rates
result_2 = calibrator.calibrate_initial(
    measurements=measurements,
    parameters=["k_dis", "k_hyd_ch"],
    objectives=["Q_ch4", "pH"]
)

# Step 3: Add yield coefficient
result_3 = calibrator.calibrate_initial(
    measurements=measurements,
    parameters=["k_dis", "k_hyd_ch", "Y_su"],
    objectives=["Q_ch4", "pH", "VFA"]
)
```

### 3. Computational Efficiency

**Quick Calibration** (< 5 min):
```python
result = calibrator.calibrate_initial(
    measurements=measurements,
    parameters=["k_dis", "Y_su"],
    method="nelder_mead",
    max_iterations=50
)
```

**Thorough Calibration** (10-30 min):
```python
result = calibrator.calibrate_initial(
    measurements=measurements,
    parameters=["k_dis", "k_hyd_ch", "Y_su"],
    method="differential_evolution",
    max_iterations=100,
    population_size=15,
    sensitivity_analysis=True
)
```

% TODO: actually measure the time needed for calibration

### 4. Result Interpretation

**Check Success Indicators**:
```python
if result.success:
    # 1. Reasonable objective value
    if result.objective_value < 1.0:  # Depends on scale
        print("✓ Good fit achieved")

    # 2. Parameters within expected ranges
    for param, value in result.parameters.items():
        bounds = calibrator.parameter_bounds.get_bounds(param)
        if bounds.lower < value < bounds.upper:
            print(f"✓ {param} within bounds")

    # 3. Validation metrics acceptable
    if result.validation_metrics.get('Q_ch4_r2', 0) > 0.75:
        print("✓ Good validation performance")
```

### 5. Documentation

Keep calibration records:
```python
# Save complete calibration result
result.to_json("calibration_result.json")

# Generate report
report = calibrator.generate_report("calibration_report.txt")

# Track calibration history
calibrator.save_history("calibration_history.json")
```

## Common Issues and Solutions

### Issue 1: Calibration Fails to Converge

**Symptoms**:
- `result.success = False`
- High objective value
- Maximum iterations reached

**Solutions**:
1. Reduce number of parameters
2. Tighten parameter bounds
3. Use different initial guess
4. Increase max iterations
5. Try different optimization method

```python
# Solution: Tighter bounds and more iterations
result = calibrator.calibrate_initial(
    measurements=measurements,
    parameters=["k_dis"],  # Reduce to one parameter
    bounds={"k_dis": (0.4, 0.6)},  # Tighter bounds
    max_iterations=200,  # More iterations
    method="differential_evolution"
)
```

### Issue 2: Parameters Not Identifiable

**Symptoms**:
- High parameter correlation (> 0.8)
- Wide confidence intervals
- Low sensitivity

**Solutions**:
1. Remove highly correlated parameters
2. Collect more discriminating measurements
3. Fix one of the correlated parameters

```python
# Check correlations
identifiability = calibrator.identifiability_analysis(
    parameters=result.parameters,
    measurements=measurements
)

# Remove problematic parameter
if not identifiability["Y_su"].is_identifiable:
    # Fix Y_su, calibrate only k_dis
    result = calibrator.calibrate_initial(
        measurements=measurements,
        parameters=["k_dis"],  # Remove Y_su
        objectives=["Q_ch4"]
    )
```

### Issue 3: Poor Validation Performance

**Symptoms**:
- Low validation R² (< 0.5)
- High PBIAS (> 25%)
- Large validation/training gap

**Solutions**:
1. Check for over-fitting (too many parameters)
2. Verify data quality
3. Ensure validation period is representative

```python
# Solution: Reduce parameters
result = calibrator.calibrate_initial(
    measurements=measurements,
    parameters=["k_dis", "k_hyd_ch"],  # Reduced from 5 parameters
    validation_split=0.3,  # More validation data
    objectives=["Q_ch4", "pH"]
)

# Check validation
if result.validation_metrics['Q_ch4_r2'] < 0.5:
    print("⚠ Still poor - check data quality")
    validation = measurements.validate()
    validation.print_report()
```

### Issue 4: Unrealistic Parameter Values

**Symptoms**:
- Parameters at bounds
- Physically impossible values

**Solutions**:
1. Check parameter bounds (too wide?)
2. Verify measurement data quality
3. Check model structure (missing processes?)

```python
# Check if parameters hit bounds
for param, value in result.parameters.items():
    bounds = calibrator.parameter_bounds.get_bounds(param)

    if value <= bounds.lower + 0.01:
        print(f"⚠ {param} at lower bound: {value:.3f}")
    elif value >= bounds.upper - 0.01:
        print(f"⚠ {param} at upper bound: {value:.3f}")
```

## Example Workflows

### Complete Initial Calibration

See `scripts/calibration_example.py` for a full working example with:
- Plant setup
- Measurement data preparation
- Parameter calibration
- Sensitivity analysis
- Identifiability assessment
- Validation
- Application to plant model

### Automated Online Monitoring

```python
import time
from datetime import datetime

# Setup
plant = BiogasPlant.from_json("plant.json", feedstock)
calibrator = Calibrator(plant)

# Configure trigger
calibrator.set_trigger(
    variance_threshold=0.15,
    time_threshold=24.0,
    consecutive_violations=3
)

# Monitoring loop (simplified)
while True:
    # Get recent measurements (last 7 days)
    recent_data = get_recent_measurements(days=7)

    # Check if re-calibration needed
    should_cal, reason = calibrator.should_recalibrate(recent_data)

    if should_cal:
        print(f"{datetime.now()}: Re-calibration triggered - {reason}")

        # Re-calibrate
        result = calibrator.calibrate_online(
            measurements=recent_data,
            max_parameter_change=0.10,
            time_window=7
        )

        if result.success:
            calibrator.apply_calibration(result)
            print("✓ Re-calibration applied")
        else:
            print(f"✗ Re-calibration failed: {result.message}")

    # Wait 1 hour
    time.sleep(3600)
```

## Next Steps

- **[Components Guide](components.md)**: Learn about model components
- **[Parallel Simulation](../examples/parallel_simulation.md)**: Run parameter sweeps
- **[API Reference](../api_reference/calibration.rst)**: Detailed API documentation

## References

1. **Batstone et al. (2002)**: Anaerobic Digestion Model No. 1 (ADM1). IWA Publishing.
2. **Gaida (2014)**: Dynamic real-time substrate feed optimization of anaerobic co-digestion plants. PhD thesis.
3. **Dochain & Vanrolleghem (2001)**: Dynamical Modelling & Estimation in Wastewater Treatment Processes.
