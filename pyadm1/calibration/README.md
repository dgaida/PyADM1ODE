# PyADM1 Calibration Package

## Parameter Bounds Module

### File: `pyadm1/calibration/parameter_bounds.py`

**Purpose**: Manage parameter bounds for calibration with physically plausible ranges, constraint handling, and penalty functions.

### Key Classes:

#### `ParameterBound` (dataclass)
Represents bounds for a single parameter with:
- Lower and upper bounds
- Default value
- Bound type (HARD, SOFT, FIXED)
- Penalty weight for soft constraints
- Methods for validation, clipping, and penalty calculation

#### `BoundType` (Enum)
- **HARD**: Must not be violated (infinite penalty)
- **SOFT**: Can be violated with penalty
- **FIXED**: Parameter is not calibrated

#### `ParameterBounds` (Manager class)
Manages all parameter bounds with methods for:
- Adding/getting bounds
- Validation
- Penalty calculation
- Parameter scaling/unscaling
- Batch operations

### Key Features:

1. **Default ADM1 Bounds** via `create_default_bounds()`:
   - 35+ ADM1 parameters with literature-based ranges
   - Substrate-dependent parameters marked
   - Different bound types and penalty weights

2. **Flexible Constraint Types**:
   - Hard constraints (must satisfy)
   - Soft constraints (with penalties)
   - Fixed parameters

3. **Multiple Penalty Functions**:
   - Quadratic: `(distance)²`
   - Linear: `|distance|`
   - Logarithmic: `-log(distance)`
   - Barrier: `1/distance`

4. **Parameter Scaling**:
   - Scale to unit interval [0,1] for optimization
   - Unscale back to original range

### Usage Example:

```python
from pyadm1.calibration.parameter_bounds import create_default_bounds

# Get default bounds
bounds = create_default_bounds()

# Check validity
is_valid = bounds.is_within_bounds("k_dis", 0.5)

# Calculate penalty
penalty = bounds.calculate_penalty("k_dis", 1.0)

# Validate multiple parameters
params = {"k_dis": 0.5, "Y_su": 0.10}
is_valid, errors = bounds.validate_parameters(params)
```

---

## Measurement Data Module

### File: `pyadm1/io/measurement_data.py`

**Purpose**: Load, validate, and preprocess biogas plant measurement data for calibration.

### Key Classes:

#### `MeasurementData`
Main container for time series measurements with methods for:
- Loading from CSV/Excel
- Validation
- Outlier detection and removal
- Gap filling
- Resampling
- Data extraction

#### `DataValidator`
Validates data quality with:
- Required column checks
- Range validation
- Missing data analysis
- Duplicate detection
- Quality score calculation

#### `OutlierDetector`
Multiple outlier detection methods:
- **Z-score**: Statistical outlier detection
- **IQR**: Interquartile range method
- **Moving window**: Local outlier detection

#### `ValidationResult` (dataclass)
Comprehensive validation report with:
- Overall status and quality score
- List of issues and warnings
- Statistics and missing data percentages
- Formatted report printing

### Key Features:

1. **Flexible Data Loading**:
   - CSV with automatic date parsing
   - Configurable timestamp columns
   - Automatic resampling
   - Metadata support

2. **Data Quality Management**:
   - Comprehensive validation
   - Quality scoring (0-1)
   - Expected range checking
   - Missing data analysis

3. **Outlier Handling**:
   - Multiple detection methods
   - Configurable thresholds
   - Per-column processing

4. **Gap Filling Methods**:
   - Linear interpolation
   - Forward/backward fill
   - Mean/median fill
   - Configurable limits

5. **Data Extraction**:
   - Time series by column
   - Substrate feed arrays
   - Time window selection
   - Statistical summaries

### Usage Example:

```python
from pyadm1.io.measurement_data import MeasurementData

# Load data
data = MeasurementData.from_csv(
    "plant_measurements.csv",
    timestamp_column="timestamp",
    resample="1H"
)

# Validate
validation = data.validate()
validation.print_report()

# Clean data
data.remove_outliers(method="zscore", threshold=3.0)
data.fill_gaps(method="interpolate", limit=3)

# Extract measurements
pH = data.get_measurement("pH")
Q = data.get_substrate_feeds()
```

---

## Integration with Existing Calibration System

These modules integrate seamlessly with the existing calibration framework:

### In `calibration/calibrator.py`:
```python
from pyadm1.calibration.parameter_bounds import create_default_bounds
from pyadm1.io.measurement_data import MeasurementData

# In Calibrator.__init__
self.parameter_bounds = create_default_bounds()

# In calibrate_initial
measurements = MeasurementData.from_csv(measurement_file)
measurements.validate()
measurements.remove_outliers()
```

### In `calibration/initial.py`:
```python
# Use bounds in optimization
bounds_list = [self.parameter_bounds.get_bounds_tuple(p)
               for p in parameters]

# Add penalty to objective
penalty = self.parameter_bounds.calculate_total_penalty(params)
objective += penalty
```

---

## File Structure

```
pyadm1/
├── calibration/
│   ├── __init__.py
│   ├── calibrator.py
│   ├── initial.py
│   ├── online.py
│   ├── parameter_bounds.py
│   └── optimization/
│       └── ...
├── io/
│   ├── __init__.py
│   ├── measurement_data.py
│   └── ...
└── ...
```

---

## Performance Considerations

### Calibration Speed

**Typical Runtime**:
- 3 parameters, 100 iterations, 30 days data: ~5-10 minutes
- 5 parameters, 200 iterations, 30 days data: ~20-30 minutes

**Optimization**:
- Use parallel evaluation in differential evolution (workers parameter)
- Reduce simulation duration if possible
- Use coarser time steps for initial calibration

### Memory Usage

**Per Simulation**:
- ~50 MB for 30-day simulation
- Optimization history can grow large (limit to 1000 entries)

**Tips**:
- Clear history between calibration runs
- Use sparse save intervals in simulation
- Don't save full time series unless needed

---

## Future Enhancements

### Planned Features
1. Multi-stage calibration (coarse then fine)
2. Bayesian calibration with uncertainty quantification
3. Real-time calibration for online operation
4. Automated parameter selection
5. Parallel simulation for faster calibration

### Integration Points
- MCP server for LLM-driven calibration
- Web interface for calibration workflows
- Database storage for calibration history
- Automated reporting and visualization

---

## References

1. **ADM1 Model**: Batstone et al. (2002). Anaerobic Digestion Model No. 1.
2. **Calibration Methods**: Gaida, D. (2014). Dynamic real-time substrate feed optimization.
3. **Sensitivity Analysis**: Saltelli et al. (2008). Global Sensitivity Analysis.
4. **Parameter Estimation**: Dochain & Vanrolleghem (2001). Dynamical Modelling of Bioprocesses.
