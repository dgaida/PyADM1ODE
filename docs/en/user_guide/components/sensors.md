# Sensors

Measurement and monitoring components for biogas plants.

## Overview

PyADM1 provides a sensor module for realistic process monitoring and control. These components model realistic measurement characteristics including noise, drift, and response times. See [API Reference](../../api/sensors.md) for the class-level documentation.

## Sensor Categories

### Physical Sensors

Sensors for physical properties:

- **pH electrodes**: With calibration drift and temperature compensation
- **Temperature sensors**: PT100, thermocouples with accuracy specifications
- **Pressure sensors**: For gas and liquid pressure
- **Level sensors**: Ultrasonic, radar, hydrostatic
- **Flow meters**: Magnetic-inductive, ultrasonic, Coriolis

### Chemical Sensors

Sensors for process liquid analysis:

- **VFA analyzers**: Online titration, GC analysis
- **Ammonia sensors**: Ion-selective electrodes
- **COD analyzers**: Online spectroscopy
- **Nutrient analyzers**: N, P, K measurement

### Gas Analyzers

Sensors for biogas composition:

- **Methane sensors**: Infrared, calorimetric
- **CO2 sensors**: NDIR technology
- **H2S sensors**: Electrochemical
- **Oxygen sensors**: For leak detection
- **Trace-gas analyzers**: With detection limits

## Usage Example

```python
from pyadm1.components.sensors import PhysicalSensor, ChemicalSensor, GasSensor

# pH sensor with realistic noise
ph_sensor = PhysicalSensor(
    "ph1",
    sensor_type="pH",
    measurement_noise=0.05,    # ±0.05 pH units
    drift_rate=0.01,           # Calibration drift per day
    response_time=30.0,        # Seconds
    calibration_interval=7     # Days between calibrations
)

# VFA analyzer with measurement delay
vfa_sensor = ChemicalSensor(
    "vfa1",
    sensor_type="VFA",
    measurement_delay=5,       # Minutes delay
    accuracy=0.1,              # ±10% accuracy
    detection_limit=0.1,       # g/L lower limit
    sampling_interval=60       # Minutes between samples
)

# Methane analyzer
ch4_sensor = GasSensor(
    "ch4_1",
    sensor_type="CH4",
    measurement_range=(0, 100),  # % CH4
    accuracy=0.5,                # ±0.5%
    drift=0.1,                   # % per week
    cross_sensitivity={'CO2': 0.01}  # Cross-sensitivity
)
```

## Sensor Characteristics

### Noise and Accuracy

Realistic sensors exhibit several error sources:

```python
# Measured value = True value + Systematic error + Random noise + Drift

measured_value = (
    true_value * (1 + systematic_error) +
    random.normal(0, noise_level) +
    drift_accumulated
)
```

**Typical accuracies:**

| Sensor | Measurement range | Accuracy | Drift |
|--------|-------------------|----------|-------|
| pH | 0–14 | ±0.05 pH | 0.01 pH/day |
| Temperature (PT100) | −50 to 200 °C | ±0.1 °C | <0.01 °C/year |
| Pressure | 0–2 bar | ±0.5% FS | <0.1% FS/year |
| VFA | 0–20 g/L | ±5% | ±0.1 g/L/week |
| CH4 | 0–100% | ±0.5% | ±0.1%/week |

### Response Time

Sensors have characteristic response times:

```python
# First-order response
sensor_value(t) = true_value * (1 - exp(-t/τ))

# where τ = time constant (63.2% response time)
```

**Typical response times:**

| Sensor | τ (time to 63% response) | t95 (time to 95% response) |
|--------|--------------------------|----------------------------|
| pH electrode | 10–30 s | 30–90 s |
| Thermocouple | 0.1–5 s | 0.3–15 s |
| CH4 sensor (IR) | 5–10 s | 15–30 s |
| VFA analyzer | 2–5 min | 6–15 min |

### Calibration and Maintenance

Sensors require regular calibration:

```python
# Calibration drift model
drift(t) = drift_rate * (t - last_calibration)

if (t - last_calibration) > calibration_interval:
    # Calibration required
    perform_calibration()
    last_calibration = t
    accumulated_drift = 0
```

## Sensor Placement

### Digester Monitoring

**Recommended sensors for a digester:**

```python
# Minimal instrumentation
sensors_minimum = {
    'temperature': ['T1'],           # Digester temperature
    'pH': ['pH1'],                   # Process pH
    'gas_flow': ['Q_gas'],           # Gas production
    'gas_composition': ['CH4_CO2']   # Methane content
}

# Standard instrumentation
sensors_standard = {
    'temperature': ['T1', 'T2'],     # Digester + ambient
    'pH': ['pH1'],
    'VFA': ['VFA1'],                 # VFA concentration
    'gas_flow': ['Q_gas'],
    'gas_composition': ['CH4_CO2', 'H2S'],
    'level': ['L1']                  # Liquid level
}

# Advanced instrumentation
sensors_advanced = {
    'temperature': ['T1', 'T2', 'T3'],  # Multiple points
    'pH': ['pH1', 'pH2'],                # Redundancy
    'VFA': ['VFA1'],
    'ammonia': ['NH3'],                  # Inhibition monitoring
    'gas_flow': ['Q_gas'],
    'gas_composition': ['CH4_CO2', 'H2S', 'O2'],
    'pressure': ['P_gas'],               # Gas storage pressure
    'level': ['L1'],
    'conductivity': ['EC1']              # Process stability
}
```

## Data Processing

### Filtering and Smoothing

Raw sensor data often needs filtering:

```python
# Exponential smoothing
def exponential_smoothing(measurements, alpha=0.3):
    """
    Smooth sensor measurements.

    alpha: smoothing factor (0–1)
           0 = no smoothing
           1 = only the current value
    """
    smoothed = [measurements[0]]
    for m in measurements[1:]:
        s = alpha * m + (1 - alpha) * smoothed[-1]
        smoothed.append(s)
    return smoothed

# Moving average
def moving_average(measurements, window=5):
    """Moving average."""
    return [
        sum(measurements[max(0, i-window):i+1]) /
        min(window, i+1)
        for i in range(len(measurements))
    ]
```

### Outlier Detection

```python
def detect_outliers(measurements, threshold=3.0):
    """
    Detect outliers via Z-score.

    threshold: standard deviations for outlier classification
    """
    mean = sum(measurements) / len(measurements)
    std = (sum((x - mean)**2 for x in measurements) / len(measurements))**0.5

    outliers = []
    for i, m in enumerate(measurements):
        z_score = abs(m - mean) / std if std > 0 else 0
        if z_score > threshold:
            outliers.append(i)

    return outliers
```

## Process Control with Sensors

### pH Control

```python
def ph_control(ph_measurement, setpoint=7.2, tolerance=0.2):
    """
    Simple pH control.

    Returns: lime dosing rate [kg/d]
    """
    error = setpoint - ph_measurement

    if error > tolerance:
        # pH too low, add lime
        dosing_rate = min(100, error * 50)  # Proportional control
    elif error < -tolerance:
        # pH too high, reduce lime
        dosing_rate = 0
    else:
        # Within tolerance
        dosing_rate = 0

    return dosing_rate
```

### Feeding Control Based on VFA

```python
def adaptive_feeding_control(vfa_measurement, vfa_limit=4.0,
                             current_feed_rate=15.0):
    """
    Adjust feeding rate based on VFA.

    vfa_limit: VFA threshold [g/L]
    current_feed_rate: current rate [m³/d]

    Returns: adjusted feed rate [m³/d]
    """
    if vfa_measurement > vfa_limit:
        # Acidification risk - reduce feeding
        reduction_factor = vfa_limit / vfa_measurement
        new_rate = current_feed_rate * reduction_factor
        print(f"VFA high ({vfa_measurement:.2f} g/L) - reducing to {new_rate:.1f} m³/d")
    elif vfa_measurement < vfa_limit * 0.5:
        # Stable - can increase feeding
        increase_factor = 1.05  # 5% increase
        new_rate = min(current_feed_rate * increase_factor, 20.0)  # Max 20 m³/d
        print(f"VFA stable ({vfa_measurement:.2f} g/L) - increasing to {new_rate:.1f} m³/d")
    else:
        # Within optimal range
        new_rate = current_feed_rate

    return new_rate
```

## Alarms and Notifications

### Alarm System

```python
class SensorAlarm:
    """Sensor alarm system."""

    def __init__(self, sensor_id, alarm_type, threshold, hysteresis=0.1):
        self.sensor_id = sensor_id
        self.alarm_type = alarm_type  # 'high', 'low', 'rate_of_change'
        self.threshold = threshold
        self.hysteresis = hysteresis
        self.is_active = False
        self.last_value = None

    def check(self, current_value):
        """Check alarm condition."""

        if self.alarm_type == 'high':
            if not self.is_active and current_value > self.threshold:
                self.is_active = True
                return f"ALARM: {self.sensor_id} high ({current_value:.2f})"
            elif self.is_active and current_value < (self.threshold - self.hysteresis):
                self.is_active = False
                return f"OK: {self.sensor_id} normal ({current_value:.2f})"

        elif self.alarm_type == 'low':
            if not self.is_active and current_value < self.threshold:
                self.is_active = True
                return f"ALARM: {self.sensor_id} low ({current_value:.2f})"
            elif self.is_active and current_value > (self.threshold + self.hysteresis):
                self.is_active = False
                return f"OK: {self.sensor_id} normal ({current_value:.2f})"

        elif self.alarm_type == 'rate_of_change':
            if self.last_value is not None:
                rate = abs(current_value - self.last_value)
                if not self.is_active and rate > self.threshold:
                    self.is_active = True
                    return f"ALARM: {self.sensor_id} rapid change ({rate:.2f}/h)"
            self.last_value = current_value

        return None

# Example usage
alarms = {
    'pH_low': SensorAlarm('pH1', 'low', 6.8, hysteresis=0.1),
    'pH_high': SensorAlarm('pH1', 'high', 8.0, hysteresis=0.1),
    'VFA_high': SensorAlarm('VFA1', 'high', 4.0, hysteresis=0.5),
    'temp_deviation': SensorAlarm('T1', 'rate_of_change', 2.0)  # 2 °C/h
}

# In the simulation loop
for measurement in ph_measurements:
    for alarm in alarms.values():
        message = alarm.check(measurement)
        if message:
            print(message)
```

## Data Logging

### Logging Sensor Data

```python
import csv
from datetime import datetime

class SensorDataLogger:
    """Log sensor data."""

    def __init__(self, filename):
        self.filename = filename
        self.file = None
        self.writer = None

    def open(self, sensor_ids):
        """Open log file."""
        self.file = open(self.filename, 'w', newline='')
        self.writer = csv.writer(self.file)
        # Header
        self.writer.writerow(['timestamp', 'time_days'] + sensor_ids)

    def log(self, time_days, sensor_values):
        """Log a single time point."""
        timestamp = datetime.now().isoformat()
        row = [timestamp, time_days] + sensor_values
        self.writer.writerow(row)

    def close(self):
        """Close the log file."""
        if self.file:
            self.file.close()

# Usage
logger = SensorDataLogger('sensor_data.csv')
logger.open(['pH', 'T', 'VFA', 'Q_gas', 'CH4'])

# In simulation
for t in range(simulation_steps):
    # ... simulation ...
    sensor_values = [
        digester.outputs_data['pH'],
        digester.T_ad,
        digester.outputs_data['VFA'],
        digester.outputs_data['Q_gas'],
        digester.outputs_data['Q_ch4']
    ]
    logger.log(t, sensor_values)

logger.close()
```

## Using Digester Outputs as Virtual Sensors

The digester component already exposes process indicators that can be used directly for monitoring, with or without dedicated sensor components:

```python
from pyadm1.components.biological import Digester

# Digester already provides process indicators
digester = Digester("dig1", feedstock, V_liq=2000)
result = digester.step(t, dt, inputs)

# Available "sensor" values
monitoring_data = {
    'pH': result['pH'],
    'VFA': result['VFA'],            # g/L
    'TAC': result['TAC'],            # g CaCO3/L
    'Q_gas': result['Q_gas'],        # m³/d
    'Q_ch4': result['Q_ch4'],        # m³/d
    'Q_co2': result['Q_co2']         # m³/d
}

# Simple process monitoring
if monitoring_data['pH'] < 6.8:
    print("Warning: low pH")

if monitoring_data['VFA'] / monitoring_data['TAC'] > 0.4:
    print("Warning: high VFA/TAC ratio")
```

## Roadmap

Planned extensions to the sensor module:

1. **Richer sensor models**
   - Noise and drift
   - Calibration cycles
   - Failure models

2. **Advanced process control**
   - PID controllers
   - Model predictive control (MPC)
   - Adaptive control

3. **Data analysis tools**
   - Trend analysis
   - Anomaly detection
   - Predictive maintenance

4. **Visualization**
   - Real-time dashboards
   - Historical trends
   - Alarm overviews

## Next Steps

- [Biological Components](biological.md): Digester and process control
- [Energy Components](energy.md): CHP and heating systems
- [Mechanical Components](mechanical.md): Pumps and mixers
- [Feeding Components](feeding.md): Storage and dosing
- [API Reference](../../api/sensors.md): Detailed class documentation
