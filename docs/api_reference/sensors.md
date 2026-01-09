# Sensors Components

Measurement and Sensor Components

Virtual sensors for process monitoring and control, with realistic measurement
characteristics including noise, drift, and response times.

Modules:
    physical: Physical property sensors including pH electrodes (with calibration drift),
             temperature sensors (PT100, thermocouples), pressure transmitters, level
             sensors, and flow meters with accuracy specifications.

    chemical: Chemical analysis sensors for process liquids including VFA analyzers
             (online titration, GC), ammonia sensors (ion-selective electrodes),
             COD analyzers (online spectroscopy), and nutrient analyzers.

    gas: Gas composition analyzers for biogas quality monitoring including methane
        sensors (infrared, calorimetric), CO2 sensors, H2S sensors (electrochemical),
        oxygen sensors, and trace gas analyzers with detection limits.

Example:

```python
    >>> from pyadm1.components.sensors import PhysicalSensor, ChemicalSensor, GasSensor
    >>>
    >>> # pH sensor with realistic noise
    >>> ph_sensor = PhysicalSensor("ph1", sensor_type="pH",
    ...                           measurement_noise=0.05, drift_rate=0.01)
    >>>
    >>> # VFA analyzer with sampling delay
    >>> vfa_sensor = ChemicalSensor("vfa1", sensor_type="VFA",
    ...                            measurement_delay=5, accuracy=0.1)
    >>>
    >>> # Methane analyzer
    >>> ch4_sensor = GasSensor("ch4_1", sensor_type="CH4",
    ...                       measurement_range=(0, 100), accuracy=0.5)
```

