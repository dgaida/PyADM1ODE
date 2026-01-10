# Sensoren

Mess- und Überwachungskomponenten für Biogasanlagen (in Entwicklung).

## Übersicht

PyADM1 wird in Zukunft ein umfassendes Sensormodul für realistische Prozessüberwachung und -steuerung enthalten. Diese Komponenten werden realistische Messcharakteristiken modellieren, einschließlich Rauschen, Drift und Ansprechzeiten.

## Geplante Sensorkategorien

### Physikalische Sensoren

Sensoren für physikalische Eigenschaften:

- **pH-Elektroden**: Mit Kalibrierungsdrift und Temperaturkompensation
- **Temperatursensoren**: PT100, Thermoelemente mit Genauigkeitsspezifikationen
- **Drucksensoren**: Für Gas- und Flüssigkeitsdruck
- **Füllstandssensoren**: Ultraschall, Radar, hydrostatisch
- **Durchflussmesser**: Magnetisch-induktiv, Ultraschall, Coriolis

### Chemische Sensoren

Sensoren für Prozessflüssigkeitsanalyse:

- **VFA-Analysatoren**: Online-Titration, GC-Analyse
- **Ammoniak-Sensoren**: Ionenselektive Elektroden
- **CSB-Analysatoren**: Online-Spektroskopie
- **Nährstoffanalysatoren**: N, P, K-Messung

### Gas-Analysatoren

Sensoren für Biogaszusammensetzung:

- **Methan-Sensoren**: Infrarot, kalorimetrisch
- **CO2-Sensoren**: NDIR-Technologie
- **H2S-Sensoren**: Elektrochemisch
- **Sauerstoff-Sensoren**: Für Leckageerkennung
- **Spurengas-Analysatoren**: Mit Nachweisgrenzen

## Beispiel-Konzept

```python
# Zukünftige Implementierung (konzeptionell)
from pyadm1.components.sensors import PhysicalSensor, ChemicalSensor, GasSensor

# pH-Sensor mit realistischem Rauschen
ph_sensor = PhysicalSensor(
    "ph1",
    sensor_type="pH",
    measurement_noise=0.05,    # ±0.05 pH Einheiten
    drift_rate=0.01,           # Kalibrierungsdrift pro Tag
    response_time=30.0,        # Sekunden
    calibration_interval=7     # Tage zwischen Kalibrierungen
)

# VFA-Analysator mit Messungsverzögerung
vfa_sensor = ChemicalSensor(
    "vfa1",
    sensor_type="VFA",
    measurement_delay=5,       # Minuten Verzögerung
    accuracy=0.1,              # ±10% Genauigkeit
    detection_limit=0.1,       # g/L untere Grenze
    sampling_interval=60       # Minuten zwischen Proben
)

# Methan-Analysator
ch4_sensor = GasSensor(
    "ch4_1",
    sensor_type="CH4",
    measurement_range=(0, 100),  # % CH4
    accuracy=0.5,                # ±0.5%
    drift=0.1,                   # % pro Woche
    cross_sensitivity={'CO2': 0.01}  # Kreuzempfindlichkeit
)
```

## Sensor-Charakteristiken

### Rauschen und Genauigkeit

Realistische Sensoren weisen verschiedene Fehlerquellen auf:

```python
# Gemessener Wert = Wahrer Wert + Systematischer Fehler + Zufälliges Rauschen + Drift

measured_value = (
    true_value * (1 + systematic_error) +
    random.normal(0, noise_level) +
    drift_accumulated
)
```

**Typische Genauigkeiten:**

| Sensor | Messbereich | Genauigkeit | Drift |
|--------|-------------|-------------|-------|
| pH | 0-14 | ±0.05 pH | 0.01 pH/Tag |
| Temperatur (PT100) | -50 bis 200°C | ±0.1°C | <0.01°C/Jahr |
| Druck | 0-2 bar | ±0.5% FS | <0.1% FS/Jahr |
| VFA | 0-20 g/L | ±5% | ±0.1 g/L/Woche |
| CH4 | 0-100% | ±0.5% | ±0.1%/Woche |

### Ansprechzeit

Sensoren haben charakteristische Ansprechzeiten:

```python
# First-order response
sensor_value(t) = true_value * (1 - exp(-t/τ))

# wobei τ = Zeitkonstante (63.2% Response-Zeit)
```

**Typische Ansprechzeiten:**

| Sensor | τ (Zeit bis 63% Response) | t95 (Zeit bis 95% Response) |
|--------|---------------------------|----------------------------|
| pH-Elektrode | 10-30 s | 30-90 s |
| Thermoelement | 0.1-5 s | 0.3-15 s |
| CH4-Sensor (IR) | 5-10 s | 15-30 s |
| VFA-Analysator | 2-5 min | 6-15 min |

### Kalibrierung und Wartung

Sensoren benötigen regelmäßige Kalibrierung:

```python
# Kalibrierungsdrift-Modell
drift(t) = drift_rate * (t - last_calibration)

if (t - last_calibration) > calibration_interval:
    # Kalibrierung erforderlich
    perform_calibration()
    last_calibration = t
    accumulated_drift = 0
```

## Sensor-Platzierung

### Fermenter-Überwachung

**Empfohlene Sensoren für Fermenter:**

```python
# Minimale Instrumentierung
sensors_minimum = {
    'temperature': ['T1'],           # Fermentertemperatur
    'pH': ['pH1'],                   # Prozess-pH
    'gas_flow': ['Q_gas'],           # Gasproduktion
    'gas_composition': ['CH4_CO2']   # Methangehalt
}

# Standard-Instrumentierung
sensors_standard = {
    'temperature': ['T1', 'T2'],     # Fermenter + Umgebung
    'pH': ['pH1'],
    'VFA': ['VFA1'],                 # VFA-Konzentration
    'gas_flow': ['Q_gas'],
    'gas_composition': ['CH4_CO2', 'H2S'],
    'level': ['L1']                  # Füllstand
}

# Erweiterte Instrumentierung
sensors_advanced = {
    'temperature': ['T1', 'T2', 'T3'],  # Multiple Punkte
    'pH': ['pH1', 'pH2'],                # Redundanz
    'VFA': ['VFA1'],
    'ammonia': ['NH3'],                  # Inhibitionsüberwachung
    'gas_flow': ['Q_gas'],
    'gas_composition': ['CH4_CO2', 'H2S', 'O2'],
    'pressure': ['P_gas'],               # Gasspeicherdruck
    'level': ['L1'],
    'conductivity': ['EC1']              # Prozessstabilität
}
```

## Datenverarbeitung

### Filterung und Glättung

Rohe Sensordaten benötigen oft Filterung:

```python
# Exponentielles Glätten
def exponential_smoothing(measurements, alpha=0.3):
    """
    Glätte Sensormessungen

    alpha: Glättungsfaktor (0-1)
           0 = keine Glättung
           1 = nur aktueller Wert
    """
    smoothed = [measurements[0]]
    for m in measurements[1:]:
        s = alpha * m + (1 - alpha) * smoothed[-1]
        smoothed.append(s)
    return smoothed

# Moving Average
def moving_average(measurements, window=5):
    """Gleitender Durchschnitt"""
    return [
        sum(measurements[max(0, i-window):i+1]) /
        min(window, i+1)
        for i in range(len(measurements))
    ]
```

### Ausreißer-Erkennung

```python
def detect_outliers(measurements, threshold=3.0):
    """
    Erkenne Ausreißer mit Z-Score

    threshold: Standardabweichungen für Ausreißer
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

## Prozesssteuerung mit Sensoren

### pH-Regelung

```python
# Zukünftiges Beispiel: pH-Steuerung
def ph_control(ph_measurement, setpoint=7.2, tolerance=0.2):
    """
    Einfache pH-Regelung

    Returns: Kalk-Dosierungsrate [kg/d]
    """
    error = setpoint - ph_measurement

    if error > tolerance:
        # pH zu niedrig, füge Kalk hinzu
        dosing_rate = min(100, error * 50)  # Proportionale Regelung
    elif error < -tolerance:
        # pH zu hoch, reduziere Kalk
        dosing_rate = 0
    else:
        # Innerhalb Toleranz
        dosing_rate = 0

    return dosing_rate
```

### Fütterungssteuerung basierend auf VFA

```python
def adaptive_feeding_control(vfa_measurement, vfa_limit=4.0,
                             current_feed_rate=15.0):
    """
    Passe Fütterungsrate basierend auf VFA an

    vfa_limit: VFA-Grenzwert [g/L]
    current_feed_rate: Aktuelle Rate [m³/d]

    Returns: Angepasste Fütterungsrate [m³/d]
    """
    if vfa_measurement > vfa_limit:
        # Übersäuerungsrisiko - reduziere Fütterung
        reduction_factor = vfa_limit / vfa_measurement
        new_rate = current_feed_rate * reduction_factor
        print(f"VFA hoch ({vfa_measurement:.2f} g/L) - Reduziere auf {new_rate:.1f} m³/d")
    elif vfa_measurement < vfa_limit * 0.5:
        # Stabil - kann Fütterung erhöhen
        increase_factor = 1.05  # 5% Erhöhung
        new_rate = min(current_feed_rate * increase_factor, 20.0)  # Max 20 m³/d
        print(f"VFA stabil ({vfa_measurement:.2f} g/L) - Erhöhe auf {new_rate:.1f} m³/d")
    else:
        # Innerhalb optimalen Bereichs
        new_rate = current_feed_rate

    return new_rate
```

## Alarme und Benachrichtigungen

### Alarm-System

```python
class SensorAlarm:
    """Sensor-Alarmsystem"""

    def __init__(self, sensor_id, alarm_type, threshold, hysteresis=0.1):
        self.sensor_id = sensor_id
        self.alarm_type = alarm_type  # 'high', 'low', 'rate_of_change'
        self.threshold = threshold
        self.hysteresis = hysteresis
        self.is_active = False
        self.last_value = None

    def check(self, current_value):
        """Prüfe Alarmbedingung"""

        if self.alarm_type == 'high':
            if not self.is_active and current_value > self.threshold:
                self.is_active = True
                return f"ALARM: {self.sensor_id} hoch ({current_value:.2f})"
            elif self.is_active and current_value < (self.threshold - self.hysteresis):
                self.is_active = False
                return f"OK: {self.sensor_id} normal ({current_value:.2f})"

        elif self.alarm_type == 'low':
            if not self.is_active and current_value < self.threshold:
                self.is_active = True
                return f"ALARM: {self.sensor_id} niedrig ({current_value:.2f})"
            elif self.is_active and current_value > (self.threshold + self.hysteresis):
                self.is_active = False
                return f"OK: {self.sensor_id} normal ({current_value:.2f})"

        elif self.alarm_type == 'rate_of_change':
            if self.last_value is not None:
                rate = abs(current_value - self.last_value)
                if not self.is_active and rate > self.threshold:
                    self.is_active = True
                    return f"ALARM: {self.sensor_id} schnelle Änderung ({rate:.2f}/h)"
            self.last_value = current_value

        return None

# Beispielverwendung
alarms = {
    'pH_low': SensorAlarm('pH1', 'low', 6.8, hysteresis=0.1),
    'pH_high': SensorAlarm('pH1', 'high', 8.0, hysteresis=0.1),
    'VFA_high': SensorAlarm('VFA1', 'high', 4.0, hysteresis=0.5),
    'temp_deviation': SensorAlarm('T1', 'rate_of_change', 2.0)  # 2°C/h
}

# In Simulationsschleife
for measurement in ph_measurements:
    for alarm in alarms.values():
        message = alarm.check(measurement)
        if message:
            print(message)
```

## Datalogging

### Sensordaten-Protokollierung

```python
import csv
from datetime import datetime

class SensorDataLogger:
    """Protokolliere Sensordaten"""

    def __init__(self, filename):
        self.filename = filename
        self.file = None
        self.writer = None

    def open(self, sensor_ids):
        """Öffne Logdatei"""
        self.file = open(self.filename, 'w', newline='')
        self.writer = csv.writer(self.file)
        # Header
        self.writer.writerow(['timestamp', 'time_days'] + sensor_ids)

    def log(self, time_days, sensor_values):
        """Protokolliere einen Zeitpunkt"""
        timestamp = datetime.now().isoformat()
        row = [timestamp, time_days] + sensor_values
        self.writer.writerow(row)

    def close(self):
        """Schließe Logdatei"""
        if self.file:
            self.file.close()

# Verwendung
logger = SensorDataLogger('sensor_data.csv')
logger.open(['pH', 'T', 'VFA', 'Q_gas', 'CH4'])

# In Simulation
for t in range(simulation_steps):
    # ... Simulation ...
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

## Aktuelle Implementierung

Während dedizierte Sensorkomponenten noch in Entwicklung sind, können Fermenter-Ausgaben bereits zur Prozessüberwachung verwendet werden:

```python
from pyadm1.components.biological import Digester

# Fermenter liefert bereits Prozessindikatoren
digester = Digester("dig1", feedstock, V_liq=2000)
result = digester.step(t, dt, inputs)

# Verfügbare "Sensor"-Werte
monitoring_data = {
    'pH': result['pH'],
    'VFA': result['VFA'],            # g/L
    'TAC': result['TAC'],            # g CaCO3/L
    'Q_gas': result['Q_gas'],        # m³/d
    'Q_ch4': result['Q_ch4'],        # m³/d
    'Q_co2': result['Q_co2']         # m³/d
}

# Einfache Prozessüberwachung
if monitoring_data['pH'] < 6.8:
    print("Warnung: Niedriger pH")

if monitoring_data['VFA'] / monitoring_data['TAC'] > 0.4:
    print("Warnung: Hohes VFA/TAC-Verhältnis")
```

## Zukünftige Entwicklung

Das Sensormodul wird erweitert um:

1. **Realistische Sensor-Modelle**
   - Rauschen und Drift
   - Kalibrierungszyklen
   - Ausfallmodelle

2. **Erweiterte Prozesssteuerung**
   - PID-Regler
   - Modellprädiktive Regelung (MPC)
   - Adaptive Steuerung

3. **Datenanalyse-Tools**
   - Trendanalyse
   - Anomalie-Erkennung
   - Prädiktive Wartung

4. **Visualisierung**
   - Echtzeit-Dashboards
   - Historische Trends
   - Alarm-Übersichten

## Beitragen

Interessiert an der Entwicklung des Sensormoduls? Beiträge sind willkommen! Siehe [Contributing Guide](../../../CONTRIBUTING.md) für Details.

## Nächste Schritte

- [Biologische Komponenten](biological.md): Fermenter und Prozesssteuerung
- [Energiekomponenten](energy.md): BHKW und Wärmesysteme
- [Mechanische Komponenten](mechanical.md): Pumpen und Rührwerke
- [Fütterungskomponenten](feeding.md): Lagerung und Dosierung
- [API-Referenz](../../api_reference/components/sensors.md): Detaillierte Klassendokumentation (wenn verfügbar)
