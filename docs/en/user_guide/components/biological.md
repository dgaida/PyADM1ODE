# Biological Components

Components for biological conversion processes in biogas plants.

## Digester

The main fermenter that implements the ADM1 model for anaerobic digestion.

### Parameters

```python
from pyadm1.configurator.plant_configurator import PlantConfigurator

configurator.add_digester(
    digester_id="main_digester",      # Unique identifier
    V_liq=2000.0,                     # Liquid volume [m³]
    V_gas=300.0,                      # Gas headspace [m³]
    T_ad=308.15,                      # Operating temperature [K]
    name="Main Digester",             # Readable name
    load_initial_state=True,          # Load steady-state initialization
    initial_state_file=None,          # Custom initial-state CSV (optional)
    Q_substrates=[15, 10, 0, ...]    # Substrate feed rates [m³/d]
)
```

### Sizing Guidelines

| Plant size | V_liq [m³] | V_gas [m³] | Feed rate [m³/d] | HRT [days] |
|------------|------------|------------|------------------|------------|
| Small      | 300–800    | 50–120     | 10–25            | 20–40      |
| Medium     | 1000–3000  | 150–450    | 25–75            | 25–45      |
| Large      | 3000–8000  | 450–1200   | 75–200           | 30–50      |

### Temperature Options

```python
# Psychrophilic (rare in practice)
T_psychro = 298.15  # 25 °C

# Mesophilic (most common)
T_meso = 308.15     # 35 °C

# Thermophilic (fiber-rich substrates)
T_thermo = 328.15   # 55 °C
```

### Outputs

```python
outputs = digester.step(t, dt, inputs)
# Returns:
{
    'Q_out': 25.0,              # Outflow [m³/d]
    'state_out': [...],         # ADM1 state for next digester
    'Q_gas': 1250.5,           # Biogas production [m³/d]
    'Q_ch4': 750.3,            # Methane production [m³/d]
    'Q_co2': 475.2,            # CO2 production [m³/d]
    'pH': 7.32,                # pH value
    'VFA': 2.45,               # Volatile fatty acids [g/L]
    'TAC': 8.50,               # Total alkalinity [g CaCO3/L]
    'gas_storage': {           # Attached gas storage info
        'stored_volume_m3': 150.0,
        'pressure_bar': 1.02,
        'vented_volume_m3': 0.0
    }
}
```

### Advanced Usage

**Multiple digesters in series:**

```python
# Hydrolysis + methanogenesis
configurator.add_digester("hydro", V_liq=500, T_ad=318.15,
                         Q_substrates=[15, 10, 0, 0, 0, 0, 0, 0, 0, 0])
configurator.add_digester("main", V_liq=2000, T_ad=308.15,
                         Q_substrates=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
configurator.connect("hydro", "main", "liquid")
```

**Custom initial state:**

```python
import pandas as pd

# Create custom state
initial = pd.DataFrame({
    'S_su': [0.01], 'S_aa': [0.001], # ... all 41 state variables
})
initial.to_csv('custom_state.csv', index=False)

# Use in digester
configurator.add_digester(
    "dig1", V_liq=2000,
    initial_state_file='custom_state.csv'
)
```

### Calibration Parameters

The digester supports applying calibration parameters for model fitting:

```python
from pyadm1.components.biological import Digester

digester = Digester("dig1", feedstock, V_liq=2000)

# Apply calibration parameters
digester.apply_calibration_parameters({
    'k_dis': 0.55,      # Disintegration rate
    'Y_su': 0.105,      # Yield coefficient for sugars
    'k_hyd_ch': 11.0    # Hydrolysis rate for carbohydrates
})

# Retrieve current parameters
params = digester.get_calibration_parameters()
print(params)

# Clear parameters (back to defaults)
digester.clear_calibration_parameters()
```

## Separator

Solid–liquid separation for digestate processing (stub for future implementation).

```python
from pyadm1.components.biological import Separator

separator = Separator(
    component_id="sep1",
    separation_efficiency=0.95  # 95% solids separation
)
```

**Use case:** Models mechanical (screw press, centrifuge) or gravity separation with configurable separation efficiency.

## Example: Two-Stage Digestion System

```python
from pyadm1.configurator import BiogasPlant, PlantConfigurator
from pyadm1.substrates import Feedstock

# Setup
feedstock = Feedstock(feeding_freq=48)
plant = BiogasPlant("Two-Stage Plant")
config = PlantConfigurator(plant, feedstock)

# Stage 1: Thermophilic hydrolysis
hydro, hydro_storage = config.add_digester(
    "hydrolysis",
    V_liq=500,
    V_gas=75,
    T_ad=328.15,  # 55 °C
    Q_substrates=[15, 10, 0, 0, 0, 0, 0, 0, 0, 0]
)

# Stage 2: Mesophilic methanogenesis
main, main_storage = config.add_digester(
    "methanogenesis",
    V_liq=2000,
    V_gas=300,
    T_ad=308.15,  # 35 °C
    Q_substrates=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Input only from stage 1
)

# Connect digesters
config.connect("hydrolysis", "methanogenesis", "liquid")

# Energy system
config.add_chp("chp1", P_el_nom=500)
config.add_heating("heat_hydro", target_temperature=328.15, heat_loss_coefficient=0.3)
config.add_heating("heat_main", target_temperature=308.15, heat_loss_coefficient=0.5)

# Auto-connections
config.auto_connect_digester_to_chp("hydrolysis", "chp1")
config.auto_connect_digester_to_chp("methanogenesis", "chp1")
config.auto_connect_chp_to_heating("chp1", "heat_hydro")
config.auto_connect_chp_to_heating("chp1", "heat_main")

# Simulate
plant.initialize()
results = plant.simulate(duration=100, dt=1/24, save_interval=1.0)

# Analyze results
final = results[-1]
print(f"Hydrolysis biogas: {final['components']['hydrolysis']['Q_gas']:.1f} m³/d")
print(f"Main biogas: {final['components']['methanogenesis']['Q_gas']:.1f} m³/d")
print(f"Total methane: {final['components']['hydrolysis']['Q_ch4'] + final['components']['methanogenesis']['Q_ch4']:.1f} m³/d")
print(f"pH hydrolysis: {final['components']['hydrolysis']['pH']:.2f}")
print(f"pH main: {final['components']['methanogenesis']['pH']:.2f}")
```

## Process Monitoring

### Key Process Indicators

```python
def monitor_digester_health(results):
    """Monitor digester health via process indicators."""

    for result in results:
        digester_data = result['components']['main_digester']

        # Check pH
        pH = digester_data['pH']
        if pH < 6.8:
            print(f"Warning: Low pH ({pH:.2f}) - acidification risk")
        elif pH > 8.0:
            print(f"Warning: High pH ({pH:.2f}) - possible ammonia inhibition")

        # VFA/TAC ratio
        VFA = digester_data['VFA']  # g HAc-eq/L
        TAC = digester_data['TAC']  # g CaCO3-eq/L

        VFA_TAC = VFA / TAC if TAC > 0 else 0

        if VFA_TAC > 0.4:
            print(f"Warning: High VFA/TAC ratio ({VFA_TAC:.2f}) - process instability")

        # Gas production
        Q_gas = digester_data['Q_gas']
        if Q_gas < 500:  # example threshold
            print(f"Warning: Low gas production ({Q_gas:.1f} m³/d)")

monitor_digester_health(results)
```

### Optimal Operating Ranges

| Parameter | Optimal | Acceptable | Critical |
|-----------|---------|------------|----------|
| pH | 7.0–7.5 | 6.8–8.0 | <6.8 or >8.0 |
| VFA [g/L] | 0.5–2.0 | 2.0–4.0 | >4.0 |
| VFA/TAC | 0.2–0.3 | 0.3–0.4 | >0.4 |
| TAC [g CaCO3/L] | 5.0–10.0 | 4.0–12.0 | <4.0 |
| Temp. mesophilic [°C] | 35–38 | 32–40 | <30 or >42 |
| Temp. thermophilic [°C] | 52–55 | 48–58 | <45 or >60 |

## Troubleshooting

### Low pH Value

**Causes:**  
- Organic loading rate (OLR) too high  
- Insufficient buffer capacity  
- Sudden substrate change  

**Solutions:**

```python
# Reduce organic loading
Q = [10, 8, 0, 0, 0, 0, 0, 0, 0, 0]  # reduced from [15, 10, ...]

# Or add lime buffer
Q = [15, 10, 0, 0, 0, 0, 0, 1, 0, 0]  # 1 m³/d lime
```

### Foaming

**Causes:**  
- High protein content in substrate  
- Sudden pH changes  
- High VFA concentrations  

**Solutions:**  
- Reduce protein-rich substrates  
- Stabilize pH via buffering  
- Implement anti-foam measures  

### Low Gas Production

**Causes:**  
- Low organic loading  
- Poor substrate quality  
- Inhibition (NH3, H2S, heavy metals)  
- Hydraulic retention time too short  

**Diagnosis:**

```python
def diagnose_low_gas_production(digester_outputs):
    """Diagnose causes of low gas production."""

    Q_gas = digester_outputs['Q_gas']
    Q_in = sum(Q_substrates)  # total input

    # Specific gas yield
    specific_gas = Q_gas / Q_in if Q_in > 0 else 0

    if specific_gas < 0.5:  # m³ biogas / m³ input
        print("Low specific gas yield - possible causes:")
        print("- Poor substrate quality")
        print("- Inhibition")
        print("- Process instability")

    # Check methane content
    CH4_content = digester_outputs['Q_ch4'] / Q_gas if Q_gas > 0 else 0

    if CH4_content < 0.55:
        print(f"Low methane content ({CH4_content:.1%}) - possible air ingress or CO2 stripping")

diagnose_low_gas_production(digester.outputs_data)
```

## Best Practices

1. **Start with realistic operating parameters**  
   - Use typical HRT values (30–40 days)  
   - Start with moderate OLR (2–4 kg VS/m³/d)  

2. **Monitor critical parameters**  
   - pH should be stable (±0.2)  
   - VFA/TAC ratio < 0.4  
   - Methane content > 55%  

3. **Implement buffering systems**  
   - Add lime or other buffers at low pH  
   - Maintain TAC > 4 g CaCO3/L  

4. **Use two-stage systems for difficult substrates**  
   - Thermophilic hydrolysis for fiber-rich substrates  
   - Mesophilic methanogenesis for stable gas production  

5. **Calibrate the model with real data**  
   - Use calibration parameters for more accurate predictions  
   - Validate against operational data  

## Next Steps

- [Energy Components](energy.md): CHP and heating systems  
- [Mechanical Components](mechanical.md): Pumps and mixers  
- [Feeding Components](feeding.md): Storage and dosing  
- [API Reference](../../api/biological.md): Detailed class documentation  
