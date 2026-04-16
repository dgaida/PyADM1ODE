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
    V_gas=300.0,                      # Gas volume [m³]
    T_ad=308.15,                      # Operating temperature [K]
    name="Main Digester",             # Readable name
    load_initial_state=True,          # Load steady-state initialization
    Q_substrates=[15, 10, 0, ...]    # Substrate feed rates [m³/d]
)
```

### Sizing Guidelines

| Plant Size | V_liq [m³] | V_gas [m³] | Feed Rate [m³/d] | HRT [Days] |
|------------|------------|------------|------------------|------------|
| Small      | 300-800    | 50-120     | 10-25            | 20-40      |
| Medium     | 1000-3000  | 150-450    | 25-75            | 25-45      |
| Large      | 3000-8000  | 450-1200   | 75-200           | 30-50      |

### Temperature Options

- **Psychrophilic**: 298.15 K (25°C)  
- **Mesophilic**: 308.15 K (35°C) - Most common  
- **Thermophilic**: 328.15 K (55°C) - For fiber-rich substrates  

### Outputs

The `step()` method returns a dictionary containing:  
- `Q_gas`, `Q_ch4`, `Q_co2`: Gas production rates [m³/d]  
- `pH`: pH value  
- `VFA`: Volatile fatty acids [g/L]  
- `TAC`: Total alkalinity [g CaCO3/L]  

## Hydrolysis

Pre-treatment tank for hydrolysis-dominated processes (Stub for future implementation).

## Separator

Solid-liquid separation for digestate processing (Stub for future implementation).

## Process Monitoring

Monitoring key indicators is crucial for stable operation:  
- **pH value**: Should be between 6.8 and 7.5.  
- **VFA/TAC ratio**: Should be below 0.4.  
- **Methane content**: Typically > 55%.  

## Troubleshooting

### Low pH Value  
- **Cause**: High OLR or insufficient buffering.  
- **Solution**: Reduce feed rate or add lime.  

## Best Practices  
- Start with moderate loading rates.  
- Monitor VFA levels regularly.  
