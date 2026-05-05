# Pre-configured Substrates

PyADM1ODE includes 10 agricultural substrates with literature-validated parameters.

## Available Substrates

| Substrate | Type | Typical Use | Biogas Potential |
|-----------|------|-------------|------------------|
| **Corn silage** | Energy crop | Main feedstock | High (600-700 L/kg VS) |
| **Liquid manure** | Animal waste | Co-substrate | Medium (200-400 L/kg VS) |
| **Green rye** | Energy crop | Early harvest | Medium-High |
| **Grass silage** | Grassland | Renewable | Medium (400-550 L/kg VS) |
| **Wheat** | Cereal | Energy crop | High |
| **GPS** | Grain silage | Whole-crop | High |
| **CCM** | Corn-cob-mix | Energy crop | High |
| **Feed lime** | Additive | pH buffer | N/A |
| **Cow manure** | Animal waste | Co-substrate | Medium (200-350 L/kg VS) |
| **Onions** | Waste | Vegetable waste | Medium-High |

## Substrate Characterization

All substrates are characterized with:  
- Dry matter (DM) and volatile solids (VS) content  
- ADM1 fractionation (carbohydrates, proteins, lipids)  
- Biochemical methane potential (BMP)  
- pH and alkalinity  

For more details on how these are mapped to ADM1, see the [ADM1 Implementation](adm1_implementation.md) page.

## Substrate Management

Substrates are characterized by:
- **Weender analysis**: Crude fiber (RF), crude protein (RP), crude fat (RL).
- **Van Soest fractions**: NDF, ADF, ADL.
- **Physical properties**: pH, TS, VS, COD.
- **Kinetic parameters**: Disintegration and hydrolysis rates.

These parameters allow for a dynamic calculation of ADM1 inputs (like $X_c$ and stoichiometry) and kinetic parameters ($k_{dis}$, $k_{hyd}$) based on the substrate characteristics.
