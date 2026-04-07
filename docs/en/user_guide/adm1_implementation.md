# ADM1 Implementation and Substrate Modeling

This page describes the technical details of the Anaerobic Digestion Model No. 1 (ADM1) implementation used in PyADM1ODE and how agricultural substrates are integrated into the model.

## ADM1 as a Pure ODE System

Unlike the standard ADM1 (IWA Task Group, 2002), which is often formulated as a system of differential-algebraic equations (DAE), this implementation is a **pure system of ordinary differential equations (ODE)**.

### Key Differences from the Standard Model

1.  **No Algebraic States**: Acid-base equilibria and gas-liquid transfer are modeled kinetically. This avoids the need for iterative algebraic solvers within each timestep of the ODE solver, enhancing numerical stability.
2.  **37 State Variables**: The model tracks a total of 37 variables to represent the entire process:
    *   **Soluble Components (12)**: Monosaccharides, amino acids, long-chain fatty acids (LCFA), valerate, butyrate, propionate, acetate, hydrogen, methane, inorganic carbon ($S_{CO2}$), inorganic nitrogen ($S_{NH4}$), soluble inerts.
    *   **Particulate Components (13)**: Composites ($X_{xc}$), carbohydrates, proteins, lipids, 7 bacterial populations, particulate inerts, particulate products ($X_p$).
    *   **Acid-Base Variables (8)**: Cations, anions, and the ionized forms of organic acids and inorganic species.
    *   **Gas Phase (4)**: Partial pressures of $H_2$, $CH_4$, $CO_2$, and total pressure.

## Modeling Agricultural Substrates

A key feature of this repository is the detailed mapping of agricultural substrates (e.g., maize silage, manure) to ADM1 input variables.

### Characterization via Weender Analysis

Substrates are not entered directly as ADM1 components but are defined via common laboratory parameters:
*   **Extended Weender Analysis**: Crude fiber (RF), crude protein (RP), crude fat (RL).
*   **Van Soest Fractions**: NDF, ADF, ADL (lignin).
*   **Physical Values**: Total solids (TS), volatile solids (VS), pH value.

### Mapping to ADM1 Input Variables

The conversion of substrate fractions into the ADM1 influent stream is performed dynamically:
1.  **Composite ($X_c$) Composition**: Based on protein, fat, and fiber content, the stoichiometric coefficients $f_{ch,xc}$, $f_{pr,xc}$, $f_{li,xc}$, $f_{xI,xc}$, and $f_{sI,xc}$ are calculated individually for each substrate.
2.  **Kinetic Parameters**: Substrates provide their own rates for disintegration ($k_{dis}$) and hydrolysis ($k_{hyd}$). For substrate mixtures, these parameters are calculated weighted by volumetric flow rate.
3.  **VFA Content**: Organic acids already present in the substrate (e.g., in silages) are directly assigned to the corresponding soluble ADM1 components.

### Mathematical Foundation

The implementation is based on the PhD thesis of **Daniel Gaida (2014)**: *Dynamic real-time substrate feed optimization of anaerobic co-digestion plants*. It combines the biochemical structure of ADM1 with a robust model for substrate influent characterization, specifically optimized for agricultural applications.

## Technical Implementation

Substrate parameter calculations and the mixed ADM1 influent stream are handled by highly optimized C# DLLs (located in the `pyadm1/dlls/` folder), integrated into the Python environment via `pythonnet`. This enables fast calculation even for complex substrate mixtures and large-scale simulation studies.

### Example: Substrate Impact on Kinetics

When you mix different substrates, the system automatically calculates the resulting kinetic rates:

```python
# The ADM1 class internally determines weighted parameters
substrate_params = adm1._get_substrate_dependent_params()
# This includes k_dis, k_hyd_ch, k_hyd_pr, k_hyd_li based on the current feed mix
```
