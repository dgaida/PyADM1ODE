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

### pH Calculation

In the original ADM1 publication, the pH value is often solved via an algebraic charge balance, requiring an iterative determination of the hydrogen ion concentration $[H^+]$.

In this implementation, the pH value is calculated directly from the charge balance of the **dynamic ion states**. Since cations ($S_{cat}$), anions ($S_{an}$), and the ionized forms of organic acids and inorganic carbon/nitrogen are maintained as separate state variables within the ODE system, the pH value can be explicitly determined at each step. This approach is more robust, especially with high solids content and varying buffer capacities common in agricultural plants.

## Enhancements for Agricultural Substrates

The implementation includes important enhancements specifically optimized for the digestion of energy crops and manure (based on **Koch et al., 2010**):

### Influence of Solids (TS) Content on Hydrolysis

In agricultural biogas plants with high total solids (TS) content, hydrolysis is often the rate-limiting step. This implementation accounts for this using a correction function:
$$ hydro\_factor = \frac{1}{1 + (\frac{TS}{K_{hyd}})^{n_{hyd}}} $$
This factor reduces the hydrolysis rates for carbohydrates, proteins, and lipids as the solids content in the digester increases, leading to a more realistic prediction of ammonium release and gas production.

### Modeling Decay Products ($X_p$)

Similar to the ASM1 (Activated Sludge Model), a separate state for particulate decay products ($X_p$) was introduced. This allows for a more precise closing of the nitrogen balance and describes the accumulation of inert organic matter from decayed biomass more accurately.

## Characterization via Weender Analysis

Substrates are not entered directly as ADM1 components but are defined via common laboratory parameters:
*   **Extended Weender Analysis**: Crude fiber (RF), crude protein (RP), crude fat (RL).
*   **Van Soest Fractions**: NDF, ADF, ADL (lignin).
*   **Physical Values**: Total solids (TS), volatile solids (VS), pH value.

### Mapping to ADM1 Input Variables

The conversion of substrate fractions into the ADM1 influent stream is performed dynamically:
1.  **Composite ($X_c$) Composition**: Based on protein, fat, and fiber content, the stoichiometric coefficients $f_{ch,xc}$, $f_{pr,xc}$, $f_{li,xc}$, $f_{xI,xc}$, and $f_{sI,xc}$ are calculated individually for each substrate.
2.  **Kinetic Parameters**: Substrates provide their own rates for disintegration ($k_{dis}$) and hydrolysis ($k_{hyd}$). For substrate mixtures, these parameters are calculated weighted by volumetric flow rate.

### Mathematical Foundation

The implementation is based on the research of:
*   **Gaida, D. (2014)**: *Dynamic real-time substrate feed optimization of anaerobic co-digestion plants*. PhD thesis, Leiden University.
*   **Koch, K. et al. (2010)**: *Biogas from grass silage – Measurements and modeling with ADM1*. Bioresource Technology.

## Technical Implementation

Substrate parameter calculations and the mixed ADM1 influent stream are handled by highly optimized C# DLLs (located in the `pyadm1/dlls/` folder), integrated into the Python environment via `pythonnet`. This enables fast calculation even for complex substrate mixtures and large-scale simulation studies.
