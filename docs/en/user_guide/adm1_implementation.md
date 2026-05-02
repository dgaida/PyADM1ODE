# ADM1 Implementation and Substrate Modeling

This page describes the technical details of the ADM1 model implemented in PyADM1ODE and how agricultural substrates are integrated into the model.

PyADM1ODE implements **ADM1da** (Schlattmann 2011) — an agricultural adaptation of the original ADM1 (Batstone et al. 2002, IWA Task Group). Compared to the classical formulation, this model adds **sub-fractioned disintegration**, **temperature-dependent kinetics**, and **modified inhibition kinetics** for agricultural co-digestion.

## ADM1 as a Pure ODE System

Unlike the standard ADM1, which is often formulated as a system of differential-algebraic equations (DAE), this implementation is a **pure system of ordinary differential equations (ODE)**.

### Key Features

1. **Kinetic acid-base equilibrium**: The ionised species (acetate, propionate, butyrate, valerate ions, $\text{HCO}_3^-$, $\text{NH}_3$) are carried as dynamic state variables with a very high reaction rate $k_{A,B} = 10^8\,\text{m}^3\,\text{kmol}^{-1}\,\text{d}^{-1}$. They therefore track the thermodynamic equilibrium essentially in real time, without an algebraic solver having to be invoked inside the ODE step.  
2. **Kinetic gas-liquid transfer**: $\text{H}_2$, $\text{CH}_4$, and $\text{CO}_2$ are transferred to the gas phase via $k_L a$ rates — no algebraic Henry equilibrium constraint.  
3. **41 state variables**: The model carries 41 variables in five blocks:  

   | Block                          | Indices | Count | Contents                                                                                                     |
   |--------------------------------|---------|-------|--------------------------------------------------------------------------------------------------------------|
   | Soluble components             | 0–11    | 12    | $S_{su},S_{aa},S_{fa},S_{va},S_{bu},S_{pro},S_{ac},S_{h2},S_{ch4},S_{IC},S_{IN},S_{I}$                       |
   | Particulate sub-fractions      | 12–21   | 10    | $X_{PS\_ch/pr/li}$ (slow), $X_{PF\_ch/pr/li}$ (fast), $X_{S\_ch/pr/li}$ (hydrolysable), $X_I$                |
   | Biomass                        | 22–28   | 7     | $X_{su},X_{aa},X_{fa},X_{c4},X_{pro},X_{ac},X_{h2}$                                                          |
   | Acid-base / charge balance     | 29–36   | 8     | $S_{cat},S_{an}$, ionised forms $S_{va^-},S_{bu^-},S_{pro^-},S_{ac^-},S_{HCO_3^-},S_{NH_3}$                  |
   | Gas phase                      | 37–40   | 4     | $p_{H_2},p_{CH_4},p_{CO_2},p_{tot}$                                                                          |

### pH calculation

The pH is determined at every evaluation step from the **charge balance** using a Newton-Raphson iteration:

$$
S_{cat} - S_{an} + (S_{NH4} - S_{NH3}) - S_{HCO_3^-} - \frac{S_{ac^-}}{64} - \frac{S_{pro^-}}{112} - \frac{S_{bu^-}}{160} - \frac{S_{va^-}}{208} + S_{H^+} - \frac{K_w}{S_{H^+}} = 0
$$

Since every contribution on the left-hand side is a state variable, this charge balance is a **purely point-wise function of the current state** within the ODE step and converges to $[H^+]$ in 5–10 Newton iterations with high accuracy. The fast equilibration of the ionised species via $k_{A,B}$ ensures the computed pH stays consistent with the thermodynamic acid-base constants.

## Sub-fractioned disintegration

The most important difference compared to classical ADM1 is the **two-pool disintegration** instead of the single composite variable $X_c$:

```
                     substrate inflow
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
   X_PS_ch/pr/li      X_PF_ch/pr/li      X_I (inert)
   (slow pool)        (fast pool)
        │                 │
        │ k_dis_PS=0.04   │ k_dis_PF=0.4
        │ d⁻¹             │ d⁻¹
        ▼                 ▼
             X_S_ch/pr/li (hydrolysable)
                   │
                   │ k_hyd ≈ 4 d⁻¹
                   ▼
               S_su / S_aa / S_fa
```

**Routing of the inflow COD** to the pools (via `Feedstock._calc_concentrations`):

| Pool       | Substrate source (Weender + sub-fractioning parameters)                                                                |
|------------|------------------------------------------------------------------------------------------------------------------------|
| $X_{PS}$   | Crude fibre (always slow pool) **plus** the share $f_{sOTS}$ of NFE carbohydrates, proteins, and lipids                |
| $X_{PF}$   | Share $f_{fOTS}$ of NFE carbohydrates, proteins, and lipids                                                            |
| $X_S$      | Produced by disintegration from $X_{PS}$ and $X_{PF}$; never sourced directly from the substrate                       |
| $X_I$      | Share $a_{XI}$ of the total raw organic COD                                                                            |
| $S_I$      | Share $a_{Si}$ of the total raw organic COD (soluble inerts, appear directly in the inflow)                            |

This makes it possible, for example, to route easily degradable starch (NFE) into the fast pool while lignocellulose (crude fibre) automatically lands in the slow pool — without changing the model structure.

## Temperature-dependent kinetics

The ADM1da variant corrects every kinetic rate against the 35 °C reference using an Arrhenius-θ function:

$$
k(T) = k(35\,°\text{C}) \cdot \theta^{(T[°\text{C}] - 35)}
$$

with group-specific exponents (values after Schlattmann 2011):

| Process group                                          | $\theta_{\exp}$ |
|--------------------------------------------------------|-----------------|
| Disintegration & hydrolysis                            | 0.024           |
| $X_{su}, X_{aa}, X_{h2}$ (growth & decay)              | 0.069           |
| $X_{fa}, X_{c4}, X_{pro}, X_{ac}$ (growth & decay)     | 0.055           |

The correction is applied once when the `ADM1` object is created — to all relevant rates via `ADMParams.apply_temperature_corrections` — and cached in the internal `_kinetic` dict.

## Modified inhibition kinetics

Compared to the standard ADM1, the ADM1da model includes the following adjustments — all implemented in `ADM1.ADM_ODE`:

| Inhibition                          | Standard ADM1                  | ADM1da                                                                  |
|-------------------------------------|--------------------------------|-------------------------------------------------------------------------|
| pH inhibition $X_{fa}/X_{c4}/X_{pro}$ | Hill, $n=1$                   | Hill, $n=2$ (sharper cut-off)                                           |
| pH inhibition $X_{ac}$              | Hill, $n=1$                    | Hill, $n=3$                                                             |
| pH inhibition $X_{h2}$              | Hill, $n=1$                    | Hill, $n=3$                                                             |
| N limitation                        | $S_{NH4}$ alone                | $S_{IN} = S_{NH4} + S_{NH3}$                                            |
| $\text{NH}_3$ inhibition $X_{ac}$   | linear in $S_{NH3}$            | squared Hill: $K_I^2/(K_I^2 + S_{NH3}^2)$, T-corrected                  |
| $\text{NH}_3$ inhibition $X_{pro}$  | not present                    | same form with its own $K_{I,nh3,pro}$                                  |
| Undissociated acids                 | not present                    | $K_{IH,pro}$ (propionic acid $\to X_{pro}$), $K_{IH,ac}$ (acetic acid $\to X_{ac}$) |
| $\text{CO}_2$ limitation $X_{h2}$   | not present                    | squared Hill in $S_{CO2}$                                               |

These extensions reproduce the behaviour typically observed in agricultural plants: a sharper pH drop on acid accumulation, more pronounced $\text{NH}_3$ inhibition under thermophilic manure-based operation, and more realistic propionate dynamics.

## Substrate characterization via Weender analysis

Substrates are still defined via common laboratory parameters, but now stored as **XML files** under `data/substrates/adm1da/`. An example structure:

```xml
<substrate name="Maize silage (milk ripeness)">
  <param name="TS"    value="320.0"/>
  <param name="NH4"   value="0.0"/>
  <param name="pH"    value="3.9"/>
  <param name="fRF"   value="0.220"/>   <!-- crude fibre -->
  <param name="fRP"   value="0.080"/>   <!-- crude protein -->
  <param name="fRFe"  value="0.030"/>   <!-- crude fat -->
  <param name="fRA"   value="0.045"/>   <!-- ash -->
  <param name="aXI"   value="0.10"/>    <!-- particulate inert COD share -->
  <param name="aSi"   value="0.02"/>    <!-- soluble inert COD share -->
  <param name="fOTSrf" value="0.40"/>   <!-- degradable share of crude fibre -->
  <param name="fsOTS"  value="0.30"/>   <!-- NFE/PR/LI into the slow pool -->
  <param name="ffOTS"  value="0.70"/>   <!-- NFE/PR/LI into the fast pool -->
  <param name="FFS"    value="0.0"/>    <!-- VFAs as acetic-acid equivalent -->
  <param name="KS43"   value="0.0"/>    <!-- acid capacity to pH 4.3 -->
  <!-- ... -->
</substrate>
```

`load_substrate_xml()` returns a `SubstrateParams` dataclass; the `Feedstock` class derives from it the full 38-column ADM1 inflow stream (37 liquid state columns + Q):

1. **Fresh-matter density** $\rho_{FM}$ from the component densities (specific-volume mixing rule).  
2. **Organic COD concentrations** $X_{ch}, X_{pr}, X_{li}$ in $\text{kg COD/m}^3$ via the COD conversion factors $M_{Xch}, M_{Xpr}, M_{Xli}$.  
3. **Routing into the sub-fractions** $X_{PS}/X_{PF}$ via $f_{sOTS}, f_{fOTS}$, and $f_{OTSrf}$ (see table above).  
4. **Dissociation** at the substrate pH: ionised VFAs, $\text{HCO}_3^-$, $\text{NH}_3$.  
5. **Charge balance** closes to $S_{anion}$ (or $S_{cation}$ if negative).  

### Volumetric ADM1da convention for Q

Substrate volumetric flows $Q_i$ are interpreted as **mass-equivalent flows** by default. Internally,

$$
Q_{actual,i} = Q_{input,i} \cdot \frac{1000}{\rho_{FM,i}}
$$

is computed. For liquid substrates ($\rho_{FM} \approx 1000$) this is a no-op; for maize silage ($\rho_{FM} \approx 1134\,\text{kg/m}^3$) the actual volumetric flow shrinks by roughly 12 %. This reproduces the behaviour of ADM1da reference results. Setting `simba_q_convention=False` disables the scaling — useful when $Q$ should be interpreted directly as the actual reactor volume.

### Co-digestion: weighted blending

For multi-substrate inflows, concentrations are blended by volumetric flow (`Feedstock._blended_concentrations`). The number of simultaneously fed substrates is unbounded — `Feedstock` accepts an arbitrarily long list of XML IDs.

## Mathematical foundation

The implementation builds on:

- **Schlattmann, M. (2011)**: ADM1da — described in *SIMBA# biogas Tutorial 4.2*, ifak e.V. Magdeburg.  
- **Batstone, D. J. et al. (2002)**: *Anaerobic Digestion Model No. 1 (ADM1)*. IWA Scientific and Technical Report No. 13.  
- **Gaida, D. (2014)**: *Dynamic real-time substrate feed optimization of anaerobic co-digestion plants*. PhD thesis, Leiden University. (Template for the volumetric blending logic.)  
- **Koch, K. et al. (2010)**: *Biogas from grass silage – measurements and modeling with ADM1*. Bioresource Technology. (Calibration values for high-strength energy crops.)  

## Technical implementation

The whole model is pure Python:

| Module                                          | Purpose                                                       |
|-------------------------------------------------|---------------------------------------------------------------|
| `pyadm1.core.adm1`                              | `ADM1` class with `ADM_ODE`, Newton-Raphson pH, gas output    |
| `pyadm1.core.adm_params`                        | Stoichiometry, kinetics, inhibition, $\theta$ corrections     |
| `pyadm1.core.solver`                            | Wrapper around `scipy.integrate.solve_ivp` (BDF, adaptive)    |
| `pyadm1.substrates.feedstock`                   | XML parser, sub-fractioning routing, blending                 |
| `pyadm1.components.biological.digester`         | Component wrapper incl. gas storage and HRT logic             |

The simulation runs in any standard Python environment and works equally well in containers, web notebooks (Colab), and CI pipelines.
