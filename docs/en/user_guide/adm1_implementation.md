# ADM1 Implementation and Substrate Modeling

This page describes the technical details of the ADM1 model implemented in
PyADM1ODE and how agricultural substrates are integrated into the model.

PyADM1ODE implements **ADM1da** (Schlattmann 2011) — an agricultural adaptation
of the original ADM1 (Batstone et al. 2002, IWA Task Group). Compared to the
classical formulation, this model adds **sub-fractioned disintegration**,
**temperature-dependent kinetics**, and **modified inhibition kinetics** for
agricultural co-digestion. The implementation reproduces the reference behaviour
of the SIMBA# biogas reactor module quantitatively (see the Validation page).
Where the published ADM1da literature is ambiguous on a specific stoichiometric
or kinetic choice, the convention followed by PyADM1ODE is stated explicitly
throughout this page.

## ADM1 as a Pure ODE System

Unlike the standard ADM1, which is often formulated as a system of differential-
algebraic equations (DAE), this implementation is a **pure system of ordinary
differential equations (ODE)**.

### Key Features

1. **Kinetic acid-base equilibrium**: the ionised species (acetate, propionate,
   butyrate, valerate ions, $\text{HCO}_3^-$, $\text{NH}_3$) are carried as
   dynamic state variables with a very high reaction rate
   $k_{A,B} = 10^8\,\text{m}^3\,\text{kmol}^{-1}\,\text{d}^{-1}$. They therefore
   track the thermodynamic equilibrium essentially in real time, without an
   algebraic solver having to be invoked inside the ODE step.
2. **Kinetic gas-liquid transfer**: $\text{H}_2$, $\text{CH}_4$, $\text{CO}_2$
   and $\text{NH}_3$ are transferred to the gas phase via $k_L a$ rates — no
   algebraic Henry equilibrium constraint.
3. **41 state variables**: the model carries 41 variables in five blocks:

   | Block | Indices | Count | Contents |
   | --- | --- | --- | --- |
   | Soluble components | 0–11 | 12 | $S_{su}, S_{aa}, S_{fa}, S_{va}, S_{bu}, S_{pro}, S_{ac}, S_{h2}, S_{ch4}, S_{IC}, S_{IN}, S_{I}$ |
   | Particulate sub-fractions | 12–21 | 10 | $X_{PS\_ch/pr/li}$ (slow), $X_{PF\_ch/pr/li}$ (fast), $X_{S\_ch/pr/li}$ (hydrolysable), $X_I$ |
   | Biomass | 22–28 | 7 | $X_{su}, X_{aa}, X_{fa}, X_{c4}, X_{pro}, X_{ac}, X_{h2}$ |
   | Acid-base / charge balance | 29–36 | 8 | $S_{cat}, S_{an}$, ionised forms $S_{va^-}, S_{bu^-}, S_{pro^-}, S_{ac^-}, S_{HCO_3^-}, S_{NH_3}$ |
   | Gas phase | 37–40 | 4 | $p_{H_2}, p_{CH_4}, p_{CO_2}, p_{tot}$ |

### pH calculation

The pH is determined at every evaluation step from the **charge balance** using
a Newton–Raphson iteration:

$$
S_{cat} - S_{an} + (S_{NH4} - S_{NH3}) - S_{HCO_3^-} - \frac{S_{ac^-}}{64}
- \frac{S_{pro^-}}{112} - \frac{S_{bu^-}}{160} - \frac{S_{va^-}}{208}
+ S_{H^+} - \frac{K_w}{S_{H^+}} = 0
$$

Since every contribution on the left-hand side is a state variable, this charge
balance is a **purely point-wise function of the current state** within the ODE
step and converges to $[H^+]$ in 5–10 Newton iterations with high accuracy. The
fast equilibration of the ionised species via $k_{A,B}$ ensures the computed
pH stays consistent with the thermodynamic acid-base constants.

## Sub-fractioned disintegration

The most important difference compared to classical ADM1 is the
**two-pool disintegration** instead of the single composite variable $X_c$:

```text
                     substrate inflow
                          │
        ┌───────────────────┼─────────────────┐
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

| Pool | Substrate source (Weender + sub-fractioning parameters) |
| --- | --- |
| $X_{PS}$ | Crude fibre (always slow pool) **plus** the share $f_{sOTS}$ of NFE carbohydrates, proteins, and lipids |
| $X_{PF}$ | Share $f_{fOTS}$ of NFE carbohydrates, proteins, and lipids |
| $X_S$ | Produced by disintegration from $X_{PS}$ and $X_{PF}$; never sourced directly from the substrate |
| $X_I$ | Share $a_{XI}$ of the total raw organic COD |
| $S_I$ | Share $a_{Si}$ of the total raw organic COD (soluble inerts, appear directly in the inflow) |

This makes it possible, for example, to route easily degradable starch (NFE)
into the fast pool while lignocellulose (crude fibre) automatically lands in
the slow pool — without changing the model structure.

## Biomass decay routing

When a biomass population $X_i$ decays, the corresponding COD is routed to the
**hydrolysable** pool $X_S$ and the **inert** pool $X_I$ — **not** to the slow
disintegration pool $X_{PS}$. This follows the biomass-decay stoichiometry of
ADM1da (Schlattmann 2011), where all seven biomass populations decay into the
same combination of hydrolysable and inert pools. The COD-basis routing
fractions are

$$
f_{CH\_XB} = \frac{f_{BM,CH} \cdot M_{Xch}}{M_{XB}}
\approx 0.246, \quad
f_{PR\_XB} = \frac{f_{BM,PR} \cdot M_{Xpr}}{M_{XB}}
\approx 0.709, \quad
f_{LI\_XB} = \frac{f_{BM,LI} \cdot M_{Xli}}{M_{XB}}
\approx 0.045
$$

with $f_{BM,CH} = 0{,}20$, $f_{BM,PR} = 0{,}70$, $f_{BM,LI} = 0{,}10$ (biomass
mass composition), and $M_{Xch} = 0{,}9375$, $M_{Xpr} = 0{,}7736$,
$M_{Xli} = 0{,}3474$ (mass-to-COD ratios). A fraction $f_P = 0{,}20$ goes to the
inert pool $X_I$; the remaining $(1 - f_P) = 0{,}80$ is split among the
$X_{S,*}$ pools using the COD-basis fractions above. The hydrolysis rate
$k_{hyd} = 4\,\text{d}^{-1}$ is 100× faster than $k_{dis,PS} = 0{,}04\,\text{d}^{-1}$, so
decayed biomass re-enters the substrate chain on the hydrolysis timescale.

## Temperature-dependent kinetics

The ADM1da variant corrects every kinetic rate against the 35 °C reference
using an Arrhenius-θ function:

$$
k(T) = k(35\,°\text{C}) \cdot \theta^{(T[°\text{C}] - 35)}
$$

with group-specific exponents (values after Schlattmann 2011):

| Process group | $\theta_{\exp}$ |
| --- | --- |
| Disintegration & hydrolysis | 0.024 |
| $X_{su}, X_{aa}, X_{h2}$ (growth & decay) | 0.069 |
| $X_{fa}, X_{c4}, X_{pro}, X_{ac}$ (growth & decay) | 0.055 |
| $K_{I,NH_3,X_{pro}}$ | 0.061 |
| $K_{I,NH_3,X_{ac}}$ | 0.086 |
| $K_{I,H_2,X_{fa}/X_{c4}/X_{pro}}$ | 0.080 |

The correction is applied once when the `ADM1` object is created — to all
relevant rates via `ADMParams.apply_temperature_corrections` — and cached in
the internal `_kinetic` dict.

## Modified inhibition kinetics

Compared to the standard ADM1, the ADM1da model includes the following
adjustments — all implemented in `ADM1.ADM_ODE` following the ADM1da uptake-
process rate laws (Schlattmann 2011), with the NH₃-inhibition forms taken
from Siegrist et al. (2002):

| Inhibition | Standard ADM1 | ADM1da (this implementation) |
| --- | --- | --- |
| pH inhibition $X_{fa}/X_{c4}/X_{pro}$ | Hill, $n=1$ | Hill, $n=2$ (sharper cut-off) |
| pH inhibition $X_{ac}$ | Hill, $n=1$ | Hill, $n=3$ |
| pH inhibition $X_{h2}$ | Hill, $n=1$ | Hill, $n=3$ |
| N limitation | $S_{NH4}$ alone | $S_{IN} = S_{NH4} + S_{NH3}$ |
| $\text{NH}_3$ inhibition $X_{ac}$ | linear in $S_{NH3}$ | squared Hill: $K_I^2/(K_I^2 + S_{NH3}^2)$, T-corrected with $\theta=0.086$ |
| $\text{NH}_3$ inhibition $X_{pro}$ | not present | squared Hill with own $K_{I,nh3,pro}$, T-corrected with $\theta=0.061$ |
| Undissociated propionate $X_{pro}$ | not present | $K_{IH,pro}/(K_{IH,pro} + S_{pro} - S_{pro^-})$ (Fukuzaki et al. 1990) |
| Undissociated acetate $X_{ac}$ | not present | $K_{IH,ac}/(K_{IH,ac} + S_{ac} - S_{ac^-})$ (Xiao et al. 2013) |
| $\text{CO}_2$ limitation $X_{h2}$ | not present | squared Hill saturation: $S_{CO2}^2/(K_S^2 + S_{CO2}^2)$ |

!!! note "Inhibition scope"
    Some descriptions of the ADM1da kinetics list an acetate / undissociated-
    acid inhibition term on $X_{fa}$ and $X_{c4}$ as well. The implementation
    reference and PyADM1ODE apply these terms only to $X_{pro}$ and $X_{ac}$,
    where they are well-supported by experimental data (Fukuzaki et al. 1990,
    Xiao et al. 2013); $X_{fa}$ and $X_{c4}$ are inhibited only by pH,
    N-limitation, and dissolved H₂.

These extensions reproduce the behaviour typically observed in agricultural
plants: a sharper pH drop on acid accumulation, more pronounced $\text{NH}_3$
inhibition under thermophilic manure-based operation, and more realistic
propionate dynamics.

## Acid-base sub-system

The acid-base reactions of the six dissociating species ($\text{NH}_4^+ /
\text{NH}_3$, $\text{CO}_2 / \text{HCO}_3^-$, and the four VFA pairs) are
implemented as **kinetic** reactions of the form

$$
\rho_{A,i} = k_{A,B} \cdot \left( S_{i^-} \cdot S_{H^+} - K_{a,i} \cdot
(S_i - S_{i^-}) \right),
\quad i \in \{ \text{va}, \text{bu}, \text{pro}, \text{ac} \}
$$

with the same kinetic coupling for $\text{CO}_2 / \text{HCO}_3^-$ and
$\text{NH}_4^+ / \text{NH}_3$. The reaction-rate constant
$k_{A,B} = 10^8\,\text{d}^{-1}$ ensures sub-second equilibration. The acid
dissociation constants are T-corrected against the 35 °C reference using van't
Hoff with the enthalpies from Batstone 2002:

$$
K_a(T) = K_a(298\,\text{K}) \cdot
\exp\left(\frac{\Delta H^\circ}{R} \cdot
\left(\frac{1}{298} - \frac{1}{T}\right)\right)
$$

## Gas-liquid transfer (Henry's law with van't Hoff correction)

The four gas-phase species $\text{H}_2$, $\text{CH}_4$, $\text{CO}_2$ and
$\text{NH}_3$ are transferred between the liquid and gas phases via

$$
r_{F,gas} = k_L a_F \cdot \left( S_F - \frac{p_F}{K_H(T)\,R\,T} \right)
\cdot \frac{V_{liq}}{V_{gas}}
$$

with a van't Hoff temperature correction of the Henry constant
(Schlattmann 2011 / Batstone et al. 2002 enthalpies):

$$
H_F(T) = H_F(T_{ref}) \cdot
\exp\left(-\frac{\Delta H^\circ_F}{R} \cdot
\left(\frac{1}{T_{ref}} - \frac{1}{T}\right)\right)
$$

| Gas | $H_{F,35°C}$ [mol/(L·bar)] | $\Delta H^\circ_F$ [J/mol] | $k_L a_F$ [d⁻¹] |
| --- | --- | --- | --- |
| CO₂ | 0.0271 | 19 410 | 200 |
| CH₄ | 0.00116 | 14 240 | 200 |
| H₂ | 7.38·10⁻⁴ | 4 180 | 200 |
| NH₃ | 60 (at 25 °C reference) | 36 584 | 200 |

Sign convention: dissolution is exothermic, so $H_F$ decreases as $T$ rises —
less gas remains dissolved at higher operating temperature.

## Sludge-volume balance and HRT

Because a significant fraction of the substrate COD leaves the reactor as gas,
the sludge volume is not conserved. The dynamic sludge-volume balance follows
the biochemical-rate variant ("Approach 2") of the ADM1da reactor model
(Schlattmann 2011):

$$
\frac{dV_S}{dt} = \dot{q}_{S,in} - \dot{q}_{S,out} - \dot{q}_{S,loss},
\quad
\dot{q}_{S,loss} = V_S \cdot \sum_i r_{hyd,i} \cdot \frac{iM_i}{\rho_i}
$$

The hydraulic retention time follows a first-order lag (Schlattmann 2011):

$$
\frac{dHRT}{dt} + HRT \cdot \frac{\dot{q}_{S,in}}{V_S} = 1,
\quad HRT_{ss} = \frac{V_S}{\dot{q}_{S,in}}
$$

The effluent is driven by an overflow weir at $V_{liq,max}$ with a small time
constant $\tau_{out}$:

$$
\dot{q}_{S,out} = \max\!\left( 0,\; \frac{V_S - V_{liq,max}}{\tau_{out}} \right)
$$

In practice $\tau_{out} = 0{,}05\,\text{d}$ pins $V_S$ within ~1 m³ of the
setpoint, reproducing SIMBA#'s essentially-instantaneous weir behaviour.

When `dynamic_volume=False` (default for backward compatibility), $V_S$ is held
constant and $\dot{q}_{S,out} = \dot{q}_{S,in} - \dot{q}_{S,loss}$ is computed
directly from the mass loss term.

## Substrate characterization via Weender analysis

Substrates are defined via common laboratory parameters and stored as
**XML files** under `data/substrates/adm1da/`. An example structure:

```xml
<substrate name="Maize silage (milk ripeness)">
  <param name="TS"     value="320.0"/>
  <param name="NH4"    value="0.0"/>
  <param name="pH"     value="3.9"/>
  <param name="fRF"    value="0.220"/>   <!-- crude fibre -->
  <param name="fRP"    value="0.080"/>   <!-- crude protein -->
  <param name="fRFe"   value="0.030"/>   <!-- crude fat -->
  <param name="fRA"    value="0.045"/>   <!-- ash -->
  <param name="aXI"    value="0.10"/>    <!-- particulate inert COD share -->
  <param name="aSi"    value="0.02"/>    <!-- soluble inert COD share -->
  <param name="fOTSrf" value="0.40"/>    <!-- degradable share of crude fibre -->
  <param name="fsOTS"  value="0.30"/>    <!-- NFE/PR/LI into the slow pool -->
  <param name="ffOTS"  value="0.70"/>    <!-- NFE/PR/LI into the fast pool -->
  <param name="FFS"    value="0.0"/>     <!-- VFAs as acetic-acid equivalent -->
  <param name="KS43"   value="0.0"/>     <!-- acid capacity to pH 4.3 -->
  <!-- ... -->
</substrate>
```

`load_substrate_xml()` returns a `SubstrateParams` dataclass; the `Feedstock`
class derives from it the full 38-column ADM1 inflow stream (37 liquid state
columns + Q):

1. **Fresh-matter density** $\rho_{FM}$ from the component densities
   (specific-volume mixing rule).
2. **Organic COD concentrations** $X_{ch}, X_{pr}, X_{li}$ in
   $\text{kg COD/m}^3$ via the COD conversion factors $M_{Xch}, M_{Xpr},
   M_{Xli}$.
3. **Routing into the sub-fractions** $X_{PS}/X_{PF}$ via $f_{sOTS}, f_{fOTS}$,
   and $f_{OTSrf}$ (see the table above).
4. **Dissociation** at the substrate pH: ionised VFAs, $\text{HCO}_3^-$,
   $\text{NH}_3$.
5. **Charge balance** for the ions. By convention $S_{cation}$ is set to zero
   for every substrate type (per the ADM1da substrate-input characterisation,
   Schlattmann 2011), and $S_{anion}$ is computed from the charge balance —
   and is allowed to be negative for net-cationic substrates.

### Volumetric ADM1da convention for Q

Substrate volumetric flows $Q_i$ are interpreted as **mass-equivalent flows**
by default. Internally,

$$
Q_{actual,i} = Q_{input,i} \cdot \frac{1000}{\rho_{FM,i}}
$$

is computed. For liquid substrates ($\rho_{FM} \approx 1000$) this is a no-op;
for maize silage ($\rho_{FM} \approx 1134\,\text{kg/m}^3$) the actual
volumetric flow shrinks by roughly 12 %. This reproduces the behaviour of
ADM1da reference results. Setting `simba_q_convention=False` disables the
scaling — useful when $Q$ should be interpreted directly as the actual reactor
volume.

### Co-digestion: weighted blending

For multi-substrate inflows, concentrations are blended by volumetric flow
(`Feedstock._blended_concentrations`). The number of simultaneously fed
substrates is unbounded — `Feedstock` accepts an arbitrarily long list of
XML IDs.

## Measurement outputs

PyADM1ODE exposes the standard ADM1da measurement outputs (Schlattmann 2011)
for plant-monitoring use cases.

### Volatile fatty acids (VFA)

The aggregate VFA value in g HAc/L is the COD-weighted acetic-acid equivalent
of the four VFA species:

$$
\text{VFA} = M_{HAc} \cdot \sum_i \frac{S_i}{\text{COD}_{MOL,i}},
\quad i \in \{ \text{va}, \text{bu}, \text{pro}, \text{ac} \}
$$

with $M_{HAc} = 60\,\text{g/mol}$ and per-VFA COD-per-mol ratios
(va = 208, bu = 160, pro = 112, ac = 64 g COD/mol).

### Acid capacity (TAC)

The pH 5 titration capacity (TAC, in g CaCO₃/L) is

$$
\text{TAC} = 50 \cdot \Bigl[
\left(F_{NH_3} - K_{A,NH_4} \cdot
\frac{F_{NH_4} + F_{NH_3}}{10^{-pH_5} + K_{A,NH_4}}\right)
+ \left(F_{HCO_3} - K_{A,CO_2} \cdot
\frac{F_{CO_2} + F_{HCO_3}}{10^{-pH_5} + K_{A,CO_2}}\right)
+ \sum_{i = \text{va},\text{bu},\text{pro},\text{ac}} (\cdots)
+ F_{AN} - F_{CAT} \Bigr]
$$

!!! note "Prefactor 50 vs 100"
    Some descriptions of the TAC formula write the prefactor as the molar
    mass of CaCO₃ (100 kg/kmol). The physically correct value is the
    **equivalent weight 50 kg/keq**, because one mole of CaCO₃ carries two
    H⁺-equivalents of acid-neutralising capacity — the standard alkalinity
    convention. PyADM1ODE applies the equivalent-weight prefactor, which
    matches the reference numerical output of the SIMBA# reactor module.

## Mathematical foundation

The implementation builds on:

- **Schlattmann, M. (2011)**: *Weiterentwicklung des Anaerobic Digestion
  Model No. 1 (ADM1) zur Anwendung auf landwirtschaftliche Substrate.*
  Dissertation, TU München. (Source of the ADM1da sub-fractioned
  disintegration, biomass-decay routing, sludge-volume balance, and
  Weender-based substrate characterisation.)
- **Batstone, D. J. et al. (2002)**: *Anaerobic Digestion Model No. 1 (ADM1)*.
  IWA Scientific and Technical Report No. 13. (Base ADM1 stoichiometry,
  kinetics, and acid-base / Henry constants.)
- **Siegrist, H., Vogt, D., Garcia-Heras, J. L., Gujer, W. (2002)**:
  Mathematical model for meso- and thermophilic anaerobic sewage sludge
  digestion. *Environmental Science & Technology* **36**, 1113–1123.
  (Source of the ADM1da NH₃-inhibition forms and temperature-correction
  exponents.)
- **Fukuzaki, S., Nishio, N., Shobayashi, M., Nagai, S. (1990)**: Inhibition
  of the fermentation of propionate to methane by hydrogen, acetate, and
  propionate. *Applied and Environmental Microbiology* **56(3)**, 719–723.
  (Undissociated-propionate inhibition constant on $X_{pro}$.)
- **Xiao, K. et al. (2013)**: Acetic acid inhibition on methanogens in a
  two-phase anaerobic process. *Biochemical Engineering Journal* **75**,
  1–7. (Undissociated-acetate inhibition constant on $X_{ac}$.)
- **Wett, B., Eladawy, A., Ogurek, M. (2006)**: Description of nitrogen
  incorporation and release in anaerobic digestion modelling. *Water
  Science & Technology* **54(4)**, 67–76. (Fraction-based biomass decay
  products.)
- **Gaida, D. (2014)**: *Dynamic real-time substrate feed optimization of
  anaerobic co-digestion plants*. PhD thesis, Leiden University.
  (Template for the volumetric blending logic.)
- **Koch, K. et al. (2010)**: *Biogas from grass silage – measurements and
  modeling with ADM1*. Bioresource Technology. (Calibration values for
  high-strength energy crops.)

## Technical implementation

The whole model is pure Python:

| Module | Purpose |
| --- | --- |
| `pyadm1.core.adm1` | `ADM1` class with `ADM_ODE`, Newton–Raphson pH, gas output |
| `pyadm1.core.adm_params` | Stoichiometry, kinetics, inhibition, $\theta$ corrections |
| `pyadm1.core.solver` | Wrapper around `scipy.integrate.solve_ivp` (BDF, adaptive) |
| `pyadm1.substrates.feedstock` | XML parser, sub-fractioning routing, blending |
| `pyadm1.components.biological.digester` | Component wrapper incl. gas storage, sludge volume, HRT logic |

The simulation runs in any standard Python environment and works equally well
in containers, web notebooks (Colab), and CI pipelines.
