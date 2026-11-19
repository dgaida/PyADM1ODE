# ADM1 Implementation and Substrate Modeling

This page describes the technical details of the ADM1 model implemented in
PyADM1ODE and how agricultural substrates are integrated into the model.

PyADM1ODE implements **ADM1da** (Schlattmann 2011) — an agricultural adaptation
of the original ADM1 (Batstone et al. 2002, IWA Task Group). Compared to the
classical formulation, this model adds **sub-fractioned disintegration**,
**temperature-dependent kinetics**, and **modified inhibition kinetics** for
agricultural co-digestion. The implementation reproduces the reference behaviour
of **SIMBA# biogas 4.2** (ifak Magdeburg) quantitatively (see the Validation page).
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
2. **Kinetic gas-liquid transfer**: $\text{H}_2$, $\text{CH}_4$ and  
   $\text{CO}_2$ are transferred to the gas phase via $k_L a$ rates — no
   algebraic Henry equilibrium constraint. Free ammonia leaves with the gas at
   its equilibrium concentration.  
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

The inert share keeps the composition of the biomass, so it holds protein in
the same proportion as the degradable share. Biomass carries
$N_{XB} = f_{BM,PR} \cdot N_{aa}$ nitrogen, its decay products
$f_{PR\_XB} \cdot N_{aa}$, with $N_{aa} = 0.00765$ kmol N per kg COD (13.84 %
nitrogen in protein). Decay therefore binds a little ammonium
($6.9 \cdot 10^{-5}$ kmol N per kg COD) instead of releasing any. Up to version
0.6.2 the inert share had a lower nitrogen content; decay then released
nitrogen and left about 2 % more ammonium in the digester.

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
| pH inhibition $X_{su}/X_{aa}/X_{fa}/X_{c4}/X_{pro}$ | Hill, $n=1$ | Hill, $n=2$ (sharper cut-off) |
| pH inhibition $X_{ac}$ | Hill, $n=1$ | Hill, $n=3$ |
| pH inhibition $X_{h2}$ | Hill, $n=1$ | Hill, $n=3$ |
| $\text{NH}_3$ inhibition $X_{ac}$ | linear in $S_{NH3}$ | squared Hill: $K_I^2/(K_I^2 + S_{NH3}^2)$, T-corrected with $\theta=0.086$ |
| $\text{NH}_3$ inhibition $X_{pro}$ | not present | squared Hill with own $K_{I,nh3,pro}$, T-corrected with $\theta=0.061$ |
| Undissociated propionate $X_{pro}$ | not present | $K_{IH,pro}/(K_{IH,pro} + S_{pro} - S_{pro^-})$ |
| Undissociated acetate $X_{ac}$ | not present | $K_{IH,ac}/(K_{IH,ac} + S_{ac} - S_{ac^-})$ |
| $\text{CO}_2$ limitation $X_{h2}$ | not present | squared Hill saturation: $S_{CO2}^2/(K_S^2 + S_{CO2}^2)$ with the dissolved $S_{CO2} = S_{IC} - S_{HCO_3^-}$ |

!!! note "Inhibition scope"
    The ADM1da kinetics as published in Schlattmann (2011) list an acetate /
    undissociated-acid inhibition term on $X_{fa}$ and $X_{c4}$ as well. The
    SIMBA# reference implementation and PyADM1ODE apply these terms only to
    $X_{pro}$ and $X_{ac}$, where they are well-supported by experimental
    data; $X_{fa}$ and $X_{c4}$ are inhibited only by pH, N-limitation, and
    dissolved H₂.

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
$\text{NH}_4^+ / \text{NH}_3$. A pair settles at the rate
$k_{A,B} \cdot (S_{H^+} + K_a)$. With
$k_{A,B} = 10^8\,\text{m}^3\,\text{kmol}^{-1}\,\text{d}^{-1}$ and pH 7.4 the
VFA pairs settle within a minute, $\text{CO}_2 / \text{HCO}_3^-$ within half an
hour and $\text{NH}_4^+ / \text{NH}_3$ within a few hours, which is fast
against the biology. The acid dissociation constants are T-corrected against
the 25 °C reference using van't Hoff with the enthalpies from Batstone 2002:

$$
K_a(T) = K_a(298\,\text{K}) \cdot
\exp\left(\frac{\Delta H^\circ}{R} \cdot
\left(\frac{1}{298} - \frac{1}{T}\right)\right)
$$

$S_{IC}$ (`S_co2`) and $S_{IN}$ (`S_nh4`) are **totals**: bicarbonate and free
ammonia are subsets of them, as the term $S_i - S_{i^-}$ of the rate shows. An
acid-base reaction only shifts between the two forms, so $\rho_A$ appears
in the balance of the ionised form alone and not in the balance of the total.
Carbon and nitrogen are conserved this way; up to version 0.6.2 the rate was
also added to the two totals, which removed about 4 % of the ammonium and
2 % of the gas carbon from the balance.

## Gas-liquid transfer (Henry's law with van't Hoff correction)

The three gas-phase species $\text{H}_2$, $\text{CH}_4$ and $\text{CO}_2$
are transferred between the liquid and gas phases via

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

| Gas | $H_{F,25°C}$ [mol/(L·bar)] | $\Delta H^\circ_F$ [J/mol] | $k_L a_F$ [d⁻¹] |
| --- | --- | --- | --- |
| CO₂ | 0.035 | 19 410 | 200 |
| CH₄ | 0.0014 | 14 240 | 200 |
| H₂ | 7.8·10⁻⁴ | 4 180 | 200 |
| NH₃ | 60 | 36 584 | – (see below) |

Sign convention: dissolution is exothermic, so $H_F$ decreases as $T$ rises —
less gas remains dissolved at higher operating temperature.

### Ammonia in the gas

Ammonia is over a thousand times more soluble than $\text{CO}_2$, so its gas
concentration follows the liquid without delay: the headspace holds
$S_{NH_3} / (H_{NH_3}\,R\,T)$ kmol N per m³. This ammonia leaves with the gas
flow and is taken from $S_{IN}$ and $S_{NH_3}$; there is no separate gas-phase
state for it. At 42 °C and 115 mg/l of free ammonia nitrogen the dry gas
carries about 300 ppm, just under 1 % of the inorganic nitrogen of a maize and
manure digester.

### Headspace pressure

The gas leaves through an outlet at the rate
$k_p \cdot (p_{dry} + p_{H_2O} - p_{ext})$. The wet headspace therefore settles
at the pressure $p_{ext}$ it vents against, and the dry gas holds
$p_{ext} - p_{H_2O}$: 0.93 bar at 42 °C with the default
$p_{ext} = 1.013$ bar. A gas system with back pressure is set through
`ADM1.p_ext`.

The dry gas pressure sets the dissolved $\text{CO}_2$ and with it the pH. Up to
version 0.6.2 the water vapour pressure was subtracted twice, which left the
dry gas at 0.85 bar and the dissolved $\text{CO}_2$ about 9 % too low.

### Reported gas flows

`Q_gas`, `Q_ch4` and `Q_co2` are **dry gas at standard conditions** (0 °C,
1.01325 bar; DIN 1343), in Nm³/d:

- The flows are the ones the gas-phase ODE removes from the headspace, split by the dry partial pressures. Methane, carbon dioxide and hydrogen add up to `Q_gas`, and the methane flow closes the COD balance.  
- The headspace is saturated with water vapour. It adds to the pressure that drives the outflow, but it is not part of `Q_gas`; `ADM1.calc_gas()` returns it separately as `q_h2o`.  
- The reference volume is the one `BMP` uses (22.414 Nm³ per kmol), so a methane yield and a methane potential can be compared directly.  
- `Q_ch4 / Q_gas` is the methane content of the dry gas, which is what a gas analyser shows.  

Up to version 0.6.2 the flows were given at 20 °C with the water vapour
counted inside `Q_gas`. Against those numbers `Q_gas` is about 7 % lower, the
methane content about 4 percentage points higher and `Q_ch4` about 1 % higher
(at 40 °C).

## Sludge-volume balance and HRT

Because a significant fraction of the substrate COD leaves the reactor as gas,
the sludge volume is not conserved. Every organic fraction takes up the volume
$v_i = iM_i / \rho_i$ per kg COD (mass-to-COD ratio over density); hydrogen,
methane and inorganic carbon take up none. The sludge loses the net volume of
what the conversions turn over:

$$
\frac{dV_S}{dt} = \dot{q}_{S,in} - \dot{q}_{S,out} - \dot{q}_{S,loss},
\quad
\dot{q}_{S,loss} = -V_S \cdot \sum_j \rho_j \sum_i \nu_{ij} \cdot v_i
$$

with the rate $\rho_j$ of process $j$ and its stoichiometric coefficients
$\nu_{ij}$.

| Fraction | $iM$ [kg/kg COD] | $\rho$ [kg/m³] |
| --- | --- | --- |
| Carbohydrates, sugars | 0.9375 | 1550 |
| Proteins, amino acids | 0.7736 | 1370 |
| Lipids, fatty acids | 0.3474 | 920 |
| Valerate / butyrate / propionate / acetate | 0.490 / 0.550 / 0.661 / 0.9375 | 940 / 960 / 990 / 1050 |
| Biomass | 0.7638 | 1361 |

Disintegration and the hydrolysis of carbohydrates and proteins keep the
volume; most of it is lost where acetate becomes methane. At steady state the
loss is about 0.59 litres per kg COD of methane formed, the ratio SIMBA#
determines for its reactors. Up to version 0.6.2 the loss was taken from the
hydrolysis rates alone, which gave about 10 % more.

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
**YAML files** under `data/substrates/`. The standard schema, after SIMBA, has
this structure (the superseded `legacy/maize_silage_milk_ripeness.yaml`):

```yaml
name: Maize silage (milk ripeness)

TS:     330.0     # kg/m^3 FM   total solids
NH4:      0.8     # kg N/m^3 FM ammonia nitrogen
BGP:    670.0     # Nm^3/t VS   biogas potential
BMP:    356.8     # Nm^3 CH4/t VS  biomethane potential

aXI:      0.1029  # particulate inert fraction of degradable COD
fOTSrf:   0.95    # biodegradable fraction of crude fibre
fsOTS:    0.0     # share of degradable VS into slow pool (XPS)
ffOTS:    1.0     # share of degradable VS into fast pool (XPF)
aSi:      0.0     # dissolved inert fraction of degradable COD

fRF:      0.235   # crude fibre   (fraction of TS)
fRP:      0.09    # crude protein
fRFe:     0.03    # crude lipid
fRA:      0.10    # ash

Temp:    20.0     # degC
pH:       4.0
KS43:     0.0     # mol/m^3   acid capacity to pH 4.3
FFS:      5.0     # kg HAc/m^3 VFAs as acetic-acid equivalent
```

The same file may instead be written as XML or TOML; the loader dispatches
on the extension. See [`data/substrates/examples/`](https://github.com/dgaida/PyADM1ODE/tree/master/data/substrates/examples)
for parallel examples in all three formats.

A substrate may also be described by measured laboratory values instead of
the COD fractionation (cell-wall analysis `NDF` / `ADL`, protein bound in the
cell wall `NDICP`, methane potential `BMP` or degradation level `D_VS`, single
acids, titrated acid capacity). The bundled substrate files are written this
way. See
[Measured parameters](../substrates/parameters.md#measured-parameters).

`load_substrate()` returns a `SubstrateParams` dataclass; the `Feedstock`
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
5. **Charge balance** for the ions, closed by the strong ions and split by  
   sign: a net-anionic material (silage at its own pH) carries the excess as
   $S_{anion}$, a net-cationic one (slurry, solid manure, where
   $\text{HCO}_3^-$ and the VFA anions exceed $\text{NH}_4^+$) as $S_{cation}$
   — the strong-base cations K⁺, Na⁺, Ca²⁺, Mg²⁺ of the material. Both pools
   stay non-negative. Only the difference $S_{cation} - S_{anion}$ enters pH,
   speciation and the acid capacity, so the chemistry equals the ADM1da
   closure (Schlattmann 2011: $S_{cation} \equiv 0$, $S_{anion}$ signed); the
   split keeps the influent inside the non-negative state box that state
   estimators clip to.

### Volumetric ADM1da convention for Q

Substrate volumetric flows $Q_i$ are interpreted as **mass-equivalent flows**
by default. Internally,

$$
Q_{actual,i} = Q_{input,i} \cdot \frac{1000}{\rho_{FM,i}}
$$

is computed. For liquid substrates ($\rho_{FM} \approx 1000$) this is a no-op;
for maize silage ($\rho_{FM} \approx 1129\,\text{kg/m}^3$) the actual
volumetric flow shrinks by roughly 11 %. This reproduces the behaviour of
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

!!! warning "Convention: the ion term is the strong-ion difference"
    The term $F_{AN} - F_{CAT}$ subtracts the strong-ion difference
    $S_{cation} - S_{anion}$, which the substrate characterisation closes from
    the charge balance (split by sign, both pools non-negative — the same
    chemistry as the ADM1da closure with $S_{cation} \equiv 0$). A state whose
    strong-ion difference does not match its weak-acid buffer therefore
    reports too little acid capacity — negative in the extreme. The
    pre-inoculated initial state is closed the same way, and tanks without a
    substrate feed of their own (post-digesters, stores) are seeded with the
    ammonium of the ADM1da reference inoculum rather than with zero nitrogen.
    Should the formula
    still yield $\text{TAC} \le 0$, PyADM1ODE reports `NaN` and warns once —
    this keeps FOS/TAC from exploding on a denominator drifting through zero.

!!! note "Prefactor 50 vs 100"
    The TAC formula as written in Schlattmann (2011) uses the molar mass of
    CaCO₃ (100 kg/kmol) as prefactor. The physically correct value is the
    **equivalent weight 50 kg/keq**, because one mole of CaCO₃ carries two
    H⁺-equivalents of acid-neutralising capacity — the standard alkalinity
    convention. PyADM1ODE applies the equivalent-weight prefactor, which
    matches the reference numerical output of the SIMBA# reactor module.

## References

The implementation builds on the following primary sources (matching the
citation list of the PyADM1ODE–SIMBA# validation report):

- **Batstone, D. J., Keller, J., Angelidaki, I., Kalyuzhnyi, S. V.,  
  Pavlostathis, S. G., Rozzi, A., Sanders, W. T. M., Siegrist, H.,
  Vavilin, V. A. (2002).** *Anaerobic Digestion Model No. 1 (ADM1).*
  IWA Scientific and Technical Report No. 13. IWA Publishing, London.  
- **Henneberg, W., Strohmann, F. (1864).** *Beiträge zur Begründung einer  
  rationellen Fütterung der Wiederkäuer.* Schwetschke, Braunschweig.
  (Origin of the Weender feed analysis used for substrate characterisation.)  
- **Schlattmann, M. (2011).** *Weiterentwicklung des Anaerobic Digestion  
  Model No. 1 (ADM1) zur Anwendung auf landwirtschaftliche Substrate.*
  Dissertation, Technische Universität München.  
- **Siegrist, H., Vogt, D., Garcia-Heras, J. L., Gujer, W. (2002).**  
  Mathematical Model for Meso- and Thermophilic Anaerobic Sewage Sludge
  Digestion. *Environmental Science & Technology* **36**(5), 1113–1123.

## Technical implementation

The whole model is pure Python:

| Module | Purpose |
| --- | --- |
| `pyadm1.core.adm1` | `ADM1` class with `ADM_ODE`, Newton–Raphson pH, gas output |
| `pyadm1.core.adm_params` | Stoichiometry, kinetics, inhibition, $\theta$ corrections |
| `pyadm1.core.solver` | Wrapper around `scipy.integrate.solve_ivp` (BDF, adaptive) |
| `pyadm1.substrates.feedstock` | XML parser, sub-fractioning routing, blending |
| `pyadm1.components.biological.digester` | Component wrapper with sludge volume and HRT logic |

The simulation runs in any standard Python environment and works equally well
in containers, web notebooks (Colab), and CI pipelines.

### Rate-family hook

`ADM1.rate_hook` lets an outside model rescale the kinetic rates without
touching balances or stoichiometry. It is `None` by default, which leaves
`ADM_ODE` unchanged. When set, it is called once per right-hand-side evaluation
as `rate_hook(state, S_H, q_ad)` and must return 13 factors in the order of
`pyadm1.core.adm1.RATE_FAMILIES`:

```text
dis_PS, dis_PF, hyd_ch, hyd_pr, hyd_li, su, aa, fa, c4, pro, ac, h2, dec
```

Each factor multiplies every rate of its family (for example `c4` scales the
valerate and the butyrate uptake, `dec` all seven decay rates). The torch
backend applies the same factors: `adm1da_rhs_torch(x, p, rate_factors)` takes
a `(…, 13)` tensor and stays differentiable in it, and `make_scipy_rhs` calls the
hook like the numpy path. Typical uses are sensitivity studies of the rate
families and hybrid models that correct the kinetics of one plant.
