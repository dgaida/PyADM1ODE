# Influence of the Values

Every value of a substrate file acts on one of four quantities of the
simulation: how much gas forms, how fast it forms, how much nitrogen reaches
the digester, and how well the digester is buffered. This page first shows
which value acts on what, then a worked example with numbers, and finally the
formulas behind it.

## Which value acts on what

| Value | Acts on | Direction |
| --- | --- | --- |
| `TS` | Gas amount, nitrogen load, loading rate | More dry matter per tonne brings proportionally more of everything: gas, nitrogen, organic load. |
| `VS` | Gas amount | More organic matter in the dry matter, correspondingly less ash. |
| `BMP` (or `D_VS`, `aXI`) | Gas amount | Sets which share of the organic matter is degradable. The rest leaves the digester as inert material. |
| `NDF` | Speed | More cell wall moves degradable material from the fast to the slow pool. At a fixed `BMP` the potential stays the same, but in the digester a larger part is washed out before it is degraded. |
| `ADL` | Almost nothing while `BMP` is given | Lignin is inert. At a fixed `BMP` the degradation level is fitted so that the potential matches; `ADL` then only shifts the make-up of the inert rest. With a fixed `D_VS`, `ADL` limits how much of the cell wall is degradable. |
| `RP` | Nitrogen, pH, inhibition | Protein releases ammonium when degraded. More protein raises ammonium and pH and can inhibit methane formation. |
| `RL` | Methane content | Lipids yield more methane per kilogram and a gas richer in methane than carbohydrates. |
| `NDICP` | Speed of protein degradation | Protein bound in the cell wall is only released with the cell wall and goes to the slow pool. |
| `NH4` / `Snh4` | Nitrogen, pH, inhibition | Ammonium the substrate already carries. Acts like `RP`, but at once. |
| `KS43` / `TAC` | Buffer | Bicarbonate of the substrate. Together with ammonium and acids it sets the charge balance of the influent. |
| `Sac`, `Spro`, `Sbu`, `Sva` (or `FFS`) | Acid load | Acids already fermented. They are available at once and burden the buffer; propionate and butyrate first have to be converted to acetate. |
| `pH` | Charge balance of the influent | Sets which share of the acids and the ammonium is dissociated, and with it the cations and anions the substrate brings along. |
| `Temp` | Heat demand | No influence on the biology (the digester has its own temperature), but on the heating: every kelvin colder costs about 1.2 kWh of heat per tonne of substrate. |
| `cost` | Costs | Only enters the cost calculation. |
| `substrate_class` | Defaults | A slurry or manure class assigns protein and lipids without a measurement to the slow pool, otherwise to the fast one. |

## Worked example

A digester of 3000 m³ at 40 °C receives 30 t of maize silage
(`maize_silage_milk_ripeness`) and 10 t of cattle slurry (`cattle_manure`) per
day, with the default kinetics. In each row **one** value of one file is
changed, everything else stays the same. Shown is the steady state after
400 days. The half-time is the time after which the methane production has
reached half of its rise when the maize amount steps up by 50 %.

| Changed value | Methane [Nm³/d] | Change | VFA [g/l] | pH | NH₄-N [g/l] | Half-time [d] |
| --- | --- | --- | --- | --- | --- | --- |
| **Base case** | 2731 | – | 0.6 | 7.45 | 2.20 | 5.0 |
| Maize `TS` 28 instead of 32.6 % | 2361 | −14 % | 0.2 | 7.42 | 1.97 | 4.3 |
| Maize `TS` 38 % | 3155 | +15 % | 2.3 | 7.43 | 2.51 | 5.6 |
| Maize `VS` 92 instead of 94 % | 2669 | −2 % | 0.7 | 7.45 | 2.23 | 5.2 |
| Maize `VS` 96 % | 2794 | +2 % | 0.5 | 7.45 | 2.18 | 4.8 |
| Maize `BMP` 300 instead of 338 | 2460 | −10 % | 1.0 | 7.45 | 2.30 | 5.0 |
| Maize `BMP` 370 | 2959 | +8 % | 0.4 | 7.44 | 2.12 | 4.9 |
| Maize `NDF` 40 instead of 54.6 % | 2817 | +3 % | 0.5 | 7.45 | 2.17 | 4.1 |
| Maize `NDF` 65 % | 2670 | −2 % | 0.7 | 7.45 | 2.23 | 6.0 |
| Maize `ADL` 2 or 7 instead of 4.6 % | 2731 | 0 % | 0.6 | 7.45 | 2.20 | 5.0 |
| Maize `RP` 5 instead of 6.4 % | 2736 | 0 % | 0.1 | 7.36 | 1.65 | 4.2 |
| Maize `RP` 10 % | 2610 | −4 % | 10.7 | 7.22 | 3.65 | 5.7 |
| Maize `RL` 2 instead of 3 % | 2721 | 0 % | 0.6 | 7.45 | 2.20 | 5.0 |
| Maize `RL` 5 % | 2752 | +1 % | 0.7 | 7.46 | 2.21 | 4.9 |
| Maize `Sac` 1 instead of 4.2 g/l | 2725 | 0 % | 0.5 | 7.44 | 2.19 | 4.9 |
| Maize `Sac` 10 g/l | 2741 | 0 % | 0.9 | 7.47 | 2.22 | 5.1 |
| Maize `pH` 3.6 or 4.3 instead of 3.9 | 2732 / 2729 | 0 % | 0.5 / 0.8 | 7.45 / 7.47 | 2.20 | 4.9 / 5.2 |
| Slurry `TS` 5 instead of 10 % | 2666 | −2 % | 0.4 | 7.44 | 2.11 | 4.7 |
| Slurry `BMP` 150 instead of 212 | 2693 | −1 % | 0.5 | 7.45 | 2.15 | 4.8 |
| Slurry `NH4` 1.5 instead of 2.4 kg N/m³ | 2735 | 0 % | 0.3 | 7.47 | 1.92 | 4.5 |
| Slurry `NH4` 4.5 kg N/m³ | 2711 | −1 % | 2.2 | 7.37 | 2.85 | 5.6 |
| Slurry `KS43` 50 instead of 109 mol/m³ | 2735 | 0 % | 0.4 | 7.41 | 2.21 | 4.6 |
| Slurry `KS43` 250 mol/m³ | 2710 | −1 % | 2.2 | 7.49 | 2.20 | 5.8 |
| Slurry `Temp` 5 or 20 instead of 10 °C | 2731 | 0 % | 0.6 | 7.45 | 2.20 | 5.0 |

What the table shows:

- **The gas amount hangs on `TS` and `BMP`.** Both act almost proportionally. They are the values to measure first for a real plant.  
- **`NDF` sets the sluggishness.** Between 40 and 65 % cell wall the half-time lies between 4.1 and 6.0 days; the gas amount changes by a few percent only.  
- **Nitrogen decides on stability.** More protein or ammonium raises the ammonium in the digester, and the acids rise. At 10 % crude protein in the maize the digester tips over (11 g/l acids).  
- **The storage state of the silage is secondary.** pH and acetic acid of the silage change the gas amount by less than 0.5 %.  
- **More buffer in the substrate does not help here.** More bicarbonate in the slurry raises the pH, with it the share of free ammonia, and the acids rise. This holds only because the base case already sits close to ammonia inhibition.  

!!! warning "The model is sensitive to nitrogen"
    With the default kinetics the base case already sits close to ammonia
    inhibition. Protein-rich rations (grass silage, poultry manure, much
    slurry) lead to high acid levels faster in the model than in practice,
    where the biology adapts to ammonia. If you compute such rations,
    calibrate the inhibition constant `K_I_nh3` against plant data.

The numbers hold for this one operating point. At another loading rate or
retention time the magnitudes change, the directions remain.

## How the measured values are used

**Cell wall instead of crude fibre.** With `NDF` and `ADL` the carbohydrates are
split by the cell wall:

```text
slow pool   X_PS_ch = d · (NDF − ADL)                 structural carbohydrates
fast pool   X_PF_ch = VS − RP − RL − NDF              non-fibre carbohydrates
inert       X_I     = ADL + (1 − d) · (NDF − ADL)     lignin and non-degradable fibre
            d       = (NDF − VS · (1 − D_VS)) / (NDF − ADL)
```

The non-degradable volatile solids `VS · (1 − D_VS)` are assigned to the cell
wall, lignin first (Koch et al. 2010). Protein and lipids count as degradable.
`d` is limited to 0…1: non-degradable matter beyond the whole cell wall is
spread over the cell contents, and lignin stays inert even if `D_VS` claims
more. Each fraction is converted to COD with its own factor, lignin with
1.563 kgCOD/kg. The total COD equals the COD of the composition, and the
degraded share of the volatile solids equals `D_VS`.

**Inert share from a measured methane potential.** With `BMP` and neither
`D_VS` nor `aXI`, the degradability is fitted until the model potential equals
the measured one:

```text
model potential = degradable COD · 0.350 Nm³ CH4/kgCOD / VS  =  BMP / BMP_recovery
```

With a cell-wall analysis the degradation level `D_VS` is fitted, otherwise the
inert share `aXI`. Both change the potential monotonically, so the result is
unique. A potential the composition cannot reach raises an error. The same
applies to a file of the standard schema that gives `BMP` but no `aXI`; in
code, `fit_degradability_to_bmp(params)` does the fit.

`BMP_recovery` defaults to 1: the model potential is matched to the measured
one directly. Two effects of opposite sign are left out by that. A batch test
recovers about 5 % less methane than the degradable COD holds, because biomass
grows (Weißbach 2009). The model counts carbohydrates as glucose, which is
about 10 % below the COD of starch and cellulose. Setting `BMP_recovery: 0.95`
accounts for the first effect alone; for the cereal whole-plant silage and
the onions the composition then no longer reaches the standard yield.

**Protein bound in the cell wall.** `NDICP` (neutral detergent insoluble crude
protein, Licitra et al. 1996) is the protein that stays in the NDF residue. It
is released only as the cell wall disintegrates and enters the slow pool; the
remaining protein enters the fast pool:

```text
slow pool   X_PS_pr = NDICP                     X_PS_ch = d · (NDF − NDICP − ADL)
fast pool   X_PF_pr = RP − NDICP                X_PF_ch = VS − RP − RL − (NDF − NDICP)
```

The bound protein is weighed with the `NDF`, so it is taken out of the
structural carbohydrates; this is the non-fibre carbohydrate formula of NRC
(2001). `fsOTS` / `ffOTS` then only split the lipids. Lipids have no such
measurement: the detergent removes them, they belong to the cell contents.

**Single acids.** Each acid enters its own ADM1 state. Propionate and butyrate
then pass acetogenesis, which `FFS` (all acetate) skips.

**Titrated acid capacity.** A titration to pH 4.3 also protonates acid anions.
They are subtracted, so only the rest counts as bicarbonate:

```text
S_hco3 = TAC − Σ S_acid · (α(pH) − α(4.3))        limited to ≥ 0
```

For the swine manure of the Gummersbach plant this leaves 51 instead of
212 mol/m³. `KS43` of the standard schema is still taken as bicarbonate
directly.

**Filtrate COD.** `COD_S` beyond the acids and the soluble inerts is matter
that is already disintegrated. It enters the hydrolysable pools `X_S_*` and is
taken out of the disintegration pools in proportion.

**Biomass.** `X_bac` and `X_meth` are split over the seven ADM1 groups in the
ratios of the digester inoculum and taken out of the disintegration pools.
`load_gummersbach_library()` adds the biomass of the legacy model to manure
classes (0.25 and 0.0003 kgCOD/m³ at a storage age of 10 days);
`manure_age=None` switches it off.

**Remuneration class.** `substrate_class` is free text and is not used by the
model. The library uses the classes of the German EEG 2012 as listed in the
BiomasseV: `EK I` for energy crops (Anlage 2), `EK II` for manure (Anlage 3),
`EK 0` for substrates without a substrate-related tariff (Anlage 1). A financial
module can key tariffs, the maize cap or greenhouse-gas factors to it. The
field is available in both schemas and is stored in `SubstrateParams`.

## What is still not measured

- The split of the **lipids** between the slow and the fast pool, and of the  
  **protein** without `NDICP` (`fsOTS` / `ffOTS`). Default: slow for manure
  classes, fast otherwise. In the library only maize silage and corn-cob mix
  carry an `NDICP`; no sourced value was found for the other substrates.  
- The **disintegration rates**. The legacy library holds a rate `kdis` per  
  substrate; ADM1da has two pools with fixed rates that are calibrated per
  digester. `kdis`, `khyd_*` and `km_*` are read and ignored.  

## Differences to the legacy model

- One disintegration pool with a rate per substrate becomes two pools, split  
  by the cell wall.  
- The legacy model subtracts all acid anions from `TAC`, not only the titrated  
  share.  
- Density uses additive volumes instead of the mass-weighted mean.  
- `cost` of the legacy library is in EUR/m³ and is converted to EUR/t FM.  

!!! note "Quality of the legacy values"
    Several values of the legacy library are not measurements of the substrate
    they describe. `NDF` / `ADF` / `ADL` of maize are literature values; green
    rye, onions, corn-cob mix and solid cattle manure carry the same `NDF`
    without a source. The degradation level of swine manure (0.95) has no
    source. `COD_S` is labelled as an assumption for five substrates and has
    no source for the other four. The library files under `data/substrates/`
    no longer use these values; `gummersbach:<id>` still does.
