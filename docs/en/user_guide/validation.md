# Validation

PyADM1ODE has been validated against the reference simulator **SIMBA# biogas**
(ifak Magdeburg) and against measurement data from real agricultural biogas
plants. This page describes the comparison with SIMBA# and gives its results.

## Reference simulator

SIMBA# biogas is the de-facto industry reference for the ADM1da agricultural
extension. The reference run uses SIMBA#'s ADM1da model (the kinetic acid-base
variant, *not* ADM1daph) in SIMBA# 5.0.57. PyADM1ODE repeats the run with the
same reactor, the same start state and **the same influent**: the influent is
the one SIMBA# computes from its own substrate blocks. The comparison therefore
tests the digestion model alone and does not depend on the substrate library.

## Reference run

A 600-day co-digestion of maize silage and cattle manure with a substrate
switch:

- **Phase 1** (0–300 d): 11.4 t/d maize silage + 6.1 m³/d cattle manure (16.2 m³/d).  
- **Phase 2** (300–600 d): 10.0 t/d maize silage + 8.0 m³/d cattle manure (16.8 m³/d).  

The reactor holds 1050 m³ of sludge and 150 m³ of gas at 42 °C, with
$k_L a = 200\,\text{d}^{-1}$ and a retention time of 62 to 65 days. The
operating point is a demanding one: 3.6 to 4.1 g/l of volatile fatty acids and
115 mg/l of free ammonia nitrogen keep the acetate degraders close to their
inhibition limit. Acetate then reacts about ten times stronger to any model
difference than the other quantities, which makes it a sensitive indicator.

The reference data (start state, influent of both phases, SIMBA# states at
300 d and 600 d and a coarse trajectory) are stored in
`tests/integration/data/simba_adm1da_cattle_600d.json`. The test
`tests/integration/test_simba_reference.py` repeats the run through the
`Digester` component and is part of the test suite.

### Headspace

SIMBA# holds its headspace at 1.017 bar with about 80 % relative humidity,
because it humidifies the gas at a finite rate. The dry gas then has 0.951 bar.
PyADM1ODE keeps the headspace saturated. The comparison fixes the dry gas
pressure of the SIMBA# run (`ADM1.p_ext = 1.033` bar); the effect of the
default headspace is listed separately below.

## Results

Deviation of PyADM1ODE from SIMBA# at the end of each phase:

| Quantity | Day 300 | Day 600 |
| --- | --- | --- |
| pH | +0.001 | +0.001 |
| Volatile fatty acids (VFA) | −0.4 % | −0.9 % |
| TAC | +0.1 % | +0.2 % |
| Methane flow | 0.0 % | 0.0 % |
| CO₂ flow | 0.0 % | 0.0 % |
| Methane content of the dry gas | +0.02 points | +0.02 points |
| Inorganic nitrogen, free ammonia | 0.0 % | 0.0 % |
| Inorganic carbon, bicarbonate | +0.2 % | +0.4 % |
| Acetate | −0.4 % | −0.9 % |
| Propionate, butyrate, valerate, sugars, amino acids, fatty acids | within 0.1 % | within 0.1 % |
| Particulate pools ($X_{PS}, X_{PF}, X_S$, $X_I$) | within 0.05 % | within 0.05 % |
| All seven biomass populations | within 0.05 % | within 0.05 % |

Over the whole run, from day 10 on:

| Quantity | Largest deviation |
| --- | --- |
| pH | 0.002 |
| TAC | 0.2 % |
| Inorganic nitrogen | 0.01 % |
| Methane flow | 0.11 % (the day after the substrate switch) |
| CO₂ flow | 0.02 % |
| Volatile fatty acids | 2.0 % (at day 10; 1 % or less from day 75 on) |

The ammonia content of the gas agrees as well: about 320 ppm in the dry gas in
both models.

### Default headspace

With the default headspace (saturated, 1.013 bar) the dry gas has 0.932 bar,
2 % less than in the SIMBA# run. Dissolved CO₂ and bicarbonate follow by the
same 2 %, and acetate rises to keep the pH:

| Quantity | Day 300 | Day 600 |
| --- | --- | --- |
| Acetate, VFA | +3.2 % | +3.4 % |
| TAC | −0.8 % | −0.7 % |
| Inorganic carbon | −1.9 % | −1.6 % |
| pH | 0.000 | +0.001 |
| Inorganic nitrogen, methane and CO₂ flow | within 0.1 % | within 0.1 % |

## Scope

- **Substrates:** The influent comes from SIMBA#. How PyADM1ODE derives an influent from a substrate file is not part of this comparison.  
- **Gas report:** SIMBA# reports wet gas (6.4 to 6.5 % water vapour in this run). Its methane and CO₂ flows compare directly with `Q_ch4` and `Q_co2`; its total gas flow and methane content have to be reduced to the dry gas first.  
- **Fast dynamics:** SIMBA# couples the acid-base pairs through the carbonate system, PyADM1ODE through the proton concentration. Both settle on the same equilibrium; the time they take differs, which only matters on the scale of hours.  
- **Earlier runs:** A 30-day run with swine manure and a 150-day run with maize silage and swine manure were compared with version 0.6. They have not been repeated, because the SIMBA# settings of those runs are no longer available.  

## Changes since version 0.6.2

Up to version 0.6.2 the same run gave acetate 20 to 30 % above SIMBA#, TAC 5
to 6 % below and the CO₂ flow 2 to 4 % below. Four deviations from the SIMBA#
model caused this, and all four are corrected:

- **Headspace pressure:** the water vapour pressure was subtracted twice, which left the dissolved CO₂ about 10 % too low.  
- **Acid-base terms:** the acid-base rates were added to the totals of inorganic carbon and nitrogen, which removed carbon from the gas.  
- **Nitrogen:** decay released nitrogen instead of binding a little, and ammonia did not leave with the gas; together about 3 % too much inorganic nitrogen.  
- **Sludge volume:** the volume loss was about 10 % too high, which raised all concentrations by about 1 %.  

The details are on the [ADM1 implementation](adm1_implementation.md) page.

## Conclusion

Given the same influent, PyADM1ODE reproduces SIMBA#'s ADM1da model within
0.2 % for pH, TAC, nitrogen, the particulate pools, the biomass and the gas
flows, and within 2 % for acetate, including the transient after a substrate
switch.
