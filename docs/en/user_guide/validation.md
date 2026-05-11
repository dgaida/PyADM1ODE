# Validation

PyADM1ODE has been validated against the reference simulator **SIMBA# biogas 4.2**
(ifak e.V. Magdeburg) and against measurement data from real agricultural biogas
plants. This page summarises the validation strategy and the headline results.

## Reference simulator

SIMBA# biogas 4.2 is the de-facto industry reference for the ADM1da agricultural
extension. The validation runs use SIMBA#'s ADM1da implementation (the kinetic
acid-base variant, *not* ADM1daph) with identical reactor geometry, temperature,
$k_L a$ value, substrate composition, and initial state. SIMBA# CSV exports
provide the reference state trajectories.

## Validation scenarios

### 30-day single-substrate run (swine manure)

20 m³/d swine manure, $V_{liq} = 1050\,\text{m}^3$, $V_{gas} = 150\,\text{m}^3$,
$T = 42\,°\text{C}$. Used to verify the basic process kinetics and the
initial-state extraction logic.

### 150-day co-digestion run (maize silage + swine manure)

11.4 m³/d maize silage + 6.1 m³/d swine manure, same reactor geometry.
Validates the multi-substrate blending logic and the `simba_q_convention`
volumetric correction.

### 600-day co-digestion run with substrate switch (maize silage + cattle manure)

The most comprehensive scenario:

- **Phase 1** (0–300 d): 11.4 m³/d maize silage + 6.1 m³/d cattle manure.
- **Phase 2** (300–600 d): 10.0 m³/d maize silage + 8.0 m³/d cattle manure.

The dynamic sludge volume is enabled (`dynamic_volume=True`,
`outflow_time_constant=0.05 d`) so that $V_{liq}$ tracks SIMBA#'s essentially-
instantaneous overflow weir within ~1 m³ of the setpoint.

## Headline results

At the end of each phase (t = 300 d and t = 600 d snapshots):

| Quantity                                                | Tolerance                                        | Status    |
| ------------------------------------------------------- | ------------------------------------------------ | --------- |
| $Q_{gas},\,Q_{CH_4},\,Q_{CO_2}$                         | 1–3 %                                            | ✓ matches |
| $pH$                                                    | within 0.01 units                                | ✓ matches |
| $HRT$                                                   | within 0.2 % (via dynamic sludge-volume balance) | ✓ matches |
| $OLR$                                                   | within 3 %                                       | ✓ matches |
| All seven biomass populations $X_*$                     | 1–4 %                                            | ✓ matches |
| All particulate substrate pools ($X_{PS}, X_{PF}, X_S$) | 1–4 %                                            | ✓ matches |
| Soluble substrates $S_{su},\,S_{aa},\,S_{fa}$           | within 1 %                                       | ✓ matches |
| VFA species $S_{va},\,S_{bu},\,S_{pro}$ (and ions)      | within 1 %                                       | ✓ matches |
| Substrate-switch transient at $t = 300\,\text{d}$       | same characteristic timescale                    | ✓ matches |

Two residual offsets persist at the validation operating point:

- **$S_{ac}$ (and aggregate VFA) is +19–21 % higher** than SIMBA#. This is a
  saturation-amplification artefact: at $S_{ac} \gg K_{S,ac} = 0.15$ the
  acetoclastic Monod kinetics are saturated and the acetate concentration is
  set by the slow dilution channel rather than by Monod feedback. A small
  upstream input-side discrepancy (~1.1 % in the effective $Q$, traced to a
  density-convention difference for maize silage) is amplified by a factor of
  ~20 into the observed S_ac offset. Under non-saturated operating conditions
  the amplifier disappears.
- **TAC is −6 %**, traceable to the same biomass excess: 2–3 % more biomass
  sequesters ~0.013 kmol C/m³ of inorganic carbon inside the biomass
  particulates, reducing the dissolved $S_{HCO_3^-}$ pool that dominates the TAC
  formula.

Both offsets are stable across the substrate switch at t = 300 d, confirming
they are steady-state characteristics rather than divergent integration errors.

## Conclusion

PyADM1ODE is considered validated as a drop-in replacement for SIMBA# biogas 4.2
in gas-yield forecasting, OLR and HRT analysis, and pH-based monitoring. The
two residuals on the acetate pool and TAC are documented and traceable to the
known density-convention difference between the two simulators.

A full report of the 600-day cattle-manure comparison, including all themed
time-series plots and the snapshot tables, is available in the
`Report/` directory of the repository.
