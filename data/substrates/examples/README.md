# Substrate file format examples

PyADM1ODE accepts substrate definitions in three formats. Pick whichever is
most convenient — the loader (`pyadm1.substrates.load_substrate`) dispatches
on the file extension and returns the same `SubstrateParams` dataclass in
every case.

| Format | Extension      | Loader                  | When to pick it                                   |
|--------|----------------|-------------------------|---------------------------------------------------|
| YAML   | `.yaml` / `.yml` | `load_substrate_yaml` | Canonical. Human-friendly, comments per parameter |
| XML    | `.xml`         | `load_substrate_xml`    | Interop with legacy substrate libraries           |
| TOML   | `.toml`        | `load_substrate_toml`   | Familiar to anyone who edits `pyproject.toml`     |

The three `cattle_manure.*` files in this directory all describe **exactly the
same substrate** in the standard schema (the SIMBA cattle manure that
`data/substrates/legacy/cattle_manure.yaml` holds). Loading any of them via

```python
from pyadm1.substrates import load_substrate
params = load_substrate("data/substrates/examples/cattle_manure.yaml")
# or .xml, or .toml — produces an equivalent SubstrateParams instance.
```

returns an equivalent `SubstrateParams` instance.

## Required vs optional parameters

In the standard schema only the **substrate-specific** parameters are mandatory:

* Proximate analysis: `TS`, `NH4`, `BGP`, `BMP`  
* COD fractionation: `aXI`, `fOTSrf`, `fsOTS`, `ffOTS`, `aSi`  
* Weender analysis: `fRF`, `fRP`, `fRFe`, `fRA`  
* Physical/chemical state: `Temp`, `pH`, `KS43`, `FFS`  

One **economic** parameter is optional in every format:

* `cost` — price free plant in EUR per tonne of fresh matter (default `0.0`).  
  It feeds `Feedstock.calc_substrate_costs` and `BiogasPlant.get_substrate_costs`.  

Two further keys are optional:

* `substrate_class` — remuneration class as free text, e.g. `"Rindergülle (EK II)"`.  
  Not used by the model; meant for a financial module.  
* `aXI` may be left out when `BMP` is a measured methane potential: the inert  
  share is then fitted to it (`BMP_recovery` sets the share of the degradable  
  COD a batch test recovers as methane, default 1).  

The remaining ADM1da model-default constants (densities `roh_*`, COD-conversion
factors `M_*`, methane potentials `MP_*`, acid-base equilibrium constants
`K*_35`, `V_m`, `CH4_cod_2_mol`, `N_aa`) inherit from `SubstrateParams`
defaults — only override them when a substrate truly differs from the
library defaults. None of the three example files in this directory
overrides them, since they're identical for every substrate.

## Measured parameters instead of the COD fractionation

`maize_silage_measured.yaml` describes a substrate by laboratory values, in the
names and units of the legacy Gummersbach library. The line `schema: measured`
(older spelling: `schema: gummersbach`) selects this parameter set; the COD
fractionation (`aXI`, `fOTSrf`, `fsOTS`, `ffOTS`, `aSi`) is then derived from the
cell-wall analysis (`NDF`, `ADL`) and the degradation level (`D_VS`) or a
measured methane potential (`BMP`). The thirteen library files in
`data/substrates/` are written this way. See the *Measured parameters* section
of the substrate user guide for all parameters and formulas.

```python
from pyadm1.substrates import load_substrate
maize = load_substrate("data/substrates/examples/maize_silage_measured.yaml")
```
