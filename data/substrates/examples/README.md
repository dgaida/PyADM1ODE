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

The three files in this directory all describe **exactly the same substrate**
(cattle manure). Loading any of them via

```python
from pyadm1.substrates import load_substrate
params = load_substrate("data/substrates/examples/cattle_manure.yaml")
# or .xml, or .toml — produces an equivalent SubstrateParams instance.
```

returns an equivalent `SubstrateParams` instance.

## Required vs optional parameters

Only the **measured / substrate-specific** parameters are mandatory:

* Proximate analysis: `TS`, `NH4`, `BGP`, `BMP`
* COD fractionation: `aXI`, `fOTSrf`, `fsOTS`, `ffOTS`, `aSi`
* Weender analysis: `fRF`, `fRP`, `fRFe`, `fRA`
* Physical/chemical state: `Temp`, `pH`, `KS43`, `FFS`

The remaining ADM1da model-default constants (densities `roh_*`, COD-conversion
factors `M_*`, methane potentials `MP_*`, acid-base equilibrium constants
`K*_35`, `V_m`, `CH4_cod_2_mol`, `N_aa`) inherit from `SubstrateParams`
defaults — only override them when a substrate truly differs from the
library defaults. None of the three example files in this directory
overrides them, since they're identical for every substrate.
