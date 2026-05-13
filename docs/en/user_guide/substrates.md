# Pre-configured Substrates

PyADM1ODE ships with 12 example ADM1da substrate characterisations under
[`data/substrates/`](https://github.com/dgaida/PyADM1ODE/tree/master/data/substrates).
Each file is a YAML mapping of the parameters listed in
[`SubstrateParams`](adm1_implementation.md) and is loaded via the
`SubstrateRegistry` by its file stem.

!!! note "Examples only — define your own for real plants"
    The bundled substrates are **examples** drawn from the literature and a
    legacy plant library. They are useful for quick demos, smoke tests, and
    reproducing the validation runs, but they are **not a substitute for
    measured data**. To model a real plant, characterise your own feedstock
    (Weender analysis, TS / VS, BMP from a batch test) and add a new file
    under `data/substrates/` in any supported format.

## Available substrates

Listed in the canonical default order — when you call `Feedstock()` with
no arguments, this is the order in which substrates land in the `Q`
array (`Q[0]` is the first row).

| Q idx | Substrate ID | Display name | Type | TS [kg/m³ FM] | BMP [Nm³ CH₄/t VS] |
| --- | --- | --- | --- | --- | --- |
| 0 | `maize_silage_milk_ripeness` | Maize silage (milk ripeness) | Energy crop | 330 | 357 |
| 1 | `cattle_manure` | Cattle manure | Animal waste | 80 | 137 |
| 2 | `swine_manure` | Swine manure | Animal waste | 80 | 149 |
| 3 | `corn_cob_mix` | Corn-cob mix (CCM) | Energy crop | 676 | 426 |
| 4 | `grass_silage` | Grass silage | Energy crop | 341 | 338 |
| 5 | `green_rye_silage` | Green rye silage | Energy crop | 193 | 322 |
| 6 | `cereal_gps_silage` | Cereal whole-plant silage (GPS) | Energy crop | 312 | 290 |
| 7 | `onion_waste` | Onion waste | Vegetable waste | 193 | 300 |
| 8 | `maize_silage_gummersbach` | Maize silage (Gummersbach plant) | Energy crop | 320 | 348 |
| 9 | `cattle_manure_solid` | Cattle manure (solid) | Animal waste | 120 | 282 |
| 10 | `swine_manure_gummersbach` | Swine manure (Gummersbach plant) | Animal waste | 61 | 203 |
| 11 | `wheat_whole_plant_silage` | Wheat whole-plant silage | Energy crop | 302 | 298 |

The `*_gummersbach` and `cattle_manure_solid` entries were converted from
the legacy [`substrate_gummersbach.xml`](https://github.com/dgaida/PyADM1ODE/blob/master/data/substrates/legacy/substrate_gummersbach.xml)
library via a Buswell-based Weender → BMP mapping. The others were taken
from the ifak Magdeburg substrate library.

## File formats

Each substrate can be defined as YAML (canonical), XML, or TOML. The loader
dispatches on the file extension and produces the same `SubstrateParams`
object in every case; see
[`data/substrates/examples/`](https://github.com/dgaida/PyADM1ODE/tree/master/data/substrates/examples)
for side-by-side examples of the same substrate in all three formats.

```python
from pyadm1.substrates import SubstrateRegistry, load_substrate

# By ID — discovers any supported format in data/substrates/
reg = SubstrateRegistry()
maize = reg.get("maize_silage_milk_ripeness")

# By explicit path — extension picks the loader
swine = load_substrate("data/substrates/swine_manure.yaml")
```

## Substrate characterisation

Every substrate carries the same parameter set:

- **Proximate analysis** — `TS`, `NH4`, biogas / biomethane potentials `BGP`, `BMP`.  
- **Weender analysis** — crude-fibre, crude-protein, crude-lipid, ash fractions of TS (`fRF`, `fRP`, `fRFe`, `fRA`).  
- **COD fractionation** — particulate-inert and dissolved-inert COD shares (`aXI`, `aSi`), biodegradable share of crude fibre (`fOTSrf`), slow / fast disintegration-pool split (`fsOTS`, `ffOTS`).  
- **Physical / chemical state** — substrate temperature, `pH`, acid capacity to pH 4.3 (`KS43`), VFAs as acetic-acid equivalent (`FFS`).  

Component densities, mass-to-COD conversion factors, methane potentials,
and the acid-base equilibrium constants are inherited from the model
defaults in `SubstrateParams` and need only be specified per substrate
when a measurement justifies overriding them.

See the [ADM1 Implementation](adm1_implementation.md) page for how the
characterisation is turned into the 38-column ADM1 inflow stream.
