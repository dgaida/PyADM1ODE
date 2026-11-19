# Example Substrates

PyADM1ODE ships 13 substrate files under
[`data/substrates/`](https://github.com/dgaida/PyADM1ODE/tree/master/data/substrates).
Each describes typical material of one substrate type and is chosen by its
file name without extension:

```python
from pyadm1 import Feedstock
from pyadm1.substrates import SubstrateRegistry

SubstrateRegistry().available()      # all IDs of the library
feedstock = Feedstock(["maize_silage_milk_ripeness", "cattle_manure"])
```

## Overview

`Feedstock()` without arguments loads all 13 substrates in the order of the
table; `Q idx` is then the position in the `Q_substrates` array.

| Q idx | Substrate ID | Remuneration class | TS [% FM] | VS [% TS] | BMP [Nm³ CH₄/t VS] | CH₄ [m³/t FM] | Degradable VS | Slow pool [%] | Carbohydrate split |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | `maize_silage_milk_ripeness` | Mais (GPS) (EK I) | 32.6 | 94.0 | 338 | 104 | 0.82 | 46 | cell wall |
| 1 | `cattle_manure` | Rindergülle (EK II) | 10.0 | 84.1 | 212 | 18 | 0.49 | 100 | crude fibre |
| 2 | `swine_manure` | Schweinegülle (EK II) | 7.9 | 84.8 | 250 | 17 | 0.51 | 100 | crude fibre |
| 3 | `corn_cob_mix` | Corn-Cob-Mix (CCM) (EK I) | 65.0 | 98.5 | 380 | 243 | 0.93 | 9 | cell wall |
| 4 | `grass_silage` | Gras/Ackergras (EK I) | 35.0 | 90.5 | 317 | 100 | 0.69 | 29 | cell wall |
| 5 | `green_rye_silage` | Grünroggen (GPS) (EK I) | 25.0 | 90.5 | 320 | 72 | 0.77 | 38 | cell wall |
| 6 | `cereal_gps_silage` | Getreide (GPS) (EK I) | 36.8 | 93.3 | 329 | 113 | 0.85 | 49 | cell wall |
| 7 | `onion_waste` | Gemüse (aussortiert) (EK 0) | 10.9 | 96.8 | 379 | 40 | 0.96 | 12 | cell wall |
| 8 | `maize_silage_gummersbach` | Mais (GPS) (EK I) | 32.0 | 96.2 | 338 | 104 | 0.82 | 30 | cell wall |
| 9 | `cattle_manure_solid` | Rinderfestmist (EK II) | 25.0 | 84.1 | 249 | 52 | 0.68 | 100 | crude fibre |
| 10 | `chicken_manure_dry` | Geflügelmist, Geflügeltrockenkot (EK II) | 45.0 | 78.0 | 273 | 96 | 0.59 | 87 | cell wall |
| 11 | `swine_manure_gummersbach` | Schweinegülle (EK II) | 6.1 | 71.6 | 250 | 11 | 0.32 | 100 | crude fibre |
| 12 | `wheat_whole_plant_silage` | Getreide (GPS) (EK I) | 36.8 | 93.3 | 329 | 113 | 0.84 | 47 | cell wall |

How to read the columns:

- **TS, VS, BMP** are stated in the file. `CH₄ [m³/t FM]` is computed from them: the methane one tonne of fresh matter yields when fully degraded.  
- **Degradable VS** is the share of the volatile solids the model can degrade. It is the degradation level `D_VS` (cell-wall route) or the degradable share of the organic COD `1 − aXI − aSi` (crude-fibre route).  
- **Slow pool** is the share of the degradable particulate material that disintegrates slowly. The higher it is, the more sluggishly the gas production follows a change in feeding.  
- **Carbohydrate split** names the fibre analysis the file uses: the cell wall after van Soest (`NDF`, `ADL`) or the crude fibre after Weender (`RF`).  

## Which file for what

### Energy crops (EK I)

| File | Material | Note |
| --- | --- | --- |
| `maize_silage_milk_ripeness` | Maize silage, harvested early (milk ripeness) | Much cell wall (55 % of TS), little starch. Today's dough-ripe silage holds about 40 % cell wall and degrades faster. |
| `maize_silage_gummersbach` | Maize silage of the Gummersbach plant | Laboratory values of a real plant; goes with `swine_manure_gummersbach`. |
| `grass_silage` | Grass silage, meadow and ley grass | Rich in protein; brings much nitrogen into the digester. |
| `green_rye_silage` | Green rye whole-plant silage | The weakest file: composition taken from grass, fermentation quality from abroad. |
| `cereal_gps_silage`, `wheat_whole_plant_silage` | Whole-crop cereal silage | Same composition (wheat); they differ only in lipids and price. |
| `corn_cob_mix` | Corn-cob-mix (CCM) | Most energy-dense substrate, almost all of it fast degradable. The blend of 90 % grain and 10 % cob is an assumption. |

### Manures (EK II)

| File | Material | Note |
| --- | --- | --- |
| `cattle_manure` | Cattle slurry, 10 % TS | Little gas per tonne, but buffer and ammonium. |
| `swine_manure` | Pig slurry, 8 % TS | More ammonium than cattle slurry. |
| `swine_manure_gummersbach` | Pig slurry of the Gummersbach plant | Thinner (6 % TS) and high in ash. |
| `cattle_manure_solid` | Solid cattle manure, 25 % TS | Stackable; the buffer is given as titrated acid capacity. |
| `chicken_manure_dry` | Dry poultry manure, 45 % TS | Very rich in nitrogen; the protein includes the uric acid. |

### Residues (EK 0)

| File | Material | Note |
| --- | --- | --- |
| `onion_waste` | Rejected onions | Almost fully and quickly degradable, no ammonium, no buffer. |

The sources behind the values of each file are listed on the page
[Origin of the values](sources.md).

## Further bundled files

### Format examples

[`data/substrates/examples/`](https://github.com/dgaida/PyADM1ODE/tree/master/data/substrates/examples)
holds the same substrate (`cattle_manure`) as YAML, XML and TOML, plus
`maize_silage_measured.yaml` as a template for a file of measured values. These
files are not part of the library and are loaded by path.

### Legacy Gummersbach library

The substrate library of the predecessor model with ten substrates (`ccm`,
`cowmanure`, `futterkalk`, `gps`, `grass`, `greenrye`, `maize`, `onions`,
`swinemanure`, `wheat`) is included as one XML file and chosen with the prefix
`gummersbach:`:

```python
from pyadm1 import Feedstock
from pyadm1.substrates import load_gummersbach_library

library = load_gummersbach_library()
feedstock = Feedstock(["gummersbach:maize", "gummersbach:swinemanure"])
```

Several of its values are not measurements of the substrate described; see
[Differences to the legacy model](influence.md#differences-to-the-legacy-model).

### Superseded files

The files the library held before are kept in
[`data/substrates/legacy/`](https://github.com/dgaida/PyADM1ODE/tree/master/data/substrates/legacy).
They are not part of the registry and are selected by the prefix `legacy:`:

```python
from pyadm1 import Feedstock

old = Feedstock(["legacy:maize_silage_milk_ripeness", "legacy:cattle_manure"])
```

Results obtained with an earlier version are reproduced this way. The most
visible changes of the new files: cattle and pig slurry hold 45 to 65 % more
degradable COD (the superseded ifak values lie well below the KTBL yields),
maize silage puts 46 instead of 23 % of its degradable COD into the slow pool,
and the bicarbonate of the Gummersbach pig slurry drops from 212 to 51 mol/m³
because its acid capacity is now read as a titration value.
