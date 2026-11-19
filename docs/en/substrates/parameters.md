# Parameters & Ranges

A substrate file holds four groups of values. This page explains what each
value means, how it is measured, and in which range it usually lies for
agricultural substrates. What a change does in the simulation is covered on
the page [Influence of the values](influence.md).

## The values at a glance

### Dry matter and composition

| Value | Unit | Meaning | Measurement |
| --- | --- | --- | --- |
| `TS` | % of fresh matter | Total solids: what remains after drying at 105 °C. The rest is water. | Drying oven |
| `VS` | % of TS | Volatile solids: the part of the TS that burns at 550 °C. The rest is ash. | Loss on ignition |
| `RP` | % of TS | Crude protein: nitrogen × 6.25. | Kjeldahl |
| `RL` | % of TS | Crude lipids. | Ether extraction |
| `RF` | % of TS | Crude fibre after Weender. Only needed when `NDF` and `ADL` are missing. | Weender analysis |
| `NDF` | % of TS | Neutral detergent fibre: the whole cell wall (cellulose, hemicellulose, lignin). | van Soest |
| `ADF` | % of TS | Acid detergent fibre: cellulose and lignin. Stored, but not used. | van Soest |
| `ADL` | % of TS | Acid detergent lignin: the non-degradable part of the cell wall. | van Soest |
| `NDICP` | % of TS | Protein bound in the cell wall. Needs `NDF` and `ADL`. | Nitrogen in the NDF residue |

What remains of the volatile solids after protein, lipids and fibre is counted
by the model as easily degradable carbohydrates (starch, sugars).

### Degradability

At least one of these three values must be given. If several are given, the
order of precedence is `D_VS`, `aXI`, `BMP`.

| Value | Unit | Meaning | Measurement |
| --- | --- | --- | --- |
| `BMP` | Nm³ CH₄ per t VS | Methane potential: the methane formed on complete degradation. The model fits the degradable share so that its potential equals this value. | Batch digestion test (VDI 4630) |
| `D_VS` | 0…1 | Degradation level: the share of the VS that is degradable. | Digestibility, digestion test |
| `aXI` | 0…1 | Inert share of the particulate COD (standard schema). | Not directly measurable |
| `BMP_recovery` | 0…1 | Share of the degradable COD the digestion test recovers as methane. Default 1. | – |

### Storage state

The state of the substrate as it reaches the digester.

| Value | Unit | Meaning |
| --- | --- | --- |
| `pH` | – | pH of the substrate. |
| `NH4` | kg N/m³ | Ammonium nitrogen. Alternatively `Snh4` in g NH₄/l. |
| `Sac`, `Spro`, `Sbu`, `Sva` | g/l | Acetic, propionic, butyric and valeric acid. Alternatively `FFS` as their sum in acetic acid equivalents. |
| `KS43` | mol/m³ | Bicarbonate. |
| `TAC` | mmol/l | Titrated acid capacity to pH 4.3. Unlike `KS43` it includes the acid anions titrated along; the loader subtracts them. |
| `Temp` | °C | Temperature of the substrate in storage. |

Lactic acid, the main fermentation acid of a silage, has no state of its own in
the model. It is not entered; its energy stays in the carbohydrates.

### Other

| Value | Unit | Meaning |
| --- | --- | --- |
| `cost` | EUR per t fresh matter | Price free plant; see [Substrate costs](#substrate-costs). |
| `substrate_class` | Text | Remuneration class, for example `"Rindergülle (EK II)"`. A slurry or manure class selects the defaults for manures. |
| `COD_S`, `SIin` | g COD/l | COD of the filtrate and soluble inert COD. Optional. |
| `X_bac`, `X_meth` | kg COD/m³ | Living biomass in the substrate (slurry, manure). Optional. |

## Realistic ranges

The ranges below come from published reference tables and serve to check your
own values for plausibility. They are spans over many farms and harvest years;
a value outside is not wrong, but it should have a reason.

!!! warning "The loader checks consistency, not plausibility"
    A substrate file with values outside the ranges below loads without a
    message. The loader rejects only what cannot be right: a misspelt
    parameter name, `TS` or `VS` outside 0–100, a pH outside 0–14, negative
    concentrations, `RP + RL + NDF` above `VS`, and detergent fibres that are
    not nested (`ADL ≤ ADF ≤ NDF`). One thing stays yours to watch: all
    contents are given in percent, not as a fraction (32.6, not 0.326).

### Dry matter, organic matter, methane yield

Range and, in brackets, mean after the Leitfaden Biogas (FNR 2016, tables
4.1–4.4 and 4.9). The guide does not list CCM and onions; their values are
single values of the database "Biogasausbeute" of LfL Bavaria, where the
methane yield is computed from biogas yield and methane content.

| Substrate | `TS` [% FM] | `VS` [% TS] | `BMP` [Nm³ CH₄/t VS] | Library value (`TS` / `VS` / `BMP`) |
| --- | --- | --- | --- | --- |
| Maize silage | 28–35 (33) | 85–98 (95) | 234–364 (340) | 32.6 / 94.0 / 338 |
| Whole-crop cereal silage | 30–35 (33) | 92–98 (95) | 290–350 (329) | 36.8 / 93.3 / 329 |
| Grass silage | 25–50 (35) | 70–95 (90) | 300–338 (310) | 35.0 / 90.5 / 317 |
| Green rye silage | (25) | (90) | (324) | 25.0 / 90.5 / 320 |
| Corn-cob-mix | 60–65 | 98 | about 350–370 | 65.0 / 98.5 / 380 |
| Cattle slurry | 6–11 (10) | 75–82 (80) | 110–275 (210) | 10.0 / 84.1 / 212 |
| Pig slurry | 4–7 (6) | 75–86 (80) | 180–360 (250) | 7.9 / 84.8 / 250 |
| Solid cattle manure | 20–25 (25) | 68–76 (80) | 130–330 (250) | 25.0 / 84.1 / 249 |
| Poultry manure | (40) | (75) | 200–360 (280) | 45.0 / 78.0 / 273 |
| Onions | 16 | 96 | about 360 | 10.9 / 96.8 / 379 |

For solid cattle manure the printed mean of the organic matter lies outside the
printed range; the table reproduces the guide. The methane content of the
biogas lies at 52–55 % for energy crops and 55–65 % for manures according to
the LfL database.

### Feed analysis

Smallest and largest value of the Feedipedia datasheets (INRAE, CIRAD, AFZ,
FAO), in percent of dry matter. For green rye silage the table values of the
Gruber Tabelle (LfL 2024) from ear emergence to after flowering.

| Substrate | `RP` | `RL` | `RF` | `NDF` | `ADF` | `ADL` | Ash (100 − `VS`) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Maize silage, 30–35 % TS | 4.7–9.3 | 1.3–4.0 | 15–26 | 38–55 | 19–33 | 1.9–5.1 | 2.1–5.4 |
| Maize silage, 25–30 % TS | 4.9–9.8 | 1.0–3.9 | 17–27 | 38–63 | 21–35 | 2.0–7.0 | 2.3–5.7 |
| Whole-crop cereal silage (barley) | 6.6–12.4 | 1.7–3.1 | 32–35 | 38–56 | 23–36 | 2.5–7.1 | 4.7–17.3 |
| Grass silage (timothy) | 9.4–17.2 | – | 28–35 | 47–64 | 28–42 | 3.6–3.8 | 5.7–11.2 |
| Green rye silage | 12–15 | 3.7–4.5 | 26–34 | 53–65 | 31–38 | – | – |
| Ear maize silage (closest to CCM) | 7.5–10.9 | 3.7–4.5 | 2.8–7.9 | 12–25 | 4.5–12 | 1.3–2.0 | 1.6–3.0 |

With rising maturity the cell wall of maize falls and the starch rises: starch
lies between 11 and 37 % of dry matter at 25–30 % TS, and between 21 and 42 %
at 35–40 % TS. The library file `maize_silage_milk_ripeness` sits at the upper
edge with 54.6 % `NDF`.

### Fermentation quality of silages

Typical ranges of well-fermented silages after Kung & Shaver (2001), in percent
of dry matter.

| Value | Maize silage (30–40 % TS) | Grass silage (30–35 % TS) |
| --- | --- | --- |
| pH | 3.7–4.2 | 4.3–4.7 |
| Lactic acid | 4–7 | 6–10 |
| Acetic acid | 1–3 | 1–3 |
| Propionic acid | < 0.1 | < 0.1 |
| Butyric acid | 0 | 0.5–1.0 |
| Ammonia N [% of crude protein] | 5–7 | 8–12 |

The file expects acids in g/l of fresh matter. The conversion:

```text
acid [g/l] ≈ acid [% of TS] · TS [% FM] / 10
```

So 1–3 % acetic acid in a maize silage of 33 % TS are 3.3–10 g/l (`Sac`). More
than 0.5 % butyric acid indicates a faulty fermentation.

### Storage state of slurry and manure

| Value | Cattle slurry | Pig slurry | Source |
| --- | --- | --- | --- |
| pH | 6.3–7.9 (7.3) | 6.5–7.9 (7.1) | Pizzul et al. 2018 |
| Ammonium nitrogen [kg/t] | 0.6–2.7 (2.0) | 1.5–5.2 (2.8) | Pizzul et al. 2018 |
| Total nitrogen [kg/t] | 1.5–5.1 (3.9) | 1.7–8.9 (4.6) | Pizzul et al. 2018 |
| `NH4` after fertiliser reference value [kg N/m³] | 1.6–2.1 | 3.0–3.4 (finishing), 2.3–2.8 (sows) | LfL Basisdaten 2024 |

For solid manure the LfL fertiliser reference values give 0.4 kg NH₄-N per
tonne (cattle manure, 18.5–23 % TS) and 7–9 kg per tonne (chicken manure and
droppings, 52–59 % TS). These values hold at the time of spreading, after
housing and storage losses.

For the acid capacity (`KS43`, `TAC`) and the fatty acids of raw slurry no
source with a range was found. The library values (bicarbonate 109–112 mol/m³,
fatty acids 8–10 g/l) rest on one fresh sample each; see
[Origin of the values](sources.md#storage-state).

### Sources of the ranges

- Fachagentur Nachwachsende Rohstoffe (ed.) (2016): Leitfaden Biogas – Von der Gewinnung zur Nutzung, 7th edition. Tables 4.1–4.4, 4.9. <https://www.fnr.de/fileadmin/Mediathek/208/Leitfaden_Biogas_web_V01.pdf>  
- Strobl, M., Keymer, U. (2016): Biogasausbeute mobil. Bayerische Landesanstalt für Landwirtschaft. <https://www.lfl.bayern.de/iba/energie/049711/>  
- Feedipedia – Animal Feed Resources Information System (INRAE, CIRAD, AFZ, FAO): datasheets Maize silage, Ear maize silage, Barley forage, Timothy grass silage. <https://www.feedipedia.org>  
- Bayerische Landesanstalt für Landwirtschaft (2024): Gruber Tabelle zur Fütterung der Milchkühe, Zuchtrinder, Schafe, Ziegen, 48th edition. <https://www.lfl.bayern.de/mam/cms07/publikationen/daten/informationen/gruber_tabelle_milchvieh-2024_lfl-information.pdf>  
- Kung, L., Shaver, R. (2001): Interpretation and use of silage fermentation analysis reports. Focus on Forage 3(13), University of Wisconsin Extension.  
- Pizzul, L., Rodhe, L., Burzynska, I. et al. (2018): Titration, buffer capacity and acid consumption of animal slurries in Baltic Sea Region countries. Baltic Slurry Acidification, RISE.  
- Bayerische Landesanstalt für Landwirtschaft (2024): Basisdaten Düngung und Nährstoffflüsse, table 5a. <https://www.lfl.bayern.de/mam/cms07/iab/dateien/bas25_tabelle_5a_20241125_neu.pdf>  

## Measured parameters

The COD fractionation above (`aXI`, `fOTSrf`, `fsOTS`, `ffOTS`, `aSi`) follows
SIMBA and cannot be measured directly. A substrate can instead be described by
laboratory values. The library files are written this way; the names and units
are those of the legacy Gummersbach substrate library.

```python
from pyadm1 import Feedstock
from pyadm1.substrates import load_gummersbach_library, load_substrate

library = load_gummersbach_library()                    # bundled legacy library, 10 substrates
feedstock = Feedstock(["gummersbach:maize", "gummersbach:swinemanure"])   # by ID
maize = load_substrate("data/substrates/examples/maize_silage_measured.yaml")
```

A substrate file selects the schema with `schema: measured` (YAML, TOML) or
`<substrate schema="measured">` (XML). The older spelling `schema: gummersbach`
is still accepted.

```yaml
name: Maize silage (measured parameters)
schema: measured
substrate_class: "Mais (GPS) (EK I)"
TS:   31.97    # % FM
VS:   96.25    # % TS
RP:    8.69    # % TS
RL:    3.68    # % TS
NDF:  43.64    # % TS
ADL:   2.15    # % TS
NDICP: 1.3     # % TS
BMP:  338      # Nm³ CH4/t VS, sets the inert share
pH:    3.93
Sac:   1.18    # g/l
Snh4:  0.66    # g/l
TAC:   6.0     # mmol/l
```

| Parameter | Unit | Meaning | Replaces |
| --- | --- | --- | --- |
| `TS`, `VS` | % FM, % TS | Total and volatile solids | `TS` [kg/t], `fRA` |
| `RF`, `RP`, `RL` | % TS | Weender analysis; `RF` is optional with `NDF` and `ADL` | `fRF`, `fRP`, `fRFe` |
| `NDF`, `ADL` | % TS | Cell wall and lignin after van Soest | `fOTSrf`, `fsOTS` / `ffOTS` for carbohydrates |
| `ADF` | % TS | Acid detergent fibre; stored, not used | — |
| `NDICP` | % TS | Protein bound in the cell wall; needs `NDF` and `ADL` | `fsOTS` / `ffOTS` for protein |
| `D_VS` | – | Degradation level of the volatile solids | `aXI`, `aSi` |
| `BMP` | Nm³ CH₄/t VS | Methane potential of a batch test; used without `D_VS` and `aXI` | `aXI` or `D_VS` |
| `BMP_recovery` | – | Share of the degradable COD the batch test recovers as methane, default 1 | — |
| `Sac`, `Spro`, `Sbu`, `Sva` | g/l | Single volatile fatty acids | `FFS` |
| `TAC` | mmol/l | Titrated acid capacity to pH 4.3 | `KS43` |
| `Snh4` | g/l | Ammonium, as NH₄ | `NH4` [kg N/m³] |
| `T` | °C | Storage temperature | `Temp` |
| `COD_S` | gCOD/l | COD of the filtrate | — |
| `SIin` | gCOD/l | Soluble inert COD | `aSi` |
| `X_bac`, `X_meth` | kgCOD/m³ | Living bacteria and methanogens (manure) | — |
| `substrate_class` | text | Remuneration class; a manure class selects the manure defaults | — |

`TS`, `VS`, `RP`, `RL` and `pH` are required; everything else is optional. The
inert share comes from `D_VS`, else from `aXI`, else from `BMP`; one of the
three has to be given. A misspelled parameter name raises an error. The keys of
the standard schema (`NH4`, `KS43`, `FFS`, `Temp`, `fOTSrf`, `aSi`, `cost`) are
accepted too, in their own units.

## Substrate characterisation

The standard schema, after SIMBA, describes a substrate by this parameter set.
The library files use the [measured parameters](#measured-parameters) instead,
from which the same set is derived:

- **Proximate analysis** — `TS`, `NH4`, biogas / biomethane potentials `BGP`, `BMP`.  
- **Weender analysis** — crude-fibre, crude-protein, crude-lipid, ash fractions of TS (`fRF`, `fRP`, `fRFe`, `fRA`).  
- **COD fractionation** — particulate-inert and dissolved-inert COD shares (`aXI`, `aSi`), biodegradable share of crude fibre (`fOTSrf`), slow / fast disintegration-pool split (`fsOTS`, `ffOTS`).  
- **Physical / chemical state** — substrate temperature, `pH`, acid capacity to pH 4.3 (`KS43`), VFAs as acetic-acid equivalent (`FFS`).  

Component densities, mass-to-COD conversion factors, methane potentials,
and the acid-base equilibrium constants are inherited from the model
defaults in `SubstrateParams` and need only be specified per substrate
when a measurement justifies overriding them.

See the [ADM1 Implementation](../user_guide/adm1_implementation.md) page for how the
characterisation is turned into the 38-column ADM1 inflow stream.

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

## Substrate costs

Each substrate may carry a `cost` in EUR per tonne of fresh matter, free plant.
It is optional and defaults to 0. Set it in the substrate file or per feedstock:

```yaml
# YAML
cost: 40.0      # EUR/t FM
```

```toml
# TOML
cost = 40.0     # EUR/t FM
```

```xml
<!-- XML -->
<param name="cost" value="40.0" unit="EUR/t_FM" description="Price free plant"/>
```

All substrates of the library carry an indicative price free plant (German
market, 2022–2025). Energy crops are priced at roughly 105–125 EUR per tonne of
dry matter, so their fresh-matter price follows the TS content; slurry carries
transport only.

| Substrate                              | `cost` [EUR/t FM] |
|----------------------------------------|-------------------|
| Maize silage                           | 40                |
| Cereal / wheat whole-plant silage      | 36 / 35           |
| Grass silage                           | 33                |
| Green rye silage                       | 22                |
| Corn-cob mix (CCM)                     | 125               |
| Chicken manure, dry                    | 30                |
| Cattle manure, solid                   | 8                 |
| Onion waste                            | 5                 |
| Cattle / swine manure (slurry)         | 2                 |

The regional and yearly spread is large (silage maize alone 30–50 EUR/t FM);
replace the value with the contract price of the plant.

```python
feedstock.set_substrate_costs({"maize_silage_milk_ripeness": 40.0, "swine_manure": 2.0})
feedstock.substrate_costs        # {'maize_silage_milk_ripeness': 40.0, 'swine_manure': 2.0}
```

There are two ways to get the costs of an operation.

**From any feed record** — configured rates, a feed log, a flow sensor or an
estimate. Pass one set of rates held for `dt` days, or a time series with one
row per interval:

```python
feedstock.calc_substrate_costs([10, 5], dt=30.0)          # 30 days at constant rates
feedstock.calc_substrate_costs(measured_feed, dt=1 / 24)  # hourly sensor readings
# {'total': 12300.0, 'by_substrate': {...}, 'mass_t': {...}}
```

**From the simulation** — the plant accumulates, on every step, what its
digesters really received. With a feeder that is the dosed flow (including an
empty storage or a blockage), not the configured rate:

```python
plant.simulate(duration=90)
plant.get_substrate_costs()
# {'total': ..., 'by_substrate': {...}, 'mass_t': {...}, 'specific_cost': ...}  # EUR/t FM
```

Comparing the two shows how far the recorded feed deviates from the simulated
one. The rates are in the unit of `Q_substrates` — tonnes per day with the
default `simba_q_convention=True`, m³/d otherwise (converted through the
substrate density).
