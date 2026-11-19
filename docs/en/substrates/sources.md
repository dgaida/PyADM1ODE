# Origin of the Values

This page documents where the values of the 13 library files come from and
where their limits lie.

Every value in a library file names its source in a comment. Five sources are
combined, because no single one covers a substrate completely:

| Source | Supplies | Parameters |
| --- | --- | --- |
| [Phyllis database](https://phyllis.nl) (TNO) | Composition of the dry matter, as the median of a documented selection of samples | `VS`, `RP`, `RL`, `NDF`, `ADF`, `ADL`; `TS` where the samples were analysed fresh |
| BiomasseV 2012, Anlagen 1–3 | Standard methane yield per tonne of fresh matter, converted with the KTBL standard dry matter | `BMP`, `substrate_class`; `TS` where Phyllis has no fresh sample |
| Published analyses of the material as it is fed (see [Storage state](#storage-state)) | Storage state | `pH`, ammonium, acids, bicarbonate or acid capacity, temperature |
| Literature | Protein bound in the cell wall | `NDICP` of maize silage and corn-cob mix |
| Superseded files; laboratory of the Gummersbach plant for its two files | Price; storage state of the plant files | `cost`; `pH`, acids, acid capacity of `maize_silage_gummersbach` and `swine_manure_gummersbach` |

Phyllis is a database for thermochemical conversion. It holds ash, elemental
and cell-wall analyses, but **no** pH, ammonium, fatty acids, acid capacity or
methane potential; its only pH values belong to biochar and wood ash. These
parameters cannot come from it.

## Phyllis samples

| Material | Samples | Used for |
| --- | --- | --- |
| Maize, whole plant, five hybrids, analysed fresh, van Soest fibre | #2603–#2607 | `TS`, `VS`, cell wall of `maize_silage_milk_ripeness` |
| Corn silage, elemental analysis | #2795 | Protein (N × 6.25) of `maize_silage_milk_ripeness` |
| Wheat, whole crop, analysed fresh, van Soest fibre | #2648–#2653 | `TS`, `VS`, protein, cell wall of `wheat_whole_plant_silage` and `cereal_gps_silage` |
| Meadow grass | #613–#617, #1699, #3193, #3197, #3286 | `VS`, protein of `grass_silage` and `green_rye_silage` |
| Grass with detergent fibre | #1038, #2377 | Cell wall of `grass_silage` and `green_rye_silage` |
| Maize grain; shelled corn | #1033; #1973 | Grain part of `corn_cob_mix` |
| Corn cob (spindle) | #979, #1035, #1037, #1102, #1240, #1454, #2068, #2302, #2791 | Spindle part of `corn_cob_mix` |
| Fresh cattle manure; cow faeces | #1882, #2782, #1528; #1391 | `VS`, protein; lipids of both cattle manures |
| Pig manure | #1366, #1715, #2246, #3069 | `TS` (#1366), `VS`, protein of `swine_manure` |
| Chicken manure, poultry litter | 25 samples; #2380; #957 | `VS`, protein; cell wall; lipids of `chicken_manure_dry` |

Samples are left out when they do not describe the material as it is fed:
composted or aged manure, digestate, ash, char, pellets, manure with soil or
sand bedding. Fibre values are left out when the lignin is a Klason lignin, the
method is unknown, or the components add up to more than 100 % of the dry
matter. Cellulose, hemicellulose and lignin of a van Soest analysis give
`NDF` (their sum), `ADF` (cellulose + lignin) and `ADL` (lignin).

## Storage state

The storage state describes the substrate as it enters the digester: pH,
ammonium, volatile fatty acids, bicarbonate or acid capacity, and temperature.
It comes from published analyses of the material, looked up in November 2026:

| File | pH | NH4-N [kg/m³] | Acids [g/l] | Buffer | Temp. [°C] | Source |
| --- | --- | --- | --- | --- | --- | --- |
| `cattle_manure` | 7.4 | 2.42 | 6.2 / 1.2 / 0.6 / 0.1 | `KS43` 109 mol/m³ from TIC | 10 | Overmeyer et al. 2020; pH: Biogashandbuch Bayern |
| `swine_manure` | 7.34 | 2.88 | 7.0 / 1.6 / 1.2 / 0.4 | `KS43` 112 mol/m³ from TIC | 20 | Overmeyer et al. 2020 |
| `cattle_manure_solid` | 7.85 | 1.59 | `FFS` 6.7 | `TAC` 250 mmol/l | 9.3 | Mahato et al. 2020 |
| `chicken_manure_dry` | 7.7 | 3.04 | `FFS` 0 (assumption) | `TAC` 170 mmol/l | 9.3 | Zhang et al. 2019, scaled to 45 % TS |
| `maize_silage_milk_ripeness` | 3.9 | 0.17 | 4.2 / 0 / 0.07 | none | 9.3 | LfL Maissilage 2018 (n = 92) |
| `grass_silage` | 4.6 | 0.38 | 6.3 / 0 / 1.9 | none | 9.3 | LfL Grassilage 2018 (n = 82) |
| `wheat_whole_plant_silage`, `cereal_gps_silage` | 4.0 | 0.38 | 7.0 / 0 / 0.4 | none | 9.3 | Leisen 2011, LUFA NRW (n = 20) |
| `green_rye_silage` | 4.45 | 0.79 | 1.0 / 0.2 / 0 | none | 9.3 | Zhao et al. 2021 (proxy) |
| `corn_cob_mix` | 4.42 | 1.24 | 1.5 / 0.3 / 0 | none | 9.3 | Li et al. 2023 (proxy) |
| `onion_waste` | 5.5 | 0 (assumption) | 0 | none | 9.3 | FDA pH of foods |

Acids are acetic / propionic / butyric (/ valeric) acid; `FFS` is the sum as
acetic acid. The conventions behind the table:

- **Slurries**: the fresh dairy-cow and fattening-pig slurry of the University  
  of Bonn (Overmeyer et al. 2020, week 0) supplies ammonium, the single acids
  and the total inorganic carbon of one consistent sample each. `KS43` is the
  bicarbonate share of that inorganic carbon at the slurry pH (pKa 6.35). The
  pH of the cattle slurry is the one of the Bavarian biogas handbook (7.42 for
  fresh slurry at 10 % TS); surveys give 7.2 (Petersen et al. 2016, n = 39)
  and 7.74 (Overmeyer et al. 2020).  
- **Solid manures**: a measured total alkalinity is written as `TAC`, from  
  which the loader subtracts the acid anions titrated along. The values of the
  chicken manure are measured at 33 % TS and scaled to the 45 % TS of the file.
  The ammonium of the poultry manure is the measured ammonium; the fertiliser
  reference values (LfL Basisdaten) count 45 % of the nitrogen as ammonium by
  convention, uric acid included, and are not used.  
- **Silages**: the laboratories report the fermentation acids in g/kg dry  
  matter and the ammonia nitrogen as a share of the total nitrogen. Both are
  converted with `TS` and `RP` of the file (N = RP / 6.25; 1 kg fresh matter
  taken as 1 l). Lactic acid is the main fermentation product (maize 45, grass
  59 g/kg TS) but has no state in ADM1: it is **not** written to the file, its
  COD stays in the carbohydrate pools. A silage below pH 4.3 has no acid
  capacity by definition. Propionic acid is not reported separately by the
  German laboratories and is set to zero where it is not measured.  
- **Temperature**: substrates stored outdoors take the mean air temperature  
  of Germany 1991–2020 (DWD, 9.3 °C). Slurry in cattle houses is 5.5–12.3 °C
  (Petersen et al. 2016; 10 °C in the file), in pig houses about 20 °C
  (Ma et al. 2023).  
- **Proxies**: no open data set for a German green rye silage or corn-cob mix  
  could be found. Green rye carries the fermentation quality of Korean forage
  rye cut at heading (Zhao et al. 2021), corn-cob mix that of untreated
  high-moisture ear corn at 63 % DM (Li et al. 2023), which lies within the
  typical ranges of Kung & Shaver. Both files say so in their comments.  

## What each file rests on

- **`maize_silage_milk_ripeness`**: Phyllis. The five hybrids hold 51–73 %  
  cell wall and 8–25 % non-structural carbohydrates, an early, starch-poor
  harvest. Today's dough-ripe silage holds about 40 % cell wall.  
- **`wheat_whole_plant_silage`, `cereal_gps_silage`**: Phyllis. Wheat is the  
  only whole-crop cereal in the database, so both files share its dry-matter
  composition and differ in lipids and price only.  
- **`grass_silage`**: Phyllis for ash, protein and cell wall. `TS` is the KTBL  
  standard, because the grass samples were dried.  
- **`corn_cob_mix`**: Phyllis holds grain and spindle separately. The file  
  mixes 90 % grain and 10 % spindle; the share is an assumption.  
- **`cattle_manure`, `cattle_manure_solid`, `swine_manure`**: Phyllis for ash,  
  protein and lipids. The fibre values of the manure samples are unusable (one
  adds up to 127 % with its ash, the others lack the hemicellulose), so these
  files keep the crude-fibre route with the crude fibre of the superseded file.  
- **`chicken_manure_dry`**: Phyllis, including one van Soest analysis. Protein  
  is N × 6.25 of the dried sample and therefore contains the uric acid.  
- **`green_rye_silage`**: no Phyllis sample. The dry-matter composition is the  
  one of grass, as the closest material; the fermentation quality is that of
  Korean forage rye cut at heading. The weakest file of the library.  
- **`onion_waste`**: Phyllis holds onion peel and ash analyses only. The  
  composition of the whole onion comes from the USDA food database; total
  dietary fibre stands in for the cell wall. Fresh bulbs carry neither
  ammonium nor fermentation acids; the pH is the one of raw onions.  
- **`maize_silage_gummersbach`, `swine_manure_gummersbach`**: plant-specific  
  files. The laboratory values of the plant are kept; only the degradation
  level, which had no source, is replaced by the standard methane yield.  

!!! warning "Limits of these values"
    Nitrogen measured on a dried manure sample misses the ammonia lost in
    drying; protein from it is an estimate. The methane yields are the guide
    values behind the BiomasseV 2012; the KTBL revised its guide values in 2021
    and that edition was not available here. The conversion of the yield to the
    volatile solids uses the KTBL standard dry matter of each substrate. The
    storage state of each slurry rests on one fresh sample of one farm; the
    fatty acids of the dry poultry manure and the ammonium of the onion waste
    are set to zero for want of a source; green rye and corn-cob mix carry
    proxies from abroad.

## References

- Koch, K., Lübken, M., Gehring, T., Wichern, M., Horn, H. (2010): Biogas from  
  grass silage – Measurements and modeling with ADM1. Bioresource Technology
  101, 8158–8165.  
- Licitra, G., Hernandez, T. M., Van Soest, P. J. (1996): Standardization of  
  procedures for nitrogen fractionation of ruminant feeds. Animal Feed Science
  and Technology 57, 347–358.  
- NRC (2001): Nutrient Requirements of Dairy Cattle, 7th revised edition.  
  National Academy Press, Washington, D.C.  
- Weißbach, F. (2009): gas formation potential from the fermentable organic  
  matter, as cited by Liebetrau, J.: Efficiency of biogas production (lecture
  slides).  
- Rossow, H.: Interpreting Forage Quality from the Cows' Perspective.  
  University of California Cooperative Extension (lecture slides; `NDICP` of
  maize silage and maize grain).  
- Biomasseverordnung in the version of 1 January 2012, Anlagen 1–3.  
- Phyllis2, database for (treated) biomass, algae, feedstocks for biogas  
  production and biochar. TNO, <https://phyllis.nl>.  

Storage state:

- Overmeyer, V., Holtkamp, F., Clemens, J., Büscher, W., Trimborn, M. (2020):  
  Dynamics of different buffer systems in slurries based on time and
  temperature of storage and their visualization by a new mathematical tool.
  Animals 10, 724. <https://doi.org/10.3390/ani10040724>  
- Bayerisches Landesamt für Umwelt (2022): Biogashandbuch Bayern, Kap. 2.2.7,  
  Tab. 1: Analyse von Frisch- und Faulgülle.
  <https://www.lfu.bayern.de/energie/biogashandbuch/doc/kap227.pdf>  
- Petersen, S. O., Olsen, A. B., Elsgaard, L., Triolo, J. M., Sommer, S. G.  
  (2016): Estimation of methane emissions from slurry pits below pig and
  cattle confinements. PLoS ONE 11, e0160968.
  <https://doi.org/10.1371/journal.pone.0160968>  
- Ma, C., Guldberg, L. B., Hansen, M. J., Feng, L., Petersen, S. O. (2023):  
  Frequent export of pig slurry for outside storage reduced methane but not
  ammonia emissions in cold and warm seasons. Waste Management 169, 223–231.
  <https://doi.org/10.1016/j.wasman.2023.07.006>  
- Bayerische Landesanstalt für Landwirtschaft (2024): Basisdaten Düngung und  
  Nährstoffflüsse, Tabelle 5a: Nährstoffgehalte von Wirtschaftsdüngern.
  <https://www.lfl.bayern.de/mam/cms07/iab/dateien/bas25_tabelle_5a_20241125_neu.pdf>  
- Mahato, P., Goyette, B., Rahaman, M. S., Rajagopal, R. (2020): Processing  
  high-solid and high-ammonia rich manures in a two-stage (liquid-solid)
  low-temperature anaerobic digestion process. Bioengineering 7, 80.
  <https://doi.org/10.3390/bioengineering7030080>  
- Zhang, C., Wang, F., Pei, M., Qiu, L., Qiang, H., Yao, Y. (2019):  
  Performance of anaerobic digestion of chicken manure under gradually
  elevated organic loading rates. International Journal of Environmental
  Research and Public Health 16, 2239.
  <https://doi.org/10.3390/ijerph16122239>  
- Bayerische Landesanstalt für Landwirtschaft (2018): Maissilage 2018 and  
  Grassilage 1. Schnitt 2018, Ergebnisse der Futtermitteluntersuchung
  (LKV-Labor Grub).
  <https://www.lfl.bayern.de/mam/cms07/ite/dateien/maissilage_2018.pdf>,
  <https://www.lfl.bayern.de/mam/cms07/ite/dateien/grassilage_2018.pdf>  
- Leisen, E. (2011): Ganzpflanzensilagen in Ökobetrieben. Landwirtschaftskammer  
  Nordrhein-Westfalen.
  <https://www.oekolandbau.nrw.de/sites/default/files/2019-09/110411GPS-form.pdf>  
- Zhao, G. Q., Wei, S. N., Liu, C., Kim, H. J., Kim, J. G. (2021): Effect of  
  harvest dates on β-carotene content and forage quality of rye (Secale
  cereale L.) silage and hay. Journal of Animal Science and Technology 63,
  354–366. <https://doi.org/10.5187/jast.2021.e28>  
- Li, J., Li, Z., Shang, S., Zhao, X., Zhang, W., Zhang, X., Bai, J., Yang,  
  Z., Guo, K. (2023): Effect of additives and moisture on the fermentation
  quality and bacterial community of high moisture ear corn. Frontiers in
  Microbiology 14, 1251946. <https://doi.org/10.3389/fmicb.2023.1251946>  
- Kung, L., Shaver, R.: Interpretation and use of silage fermentation analysis  
  reports. Focus on Forage 3(13), University of Wisconsin Extension.  
- U.S. Food and Drug Administration: Approximate pH of foods and food  
  products (onions).  
- Deutscher Wetterdienst (2021): Klimastatusbericht Deutschland Jahr 2020  
  (mean temperature of the reference period 1991–2020: 9.3 °C).  
