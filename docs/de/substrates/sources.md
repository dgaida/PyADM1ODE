# Herkunft der Werte

Diese Seite belegt, woher die Werte der 13 Bibliotheksdateien stammen und wo
ihre Grenzen liegen.

Jeder Wert einer Bibliotheksdatei nennt seine Quelle in einem Kommentar. Fünf
Quellen sind kombiniert, weil keine ein Substrat vollständig abdeckt:

| Quelle | Liefert | Parameter |
| --- | --- | --- |
| [Phyllis-Datenbank](https://phyllis.nl) (TNO) | Zusammensetzung der Trockensubstanz, als Median einer dokumentierten Probenauswahl | `VS`, `RP`, `RL`, `NDF`, `ADF`, `ADL`; `TS`, wo die Proben frisch analysiert wurden |
| BiomasseV 2012, Anlagen 1–3 | Standard-Methanertrag je Tonne Frischmasse, umgerechnet mit der KTBL-Standard-Trockensubstanz | `BMP`, `substrate_class`; `TS`, wo Phyllis keine frische Probe hat |
| Veröffentlichte Analysen des Materials, wie es gefüttert wird (siehe [Lagerzustand](#lagerzustand)) | Lagerzustand | `pH`, Ammonium, Säuren, Hydrogencarbonat oder Säurekapazität, Temperatur |
| Literatur | In der Zellwand gebundenes Protein | `NDICP` von Maissilage und Corn-Cob-Mix |
| Abgelöste Dateien; Labor der Gummersbacher Anlage für ihre zwei Dateien | Preis; Lagerzustand der Anlagendateien | `cost`; `pH`, Säuren, Säurekapazität von `maize_silage_gummersbach` und `swine_manure_gummersbach` |

Phyllis ist eine Datenbank für die thermochemische Konversion. Sie enthält
Asche-, Elementar- und Zellwandanalysen, aber **keinen** pH-Wert, kein
Ammonium, keine Fettsäuren, keine Säurekapazität und kein Methanpotenzial;
ihre einzigen pH-Werte gehören zu Biokohle und Holzasche. Diese Parameter
können nicht aus ihr stammen.

## Phyllis-Proben

| Material | Proben | Verwendet für |
| --- | --- | --- |
| Mais, Ganzpflanze, fünf Hybriden, frisch analysiert, Faser nach van Soest | #2603–#2607 | `TS`, `VS`, Zellwand von `maize_silage_milk_ripeness` |
| Maissilage, Elementaranalyse | #2795 | Protein (N × 6,25) von `maize_silage_milk_ripeness` |
| Weizen, Ganzpflanze, frisch analysiert, Faser nach van Soest | #2648–#2653 | `TS`, `VS`, Protein, Zellwand von `wheat_whole_plant_silage` und `cereal_gps_silage` |
| Wiesengras | #613–#617, #1699, #3193, #3197, #3286 | `VS`, Protein von `grass_silage` und `green_rye_silage` |
| Gras mit Detergenzienfaser | #1038, #2377 | Zellwand von `grass_silage` und `green_rye_silage` |
| Maiskorn; entkörnter Mais | #1033; #1973 | Kornanteil von `corn_cob_mix` |
| Maisspindel | #979, #1035, #1037, #1102, #1240, #1454, #2068, #2302, #2791 | Spindelanteil von `corn_cob_mix` |
| Frischer Rindermist; Rinderkot | #1882, #2782, #1528; #1391 | `VS`, Protein; Fett beider Rinderdünger |
| Schweinemist und -gülle | #1366, #1715, #2246, #3069 | `TS` (#1366), `VS`, Protein von `swine_manure` |
| Hühnermist, Geflügeleinstreu | 25 Proben; #2380; #957 | `VS`, Protein; Zellwand; Fett von `chicken_manure_dry` |

Proben bleiben außen vor, wenn sie das Material nicht so beschreiben, wie es
gefüttert wird: kompostierter oder gealterter Mist, Gärrest, Asche, Kohle,
Pellets, Mist mit Erde oder Sandeinstreu. Faserwerte bleiben außen vor, wenn das
Lignin ein Klason-Lignin ist, die Methode unbekannt ist oder die Bestandteile
mehr als 100 % der Trockensubstanz ergeben. Cellulose, Hemicellulose und Lignin
einer van-Soest-Analyse ergeben `NDF` (ihre Summe), `ADF` (Cellulose + Lignin)
und `ADL` (Lignin).

## Lagerzustand

Der Lagerzustand beschreibt das Substrat, wie es in den Fermenter gelangt:
pH-Wert, Ammonium, flüchtige Fettsäuren, Hydrogencarbonat oder Säurekapazität
und Temperatur. Er stammt aus veröffentlichten Analysen des Materials,
recherchiert im November 2026:

| Datei | pH | NH4-N [kg/m³] | Säuren [g/l] | Puffer | Temp. [°C] | Quelle |
| --- | --- | --- | --- | --- | --- | --- |
| `cattle_manure` | 7,4 | 2,42 | 6,2 / 1,2 / 0,6 / 0,1 | `KS43` 109 mol/m³ aus TIC | 10 | Overmeyer et al. 2020; pH: Biogashandbuch Bayern |
| `swine_manure` | 7,34 | 2,88 | 7,0 / 1,6 / 1,2 / 0,4 | `KS43` 112 mol/m³ aus TIC | 20 | Overmeyer et al. 2020 |
| `cattle_manure_solid` | 7,85 | 1,59 | `FFS` 6,7 | `TAC` 250 mmol/l | 9,3 | Mahato et al. 2020 |
| `chicken_manure_dry` | 7,7 | 3,04 | `FFS` 0 (Annahme) | `TAC` 170 mmol/l | 9,3 | Zhang et al. 2019, auf 45 % TS skaliert |
| `maize_silage_milk_ripeness` | 3,9 | 0,17 | 4,2 / 0 / 0,07 | keiner | 9,3 | LfL Maissilage 2018 (n = 92) |
| `grass_silage` | 4,6 | 0,38 | 6,3 / 0 / 1,9 | keiner | 9,3 | LfL Grassilage 2018 (n = 82) |
| `wheat_whole_plant_silage`, `cereal_gps_silage` | 4,0 | 0,38 | 7,0 / 0 / 0,4 | keiner | 9,3 | Leisen 2011, LUFA NRW (n = 20) |
| `green_rye_silage` | 4,45 | 0,79 | 1,0 / 0,2 / 0 | keiner | 9,3 | Zhao et al. 2021 (Proxy) |
| `corn_cob_mix` | 4,42 | 1,24 | 1,5 / 0,3 / 0 | keiner | 9,3 | Li et al. 2023 (Proxy) |
| `onion_waste` | 5,5 | 0 (Annahme) | 0 | keiner | 9,3 | FDA pH of foods |

Säuren sind Essig- / Propion- / Butter- (/ Valerian-)säure; `FFS` ist die
Summe als Essigsäure. Die Konventionen hinter der Tabelle:

- **Gülle**: Die frische Milchvieh- und Mastschweinegülle der Universität  
  Bonn (Overmeyer et al. 2020, Woche 0) liefert Ammonium, Einzelsäuren und
  den gesamten anorganischen Kohlenstoff je einer in sich konsistenten Probe.
  `KS43` ist der Hydrogencarbonat-Anteil dieses anorganischen Kohlenstoffs
  beim pH der Gülle (pKs 6,35). Der pH der Rindergülle ist der des
  Biogashandbuchs Bayern (7,42 für Frischgülle mit 10 % TS); Erhebungen nennen
  7,2 (Petersen et al. 2016, n = 39) und 7,74 (Overmeyer et al. 2020).  
- **Festmist**: Eine gemessene Gesamtalkalinität steht als `TAC`; der Lader  
  zieht die mittitrierten Säureanionen ab. Die Werte des Hühnerkots sind bei
  33 % TS gemessen und auf die 45 % TS der Datei skaliert. Das Ammonium des
  Geflügelkots ist das gemessene Ammonium; die Düngerichtwerte (LfL
  Basisdaten) zählen per Konvention 45 % des Stickstoffs einschließlich
  Harnsäure als Ammonium und werden nicht verwendet.  
- **Silagen**: Die Labore berichten die Gärsäuren in g/kg Trockenmasse und  
  den Ammoniakstickstoff als Anteil am Gesamtstickstoff. Beides wird mit `TS`
  und `RP` der Datei umgerechnet (N = RP / 6,25; 1 kg Frischmasse als 1 l).
  Milchsäure ist das Hauptgärprodukt (Mais 45, Gras 59 g/kg TS), hat aber
  keinen Zustand im ADM1: Sie wird **nicht** in die Datei geschrieben, ihr COD
  bleibt in den Kohlenhydratpools. Eine Silage unter pH 4,3 hat per Definition
  keine Säurekapazität. Propionsäure weisen die deutschen Labore nicht
  getrennt aus; wo sie nicht gemessen ist, steht null.  
- **Temperatur**: Im Freien gelagerte Substrate erhalten das Jahresmittel der  
  Lufttemperatur in Deutschland 1991–2020 (DWD, 9,3 °C). Gülle im Rinderstall
  hat 5,5–12,3 °C (Petersen et al. 2016; 10 °C in der Datei), im Schweinestall
  etwa 20 °C (Ma et al. 2023).  
- **Proxys**: Für Grünroggensilage und Corn-Cob-Mix aus Deutschland war kein  
  offener Datensatz zu finden. Grünroggen trägt die Gärqualität von
  koreanischem Futterroggen im Ährenschieben (Zhao et al. 2021), CCM die von
  unbehandeltem Maiskolbenschrot mit 63 % TS (Li et al. 2023), die in den
  typischen Bereichen von Kung & Shaver liegt. Beide Dateien sagen das in
  ihren Kommentaren.  

## Worauf jede Datei beruht

- **`maize_silage_milk_ripeness`**: Phyllis. Die fünf Hybriden enthalten  
  51–73 % Zellwand und 8–25 % Nichtstruktur-Kohlenhydrate, eine frühe,
  stärkearme Ernte. Heutige teigreife Silage enthält etwa 40 % Zellwand.  
- **`wheat_whole_plant_silage`, `cereal_gps_silage`**: Phyllis. Weizen ist das  
  einzige Ganzpflanzengetreide der Datenbank. Beide Dateien teilen sich deshalb
  seine Zusammensetzung der Trockensubstanz und unterscheiden sich nur in
  Fett und Preis.  
- **`grass_silage`**: Phyllis für Asche, Protein und Zellwand. `TS` ist der  
  KTBL-Standardwert, weil die Grasproben getrocknet waren.  
- **`corn_cob_mix`**: Phyllis führt Korn und Spindel getrennt. Die Datei mischt  
  90 % Korn und 10 % Spindel; der Anteil ist eine Annahme.  
- **`cattle_manure`, `cattle_manure_solid`, `swine_manure`**: Phyllis für  
  Asche, Protein und Fett. Die Faserwerte der Mistproben sind unbrauchbar (eine
  ergibt mit ihrer Asche 127 %, den anderen fehlt die Hemicellulose). Diese
  Dateien bleiben deshalb beim Rohfaserweg mit der Rohfaser der abgelösten
  Datei.  
- **`chicken_manure_dry`**: Phyllis, einschließlich einer van-Soest-Analyse.  
  Das Protein ist N × 6,25 der getrockneten Probe und enthält damit die
  Harnsäure.  
- **`green_rye_silage`**: keine Phyllis-Probe. Die Zusammensetzung der  
  Trockensubstanz ist die von Gras als nächstliegendem Material; die
  Gärqualität ist die von koreanischem Futterroggen im Ährenschieben. Die
  schwächste Datei der Bibliothek.  
- **`onion_waste`**: Phyllis führt nur Zwiebelschalen und Ascheanalysen. Die  
  Zusammensetzung der ganzen Zwiebel stammt aus der Lebensmitteldatenbank des
  USDA; die Gesamtballaststoffe stehen für die Zellwand. Frische Zwiebeln
  enthalten weder Ammonium noch Gärsäuren; der pH ist der roher Zwiebeln.  
- **`maize_silage_gummersbach`, `swine_manure_gummersbach`**: anlagenspezifische  
  Dateien. Die Laborwerte der Anlage bleiben erhalten; nur der Abbaugrad, der
  keine Quelle hatte, ist durch den Standard-Methanertrag ersetzt.  

!!! warning "Grenzen dieser Werte"
    Stickstoff aus einer getrockneten Mistprobe erfasst das beim Trocknen
    entwichene Ammoniak nicht; das Protein daraus ist eine Schätzung. Die
    Methanerträge sind die Richtwerte hinter der BiomasseV 2012; das KTBL hat
    seine Richtwerte 2021 überarbeitet, diese Ausgabe lag hier nicht vor. Die
    Umrechnung des Ertrags auf die organische Trockensubstanz nutzt die
    KTBL-Standard-Trockensubstanz des jeweiligen Substrats. Der Lagerzustand
    jeder Gülle beruht auf einer frischen Probe eines Betriebs; die Fettsäuren
    des Hühnertrockenkots und das Ammonium der Zwiebeln stehen mangels Quelle
    auf null; Grünroggen und Corn-Cob-Mix tragen Proxys aus dem Ausland.

## Quellen

- Koch, K., Lübken, M., Gehring, T., Wichern, M., Horn, H. (2010): Biogas from  
  grass silage – Measurements and modeling with ADM1. Bioresource Technology
  101, 8158–8165.  
- Licitra, G., Hernandez, T. M., Van Soest, P. J. (1996): Standardization of  
  procedures for nitrogen fractionation of ruminant feeds. Animal Feed Science
  and Technology 57, 347–358.  
- NRC (2001): Nutrient Requirements of Dairy Cattle, 7th revised edition.  
  National Academy Press, Washington, D.C.  
- Weißbach, F. (2009): Gasbildungspotenzial aus der fermentierbaren organischen  
  Trockensubstanz, zitiert nach Liebetrau, J.: Efficiency of biogas production
  (Vortragsfolien).  
- Rossow, H.: Interpreting Forage Quality from the Cows' Perspective.  
  University of California Cooperative Extension (Vortragsfolien; `NDICP` von
  Maissilage und Maiskorn).  
- Biomasseverordnung in der Fassung vom 1. Januar 2012, Anlagen 1–3.  
- Phyllis2, database for (treated) biomass, algae, feedstocks for biogas  
  production and biochar. TNO, <https://phyllis.nl>.  

Lagerzustand:

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
- Bayerische Landesanstalt für Landwirtschaft (2018): Maissilage 2018 und  
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
  products (Zwiebeln).  
- Deutscher Wetterdienst (2021): Klimastatusbericht Deutschland Jahr 2020  
  (Mitteltemperatur der Referenzperiode 1991–2020: 9,3 °C).  
