# Beispiel-Substrate

PyADM1ODE bringt 13 Substratdateien unter
[`data/substrates/`](https://github.com/dgaida/PyADM1ODE/tree/master/data/substrates)
mit. Jede beschreibt typisches Material einer Substratart und wird über ihren
Dateinamen ohne Endung gewählt:

```python
from pyadm1 import Feedstock
from pyadm1.substrates import SubstrateRegistry

SubstrateRegistry().available()      # alle IDs der Bibliothek
feedstock = Feedstock(["maize_silage_milk_ripeness", "cattle_manure"])
```

## Übersicht

`Feedstock()` ohne Argumente lädt alle 13 Substrate in der Reihenfolge der
Tabelle; `Q-Idx` ist dann die Position im `Q_substrates`-Array.

| Q-Idx | Substrat-ID | Vergütungsklasse | TS [% FM] | oTS [% TS] | BMP [Nm³ CH₄/t oTS] | CH₄ [m³/t FM] | Abbaubare oTS | Langsamer Pool [%] | Aufteilung der Kohlenhydrate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | `maize_silage_milk_ripeness` | Mais (GPS) (EK I) | 32,6 | 94,0 | 338 | 104 | 0,82 | 46 | Zellwand |
| 1 | `cattle_manure` | Rindergülle (EK II) | 10,0 | 84,1 | 212 | 18 | 0,49 | 100 | Rohfaser |
| 2 | `swine_manure` | Schweinegülle (EK II) | 7,9 | 84,8 | 250 | 17 | 0,51 | 100 | Rohfaser |
| 3 | `corn_cob_mix` | Corn-Cob-Mix (CCM) (EK I) | 65,0 | 98,5 | 380 | 243 | 0,93 | 9 | Zellwand |
| 4 | `grass_silage` | Gras/Ackergras (EK I) | 35,0 | 90,5 | 317 | 100 | 0,69 | 29 | Zellwand |
| 5 | `green_rye_silage` | Grünroggen (GPS) (EK I) | 25,0 | 90,5 | 320 | 72 | 0,77 | 38 | Zellwand |
| 6 | `cereal_gps_silage` | Getreide (GPS) (EK I) | 36,8 | 93,3 | 329 | 113 | 0,85 | 49 | Zellwand |
| 7 | `onion_waste` | Gemüse (aussortiert) (EK 0) | 10,9 | 96,8 | 379 | 40 | 0,96 | 12 | Zellwand |
| 8 | `maize_silage_gummersbach` | Mais (GPS) (EK I) | 32,0 | 96,2 | 338 | 104 | 0,82 | 30 | Zellwand |
| 9 | `cattle_manure_solid` | Rinderfestmist (EK II) | 25,0 | 84,1 | 249 | 52 | 0,68 | 100 | Rohfaser |
| 10 | `chicken_manure_dry` | Geflügelmist, Geflügeltrockenkot (EK II) | 45,0 | 78,0 | 273 | 96 | 0,59 | 87 | Zellwand |
| 11 | `swine_manure_gummersbach` | Schweinegülle (EK II) | 6,1 | 71,6 | 250 | 11 | 0,32 | 100 | Rohfaser |
| 12 | `wheat_whole_plant_silage` | Getreide (GPS) (EK I) | 36,8 | 93,3 | 329 | 113 | 0,84 | 47 | Zellwand |

So liest man die Spalten:

- **TS, oTS, BMP** stehen in der Datei. `CH₄ [m³/t FM]` ist daraus berechnet: das Methan, das eine Tonne Frischmasse bei vollständigem Abbau liefert.  
- **Abbaubare oTS** ist der Anteil der organischen Trockensubstanz, den das Modell abbauen kann. Es ist der Abbaugrad `D_VS` (Zellwandweg) oder der abbaubare Anteil des organischen COD `1 − aXI − aSi` (Rohfaserweg).  
- **Langsamer Pool** ist der Anteil des abbaubaren partikulären Materials, der langsam zerfällt. Je höher er ist, desto träger folgt die Gasproduktion einer Fütterungsänderung.  
- **Aufteilung der Kohlenhydrate** nennt, welche Faseranalyse die Datei nutzt: die Zellwand nach van Soest (`NDF`, `ADL`) oder die Rohfaser nach Weender (`RF`).  

## Welche Datei wofür

### Energiepflanzen (EK I)

| Datei | Material | Hinweis |
| --- | --- | --- |
| `maize_silage_milk_ripeness` | Maissilage, früh geerntet (Milchreife) | Viel Zellwand (55 % der TS), wenig Stärke. Heutige teigreife Silage enthält etwa 40 % Zellwand und baut schneller ab. |
| `maize_silage_gummersbach` | Maissilage der Anlage Gummersbach | Laborwerte einer realen Anlage; passt zu `swine_manure_gummersbach`. |
| `grass_silage` | Grassilage, Wiesen- und Ackergras | Proteinreich; bringt viel Stickstoff in den Fermenter. |
| `green_rye_silage` | Grünroggen-Ganzpflanzensilage | Die schwächste Datei: Zusammensetzung von Gras übernommen, Gärqualität aus dem Ausland. |
| `cereal_gps_silage`, `wheat_whole_plant_silage` | Getreide-Ganzpflanzensilage | Gleiche Zusammensetzung (Weizen); sie unterscheiden sich nur in Fett und Preis. |
| `corn_cob_mix` | Corn-Cob-Mix (CCM) | Energiereichstes Substrat, fast alles schnell abbaubar. Die Mischung aus 90 % Korn und 10 % Spindel ist eine Annahme. |

### Wirtschaftsdünger (EK II)

| Datei | Material | Hinweis |
| --- | --- | --- |
| `cattle_manure` | Rindergülle, 10 % TS | Wenig Gas je Tonne, aber Puffer und Ammonium. |
| `swine_manure` | Schweinegülle, 8 % TS | Mehr Ammonium als Rindergülle. |
| `swine_manure_gummersbach` | Schweinegülle der Anlage Gummersbach | Dünner (6 % TS) und mit hohem Aschegehalt. |
| `cattle_manure_solid` | Rinderfestmist, 25 % TS | Stapelbar; der Puffer ist als titrierte Säurekapazität angegeben. |
| `chicken_manure_dry` | Hühnertrockenkot, 45 % TS | Sehr stickstoffreich; das Protein enthält die Harnsäure. |

### Reststoffe (EK 0)

| Datei | Material | Hinweis |
| --- | --- | --- |
| `onion_waste` | Aussortierte Zwiebeln | Fast vollständig und schnell abbaubar, kein Ammonium, kein Puffer. |

Aus welchen Quellen die Werte jeder Datei stammen, steht auf der Seite
[Herkunft der Werte](sources.md).

## Weitere mitgelieferte Dateien

### Formatbeispiele

Unter
[`data/substrates/examples/`](https://github.com/dgaida/PyADM1ODE/tree/master/data/substrates/examples)
liegt dasselbe Substrat (`cattle_manure`) als YAML, XML und TOML, dazu
`maize_silage_measured.yaml` als Vorlage für eine Datei mit Messwerten. Diese
Dateien gehören nicht zur Bibliothek und werden über ihren Pfad geladen.

### Alte Gummersbach-Bibliothek

Die Substratbibliothek des Vorgängermodells mit zehn Substraten (`ccm`,
`cowmanure`, `futterkalk`, `gps`, `grass`, `greenrye`, `maize`, `onions`,
`swinemanure`, `wheat`) ist als eine XML-Datei enthalten und wird über das
Präfix `gummersbach:` gewählt:

```python
from pyadm1 import Feedstock
from pyadm1.substrates import load_gummersbach_library

library = load_gummersbach_library()
feedstock = Feedstock(["gummersbach:maize", "gummersbach:swinemanure"])
```

Mehrere ihrer Werte sind keine Messwerte des beschriebenen Substrats; siehe
[Unterschiede zum alten Modell](influence.md#unterschiede-zum-alten-modell).

### Abgelöste Dateien

Die bisherigen Dateien der Bibliothek liegen weiter in
[`data/substrates/legacy/`](https://github.com/dgaida/PyADM1ODE/tree/master/data/substrates/legacy).
Sie gehören nicht zum Registry und werden über das Präfix `legacy:` gewählt:

```python
from pyadm1 import Feedstock

old = Feedstock(["legacy:maize_silage_milk_ripeness", "legacy:cattle_manure"])
```

Ergebnisse einer früheren Version lassen sich so reproduzieren. Die
sichtbarsten Änderungen der neuen Dateien: Rinder- und Schweinegülle enthalten
45 bis 65 % mehr abbaubaren COD (die abgelösten ifak-Werte liegen deutlich
unter den KTBL-Erträgen), Maissilage gibt 46 statt 23 % ihres abbaubaren COD in
den langsamen Pool, und das Hydrogencarbonat der Gummersbacher Schweinegülle
sinkt von 212 auf 51 mol/m³, weil ihre Säurekapazität jetzt als Titrationswert
gelesen wird.
