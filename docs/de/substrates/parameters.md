# Parameter & Wertebereiche

Eine Substratdatei enthält vier Gruppen von Werten. Diese Seite erklärt, was
jeder Wert bedeutet, wie er gemessen wird und in welchem Bereich er bei
landwirtschaftlichen Substraten üblicherweise liegt. Was eine Änderung in der
Simulation bewirkt, steht auf der Seite [Einfluss der Werte](influence.md).

## Die Werte im Überblick

### Trockensubstanz und Zusammensetzung

| Wert | Einheit | Bedeutung | Messung |
| --- | --- | --- | --- |
| `TS` | % der Frischmasse | Trockensubstanz: was nach dem Trocknen bei 105 °C übrig bleibt. Der Rest ist Wasser. | Trockenschrank |
| `VS` | % der TS | Organische Trockensubstanz (oTS): der Teil der TS, der bei 550 °C verbrennt. Der Rest ist Asche. | Glühverlust |
| `RP` | % der TS | Rohprotein: Stickstoff × 6,25. | Kjeldahl |
| `RL` | % der TS | Rohfett. | Etherextraktion |
| `RF` | % der TS | Rohfaser nach Weender. Nur nötig, wenn `NDF` und `ADL` fehlen. | Weender-Analyse |
| `NDF` | % der TS | Neutral-Detergenzien-Faser: die ganze Zellwand (Cellulose, Hemicellulose, Lignin). | van Soest |
| `ADF` | % der TS | Säure-Detergenzien-Faser: Cellulose und Lignin. Wird abgelegt, aber nicht verwendet. | van Soest |
| `ADL` | % der TS | Säure-Detergenzien-Lignin: der nicht abbaubare Teil der Zellwand. | van Soest |
| `NDICP` | % der TS | Protein, das in der Zellwand gebunden ist. Braucht `NDF` und `ADL`. | Stickstoff im NDF-Rückstand |

Was von der organischen Trockensubstanz nach Protein, Fett und Faser übrig
bleibt, zählt das Modell als leicht abbaubare Kohlenhydrate (Stärke, Zucker).

### Abbaubarkeit

Mindestens einer dieser drei Werte muss angegeben sein. Sind mehrere angegeben, gilt
die Reihenfolge `D_VS`, `aXI`, `BMP`.

| Wert | Einheit | Bedeutung | Messung |
| --- | --- | --- | --- |
| `BMP` | Nm³ CH₄ je t oTS | Methanpotenzial: das Methan, das bei vollständigem Abbau entsteht. Das Modell passt den abbaubaren Anteil so an, dass sein Potenzial diesem Wert entspricht. | Batch-Gärtest (VDI 4630) |
| `D_VS` | 0…1 | Abbaugrad: der Anteil der oTS, der abbaubar ist. | Verdaulichkeit, Gärtest |
| `aXI` | 0…1 | Inerter Anteil des partikulären COD (Standardschema). | nicht direkt messbar |
| `BMP_recovery` | 0…1 | Anteil des abbaubaren COD, den der Gärtest als Methan wiederfindet. Vorgabe 1. | – |

### Lagerzustand

Der Zustand des Substrats, wie es in den Fermenter gelangt.

| Wert | Einheit | Bedeutung |
| --- | --- | --- |
| `pH` | – | pH-Wert des Substrats. |
| `NH4` | kg N/m³ | Ammoniumstickstoff. Alternativ `Snh4` in g NH₄/l. |
| `Sac`, `Spro`, `Sbu`, `Sva` | g/l | Essig-, Propion-, Butter- und Valeriansäure. Alternativ `FFS` als Summe in Essigsäure-Äquivalenten. |
| `KS43` | mol/m³ | Hydrogencarbonat. |
| `TAC` | mmol/l | Titrierte Säurekapazität bis pH 4,3. Anders als `KS43` enthält sie die mittitrierten Säureanionen; der Lader zieht sie ab. |
| `Temp` | °C | Temperatur des Substrats im Lager. |

Milchsäure, die wichtigste Gärsäure einer Silage, hat im Modell keinen eigenen
Zustand. Sie wird nicht eingetragen; ihr Energiegehalt bleibt in den
Kohlenhydraten.

### Sonstiges

| Wert | Einheit | Bedeutung |
| --- | --- | --- |
| `cost` | EUR je t Frischmasse | Preis frei Anlage; siehe [Substratkosten](#substratkosten). |
| `substrate_class` | Text | Vergütungsklasse, zum Beispiel `"Rindergülle (EK II)"`. Eine Gülle- oder Mistklasse wählt die Vorgaben für Wirtschaftsdünger. |
| `COD_S`, `SIin` | g COD/l | COD des Filtrats und gelöster inerter COD. Optional. |
| `X_bac`, `X_meth` | kg COD/m³ | Lebende Biomasse im Substrat (Gülle, Mist). Optional. |

## Realistische Wertebereiche

Die Bereiche unten stammen aus veröffentlichten Tabellenwerken und dienen der
Plausibilitätsprüfung eigener Werte. Sie sind Spannen über viele Betriebe und
Erntejahre; ein Wert außerhalb ist nicht falsch, sollte aber einen Grund haben.

!!! warning "Der Lader prüft Widerspruchsfreiheit, nicht Plausibilität"
    Eine Substratdatei mit Werten außerhalb der Bereiche unten wird ohne
    Meldung geladen. Abgelehnt wird nur, was nicht stimmen kann: ein falsch
    geschriebener Parametername, `TS` oder `VS` außerhalb von 0–100, ein pH
    außerhalb von 0–14, negative Konzentrationen, `RP + RL + NDF` über `VS`
    und Faserwerte, die nicht ineinander liegen (`ADL ≤ ADF ≤ NDF`). Auf eines
    muss man weiterhin selbst achten: Alle Gehalte stehen in Prozent, nicht
    als Bruch (32,6, nicht 0,326).

### Trockensubstanz, organische Substanz, Methanertrag

Spanne und, in Klammern, Mittelwert nach dem Leitfaden Biogas (FNR 2016,
Tab. 4.1–4.4 und 4.9). CCM und Zwiebeln führt der Leitfaden nicht; ihre Werte
sind Einzelwerte der Datenbank „Biogasausbeute" der LfL Bayern, der Methanertrag
ist dort aus Biogasertrag und Methangehalt berechnet.

| Substrat | `TS` [% FM] | `VS` [% TS] | `BMP` [Nm³ CH₄/t oTS] | Wert der Bibliothek (`TS` / `VS` / `BMP`) |
| --- | --- | --- | --- | --- |
| Maissilage | 28–35 (33) | 85–98 (95) | 234–364 (340) | 32,6 / 94,0 / 338 |
| Getreide-Ganzpflanzensilage | 30–35 (33) | 92–98 (95) | 290–350 (329) | 36,8 / 93,3 / 329 |
| Grassilage | 25–50 (35) | 70–95 (90) | 300–338 (310) | 35,0 / 90,5 / 317 |
| Grünroggensilage | (25) | (90) | (324) | 25,0 / 90,5 / 320 |
| Corn-Cob-Mix | 60–65 | 98 | etwa 350–370 | 65,0 / 98,5 / 380 |
| Rindergülle | 6–11 (10) | 75–82 (80) | 110–275 (210) | 10,0 / 84,1 / 212 |
| Schweinegülle | 4–7 (6) | 75–86 (80) | 180–360 (250) | 7,9 / 84,8 / 250 |
| Rinderfestmist | 20–25 (25) | 68–76 (80) | 130–330 (250) | 25,0 / 84,1 / 249 |
| Geflügelmist | (40) | (75) | 200–360 (280) | 45,0 / 78,0 / 273 |
| Zwiebeln | 16 | 96 | etwa 360 | 10,9 / 96,8 / 379 |

Beim Rinderfestmist liegt der gedruckte Mittelwert der organischen Substanz
außerhalb der gedruckten Spanne; die Tabelle gibt den Leitfaden wieder. Der
Methangehalt des Biogases liegt nach der LfL-Datenbank bei 52–55 % für
Energiepflanzen und 55–65 % für Wirtschaftsdünger.

### Futtermittelanalyse

Kleinster und größter Wert der Datenblätter von Feedipedia (INRAE, CIRAD, AFZ,
FAO), in Prozent der Trockensubstanz. Für Grünroggensilage die Tabellenwerte
der Gruber Tabelle (LfL 2024) vom Ährenschieben bis nach der Blüte.

| Substrat | `RP` | `RL` | `RF` | `NDF` | `ADF` | `ADL` | Asche (100 − `VS`) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Maissilage, 30–35 % TS | 4,7–9,3 | 1,3–4,0 | 15–26 | 38–55 | 19–33 | 1,9–5,1 | 2,1–5,4 |
| Maissilage, 25–30 % TS | 4,9–9,8 | 1,0–3,9 | 17–27 | 38–63 | 21–35 | 2,0–7,0 | 2,3–5,7 |
| Getreide-Ganzpflanzensilage (Gerste) | 6,6–12,4 | 1,7–3,1 | 32–35 | 38–56 | 23–36 | 2,5–7,1 | 4,7–17,3 |
| Grassilage (Wiesenlieschgras) | 9,4–17,2 | – | 28–35 | 47–64 | 28–42 | 3,6–3,8 | 5,7–11,2 |
| Grünroggensilage | 12–15 | 3,7–4,5 | 26–34 | 53–65 | 31–38 | – | – |
| Maiskolbensilage (nächst an CCM) | 7,5–10,9 | 3,7–4,5 | 2,8–7,9 | 12–25 | 4,5–12 | 1,3–2,0 | 1,6–3,0 |

Mit steigender Reife sinkt bei Mais die Zellwand und steigt die Stärke: Die
Stärke liegt bei 25–30 % TS zwischen 11 und 37 % der Trockensubstanz, bei
35–40 % TS zwischen 21 und 42 %. Die Bibliotheksdatei
`maize_silage_milk_ripeness` liegt mit 54,6 % `NDF` am oberen Rand.

### Gärqualität von Silagen

Typische Bereiche gut vergorener Silagen nach Kung & Shaver (2001), in Prozent
der Trockensubstanz.

| Wert | Maissilage (30–40 % TS) | Grassilage (30–35 % TS) |
| --- | --- | --- |
| pH | 3,7–4,2 | 4,3–4,7 |
| Milchsäure | 4–7 | 6–10 |
| Essigsäure | 1–3 | 1–3 |
| Propionsäure | < 0,1 | < 0,1 |
| Buttersäure | 0 | 0,5–1,0 |
| Ammoniak-N [% des Rohproteins] | 5–7 | 8–12 |

Die Datei erwartet Säuren in g/l Frischmasse. Die Umrechnung:

```text
Säure [g/l] ≈ Säure [% der TS] · TS [% FM] / 10
```

1–3 % Essigsäure in einer Maissilage mit 33 % TS sind also 3,3–10 g/l (`Sac`).
Mehr als 0,5 % Buttersäure zeigt eine Fehlgärung an.

### Lagerzustand von Gülle und Mist

| Wert | Rindergülle | Schweinegülle | Quelle |
| --- | --- | --- | --- |
| pH | 6,3–7,9 (7,3) | 6,5–7,9 (7,1) | Pizzul et al. 2018 |
| Ammoniumstickstoff [kg/t] | 0,6–2,7 (2,0) | 1,5–5,2 (2,8) | Pizzul et al. 2018 |
| Gesamtstickstoff [kg/t] | 1,5–5,1 (3,9) | 1,7–8,9 (4,6) | Pizzul et al. 2018 |
| `NH4` nach Düngerichtwert [kg N/m³] | 1,6–2,1 | 3,0–3,4 (Mast), 2,3–2,8 (Zucht) | LfL Basisdaten 2024 |

Für Festmist nennen die Düngerichtwerte der LfL 0,4 kg NH₄-N je Tonne
(Rindermist, 18,5–23 % TS) und 7–9 kg je Tonne (Hühnermist und -kot,
52–59 % TS). Diese Werte gelten zum Zeitpunkt der Ausbringung, nach Stall- und
Lagerverlusten.

Für die Säurekapazität (`KS43`, `TAC`) und die Fettsäuren von Rohgülle fand sich
keine Quelle mit einer Spanne. Die Werte der Bibliothek (Hydrogencarbonat
109–112 mol/m³, Fettsäuren 8–10 g/l) beruhen auf je einer frischen Probe; siehe
[Herkunft der Werte](sources.md#lagerzustand).

### Quellen der Wertebereiche

- Fachagentur Nachwachsende Rohstoffe (Hrsg.) (2016): Leitfaden Biogas – Von der Gewinnung zur Nutzung, 7. Auflage. Tab. 4.1–4.4, 4.9. <https://www.fnr.de/fileadmin/Mediathek/208/Leitfaden_Biogas_web_V01.pdf>  
- Strobl, M., Keymer, U. (2016): Biogasausbeute mobil. Bayerische Landesanstalt für Landwirtschaft. <https://www.lfl.bayern.de/iba/energie/049711/>  
- Feedipedia – Animal Feed Resources Information System (INRAE, CIRAD, AFZ, FAO): Datenblätter Maize silage, Ear maize silage, Barley forage, Timothy grass silage. <https://www.feedipedia.org>  
- Bayerische Landesanstalt für Landwirtschaft (2024): Gruber Tabelle zur Fütterung der Milchkühe, Zuchtrinder, Schafe, Ziegen, 48. Auflage. <https://www.lfl.bayern.de/mam/cms07/publikationen/daten/informationen/gruber_tabelle_milchvieh-2024_lfl-information.pdf>  
- Kung, L., Shaver, R. (2001): Interpretation and use of silage fermentation analysis reports. Focus on Forage 3(13), University of Wisconsin Extension.  
- Pizzul, L., Rodhe, L., Burzynska, I. et al. (2018): Titration, buffer capacity and acid consumption of animal slurries in Baltic Sea Region countries. Baltic Slurry Acidification, RISE.  
- Bayerische Landesanstalt für Landwirtschaft (2024): Basisdaten Düngung und Nährstoffflüsse, Tabelle 5a. <https://www.lfl.bayern.de/mam/cms07/iab/dateien/bas25_tabelle_5a_20241125_neu.pdf>  

## Messwert-Parameter

Die COD-Fraktionierung oben (`aXI`, `fOTSrf`, `fsOTS`, `ffOTS`, `aSi`) folgt
SIMBA und lässt sich nicht direkt messen. Ein Substrat kann stattdessen über
Laborwerte beschrieben werden. Die Bibliotheksdateien sind so geschrieben; Namen
und Einheiten sind die der alten Gummersbach-Substratbibliothek.

```python
from pyadm1 import Feedstock
from pyadm1.substrates import load_gummersbach_library, load_substrate

library = load_gummersbach_library()                    # mitgelieferte alte Bibliothek, 10 Substrate
feedstock = Feedstock(["gummersbach:maize", "gummersbach:swinemanure"])   # über die ID
maize = load_substrate("data/substrates/examples/maize_silage_measured.yaml")
```

Eine Substratdatei wählt das Schema mit `schema: measured` (YAML, TOML) oder
`<substrate schema="measured">` (XML). Die ältere Schreibweise
`schema: gummersbach` wird weiter akzeptiert.

```yaml
name: Maissilage (Messwert-Parameter)
schema: measured
substrate_class: "Mais (GPS) (EK I)"
TS:   31.97    # % FM
VS:   96.25    # % TS
RP:    8.69    # % TS
RL:    3.68    # % TS
NDF:  43.64    # % TS
ADL:   2.15    # % TS
NDICP: 1.3     # % TS
BMP:  338      # Nm³ CH4/t oTS, legt den Inertanteil fest
pH:    3.93
Sac:   1.18    # g/l
Snh4:  0.66    # g/l
TAC:   6.0     # mmol/l
```

| Parameter | Einheit | Bedeutung | Ersetzt |
| --- | --- | --- | --- |
| `TS`, `VS` | % FM, % TS | Trockensubstanz und organische Trockensubstanz | `TS` [kg/t], `fRA` |
| `RF`, `RP`, `RL` | % TS | Weender-Analyse; `RF` ist mit `NDF` und `ADL` optional | `fRF`, `fRP`, `fRFe` |
| `NDF`, `ADL` | % TS | Zellwand und Lignin nach van Soest | `fOTSrf`, `fsOTS` / `ffOTS` für Kohlenhydrate |
| `ADF` | % TS | Säure-Detergenzien-Faser; abgelegt, nicht verwendet | — |
| `NDICP` | % TS | In der Zellwand gebundenes Protein; braucht `NDF` und `ADL` | `fsOTS` / `ffOTS` für Protein |
| `D_VS` | – | Abbaugrad der organischen Trockensubstanz | `aXI`, `aSi` |
| `BMP` | Nm³ CH₄/t oTS | Methanpotenzial eines Batch-Tests; wirkt ohne `D_VS` und `aXI` | `aXI` oder `D_VS` |
| `BMP_recovery` | – | Anteil des abbaubaren COD, den der Batch-Test als Methan wiederfindet, Vorgabe 1 | — |
| `Sac`, `Spro`, `Sbu`, `Sva` | g/l | Einzelne flüchtige Fettsäuren | `FFS` |
| `TAC` | mmol/l | Titrierte Säurekapazität bis pH 4,3 | `KS43` |
| `Snh4` | g/l | Ammonium, als NH₄ | `NH4` [kg N/m³] |
| `T` | °C | Lagertemperatur | `Temp` |
| `COD_S` | gCOD/l | COD des Filtrats | — |
| `SIin` | gCOD/l | Gelöster inerter COD | `aSi` |
| `X_bac`, `X_meth` | kgCOD/m³ | Lebende Bakterien und Methanogene (Gülle, Mist) | — |
| `substrate_class` | Text | Vergütungsklasse; eine Gülle- oder Mistklasse wählt die Vorgaben für Wirtschaftsdünger | — |

`TS`, `VS`, `RP`, `RL` und `pH` sind Pflicht, alles andere ist optional. Der
Inertanteil kommt aus `D_VS`, sonst aus `aXI`, sonst aus `BMP`; einer der drei
muss angegeben sein. Ein falsch geschriebener Parametername führt zu einem
Fehler. Die Schlüssel des Standardschemas (`NH4`, `KS43`, `FFS`, `Temp`,
`fOTSrf`, `aSi`, `cost`) werden in ihren eigenen Einheiten ebenfalls akzeptiert.

## Substratcharakterisierung

Das Standardschema nach SIMBA beschreibt ein Substrat über diesen Parametersatz.
Die Bibliotheksdateien verwenden stattdessen die [Messwert-Parameter](#messwert-parameter),
aus denen derselbe Satz abgeleitet wird:

- **Rohzusammensetzung** — `TS`, `NH4`, Biogas- / Biomethanpotenzial `BGP`, `BMP`.  
- **Weender-Analyse** — Rohfaser-, Rohprotein-, Rohfett- und Asche-Anteile der TS (`fRF`, `fRP`, `fRFe`, `fRA`).  
- **COD-Fraktionierung** — partikulär- und löslich-inerter COD-Anteil (`aXI`, `aSi`), abbaubarer Anteil der Rohfaser (`fOTSrf`), Aufteilung in langsame/schnelle Desintegrationspools (`fsOTS`, `ffOTS`).  
- **Physikalisch-chemischer Zustand** — Substrattemperatur, `pH`, Säurekapazität bis pH 4,3 (`KS43`), VFA als Essigsäure-Äquivalent (`FFS`).  

Komponentendichten, Masse-zu-COD-Umrechnungsfaktoren, Methanpotenziale und
die Säure-Base-Gleichgewichtskonstanten sind als Modelldefaults in
`SubstrateParams` hinterlegt und müssen pro Substrat nur dann angegeben
werden, wenn eine Messung das Überschreiben rechtfertigt.

Die Seite [ADM1-Implementierung](../user_guide/adm1_implementation.md) beschreibt, wie aus
der Charakterisierung der 38-spaltige ADM1-Zulaufstrom erzeugt wird.

## Dateiformate

Jedes Substrat kann als YAML (kanonisch), XML oder TOML definiert werden.
Der Loader wählt anhand der Dateiendung das passende Format aus und liefert
in jedem Fall dasselbe `SubstrateParams`-Objekt; in
[`data/substrates/examples/`](https://github.com/dgaida/PyADM1ODE/tree/master/data/substrates/examples)
liegen Side-by-Side-Beispiele desselben Substrats in allen drei Formaten.

```python
from pyadm1.substrates import SubstrateRegistry, load_substrate

# Per ID — erkennt jedes unterstützte Format in data/substrates/
reg = SubstrateRegistry()
maize = reg.get("maize_silage_milk_ripeness")

# Per expliziten Pfad — die Endung bestimmt den Loader
swine = load_substrate("data/substrates/swine_manure.yaml")
```

## Substratkosten

Jedes Substrat kann einen Preis `cost` in EUR pro Tonne Frischmasse frei Anlage
tragen. Er ist optional und standardmäßig 0. Er wird in der Substratdatei oder
pro Feedstock gesetzt:

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

Alle Substrate der Bibliothek tragen einen Richtpreis frei Anlage (deutscher
Markt, 2022–2025). Energiepflanzen liegen bei etwa 105–125 EUR pro Tonne
Trockenmasse, ihr Frischmassepreis folgt also dem TS-Gehalt; Gülle trägt nur
die Transportkosten.

| Substrat                               | `cost` [EUR/t FM] |
|----------------------------------------|-------------------|
| Maissilage                             | 40                |
| Getreide- / Weizen-GPS                 | 36 / 35           |
| Grassilage                             | 33                |
| Grünroggensilage                       | 22                |
| Corn-Cob-Mix (CCM)                     | 125               |
| Hühnertrockenkot                       | 30                |
| Rinderfestmist                         | 8                 |
| Zwiebelabfälle                         | 5                 |
| Rinder- / Schweinegülle                | 2                 |

Die regionale und jährliche Streuung ist groß (allein Silomais 30–50 EUR/t FM);
ersetze den Wert durch den Vertragspreis der Anlage.

```python
feedstock.set_substrate_costs({"maize_silage_milk_ripeness": 40.0, "swine_manure": 2.0})
feedstock.substrate_costs        # {'maize_silage_milk_ripeness': 40.0, 'swine_manure': 2.0}
```

Die Kosten eines Betriebs lassen sich auf zwei Wegen berechnen.

**Aus beliebigen Fütterungsdaten** — eingestellte Raten, ein Fütterungsprotokoll,
ein Durchflusssensor oder eine Schätzung. Übergeben wird entweder ein Satz
Raten, der `dt` Tage gilt, oder eine Zeitreihe mit einer Zeile pro Intervall:

```python
feedstock.calc_substrate_costs([10, 5], dt=30.0)          # 30 Tage mit konstanten Raten
feedstock.calc_substrate_costs(measured_feed, dt=1 / 24)  # stündliche Sensorwerte
# {'total': 12300.0, 'by_substrate': {...}, 'mass_t': {...}}
```

**Aus der Simulation** — die Anlage summiert in jedem Schritt, was ihre
Fermenter tatsächlich erhalten haben. Mit einem Dosierer ist das der dosierte
Strom (einschließlich leerem Lager oder Blockade), nicht die eingestellte Rate:

```python
plant.simulate(duration=90)
plant.get_substrate_costs()
# {'total': ..., 'by_substrate': {...}, 'mass_t': {...}, 'specific_cost': ...}  # EUR/t FM
```

Der Vergleich beider zeigt, wie weit die aufgezeichnete Fütterung von der
simulierten abweicht. Die Raten haben die Einheit von `Q_substrates` — Tonnen
pro Tag mit dem Standard `simba_q_convention=True`, sonst m³/d (umgerechnet über
die Substratdichte).
