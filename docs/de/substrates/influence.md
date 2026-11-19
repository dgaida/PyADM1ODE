# Einfluss der Werte

Jeder Wert einer Substratdatei wirkt auf eine von vier Größen der Simulation:
wie viel Gas entsteht, wie schnell es entsteht, wie viel Stickstoff in den
Fermenter gelangt und wie gut der Fermenter gepuffert ist. Diese Seite zeigt
zuerst, welcher Wert worauf wirkt, dann eine Beispielrechnung mit Zahlen und
zuletzt die Formeln dahinter.

## Welcher Wert wirkt worauf

| Wert | Wirkt auf | Richtung |
| --- | --- | --- |
| `TS` | Gasmenge, Stickstofffracht, Belastung | Mehr Trockensubstanz je Tonne bringt proportional mehr von allem: Gas, Stickstoff, organische Belastung. |
| `VS` | Gasmenge | Mehr organische Substanz in der Trockensubstanz, entsprechend weniger Asche. |
| `BMP` (oder `D_VS`, `aXI`) | Gasmenge | Legt fest, welcher Anteil der organischen Substanz abbaubar ist. Der Rest verlässt den Fermenter als inertes Material. |
| `NDF` | Geschwindigkeit | Mehr Zellwand verschiebt abbaubares Material vom schnellen in den langsamen Pool. Bei festem `BMP` bleibt das Potenzial gleich, aber im Fermenter wird ein größerer Teil ausgetragen, bevor er abgebaut ist. |
| `ADL` | fast nichts, solange `BMP` gegeben ist | Lignin ist inert. Bei festem `BMP` wird der Abbaugrad so angepasst, dass das Potenzial stimmt; `ADL` verschiebt dann nur die Zusammensetzung des inerten Rests. Mit festem `D_VS` begrenzt `ADL` dagegen, wie viel Zellwand abbaubar ist. |
| `RP` | Stickstoff, pH, Hemmung | Protein setzt beim Abbau Ammonium frei. Mehr Protein hebt Ammonium und pH und kann die Methanbildung hemmen. |
| `RL` | Methangehalt | Fett liefert mehr Methan je Kilogramm und ein methanreicheres Gas als Kohlenhydrate. |
| `NDICP` | Geschwindigkeit des Proteinabbaus | Zellwandgebundenes Protein wird erst mit der Zellwand frei und geht in den langsamen Pool. |
| `NH4` / `Snh4` | Stickstoff, pH, Hemmung | Ammonium, das das Substrat schon mitbringt. Wirkt wie `RP`, aber sofort. |
| `KS43` / `TAC` | Puffer | Hydrogencarbonat des Substrats. Bestimmt zusammen mit Ammonium und Säuren die Ladungsbilanz des Zulaufs. |
| `Sac`, `Spro`, `Sbu`, `Sva` (oder `FFS`) | Säurefracht | Bereits vergorene Säuren. Sie sind sofort verfügbar und belasten den Puffer; Propionat und Butyrat müssen erst zu Acetat umgesetzt werden. |
| `pH` | Ladungsbilanz des Zulaufs | Legt fest, welcher Anteil der Säuren und des Ammoniums dissoziiert vorliegt, und damit die Kationen und Anionen, die das Substrat mitbringt. |
| `Temp` | Wärmebedarf | Hat keinen Einfluss auf die Biologie (der Fermenter hat seine eigene Temperatur), aber auf die Heizung: Jedes Kelvin kälter kostet etwa 1,2 kWh Wärme je Tonne Substrat. |
| `cost` | Kosten | Geht nur in die Kostenrechnung ein. |
| `substrate_class` | Vorgaben | Eine Gülle- oder Mistklasse ordnet Protein und Fett ohne Messung dem langsamen Pool zu, sonst dem schnellen. |

## Beispielrechnung

Ein Fermenter mit 3000 m³ bei 40 °C erhält täglich 30 t Maissilage
(`maize_silage_milk_ripeness`) und 10 t Rindergülle (`cattle_manure`), mit den
Standardwerten der Kinetik. Je Zeile ist **ein** Wert einer Datei geändert,
alles andere bleibt gleich. Angegeben ist der eingeschwungene Zustand nach
400 Tagen. Die Halbwertszeit ist die Zeit, nach der die Methanproduktion die
Hälfte des Anstiegs erreicht hat, wenn die Maismenge sprunghaft um 50 % steigt.

| Geänderter Wert | Methan [Nm³/d] | Änderung | FOS [g/l] | pH | NH₄-N [g/l] | Halbwertszeit [d] |
| --- | --- | --- | --- | --- | --- | --- |
| **Ausgangsfall** | 2731 | – | 0,6 | 7,45 | 2,20 | 5,0 |
| Mais `TS` 28 statt 32,6 % | 2361 | −14 % | 0,2 | 7,42 | 1,97 | 4,3 |
| Mais `TS` 38 % | 3155 | +15 % | 2,3 | 7,43 | 2,51 | 5,6 |
| Mais `VS` 92 statt 94 % | 2669 | −2 % | 0,7 | 7,45 | 2,23 | 5,2 |
| Mais `VS` 96 % | 2794 | +2 % | 0,5 | 7,45 | 2,18 | 4,8 |
| Mais `BMP` 300 statt 338 | 2460 | −10 % | 1,0 | 7,45 | 2,30 | 5,0 |
| Mais `BMP` 370 | 2959 | +8 % | 0,4 | 7,44 | 2,12 | 4,9 |
| Mais `NDF` 40 statt 54,6 % | 2817 | +3 % | 0,5 | 7,45 | 2,17 | 4,1 |
| Mais `NDF` 65 % | 2670 | −2 % | 0,7 | 7,45 | 2,23 | 6,0 |
| Mais `ADL` 2 oder 7 statt 4,6 % | 2731 | 0 % | 0,6 | 7,45 | 2,20 | 5,0 |
| Mais `RP` 5 statt 6,4 % | 2736 | 0 % | 0,1 | 7,36 | 1,65 | 4,2 |
| Mais `RP` 10 % | 2610 | −4 % | 10,7 | 7,22 | 3,65 | 5,7 |
| Mais `RL` 2 statt 3 % | 2721 | 0 % | 0,6 | 7,45 | 2,20 | 5,0 |
| Mais `RL` 5 % | 2752 | +1 % | 0,7 | 7,46 | 2,21 | 4,9 |
| Mais `Sac` 1 statt 4,2 g/l | 2725 | 0 % | 0,5 | 7,44 | 2,19 | 4,9 |
| Mais `Sac` 10 g/l | 2741 | 0 % | 0,9 | 7,47 | 2,22 | 5,1 |
| Mais `pH` 3,6 oder 4,3 statt 3,9 | 2732 / 2729 | 0 % | 0,5 / 0,8 | 7,45 / 7,47 | 2,20 | 4,9 / 5,2 |
| Gülle `TS` 5 statt 10 % | 2666 | −2 % | 0,4 | 7,44 | 2,11 | 4,7 |
| Gülle `BMP` 150 statt 212 | 2693 | −1 % | 0,5 | 7,45 | 2,15 | 4,8 |
| Gülle `NH4` 1,5 statt 2,4 kg N/m³ | 2735 | 0 % | 0,3 | 7,47 | 1,92 | 4,5 |
| Gülle `NH4` 4,5 kg N/m³ | 2711 | −1 % | 2,2 | 7,37 | 2,85 | 5,6 |
| Gülle `KS43` 50 statt 109 mol/m³ | 2735 | 0 % | 0,4 | 7,41 | 2,21 | 4,6 |
| Gülle `KS43` 250 mol/m³ | 2710 | −1 % | 2,2 | 7,49 | 2,20 | 5,8 |
| Gülle `Temp` 5 oder 20 statt 10 °C | 2731 | 0 % | 0,6 | 7,45 | 2,20 | 5,0 |

Was die Tabelle zeigt:

- **Die Gasmenge hängt an `TS` und `BMP`.** Beide wirken fast proportional. Sie sind die Werte, die man für eine reale Anlage zuerst messen sollte.  
- **`NDF` bestimmt die Trägheit.** Zwischen 40 und 65 % Zellwand liegt die Halbwertszeit zwischen 4,1 und 6,0 Tagen; die Gasmenge ändert sich nur um wenige Prozent.  
- **Stickstoff entscheidet über die Stabilität.** Mehr Protein oder Ammonium hebt das Ammonium im Fermenter, und die Säuren steigen. Bei 10 % Rohprotein im Mais kippt der Fermenter (11 g/l Säuren).  
- **Lagerzustand der Silage ist zweitrangig.** pH und Essigsäure der Silage ändern die Gasmenge um weniger als 0,5 %.  
- **Mehr Puffer im Substrat hilft hier nicht.** Mehr Hydrogencarbonat in der Gülle hebt den pH, damit den Anteil an freiem Ammoniak, und die Säuren steigen. Das gilt nur, weil der Ausgangsfall schon nahe an der Ammoniakhemmung liegt.  

!!! warning "Das Modell reagiert empfindlich auf Stickstoff"
    Mit den Standardwerten der Kinetik liegt schon der Ausgangsfall nahe an
    der Ammoniakhemmung. Proteinreiche Rationen (Grassilage, Hühnerkot, viel
    Gülle) führen im Modell schneller zu hohen Säurewerten als in der Praxis,
    wo sich die Biologie an Ammoniak anpasst. Wer solche Rationen rechnet,
    sollte die Hemmkonstante `K_I_nh3` an Anlagendaten kalibrieren.

Die Zahlen gelten für diesen einen Betriebspunkt. Bei anderer Belastung oder
Verweilzeit ändern sich die Beträge, die Richtungen bleiben.

## Wie die Messwerte verwendet werden

**Zellwand statt Rohfaser.** Mit `NDF` und `ADL` werden die Kohlenhydrate nach
der Zellwand aufgeteilt:

```text
langsam   X_PS_ch = d · (NDF − ADL)                 Gerüstkohlenhydrate
schnell   X_PF_ch = VS − RP − RL − NDF              Nicht-Faser-Kohlenhydrate
inert     X_I     = ADL + (1 − d) · (NDF − ADL)     Lignin und nicht abbaubare Faser
          d       = (NDF − VS · (1 − D_VS)) / (NDF − ADL)
```

Die nicht abbaubare organische Trockensubstanz `VS · (1 − D_VS)` wird der
Zellwand zugeordnet, zuerst dem Lignin (Koch et al. 2010). Protein und Fett
gelten als abbaubar. `d` ist auf 0…1 begrenzt: Nicht Abbaubares über die ganze
Zellwand hinaus wird auf den Zellinhalt verteilt, und Lignin bleibt inert, auch
wenn `D_VS` mehr verlangt. Jede Fraktion wird mit ihrem eigenen Faktor in COD
umgerechnet, Lignin mit 1,563 kgCOD/kg. Der Gesamt-COD entspricht dem COD der
Zusammensetzung, und der abgebaute Anteil der organischen Trockensubstanz
entspricht `D_VS`.

**Inertanteil aus einem gemessenen Methanpotenzial.** Mit `BMP` und ohne
`D_VS` und `aXI` wird die Abbaubarkeit angepasst, bis das Modellpotenzial dem
gemessenen entspricht:

```text
Modellpotenzial = abbaubarer COD · 0,350 Nm³ CH4/kgCOD / oTS  =  BMP / BMP_recovery
```

Mit einer Zellwandanalyse wird der Abbaugrad `D_VS` angepasst, sonst der
Inertanteil `aXI`. Beide ändern das Potenzial monoton, das Ergebnis ist also
eindeutig. Ein Potenzial, das die Zusammensetzung nicht erreichen kann, führt zu
einem Fehler. Dasselbe gilt für eine Datei des Standardschemas, die `BMP`, aber
kein `aXI` angibt; im Code übernimmt `fit_degradability_to_bmp(params)` die
Anpassung.

`BMP_recovery` hat die Vorgabe 1: Das Modellpotenzial wird direkt dem gemessenen
gleichgesetzt. Zwei Effekte mit entgegengesetztem Vorzeichen bleiben dabei
unberücksichtigt. Ein Batch-Test findet etwa 5 % weniger Methan wieder, als der
abbaubare COD enthält, weil Biomasse wächst (Weißbach 2009). Das Modell zählt
Kohlenhydrate als Glucose, das liegt etwa 10 % unter dem COD von Stärke und
Cellulose. `BMP_recovery: 0.95` berücksichtigt nur den ersten Effekt; bei der
Getreide-Ganzpflanzensilage und den Zwiebeln erreicht die Zusammensetzung den
Standardertrag dann nicht mehr.

**In der Zellwand gebundenes Protein.** `NDICP` (neutral-detergenzien-unlösliches
Rohprotein, Licitra et al. 1996) ist das Protein, das im NDF-Rückstand bleibt.
Es wird erst frei, wenn die Zellwand zerfällt, und geht in den langsamen Pool;
das übrige Protein geht in den schnellen Pool:

```text
langsam   X_PS_pr = NDICP                     X_PS_ch = d · (NDF − NDICP − ADL)
schnell   X_PF_pr = RP − NDICP                X_PF_ch = VS − RP − RL − (NDF − NDICP)
```

Das gebundene Protein wird mit der `NDF` gewogen und deshalb aus den
Gerüstkohlenhydraten herausgenommen; das ist die Formel der
Nicht-Faser-Kohlenhydrate nach NRC (2001). `fsOTS` / `ffOTS` teilen dann nur
noch das Fett auf. Für Fett gibt es keine solche Messung: Das Detergens löst es
heraus, es gehört zum Zellinhalt.

**Einzelsäuren.** Jede Säure geht in ihren eigenen ADM1-Zustand. Propionat und
Butyrat durchlaufen dann die Acetogenese, die `FFS` (alles Acetat) überspringt.

**Titrierte Säurekapazität.** Eine Titration bis pH 4,3 protoniert auch
Säureanionen. Sie werden abgezogen, nur der Rest zählt als Hydrogencarbonat:

```text
S_hco3 = TAC − Σ S_Säure · (α(pH) − α(4,3))        begrenzt auf ≥ 0
```

Für die Schweinegülle der Anlage Gummersbach bleiben 51 statt 212 mol/m³. `KS43` des
Standardschemas wird weiterhin direkt als Hydrogencarbonat übernommen.

**Filtrat-COD.** `COD_S` über die Säuren und die gelösten Inerte hinaus ist
bereits desintegriertes Material. Es geht in die hydrolysierbaren Pools `X_S_*`
und wird den Desintegrationspools anteilig entnommen.

**Biomasse.** `X_bac` und `X_meth` werden im Verhältnis des Fermenter-Inokulums
auf die sieben ADM1-Gruppen verteilt und den Desintegrationspools entnommen.
`load_gummersbach_library()` gibt Gülle- und Mistklassen die Biomasse des alten
Modells mit (0,25 und 0,0003 kgCOD/m³ bei 10 Tagen Lageralter);
`manure_age=None` schaltet das ab.

**Vergütungsklasse.** `substrate_class` ist Freitext und wird vom Modell nicht
verwendet. Die Bibliothek nutzt die Klassen des EEG 2012 nach der BiomasseV:
`EK I` für Energiepflanzen (Anlage 2), `EK II` für Wirtschaftsdünger (Anlage 3),
`EK 0` für Einsatzstoffe ohne einsatzstoffbezogene Vergütung (Anlage 1). Ein
Finanzmodul kann daran Vergütungssätze, den Maisdeckel oder
Treibhausgasfaktoren knüpfen. Das Feld gibt es in beiden Schemata, es wird in
`SubstrateParams` abgelegt.

## Was weiterhin nicht gemessen ist

- Die Aufteilung des **Fetts** auf den langsamen und den schnellen Pool und die  
  des **Proteins** ohne `NDICP` (`fsOTS` / `ffOTS`). Vorgabe: langsam für Gülle-
  und Mistklassen, sonst schnell. In der Bibliothek tragen nur Maissilage und
  Corn-Cob-Mix ein `NDICP`; für die übrigen Substrate fand sich kein belegter
  Wert.  
- Die **Desintegrationsraten**. Die alte Bibliothek führt eine Rate `kdis` je  
  Substrat; ADM1da hat zwei Pools mit festen Raten, die je Fermenter kalibriert
  werden. `kdis`, `khyd_*` und `km_*` werden gelesen und ignoriert.  

## Unterschiede zum alten Modell

- Aus einem Desintegrationspool mit einer Rate je Substrat werden zwei Pools,  
  aufgeteilt nach der Zellwand.  
- Das alte Modell zieht alle Säureanionen von `TAC` ab, nicht nur den titrierten  
  Anteil.  
- Die Dichte rechnet mit additiven Volumen statt mit dem massengewichteten  
  Mittel.  
- `cost` der alten Bibliothek ist in EUR/m³ angegeben und wird in EUR/t FM  
  umgerechnet.  

!!! note "Qualität der alten Werte"
    Mehrere Werte der alten Bibliothek sind keine Messwerte des beschriebenen
    Substrats. `NDF` / `ADF` / `ADL` von Mais sind Literaturwerte; Grünroggen,
    Zwiebeln, Corn-Cob-Mix und Rindermist tragen dasselbe `NDF` ohne Quelle.
    Der Abbaugrad der Schweinegülle (0,95) hat keine Quelle. `COD_S` ist bei
    fünf Substraten als Annahme beschriftet und bei den übrigen vier ohne
    Quelle. Die Bibliotheksdateien unter `data/substrates/` verwenden diese
    Werte nicht mehr; `gummersbach:<id>` verwendet sie weiterhin.
