# ADM1-Implementierung und Substratmodellierung

Diese Seite beschreibt die technischen Details des in PyADM1ODE umgesetzten
ADM1-Modells und wie landwirtschaftliche Substrate in das Modell integriert
werden.

PyADM1ODE implementiert **ADM1da** (Schlattmann 2011) — eine landwirtschaftliche
Adaption des ursprünglichen ADM1 (Batstone et al. 2002, IWA Task Group). Im
Vergleich zur klassischen Formulierung enthält dieses Modell eine
**Sub-Fraktionierung der Desintegration**, **temperaturabhängige Kinetiken**
sowie **modifizierte Inhibitionskinetiken** für agrarische Co-Vergärung. Die
Umsetzung reproduziert das Referenzverhalten von **SIMBA# biogas 4.2**
(ifak Magdeburg) quantitativ (siehe die Validierungs-Seite). Wo die veröffentlichte
ADM1da-Literatur bei einer konkreten stöchiometrischen oder kinetischen Wahl
mehrdeutig ist, wird die in PyADM1ODE getroffene Konvention auf dieser Seite
explizit angegeben.

## ADM1 als reines ODE-System

Im Gegensatz zum Standard-ADM1, das oft als System differentiell-algebraischer
Gleichungen (DAE) formuliert wird, ist diese Implementierung ein **reines
System gewöhnlicher Differentialgleichungen (ODE)**.

### Hauptmerkmale

1. **Kinetisches Säure-Base-Gleichgewicht**: Die ionisierten Spezies (Acetat-,  
   Propionat-, Butyrat-, Valerat-Ion, $\text{HCO}_3^-$, $\text{NH}_3$) werden
   als dynamische Zustandsvariablen mit einer sehr hohen Reaktionsrate
   $k_{A,B} = 10^8\,\text{m}^3\,\text{kmol}^{-1}\,\text{d}^{-1}$ geführt.
   Dadurch verfolgen sie das thermodynamische Gleichgewicht praktisch direkt,
   ohne dass innerhalb des ODE-Schritts ein algebraischer Solver aufgerufen
   werden muss.  
2. **Kinetischer Gas-Flüssig-Transfer**: $\text{H}_2$, $\text{CH}_4$ und  
   $\text{CO}_2$ werden über $k_L a$-Raten in die Gasphase überführt — keine
   algebraische Henry-Gleichgewichtsbedingung. Freies Ammoniak verlässt den
   Fermenter mit dem Gas in seiner Gleichgewichtskonzentration.  
3. **41 Zustandsvariablen**: Das Modell führt 41 Variablen in fünf Blöcken:  

   | Block | Indizes | Anzahl | Inhalt |
   | --- | --- | --- | --- |
   | Gelöste Komponenten | 0–11 | 12 | $S_{su}, S_{aa}, S_{fa}, S_{va}, S_{bu}, S_{pro}, S_{ac}, S_{h2}, S_{ch4}, S_{IC}, S_{IN}, S_{I}$ |
   | Partikuläre Sub-Fraktionen | 12–21 | 10 | $X_{PS\_ch/pr/li}$ (langsam), $X_{PF\_ch/pr/li}$ (schnell), $X_{S\_ch/pr/li}$ (hydrolisierbar), $X_I$ |
   | Biomasse | 22–28 | 7 | $X_{su}, X_{aa}, X_{fa}, X_{c4}, X_{pro}, X_{ac}, X_{h2}$ |
   | Säure-Base / Ladungsbilanz | 29–36 | 8 | $S_{cat}, S_{an}$, ionisierte Formen $S_{va^-}, S_{bu^-}, S_{pro^-}, S_{ac^-}, S_{HCO_3^-}, S_{NH_3}$ |
   | Gasphase | 37–40 | 4 | $p_{H_2}, p_{CH_4}, p_{CO_2}, p_{tot}$ |

### pH-Wert-Berechnung

Der pH-Wert wird in jedem Auswertungsschritt über die **Ladungsbilanz** und
ein Newton–Raphson-Verfahren bestimmt:

$$
S_{cat} - S_{an} + (S_{NH4} - S_{NH3}) - S_{HCO_3^-} - \frac{S_{ac^-}}{64}  
- \frac{S_{pro^-}}{112} - \frac{S_{bu^-}}{160} - \frac{S_{va^-}}{208}  
+ S_{H^+} - \frac{K_w}{S_{H^+}} = 0  
$$

Da alle Beiträge zur linken Seite Zustandsvariablen sind, ist diese
Ladungsbilanz innerhalb des ODE-Schritts eine **rein punktweise Funktion des
aktuellen Zustands** und liefert in 5–10 Newton-Iterationen $[H^+]$ mit hoher
Genauigkeit. Die schnelle Equilibrierung der ionisierten Spezies über
$k_{A,B}$ stellt sicher, dass der berechnete pH konsistent mit den
thermodynamischen Säure-Base-Konstanten bleibt.

## Sub-Fraktionierung der Desintegration

Der wichtigste Unterschied zum klassischen ADM1 ist die
**zwei-Pool-Desintegration** anstelle der monolithischen Verbundstoff-Variable
$X_c$:

```text
                    Substratzulauf
                         │
       ┌─────────────────┼─────────────────┐
       │                 │                 │
   X_PS_ch/pr/li     X_PF_ch/pr/li      X_I (inert)
   (slow pool)       (fast pool)
       │                 │
       │ k_dis_PS=0.04   │ k_dis_PF=0.4
       │ d⁻¹             │ d⁻¹
       ▼                 ▼
            X_S_ch/pr/li (hydrolisierbar)
                  │
                  │ k_hyd ≈ 4 d⁻¹
                  ▼
              S_su / S_aa / S_fa
```

**Routing der Eingangs-COD** auf die Pools (via `Feedstock._calc_concentrations`):

| Pool | Substratquelle (Weender + Sub-Fraktionierungs-Parameter) |
| --- | --- |
| $X_{PS}$ | Rohfaser (immer langsamer Pool) **plus** der Anteil $f_{sOTS}$ der NFE-Kohlenhydrate, Proteine und Fette |
| $X_{PF}$ | Anteil $f_{fOTS}$ der NFE-Kohlenhydrate, Proteine und Fette |
| $X_S$ | Wird durch Desintegration aus $X_{PS}$ und $X_{PF}$ erzeugt; nicht direkt aus dem Substrat |
| $X_I$ | Anteil $a_{XI}$ der gesamten organischen Roh-COD |
| $S_I$ | Anteil $a_{Si}$ der gesamten organischen Roh-COD (gelöste Inerte, treten direkt im Zulauf auf) |

Damit lässt sich beispielsweise leichtabbaubare Stärke (NFE) gezielt in den
schnellen Pool routen, während Lignocellulose (Rohfaser) automatisch in den
langsamen Pool fällt — ohne Modellumbau.

## Routing der Biomasse-Zerfallsprodukte

Wenn eine Biomasse-Population $X_i$ zerfällt, wird die zugehörige COD in den
**hydrolisierbaren** Pool $X_S$ und in den **inerten** Pool $X_I$ umgeleitet —
**nicht** in den langsamen Desintegrationspool $X_{PS}$. Dies folgt der
Biomasse-Zerfallsstöchiometrie von ADM1da (Schlattmann 2011), bei der alle
sieben Biomasse-Populationen in dieselbe Kombination aus hydrolisierbarem
und inertem Pool zerfallen. Die COD-basierten Routing-Anteile lauten

$$
f_{CH\_XB} = \frac{f_{BM,CH} \cdot M_{Xch}}{M_{XB}}
\approx 0{,}246, \quad
f_{PR\_XB} = \frac{f_{BM,PR} \cdot M_{Xpr}}{M_{XB}}
\approx 0{,}709, \quad
f_{LI\_XB} = \frac{f_{BM,LI} \cdot M_{Xli}}{M_{XB}}
\approx 0{,}045
$$

mit $f_{BM,CH} = 0{,}20$, $f_{BM,PR} = 0{,}70$, $f_{BM,LI} = 0{,}10$
(Massenanteile in der Biomasse) und $M_{Xch} = 0{,}9375$,
$M_{Xpr} = 0{,}7736$, $M_{Xli} = 0{,}3474$ (Masse-COD-Verhältnisse). Ein
Anteil $f_P = 0{,}20$ geht in den Inertpool $X_I$; die restlichen
$(1 - f_P) = 0{,}80$ werden über die obigen COD-Anteile auf die $X_{S,*}$-Pools
verteilt. Die Hydrolyserate $k_{hyd} = 4\,\text{d}^{-1}$ ist 100× schneller als
$k_{dis,PS} = 0{,}04\,\text{d}^{-1}$, sodass zerfallene Biomasse auf der
Hydrolyse-Zeitskala wieder in die Substratkette eintritt.

Der inerte Anteil behält die Zusammensetzung der Biomasse und enthält damit
Protein im selben Verhältnis wie der abbaubare Anteil. Die Biomasse trägt
$N_{XB} = f_{BM,PR} \cdot N_{aa}$ Stickstoff, ihre Zerfallsprodukte
$f_{PR\_XB} \cdot N_{aa}$, mit $N_{aa} = 0{,}00765$ kmol N je kg COD (13,84 %
Stickstoff im Protein). Der Zerfall bindet deshalb etwas Ammonium
($6{,}9 \cdot 10^{-5}$ kmol N je kg COD), statt welches freizusetzen. Bis
Version 0.6.2 hatte der inerte Anteil einen niedrigeren Stickstoffgehalt; der
Zerfall setzte dann Stickstoff frei und ließ rund 2 % mehr Ammonium im
Fermenter.

## Temperaturabhängige Kinetiken

Die ADM1da-Variante korrigiert sämtliche kinetischen Raten gegenüber der
35 °C-Referenz mit einer Arrhenius-θ-Funktion:

$$
k(T) = k(35\,°\text{C}) \cdot \theta^{(T[°\text{C}] - 35)}
$$

mit gruppenspezifischen Exponenten (Werte nach Schlattmann 2011):

| Prozess­gruppe | $\theta_{\exp}$ |
| --- | --- |
| Desintegration & Hydrolyse | 0,024 |
| $X_{su}, X_{aa}, X_{h2}$ (Wachstum & Zerfall) | 0,069 |
| $X_{fa}, X_{c4}, X_{pro}, X_{ac}$ (Wachstum & Zerfall) | 0,055 |
| $K_{I,NH_3,X_{pro}}$ | 0,061 |
| $K_{I,NH_3,X_{ac}}$ | 0,086 |
| $K_{I,H_2,X_{fa}/X_{c4}/X_{pro}}$ | 0,080 |

Die Korrektur wird einmalig beim Anlegen des `ADM1`-Objekts auf alle relevanten
Raten angewandt (`ADMParams.apply_temperature_corrections`) und im internen
`_kinetic`-Dict zwischengespeichert.

## Modifizierte Inhibitionskinetiken

Gegenüber dem Standard-ADM1 weist das ADM1da-Modell folgende Anpassungen auf —
alle in `ADM1.ADM_ODE` umgesetzt und folgen den ADM1da-Aufnahme-Prozessraten
(Schlattmann 2011), wobei die NH₃-Inhibitionsformen aus Siegrist et al. (2002)
übernommen sind:

| Inhibition | Standard-ADM1 | ADM1da (diese Implementierung) |
| --- | --- | --- |
| pH-Inhibition $X_{su}/X_{aa}/X_{fa}/X_{c4}/X_{pro}$ | Hill, $n=1$ | Hill, $n=2$ (steilerer Cut-Off) |
| pH-Inhibition $X_{ac}$ | Hill, $n=1$ | Hill, $n=3$ |
| pH-Inhibition $X_{h2}$ | Hill, $n=1$ | Hill, $n=3$ |
| $\text{NH}_3$-Inhibition $X_{ac}$ | linear in $S_{NH3}$ | quadratischer Hill: $K_I^2/(K_I^2 + S_{NH3}^2)$, T-korrigiert mit $\theta=0{,}086$ |
| $\text{NH}_3$-Inhibition $X_{pro}$ | nicht enthalten | quadratischer Hill mit eigenem $K_{I,nh3,pro}$, T-korrigiert mit $\theta=0{,}061$ |
| Undissoziiertes Propionat $X_{pro}$ | nicht enthalten | $K_{IH,pro}/(K_{IH,pro} + S_{pro} - S_{pro^-})$ |
| Undissoziiertes Acetat $X_{ac}$ | nicht enthalten | $K_{IH,ac}/(K_{IH,ac} + S_{ac} - S_{ac^-})$ |
| $\text{CO}_2$-Limitation $X_{h2}$ | nicht enthalten | quadratische Hill-Sättigung: $S_{CO2}^2/(K_S^2 + S_{CO2}^2)$ mit dem gelösten $S_{CO2} = S_{IC} - S_{HCO_3^-}$ |

!!! note "Anwendungsbereich der Inhibition"
    Die ADM1da-Kinetik in Schlattmann (2011) führt zusätzlich eine Inhibition
    durch Acetat bzw. undissoziierte Säuren auf $X_{fa}$ und $X_{c4}$ auf. Die
    SIMBA#-Referenzimplementierung und PyADM1ODE wenden diese Terme nur auf
    $X_{pro}$ und $X_{ac}$ an, wo sie experimentell gut belegt sind; $X_{fa}$
    und $X_{c4}$ werden nur durch pH, N-Limitation und gelöstes H₂ inhibiert.

Diese Erweiterungen reproduzieren das in landwirtschaftlichen Anlagen typische
Verhalten: schärferer pH-Abfall bei Säureakkumulation, ausgeprägtere
$\text{NH}_3$-Hemmung bei thermophilem Betrieb auf Güllebasis und realistischere
Propionatdynamik.

## Säure-Base-Subsystem

Die Säure-Base-Reaktionen der sechs dissoziierenden Spezies ($\text{NH}_4^+ /
\text{NH}_3$, $\text{CO}_2 / \text{HCO}_3^-$ sowie die vier VFA-Paare) sind als
**kinetische** Reaktionen folgender Form implementiert:

$$
\rho_{A,i} = k_{A,B} \cdot \left( S_{i^-} \cdot S_{H^+} - K_{a,i} \cdot
(S_i - S_{i^-}) \right),
\quad i \in \{ \text{va}, \text{bu}, \text{pro}, \text{ac} \}
$$

mit derselben kinetischen Kopplung für $\text{CO}_2 / \text{HCO}_3^-$ und
$\text{NH}_4^+ / \text{NH}_3$. Ein Paar stellt sich mit der Rate
$k_{A,B} \cdot (S_{H^+} + K_a)$ ein. Mit
$k_{A,B} = 10^8\,\text{m}^3\,\text{kmol}^{-1}\,\text{d}^{-1}$ und pH 7,4
brauchen die VFA-Paare weniger als eine Minute, $\text{CO}_2 / \text{HCO}_3^-$
eine halbe Stunde und $\text{NH}_4^+ / \text{NH}_3$ einige Stunden; das ist
schnell gegenüber der Biologie. Die Säuredissoziationskonstanten werden
gegenüber der 25 °C-Referenz mittels van't Hoff mit den Enthalpien aus
Batstone 2002 T-korrigiert:

$$
K_a(T) = K_a(298\,\text{K}) \cdot
\exp\left(\frac{\Delta H^\circ}{R} \cdot
\left(\frac{1}{298} - \frac{1}{T}\right)\right)
$$

$S_{IC}$ (`S_co2`) und $S_{IN}$ (`S_nh4`) sind **Gesamtmengen**: Hydrogencarbonat
und freies Ammoniak sind Teilmengen davon, wie der Term $S_i - S_{i^-}$ der Rate
zeigt. Eine Säure-Base-Reaktion verschiebt nur zwischen den beiden Formen;
$\rho_A$ steht deshalb allein in der Bilanz der ionisierten Form und nicht in
der Bilanz der Gesamtmenge. So bleiben Kohlenstoff und Stickstoff erhalten; bis
Version 0.6.2 wurde die Rate auch den beiden Gesamtmengen zugeschlagen, was rund
4 % des Ammoniums und 2 % des Gaskohlenstoffs aus der Bilanz nahm.

## Gas-Flüssig-Transfer (Henry-Gesetz mit van't Hoff-Korrektur)

Die drei Gasphasenspezies $\text{H}_2$, $\text{CH}_4$ und $\text{CO}_2$
werden zwischen Flüssig- und Gasphase über

$$
r_{F,gas} = k_L a_F \cdot \left( S_F - \frac{p_F}{K_H(T)\,R\,T} \right)
\cdot \frac{V_{liq}}{V_{gas}}
$$

mit einer van't Hoff-Temperaturkorrektur der Henry-Konstante
(Schlattmann 2011 / Batstone et al. 2002 Enthalpien) transferiert:

$$
H_F(T) = H_F(T_{ref}) \cdot
\exp\left(-\frac{\Delta H^\circ_F}{R} \cdot
\left(\frac{1}{T_{ref}} - \frac{1}{T}\right)\right)
$$

| Gas | $H_{F,25°C}$ [mol/(L·bar)] | $\Delta H^\circ_F$ [J/mol] | $k_L a_F$ [d⁻¹] |
| --- | --- | --- | --- |
| CO₂ | 0,035 | 19 410 | 200 |
| CH₄ | 0,0014 | 14 240 | 200 |
| H₂ | 7,8·10⁻⁴ | 4 180 | 200 |
| NH₃ | 60 | 36 584 | – (siehe unten) |

Vorzeichenkonvention: Die Lösung ist exotherm, daher sinkt $H_F$ mit
steigender Temperatur — bei höherer Betriebstemperatur bleibt weniger Gas
gelöst.

### Ammoniak im Gas

Ammoniak ist über tausendmal löslicher als $\text{CO}_2$; seine Konzentration
im Gas folgt der Flüssigkeit deshalb ohne Verzögerung: Der Gasraum enthält
$S_{NH_3} / (H_{NH_3}\,R\,T)$ kmol N je m³. Dieses Ammoniak verlässt den
Fermenter mit dem Gasstrom und wird $S_{IN}$ und $S_{NH_3}$ entnommen; einen
eigenen Gasphasenzustand gibt es dafür nicht. Bei 42 °C und 115 mg/l freiem
Ammoniakstickstoff enthält das trockene Gas rund 300 ppm, knapp 1 % des
anorganischen Stickstoffs eines Mais-Gülle-Fermenters.

### Druck im Gasraum

Das Gas strömt über einen Auslass mit der Rate
$k_p \cdot (p_{trocken} + p_{H_2O} - p_{ext})$ ab. Der feuchte Gasraum stellt
sich deshalb auf den Druck $p_{ext}$ ein, gegen den er abströmt, und das
trockene Gas hält $p_{ext} - p_{H_2O}$: 0,93 bar bei 42 °C mit dem Standardwert
$p_{ext} = 1{,}013$ bar. Ein Gassystem mit Gegendruck wird über `ADM1.p_ext`
eingestellt.

Der trockene Gasdruck bestimmt das gelöste $\text{CO}_2$ und damit den pH. Bis
Version 0.6.2 wurde der Wasserdampfdruck doppelt abgezogen; das trockene Gas
lag dadurch bei 0,85 bar und das gelöste $\text{CO}_2$ rund 9 % zu niedrig.

### Gemeldete Gasströme

`Q_gas`, `Q_ch4` und `Q_co2` sind **trockenes Gas im Normzustand** (0 °C,
1,01325 bar; DIN 1343), in Nm³/d:

- Die Ströme sind die, die die Gasphasen-ODE aus dem Gasraum entfernt, aufgeteilt nach den trockenen Partialdrücken. Methan, Kohlendioxid und Wasserstoff ergeben zusammen `Q_gas`, und der Methanstrom schließt die CSB-Bilanz.  
- Der Gasraum ist mit Wasserdampf gesättigt. Er trägt zum Druck bei, der den Abstrom treibt, gehört aber nicht zu `Q_gas`; `ADM1.calc_gas()` gibt ihn getrennt als `q_h2o` zurück.  
- Das Bezugsvolumen ist dasselbe wie bei `BMP` (22,414 Nm³ je kmol); Methanertrag und Methanpotenzial sind damit direkt vergleichbar.  
- `Q_ch4 / Q_gas` ist der Methangehalt des trockenen Gases, wie ihn ein Gasanalysator anzeigt.  

Bis Version 0.6.2 galten die Ströme bei 20 °C, und der Wasserdampf war in
`Q_gas` enthalten. Gegenüber diesen Zahlen liegt `Q_gas` rund 7 % niedriger,
der Methangehalt rund 4 Prozentpunkte höher und `Q_ch4` rund 1 % höher
(bei 40 °C).

## Schlammvolumenbilanz und HRT

Da ein erheblicher Anteil der Substrat-COD den Reaktor als Gas verlässt, ist
das Schlammvolumen nicht erhalten. Jede organische Fraktion nimmt je kg COD das
Volumen $v_i = iM_i / \rho_i$ ein (Masse-COD-Verhältnis durch Dichte);
Wasserstoff, Methan und anorganischer Kohlenstoff nehmen keines ein. Der
Schlamm verliert das Nettovolumen dessen, was die Umsetzungen umwandeln:

$$
\frac{dV_S}{dt} = \dot{q}_{S,in} - \dot{q}_{S,out} - \dot{q}_{S,loss},
\quad
\dot{q}_{S,loss} = -V_S \cdot \sum_j \rho_j \sum_i \nu_{ij} \cdot v_i
$$

mit der Rate $\rho_j$ des Prozesses $j$ und seinen stöchiometrischen
Koeffizienten $\nu_{ij}$.

| Fraktion | $iM$ [kg/kg COD] | $\rho$ [kg/m³] |
| --- | --- | --- |
| Kohlenhydrate, Zucker | 0,9375 | 1550 |
| Proteine, Aminosäuren | 0,7736 | 1370 |
| Lipide, Fettsäuren | 0,3474 | 920 |
| Valerat / Butyrat / Propionat / Acetat | 0,490 / 0,550 / 0,661 / 0,9375 | 940 / 960 / 990 / 1050 |
| Biomasse | 0,7638 | 1361 |

Desintegration und die Hydrolyse von Kohlenhydraten und Proteinen erhalten das
Volumen; der größte Teil geht dort verloren, wo Acetat zu Methan wird. Im
Gleichgewicht beträgt der Verlust rund 0,59 Liter je kg COD gebildetem Methan,
das Verhältnis, das SIMBA# für seine Reaktoren ermittelt. Bis Version 0.6.2
wurde der Verlust allein aus den Hydrolyseraten gebildet, was rund 10 % mehr
ergab.

Die hydraulische Verweilzeit folgt einer Verzögerung erster Ordnung
(Schlattmann 2011):

$$
\frac{dHRT}{dt} + HRT \cdot \frac{\dot{q}_{S,in}}{V_S} = 1,
\quad HRT_{ss} = \frac{V_S}{\dot{q}_{S,in}}
$$

Der Ablauf wird über ein Überlaufwehr bei $V_{liq,max}$ mit kleiner Zeit­konstante
$\tau_{out}$ getrieben:

$$
\dot{q}_{S,out} = \max\!\left( 0,\; \frac{V_S - V_{liq,max}}{\tau_{out}} \right)
$$

In der Praxis hält $\tau_{out} = 0{,}05\,\text{d}$ $V_S$ auf ~1 m³ am Sollwert
und reproduziert damit das nahezu instantane Wehrverhalten von SIMBA#.

Bei `dynamic_volume=False` (Default zur Rückwärts­kompatibilität) wird $V_S$
konstant gehalten und $\dot{q}_{S,out} = \dot{q}_{S,in} - \dot{q}_{S,loss}$
direkt aus dem Massenverlustterm berechnet.

## Substratcharakterisierung via Weender-Analyse

Substrate werden über praxisübliche Laborparameter definiert und als
**YAML-Dateien** unter `data/substrates/` gespeichert. Das Standardschema nach
SIMBA hat diese Struktur (die abgelöste `legacy/maize_silage_milk_ripeness.yaml`):

```yaml
name: Maissilage (Milchreife)

TS:     330.0     # kg/m^3 FM   Trockensubstanz
NH4:      0.8     # kg N/m^3 FM Ammoniumstickstoff
BGP:    670.0     # Nm^3/t VS   Biogaspotenzial
BMP:    356.8     # Nm^3 CH4/t VS  Biomethanpotenzial

aXI:      0.1029  # partikulär inerter Anteil der abbaubaren COD
fOTSrf:   0.95    # abbaubarer Anteil der Rohfaser
fsOTS:    0.0     # Anteil abbaubarer VS in den langsamen Pool (XPS)
ffOTS:    1.0     # Anteil abbaubarer VS in den schnellen Pool (XPF)
aSi:      0.0     # gelöst-inerter Anteil der abbaubaren COD

fRF:      0.235   # Rohfaser   (Anteil der TS)
fRP:      0.09    # Rohprotein
fRFe:     0.03    # Rohfett
fRA:      0.10    # Rohasche

Temp:    20.0     # degC
pH:       4.0
KS43:     0.0     # mol/m^3   Säurekapazität bis pH 4,3
FFS:      5.0     # kg HAc/m^3 VFA als Essigsäure-Äquivalent
```

Dieselbe Datei kann alternativ als XML oder TOML geschrieben werden; der
Loader wählt das Format anhand der Endung. Side-by-Side-Beispiele in allen
drei Formaten finden sich in
[`data/substrates/examples/`](https://github.com/dgaida/PyADM1ODE/tree/master/data/substrates/examples).

Ein Substrat kann statt über die COD-Fraktionierung auch über gemessene
Laborwerte beschrieben werden (Zellwandanalyse `NDF` / `ADL`, in der Zellwand
gebundenes Protein `NDICP`, Methanpotenzial `BMP` oder Abbaugrad `D_VS`,
Einzelsäuren, titrierte Säurekapazität). Die mitgelieferten Substratdateien
sind so geschrieben. Siehe
[Messwert-Parameter](../substrates/parameters.md#messwert-parameter).

`load_substrate()` liefert eine `SubstrateParams`-Dataclass; die
`Feedstock`-Klasse berechnet daraus den vollständigen 38-spaltigen ADM1-
Zulaufstrom (37 flüssige Zustandsspalten + Q):

1. **Frischmasse-Dichte** $\rho_{FM}$ aus den Komponentendichten  
   (Mischungsregel auf spezifischem Volumen).  
2. **Organische COD-Konzentrationen** $X_{ch}, X_{pr}, X_{li}$ in  
   $\text{kg COD/m}^3$ über die COD-Umrechnungsfaktoren $M_{Xch}, M_{Xpr},
   M_{Xli}$.  
3. **Routing in die Sub-Fraktionen** $X_{PS}/X_{PF}$ via $f_{sOTS}, f_{fOTS}$  
   und $f_{OTSrf}$ (siehe Tabelle oben).  
4. **Dissoziation** beim Substrat-pH: ionisierte VFA, $\text{HCO}_3^-$,  
   $\text{NH}_3$.  
5. **Ladungsbilanz** für die Ionen, geschlossen über die starken Ionen und  
   nach Vorzeichen aufgeteilt: ein netto-anionisches Material (Silage bei
   ihrem eigenen pH) trägt den Überschuss als $S_{anion}$, ein
   netto-kationisches (Gülle, Festmist, wo $\text{HCO}_3^-$ und die
   VFA-Anionen $\text{NH}_4^+$ übersteigen) als $S_{cation}$ — die starken
   Kationen K⁺, Na⁺, Ca²⁺, Mg²⁺ des Materials. Beide Pools bleiben
   nicht-negativ. In pH, Speziation und Säurekapazität geht nur die Differenz
   $S_{cation} - S_{anion}$ ein, die Chemie entspricht also dem
   ADM1da-Abschluss (Schlattmann 2011: $S_{cation} \equiv 0$, $S_{anion}$
   vorzeichenbehaftet); die Aufteilung hält den Zulauf in der nicht-negativen
   Zustandsbox, auf die Zustandsschätzer begrenzen.

### Volumetrische ADM1da-Konvention für Q

Die Substrat-Volumenströme $Q_i$ werden standardmäßig als **massenäquivalente
Ströme** interpretiert. Intern wird

$$
Q_{actual,i} = Q_{input,i} \cdot \frac{1000}{\rho_{FM,i}}
$$

berechnet. Für flüssige Substrate ($\rho_{FM} \approx 1000$) ist das ein
No-Op; für Maissilage ($\rho_{FM} \approx 1129\,\text{kg/m}^3$) reduziert
sich der tatsächliche Volumenstrom etwa um 11 %. Damit ist Parität zu
ADM1da-Referenzergebnissen gegeben. Über `simba_q_convention=False` lässt
sich diese Skalierung abschalten, falls $Q$ direkt als tatsächliches Reaktor­
volumen interpretiert werden soll.

### Co-Vergärung: gewichtete Mischung

Bei Multi-Substrat-Zuläufen werden die Konzentrationen volumetrisch
flussgewichtet gemischt (`Feedstock._blended_concentrations`). Die Anzahl
gleichzeitig beschickter Substrate ist nicht begrenzt — `Feedstock`
akzeptiert eine beliebig lange Liste von XML-IDs.

## Messgrößen

PyADM1ODE liefert die standardmäßigen ADM1da-Messgrößen (Schlattmann 2011)
für die Anlagenüberwachung.

### Flüchtige Fettsäuren (VFA)

Der aggregierte VFA-Wert in g HAc/L ist das COD-gewichtete Essigsäure-Äquivalent
der vier VFA-Spezies:

$$
\text{VFA} = M_{HAc} \cdot \sum_i \frac{S_i}{\text{COD}_{MOL,i}},
\quad i \in \{ \text{va}, \text{bu}, \text{pro}, \text{ac} \}
$$

mit $M_{HAc} = 60\,\text{g/mol}$ und den COD-pro-mol-Verhältnissen
pro VFA (va = 208, bu = 160, pro = 112, ac = 64 g COD/mol).

### Säurekapazität (TAC)

Die pH-5-Titrationskapazität (TAC, in g CaCO₃/L) ist

$$
\text{TAC} = 50 \cdot \Bigl[
\left(F_{NH_3} - K_{A,NH_4} \cdot
\frac{F_{NH_4} + F_{NH_3}}{10^{-pH_5} + K_{A,NH_4}}\right)  
+ \left(F_{HCO_3} - K_{A,CO_2} \cdot  
\frac{F_{CO_2} + F_{HCO_3}}{10^{-pH_5} + K_{A,CO_2}}\right)  
+ \sum_{i = \text{va},\text{bu},\text{pro},\text{ac}} (\cdots)  
+ F_{AN} - F_{CAT} \Bigr]  
$$

!!! warning "Konvention: der Ionenterm ist die Starkionen-Differenz"
    Der Term $F_{AN} - F_{CAT}$ zieht die Starkionen-Differenz
    $S_{cation} - S_{anion}$ ab, die die Substratcharakterisierung aus der
    Ladungsbilanz schließt (nach Vorzeichen aufgeteilt, beide Pools
    nicht-negativ — dieselbe Chemie wie der ADM1da-Abschluss mit
    $S_{cation} \equiv 0$). Ein Zustand, dessen Starkionen-Differenz nicht zu
    seinem Schwachsäure-Puffer passt, meldet dadurch zu wenig Säurekapazität —
    im Extremfall negativ. Der vorinokulierte Startzustand wird genauso
    geschlossen, und Behälter ohne eigenen Substratzulauf (Nachgärer, Lager)
    werden mit dem Ammonium des ADM1da-Referenzinokulums initialisiert statt mit
    null Stickstoff. Ergibt die Formel dennoch $\text{TAC} \le 0$, liefert
    PyADM1ODE `NaN` und warnt einmalig — das verhindert FOS/TAC-Werte, die durch
    eine gegen null laufende Bezugsgröße ins Unendliche laufen.

!!! note "Vorfaktor 50 vs 100"
    Die TAC-Formel in Schlattmann (2011) verwendet die Molmasse von CaCO₃
    (100 kg/kmol) als Vorfaktor. Der physikalisch korrekte Wert ist das
    **Äquivalentgewicht 50 kg/keq**, da ein Mol CaCO₃ zwei H⁺-Äquivalente
    säureneutralisierende Kapazität trägt — die übliche Alkalinitäts­
    konvention. PyADM1ODE verwendet den Äquivalentgewichts­vorfaktor, was
    mit dem Referenz-Output des SIMBA#-Reaktormoduls übereinstimmt.

## Referenzen

Die Implementierung basiert auf folgenden Primärquellen (übereinstimmend mit
der Literaturliste des PyADM1ODE–SIMBA#-Validierungsberichts):

- **Batstone, D. J., Keller, J., Angelidaki, I., Kalyuzhnyi, S. V.,  
  Pavlostathis, S. G., Rozzi, A., Sanders, W. T. M., Siegrist, H.,
  Vavilin, V. A. (2002).** *Anaerobic Digestion Model No. 1 (ADM1).*
  IWA Scientific and Technical Report No. 13. IWA Publishing, London.  
- **Henneberg, W., Strohmann, F. (1864).** *Beiträge zur Begründung einer  
  rationellen Fütterung der Wiederkäuer.* Schwetschke, Braunschweig.
  (Ursprung der Weender Futtermittelanalyse, die der Substrat­charakterisierung
  zugrunde liegt.)  
- **Schlattmann, M. (2011).** *Weiterentwicklung des Anaerobic Digestion  
  Model No. 1 (ADM1) zur Anwendung auf landwirtschaftliche Substrate.*
  Dissertation, Technische Universität München.  
- **Siegrist, H., Vogt, D., Garcia-Heras, J. L., Gujer, W. (2002).**  
  Mathematical Model for Meso- and Thermophilic Anaerobic Sewage Sludge
  Digestion. *Environmental Science & Technology* **36**(5), 1113–1123.

## Technische Umsetzung

Das gesamte Modell ist pure Python:

| Modul | Zweck |
| --- | --- |
| `pyadm1.core.adm1` | `ADM1`-Klasse mit `ADM_ODE`, Newton–Raphson-pH, Gasausgang |
| `pyadm1.core.adm_params` | Stöchiometrie, Kinetiken, Inhibition, $\theta$-Korrekturen |
| `pyadm1.core.solver` | Wrapper um `scipy.integrate.solve_ivp` (BDF, adaptive) |
| `pyadm1.substrates.feedstock` | XML-Parser, Sub-Fraktionierungs-Routing, Mischung |
| `pyadm1.components.biological.digester` | Komponenten-Wrapper mit Schlammvolumen und HRT-Logik |

Damit läuft die Simulation in jeder Standard-Python-Umgebung und auch in
Containern, Web-Notebooks (Colab) und CI-Pipelines problemlos.

### Ratenfamilien-Haken

`ADM1.rate_hook` erlaubt einem externen Modell, die kinetischen Raten zu
skalieren, ohne Bilanzen oder Stöchiometrie anzufassen. Standardmäßig ist er
`None`, dann bleibt `ADM_ODE` unverändert. Ist er gesetzt, wird er einmal je
Auswertung der rechten Seite als `rate_hook(state, S_H, q_ad)` aufgerufen und muss
13 Faktoren in der Reihenfolge von `pyadm1.core.adm1.RATE_FAMILIES` liefern:

```text
dis_PS, dis_PF, hyd_ch, hyd_pr, hyd_li, su, aa, fa, c4, pro, ac, h2, dec
```

Jeder Faktor multipliziert alle Raten seiner Familie (zum Beispiel skaliert `c4`
die Valerat- und die Butyrat-Aufnahme, `dec` alle sieben Zerfallsraten). Das
Torch-Backend wendet dieselben Faktoren an: `adm1da_rhs_torch(x, p, rate_factors)`
nimmt einen `(…, 13)`-Tensor und bleibt darin differenzierbar, und
`make_scipy_rhs` ruft den Haken wie der NumPy-Pfad auf. Typische Anwendungen sind
Sensitivitätsstudien der Ratenfamilien und hybride Modelle, die die Kinetik
einer Anlage korrigieren.
