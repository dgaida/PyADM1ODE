# ADM1-Implementierung und Substratmodellierung

Diese Seite beschreibt die technischen Details des in PyADM1ODE umgesetzten
ADM1-Modells und wie landwirtschaftliche Substrate in das Modell integriert
werden.

PyADM1ODE implementiert **ADM1da** (Schlattmann 2011) — eine landwirtschaftliche
Adaption des ursprünglichen ADM1 (Batstone et al. 2002, IWA Task Group). Im
Vergleich zur klassischen Formulierung enthält dieses Modell eine
**Sub-Fraktionierung der Desintegration**, **temperaturabhängige Kinetiken**
sowie **modifizierte Inhibitionskinetiken** für agrarische Co-Vergärung. Die
Umsetzung folgt der Referenz `adm1da.asm` von SIMBA# biogas 4.2 bitgenau;
Abweichungen vom SIMBA#-Tutorial (PDF), wenn das PDF und die `.asm`-Datei
unterschiedliche Werte angeben, werden auf dieser Seite explizit gekennzeichnet.

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
2. **Kinetischer Gas-Flüssig-Transfer**: $\text{H}_2$, $\text{CH}_4$,  
   $\text{CO}_2$ und $\text{NH}_3$ werden über $k_L a$-Raten in die Gasphase
   überführt — keine algebraische Henry-Gleichgewichtsbedingung.  
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
**nicht** in den langsamen Desintegrationspool $X_{PS}$. Das deckt sich mit
den asm-Prozessen p13–p19 in `adm1da.asm`. Die COD-basierten Routing-Anteile
lauten

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
alle in `ADM1.ADM_ODE` umgesetzt und aus den asm-Prozessen p5–p12 abgeleitet:

| Inhibition | Standard-ADM1 | ADM1da (diese Implementierung) |
| --- | --- | --- |
| pH-Inhibition $X_{fa}/X_{c4}/X_{pro}$ | Hill, $n=1$ | Hill, $n=2$ (steilerer Cut-Off) |
| pH-Inhibition $X_{ac}$ | Hill, $n=1$ | Hill, $n=3$ |
| pH-Inhibition $X_{h2}$ | Hill, $n=1$ | Hill, $n=3$ (asm-Prozess p12) |
| N-Limitation | $S_{NH4}$ allein | $S_{IN} = S_{NH4} + S_{NH3}$ |
| $\text{NH}_3$-Inhibition $X_{ac}$ | linear in $S_{NH3}$ | quadratischer Hill: $K_I^2/(K_I^2 + S_{NH3}^2)$, T-korrigiert mit $\theta=0{,}086$ |
| $\text{NH}_3$-Inhibition $X_{pro}$ | nicht enthalten | quadratischer Hill mit eigenem $K_{I,nh3,pro}$, T-korrigiert mit $\theta=0{,}061$ |
| Undissoziiertes Propionat $X_{pro}$ | nicht enthalten | $K_{IH,pro}/(K_{IH,pro} + S_{pro} - S_{pro^-})$ |
| Undissoziiertes Acetat $X_{ac}$ | nicht enthalten | $K_{IH,ac}/(K_{IH,ac} + S_{ac} - S_{ac^-})$ |
| $\text{CO}_2$-Limitation $X_{h2}$ | nicht enthalten | quadratische Hill-Sättigung: $S_{CO2}^2/(K_S^2 + S_{CO2}^2)$ (asm p12) |

!!! note "Tutorial vs. Implementierung"
    Das SIMBA#-Tutorial (PDF, §7.1–§7.3) listet zusätzlich eine Inhibition
    durch undissoziierte Säuren auf $X_{fa}$ und $X_{c4}$ auf. Die
    tatsächlichen asm-Prozesse p7/p8/p9 wenden diese Inhibition **nicht** an —
    nur pH-, N-Limitation und H₂-Inhibition. PyADM1ODE folgt der `.asm`-Datei,
    nicht dem PDF, da die `.asm`-Datei das ist, was SIMBA# tatsächlich
    ausführt.

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
$\text{NH}_4^+ / \text{NH}_3$. Die Reaktionsratenkonstante
$k_{A,B} = 10^8\,\text{d}^{-1}$ sorgt für eine Sub-Sekunden-Equilibrierung. Die
Säuredissoziationskonstanten werden gegenüber der 35 °C-Referenz mittels
van't Hoff mit den Enthalpien aus Batstone 2002 T-korrigiert:

$$
K_a(T) = K_a(298\,\text{K}) \cdot
\exp\left(\frac{\Delta H^\circ}{R} \cdot
\left(\frac{1}{298} - \frac{1}{T}\right)\right)
$$

## Gas-Flüssig-Transfer (Henry-Gesetz mit van't Hoff-Korrektur)

Die vier Gasphasenspezies $\text{H}_2$, $\text{CH}_4$, $\text{CO}_2$ und
$\text{NH}_3$ werden zwischen Flüssig- und Gasphase über

$$
r_{F,gas} = k_L a_F \cdot \left( S_F - \frac{p_F}{K_H(T)\,R\,T} \right)
\cdot \frac{V_{liq}}{V_{gas}}
$$

mit einer temperaturabhängigen Henry-Konstante gemäß SIMBA#-Tutorial §9 /
asm-Parametern transferiert:

$$
H_F(T) = H_F(T_{ref}) \cdot
\exp\left(-\frac{\Delta H^\circ_F}{R} \cdot
\left(\frac{1}{T_{ref}} - \frac{1}{T}\right)\right)
$$

| Gas | $H_{F,35°C}$ [mol/(L·bar)] | $\Delta H^\circ_F$ [J/mol] | $k_L a_F$ [d⁻¹] |
| --- | --- | --- | --- |
| CO₂ | 0,0271 | 19 410 | 200 |
| CH₄ | 0,00116 | 14 240 | 200 |
| H₂ | 7,38·10⁻⁴ | 4 180 | 200 |
| NH₃ | 60 (bei 25 °C Referenz) | 36 584 | 200 |

Vorzeichenkonvention: Die Lösung ist exotherm, daher sinkt $H_F$ mit
steigender Temperatur — bei höherer Betriebstemperatur bleibt weniger Gas
gelöst.

## Schlammvolumenbilanz und HRT

Da ein erheblicher Anteil der Substrat-COD den Reaktor als Gas verlässt, ist
das Schlammvolumen nicht erhalten. Die dynamische Schlammvolumenbilanz folgt
dem SIMBA#-Tutorial §1.3–§1.4 (Ansatz 2):

$$
\frac{dV_S}{dt} = \dot{q}_{S,in} - \dot{q}_{S,out} - \dot{q}_{S,loss},
\quad
\dot{q}_{S,loss} = V_S \cdot \sum_i r_{hyd,i} \cdot \frac{iM_i}{\rho_i}
$$

Die hydraulische Verweilzeit folgt einer Verzögerung erster Ordnung
(Tutorial §8.11):

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
**XML-Dateien** unter `data/substrates/adm1da/` gespeichert. Eine Beispielstruktur:

```xml
<substrate name="Maissilage (Milchreife)">
  <param name="TS"     value="320.0"/>
  <param name="NH4"    value="0.0"/>
  <param name="pH"     value="3.9"/>
  <param name="fRF"    value="0.220"/>   <!-- Rohfaser -->
  <param name="fRP"    value="0.080"/>   <!-- Rohprotein -->
  <param name="fRFe"   value="0.030"/>   <!-- Rohfett -->
  <param name="fRA"    value="0.045"/>   <!-- Rohasche -->
  <param name="aXI"    value="0.10"/>    <!-- partikulär inerter COD-Anteil -->
  <param name="aSi"    value="0.02"/>    <!-- löslich inerter COD-Anteil -->
  <param name="fOTSrf" value="0.40"/>    <!-- abbaubarer Anteil der Rohfaser -->
  <param name="fsOTS"  value="0.30"/>    <!-- NFE/PR/LI in den langsamen Pool -->
  <param name="ffOTS"  value="0.70"/>    <!-- NFE/PR/LI in den schnellen Pool -->
  <param name="FFS"    value="0.0"/>     <!-- VFA als Essigsäure-Äquivalent -->
  <param name="KS43"   value="0.0"/>     <!-- Säurekapazität bis pH 4,3 -->
  <!-- ... -->
</substrate>
```

`load_substrate_xml()` liefert eine `SubstrateParams`-Dataclass; die
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
5. **Ladungsbilanz** für die Ionen. Konventionsgemäß wird $S_{cation}$ für  
   jeden Substrattyp auf null gesetzt (gemäß SIMBA#-Tutorial §5.7.3–§5.7.5),
   und $S_{anion}$ ergibt sich aus der Ladungsbilanz — wobei es für netto-
   kationische Substrate auch negativ werden darf.

### Volumetrische ADM1da-Konvention für Q

Die Substrat-Volumenströme $Q_i$ werden standardmäßig als **massenäquivalente
Ströme** interpretiert. Intern wird

$$
Q_{actual,i} = Q_{input,i} \cdot \frac{1000}{\rho_{FM,i}}
$$

berechnet. Für flüssige Substrate ($\rho_{FM} \approx 1000$) ist das ein
No-Op; für Maissilage ($\rho_{FM} \approx 1134\,\text{kg/m}^3$) reduziert
sich der tatsächliche Volumenstrom etwa um 12 %. Damit ist Parität zu
ADM1da-Referenzergebnissen gegeben. Über `simba_q_convention=False` lässt
sich diese Skalierung abschalten, falls $Q$ direkt als tatsächliches Reaktor­
volumen interpretiert werden soll.

### Co-Vergärung: gewichtete Mischung

Bei Multi-Substrat-Zuläufen werden die Konzentrationen volumetrisch
flussgewichtet gemischt (`Feedstock._blended_concentrations`). Die Anzahl
gleichzeitig beschickter Substrate ist nicht begrenzt — `Feedstock`
akzeptiert eine beliebig lange Liste von XML-IDs.

## Messgrößen

PyADM1ODE liefert die gleichen Messgrößen wie SIMBA# (Tutorial §5.3.7) zur
Anlagenüberwachung.

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

!!! note "Vorfaktor 50 vs 100"
    Das SIMBA#-Tutorial (PDF) dokumentiert den Vorfaktor als
    $f_{M,MOL,CaCO_3} = 100\,\text{kg/kmol}$ (die Molmasse), aber die
    tatsächliche `adm1da.asm`-Umsetzung verwendet **50 kg/keq** (das
    Äquivalent­gewicht, da 1 mol CaCO₃ zwei H⁺-Äquivalente trägt). PyADM1ODE
    folgt der `.asm`-Datei (Vorfaktor 50), was mit dem numerischen Output
    von SIMBA# übereinstimmt.

## Mathematische Grundlage

Die Implementierung basiert auf:

- **Schlattmann, M. (2011)**: ADM1da — Beschreibung im *SIMBA#-biogas-Tutorial  
  4.2*, ifak e.V. Magdeburg.  
- **Batstone, D. J. et al. (2002)**: *Anaerobic Digestion Model No. 1 (ADM1)*.  
  IWA Scientific and Technical Report No. 13.  
- **Siegrist, H., Vogt, D., Garcia-Heras, J. L., Gujer, W. (2002)**:  
  Mathematical model for meso- and thermophilic anaerobic sewage sludge
  digestion. *Environmental Science & Technology* **36**, 1113–1123.
  (Quelle der ADM1da-NH₃-Inhibitions-Formulierungen.)  
- **Gaida, D. (2014)**: *Dynamic real-time substrate feed optimization of  
  anaerobic co-digestion plants*. PhD thesis, Leiden University. (Vorbild
  für die volumetrische Mischungslogik.)  
- **Koch, K. et al. (2010)**: *Biogas from grass silage – measurements and  
  modeling with ADM1*. Bioresource Technology. (Kalibrierwerte für
  hochfeste Energiepflanzen.)

## Technische Umsetzung

Das gesamte Modell ist pure Python:

| Modul | Zweck |
| --- | --- |
| `pyadm1.core.adm1` | `ADM1`-Klasse mit `ADM_ODE`, Newton–Raphson-pH, Gasausgang |
| `pyadm1.core.adm_params` | Stöchiometrie, Kinetiken, Inhibition, $\theta$-Korrekturen |
| `pyadm1.core.solver` | Wrapper um `scipy.integrate.solve_ivp` (BDF, adaptive) |
| `pyadm1.substrates.feedstock` | XML-Parser, Sub-Fraktionierungs-Routing, Mischung |
| `pyadm1.components.biological.digester` | Komponenten-Wrapper inkl. Gasspeicher, Schlammvolumen und HRT-Logik |

Damit läuft die Simulation in jeder Standard-Python-Umgebung und auch in
Containern, Web-Notebooks (Colab) und CI-Pipelines problemlos.
