# ADM1-Implementierung und Substratmodellierung

Diese Seite beschreibt die technischen Details des in PyADM1ODE umgesetzten ADM1-Modells und wie landwirtschaftliche Substrate in das Modell integriert werden.

PyADM1ODE implementiert **ADM1da** (Schlattmann 2011) — eine landwirtschaftliche Adaption des ursprünglichen ADM1 (Batstone et al. 2002, IWA Task Group). Im Vergleich zur klassischen Formulierung enthält dieses Modell eine **Sub-Fraktionierung der Desintegration**, **temperaturabhängige Kinetiken** sowie **modifizierte Inhibitionskinetiken** für agrarische Co-Vergärung.

## ADM1 als reines ODE-System

Im Gegensatz zum Standard-ADM1, das oft als System differentiell-algebraischer Gleichungen (DAE) formuliert wird, ist diese Implementierung ein **reines System gewöhnlicher Differentialgleichungen (ODE)**.

### Hauptmerkmale

1. **Kinetisches Säure-Base-Gleichgewicht**: Die ionisierten Spezies (Acetat-, Propionat-, Butyrat-, Valerat-Ion, $\text{HCO}_3^-$, $\text{NH}_3$) werden als dynamische Zustandsvariablen mit einer sehr hohen Reaktionsrate $k_{A,B} = 10^8\,\text{m}^3\,\text{kmol}^{-1}\,\text{d}^{-1}$ geführt. Dadurch verfolgen sie das thermodynamische Gleichgewicht praktisch direkt, ohne dass innerhalb des ODE-Schritts ein algebraischer Solver aufgerufen werden muss.  
2. **Kinetischer Gas-Flüssig-Transfer**: $\text{H}_2$, $\text{CH}_4$ und $\text{CO}_2$ werden über $k_L a$-Raten in die Gasphase überführt — keine algebraische Henry-Gleichgewichtsbedingung.  
3. **41 Zustandsvariablen**: Das Modell verfolgt insgesamt 41 Variablen, gegliedert in fünf Blöcke:  

   | Block                          | Indizes | Anzahl | Inhalt                                                                                                       |
   |--------------------------------|---------|--------|--------------------------------------------------------------------------------------------------------------|
   | Gelöste Komponenten            | 0–11    | 12     | $S_{su},S_{aa},S_{fa},S_{va},S_{bu},S_{pro},S_{ac},S_{h2},S_{ch4},S_{IC},S_{IN},S_{I}$                       |
   | Partikuläre Sub-Fraktionen     | 12–21   | 10     | $X_{PS\_ch/pr/li}$ (langsam), $X_{PF\_ch/pr/li}$ (schnell), $X_{S\_ch/pr/li}$ (hydrolisierbar), $X_I$         |
   | Biomasse                       | 22–28   | 7      | $X_{su},X_{aa},X_{fa},X_{c4},X_{pro},X_{ac},X_{h2}$                                                          |
   | Säure-Base / Ladungsbilanz     | 29–36   | 8      | $S_{cat},S_{an}$, ionisierte Formen $S_{va^-},S_{bu^-},S_{pro^-},S_{ac^-},S_{HCO_3^-},S_{NH_3}$              |
   | Gasphase                       | 37–40   | 4      | $p_{H_2},p_{CH_4},p_{CO_2},p_{tot}$                                                                          |

### pH-Wert-Berechnung

Der pH-Wert wird in jedem Auswertungsschritt über die **Ladungsbilanz** und ein Newton-Raphson-Verfahren bestimmt:

$$
S_{cat} - S_{an} + (S_{NH4} - S_{NH3}) - S_{HCO_3^-} - \frac{S_{ac^-}}{64} - \frac{S_{pro^-}}{112} - \frac{S_{bu^-}}{160} - \frac{S_{va^-}}{208} + S_{H^+} - \frac{K_w}{S_{H^+}} = 0
$$

Da alle Beiträge zur linken Seite Zustandsvariablen sind, ist diese Ladungsbilanz innerhalb des ODE-Schritts eine **rein punktweise Funktion des aktuellen Zustands** und liefert in 5–10 Newton-Iterationen $[H^+]$ mit hoher Genauigkeit. Die fast-Equilibrierung der ionisierten Spezies durch $k_{A,B}$ stellt sicher, dass der berechnete pH konsistent mit den thermodynamischen Säure-Base-Konstanten bleibt.

## Sub-Fraktionierung der Desintegration

Der wichtigste Unterschied zum klassischen ADM1 ist die **zwei-Pool-Desintegration** anstelle der monolithischen Verbundstoff-Variable $X_c$:

```
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

| Pool       | Substratquelle (Weender + Sub-Fraktionierungs-Parameter)                                                              |
|------------|-----------------------------------------------------------------------------------------------------------------------|
| $X_{PS}$   | Rohfaser (immer langsamer Pool) **plus** der Anteil $f_{sOTS}$ der NFE-Kohlenhydrate, Proteine und Fette              |
| $X_{PF}$   | Anteil $f_{fOTS}$ der NFE-Kohlenhydrate, Proteine und Fette                                                           |
| $X_S$      | Wird durch Desintegration aus $X_{PS}$ und $X_{PF}$ erzeugt; nicht direkt aus dem Substrat                            |
| $X_I$      | Anteil $a_{XI}$ der gesamten organischen Roh-COD                                                                      |
| $S_I$      | Anteil $a_{Si}$ der gesamten organischen Roh-COD (gelöste Inerte, treten direkt im Zulauf auf)                        |

Damit lässt sich beispielsweise leichtabbaubare Stärke (NFE) gezielt in den schnellen Pool routen, während Lignocellulose (Rohfaser) automatisch in den langsamen Pool fällt — ohne Modellumbau.

## Temperaturabhängige Kinetiken

Die ADM1da-Variante korrigiert sämtliche kinetischen Raten gegenüber der 35 °C-Referenz mit einer Arrhenius-θ-Funktion:

$$
k(T) = k(35\,°\text{C}) \cdot \theta^{(T[°\text{C}] - 35)}
$$

mit gruppenspezifischen Exponenten (Werte nach Schlattmann 2011):

| Prozess­gruppe                                    | $\theta_{\exp}$ |
|---------------------------------------------------|-----------------|
| Desintegration & Hydrolyse                        | 0,024           |
| $X_{su}, X_{aa}, X_{h2}$ (Wachstum & Zerfall)     | 0,069           |
| $X_{fa}, X_{c4}, X_{pro}, X_{ac}$ (Wachstum & Zerfall) | 0,055           |

Die Korrektur wird einmalig beim Anlegen des `ADM1`-Objekts auf alle relevanten Raten angewandt (`ADMParams.apply_temperature_corrections`) und im internen `_kinetic`-Dict zwischengespeichert.

## Modifizierte Inhibitionskinetiken

Gegenüber dem Standard-ADM1 weist das ADM1da-Modell folgende Anpassungen auf — alle in `ADM1.ADM_ODE` umgesetzt:

| Inhibition                  | Standard-ADM1                  | ADM1da                                                           |
|-----------------------------|--------------------------------|------------------------------------------------------------------|
| pH-Inhibition $X_{fa}/X_{c4}/X_{pro}$ | Hill, $n=1$            | Hill, $n=2$ (steilerer Cut-Off)                                  |
| pH-Inhibition $X_{ac}$      | Hill, $n=1$                    | Hill, $n=3$                                                      |
| pH-Inhibition $X_{h2}$      | Hill, $n=1$                    | Hill, $n=3$                                                      |
| N-Limitation                | $S_{NH4}$ allein               | $S_{IN} = S_{NH4} + S_{NH3}$                                     |
| $\text{NH}_3$-Inhibition $X_{ac}$ | linear in $S_{NH3}$       | Hill quadratisch: $K_I^2/(K_I^2 + S_{NH3}^2)$, T-korrigiert      |
| $\text{NH}_3$-Inhibition $X_{pro}$ | nicht enthalten           | gleiche Form mit eigenem $K_{I,nh3,pro}$                         |
| Undissoziierte Säuren       | nicht enthalten                | $K_{IH,pro}$ (Propionsäure $\to X_{pro}$), $K_{IH,ac}$ (Essigsäure $\to X_{ac}$) |
| $\text{CO}_2$-Limitation $X_{h2}$ | nicht enthalten          | Hill quadratisch in $S_{CO2}$                                    |

Diese Erweiterungen reproduzieren das in landwirtschaftlichen Anlagen typische Verhalten: schärferer pH-Abfall bei Säureakkumulation, ausgeprägtere $\text{NH}_3$-Hemmung bei thermophilem Betrieb auf Güllebasis und realistischere Propionatdynamik.

## Substratcharakterisierung via Weender-Analyse

Substrate werden weiterhin über praxisübliche Laborparameter definiert, jetzt aber als **XML-Dateien** unter `data/substrates/adm1da/`. Eine Beispielstruktur:

```xml
<substrate name="Maissilage (Milchreife)">
  <param name="TS"    value="320.0"/>
  <param name="NH4"   value="0.0"/>
  <param name="pH"    value="3.9"/>
  <param name="fRF"   value="0.220"/>   <!-- Rohfaser -->
  <param name="fRP"   value="0.080"/>   <!-- Rohprotein -->
  <param name="fRFe"  value="0.030"/>   <!-- Rohfett -->
  <param name="fRA"   value="0.045"/>   <!-- Rohasche -->
  <param name="aXI"   value="0.10"/>    <!-- partikulär inerter COD-Anteil -->
  <param name="aSi"   value="0.02"/>    <!-- löslich inerter COD-Anteil -->
  <param name="fOTSrf" value="0.40"/>   <!-- abbaubarer Anteil der Rohfaser -->
  <param name="fsOTS"  value="0.30"/>   <!-- NFE/PR/LI in den langsamen Pool -->
  <param name="ffOTS"  value="0.70"/>   <!-- NFE/PR/LI in den schnellen Pool -->
  <param name="FFS"    value="0.0"/>    <!-- VFA als Essigsäure-Äquivalent -->
  <param name="KS43"   value="0.0"/>    <!-- Säurekapazität bis pH 4,3 -->
  <!-- ... -->
</substrate>
```

`load_substrate_xml()` liefert eine `SubstrateParams`-Dataclass; die `Feedstock`-Klasse berechnet daraus den vollständigen 38-spaltigen ADM1-Zulaufstrom (37 flüssige Zustandsspalten + Q):

1. **Frischmasse-Dichte** $\rho_{FM}$ aus den Komponentendichten (Mischungsregel auf spezifischem Volumen).  
2. **Organische COD-Konzentrationen** $X_{ch}, X_{pr}, X_{li}$ in $\text{kg COD/m}^3$ über die COD-Umrechnungsfaktoren $M_{Xch}, M_{Xpr}, M_{Xli}$.  
3. **Routing in die Sub-Fraktionen** $X_{PS}/X_{PF}$ via $f_{sOTS}, f_{fOTS}$ und $f_{OTSrf}$ (siehe Tabelle oben).  
4. **Berechnung der Dissoziation** bei Substrat-pH: ionisierte VFA, $\text{HCO}_3^-$, $\text{NH}_3$.  
5. **Ladungsbilanz** schließt zu $S_{anion}$ (oder $S_{cation}$, falls negativ).  

### Volumetrische ADM1da-Konvention für Q

Die Substrat-Volumenströme $Q_i$ werden standardmäßig als **massenäquivalente Ströme** interpretiert. Intern wird

$$
Q_{actual,i} = Q_{input,i} \cdot \frac{1000}{\rho_{FM,i}}
$$

berechnet. Für flüssige Substrate ($\rho_{FM} \approx 1000$) ist das ein No-Op; für Maissilage ($\rho_{FM} \approx 1134\,\text{kg/m}^3$) reduziert sich der tatsächliche Volumenstrom etwa um 12 %. Damit ist Parität zu ADM1da-Referenzergebnissen gegeben. Über `simba_q_convention=False` lässt sich diese Skalierung abschalten, falls $Q$ direkt als tatsächliches Reaktorvolumen interpretiert werden soll.

### Co-Vergärung: gewichtete Mischung

Bei Multi-Substrat-Zuläufen werden die Konzentrationen volumetrisch flussgewichtet gemischt (`Feedstock._blended_concentrations`). Die Anzahl gleichzeitig beschickter Substrate ist nicht begrenzt — `Feedstock` akzeptiert eine beliebig lange Liste von XML-IDs.

## Mathematische Grundlage

Die Implementierung basiert auf:

- **Schlattmann, M. (2011)**: ADM1da — Beschreibung im *SIMBA#-biogas-Tutorial 4.2*, ifak e.V. Magdeburg.  
- **Batstone, D. J. et al. (2002)**: *Anaerobic Digestion Model No. 1 (ADM1)*. IWA Scientific and Technical Report No. 13.  
- **Gaida, D. (2014)**: *Dynamic real-time substrate feed optimization of anaerobic co-digestion plants*. PhD thesis, Leiden University. (Vorbild für die volumetrische Mischungslogik.)  
- **Koch, K. et al. (2010)**: *Biogas from grass silage – measurements and modeling with ADM1*. Bioresource Technology. (Kalibrierwerte für hochfeste Energiepflanzen.)  

## Technische Umsetzung

Das gesamte Modell ist pure Python:

| Modul                                          | Zweck                                                        |
|------------------------------------------------|--------------------------------------------------------------|
| `pyadm1.core.adm1`                             | `ADM1`-Klasse mit `ADM_ODE`, Newton-Raphson-pH, Gasausgang   |
| `pyadm1.core.adm_params`                       | Stöchiometrie, Kinetiken, Inhibition, $\theta$-Korrekturen   |
| `pyadm1.core.solver`                           | Wrapper um `scipy.integrate.solve_ivp` (BDF, adaptive)       |
| `pyadm1.substrates.feedstock`                  | XML-Parser, Sub-Fraktionierungs-Routing, Mischung            |
| `pyadm1.components.biological.digester`        | Komponentenwrapper inkl. Gasspeicher und HRT-Logik           |

Damit läuft die Simulation in jeder Python-Umgebung ohne .NET-/Mono-Runtime und auch in Containern, Web-Notebooks (Colab) und CI-Pipelines problemlos.
