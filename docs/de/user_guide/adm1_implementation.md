# ADM1-Implementierung und Substratmodellierung

Diese Seite beschreibt die technischen Details der in PyADM1ODE verwendeten Implementierung des Anaerobic Digestion Model No. 1 (ADM1) und wie landwirtschaftliche Substrate in das Modell integriert werden.

## ADM1 als reines ODE-System

Im Gegensatz zum Standard-ADM1 (IWA Task Group, 2002), das oft als System von differentiell-algebraischen Gleichungen (DAE) formuliert wird, ist diese Implementierung ein **reines System gewöhnlicher Differentialgleichungen (ODE)**.

### Hauptunterschiede zum Standard-Modell

1.  **Keine algebraischen Zustände**: Säure-Base-Gleichgewichte und der Gas-Flüssig-Transfer werden kinetisch modelliert. Dies vermeidet die Notwendigkeit von iterativen algebraischen Solvern innerhalb jedes Zeitschritts des ODE-Solvers, was die numerische Stabilität erhöht.
2.  **37 Zustandsvariablen**: Das Modell trackt insgesamt 37 Variablen, um den vollständigen Prozess abzubilden:
    *   **Gelöste Komponenten (12)**: Monosaccharide, Aminosäuren, langkettige Fettsäuren (LCFA), Valerat, Butyrat, Propionat, Acetat, Wasserstoff, Methan, anorganischer Kohlenstoff ($S_{CO2}$), anorganischer Stickstoff ($S_{NH4}$), lösliche Inerte.
    *   **Partikuläre Komponenten (13)**: Verbundstoffe ($X_{xc}$), Kohlenhydrate, Proteine, Lipide, 7 bakterielle Populationen, partikuläre Inerte, partikuläre Zerfallsprodukte ($X_p$).
    *   **Säure-Base-Variablen (8)**: Kationen, Anionen sowie die ionisierten Formen der organischen Säuren und anorganischen Spezies.
    *   **Gasphase (4)**: Partialdrücke von $H_2$, $CH_4$, $CO_2$ und der Gesamtdruck.

## Modellierung landwirtschaftlicher Substrate

Ein wesentliches Merkmal dieses Repositories ist die detaillierte Abbildung landwirtschaftlicher Substrate (z. B. Maissilage, Gülle) auf die ADM1-Inputvariablen.

### Charakterisierung via Weender-Analyse

Substrate werden nicht direkt als ADM1-Komponenten eingegeben, sondern über praxisübliche Laborparameter definiert:
*   **Erweiterte Weender-Analyse**: Rohfaser (RF), Rohprotein (RP), Rohfett (RL).
*   **Van-Soest-Fraktionen**: NDF, ADF, ADL (Lignin).
*   **Physikalische Werte**: Trockensubstanz (TS), organische Trockensubstanz (oTS/VS), pH-Wert.

### Mapping auf ADM1-Eingangsgrößen

Die Umrechnung der Substratfraktionen in den ADM1-Zulaufstrom erfolgt dynamisch:
1.  **Zusammensetzung der Verbundstoffe ($X_c$)**: Basierend auf den Protein-, Fett- und Faseranteilen werden die stöchiometrischen Koeffizienten $f_{ch,xc}$, $f_{pr,xc}$, $f_{li,xc}$, $f_{xI,xc}$ und $f_{sI,xc}$ für jedes Substrat individuell berechnet.
2.  **Kinetische Parameter**: Substrate bringen ihre eigenen Raten für Desintegration ($k_{dis}$) und Hydrolyse ($k_{hyd}$) mit. Bei Substratgemischen werden diese Parameter gewichtet nach dem Volumenstrom berechnet.
3.  **VFA-Gehalt**: Bereits im Substrat vorhandene organische Säuren (z. B. in Silagen) werden direkt den entsprechenden gelösten ADM1-Komponenten zugeordnet.

### Mathematische Grundlage

Die Implementierung basiert auf der Dissertation von **Daniel Gaida (2014)**: *Dynamic real-time substrate feed optimization of anaerobic co-digestion plants*. Sie kombiniert die biochemische Struktur des ADM1 mit einem robusten Modell für die Substrat-Zulaufcharakterisierung, das speziell für landwirtschaftliche Anwendungen optimiert wurde.

## Technische Umsetzung

Die Berechnung der Substratparameter und des gemischten ADM1-Zulaufstroms erfolgt über hochoptimierte C#-DLLs (im Ordner `pyadm1/dlls/`), die via `pythonnet` in die Python-Umgebung eingebunden sind. Dies ermöglicht eine schnelle Berechnung auch bei komplexen Substratgemischen und großen Simulationsstudien.

### Beispiel: Substrat-Einfluss auf die Kinetik

Wenn Sie verschiedene Substrate mischen, berechnet das System automatisch die resultierenden kinetischen Raten:

```python
# Die ADM1-Klasse ermittelt intern die gemittelten Parameter
substrate_params = adm1._get_substrate_dependent_params()
# Dies beinhaltet k_dis, k_hyd_ch, k_hyd_pr, k_hyd_li basierend auf dem aktuellen Feed-Mix
```
