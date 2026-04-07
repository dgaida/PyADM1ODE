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

### pH-Wert Berechnung

In der Original-Publikation des ADM1 wird der pH-Wert oft über eine algebraische Ladungsbilanz gelöst, die eine iterative Bestimmung der Wasserstoffionen-Konzentration $[H^+]$ erfordert.

In dieser Implementierung wird der pH-Wert direkt aus der Ladungsbilanz der **dynamischen Ionen-Zustände** berechnet. Da Kationen ($S_{cat}$), Anionen ($S_{an}$) und die ionisierten Formen der organischen Säuren sowie des anorganischen Kohlenstoffs/Stickstoffs als eigene Zustandsvariablen im ODE-System geführt werden, kann der pH-Wert in jedem Schritt explizit bestimmt werden. Dieser Ansatz erhöht die numerische Stabilität (Robustheit) des Solvers, da keine algebraischen Gleichungen innerhalb des Integrationsschritts gelöst werden müssen.

## Erweiterungen für landwirtschaftliche Substrate

Die Implementierung enthält wichtige Erweiterungen, die speziell für die Vergärung von Energiepflanzen und Gülle optimiert wurden (nach **Koch et al., 2010**):

### Einfluss des TS-Gehalts auf die Hydrolyse

In landwirtschaftlichen Biogasanlagen mit hohen Feststoffgehalten (TS) ist die Hydrolyse oft der ratenlimitierende Schritt. Das Modell sieht hierfür eine Korrekturfunktion vor:
$$ hydro\_factor = \frac{1}{1 + (\frac{TS}{K_{hyd}})^{n_{hyd}}} $$

!!! info "Hinweis zur aktuellen Implementierung"
    Obwohl die mathematische Struktur zur TS-abhängigen Hydrolysekorrektur im Code implementiert ist (siehe `adm_equations.py`), ist sie in der aktuellen Version standardmäßig **deaktiviert**. Der `hydro_factor` wird beim Aufruf der Prozessraten fest auf `1.0` gesetzt, sodass die Korrekturgleichung im Standardbetrieb übersprungen wird.

### Modellierung von Zerfallsprodukten ($X_p$)

Analog zum ASM1 (Activated Sludge Model) wurde ein separater Zustand für partikuläre Zerfallsprodukte ($X_p$) eingeführt. Dies ermöglicht eine präzisere Schließung der Stickstoffbilanz und beschreibt die Akkumulation von inerten organischen Stoffen aus abgestorbener Biomasse genauer.

## Charakterisierung via Weender-Analyse

Substrate werden nicht direkt als ADM1-Komponenten eingegeben, sondern über praxisübliche Laborparameter definiert:
*   **Erweiterte Weender-Analyse**: Rohfaser (RF), Rohprotein (RP), Rohfett (RL).
*   **Van-Soest-Fraktionen**: NDF, ADF, ADL (Lignin).
*   **Physikalische Werte**: Trockensubstanz (TS), organische Trockensubstanz (oTS/VS), pH-Wert.

### Mapping auf ADM1-Eingangsgrößen

Die Umrechnung der Substratfraktionen in den ADM1-Zulaufstrom erfolgt dynamisch:
1.  **Zusammensetzung der Verbundstoffe ($X_c$)**: Basierend auf den Protein-, Fett- und Faseranteilen werden die stöchiometrischen Koeffizienten $f_{ch,xc}$, $f_{pr,xc}$, $f_{li,xc}$, $f_{xI,xc}$ und $f_{sI,xc}$ für jedes Substrat individuell berechnet.
2.  **Kinetische Parameter**: Substrate bringen ihre eigenen Raten für Desintegration ($k_{dis}$) und Hydrolyse ($k_{hyd}$) mit. Bei Substratgemischen werden diese Parameter gewichtet nach dem Volumenstrom berechnet.

### Mathematische Grundlage

Die Implementierung basiert auf den Arbeiten von:
*   **Gaida, D. (2014)**: *Dynamic real-time substrate feed optimization of anaerobic co-digestion plants*. PhD thesis, Leiden University.
*   **Koch, K. et al. (2010)**: *Biogas from grass silage – Measurements and modeling with ADM1*. Bioresource Technology.

## Technische Umsetzung

Die Berechnung der Substratparameter und des gemischten ADM1-Zulaufstroms erfolgt über hochoptimierte C#-DLLs (im Ordner `pyadm1/dlls/`), die via `pythonnet` in die Python-Umgebung eingebunden sind. Dies ermöglicht eine schnelle Berechnung auch bei komplexen Substratgemischen und großen Simulationsstudien.
