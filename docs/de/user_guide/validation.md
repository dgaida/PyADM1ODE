# Validierung

PyADM1ODE wurde gegen den Referenzsimulator **SIMBA# biogas 4.2**
(ifak e.V. Magdeburg) sowie gegen Messdaten realer landwirtschaftlicher
Biogasanlagen validiert. Diese Seite fasst die Validierungsstrategie und die
wichtigsten Ergebnisse zusammen.

## Referenzsimulator

SIMBA# biogas 4.2 ist die de-facto Industriereferenz für die landwirtschaftliche
ADM1-Erweiterung ADM1da. Die Validierungsläufe nutzen SIMBA#'s ADM1da-Implementierung
(die kinetische Säure-Base-Variante, *nicht* ADM1daph) mit identischer Reaktor­
geometrie, Temperatur, $k_L a$-Wert, Substratzusammensetzung und Anfangszustand.
SIMBA#-CSV-Exporte liefern die Referenz-Zustandstrajektorien.

## Validierungs­szenarien

### 30-tägiger Einzelsubstrat-Lauf (Schweinegülle)

20 m³/d Schweinegülle, $V_{liq} = 1050\,\text{m}^3$, $V_{gas} = 150\,\text{m}^3$,
$T = 42\,°\text{C}$. Dient der Verifikation der Prozesskinetik und der Logik
zur Übernahme des Anfangszustands.

### 150-tägiger Co-Vergärungs-Lauf (Maissilage + Schweinegülle)

11,4 m³/d Maissilage + 6,1 m³/d Schweinegülle, gleiche Reaktorgeometrie.
Validiert die Multi-Substrat-Mischlogik und die volumetrische Korrektur
`simba_q_convention`.

### 600-tägiger Co-Vergärungs-Lauf mit Substratwechsel (Maissilage + Rindergülle)

Das umfassendste Szenario:

- **Phase 1** (0–300 d): 11,4 m³/d Maissilage + 6,1 m³/d Rindergülle.  
- **Phase 2** (300–600 d): 10,0 m³/d Maissilage + 8,0 m³/d Rindergülle.  

Die dynamische Schlammvolumenbilanz ist aktiviert (`dynamic_volume=True`,
`outflow_time_constant=0.05 d`), sodass $V_{liq}$ dem nahezu instantanen
Überlaufwehr von SIMBA# folgt und auf ~1 m³ am Sollwert bleibt.

## Wichtigste Ergebnisse

Am Ende jeder Phase (Snapshots bei $t = 300\,\text{d}$ und $t = 600\,\text{d}$):

| Größe                                                   | Toleranz                                         | Status     |
| ------------------------------------------------------- | ------------------------------------------------ | ---------- |
| $Q_{gas},\,Q_{CH_4},\,Q_{CO_2}$                         | 1–3 %                                            | ✓ erfüllt  |
| $pH$                                                    | innerhalb 0,01 Einheiten                         | ✓ erfüllt  |
| $HRT$                                                   | innerhalb 0,2 % (über dyn. Schlammvolumenbilanz) | ✓ erfüllt  |
| $OLR$                                                   | innerhalb 3 %                                    | ✓ erfüllt  |
| Alle sieben Biomasse-Populationen $X_*$                 | 1–4 %                                            | ✓ erfüllt  |
| Partikuläre Substrat-Pools ($X_{PS}, X_{PF}, X_S$)      | 1–4 %                                            | ✓ erfüllt  |
| Gelöste Substrate $S_{su},\,S_{aa},\,S_{fa}$            | innerhalb 1 %                                    | ✓ erfüllt  |
| VFA-Spezies $S_{va},\,S_{bu},\,S_{pro}$ (und Ionen)     | innerhalb 1 %                                    | ✓ erfüllt  |
| Substratwechsel-Transient bei $t = 300\,\text{d}$       | gleiche charakteristische Zeitkonstante          | ✓ erfüllt  |

Zwei verbleibende Abweichungen am Validierungs-Betriebspunkt:

- **$S_{ac}$ (und damit die aggregierte VFA) liegt +19–21 % höher** als  
  SIMBA#. Dies ist ein Sättigungs-Verstärkungs-Artefakt: bei $S_{ac} \gg
  K_{S,ac} = 0{,}15$ sind die acetoklastischen Monod-Kinetiken gesättigt, und
  die Acetatkonzentration wird über den langsamen Verdünnungskanal eingestellt
  und nicht über eine Monod-Rückkopplung. Eine kleine eingangs­seitige
  Diskrepanz (~1,1 % im effektiven $Q$, zurückgeführt auf eine Dichte-
  Konvention für Maissilage) wird um den Faktor ~20 in den beobachteten
  S_ac-Offset verstärkt. Bei nicht-gesättigtem Betrieb verschwindet der
  Verstärkungseffekt.  
- **TAC ist −6 %**, ebenfalls auf den Biomasse­überschuss zurückzuführen:  
  2–3 % mehr Biomasse binden ~0,013 kmol C/m³ anorganischen Kohlenstoff in den
  Biomasse-Partikeln, was den gelösten $S_{HCO_3^-}$-Pool reduziert, der die
  TAC-Formel dominiert.

Beide Offsets sind über den Substratwechsel bei $t = 300\,\text{d}$ stabil und
sind damit stationäre Charakteristika und keine divergierenden Integrations­fehler.

## Fazit

PyADM1ODE ist als Drop-in-Ersatz für SIMBA# biogas 4.2 in Gasertrag-Prognosen,
OLR- und HRT-Analysen sowie pH-basierter Überwachung validiert. Die zwei
verbleibenden Residuen auf dem Acetat-Pool und auf der TAC sind dokumentiert
und auf die bekannte Dichte-Konventionsdifferenz zwischen den beiden Simulatoren
zurückführbar.

Ein vollständiger Bericht des 600-tägigen Rindergülle-Vergleichs inkl. aller
thematischen Zeitreihen-Plots und Snapshot-Tabellen ist im `Report/`-Verzeichnis
des Repositorys verfügbar.
