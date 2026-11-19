# Validierung

PyADM1ODE wurde gegen den Referenzsimulator **SIMBA# biogas** (ifak Magdeburg)
sowie gegen Messdaten realer landwirtschaftlicher Biogasanlagen validiert.
Diese Seite beschreibt den Vergleich mit SIMBA# und nennt seine Ergebnisse.

## Referenzsimulator

SIMBA# biogas ist die de-facto Industriereferenz für die landwirtschaftliche
ADM1-Erweiterung ADM1da. Der Referenzlauf nutzt das ADM1da-Modell von SIMBA#
(die kinetische Säure-Base-Variante, *nicht* ADM1daph) in SIMBA# 5.0.57.
PyADM1ODE wiederholt den Lauf mit demselben Reaktor, demselben Anfangszustand
und **demselben Zulauf**: Der Zulauf ist der, den SIMBA# aus seinen eigenen
Substratblöcken berechnet. Der Vergleich prüft damit allein das Vergärungsmodell
und hängt nicht von der Substratbibliothek ab.

## Referenzlauf

Eine 600-tägige Co-Vergärung von Maissilage und Rindergülle mit Substratwechsel:

- **Phase 1** (0–300 d): 11,4 t/d Maissilage + 6,1 m³/d Rindergülle (16,2 m³/d).  
- **Phase 2** (300–600 d): 10,0 t/d Maissilage + 8,0 m³/d Rindergülle (16,8 m³/d).  

Der Reaktor fasst 1050 m³ Schlamm und 150 m³ Gas bei 42 °C, mit
$k_L a = 200\,\text{d}^{-1}$ und einer Verweilzeit von 62 bis 65 Tagen. Der
Betriebspunkt ist anspruchsvoll: 3,6 bis 4,1 g/l flüchtige Fettsäuren und
115 mg/l freier Ammoniakstickstoff halten die Acetat-Verwerter nahe an ihrer
Hemmgrenze. Acetat reagiert dann etwa zehnmal stärker auf jeden Modellunterschied
als die übrigen Größen und ist damit ein empfindlicher Indikator.

Die Referenzdaten (Anfangszustand, Zulauf beider Phasen, SIMBA#-Zustände bei
300 d und 600 d sowie eine grobe Trajektorie) liegen in
`tests/integration/data/simba_adm1da_cattle_600d.json`. Der Test
`tests/integration/test_simba_reference.py` wiederholt den Lauf über die
Komponente `Digester` und gehört zur Testsuite.

### Gasraum

SIMBA# hält seinen Gasraum bei 1,017 bar mit rund 80 % relativer Feuchte, weil
es das Gas mit endlicher Rate befeuchtet. Das trockene Gas hat dann 0,951 bar.
PyADM1ODE hält den Gasraum gesättigt. Der Vergleich legt den trockenen Gasdruck
des SIMBA#-Laufs fest (`ADM1.p_ext = 1.033` bar); die Wirkung des
Standard-Gasraums steht getrennt weiter unten.

## Ergebnisse

Abweichung von PyADM1ODE gegenüber SIMBA# am Ende jeder Phase:

| Größe | Tag 300 | Tag 600 |
| --- | --- | --- |
| pH | +0,001 | +0,001 |
| Flüchtige Fettsäuren (FOS) | −0,4 % | −0,9 % |
| TAC | +0,1 % | +0,2 % |
| Methanstrom | 0,0 % | 0,0 % |
| CO₂-Strom | 0,0 % | 0,0 % |
| Methangehalt des trockenen Gases | +0,02 Punkte | +0,02 Punkte |
| Anorganischer Stickstoff, freies Ammoniak | 0,0 % | 0,0 % |
| Anorganischer Kohlenstoff, Hydrogencarbonat | +0,2 % | +0,4 % |
| Acetat | −0,4 % | −0,9 % |
| Propionat, Butyrat, Valerat, Zucker, Aminosäuren, Fettsäuren | innerhalb 0,1 % | innerhalb 0,1 % |
| Partikuläre Pools ($X_{PS}, X_{PF}, X_S$, $X_I$) | innerhalb 0,05 % | innerhalb 0,05 % |
| Alle sieben Biomasse-Populationen | innerhalb 0,05 % | innerhalb 0,05 % |

Über den ganzen Lauf, ab Tag 10:

| Größe | Größte Abweichung |
| --- | --- |
| pH | 0,002 |
| TAC | 0,2 % |
| Anorganischer Stickstoff | 0,01 % |
| Methanstrom | 0,11 % (am Tag nach dem Substratwechsel) |
| CO₂-Strom | 0,02 % |
| Flüchtige Fettsäuren | 2,0 % (an Tag 10; ab Tag 75 höchstens 1 %) |

Auch der Ammoniakgehalt des Gases stimmt überein: rund 320 ppm im trockenen Gas
in beiden Modellen.

### Standard-Gasraum

Mit dem Standard-Gasraum (gesättigt, 1,013 bar) hat das trockene Gas 0,932 bar,
2 % weniger als im SIMBA#-Lauf. Gelöstes CO₂ und Hydrogencarbonat folgen um
dieselben 2 %, und Acetat steigt, um den pH zu halten:

| Größe | Tag 300 | Tag 600 |
| --- | --- | --- |
| Acetat, FOS | +3,2 % | +3,4 % |
| TAC | −0,8 % | −0,7 % |
| Anorganischer Kohlenstoff | −1,9 % | −1,6 % |
| pH | 0,000 | +0,001 |
| Anorganischer Stickstoff, Methan- und CO₂-Strom | innerhalb 0,1 % | innerhalb 0,1 % |

## Geltungsbereich

- **Substrate:** Der Zulauf stammt aus SIMBA#. Wie PyADM1ODE aus einer Substratdatei einen Zulauf ableitet, ist nicht Teil dieses Vergleichs.  
- **Gasmeldung:** SIMBA# meldet feuchtes Gas (6,4 bis 6,5 % Wasserdampf in diesem Lauf). Seine Methan- und CO₂-Ströme sind direkt mit `Q_ch4` und `Q_co2` vergleichbar; Gesamtgasstrom und Methangehalt müssen zuerst auf das trockene Gas umgerechnet werden.  
- **Schnelle Dynamik:** SIMBA# koppelt die Säure-Base-Paare über das Carbonatsystem, PyADM1ODE über die Protonenkonzentration. Beide stellen dasselbe Gleichgewicht ein; die Zeit dafür unterscheidet sich, was nur im Stundenbereich eine Rolle spielt.  
- **Frühere Läufe:** Ein 30-tägiger Lauf mit Schweinegülle und ein 150-tägiger Lauf mit Maissilage und Schweinegülle wurden mit Version 0.6 verglichen. Sie wurden nicht wiederholt, weil die SIMBA#-Einstellungen dieser Läufe nicht mehr vorliegen.  

## Änderungen seit Version 0.6.2

Bis Version 0.6.2 ergab derselbe Lauf Acetat 20 bis 30 % über SIMBA#, TAC 5 bis
6 % darunter und den CO₂-Strom 2 bis 4 % darunter. Ursache waren vier
Abweichungen vom SIMBA#-Modell, die alle behoben sind:

- **Druck im Gasraum:** Der Wasserdampfdruck wurde doppelt abgezogen; das gelöste CO₂ lag dadurch rund 10 % zu niedrig.  
- **Säure-Base-Terme:** Die Säure-Base-Raten wurden den Gesamtmengen an anorganischem Kohlenstoff und Stickstoff zugeschlagen, was dem Gas Kohlenstoff entzog.  
- **Stickstoff:** Der Zerfall setzte Stickstoff frei, statt etwas zu binden, und Ammoniak verließ den Fermenter nicht mit dem Gas; zusammen rund 3 % zu viel anorganischer Stickstoff.  
- **Schlammvolumen:** Der Volumenverlust war rund 10 % zu hoch, was alle Konzentrationen um rund 1 % anhob.  

Die Einzelheiten stehen auf der Seite [ADM1-Implementierung](adm1_implementation.md).

## Fazit

Bei gleichem Zulauf gibt PyADM1ODE das ADM1da-Modell von SIMBA# auf 0,2 % genau
wieder (pH, TAC, Stickstoff, partikuläre Pools, Biomasse, Gasströme) und Acetat
auf 2 %, einschließlich des Übergangs nach einem Substratwechsel.
