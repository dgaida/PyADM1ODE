# Validierung

Das Framework wurde gegen verschiedene Referenzmodelle und Realdaten validiert:

- **SIMBA#**: Kommerzielle Biogas-Simulationssoftware (ifak e.V.).  
- **[ADM1F](https://github.com/lanl/ADM1F)**: ADM1-Implementierung in Fortran des LANL.  
- **Realdaten**: Daten von mehreren landwirtschaftlichen Biogasanlagen.  

## Vergleich mit SIMBA#
Die Implementierung des ADM1da-Modells wurde intensiv mit SIMBA# abgeglichen, um die Korrektheit der Stöchiometrie, Kinetik und der pH-Wert-Berechnung sicherzustellen.

## Vergleich mit ADM1F
Der Kern-ADM1-Code wurde gegen die ADM1F-Implementierung validiert, um die mathematische Konsistenz der ODE-Löser und der Prozessraten zu bestätigen.

## Validierung mit Realdaten
PyADM1ODE wird kontinuierlich mit Daten aus dem realen Anlagenbetrieb validiert, insbesondere im Hinblick auf die Gasproduktion und die Substratabbaugeschwindigkeiten unter praxisnahen Bedingungen.
