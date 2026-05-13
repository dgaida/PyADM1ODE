# Troubleshooting Guide

Diese Seite bietet eine Übersicht über häufige Probleme und deren Lösungen in PyADM1ODE. Detaillierte Troubleshooting-Anleitungen finden Sie in den jeweiligen Komponentendokumentationen.

## Übersicht

PyADM1ODE ist ein komplexes System mit biologischen, energetischen und mechanischen Komponenten. Probleme können in verschiedenen Bereichen auftreten:

- **[Installation und Setup](#installation-und-setup)**: Python-Umgebung, Abhängigkeiten  
- **[Biologische Prozesse](#biologische-prozesse)**: Fermenter-Instabilität, pH-Probleme, VFA-Akkumulation  
- **[Energiesystem](#energiesystem)**: BHKW, Heizung, Gasspeicher  
- **[Mechanische Komponenten](#mechanische-komponenten)**: Pumpen, Rührwerke  
- **[Fütterungssystem](#fütterungssystem)**: Substratqualität, Dosiergenauigkeit  
- **[Simulation und Performance](#simulation-und-performance)**: Laufzeit, Konvergenz, Numerik  

## Installation und Setup

Für Installationsprobleme siehe:

**[→ Installation Guide – Troubleshooting](installation.md#fehlerbehebung)**

Häufige Themen:

- Paket nicht in der aktiven Umgebung gefunden  
- Substrat-Dateien nicht gefunden  

## Biologische Prozesse

### Fermenter-Probleme

Für Diagnose und Lösung biologischer Prozessprobleme siehe:

**[→ Biologische Komponenten – Fehlerbehebung](components/biological.md#fehlerbehebung)**

Behandelte Themen:

#### Niedriger pH-Wert

- **Ursachen**: Hohe organische Raumbelastung (OLR), unzureichende Pufferkapazität  
- **Diagnose**: pH < 6.8, steigende VFA  
- **Lösungen**: OLR reduzieren, Kalkpuffer hinzufügen, Substratmischung anpassen  

#### Schaumbildung

- **Ursachen**: Hoher Proteingehalt, pH-Änderungen, hohe VFA  
- **Lösungen**: Proteinreiche Substrate reduzieren, pH stabilisieren  

#### Geringe Gasproduktion

- **Ursachen**: Niedrige OLR, schlechte Substratqualität, Inhibition, kurze HRT  
- **Diagnose-Tools**: Spezifische Gasproduktion, Methangehalt prüfen  
- **Lösungen**: Substratqualität verbessern, Inhibitoren identifizieren  

### Prozessüberwachung

**[→ Biologische Komponenten – Prozessüberwachung](components/biological.md#prozessuberwachung)**

Wichtige Prozessindikatoren:

- pH-Wert: 6.8–7.5 optimal  
- VFA/TAC-Verhältnis: < 0.4  
- Methangehalt: > 55 %  
- Temperaturstabilität  

## Energiesystem

### BHKW und Wärmesysteme

Für Energiekomponenten-Probleme siehe:

**[→ Energiekomponenten – Fehlerbehebung](components/energy.md#fehlerbehebung)**

Behandelte Themen:

#### BHKW läuft nicht

- **Diagnose**: Gasverfügbarkeit, minimaler Gasbedarf, Speicherdruck prüfen  
- **Lösungen**: Gasversorgung sicherstellen, Speicherdruck anpassen  

#### Übermäßiges Venting

- **Ursache**: Gasproduktion > BHKW-Verbrauch  
- **Lösungen**:  
  - BHKW-Kapazität erhöhen  
  - Zweites BHKW hinzufügen  
  - Gasspeicher vergrößern  

#### Unzureichende Wärme

- **Diagnose**: Hoher Zusatzheizungsbedarf  
- **Lösungen**: Isolierung verbessern, BHKW vergrößern, Fermentertemperatur reduzieren  

### Optimierungsstrategien

**[→ Energiekomponenten – Optimierungsstrategien](components/energy.md#optimierungsstrategien)**

## Mechanische Komponenten

### Pumpen- und Rührwerksprobleme

Für mechanische Komponentenprobleme siehe:

**[→ Mechanische Komponenten – Fehlerbehebung](components/mechanical.md#fehlerbehebung)**

Behandelte Themen:

#### Pumpe liefert unzureichenden Durchfluss

- **Diagnose**: Wirkungsgrad, Druckhöhe, Dimensionierung prüfen  
- **Lösungen**: Pumpengröße erhöhen, Reibungsverluste reduzieren, auf Blockaden prüfen  

#### Rührwerk verbraucht zu viel Energie

- **Diagnose**: Spezifische Leistung > 6.0 W/m³  
- **Lösungen**: Intermittierenden Betrieb aktivieren, Intensität reduzieren  

#### Schlechte Mischqualität

- **Diagnose**: Mischqualität < 0.7, lange Mischzeit  
- **Lösungen**: Intensität erhöhen, Einschaltzeit verlängern, größeres Rührblatt  

## Fütterungssystem

### Lager- und Dosierprobleme

Für Fütterungskomponenten-Probleme siehe:

**[→ Fütterungskomponenten – Fehlerbehebung](components/feeding.md#fehlerbehebung)**

Behandelte Themen:

#### Schneller Qualitätsverlust

- **Diagnose**: Qualitätsfaktor < 0.95 bei kurzer Lagerzeit  
- **Lösungen**: Lagertyp verbessern, Temperatur reduzieren, schnellere Verwendung  

#### Dosierer-Blockaden

- **Diagnose**: Häufige Blockaden (> 5)  
- **Lösungen**: Robusteren Dosierertyp wählen, Substratvorbereitung verbessern  

#### Inkonsistente Dosierung

- **Diagnose**: Durchschnittlicher Dosierfehler > 10 %  
- **Lösungen**: Präziseren Dosierertyp wählen, Kalibrierung prüfen  

## Simulation und Performance

### Simulationsprobleme

Für allgemeine Simulationsprobleme siehe:

**[→ Schnellstart – Fehlerbehebung](quickstart.md#fehlerbehebung)**

Behandelte Themen:

#### Simulation instabil

- **Symptome**: pH fällt, VFA steigt, Methanproduktion sinkt  
- **Lösungen**: Substratfütterungsrate reduzieren, Verweilzeit erhöhen, Puffermaterial hinzufügen  

#### Geringe Gasproduktion

- **Lösungen**: Substratfütterung erhöhen, Abbaubarkeit prüfen, Temperatur optimieren  

#### Langsame Simulation

- **Lösungen**: Zeitschritt (dt) erhöhen, save_interval reduzieren, parallele Simulation verwenden  

## FAQ

### Warum ist mein pH-Wert niedrig?

**Antwort**: Siehe [Biologische Komponenten – Niedriger pH-Wert](components/biological.md#niedriger-ph-wert)

### Warum läuft mein BHKW nicht?

**Antwort**: Siehe [Energiekomponenten – BHKW läuft nicht](components/energy.md#problem-bhkw-lauft-nicht)

## Support

Wenn Sie in dieser Dokumentation keine Lösung finden:

1. **GitHub Issues prüfen**: [Existing Issues](https://github.com/dgaida/PyADM1ODE/issues)  
2. **Neues Issue erstellen**  
3. **Kontakt**: <daniel.gaida@th-koeln.de>  
