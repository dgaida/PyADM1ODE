# Troubleshooting Guide

Diese Seite bietet eine Übersicht über häufige Probleme und deren Lösungen in PyADM1ODE. Detaillierte Troubleshooting-Anleitungen finden Sie in den jeweiligen Komponentendokumentationen.

## Übersicht

PyADM1ODE ist ein komplexes System mit biologischen, energetischen und mechanischen Komponenten. Probleme können in verschiedenen Bereichen auftreten:

- **[Installation und Setup](#installation-und-setup)**: Python-Umgebung, Abhängigkeiten, C# DLLs
- **[Biologische Prozesse](#biologische-prozesse)**: Fermenter-Instabilität, pH-Probleme, VFA-Akkumulation
- **[Energiesystem](#energiesystem)**: BHKW, Heizung, Gasspeicher
- **[Mechanische Komponenten](#mechanische-komponenten)**: Pumpen, Rührwerke
- **[Fütterungssystem](#fütterungssystem)**: Substratqualität, Dosiergenauigkeit
- **[Simulation und Performance](#simulation-und-performance)**: Laufzeit, Konvergenz, Numerik

## Installation und Setup

Für Installationsprobleme siehe:

**[→ Installation Guide - Troubleshooting Section](installation.md#troubleshooting)**

Häufige Themen:
- C# DLL-Dateien nicht gefunden
- pythonnet Import-Fehler
- Mono/.NET Framework-Probleme
- Erste Import-Verzögerungen
- Modul-Attributfehler

## Biologische Prozesse

### Fermenter-Probleme

Für Diagnose und Lösung biologischer Prozessprobleme siehe:

**[→ Biological Components - Troubleshooting Section](components/biological.md#fehlerbehebung)**

Behandelte Themen:

#### Niedriger pH-Wert
- **Ursachen**: Zu hohe organische Raumbelastung, unzureichende Pufferkapazität
- **Diagnose**: pH < 6.8, steigende VFA
- **Lösungen**: OLR reduzieren, Kalkpuffer hinzufügen, Substratmischung anpassen

#### Schaumbildung
- **Ursachen**: Hoher Proteingehalt, pH-Änderungen, hohe VFA
- **Lösungen**: Proteinreiche Substrate reduzieren, pH stabilisieren

#### Geringe Gasproduktion
- **Ursachen**: Niedrige OLR, Substratqualität, Inhibition, kurze HRT
- **Diagnose-Tools**: Spezifische Gasproduktion, Methangehalt prüfen
- **Lösungen**: Substratqualität verbessern, Inhibitoren identifizieren

### Prozessüberwachung

**[→ Biological Components - Process Monitoring](components/biological.md#prozessüberwachung)**

Wichtige Prozessindikatoren:
- pH-Wert: 6.8-7.5 optimal
- VFA/TAC-Verhältnis: < 0.4
- Methangehalt: > 55%
- Temperaturstabilität

## Energiesystem

### BHKW und Wärmesysteme

Für Energiekomponenten-Probleme siehe:

**[→ Energy Components - Troubleshooting Section](components/energy.md#fehlerbehebung)**

Behandelte Themen:

#### BHKW läuft nicht
- **Diagnose**: Gasverfügbarkeit, Mindestgasbedarf, Speicherdruck prüfen
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

**[→ Energy Components - Optimization Strategies](components/energy.md#optimierungsstrategien)**

Themen:
- Wärmenutzungsoptimierung
- Lastfolgebetrieb
- Gasspeicher-Management

## Mechanische Komponenten

### Pumpen- und Rührwerkprobleme

Für mechanische Komponentenprobleme siehe:

**[→ Mechanical Components - Troubleshooting Section](components/mechanical.md#fehlerbehebung)**

Behandelte Themen:

#### Pumpe liefert unzureichenden Durchfluss
- **Diagnose**: Effizienz, Druckhöhe, Dimensionierung prüfen
- **Lösungen**: Pumpengröße erhöhen, Reibungsverluste reduzieren, Blockaden prüfen

#### Rührwerk verbraucht zu viel Energie
- **Diagnose**: Spezifische Leistung > 6.0 W/m³
- **Lösungen**: Intermittierenden Betrieb aktivieren, Intensität reduzieren

#### Schlechte Mischqualität
- **Diagnose**: Mixing quality < 0.7, lange Mischzeit
- **Lösungen**: Intensität erhöhen, Einschaltzeit verlängern, größeres Rührblatt

### Optimierungsstrategien

**[→ Mechanical Components - Optimization Strategies](components/mechanical.md#optimierungsstrategien)**

Themen:
- Pumpenoptimierung (80-90% Nennlast)
- Mischstrategie-Auswahl
- Energieminimierung

## Fütterungssystem

### Lager- und Dosierprobleme

Für Fütterungskomponenten-Probleme siehe:

**[→ Feeding Components - Troubleshooting Section](components/feeding.md#fehlerbehebung)**

Behandelte Themen:

#### Schneller Qualitätsverlust
- **Diagnose**: Qualitätsfaktor < 0.95 bei kurzer Lagerzeit
- **Lösungen**: Lagertyp verbessern, Temperatur reduzieren, schnellere Verwendung

#### Dosierer-Blockaden
- **Diagnose**: Häufige Blockaden (> 5)
- **Lösungen**: Robusteren Dosierertyp wählen, Substratvorbereitung verbessern

#### Inkonsistente Dosierung
- **Diagnose**: Durchschnittlicher Dosierfehler > 10%
- **Lösungen**: Präziseren Dosierertyp erwägen, Kalibrierung prüfen

### Substratmanagement

**[→ Feeding Components - Optimization Strategies](components/feeding.md#optimierungsstrategien)**

Themen:
- Substratmischoptimierung
- Bestandsmanagement
- Dosiergenauigkeitsoptimierung

## Simulation und Performance

### Simulationsprobleme

Für allgemeine Simulationsprobleme siehe:

**[→ Quickstart Guide - Troubleshooting Section](quickstart.md#troubleshooting)**

Behandelte Themen:

#### Simulation instabil
- **Symptome**: pH fällt, VFA steigt, Methanproduktion sinkt
- **Lösungen**: Substratfütterungsrate reduzieren, Verweilzeit erhöhen, Puffermaterial hinzufügen

#### Niedrige Gasproduktion
- **Lösungen**: Substratfütterung erhöhen, Abbaubarkeit prüfen, Temperatur optimieren

#### Langsame Simulation
- **Lösungen**: Zeitschritt erhöhen (dt), save_interval reduzieren, parallele Simulation nutzen

## Best Practices nach Komponententyp

### Biologische Komponenten

**[→ Biological Components - Best Practices](components/biological.md#best-practices)**

1. Mit realistischen Betriebsparametern starten
2. Kritische Parameter überwachen
3. Puffersysteme implementieren
4. Zweistufige Systeme für schwierige Substrate nutzen
5. Modell mit realen Daten kalibrieren

### Energiekomponenten

**[→ Energy Components - Performance Metrics](components/energy.md#leistungsmetriken)**

- BHKW-Verfügbarkeit berechnen
- Volllaststunden tracken
- Wärmenutzungsgrad optimieren
- Parasitäre Last minimieren

### Mechanische Komponenten

**[→ Mechanical Components - Best Practices](components/mechanical.md#best-practices)**

1. Pumpen für optimalen Wirkungsgrad dimensionieren (80-90% Nennlast)
2. Intermittierendes Mischen implementieren
3. Regelmäßige Wartung
4. Systemdesign optimieren
5. Energieverbrauch überwachen

### Fütterungskomponenten

**[→ Feeding Components - Best Practices](components/feeding.md#best-practices)**

1. Qualitätsüberwachung implementieren
2. Bestandssicherheit aufrechterhalten (7-14 Tage)
3. Energieverbrauch optimieren
4. Substratmischung optimieren
5. Wartung planen

## Diagnose-Tools und Checklisten

### Fermenter-Gesundheitscheck

```python
def monitor_digester_health(results):
    """
    Umfassender Fermenter-Gesundheitscheck

    Siehe: components/biological.md#prozessüberwachung
    """
    for result in results:
        digester_data = result['components']['main_digester']

        # pH-Wert
        pH = digester_data['pH']
        if pH < 6.8:
            print(f"⚠ Niedriger pH ({pH:.2f}) - Übersäuerungsrisiko")
        elif pH > 8.0:
            print(f"⚠ Hoher pH ({pH:.2f}) - Mögliche Ammoniakinhibition")

        # VFA/TAC-Verhältnis
        VFA = digester_data['VFA']
        TAC = digester_data['TAC']
        VFA_TAC = VFA / TAC if TAC > 0 else 0

        if VFA_TAC > 0.4:
            print(f"⚠ Hohes VFA/TAC ({VFA_TAC:.2f}) - Prozessinstabilität")

        # Gasproduktion
        Q_gas = digester_data['Q_gas']
        if Q_gas < 500:  # Beispielschwelle
            print(f"⚠ Niedrige Gasproduktion ({Q_gas:.1f} m³/d)")
```

### Energiebilanz-Analyse

```python
def energy_balance_check(results):
    """
    Energiesystem-Diagnose

    Siehe: components/energy.md#vollständiges-energiesystem
    """
    final = results[-1]
    comp = final['components']

    # BHKW-Leistung
    P_el = comp['chp1']['P_el']
    P_th = comp['chp1']['P_th']

    # Wärmebedarf
    Q_heat = comp['heat1']['Q_heat_supplied']
    P_aux = comp['heat1']['P_aux_heat']

    # Warnungen
    if P_aux > 50:
        print(f"⚠ Hoher Zusatzheizungsbedarf: {P_aux:.1f} kW")
        print("  → Isolierung verbessern oder BHKW vergrößern")

    heat_utilization = Q_heat / P_th if P_th > 0 else 0
    if heat_utilization < 0.5:
        print(f"⚠ Niedrige Wärmenutzung: {heat_utilization:.1%}")
        print("  → Zusätzliche Wärmeverbraucher erwägen")
```

### Mechanische Energie-Audit

```python
def mechanical_energy_audit(results):
    """
    Parasitäre Last-Analyse

    Siehe: components/mechanical.md#integrationsbeispiel
    """
    final = results[-1]
    comp = final['components']

    # Gesamtverbrauch
    pump_energy = comp.get('pump1', {}).get('energy_consumed', 0)
    mixer_energy = comp.get('mix1', {}).get('energy_consumed', 0)
    total_parasitic = pump_energy + mixer_energy

    # BHKW-Produktion
    chp_energy = comp['chp1']['P_el'] * 30 * 24  # kWh

    # Parasitäre Last
    parasitic_fraction = total_parasitic / chp_energy if chp_energy > 0 else 0

    if parasitic_fraction > 0.10:
        print(f"⚠ Hohe parasitäre Last: {parasitic_fraction:.1%}")
        print("  → Intermittierendes Mischen aktivieren")
        print("  → Pumpendimensionierung prüfen")
```

## Optimale Betriebsbereiche - Schnellreferenz

### Fermenter

| Parameter | Optimal | Akzeptabel | Kritisch |
|-----------|---------|------------|----------|
| pH | 7.0-7.5 | 6.8-8.0 | <6.8 oder >8.0 |
| VFA [g/L] | 0.5-2.0 | 2.0-4.0 | >4.0 |
| VFA/TAC | 0.2-0.3 | 0.3-0.4 | >0.4 |
| TAC [g CaCO₃/L] | 5.0-10.0 | 4.0-12.0 | <4.0 |
| CH₄-Gehalt [%] | 58-62 | 55-65 | <55 |

Quelle: [Biological Components - Process Monitoring](components/biological.md#optimale-betriebsbereiche)

### Energiesystem

| Komponente | Optimaler Bereich | Warnung |
|-----------|------------------|---------|
| BHKW-Last | 80-100% | <40% oder >100% |
| Wärmenutzung | >70% | <50% |
| Gasspeicher | 30-70% | <20% oder >80% |
| Parasitäre Last | <10% der Produktion | >15% |

Quelle: [Energy Components - Performance Metrics](components/energy.md#leistungsmetriken)

### Mechanische Komponenten

| Komponente | Optimaler Betrieb | Ineffizient |
|-----------|------------------|-------------|
| Pumpe | 80-90% Q_nom | <50% oder >95% |
| Rührwerk (kontinuierlich) | 5 W/m³ | >8 W/m³ |
| Rührwerk (intermittierend) | 25% Einschaltzeit | >40% |
| Pumpenwirkungsgrad | >65% | <50% |

Quelle: [Mechanical Components - Dimensioning Guidelines](components/mechanical.md#dimensionierungsrichtlinien)

## Häufig gestellte Fragen (FAQ)

### Warum ist mein pH-Wert niedrig?

**Antwort**: Siehe [Biological Components - Niedriger pH-Wert](components/biological.md#niedriger-ph-wert)

Hauptursachen: Überfütterung, unzureichende Pufferung, plötzliche Substratänderungen

### Warum läuft mein BHKW nicht?

**Antwort**: Siehe [Energy Components - BHKW läuft nicht](components/energy.md#problem-bhkw-läuft-nicht)

Prüfen: Gasverfügbarkeit, Mindestgasbedarf, Speicherdruck

### Warum verbraucht mein Rührwerk so viel Energie?

**Antwort**: Siehe [Mechanical Components - Rührwerk verbraucht zu viel](components/mechanical.md#problem-rührwerk-verbraucht-zu-viel-energie)

Lösung: Intermittierenden Betrieb aktivieren (70% Einsparung)

### Warum verschlechtert sich meine Substratqualität schnell?

**Antwort**: Siehe [Feeding Components - Schneller Qualitätsverlust](components/feeding.md#problem-schneller-qualitätsverlust)

Prüfen: Lagertyp, Temperatur, Lagerzeit

### Warum ist meine Simulation langsam?

**Antwort**: Siehe [Quickstart - Slow Simulation](quickstart.md#issue-slow-simulation)

Lösungen: Zeitschritt erhöhen, save_interval reduzieren, parallele Simulation nutzen

## Weiterführende Ressourcen

### Komponenten-spezifische Dokumentation

- **[Biological Components](components/biological.md)**: Fermenter, Hydrolyse, Separatoren
- **[Energy Components](components/energy.md)**: BHKW, Heizung, Gasspeicher, Fackel
- **[Mechanical Components](components/mechanical.md)**: Pumpen, Rührwerke
- **[Feeding Components](components/feeding.md)**: Lager, Dosierer

### Allgemeine Anleitungen

- **[Installation Guide](installation.md)**: Setup und Plattform-spezifische Probleme
- **[Quickstart Guide](quickstart.md)**: Erste Schritte und grundlegende Simulation
- **[Components Overview](components/index.md)**: Architektur und Integrationsmuster

### Externe Ressourcen

- **GitHub Issues**: [PyADM1ODE Issues](https://github.com/dgaida/PyADM1ODE/issues)
- **Leitfaden Biogas**: FNR (2016) - Praktische Betriebsempfehlungen
- **ADM1 Documentation**: IWA - Wissenschaftliche Grundlagen

## Support

Wenn Sie in dieser Dokumentation keine Lösung finden:

1. **Prüfen Sie GitHub Issues**: [Existing Issues](https://github.com/dgaida/PyADM1ODE/issues)
2. **Erstellen Sie ein neues Issue** mit:
   - Betriebssystem und Version
   - Python-Version
   - Fehlermeldungen und Stack Traces
   - Minimales reproduzierbares Beispiel
3. **Kontakt**: daniel.gaida@th-koeln.de

## Checkliste für Problem-Reports

Bei der Meldung von Problemen bitte folgende Informationen angeben:

- [ ] **System-Info**: OS, Python-Version (`python --version`)
- [ ] **PyADM1-Version**: `import pyadm1; print(pyadm1.__version__)`
- [ ] **Problemkategorie**: Installation, Biologisch, Energie, Mechanisch, etc.
- [ ] **Fehlermeldung**: Vollständiger Stack Trace
- [ ] **Minimales Beispiel**: Code, der das Problem reproduziert
- [ ] **Erwartetes Verhalten**: Was sollte passieren
- [ ] **Tatsächliches Verhalten**: Was passiert stattdessen
- [ ] **Logs**: Relevante Ausgaben oder Logdateien
