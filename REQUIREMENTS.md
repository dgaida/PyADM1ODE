# Systemanforderungen und Lastenheft (PyADM1ODE)

Dieses Dokument beschreibt die funktionalen und nicht-funktionalen Anforderungen für das **PyADM1ODE**-Framework. Es dient als Referenz für Entwickler, Anwender und zur Qualitätssicherung des Systems.

---

## 1. Einleitung und Systemübersicht

### 1.1 Zweck des Frameworks
PyADM1ODE ist ein modulares, wissenschaftlich fundiertes Python-Framework zur Modellierung, Simulation und Optimierung von landwirtschaftlichen Biogasanlagen. Es basiert auf dem mathematischen Modell **Anaerobic Digestion Model No. 1 (ADM1)** und erweitert dieses um landwirtschaftliche Anpassungen (ADM1da).

### 1.2 Zielgruppe
- **Wissenschaftler und Forscher** im Bereich der Bioenergie und Verfahrenstechnik.
- **Ingenieure und Planer** von Biogasanlagen zur Auslegungs- und Betriebssimulation.
- **Softwareentwickler** für KI-gestützte Betriebsoptimierung und Agenten-basierte Steuerungen.

---

## 2. Funktionale Anforderungen (FA)

### 2.1 Core-Simulation und ADM1-Modell (FA-1)
- **FA-1.1: ODE-basierte Modellierung**
  Das System muss das ADM1-Modell als reines System gewöhnlicher Differentialgleichungen (ODE) ohne differentiell-algebraische Gleichungen (DAE) abbilden. Es umfasst mindestens 37 (bzw. erweitert 41) kontinuierliche Zustandsvariablen (gelöste Stoffe, partikuläre Fraktionen, Biomasse, Säure-Base-Ionen und Gasphasenkomponenten).
- **FA-1.2: Kinetisches Säure-Base-Gleichgewicht**
  Dissoziierte Ionenarten (Acetat-, Propionat-, Butyrat-, Valerat-Ionen, Hydrogencarbonat und Ammonium/Ammoniak) müssen über kinetische Reaktionen mit extrem hoher Geschwindigkeitskonstante ($k_{A,B} = 10^8 \text{m}^3 \text{kmol}^{-1} \text{d}^{-1}$) abgebildet werden, um ein dynamisches Gleichgewicht ohne algebraische Solver zu gewährleisten.
- **FA-1.3: Ladungsbilanzierte pH-Wert-Bestimmung**
  Der pH-Wert des Fermenters muss in jedem Integrationsschritt aus der Ladungsbilanz der kinetischen Ionen über ein schnelles, numerisch robustes Newton-Raphson-Verfahren berechnet werden.
- **FA-1.4: Zwei-Pool-Desintegration (ADM1da)**
  Die Desintegration organischer Feststoffe muss in einen schnellen Pool ($X_{PF}$ mit $k_{dis} \approx 0.4 \text{d}^{-1}$) und einen langsamen Pool ($X_{PS}$ mit $k_{dis} \approx 0.04 \text{d}^{-1}$) aufgeteilt sein, um leicht- und schwerverdauliche Organik realitätsnah abzubilden.
- **FA-1.5: Biomasse-Zerfallsstöchiometrie**
  Der Zerfall von Biomassepopulationen muss direkt in hydrolisierbare Kohlenhydrate, Proteine und Fette ($X_S$) sowie inerte organische Feststoffe ($X_I$) geroutet werden (nach Wett et al. 2006).
- **FA-1.6: Temperaturabhängigkeit (Arrhenius)**
  Sämtliche kinetischen Raten und Halbsättigungskonstanten müssen über eine thermodynamische Arrhenius-Beziehung temperaturabhängig korrigiert werden.
- **FA-1.7: Erweiterte Inhibitionskinetiken**
  Das System muss landwirtschaftliche Hemmungen abbilden:
  - Steilere pH-Inhibition der methanogenen Gruppen (Hill-Exponenten $n=2$ und $n=3$).
  - Ammoniak-Hemmung ($S_{NH3}$) auf Essigsäure- und Propionsäure-Verwerter mittels quadratischer Hill-Kinetik.
  - Hemmung durch undissoziierte Säuren (Propionsäure auf $X_{pro}$ und Essigsäure auf $X_{ac}$).
  - Stickstofflimitierung basierend auf der Summe der anorganischen Stickstoff-Spezies ($S_{IN}$).
- **FA-1.8: Gas-Flüssig-Phasentransfer**
  Der Transfer der Gase ($	ext{CH}_4, \text{CO}_2, \text{H}_2, \text{NH}_3$) in den Gasraum muss kinetisch über volumenspezifische Stoffdurchgangskoeffizienten ($k_L a = 200 \text{d}^{-1}$) und temperaturkorrigierte Henry-Konstanten abgebildet werden.
- **FA-1.9: Schlammvolumen- und HRT-Bilanzierung**
  Das System muss eine dynamische Massen- und Volumenbilanz des Flüssigkeitsraums bereitstellen. Massenverluste durch Gasfreisetzung müssen das Reaktorvolumen verändern, und der Abfluss muss über eine Wehr-Gleichung sowie eine Verzögerung 1. Ordnung für die Verweilzeit (HRT) modelliert werden.

### 2.2 Komponentenbibliothek (FA-2)
- **FA-2.1: Biologische Komponenten**
  - **Fermenter (Digester)**: Kontinuierlich gerührter Reaktor (CSTR) mit Flüssig- und Gasvolumen, Heizmantel und biologischen Abbauprozessen.
  - **Separator**: Phasentrennung des Gärrests in feste und flüssige Phasen basierend auf Feststoff-Rückhaltegraden.
- **FA-2.2: Energiekomponenten**
  - **Blockheizkraftwerk (BHKW/CHP)**: Wandelt produziertes Biogas in elektrische und thermische Energie um. Muss Teillastverhalten und minimalen Gasqualitätsbedarf unterstützen.
  - **Heizkessel (Boiler)**: Erzeugt Hilfswärme zur Einhaltung der Solltemperatur bei unzureichender BHKW-Wärme.
  - **Gasspeicher (Gas Storage)**: Puffert Biogas unter variablem Druck und verfügt über eine automatische Überdruckfackel (Flare) zum schadlosen Abblasen.
  - **Biogas-Aufbereitungsanlage**: Trennt Methan und Kohlendioxid zur Biomethan-Netzeinspeisung.
- **FA-2.3: Fütterung und Materialfluss**
  - **Substrat-Lager (Substrate Storage)**: Speichert Rohsubstrate ohne biologische Aktivität.
  - **Dosierer (Feeder)**: Dosiert Feststoffe und Flüssigkeiten zeitgesteuert oder intervallbasiert in den Fermenter.
- **FA-2.4: Mechanische Komponenten**
  - **Pumpen (Pumps)**: Fördern Flüssigkeiten und Suspensionen zwischen Komponenten mit volumetrischer Ratensteuerung.
  - **Rührwerke (Mixers)**: Sichern die Homogenität im Fermenter und verursachen einen elektrischen Eigenenergiebedarf.
- **FA-2.5: Sensorik**
  - **Physikalische Sensoren**: Erfassung von Füllstand, Temperatur, Volumenströmen und Druck.
  - **Chemische Sensoren**: Erfassung von pH-Wert, VFA-Konzentration, TAC (Säurekapazität) und Ammoniumgehalt.
  - **Gassensoren**: Analyse der Methan- und Kohlendioxidkonzentration sowie des Gasvolumenstroms.

### 2.3 Substratmanagement (FA-3)
- **FA-3.1: Weender- und Van-Soest-Charakterisierung**
  Die Konvertierung realer Substrate in ADM1-Zustandskoordinaten muss auf Laboranalysen basieren (Trockensubstanz, organische Trockensubstanz, Rohfaser, Rohprotein, Rohfett, Rohasche, Ammonium und flüchtige Säuren).
- **FA-3.2: Automatisches Routing und Dissoziation**
  Der Feedstock-Manager muss die Laborfraktionen mathematisch auf die ADM1-Zustände (z.B. $X_{PS\_ch}, X_{PF\_pr}, X_I$) verteilen und die Ionenkonzentrationen basierend auf dem Substrat-pH-Wert vorkalkulieren.
- **FA-3.3: Datenformate für Substrate**
  Substratdatenbanken müssen flexibel über strukturierte Textformate (YAML, XML, TOML) geladen, modifiziert und gespeichert werden können.
- **FA-3.4: Volumenstrom-Konventionen**
  Das System muss massenäquivalente Volumenströme (SIMBA#-Konvention) unterstützen, um die Dichte von Maissilage und Gülle korrekt zu berücksichtigen, sowie rein volumetrische Ströme erlauben.

### 2.4 Anlagenkonfiguration und Simulation (FA-4)
- **FA-4.1: Programmatischer Konfigurator**
  Ein modularer `PlantConfigurator` muss es ermöglichen, eine Biogasanlage im Python-Code durch Instanziierung und softwareseitiges Verbinden der Ein- und Ausgänge von Komponenten flexibel aufzubauen.
- **FA-4.2: Drei-Pass-Simulationszyklus**
  Zur korrekten zeitlichen Auflösung rückwirkender Stoff- und Energieflüsse (z.B. Gasproduktion -> Speicherdruck -> BHKW-Verbrauch) muss jeder Simulationsschritt in drei sequentiellen Schritten berechnet werden:
  1. *Pass 1 (Produktion)*: Biologische und mechanische Komponenten berechnen ihren Zustandsschritt.
  2. *Pass 2 (Speicher)*: Pufferspeicher aktualisieren Drücke und Füllstände.
  3. *Pass 3 (Verbrauch)*: Verbraucher fordern Medien an und korrigieren ggf. ihre Leistung.
- **FA-4.3: Parallele Simulation**
  Das System muss über einen `ParallelSimulator` die gleichzeitige Ausführung von hunderten Simulationen (z.B. für Sensitivitätsanalysen oder Monte-Carlo-Simulationen) auf Multi-Core-CPUs ermöglichen und aggregierte Statistiken (Erfolgsquote, Fehlerraten, durchschnittliche Verarbeitungszeit) zurückliefern.

### 2.5 Benchmark-Datensatz und Oracle (FA-5)
- **FA-5.1: Validierungs- und Benchmark-Framework**
  Das System muss standardisierte Datenpunkte und Szenarien bereitstellen, um die Qualität von Optimierungsalgorithmen vergleichen zu können.
- **FA-5.2: Oracle-Bewertungssystem**
  Ein integriertes Oracle-Modul muss Simulationsergebnisse bewerten (z.B. biologische Stabilität, Methanausbeute, Wirtschaftlichkeit) und einen normierten Score ausgeben.

---

## 3. Nicht-funktionale Anforderungen (NFA)

### 3.1 Performance und Skalierbarkeit (NFA-1)
- **NFA-1.1: Adaptive ODE-Integration**
  Das System muss moderne, adaptive Schrittweitensteuerungen (z.B. BDF - Backward Differentiation Formula für steife Systeme) über SciPy nutzen, um eine 30-Tage-Simulation eines komplexen Fermenters in unter 1,0 Sekunden Rechenzeit auf Standard-CPUs auszuführen.
- **NFA-1.2: Parallelisierungseffizienz**
  Der `ParallelSimulator` muss eine nahezu lineare Skalierung mit der Anzahl der physischen CPU-Kerne aufweisen, um umfangreiche Parametersuchen effizient zu bewältigen.

### 3.2 Zuverlässigkeit, Genauigkeit und Robustheit (NFA-2)
- **NFA-1.1: Quantitative Validierung**
  Die Simulationsergebnisse (pH-Wert, Gaskonzentrationen, Abbauraten) müssen eine Abweichung von maximal 1% gegenüber der etablierten Referenzsoftware **SIMBA# biogas 4.2** aufweisen.
- **NFA-1.2: Numerische Stabilität**
  Durch die Formulierung als rein kinetisches ODE-System anstelle eines DAE-Systems muss der Solver auch bei extremen Schockbelastungen (z.B. plötzliche Überfütterung, saure Milieus, pH < 5) stabil konvergieren, ohne numerische Singularitäten zu erzeugen.
- **NFA-1.3: Fehlerbehandlung (Inputs)**
  Ungültige physikalische Parameter (z.B. negative Fütterungsmengen, Temperaturen außerhalb realistischer Bereiche) müssen von den Komponenten und Validierungsprüfungen abgefangen und mit aussagekräftigen Ausnahmen (`ValueError`) quittiert werden.

### 3.3 Interoperabilität und Plattformkompatibilität (NFA-3)
- **NFA-3.1: .NET C#-Integration via pythonnet**
  Das Framework nutzt hochoptimierte, in C# geschriebene Physico-Chemicals-DLLs (`biogas.dll`). Die Kommunikation über `pythonnet` muss stabil und typsicher erfolgen. Zur Gewährleistung der Dimensionalität müssen 1D-Python-Listen nativer Floats für Methoden wie `calcPHOfADMstate` genutzt werden.
- **NFA-3.2: Plattformübergreifende Portabilität**
  Das System muss nativ unter Windows und Linux (Debian, Ubuntu) lauffähig sein. Unter Linux-Systemen und CI/CD-Pipelines muss die Ausführung durch automatische Erkennung und Nutzung einer Mono-Laufzeitumgebung (`mono-complete`) sichergestellt sein.
- **NFA-3.3: Docker-Unterstützung**
  Ein optimiertes Dockerfile basierend auf `python:3.11-slim` mit vorinstallierter Mono-Laufzeit muss bereitgestellt werden, um eine reproduzierbare Container-Umgebung bereitzustellen.
- **NFA-3.4: Cloud-Kompatibilität (Google Colab)**
  Das Framework und alle Beispiele müssen nahtlos in Google Colab ausführbar sein, indem externe Abhängigkeiten (Mono) und das Python-Paket dynamisch installiert werden können.

### 3.4 Usability und Dokumentation (NFA-4)
- **NFA-4.1: Bilingualität (DE/EN)**
  Die gesamte Online-Dokumentation, Tutorials und Installationsanleitungen müssen vollständig synchronisiert in deutscher und englischer Sprache über das `mkdocs-static-i18n`-Plugin bereitgestellt werden.
- **NFA-4.2: Interaktive Tutorials**
  Mindestens zwei voll funktionsfähige Jupyter Notebooks (Einstufiger Basis-Fermenter, Komplexe zweistufige Anlage mit BHKW und Gasspeicher) müssen als Direkteinstieg mit "Open in Colab"-Badges in der Dokumentation verankert sein.
- **NFA-4.3: API-Dokumentation**
  Die API-Referenz muss automatisch aus den Python-Quelltexten generiert werden. Alle öffentlichen Methoden und Klassen müssen vollständige Docstrings im standardisierten Google-Style aufweisen.
- **NFA-4.4: Visualisierung**
  Simulationsergebnisse müssen einfach über integrierte Plot-Funktionen oder standardisierte Datenexporte (z.B. Pandas DataFrames) visualisiert werden können.

### 3.5 Wartbarkeit und Code-Qualität (NFA-5)
- **NFA-5.1: Testabdeckung**
  Das Projekt muss eine automatisierte Unit- und Integrationstest-Suite mit einer Mindest-Code-Abdeckung (Coverage) von **80%** aufweisen.
- **NFA-5.2: Automatische API-Abdeckung (interrogate)**
  Die Docstring-Abdeckung des gesamten Frameworks wird per `interrogate` geprüft und muss einen Schwellenwert von **95%** überschreiten, um im CI-Prozess erfolgreich zu bauen.
- **NFA-5.3: Automatisches Versionierungs- und Changelog-System**
  Versionen werden automatisch über ein CI-Workflow-Skript erhöht (`auto-version.yml`). Das Changelog (`CHANGELOG.md`) wird automatisch mittels `git-cliff` und `cliff.toml` aus strukturierten Git-Commit-Nachrichten generiert.
- **NFA-5.4: Formatierungs- und Linting-Standards**
  Der Quelltext muss strikt PEP-8-konform sein. Dies wird über Pre-Commit-Hooks und CI-Workflows mit **Black** (Code-Formatter) und **Ruff** (Linter) erzwungen.
- **NFA-5.5: Dokumentationsprüfung (Markdown-Listen)**
  Um Darstellungsfehler in der gebauten Dokumentation zu vermeiden, müssen alle Markdown-Dateien automatisch auf das Vorhandensein von zwei Leerzeichen am Ende von Listen-Zeilen überprüft und ggf. korrigiert werden (`fix_markdown_lists.py`).
