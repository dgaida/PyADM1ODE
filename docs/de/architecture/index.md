# Architektur

Diese Seite beschreibt die Systemarchitektur und den Datenfluss von PyADM1ODE.

## Systemübersicht

Das Framework besteht aus mehreren modularen Schichten:

```mermaid
graph TD
    UI[Benutzeroberfläche / Skripte] --> Config[Anlagenkonfigurator]
    Config --> Builder[Plant Builder]
    Builder --> Components[Komponentensystem]

    subgraph Components
        Bio[Biologisch: Fermenter]
        Energy[Energie: BHKW, Speicher, Fackel]
        Mech[Mechanisch: Pumpen, Rührwerke]
        Feed[Fütterung: Lagerung, Dosierer]
    end

    Bio --> Core[Kern-ADM1-Engine]
    Core --> Params[ADM-Parameter]
    Core --> Equations[Prozessgleichungen]
```

## Datenfluss

```mermaid
graph LR
    Sub[Substrate] --> Feedstock[Feedstock-Manager]
    Feedstock --> ADM_In[ADM1-Zulauf]
    ADM_In --> Digester[Fermenter-Komponente]
    Digester --> Biogas[Biogasproduktion]
    Biogas --> Storage[Gasspeicher]
    Storage --> CHP[BHKW-Einheit]
    CHP --> Power[Strom & Wärme]
    Storage -- Überdruck --> Flare[Sicherheitsfackel]
```

## Drei-Pass-Simulationsprozess

Um Gasflussabhängigkeiten korrekt zu behandeln, verwendet die Simulation ein Drei-Pass-Modell für jeden Zeitschritt:

```mermaid
sequenceDiagram
    participant S as Simulator
    participant D as Fermenter
    participant G as Gasspeicher
    participant C as BHKW / Verbraucher

    Note over S,G: Zeitschritt (t -> t + dt)

    rect rgb(200, 220, 240)
    Note right of S: Pass 1: Produktion
    S->>D: step()
    D->>G: Produziertes Gas hinzufügen
    end

    rect rgb(220, 240, 200)
    Note right of S: Pass 2: Speicher-Update
    S->>G: step()
    G->>G: Druck & Volumen aktualisieren
    Note right of G: Abblasen falls voll
    end

    rect rgb(240, 220, 200)
    Note right of S: Pass 3: Verbrauch
    S->>C: step()
    C->>G: Gas anfordern
    G-->>C: Tatsächlich verfügbares Gas liefern
    C->>C: Neu ausführen mit tatsächlichem Angebot
    end
```
