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

## Simulationsschritt in vier Durchläufen

Gasverbraucher hängen davon ab, was die Speicher liefern können, und
Heizsysteme von der Wärme des BHKW. `BiogasPlant.step` führt deshalb jeden
Zeitschritt in vier Durchläufen aus; jede Komponente wird genau einmal
berechnet.

```mermaid
sequenceDiagram
    participant S as BiogasPlant.step
    participant D as Fermenter und Fütterung
    participant G as Gasspeicher
    participant C as BHKW / Aufbereitung / Heizung
    participant F as Fackeln

    Note over S,F: Zeitschritt (t -> t + dt)

    rect rgb(200, 220, 240)
    Note right of S: Durchlauf 1: Produktion
    S->>D: step()
    D->>G: Produziertes Gas
    end

    rect rgb(220, 240, 200)
    Note right of S: Durchlauf 2: Speicher
    C->>G: Gasbedarf
    S->>G: step() mit Produktion und Bedarf
    Note right of G: Gas über der Kapazität wird abgelassen
    end

    rect rgb(240, 220, 200)
    Note right of S: Durchlauf 3: Verbraucher
    G-->>C: Tatsächlich geliefertes Gas
    S->>C: step(), Heizung nach dem BHKW
    end

    rect rgb(240, 230, 210)
    Note right of S: Durchlauf 4: Fackeln
    G-->>F: Abgelassenes Gas
    S->>F: step()
    end
```
