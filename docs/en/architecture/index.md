# Architecture

This page describes the system architecture and data flow of PyADM1ODE.

## System Overview

The framework is composed of several modular layers:

```mermaid
graph TD
    UI[User Interface / Scripts] --> Config[Plant Configurator]
    Config --> Builder[Plant Builder]
    Builder --> Components[Component System]

    subgraph Components
        Bio[Biological: Digesters]
        Energy[Energy: CHP, Storage, Flare]
        Mech[Mechanical: Pumps, Mixers]
        Feed[Feeding: Storage, Feeders]
    end

    Bio --> Core[Core ADM1 Engine]
    Core --> Params[ADM Parameters]
    Core --> Equations[Process Equations]
```

## Data Flow

```mermaid
graph LR
    Sub[Substrates] --> Feedstock[Feedstock Manager]
    Feedstock --> ADM_In[ADM1 Influent]
    ADM_In --> Digester[Digester Component]
    Digester --> Biogas[Biogas Production]
    Biogas --> Storage[Gas Storage]
    Storage --> CHP[CHP Unit]
    CHP --> Power[Electricity & Heat]
    Storage -- Overpressure --> Flare[Safety Flare]
```

## Four-Pass Simulation Step

Gas consumers depend on what the storages can supply, and heating systems on the
heat of the CHP. `BiogasPlant.step` therefore runs every time step in four
passes; each component is stepped exactly once.

```mermaid
sequenceDiagram
    participant S as BiogasPlant.step
    participant D as Digesters and feed chain
    participant G as Gas storages
    participant C as CHP / upgrading / heating
    participant F as Flares

    Note over S,F: Time Step (t -> t + dt)

    rect rgb(200, 220, 240)
    Note right of S: Pass 1: Production
    S->>D: step()
    D->>G: Produced gas
    end

    rect rgb(220, 240, 200)
    Note right of S: Pass 2: Storages
    C->>G: Gas demand
    S->>G: step() with production and demand
    Note right of G: Gas beyond capacity is vented
    end

    rect rgb(240, 220, 200)
    Note right of S: Pass 3: Consumers
    G-->>C: Gas actually supplied
    S->>C: step(), heating after the CHP
    end

    rect rgb(240, 230, 210)
    Note right of S: Pass 4: Flares
    G-->>F: Vented gas
    S->>F: step()
    end
```
