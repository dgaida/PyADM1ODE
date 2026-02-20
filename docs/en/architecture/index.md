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

## Three-Pass Simulation Process

To handle gas flow dependencies correctly, the simulation uses a three-pass model for each time step:

```mermaid
sequenceDiagram
    participant S as Simulator
    participant D as Digesters
    participant G as Gas Storage
    participant C as CHP / Consumers

    Note over S,C: Time Step (t -> t + dt)

    rect rgb(200, 220, 240)
    Note right of S: Pass 1: Production
    S->>D: step()
    D->>G: Add produced gas
    end

    rect rgb(220, 240, 200)
    Note right of S: Pass 2: Storage Update
    S->>G: step()
    G->>G: Update pressure & volume
    Note right of G: Vent if full
    end

    rect rgb(240, 220, 200)
    Note right of S: Pass 3: Consumption
    S->>C: step()
    C->>G: Request gas
    G-->>C: Supply actual available
    C->>C: Re-execute with actual supply
    end
```
