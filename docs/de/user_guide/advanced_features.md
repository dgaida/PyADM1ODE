# Fortgeschrittene Funktionen

PyADM1ODE bietet fortgeschrittene Funktionen für umfangreiche Studien und das Konfigurationsmanagement.

## Parallele Simulation

Führen Sie mehrere Szenarien gleichzeitig aus, um Parameterstudien oder Monte-Carlo-Simulationen zu beschleunigen.

```python
from pyadm1.simulation import ParallelSimulator

# Parameterstudie
parallel = ParallelSimulator(adm1, n_workers=4)
scenarios = [
    {"k_dis": 0.5, "Q": [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]},
    {"k_dis": 0.6, "Q": [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]},
    {"k_dis": 0.7, "Q": [15, 10, 0, 0, 0, 0, 0, 0, 0, 0]}
]

results = parallel.run_scenarios(scenarios, duration=30, initial_state=state)
```

Weitere Details finden Sie im [Beispiel für parallele Simulation](../examples/parallel_simulation.md).

## Konfigurationsmanagement

Speichern und verwenden Sie Anlagendesigns mithilfe der JSON-Serialisierung wieder.

```python
# Konfiguration speichern
plant.to_json("zweistufige_anlage.json")

# Später laden
from pyadm1.configurator import BiogasPlant
plant = BiogasPlant.from_json("zweistufige_anlage.json", feedstock)
plant.initialize()
results = plant.simulate(duration=30, dt=1/24)
```
