# Substrate

Ein Substrat ist in PyADM1ODE eine Datei mit Laborwerten: Trockensubstanz,
Futtermittelanalyse, Methanpotenzial und der Zustand, in dem das Material in
den Fermenter gelangt. Aus diesen Werten berechnet das Modell den Zulauf des
Fermenters – wie viel abbaubares Material ankommt, wie schnell es abgebaut wird
und wie viel Stickstoff und Puffer es mitbringt.

```python
from pyadm1 import Feedstock

# Zwei Substrate der Bibliothek, gewählt über den Dateinamen ohne Endung
feedstock = Feedstock(["maize_silage_milk_ripeness", "cattle_manure"])
```

Die Dateien liegen unter
[`data/substrates/`](https://github.com/dgaida/PyADM1ODE/tree/master/data/substrates).
Die Mengen je Substrat (`Q_substrates`, in Tonnen Frischmasse pro Tag) gibt
man am Fermenter an, in der Reihenfolge der Liste oben.

## Was diese Sektion beantwortet

| Frage | Seite |
| --- | --- |
| Welche Beispiel-Substrate gibt es, und wofür eignet sich welches? | [Beispiel-Substrate](library.md) |
| Was bedeuten die Werte einer Substratdatei, und welche Bereiche sind realistisch? | [Parameter & Wertebereiche](parameters.md) |
| Was ändert sich in der Simulation, wenn ein Wert anders ist? | [Einfluss der Werte](influence.md) |
| Woher stammen die Werte der mitgelieferten Dateien? | [Herkunft der Werte](sources.md) |

!!! note "Richtwerte – für reale Anlagen eigene Substrate definieren"
    Die mitgelieferten Substrate beschreiben **typisches Material**, nicht das
    Substrat einer bestimmten Anlage. Sie eignen sich für Demos, Tests und
    Studien, **ersetzen aber keine gemessenen Daten**. Um eine reale Anlage zu
    modellieren, sollten Sie das eigene Substrat charakterisieren (TS / oTS,
    Weender- und Zellwandanalyse, BMP aus einem Batch-Test) und eine neue Datei
    unter `data/substrates/` anlegen. Die Seite
    [Parameter & Wertebereiche](parameters.md) zeigt, welche Werte dafür nötig
    sind.

## Ein eigenes Substrat in drei Schritten

1. Eine Bibliotheksdatei des ähnlichsten Materials kopieren, zum Beispiel  
   `maize_silage_milk_ripeness.yaml` nach `my_maize.yaml`.  
2. Die eigenen Laborwerte eintragen. Pflicht sind `TS`, `VS`, `RP`, `RL`, `pH`  
   und einer der drei Werte `BMP`, `D_VS` oder `aXI`; alles andere ist optional.  
3. Die Datei über ihren Namen oder ihren Pfad laden:  

```python
from pyadm1 import Feedstock
from pyadm1.substrates import load_substrate

feedstock = Feedstock(["my_maize", "cattle_manure"])       # liegt in data/substrates/
my_maize = load_substrate("pfad/zu/my_maize.yaml")          # oder über den Pfad
```

Ein falsch geschriebener Parametername führt zu einem Fehler, er wird nicht
stillschweigend übergangen.

Wie aus der Datei der 38-spaltige ADM1-Zulaufstrom entsteht, beschreibt die
Seite [ADM1-Implementierung](../user_guide/adm1_implementation.md).
