# Vorkonfigurierte Substrate

PyADM1ODE bringt 12 Beispiel-Substratcharakterisierungen für ADM1da unter
[`data/substrates/`](https://github.com/dgaida/PyADM1ODE/tree/master/data/substrates)
mit. Jede Datei ist eine YAML-Abbildung der in
[`SubstrateParams`](adm1_implementation.md) aufgeführten Parameter und wird
über das `SubstrateRegistry` per Dateinamen-Stamm geladen.

!!! note "Nur Beispiele — für reale Anlagen eigene Substrate definieren"
    Die mitgelieferten Substrate sind **Beispiele** aus der Literatur und
    einer älteren Anlagen-Bibliothek. Sie eignen sich für schnelle Demos,
    Smoke-Tests und das Reproduzieren der Validierungsläufe, **ersetzen
    aber keine gemessenen Daten**. Um eine reale Anlage zu modellieren,
    sollten Sie das eigene Substrat charakterisieren (Weender-Analyse,
    TS / oTS, BMP aus einem Batch-Test) und eine neue Datei unter
    `data/substrates/` in einem der unterstützten Formate anlegen.

## Verfügbare Substrate

Sortiert in der kanonischen Standardreihenfolge — beim Aufruf von
`Feedstock()` ohne Argumente landen die Substrate genau in dieser
Reihenfolge im `Q`-Array (`Q[0]` ist die erste Zeile).

| Q-Index | Substrat-ID | Anzeigename | Typ | TS [kg/m³ FM] | BMP [Nm³ CH₄/t VS] |
| --- | --- | --- | --- | --- | --- |
| 0 | `maize_silage_milk_ripeness` | Maissilage (Milchreife) | Energiepflanze | 330 | 357 |
| 1 | `cattle_manure` | Rindergülle | Tierische Abfälle | 80 | 137 |
| 2 | `swine_manure` | Schweinegülle | Tierische Abfälle | 80 | 149 |
| 3 | `corn_cob_mix` | Corn-Cob-Mix (CCM) | Energiepflanze | 676 | 426 |
| 4 | `grass_silage` | Grassilage | Energiepflanze | 341 | 338 |
| 5 | `green_rye_silage` | Grünroggensilage | Energiepflanze | 193 | 322 |
| 6 | `cereal_gps_silage` | Getreide-Ganzpflanzensilage (GPS) | Energiepflanze | 312 | 290 |
| 7 | `onion_waste` | Zwiebelabfälle | Gemüseabfall | 193 | 300 |
| 8 | `maize_silage_gummersbach` | Maissilage (Anlage Gummersbach) | Energiepflanze | 320 | 348 |
| 9 | `cattle_manure_solid` | Rinderfestmist | Tierische Abfälle | 120 | 282 |
| 10 | `swine_manure_gummersbach` | Schweinegülle (Anlage Gummersbach) | Tierische Abfälle | 61 | 203 |
| 11 | `wheat_whole_plant_silage` | Weizen-Ganzpflanzensilage | Energiepflanze | 302 | 298 |

Die Einträge `*_gummersbach` und `cattle_manure_solid` wurden aus der
älteren [`substrate_gummersbach.xml`](https://github.com/dgaida/PyADM1ODE/blob/master/data/substrates/legacy/substrate_gummersbach.xml)-Bibliothek
über eine Buswell-basierte Weender-zu-BMP-Zuordnung konvertiert. Die
übrigen stammen aus der Substratbibliothek von ifak Magdeburg.

## Dateiformate

Jedes Substrat kann als YAML (kanonisch), XML oder TOML definiert werden.
Der Loader wählt anhand der Dateiendung das passende Format aus und liefert
in jedem Fall dasselbe `SubstrateParams`-Objekt; in
[`data/substrates/examples/`](https://github.com/dgaida/PyADM1ODE/tree/master/data/substrates/examples)
liegen Side-by-Side-Beispiele desselben Substrats in allen drei Formaten.

```python
from pyadm1.substrates import SubstrateRegistry, load_substrate

# Per ID — erkennt jedes unterstützte Format in data/substrates/
reg = SubstrateRegistry()
maize = reg.get("maize_silage_milk_ripeness")

# Per expliziten Pfad — die Endung bestimmt den Loader
swine = load_substrate("data/substrates/swine_manure.yaml")
```

## Substratcharakterisierung

Jedes Substrat trägt denselben Parametersatz:

- **Rohzusammensetzung** — `TS`, `NH4`, Biogas- / Biomethanpotenzial `BGP`, `BMP`.
- **Weender-Analyse** — Rohfaser-, Rohprotein-, Rohfett- und Asche-Anteile der TS (`fRF`, `fRP`, `fRFe`, `fRA`).
- **COD-Fraktionierung** — partikulär- und löslich-inerter COD-Anteil (`aXI`, `aSi`), abbaubarer Anteil der Rohfaser (`fOTSrf`), Aufteilung in langsame/schnelle Desintegrationspools (`fsOTS`, `ffOTS`).
- **Physikalisch-chemischer Zustand** — Substrattemperatur, `pH`, Säurekapazität bis pH 4,3 (`KS43`), VFA als Essigsäure-Äquivalent (`FFS`).

Komponentendichten, Masse-zu-COD-Umrechnungsfaktoren, Methanpotenziale und
die Säure-Base-Gleichgewichtskonstanten sind als Modelldefaults in
`SubstrateParams` hinterlegt und müssen pro Substrat nur dann angegeben
werden, wenn eine Messung das Überschreiben rechtfertigt.

Die Seite [ADM1-Implementierung](adm1_implementation.md) beschreibt, wie aus
der Charakterisierung der 38-spaltige ADM1-Zulaufstrom erzeugt wird.
