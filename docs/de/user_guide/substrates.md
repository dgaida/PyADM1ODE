# Vorkonfigurierte Substrate

PyADM1ODE enthält 10 landwirtschaftliche Substrate mit in der Literatur validierten Parametern.

## Verfügbare Substrate

| Substrat | Typ | Typische Verwendung | Biogaspotenzial |
|----------|-----|--------------------|-----------------|
| **Maissilage** | Energiepflanze | Hauptsubstrat | Hoch (600-700 L/kg VS) |
| **Gülle** | Tierische Abfälle | Co-Substrat | Mittel (200-400 L/kg VS) |
| **Grünroggen** | Energiepflanze | Frühernte | Mittel-Hoch |
| **Grassilage** | Grünland | Erneuerbar | Mittel (400-550 L/kg VS) |
| **Weizen** | Getreide | Energiepflanze | Hoch |
| **GPS** | Ganzpflanzensilage | Ganze Pflanze | Hoch |
| **CCM** | Corn-Cob-Mix | Energiepflanze | Hoch |
| **Futterkalk** | Zusatzstoff | pH-Puffer | N/V |
| **Rindergülle** | Tierische Abfälle | Co-Substrat | Mittel (200-350 L/kg VS) |
| **Zwiebeln** | Abfall | Gemüseabfälle | Mittel-Hoch |

## Substratcharakterisierung

Alle Substrate sind charakterisiert durch:  
- Trockensubstanz (TS) und organische Trockensubstanz (oTS)  
- ADM1-Fraktionierung (Kohlenhydrate, Proteine, Lipide)  
- Biochemisches Methanpotenzial (BMP)  
- pH-Wert und Alkalinität  

Weitere Details zur Abbildung auf ADM1 finden Sie auf der Seite [ADM1-Implementierung](adm1_implementation.md).

## Substratmanagement

Substrate werden charakterisiert durch:
- **Weender-Analyse**: Rohfaser (RF), Rohprotein (RP), Rohfett (RL).
- **Van-Soest-Fraktionen**: NDF, ADF, ADL.
- **Physikalische Eigenschaften**: pH-Wert, TS, oTS, COD.
- **Kinetische Parameter**: Desintegrations- und Hydrolyseraten.

Diese Parameter ermöglichen eine dynamische Berechnung der ADM1-Zulaufwerte (wie $X_c$ und Stöchiometrie) und der kinetischen Parameter ($k_{dis}$, $k_{hyd}$) basierend auf den spezifischen Substrateigenschaften.
