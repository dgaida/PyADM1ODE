# Docstring-Styleguide

PyADM1ODE folgt dem **Google Python Style Guide** für Docstrings. Dies gewährleistet eine konsistente, lesbare und automatisch auswertbare Dokumentation.

## Allgemeines Format

```python
def funktion(arg1: int, arg2: str) -> bool:
    """
    Kurze Zusammenfassung.

    Erweiterte Beschreibung der Funktion und ihres Verhaltens.

    Args:
        arg1: Beschreibung von arg1.
        arg2: Beschreibung von arg2.

    Returns:
        Beschreibung des Rückgabewerts.

    Raises:
        ValueError: Wenn arg1 negativ ist.
    """
```

## Klassen

```python
class MeineKlasse:
    """
    Zusammenfassung.

    Erweiterte Beschreibung.

    Attributes:
        attr1: Beschreibung von attr1.
    """
```

## Werkzeuge

Wir verwenden `interrogate`, um die Docstring-Abdeckung zu erzwingen, und `mkdocstrings`, um die API-Referenz zu generieren.
Der aktuelle Schwellenwert liegt bei **95%**.
