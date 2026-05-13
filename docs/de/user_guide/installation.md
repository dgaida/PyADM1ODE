# Installationsanleitung

Diese Anleitung behandelt die Installation von PyADM1ODE auf verschiedenen Betriebssystemen.

## Systemanforderungen

### Mindestanforderungen

- **Python**: 3.8 oder höher (3.10+ empfohlen, benötigt vom [fastmcp](https://github.com/jlowin/fastmcp)-Paket, das im optionalen Paket [PyADM1ODE_mcp](https://github.com/dgaida/PyADM1ODE_mcp) verwendet wird)
- **Betriebssystem**: Windows, Linux oder macOS
- **Arbeitsspeicher**: Mindestens 2 GB RAM (4 GB empfohlen)
- **Festplattenspeicher**: 10 MB für die Installation

PyADM1ODE ist reines Python ohne native Laufzeitabhängigkeiten — die Installation des Python-Pakets genügt.

## Installationsmethoden

### Methode 1: Installation über PyPI (Empfohlen, aber noch nicht unterstützt)

Sobald veröffentlicht, installieren Sie über pip:

```bash
pip install pyadm1ode
```

### Methode 2: Installation aus dem Quellcode

Für die Entwicklung oder die neuesten Funktionen:

```bash
# Repository klonen
git clone https://github.com/dgaida/PyADM1ODE.git
cd PyADM1ODE

# Im Entwicklungsmodus installieren
pip install -e .
```

### Methode 3: Verwendung von Conda

Erstellen Sie eine dedizierte Umgebung:

```bash
# Umgebung aus environment.yml erstellen
conda env create -f environment.yml

# Umgebung aktivieren
conda activate biogas

# PyADM1 installieren
pip install -e .
```

## Plattformspezifische Einrichtung

### Windows-Installation

1. **Python installieren** (falls noch nicht geschehen):
   - Von [python.org](https://www.python.org/downloads/) herunterladen.
   - Stellen Sie sicher, dass "Add Python to PATH" während der Installation aktiviert ist.

2. **PyADM1 installieren**:

   ```cmd
   pip install pyadm1ode  # pip noch nicht unterstützt
   # oder aus dem Quellcode:
   git clone https://github.com/dgaida/PyADM1ODE.git
   cd PyADM1ODE
   pip install -e .
   ```

3. **Installation überprüfen**:

   ```cmd
   python -c "import pyadm1; print(pyadm1.__version__)"
   ```

### Linux-Installation (Ubuntu/Debian)

1. **Python und Abhängigkeiten installieren**:

   ```bash
   sudo apt-get update
   sudo apt-get install python3 python3-pip
   ```

2. **PyADM1ODE installieren**:

   ```bash
   pip install pyadm1ode
   # oder aus dem Quellcode:
   git clone https://github.com/dgaida/PyADM1ODE.git
   cd PyADM1ODE
   pip install -e .
   ```

3. **Installation überprüfen**:

   ```bash
   python3 -c "import pyadm1; print(pyadm1.__version__)"
   ```

### macOS-Installation

1. **Homebrew installieren** (falls noch nicht geschehen):

   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Python installieren**:

   ```bash
   brew install python@3.11
   ```

3. **PyADM1ODE installieren**:

   ```bash
   pip3 install pyadm1ode
   # oder aus dem Quellcode:
   git clone https://github.com/dgaida/PyADM1ODE.git
   cd PyADM1ODE
   pip3 install -e .
   ```

4. **Installation überprüfen**:

   ```bash
   python3 -c "import pyadm1; print(pyadm1.__version__)"
   ```

## Kernabhängigkeiten

PyADM1 installiert diese Kernabhängigkeiten automatisch:

```text
numpy>=1.20.0         # Numerisches Rechnen
pandas>=1.3.0         # Datenmanipulation
scipy>=1.7.0          # Wissenschaftliches Rechnen
matplotlib>=3.5.0     # Plotten
```

## Optionale Abhängigkeiten

### Für die Entwicklung

```bash
pip install pytest pytest-cov black ruff mypy
```

## Installation überprüfen

### Schnellüberprüfung

Führen Sie dieses Python-Skript aus, um alle Komponenten zu überprüfen:

```python
#!/usr/bin/env python3
"""PyADM1-Installation überprüfen."""

def verify_installation():
    """Prüft alle PyADM1-Komponenten."""

    # 1. Core-Import prüfen
    try:
        import pyadm1
        print(f"✓ PyADM1 Version: {pyadm1.__version__}")
    except ImportError as e:
        print(f"✗ Import von pyadm1 fehlgeschlagen: {e}")
        return False

    # 2. Kernmodule prüfen
    try:
        from pyadm1.core import ADM1
        from pyadm1.substrates import Feedstock
        from pyadm1.simulation import Simulator
        print("✓ Kernmodule erfolgreich importiert")
    except ImportError as e:
        print(f"✗ Import der Kernmodule fehlgeschlagen: {e}")
        return False

    # 3. Feedstock aus der mitgelieferten Substratbibliothek laden
    try:
        feedstock = Feedstock(["maize_silage_milk_ripeness", "swine_manure"],
                              feeding_freq=24)
        print("✓ Feedstock aus data/substrates/ geladen")
    except Exception as e:
        print(f"✗ Feedstock-Laden fehlgeschlagen: {e}")
        return False

    # 4. Schneller Simulationstest
    try:
        from pyadm1.core.adm1 import ADM1, STATE_SIZE
        adm1 = ADM1(feedstock, V_liq=2000, T_ad=308.15)
        initial_state = [0.01] * STATE_SIZE  # 41-State ADM1da-Vektor
        adm1.create_influent([15, 10], 0)
        print(f"✓ Grundlegender Simulationsaufbau funktioniert ({STATE_SIZE} States)")
    except Exception as e:
        print(f"✗ Simulationstest fehlgeschlagen: {e}")
        return False

    print("\n✅ Alle Überprüfungen erfolgreich!")
    return True

if __name__ == "__main__":
    verify_installation()
```

Speichern Sie das Skript als `verify_install.py` und führen Sie es aus:

```bash
python verify_install.py
```

## Fehlerbehebung

### Häufige Probleme

#### 1. "ModuleNotFoundError: No module named 'pyadm1'"

**Problem**: Python findet das Paket nicht.

**Lösung**: Stellen Sie sicher, dass das Paket in der aktiven Umgebung installiert ist:

```bash
pip show pyadm1ode
# Falls leer, aus dem Quellcode installieren:
git clone https://github.com/dgaida/PyADM1ODE.git
cd PyADM1ODE
pip install -e .
```

#### 2. Substratdatei nicht gefunden

**Problem**: `Feedstock([...])` schlägt fehl, weil eine Substrat-ID unbekannt ist.

**Lösung**: Substrat-IDs sind die Dateinamen-Stämme unter `data/substrates/` (YAML, XML oder TOML — jedes unterstützte Format). Auflisten:

```bash
ls data/substrates/
```

Verwenden Sie die Dateinamen (ohne Endung) als Substrat-IDs.

### Hilfe erhalten

Falls Sie auf Probleme stoßen:

1. **GitHub Issues prüfen**: [PyADM1ODE Issues](https://github.com/dgaida/PyADM1ODE/issues)
2. **Neues Issue erstellen**: Folgende Informationen angeben:
   - Betriebssystem und Version
   - Python-Version (`python --version`)
   - Fehlermeldungen und Stack Traces
   - Ausgabe von `verify_install.py`

3. **Kontakt**: <daniel.gaida@th-koeln.de>

## Nächste Schritte

Nach erfolgreicher Installation:

1. **Schnellstart ausprobieren**: Siehe [Schnellstart-Anleitung](quickstart.md)
2. **Beispiele erkunden**: Siehe [Beispiel: Basis-Fermenter](../examples/basic_digester.md)
3. **Komponenten-Dokumentation lesen**: [Komponenten-Leitfaden](components/index.md)

## PyADM1ODE aktualisieren

### Update über PyPI (noch nicht unterstützt)

```bash
pip install --upgrade pyadm1
```

### Update aus dem Quellcode

```bash
cd PyADM1ODE
git pull origin master
pip install -e . --upgrade
```

## Optionale Pakete

### [PyADM1ODE_mcp](https://github.com/dgaida/PyADM1ODE_mcp) – Model-Context-Protocol-Server

Für LLM-gesteuerte Biogasanlagen-Modellierung mit natürlichsprachiger Schnittstelle:

```bash
# Aus GitHub installieren
git clone https://github.com/dgaida/PyADM1ODE_mcp.git
cd PyADM1ODE_mcp
pip install -e .
```

**Funktionen:**

- Natürlichsprachige Anlagenauslegung über LLM (z. B. Claude)
- MCP-Server für LLM-Integration
- Interaktive Anlagenkonfiguration

**Anwendungsfälle:** Anlagenauslegung für Nicht-Experten, schnelles Prototyping, Lehrwerkzeuge

Siehe [PyADM1ODE_mcp-Dokumentation](https://github.com/dgaida/PyADM1ODE_mcp) für Details.

### [PyADM1ODE_calibration](https://github.com/dgaida/PyADM1ODE_calibration) – Parameter-Kalibrierungs-Framework

Für automatische Modellkalibrierung anhand von Messdaten:

```bash
# Aus GitHub installieren
git clone https://github.com/dgaida/PyADM1ODE_calibration.git
cd PyADM1ODE_calibration
pip install -e .
```

**Funktionen:**

- Erstkalibrierung aus historischen Daten
- Online-Rekalibrierung während des Betriebs
- Mehrere Optimierungsalgorithmen (DE, PSO, Nelder-Mead)
- Umfassende Validierungsmetriken
- Datenbankanbindung für Messdaten

**Anwendungsfälle:** Modellparametrisierung, Anpassung an reale Anlagen, Unsicherheitsquantifizierung

Siehe [PyADM1ODE_calibration-Dokumentation](https://github.com/dgaida/PyADM1ODE_calibration) für Details.

## Deinstallation

PyADM1ODE entfernen (noch nicht unterstützt):

```bash
pip uninstall pyadm1ode
```

Auch Abhängigkeiten entfernen:

```bash
pip uninstall pyadm1ode numpy pandas scipy matplotlib
```

Optionale Pakete entfernen (noch nicht unterstützt):

```bash
pip uninstall pyadm1ode_mcp pyadm1ode_calibration
```
