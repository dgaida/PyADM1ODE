# Installationsanleitung

Diese Anleitung behandelt die Installation von PyADM1ODE auf verschiedenen Betriebssystemen.

## Systemanforderungen

### Mindestanforderungen
- **Python**: 3.8 oder höher (3.10+ empfohlen)
- **Betriebssystem**: Windows, Linux oder macOS
- **Arbeitsspeicher**: Mindestens 2 GB RAM (4 GB empfohlen)
- **Festplattenspeicher**: 10 MB für die Installation

### Laufzeitanforderungen
PyADM1ODE verwendet C#-DLLs für die Substratcharakterisierung, was Folgendes erfordert:
- **Linux/macOS**: Mono-Laufzeitumgebung
- **Windows**: .NET Framework (normalerweise vorinstalliert)

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
   git clone https://github.com/dgaida/PyADM1ODE.git
   cd PyADM1ODE
   pip install -e .
   pip install -r requirements-windows.txt
   ```

3. **.NET Framework** sollte unter Windows 10/11 vorinstalliert sein.

### Linux-Installation (Ubuntu/Debian)

1. **Python und Abhängigkeiten installieren**:
   ```bash
   sudo apt-get update
   sudo apt-get install python3 python3-pip
   ```

2. **Mono-Laufzeitumgebung installieren** (erforderlich für C#-DLLs):
   ```bash
   sudo apt-get install mono-complete
   mono --version
   ```

3. **PyADM1ODE installieren**:
   ```bash
   git clone https://github.com/dgaida/PyADM1ODE.git
   cd PyADM1ODE
   pip install -e .
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

3. **Mono-Laufzeitumgebung installieren**:
   ```bash
   brew install mono
   mono --version
   ```

4. **PyADM1ODE installieren**:
   ```bash
   git clone https://github.com/dgaida/PyADM1ODE.git
   cd PyADM1ODE
   pip3 install -e .
   ```

## Verifizierung der Installation

Führen Sie folgendes Skript aus, um die Installation zu prüfen:

```python
import pyadm1
print(f"PyADM1 Version: {pyadm1.__version__}")
from pyadm1.core import ADM1
print("Core-Module erfolgreich geladen.")
```
