# Installation Guide

This guide covers the installation of PyADM1ODE on different operating systems.

## System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher (3.10+ recommended, needed by [fastmcp](https://github.com/jlowin/fastmcp) package, used in the optional package [PyADM1ODE_mcp](https://github.com/dgaida/PyADM1ODE_mcp))
- **Operating System**: Windows, Linux, or macOS
- **Memory**: 2 GB RAM minimum (4 GB recommended)
- **Disk Space**: 10 MB for installation

### Runtime Requirements
PyADM1ODE uses C# DLLs for substrate characterization, which requires:  
- **Linux/macOS**: Mono runtime  
- **Windows**: .NET Framework (usually pre-installed)

## Installation Methods

### Method 1: Install from PyPI (Recommended, but not yet supported)

Once released, install via pip:

```bash
pip install pyadm1ode
```

### Method 2: Install from Source

For development or the latest features:

```bash
# Clone the repository
git clone https://github.com/dgaida/PyADM1ODE.git
cd PyADM1ODE

# Install in development mode
pip install -e .
```

### Method 3: Using Conda

Create a dedicated environment:

```bash
# Create environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate biogas

# Install PyADM1
pip install -e .
```

## Platform-Specific Setup

### Windows Installation

1. **Install Python** (if not already installed):
   - Download from [python.org](https://www.python.org/downloads/)
   - Ensure "Add Python to PATH" is checked during installation

2. **Install PyADM1**:
   ```cmd
   pip install pyadm1ode  # pip not yet supported
   # or from source:
   git clone https://github.com/dgaida/PyADM1ODE.git
   cd PyADM1ODE
   pip install -e .

   # Windows-specific dependencies
   pip install -r requirements-windows.txt
   ```

3. **.NET Framework** should be pre-installed on Windows 10/11. If needed:
   - Download from [Microsoft .NET Framework](https://dotnet.microsoft.com/download/dotnet-framework)

4. **Verify Installation**:
   ```cmd
   python -c "import pyadm1; print(pyadm1.__version__)"
   ```

### Linux Installation (Ubuntu/Debian)

1. **Install Python and dependencies**:
   ```bash
   sudo apt-get update
   sudo apt-get install python3 python3-pip
   ```

2. **Install Mono runtime** (required for C# DLLs):
   ```bash
   sudo apt-get install mono-complete
   mono --version
   ```

3. **Install PyADM1ODE**:
   ```bash
   pip install pyadm1ode
   # or from source:
   git clone https://github.com/dgaida/PyADM1ODE.git
   cd PyADM1ODE
   pip install -e .
   ```

4. **Verify Installation**:
   ```bash
   python3 -c "import pyadm1; print(pyadm1.__version__)"
   ```

### macOS Installation

1. **Install Homebrew** (if not already installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install Python**:
   ```bash
   brew install python@3.11
   ```

3. **Install Mono runtime**:
   ```bash
   brew install mono
   mono --version
   ```

4. **Install PyADM1ODE**:
   ```bash
   pip3 install pyadm1ode
   # or from source:
   git clone https://github.com/dgaida/PyADM1ODE.git
   cd PyADM1ODE
   pip3 install -e .
   ```

5. **Verify Installation**:
   ```bash
   python3 -c "import pyadm1; print(pyadm1.__version__)"
   ```

## Core Dependencies

PyADM1 automatically installs these core dependencies:

```
pythonnet>=3.0.0      # .NET interop for C# DLLs
numpy>=1.20.0         # Numerical computing
pandas>=1.3.0         # Data manipulation
scipy>=1.7.0          # Scientific computing
matplotlib>=3.5.0     # Plotting
```

## Optional Dependencies

### For Development
```bash
pip install pytest pytest-cov black ruff mypy
```

## Verifying Your Installation

### Quick Verification

Run this Python script to verify all components:

```python
#!/usr/bin/env python3
"""Verify PyADM1 installation."""

def verify_installation():
    """Check all PyADM1 components."""

    # 1. Check core import
    try:
        import pyadm1
        print(f"✓ PyADM1 version: {pyadm1.__version__}")
    except ImportError as e:
        print(f"✗ Failed to import pyadm1: {e}")
        return False

    # 2. Check core modules
    try:
        from pyadm1.core import ADM1
        from pyadm1.substrates import Feedstock
        from pyadm1.simulation import Simulator
        print("✓ Core modules imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import core modules: {e}")
        return False

    # 3. Check .NET/Mono runtime
    try:
        import clr
        print("✓ pythonnet (CLR) available")
    except ImportError:
        print("✗ pythonnet not available")
        return False

    # 4. Check C# DLLs
    try:
        feedstock = Feedstock(feeding_freq=48)
        substrates = feedstock.mySubstrates()
        n_substrates = substrates.getNumSubstrates()
        print(f"✓ C# DLLs working ({n_substrates} substrates loaded)")
    except Exception as e:
        print(f"✗ C# DLL access failed: {e}")
        return False

    # 5. Quick simulation test
    try:
        from pyadm1.core.adm1 import ADM1
        adm1 = ADM1(feedstock, V_liq=2000, T_ad=308.15)
        initial_state = [0.01] * 37
        adm1.create_influent([15, 10, 0, 0, 0, 0, 0, 0, 0, 0], 0)
        print("✓ Basic simulation setup works")
    except Exception as e:
        print(f"✗ Simulation test failed: {e}")
        return False

    print("\n✅ All verification checks passed!")
    return True

if __name__ == "__main__":
    verify_installation()
```

Save as `verify_install.py` and run:
```bash
python verify_install.py
```

## Troubleshooting

### Common Issues

#### 1. "Cannot find C# DLLs"

**Problem**: Python can't locate the C# DLL files.

**Solution**:
```bash
# Verify DLL files exist
ls pyadm1/dlls/

# Should show: plant.dll, substrates.dll, biogas.dll, physchem.dll
```

If missing, reinstall from source:
```bash
git clone https://github.com/dgaida/PyADM1ODE.git
cd PyADM1ODE
pip install -e .
```

#### 2. "pythonnet import error"

**Problem**: pythonnet fails to import or find .NET runtime.

**Linux/macOS Solution**:
```bash
# Install Mono
sudo apt-get install mono-complete  # Ubuntu/Debian
brew install mono                    # macOS

# Verify Mono
mono --version
```

**Windows Solution**:
```cmd
# Install .NET Framework
# Download from: https://dotnet.microsoft.com/download/dotnet-framework
```

#### 3. "Module 'biogas' has no attribute..."

**Problem**: C# DLL methods not accessible.

**Solution**: This usually indicates Mono/CLR issues.
```bash
# Reinstall pythonnet
pip uninstall pythonnet
pip install pythonnet>=3.0.0

# Restart Python interpreter
```

#### 4. Import errors on first run

**Problem**: First import takes long or fails.

**Solution**: pythonnet needs to compile CLR bindings on first run:
```python
import clr  # First import may take 10-30 seconds
# Subsequent imports will be fast
```

### Getting Help

If you encounter issues:

1. **Check GitHub Issues**: [PyADM1ODE Issues](https://github.com/dgaida/PyADM1ODE/issues)
2. **Create New Issue**: Include:
   - Operating system and version
   - Python version (`python --version`)
   - Error messages and stack traces
   - Output from `verify_install.py`

3. **Contact**: daniel.gaida@th-koeln.de

## Next Steps

After successful installation:

1. **Try the Quickstart**: See [Quickstart Guide](quickstart.md)
2. **Explore Examples**: See [Example: Basic Digester](../examples/basic_digester.md)
3. **Read Component Documentation**: [Components Guide](components/index.md)

## Updating PyADM1ODE

### Update from PyPI (not yet supported)
```bash
pip install --upgrade pyadm1
```

### Update from Source
```bash
cd PyADM1ODE
git pull origin master
pip install -e . --upgrade
```

## Optional Packages

### [PyADM1ODE_mcp]((https://github.com/dgaida/PyADM1ODE_mcp)) - Model Context Protocol Server

For LLM-driven biogas plant modeling with natural language interface:

```bash
# Install from GitHub
git clone https://github.com/dgaida/PyADM1ODE_mcp.git
cd PyADM1ODE_mcp
pip install -e .
```

**Features:**
- Natural language plant design via LLM (e.g., Claude)
- MCP server for LLM integration
- Interactive plant configuration

**Use cases:** Non-expert plant design, rapid prototyping, educational tools

See [PyADM1ODE_mcp documentation](https://github.com/dgaida/PyADM1ODE_mcp) for details.

### [PyADM1ODE_calibration](https://github.com/dgaida/PyADM1ODE_calibration) - Parameter Calibration Framework

For automated model calibration from measurement data:

```bash
# Install from GitHub
git clone https://github.com/dgaida/PyADM1ODE_calibration.git
cd PyADM1ODE_calibration
pip install -e .
```

**Features:**
- Initial calibration from historical data
- Online re-calibration during operation
- Multiple optimization algorithms (DE, PSO, Nelder-Mead)
- Comprehensive validation metrics
- Database integration for measurement data

**Use cases:** Model parameterization, real plant adaptation, uncertainty quantification

See [PyADM1ODE_calibration documentation](https://github.com/dgaida/PyADM1ODE_calibration) for details.

## Uninstallation

To remove PyADM1ODE (not yet supported):
```bash
pip uninstall pyadm1ode
```

To also remove dependencies:
```bash
pip uninstall pyadm1ode pythonnet numpy pandas scipy matplotlib
```

To remove optional packages (not yet supported):
```bash
pip uninstall pyadm1ode_mcp pyadm1ode_calibration
```
