# Testing PyADM1ODE

## Overview

This project includes unit tests for the core ADM1 functionality. However, some tests require the .NET/Mono runtime to be available because they test components that interact with C# DLLs via `pythonnet`.

## Requirements

### For Basic Tests
- Python 3.8+
- pytest
- numpy
- pandas
- scipy

### For Full Test Suite (including .NET-dependent tests)
- All basic requirements
- Mono runtime (Linux/macOS) or .NET Framework (Windows)
- pythonnet
- C# DLLs in the `dlls/` directory (included in repository)
- **Windows only**: vs2015_runtime

## Running Tests

### Run All Available Tests

```bash
pytest
```

This will automatically skip tests that require .NET if the runtime is not available.

### Run Only Tests That Don't Require .NET

```bash
pytest -m "not requires_dotnet"
```

### View Skipped Test Information

```bash
pytest -v -rs
```

The `-rs` flag shows detailed information about skipped tests.

### Install .NET/Mono Runtime

#### On Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install mono-complete
```

#### On macOS:
```bash
brew install mono
```

#### On Windows:
Install .NET Framework or Mono from the official websites.

## Test Structure

```
tests/
├── test_adm_params.py      # Tests for ADMparams (requires .NET)
├── test_pyadm1.py           # Tests for PyADM1 (requires .NET)
├── test_simulator.py        # Tests for Simulator (requires .NET)
└── test_feedstock.py        # Tests for Feedstock (requires .NET)
```

## Continuous Integration

The GitHub Actions workflow automatically installs Mono (on Linux/macOS) or uses the pre-installed .NET Framework (on Windows) to run the complete test suite. All tests should pass in CI.

## Writing New Tests

When writing tests that require the C# DLLs, add this at the top of your test module:

```python
import pytest

# Check if .NET runtime is available
try:
    import clr
    clr.AddReference("System")
    DOTNET_AVAILABLE = True
except (ImportError, RuntimeError):
    DOTNET_AVAILABLE = False

# Skip entire module if .NET is not available
if not DOTNET_AVAILABLE:
    pytest.skip("Requires .NET/Mono runtime", allow_module_level=True)
```

## Known Issues

1. **First-time Mono Installation**: When running tests for the first time after installing Mono, you may need to restart your terminal or IDE.

2. **DLL Files**: The C# DLL files are included in the `dlls/` directory of the repository. Make sure they are present and accessible.

3. **Platform Differences**: Some tests may behave differently on Windows vs Linux/macOS due to .NET Framework vs Mono differences.

4. **pythonnet Configuration**: On some systems, you may need to configure pythonnet to use the correct runtime. See the pythonnet documentation for details.

## Contributing

When contributing tests:
1. Ensure tests can run without .NET when possible
2. Clearly mark tests that require .NET
3. Provide mock data or fixtures to enable testing without external dependencies
4. Document any special requirements in test docstrings
