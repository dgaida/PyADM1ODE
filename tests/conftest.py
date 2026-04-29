"""
pytest configuration and shared fixtures for PyADM1ODE tests.

PyADM1 is now pure Python (no .NET / DLL dependency), so this file no longer
needs to mock CLR or skip DLL-bound tests.
"""

from pathlib import Path
from typing import List

import numpy as np
import pytest


@pytest.fixture
def sample_state_vector() -> List[float]:
    """
    Provide a representative ADM1 state vector (41 elements) for testing.
    """
    return [
        # Dissolved (0–11)
        0.005,
        0.0025,
        0.04,
        0.005,
        0.01,
        0.028,
        0.7,
        1.0e-7,
        0.05,
        0.18,
        0.04,
        0.5,
        # Particulate sub-fractions (12–21)
        2.0,
        1.0,
        0.3,
        0.5,
        0.2,
        0.1,
        0.5,
        0.2,
        0.1,
        5.0,
        # Biomass (22–28)
        0.5,
        0.5,
        0.3,
        0.4,
        0.3,
        1.2,
        0.3,
        # Charge balance (29–36)
        0.04,
        0.04,
        0.005,
        0.01,
        0.028,
        0.69,
        0.16,
        0.005,
        # Gas phase (37–40)
        1.02e-5,
        0.65,
        0.33,
        0.98,
    ]


@pytest.fixture
def test_data_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory for test data files."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def sample_flow_rates() -> List[float]:
    """Per-substrate flow rates [m³/d], shape (10,)."""
    return [11.4, 6.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


@pytest.fixture
def random_seed() -> None:
    """Set a deterministic random seed for tests using np.random."""
    np.random.seed(42)


def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
