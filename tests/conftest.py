"""
pytest configuration and shared fixtures for PyADM1ODE tests.

This module provides common fixtures and configuration used across
all test modules.
"""

import pytest
import numpy as np
from pathlib import Path
from typing import List


@pytest.fixture
def sample_state_vector() -> List[float]:
    """
    Provide a sample ADM1 state vector for testing.

    Returns:
        List of 37 float values representing a typical ADM1 state.
    """
    return [
        0.0055,  # S_su
        0.0025,  # S_aa
        0.0398,  # S_fa
        0.0052,  # S_va
        0.0101,  # S_bu
        0.0281,  # S_pro
        0.9686,  # S_ac
        1.075e-7,  # S_h2
        0.0556,  # S_ch4
        0.0117,  # S_co2
        0.2438,  # S_nh4
        10.738,  # S_I
        18.015,  # X_xc
        0.2544,  # X_ch
        0.0554,  # X_pr
        0.0184,  # X_li
        7.8300,  # X_su
        1.3649,  # X_aa
        0.3288,  # X_fa
        1.0153,  # X_c4
        0.8791,  # X_pro
        3.1754,  # X_ac
        1.6683,  # X_h2
        38.294,  # X_I
        2.0409,  # X_p
        0.0,  # S_cation
        0.0,  # S_anion
        0.0052,  # S_va_ion
        0.0101,  # S_bu_ion
        0.0281,  # S_pro_ion
        0.9672,  # S_ac_ion
        0.2284,  # S_hco3_ion
        0.0128,  # S_nh3
        5.935e-6,  # pi_Sh2
        0.5592,  # pi_Sch4
        0.4253,  # pi_Sco2
        0.9845,  # pTOTAL
    ]


@pytest.fixture
def test_data_dir(tmp_path: Path) -> Path:
    """
    Create a temporary directory for test data files.

    Args:
        tmp_path: pytest fixture providing temporary directory.

    Returns:
        Path to test data directory.
    """
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def sample_flow_rates() -> List[float]:
    """
    Provide sample volumetric flow rates for testing.

    Returns:
        List of 10 float values representing substrate flow rates in m³/d.
    """
    return [15.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


@pytest.fixture
def random_seed() -> None:
    """
    Set random seed for reproducible tests.

    This fixture ensures that tests involving random number generation
    are deterministic and reproducible.
    """
    np.random.seed(42)


# Configure pytest markers
def pytest_configure(config):
    """
    Configure custom pytest markers.

    Args:
        config: pytest configuration object.
    """
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "requires_dlls: marks tests that require C# DLLs")
