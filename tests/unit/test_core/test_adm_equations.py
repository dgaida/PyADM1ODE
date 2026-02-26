"""Unit tests for adm_equations helper functions."""

from pyadm1.core.adm_equations import InhibitionFunctions


def test_substrate_inhibition_monod_factor():
    """Cover substrate_inhibition helper."""
    value = InhibitionFunctions.substrate_inhibition(S=2.0, K_S=3.0)
    assert value == 2.0 / 5.0
