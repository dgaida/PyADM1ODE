# -*- coding: utf-8 -*-
"""
Unit tests

- Test bound validation
- Test penalty calculation
- Test scaling/unscaling
"""

import numpy as np
import pytest

from pyadm1.calibration.parameter_bounds import (
    ParameterBound,
    ParameterBounds,
    BoundType,
    create_default_bounds,
)


def test_parameter_bound_within_bounds():
    pb = ParameterBound("k_dis", lower=0.3, upper=0.8, default=0.5)

    assert pb.is_within_bounds(0.5)
    assert not pb.is_within_bounds(0.1)
    assert not pb.is_within_bounds(1.0)


def test_parameter_bound_clipping():
    pb = ParameterBound("k_dis", lower=0.3, upper=0.8, default=0.5)

    assert pb.clip_to_bounds(0.1) == 0.3
    assert pb.clip_to_bounds(1.0) == 0.8
    assert pb.clip_to_bounds(0.5) == 0.5


def test_hard_bounds_penalty():
    pb = ParameterBound("k_dis", lower=0.3, upper=0.8, default=0.5, bound_type=BoundType.HARD)

    assert pb.calculate_penalty(0.5) == 0.0
    assert np.isinf(pb.calculate_penalty(0.1))


def test_soft_bounds_penalty_quadratic():
    pb = ParameterBound("k_dis", lower=0.3, upper=0.8, default=0.5, bound_type=BoundType.SOFT, penalty_weight=2.0)

    # Below lower bound: distance = 0.2 → penalty = 2 * 0.04 = 0.08
    assert pytest.approx(pb.calculate_penalty(0.1)) == 0.08


def test_soft_bounds_penalty_linear():
    pb = ParameterBound(
        "k_dis",
        lower=0.3,
        upper=0.8,
        default=0.5,
        bound_type=BoundType.SOFT,
        penalty_weight=1.0,
    )

    assert pb.calculate_penalty(0.0, penalty_type="linear") == 0.3


def test_soft_bounds_penalty_barrier():
    pb = ParameterBound(
        "k_dis",
        lower=0.3,
        upper=0.8,
        default=0.5,
        bound_type=BoundType.SOFT,
        penalty_weight=1.0,
    )

    assert pb.calculate_penalty(0.0, penalty_type="barrier") > 3.0


def test_relative_position():
    pb = ParameterBound("k_dis", lower=0.0, upper=10.0, default=5.0)

    assert pb.get_relative_position(0.0) == 0.0
    assert pb.get_relative_position(10.0) == 1.0
    assert pb.get_relative_position(5.0) == 0.5


def test_parameter_bounds_add_get():
    bounds = ParameterBounds()
    bounds.add_bound("k_dis", 0.3, 0.8, 0.5)

    b = bounds.get_bounds("k_dis")
    assert b.lower == 0.3
    assert b.upper == 0.8


def test_parameter_bounds_is_within():
    bounds = ParameterBounds()
    bounds.add_bound("k_dis", 0.3, 0.8, 0.5)

    assert bounds.is_within_bounds("k_dis", 0.5)
    assert not bounds.is_within_bounds("k_dis", 0.1)


def test_parameter_bounds_default_values():
    bounds = create_default_bounds()
    params = ["k_dis", "Y_su", "k_m_ac"]

    defaults = bounds.get_default_values(params)

    for p in params:
        assert p in defaults
        assert isinstance(defaults[p], float)


def test_total_penalty_sum():
    bounds = ParameterBounds()
    bounds.add_bound("k1", 0, 1, 0.5, BoundType.SOFT, penalty_weight=1.0)
    bounds.add_bound("k2", 0, 2, 1.0, BoundType.SOFT, penalty_weight=1.0)

    params = {"k1": -1, "k2": 5}
    penalty = bounds.calculate_total_penalty(params)

    assert penalty > 0
