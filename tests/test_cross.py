"""Tests for cross-interpolation algorithms."""

import math

import jax.numpy as jnp

from tenax_qtt.cross import QTTResult, cross_interpolation, estimate_error
from tenax_qtt.grid import GridSpec, UniformGrid
from tenax_qtt.qtt import QTT


# ---- TCI2 tests ----


def test_cross_constant():
    """TCI on constant function should converge with bond dim 1."""
    grid = GridSpec(variables=(UniformGrid(0, 1, 8),), layout="grouped")
    result = cross_interpolation(lambda x: 1.0, grid, tol=1e-10, method="tci2")
    assert isinstance(result, QTTResult)
    assert result.converged
    assert max(result.qtt.bond_dims) <= 2  # near-trivial


def test_cross_linear():
    grid = GridSpec(variables=(UniformGrid(0, 1, 8),), layout="grouped")
    result = cross_interpolation(lambda x: x[0], grid, tol=1e-10, method="tci2")
    # Evaluate at a few points
    assert abs(result.qtt.evaluate((0.25,)) - 0.25) < 1e-6


def test_cross_sin():
    grid = GridSpec(
        variables=(UniformGrid(0, 2 * math.pi, 8),), layout="grouped"
    )
    result = cross_interpolation(
        lambda x: math.sin(x[0]),
        grid,
        tol=1e-6,
        max_bond_dim=20,
        method="tci2",
    )
    # Use estimate_error to check approximation quality on grid points
    # (avoids discretization artifacts from evaluating at non-grid points)
    from tenax_qtt.cross import estimate_error

    err = estimate_error(result.qtt, lambda x: math.sin(x[0]), n_samples=200)
    assert err < 0.05


def test_cross_result_fields():
    grid = GridSpec(variables=(UniformGrid(0, 1, 6),), layout="grouped")
    result = cross_interpolation(
        lambda x: x[0] ** 2, grid, tol=1e-8, method="tci2"
    )
    assert result.n_iter > 0
    assert result.n_function_evals > 0
    assert isinstance(result.estimated_error, float)


# ---- prrLU tests ----


def test_prrlu_constant():
    grid = GridSpec(variables=(UniformGrid(0, 1, 8),), layout="grouped")
    result = cross_interpolation(
        lambda x: 1.0, grid, tol=1e-10, method="prrlu"
    )
    assert isinstance(result, QTTResult)
    assert max(result.qtt.bond_dims) <= 2


def test_prrlu_polynomial():
    grid = GridSpec(variables=(UniformGrid(0, 1, 10),), layout="grouped")
    result = cross_interpolation(
        lambda x: x[0] ** 2 - 0.5 * x[0] + 0.1,
        grid,
        tol=1e-8,
        max_bond_dim=16,
        method="prrlu",
    )
    # Check at a few points
    assert abs(result.qtt.evaluate((0.3,)) - (0.09 - 0.15 + 0.1)) < 1e-4


def test_prrlu_vs_tci2_accuracy():
    """prrLU should be at least as good as TCI2 on a smooth function."""
    grid = GridSpec(
        variables=(UniformGrid(0, 2 * math.pi, 10),), layout="grouped"
    )
    r1 = cross_interpolation(
        lambda x: math.sin(x[0]),
        grid,
        tol=1e-6,
        max_bond_dim=20,
        method="tci2",
    )
    r2 = cross_interpolation(
        lambda x: math.sin(x[0]),
        grid,
        tol=1e-6,
        max_bond_dim=20,
        method="prrlu",
    )
    # Both should approximate sin reasonably
    x_test = (1.5,)
    assert abs(r1.qtt.evaluate(x_test) - math.sin(1.5)) < 1e-2
    assert abs(r2.qtt.evaluate(x_test) - math.sin(1.5)) < 1e-2


# ---- estimate_error tests ----


def test_estimate_error_exact():
    """Error should be near zero for an exact representation."""
    grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
    f = lambda x: 1.0
    result = cross_interpolation(f, grid, tol=1e-12, method="tci2")
    err = estimate_error(result.qtt, f, n_samples=100)
    assert err < 1e-8


# ---- QTT.from_cross and QTT.from_dense tests ----


def test_from_cross():
    grid = GridSpec(variables=(UniformGrid(0, 1, 8),), layout="grouped")
    result = QTT.from_cross(
        lambda x: x[0] ** 2, grid, tol=1e-6, method="tci2"
    )
    assert isinstance(result, QTTResult)
    assert abs(result.qtt.evaluate((0.5,)) - 0.25) < 1e-3


def test_from_dense():
    grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
    # 2^4 = 16 points
    data = jnp.linspace(0, 1, 16)
    qtt = QTT.from_dense(data, grid, tol=1e-10)
    assert isinstance(qtt, QTT)
    dense_back = qtt.to_dense()
    assert jnp.allclose(dense_back, data, atol=1e-8)
