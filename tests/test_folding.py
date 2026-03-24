"""Tests for SVD-based QTT construction."""

import jax.numpy as jnp

from tenax_qtt.folding import fold_to_qtt
from tenax_qtt.grid import GridSpec, UniformGrid


def test_fold_constant():
    """Folding a constant vector should produce bond-dim-1 QTT."""
    grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
    data = jnp.ones(16)  # 2^4
    qtt = fold_to_qtt(data, grid)
    assert max(qtt.bond_dims) == 1
    assert jnp.allclose(qtt.to_dense(), 1.0, atol=1e-12)


def test_fold_linear():
    """Folding a linear function should compress well."""
    grid = GridSpec(variables=(UniformGrid(0, 1, 8),), layout="grouped")
    N = 256  # 2^8
    data = jnp.linspace(0, 1, N, endpoint=False)
    qtt = fold_to_qtt(data, grid, tol=1e-10)
    assert jnp.allclose(qtt.to_dense(), data, atol=1e-8)


def test_fold_with_max_bond_dim():
    grid = GridSpec(variables=(UniformGrid(0, 1, 8),), layout="grouped")
    data = jnp.sin(jnp.linspace(0, 2 * jnp.pi, 256, endpoint=False))
    qtt = fold_to_qtt(data, grid, max_bond_dim=4)
    assert max(qtt.bond_dims) <= 4


def test_fold_roundtrip_exact():
    """A rank-1 tensor should be exactly representable."""
    grid = GridSpec(variables=(UniformGrid(0, 1, 3),), layout="grouped")
    data = jnp.ones(8)  # constant = rank 1
    qtt = fold_to_qtt(data, grid)
    assert jnp.allclose(qtt.to_dense(), data, atol=1e-14)
