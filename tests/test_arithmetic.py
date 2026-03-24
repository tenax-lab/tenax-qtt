"""Tests for QTT arithmetic operations."""

import jax.numpy as jnp
import pytest
from tenax_qtt.arithmetic import add, hadamard, recompress, scalar_multiply, subtract
from tenax_qtt.folding import fold_to_qtt
from tenax_qtt.grid import GridSpec, UniformGrid
from tenax_qtt.qtt import QTT


# -- 4a: scalar_multiply and recompress --


def test_scalar_multiply():
    grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
    qtt = QTT.ones(grid)
    scaled = scalar_multiply(qtt, 3.0)
    assert jnp.allclose(scaled.to_dense(), 3.0, atol=1e-12)


def test_scalar_multiply_complex():
    grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
    qtt = QTT.ones(grid)
    scaled = scalar_multiply(qtt, 1j)
    dense = scaled.to_dense()
    assert jnp.allclose(dense, 1j, atol=1e-12)


def test_recompress_no_change():
    grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
    qtt = QTT.ones(grid)
    compressed = recompress(qtt)
    assert max(compressed.bond_dims) == 1


# -- 4b: add, subtract, hadamard --


def test_add_ones():
    grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
    a = QTT.ones(grid)
    b = QTT.ones(grid)
    result = add(a, b)
    assert jnp.allclose(result.to_dense(), 2.0, atol=1e-10)


def test_subtract():
    grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
    a = QTT.ones(grid)
    b = QTT.ones(grid)
    result = subtract(a, b)
    assert jnp.allclose(result.to_dense(), 0.0, atol=1e-10)


def test_add_mismatched_grids():
    g1 = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
    g2 = GridSpec(variables=(UniformGrid(0, 2, 4),), layout="grouped")
    a = QTT.ones(g1)
    b = QTT.ones(g2)
    with pytest.raises(ValueError, match="identical GridSpec"):
        add(a, b)


def test_hadamard():
    grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
    data = jnp.linspace(0, 1, 16, endpoint=False)
    a = fold_to_qtt(data, grid)
    b = fold_to_qtt(data, grid)
    result = hadamard(a, b)
    expected = data**2
    assert jnp.allclose(result.to_dense(), expected, atol=1e-8)


def test_add_recompresses():
    grid = GridSpec(variables=(UniformGrid(0, 1, 8),), layout="grouped")
    a = QTT.ones(grid)
    b = QTT.ones(grid)
    result = add(a, b, max_bond_dim=2)
    assert max(result.bond_dims) <= 2


# -- 4c: QTT dunder methods --


def test_qtt_add_operator():
    grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
    a = QTT.ones(grid)
    b = QTT.ones(grid)
    result = a + b
    assert jnp.allclose(result.to_dense(), 2.0, atol=1e-10)


def test_qtt_sub_operator():
    grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
    a = QTT.ones(grid)
    b = QTT.ones(grid)
    result = a - b
    assert jnp.allclose(result.to_dense(), 0.0, atol=1e-10)


def test_qtt_mul_operator():
    grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
    a = QTT.ones(grid)
    result = a * 5.0
    assert jnp.allclose(result.to_dense(), 5.0, atol=1e-12)


def test_qtt_rmul_operator():
    grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
    a = QTT.ones(grid)
    result = 5.0 * a
    assert jnp.allclose(result.to_dense(), 5.0, atol=1e-12)
