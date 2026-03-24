"""Tests for Fourier transform MPO construction."""

import jax.numpy as jnp
import pytest

from tenax_qtt.folding import fold_to_qtt
from tenax_qtt.fourier import fourier_mpo
from tenax_qtt.grid import GridSpec, UniformGrid


def test_fourier_delta():
    """DFT of delta function at index 0 should give constant magnitude."""
    grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
    N = 16
    data = jnp.zeros(N).at[0].set(1.0)
    qtt = fold_to_qtt(data, grid)
    F = fourier_mpo(grid)
    result = F.apply(qtt, method="naive")
    dense = result.to_dense()
    # DFT of delta(0) = 1/sqrt(N) * [1, 1, ..., 1]
    magnitudes = jnp.abs(dense)
    assert jnp.allclose(magnitudes, magnitudes[0], atol=1e-8)


def test_fourier_roundtrip():
    """Forward then inverse DFT should recover original."""
    grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
    N = 16
    data = jnp.sin(2 * jnp.pi * jnp.arange(N) / N)
    qtt = fold_to_qtt(data, grid)
    F = fourier_mpo(grid)
    F_inv = fourier_mpo(grid, inverse=True)
    transformed = F.apply(qtt, method="naive")
    recovered = F_inv.apply(transformed, method="naive")
    assert jnp.allclose(recovered.to_dense(), data, atol=1e-6)


def test_fourier_multi_d_raises():
    """Multi-D grids should raise NotImplementedError."""
    grid = GridSpec(
        variables=(UniformGrid(0, 1, 3), UniformGrid(0, 1, 3)),
        layout="grouped",
    )
    with pytest.raises(NotImplementedError, match="Multi-D"):
        fourier_mpo(grid)


def test_fourier_unitarity():
    """F @ F_inv should approximate the identity."""
    grid = GridSpec(variables=(UniformGrid(0, 1, 3),), layout="grouped")
    N = 8
    F = fourier_mpo(grid)
    F_inv = fourier_mpo(grid, inverse=True)
    # Apply to each basis vector
    for k in range(N):
        data = jnp.zeros(N).at[k].set(1.0)
        qtt = fold_to_qtt(data, grid)
        roundtrip = F_inv.apply(F.apply(qtt, method="naive"), method="naive")
        assert jnp.allclose(roundtrip.to_dense(), data, atol=1e-6), (
            f"Roundtrip failed for basis vector {k}"
        )
