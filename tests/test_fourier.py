"""Tests for Fourier transform MPO construction."""

import jax.numpy as jnp
import numpy as np
import pytest

from tenax_qtt.folding import fold_to_qtt
from tenax_qtt.fourier import (
    _build_fourier_site_tensors,
    _svd_compress_site_tensors,
    fourier_mpo,
)
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


# ---- Tests for the analytical (binary phase decomposition) construction ----


def _mpo_to_dense(tensors, R):
    """Contract raw MPO site tensors into a dense matrix for verification."""
    N = 2**R
    result = tensors[0][0]  # remove left bond dim (=1)
    for i in range(1, R):
        result = np.tensordot(result, tensors[i], axes=([-1], [0]))
    result = result[..., 0]  # remove right bond dim (=1)
    # Current shape: (j_0, k_0, j_1, k_1, ..., j_{R-1}, k_{R-1})
    out_axes = list(range(0, 2 * R, 2))
    in_axes = list(range(1, 2 * R, 2))
    result = np.transpose(result, out_axes + in_axes)
    return result.reshape(N, N)


@pytest.mark.parametrize("R", [2, 3, 4, 5, 6])
def test_analytical_matches_dense_dft(R):
    """Analytical site tensors reproduce the exact dense DFT matrix."""
    N = 2**R
    sign = -1.0
    F_ref = np.exp(sign * 2j * np.pi * np.outer(np.arange(N), np.arange(N)) / N)
    F_ref /= np.sqrt(N)

    tensors = _build_fourier_site_tensors(R, sign)
    compressed = _svd_compress_site_tensors(tensors)
    F_mpo = _mpo_to_dense(compressed, R)

    np.testing.assert_allclose(F_mpo, F_ref, atol=1e-12)


@pytest.mark.parametrize("R", [2, 3, 4, 5])
def test_analytical_inverse(R):
    """Analytical construction of the inverse DFT is correct."""
    N = 2**R
    sign = 1.0
    F_ref = np.exp(sign * 2j * np.pi * np.outer(np.arange(N), np.arange(N)) / N)
    F_ref /= np.sqrt(N)

    tensors = _build_fourier_site_tensors(R, sign)
    compressed = _svd_compress_site_tensors(tensors)
    F_mpo = _mpo_to_dense(compressed, R)

    np.testing.assert_allclose(F_mpo, F_ref, atol=1e-12)


@pytest.mark.parametrize("R", [3, 4, 5, 6])
def test_analytical_bond_dimensions(R):
    """After compression, bond dims match 4^{min(a+1, R-a-1)}."""
    tensors = _build_fourier_site_tensors(R, -1.0)
    compressed = _svd_compress_site_tensors(tensors)

    for a in range(R - 1):
        expected_bond = 4 ** min(a + 1, R - a - 1)
        actual_bond = compressed[a].shape[3]
        assert actual_bond == expected_bond, (
            f"Bond {a}: expected {expected_bond}, got {actual_bond}"
        )


def test_analytical_large_R_no_dense_matrix():
    """For large R, the analytical path builds MPO without N x N matrix.

    Verifies the construction completes for R=16 (N=65536) where the
    dense matrix would require 32 GB of memory.
    """
    grid = GridSpec(variables=(UniformGrid(0, 1, 16),), layout="grouped")
    F = fourier_mpo(grid, max_bond_dim=32)
    # Basic sanity: check the MPO has the right number of site tensors
    assert len(F.site_tensors) == 16
    # Bond dims should all be at most 32
    for i, W in enumerate(F.site_tensors):
        W = jnp.array(W)
        assert W.shape[0] <= 32, f"Left bond at site {i} exceeds max_bond_dim"
        assert W.shape[3] <= 32, f"Right bond at site {i} exceeds max_bond_dim"


def test_analytical_large_R_roundtrip():
    """Forward + inverse DFT roundtrip for R=10 with bounded bond dim."""
    R = 10
    N = 2**R
    grid = GridSpec(variables=(UniformGrid(0, 1, R),), layout="grouped")

    # Build a simple test signal: single frequency
    data = jnp.sin(2 * jnp.pi * 3 * jnp.arange(N) / N)
    qtt = fold_to_qtt(data, grid, max_bond_dim=64)

    F = fourier_mpo(grid, max_bond_dim=64)
    F_inv = fourier_mpo(grid, inverse=True, max_bond_dim=64)

    transformed = F.apply(qtt, method="naive", max_bond_dim=128)
    recovered = F_inv.apply(transformed, method="naive", max_bond_dim=128)

    # With max_bond_dim=64 for the DFT MPO, some approximation error is
    # expected.  The roundtrip should still preserve the signal shape.
    rec_dense = recovered.to_dense().real
    # Normalize both to unit norm for comparison
    data_norm = data / jnp.linalg.norm(data)
    rec_norm = rec_dense / jnp.linalg.norm(rec_dense)
    overlap = jnp.abs(jnp.dot(data_norm, rec_norm))
    assert overlap > 0.99, f"Roundtrip overlap too low: {overlap}"


def test_analytical_truncated_precision():
    """Truncated precision produces approximate but bounded-bond MPO."""
    R = 8
    N = 2**R
    sign = -1.0
    F_ref = np.exp(sign * 2j * np.pi * np.outer(np.arange(N), np.arange(N)) / N)
    F_ref /= np.sqrt(N)

    for p in [2, 3, 4]:
        tensors = _build_fourier_site_tensors(R, sign, max_precision=p)
        compressed = _svd_compress_site_tensors(tensors)
        F_mpo = _mpo_to_dense(compressed, R)

        err = np.max(np.abs(F_mpo - F_ref))
        max_bond = max(t.shape[3] for t in compressed)

        # Bond dim should be bounded by 4^p
        assert max_bond <= 4**p, f"p={p}: max bond {max_bond} exceeds 4^{p}={4**p}"
        # Higher precision should give lower error
        if p >= 4:
            assert err < 0.05, f"p={p}: error {err} too large"
