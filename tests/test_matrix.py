"""Tests for QTTMatrix class."""

import jax.numpy as jnp

from tenax_qtt.folding import fold_to_qtt
from tenax_qtt.grid import GridSpec, UniformGrid
from tenax_qtt.matrix import QTTMatrix
from tenax_qtt.qtt import QTT


# ---------------------------------------------------------------------------
# 6a: QTTMatrix dataclass, identity, and apply methods
# ---------------------------------------------------------------------------


class TestIdentity:
    def test_identity_construction(self):
        grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
        I = QTTMatrix.identity(grid)
        assert I.grid_in == grid
        assert I.grid_out == grid
        assert len(I.site_tensors) == 4  # 4 bits => 4 sites

    def test_identity_site_tensor_shape(self):
        grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
        I = QTTMatrix.identity(grid)
        for W in I.site_tensors:
            assert W.shape == (1, 2, 2, 1)

    def test_identity_to_dense(self):
        grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
        I = QTTMatrix.identity(grid)
        mat = I.to_dense()
        assert mat.shape == (16, 16)
        assert jnp.allclose(mat, jnp.eye(16), atol=1e-12)


class TestNaiveApply:
    def test_identity_apply_ones(self):
        grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
        I = QTTMatrix.identity(grid)
        qtt = QTT.ones(grid)
        result = I.apply(qtt, method="naive")
        assert jnp.allclose(result.to_dense(), 1.0, atol=1e-10)

    def test_identity_apply_linear(self):
        grid = GridSpec(variables=(UniformGrid(0, 1, 6),), layout="grouped")
        N = 64
        x = jnp.linspace(0, 1, N, endpoint=False)
        qtt = fold_to_qtt(x, grid)
        I = QTTMatrix.identity(grid)
        result = I.apply(qtt, method="naive")
        assert jnp.allclose(result.to_dense(), x, atol=1e-6)

    def test_identity_preserves_values(self):
        """I @ v should equal v for an arbitrary vector."""
        grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
        N = 16
        data = jnp.sin(jnp.linspace(0, 2 * jnp.pi, N, endpoint=False))
        qtt = fold_to_qtt(data, grid)
        I = QTTMatrix.identity(grid)
        result = I.apply(qtt, method="naive")
        assert jnp.allclose(result.to_dense(), data, atol=1e-6)


class TestZipupApply:
    def test_identity_apply_ones(self):
        grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
        I = QTTMatrix.identity(grid)
        qtt = QTT.ones(grid)
        result = I.apply(qtt, method="zipup")
        assert jnp.allclose(result.to_dense(), 1.0, atol=1e-10)

    def test_identity_apply_linear(self):
        grid = GridSpec(variables=(UniformGrid(0, 1, 6),), layout="grouped")
        N = 64
        x = jnp.linspace(0, 1, N, endpoint=False)
        qtt = fold_to_qtt(x, grid)
        I = QTTMatrix.identity(grid)
        result = I.apply(qtt, method="zipup")
        assert jnp.allclose(result.to_dense(), x, atol=1e-6)


class TestTCIApply:
    def test_identity_apply_ones(self):
        grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
        I = QTTMatrix.identity(grid)
        qtt = QTT.ones(grid)
        result = I.apply(qtt, method="tci", tol=1e-10, max_bond_dim=16)
        assert jnp.allclose(result.to_dense(), 1.0, atol=1e-4)


class TestToDense:
    def test_identity_to_dense(self):
        grid = GridSpec(variables=(UniformGrid(0, 1, 3),), layout="grouped")
        I = QTTMatrix.identity(grid)
        mat = I.to_dense()
        assert mat.shape == (8, 8)
        assert jnp.allclose(mat, jnp.eye(8), atol=1e-12)
