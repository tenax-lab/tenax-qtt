"""Tests for QTTMatrix class."""

import jax.numpy as jnp
import pytest

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


class TestFromDenseMatrix:
    def test_identity_roundtrip(self):
        grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
        mat = jnp.eye(16)
        M = QTTMatrix._from_dense_matrix(mat, grid, grid)
        reconstructed = M.to_dense()
        assert jnp.allclose(reconstructed, mat, atol=1e-10)

    def test_tridiagonal(self):
        """A tridiagonal matrix should round-trip through MPO format."""
        grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
        N = 16
        mat = jnp.zeros((N, N))
        for i in range(N):
            mat = mat.at[i, i].set(2.0)
            if i > 0:
                mat = mat.at[i, i - 1].set(-1.0)
            if i < N - 1:
                mat = mat.at[i, i + 1].set(-1.0)
        M = QTTMatrix._from_dense_matrix(mat, grid, grid)
        reconstructed = M.to_dense()
        assert jnp.allclose(reconstructed, mat, atol=1e-10)


class TestToDense:
    def test_identity_to_dense(self):
        grid = GridSpec(variables=(UniformGrid(0, 1, 3),), layout="grouped")
        I = QTTMatrix.identity(grid)
        mat = I.to_dense()
        assert mat.shape == (8, 8)
        assert jnp.allclose(mat, jnp.eye(8), atol=1e-12)


# ---------------------------------------------------------------------------
# 6b: Analytical operators — derivative_1d, laplacian_1d
# ---------------------------------------------------------------------------


class TestDerivative1D:
    def test_derivative_linear(self):
        """Derivative of f(x) = x should be ~1 at interior points."""
        grid = GridSpec(variables=(UniformGrid(0, 1, 6),), layout="grouped")
        N = 64
        x = jnp.linspace(0, 1, N, endpoint=False)
        data = x  # f(x) = x, derivative = 1
        qtt = fold_to_qtt(data, grid)
        D = QTTMatrix.derivative_1d(grid)
        result = D.apply(qtt, method="naive")
        dense = result.to_dense()
        # Interior points should be ~1 (boundary effects at edges)
        assert jnp.allclose(dense[2:-2], 1.0, atol=0.1)

    def test_derivative_quadratic(self):
        """Derivative of f(x) = x^2 should be ~2x at interior points."""
        grid = GridSpec(variables=(UniformGrid(0, 1, 6),), layout="grouped")
        N = 64
        x = jnp.linspace(0, 1, N, endpoint=False)
        data = x**2
        qtt = fold_to_qtt(data, grid)
        D = QTTMatrix.derivative_1d(grid)
        result = D.apply(qtt, method="naive")
        dense = result.to_dense()
        expected = 2.0 * x
        # Interior points should match (skip boundaries)
        assert jnp.allclose(dense[2:-2], expected[2:-2], atol=0.1)

    def test_derivative_dense_matrix(self):
        """Check the dense matrix form of the derivative operator."""
        grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
        D = QTTMatrix.derivative_1d(grid)
        mat = D.to_dense()
        N = 16
        dx = 1.0 / N
        # Check a few entries: D[1,0] = -1/(2dx), D[1,2] = 1/(2dx)
        assert jnp.isclose(mat[1, 0], -1.0 / (2 * dx), atol=1e-8)
        assert jnp.isclose(mat[1, 2], 1.0 / (2 * dx), atol=1e-8)
        assert jnp.isclose(mat[1, 1], 0.0, atol=1e-8)

    def test_derivative_requires_1d(self):
        grid = GridSpec(
            variables=(UniformGrid(0, 1, 4), UniformGrid(0, 1, 4)),
            layout="interleaved",
        )
        with pytest.raises(ValueError, match="1D grid"):
            QTTMatrix.derivative_1d(grid)


class TestLaplacian1D:
    def test_laplacian_quadratic(self):
        """Laplacian of f(x) = x^2 should be ~2 at interior points."""
        grid = GridSpec(variables=(UniformGrid(0, 1, 6),), layout="grouped")
        N = 64
        x = jnp.linspace(0, 1, N, endpoint=False)
        data = x**2
        qtt = fold_to_qtt(data, grid)
        L = QTTMatrix.laplacian_1d(grid)
        result = L.apply(qtt, method="naive")
        dense = result.to_dense()
        # Interior points should be ~2
        assert jnp.allclose(dense[2:-2], 2.0, atol=0.5)

    def test_laplacian_dense_matrix(self):
        """Check the dense matrix form of the Laplacian operator."""
        grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
        L_op = QTTMatrix.laplacian_1d(grid)
        mat = L_op.to_dense()
        N = 16
        dx = 1.0 / N
        # Check diagonal: -2/dx^2
        assert jnp.isclose(mat[3, 3], -2.0 / dx**2, atol=1e-6)
        # Check off-diagonal: 1/dx^2
        assert jnp.isclose(mat[3, 2], 1.0 / dx**2, atol=1e-6)
        assert jnp.isclose(mat[3, 4], 1.0 / dx**2, atol=1e-6)

    def test_laplacian_requires_1d(self):
        grid = GridSpec(
            variables=(UniformGrid(0, 1, 4), UniformGrid(0, 1, 4)),
            layout="interleaved",
        )
        with pytest.raises(ValueError, match="1D grid"):
            QTTMatrix.laplacian_1d(grid)


class TestApplyMethodsConsistency:
    """Verify naive and zipup produce consistent results on a nontrivial operator."""

    def test_naive_vs_zipup_derivative(self):
        grid = GridSpec(variables=(UniformGrid(0, 1, 5),), layout="grouped")
        N = 32
        x = jnp.linspace(0, 1, N, endpoint=False)
        data = jnp.sin(2 * jnp.pi * x)
        qtt = fold_to_qtt(data, grid)
        D = QTTMatrix.derivative_1d(grid)
        result_naive = D.apply(qtt, method="naive").to_dense()
        result_zipup = D.apply(qtt, method="zipup").to_dense()
        assert jnp.allclose(result_naive, result_zipup, atol=1e-4)


# ---------------------------------------------------------------------------
# 11a: transpose, compose, dunder methods
# ---------------------------------------------------------------------------


class TestTranspose:
    def test_transpose_identity(self):
        grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
        I = QTTMatrix.identity(grid)
        It = I.transpose()
        qtt = QTT.ones(grid)
        result = It.apply(qtt, method="naive")
        assert jnp.allclose(result.to_dense(), 1.0, atol=1e-10)

    def test_transpose_swaps_grids(self):
        grid_in = GridSpec(variables=(UniformGrid(0, 1, 3),), layout="grouped")
        grid_out = GridSpec(variables=(UniformGrid(0, 2, 3),), layout="grouped")
        I = QTTMatrix.identity(grid_in)
        # Build a matrix with different in/out grids
        M = QTTMatrix(site_tensors=I.site_tensors, grid_in=grid_in, grid_out=grid_out)
        Mt = M.transpose()
        assert Mt.grid_in == grid_out
        assert Mt.grid_out == grid_in

    def test_transpose_dense_roundtrip(self):
        """Transpose of a dense matrix should match np transpose."""
        grid = GridSpec(variables=(UniformGrid(0, 1, 3),), layout="grouped")
        N = 8
        mat = jnp.arange(N * N, dtype=float).reshape(N, N)
        M = QTTMatrix._from_dense_matrix(mat, grid, grid)
        Mt = M.transpose()
        assert jnp.allclose(Mt.to_dense(), mat.T, atol=1e-8)


class TestCompose:
    def test_compose_identity(self):
        grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
        I = QTTMatrix.identity(grid)
        I2 = I.compose(I, method="naive")
        qtt = QTT.ones(grid)
        result = I2.apply(qtt, method="naive")
        assert jnp.allclose(result.to_dense(), 1.0, atol=1e-10)

    def test_compose_dense_product(self):
        """Compose two dense-folded matrices and check against matmul."""
        grid = GridSpec(variables=(UniformGrid(0, 1, 3),), layout="grouped")
        N = 8
        A = jnp.eye(N) * 2.0
        B = jnp.eye(N) * 3.0
        MA = QTTMatrix._from_dense_matrix(A, grid, grid)
        MB = QTTMatrix._from_dense_matrix(B, grid, grid)
        MC = MA.compose(MB, method="naive")
        expected = A @ B
        assert jnp.allclose(MC.to_dense(), expected, atol=1e-8)

    def test_compose_unsupported_method(self):
        grid = GridSpec(variables=(UniformGrid(0, 1, 3),), layout="grouped")
        I = QTTMatrix.identity(grid)
        with pytest.raises(NotImplementedError):
            I.compose(I, method="zipup")


class TestDunderMethods:
    def test_qttmatrix_add(self):
        grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
        I = QTTMatrix.identity(grid)
        two_I = I + I
        qtt = QTT.ones(grid)
        result = two_I.apply(qtt, method="naive")
        assert jnp.allclose(result.to_dense(), 2.0, atol=1e-10)

    def test_qttmatrix_add_dense(self):
        """(A + B).to_dense() should equal A.to_dense() + B.to_dense()."""
        grid = GridSpec(variables=(UniformGrid(0, 1, 3),), layout="grouped")
        N = 8
        A_mat = jnp.eye(N) * 2.0
        B_mat = jnp.diag(jnp.arange(N, dtype=float))
        MA = QTTMatrix._from_dense_matrix(A_mat, grid, grid)
        MB = QTTMatrix._from_dense_matrix(B_mat, grid, grid)
        MC = MA + MB
        assert jnp.allclose(MC.to_dense(), A_mat + B_mat, atol=1e-8)

    def test_qttmatrix_scalar_mul(self):
        grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
        I = QTTMatrix.identity(grid)
        scaled = I * 3.0
        qtt = QTT.ones(grid)
        result = scaled.apply(qtt, method="naive")
        assert jnp.allclose(result.to_dense(), 3.0, atol=1e-10)

    def test_qttmatrix_rmul(self):
        grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
        I = QTTMatrix.identity(grid)
        scaled = 5.0 * I
        qtt = QTT.ones(grid)
        result = scaled.apply(qtt, method="naive")
        assert jnp.allclose(result.to_dense(), 5.0, atol=1e-10)

    def test_qttmatrix_sub(self):
        grid = GridSpec(variables=(UniformGrid(0, 1, 4),), layout="grouped")
        I = QTTMatrix.identity(grid)
        zero = I - I
        qtt = QTT.ones(grid)
        result = zero.apply(qtt, method="naive")
        assert jnp.allclose(result.to_dense(), 0.0, atol=1e-10)
