"""Analytical benchmark tests validating QTT correctness against known results.

References:
- Oseledets, "Tensor-Train Decomposition", SIAM J. Sci. Comput. 33 (2011)
- Khoromskij, "O(d log N)-Quantics Approximation", Constr. Approx. 34 (2011)
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import erf

from tenax_qtt.arithmetic import hadamard
from tenax_qtt.cross import cross_interpolation
from tenax_qtt.folding import fold_to_qtt
from tenax_qtt.fourier import fourier_mpo
from tenax_qtt.grid import GridSpec, UniformGrid, _index_to_coord
from tenax_qtt.matrix import QTTMatrix
from tenax_qtt.qtt import QTT

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_1d_grid(a: float, b: float, n_bits: int = 10, **kwargs) -> GridSpec:
    return GridSpec(variables=(UniformGrid(a, b, n_bits, **kwargs),), layout="grouped")


def _sample_on_grid(f, grid: GridSpec) -> jnp.ndarray:
    """Evaluate a scalar function at every grid point, returning a flat array."""
    v = grid.variables[0]
    N = v.n_points
    return jnp.array([f(_index_to_coord(v, i)) for i in range(N)])


# ===========================================================================
# 1. Known QTT Ranks
# ===========================================================================


@pytest.mark.algorithm
class TestKnownRanks:
    """Verify that well-known functions achieve their theoretical QTT ranks.

    On a dyadic grid x = sum_k b_k * 2^{-(k+1)}, functions that factor
    as products over bits have low QTT rank:
      - constant: rank 1
      - exp(a*x): rank 1 (product of per-bit factors)
      - x (linear): rank 2
      - x^n: rank n+1
      - sin(2*pi*x): rank 2
    """

    def test_constant_rank1(self):
        """Constant function has exact QTT rank 1."""
        grid = _make_1d_grid(0, 1, n_bits=10)
        data = jnp.ones(grid.variables[0].n_points) * 3.14
        qtt = fold_to_qtt(data, grid, tol=1e-14)
        assert max(qtt.bond_dims) == 1

    def test_exp_rank1(self):
        """exp(x) on dyadic grid has QTT rank 1.

        On a dyadic grid x = sum_k b_k * h_k, exp(x) = prod_k exp(b_k * h_k)
        which is a rank-1 tensor product over the bit sites.
        """
        grid = _make_1d_grid(0, 1, n_bits=10)
        data = _sample_on_grid(math.exp, grid)
        qtt = fold_to_qtt(data, grid, tol=1e-12)
        assert max(qtt.bond_dims) == 1

    def test_linear_rank2(self):
        """x on [0,1] has QTT rank 2."""
        grid = _make_1d_grid(0, 1, n_bits=10)
        data = _sample_on_grid(lambda x: x, grid)
        qtt = fold_to_qtt(data, grid, tol=1e-12)
        assert max(qtt.bond_dims) == 2

    def test_polynomial_rank_x2(self):
        """x^2 has QTT rank 3 (rank n+1 for x^n)."""
        grid = _make_1d_grid(0, 1, n_bits=10)
        data = _sample_on_grid(lambda x: x**2, grid)
        qtt = fold_to_qtt(data, grid, tol=1e-12)
        assert max(qtt.bond_dims) == 3

    def test_polynomial_rank_x3(self):
        """x^3 has QTT rank 4 (rank n+1 for x^n)."""
        grid = _make_1d_grid(0, 1, n_bits=10)
        data = _sample_on_grid(lambda x: x**3, grid)
        qtt = fold_to_qtt(data, grid, tol=1e-12)
        assert max(qtt.bond_dims) == 4

    def test_sin_rank2(self):
        """sin(2*pi*x) on [0,1] has QTT rank 2.

        sin(2*pi*x) = Im(exp(2*pi*i*x)) and the complex exponential
        of a linear function of the bits is rank 1 per component,
        giving real/imaginary parts with rank 2 jointly.
        """
        grid = _make_1d_grid(0, 1, n_bits=10)
        data = _sample_on_grid(lambda x: math.sin(2 * math.pi * x), grid)
        qtt = fold_to_qtt(data, grid, tol=1e-12)
        assert max(qtt.bond_dims) == 2


# ===========================================================================
# 2. Known Integrals
# ===========================================================================


@pytest.mark.algorithm
class TestKnownIntegrals:
    """Verify QTT integration against analytically known integrals.

    The integrate() method uses a simple Riemann sum (sum * dx), so the
    quadrature error scales as O(dx) = O(1/N).  With N=1024 (n_bits=10)
    on [0,1], dx ~ 1e-3, so we allow tolerances of ~1e-3.
    """

    def test_x_squared_integral(self):
        """integral_0^1 x^2 dx = 1/3."""
        grid = _make_1d_grid(0, 1, n_bits=10)
        data = _sample_on_grid(lambda x: x**2, grid)
        qtt = fold_to_qtt(data, grid, tol=1e-12)
        result = qtt.integrate()
        assert abs(result - 1.0 / 3) < 1e-3

    def test_exp_integral(self):
        """integral_0^1 exp(x) dx = e - 1."""
        grid = _make_1d_grid(0, 1, n_bits=10)
        data = _sample_on_grid(math.exp, grid)
        qtt = fold_to_qtt(data, grid, tol=1e-12)
        result = qtt.integrate()
        expected = math.e - 1
        assert abs(result - expected) < 1e-3

    def test_sin_full_period(self):
        """integral_0^{2pi} sin(x) dx = 0."""
        grid = _make_1d_grid(0, 2 * math.pi, n_bits=10)
        data = _sample_on_grid(math.sin, grid)
        qtt = fold_to_qtt(data, grid, tol=1e-12)
        result = qtt.integrate()
        assert abs(result) < 1e-3

    def test_gaussian_integral(self):
        """integral_{-3}^{3} exp(-x^2) dx ~ erf(3) * sqrt(pi)."""
        grid = _make_1d_grid(-3, 3, n_bits=10)
        data = _sample_on_grid(lambda x: math.exp(-(x**2)), grid)
        qtt = fold_to_qtt(data, grid, tol=1e-12)
        result = qtt.integrate()
        expected = float(erf(3)) * math.sqrt(math.pi)
        assert abs(result - expected) < 1e-3


# ===========================================================================
# 3. TCI vs Folding Consistency
# ===========================================================================


@pytest.mark.algorithm
class TestTCIAccuracy:
    """Verify that TCI and SVD folding agree on the same function."""

    def test_tci_exp(self):
        """TCI of exp(x) matches folded version."""
        grid = _make_1d_grid(0, 1, n_bits=8)
        # Folded (exact SVD)
        data = _sample_on_grid(math.exp, grid)
        qtt_fold = fold_to_qtt(data, grid, tol=1e-12)
        # TCI
        result = cross_interpolation(
            lambda x: math.exp(x[0]),
            grid,
            tol=1e-8,
            max_bond_dim=16,
            method="prrlu",
        )
        dense_fold = qtt_fold.to_dense()
        dense_tci = result.qtt.to_dense()
        assert jnp.allclose(dense_fold, dense_tci, atol=1e-4)

    def test_tci_polynomial(self):
        """TCI of x^3 - x matches folded version."""
        grid = _make_1d_grid(0, 1, n_bits=8)
        data = _sample_on_grid(lambda x: x**3 - x, grid)
        qtt_fold = fold_to_qtt(data, grid, tol=1e-12)
        result = cross_interpolation(
            lambda x: x[0] ** 3 - x[0],
            grid,
            tol=1e-8,
            max_bond_dim=16,
            method="prrlu",
        )
        dense_fold = qtt_fold.to_dense()
        dense_tci = result.qtt.to_dense()
        assert jnp.allclose(dense_fold, dense_tci, atol=1e-4)


# ===========================================================================
# 4. Operator Eigenvalues
# ===========================================================================


@pytest.mark.algorithm
class TestOperatorEigenvalues:
    """Verify that the Laplacian applied to sin gives the correct eigenvalue."""

    def test_laplacian_on_sine(self):
        """L sin(k*pi*x) ~ -(k*pi)^2 sin(k*pi*x) for k=1.

        Check at interior grid points (skip boundaries where finite
        difference stencil is one-sided / truncated).
        """
        n_bits = 8
        grid = _make_1d_grid(0, 1, n_bits=n_bits, include_endpoint=True)
        N = grid.variables[0].n_points
        k = 1
        # Build sin(pi*x) on the grid
        data = _sample_on_grid(lambda x: math.sin(k * math.pi * x), grid)
        qtt_sin = fold_to_qtt(data, grid, tol=1e-12)

        # Apply Laplacian
        lap = QTTMatrix.laplacian_1d(grid)
        qtt_lap = lap.apply(qtt_sin, method="naive", tol=1e-10, max_bond_dim=64)

        # Expected: -(k*pi)^2 * sin(k*pi*x)
        eigenvalue = -((k * math.pi) ** 2)
        expected = data * eigenvalue
        result = qtt_lap.to_dense()

        # Compare at interior points (skip boundaries where finite-difference
        # stencil is one-sided)
        margin = 4
        interior = slice(margin, N - margin)
        rel_err = float(
            jnp.max(jnp.abs(result[interior] - expected[interior]))
            / jnp.max(jnp.abs(expected[interior]))
        )
        assert rel_err < 0.05, f"Relative error {rel_err:.4e} exceeds 5%"


# ===========================================================================
# 5. Fourier / Parseval
# ===========================================================================


@pytest.mark.algorithm
class TestFourierProperties:
    """Verify DFT properties in QTT format."""

    def test_parseval_theorem(self):
        """||f||^2 = ||F(f)||^2 (Parseval's theorem).

        The DFT is unitary so the L2 norm of the coefficient vector
        should be preserved.
        """
        n_bits = 6  # 64 points -- small enough for DFT matrix
        grid = _make_1d_grid(0, 1, n_bits=n_bits)
        # Build a test function
        data = _sample_on_grid(lambda x: math.sin(2 * math.pi * x) + 0.5, grid)
        qtt_f = fold_to_qtt(data, grid, tol=1e-12)

        # DFT
        F = fourier_mpo(grid, max_bond_dim=64, tol=1e-12)
        qtt_Ff = F.apply(qtt_f, method="naive", tol=1e-10, max_bond_dim=64)

        norm_f = float(jnp.linalg.norm(qtt_f.to_dense()))
        norm_Ff = float(jnp.linalg.norm(qtt_Ff.to_dense()))
        assert abs(norm_f - norm_Ff) / norm_f < 1e-4, (
            f"|norm_f - norm_Ff| / norm_f = {abs(norm_f - norm_Ff) / norm_f:.4e}"
        )

    def test_dft_of_pure_frequency(self):
        """DFT of sin(2*pi*k*x) should have peaks at indices k and N-k.

        Using the unitary DFT convention:
          F[j,m] = exp(-2*pi*i*j*m/N) / sqrt(N)
        For sin(2*pi*k*n/N) = (exp(i*...) - exp(-i*...)) / (2i),
        the DFT has peaks at m=k and m=N-k.
        """
        n_bits = 6  # 64 points
        grid = _make_1d_grid(0, 1, n_bits=n_bits)
        N = grid.variables[0].n_points
        k = 3  # frequency index
        data = _sample_on_grid(lambda x: math.sin(2 * math.pi * k * x), grid)
        qtt_f = fold_to_qtt(data, grid, tol=1e-12)

        F = fourier_mpo(grid, max_bond_dim=64, tol=1e-12)
        qtt_Ff = F.apply(qtt_f, method="naive", tol=1e-10, max_bond_dim=64)

        spectrum = jnp.abs(qtt_Ff.to_dense())

        # The two largest peaks should be at indices k and N-k
        top2 = jnp.argsort(-spectrum)[:2]
        top2_set = set(int(t) for t in top2)
        assert top2_set == {k, N - k}, (
            f"Expected peaks at {{{k}, {N - k}}}, got {top2_set}"
        )


# ===========================================================================
# 6. 2D Integration (partial sum)
# ===========================================================================


@pytest.mark.algorithm
class TestMultiDimensional:
    """Verify multi-variable integration via partial contraction."""

    def test_separable_2d_integral(self):
        """integral_0^1 integral_0^1 x*y dx dy = 1/4.

        Build f(x,y) = x*y as a flat array on the grouped 2D grid,
        fold into QTT, then integrate both variables.
        """
        n_bits = 8
        gx = UniformGrid(0, 1, n_bits)
        gy = UniformGrid(0, 1, n_bits)
        grid = GridSpec(variables=(gx, gy), layout="grouped")
        N = gx.n_points  # 256

        # f(x,y) = x * y on a 2D grid (grouped layout: x-bits then y-bits)
        xs = jnp.array([_index_to_coord(gx, i) for i in range(N)])
        ys = jnp.array([_index_to_coord(gy, j) for j in range(N)])
        data = jnp.outer(xs, ys).ravel()
        qtt = fold_to_qtt(data, grid, tol=1e-12)

        result = qtt.integrate()
        expected = 0.25
        # Riemann sum quadrature error O(dx) ~ O(1/256) ~ 4e-3 per dimension
        assert abs(result - expected) < 5e-3, (
            f"integral of x*y = {result}, expected {expected}"
        )
