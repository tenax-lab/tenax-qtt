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

from tenax_qtt.arithmetic import add, hadamard, recompress
from tenax_qtt.cross import cross_interpolation, estimate_error
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
        n_bits = 4  # 16 points -- exact DFT MPO at this size
        grid = _make_1d_grid(0, 1, n_bits=n_bits)
        # Build a test function
        data = _sample_on_grid(lambda x: math.sin(2 * math.pi * x) + 0.5, grid)
        qtt_f = fold_to_qtt(data, grid, tol=1e-12)

        # DFT (bond dim 4^2=16 at peak, well within budget)
        F = fourier_mpo(grid, max_bond_dim=256, tol=1e-12)
        qtt_Ff = F.apply(qtt_f, method="naive", tol=1e-10, max_bond_dim=256)

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
        n_bits = 4  # 16 points -- exact DFT at this size
        grid = _make_1d_grid(0, 1, n_bits=n_bits)
        N = grid.variables[0].n_points
        k = 3  # frequency index
        data = _sample_on_grid(lambda x: math.sin(2 * math.pi * k * x), grid)
        qtt_f = fold_to_qtt(data, grid, tol=1e-12)

        F = fourier_mpo(grid, max_bond_dim=256, tol=1e-12)
        qtt_Ff = F.apply(qtt_f, method="naive", tol=1e-10, max_bond_dim=256)

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


# ===========================================================================
# 7. Runge Function (Fernandez et al. 2024)
# ===========================================================================


@pytest.mark.algorithm
class TestRungeFunction:
    """Verify QTT/TCI approximation of the Runge function 1/(1+25x^2)."""

    def test_runge_tci(self):
        """1/(1+25x^2) on [-1,1] -- TCI should approximate well despite narrow peak."""
        grid = _make_1d_grid(-1, 1, n_bits=10)
        result = cross_interpolation(
            lambda x: 1.0 / (1.0 + 25.0 * x[0] ** 2),
            grid,
            tol=1e-6,
            max_bond_dim=20,
            method="tci2",
        )
        # Evaluate at x=0: should give 1.0
        val_0 = result.qtt.evaluate((0.0,))
        assert abs(val_0 - 1.0) < 0.05, f"Runge(0) = {val_0}, expected 1.0"

        # Evaluate at x=1: should give 1/26
        val_1 = result.qtt.evaluate((1.0,))
        expected_1 = 1.0 / 26.0
        assert abs(val_1 - expected_1) < 0.05, (
            f"Runge(1) = {val_1}, expected {expected_1}"
        )

        # Overall error should be modest
        err = estimate_error(
            result.qtt,
            lambda x: 1.0 / (1.0 + 25.0 * x[0] ** 2),
            n_samples=500,
        )
        assert err < 0.1, f"Runge TCI error {err:.4e} exceeds threshold"

    def test_runge_folding_vs_tci(self):
        """Folded and TCI versions should agree on the Runge function."""
        grid = _make_1d_grid(-1, 1, n_bits=8)

        def runge(x):
            return 1.0 / (1.0 + 25.0 * x**2)

        # Folded (exact SVD)
        data = _sample_on_grid(runge, grid)
        qtt_fold = fold_to_qtt(data, grid, tol=1e-12)

        # TCI
        result = cross_interpolation(
            lambda x: runge(x[0]),
            grid,
            tol=1e-6,
            max_bond_dim=20,
            method="tci2",
        )

        dense_fold = qtt_fold.to_dense()
        dense_tci = result.qtt.to_dense()
        # Both should approximate the same function
        assert jnp.allclose(dense_fold, dense_tci, atol=0.05)


# ===========================================================================
# 8. High-Dimensional (Fernandez et al. 2024)
# ===========================================================================


@pytest.mark.algorithm
class TestHighDimensional:
    """Verify QTT integration and TCI for multi-dimensional functions."""

    def test_4d_gaussian_integral(self):
        """integral exp(-sum(xi^2)) over [-3,3]^4 ~ (sqrt(pi)*erf(3))^4."""
        n_bits = 4  # 2^4 = 16 points per variable, 2^16 total
        variables = tuple(UniformGrid(-3, 3, n_bits) for _ in range(4))
        grid = GridSpec(variables=variables, layout="grouped")

        result = cross_interpolation(
            lambda x: float(np.exp(-(x[0] ** 2 + x[1] ** 2 + x[2] ** 2 + x[3] ** 2))),
            grid,
            tol=1e-6,
            max_bond_dim=16,
            method="tci2",
        )
        integral = result.qtt.integrate()
        expected = (math.sqrt(math.pi) * float(erf(3))) ** 4
        # Coarse grid (only 16 pts per dim) gives larger quadrature error
        rel_err = abs(integral - expected) / expected
        assert rel_err < 0.15, (
            f"4D Gaussian integral = {integral:.4f}, expected {expected:.4f}, "
            f"rel_err = {rel_err:.4e}"
        )

    def test_separable_4d_product(self):
        """f(x1,x2,x3,x4) = x1*x2*x3*x4, integral over [0,1]^4 = (1/2)^4 = 1/16."""
        n_bits = 4  # 16 points per variable
        variables = tuple(UniformGrid(0, 1, n_bits) for _ in range(4))
        grid = GridSpec(variables=variables, layout="grouped")

        # Build via folding -- 2^16 = 65536, manageable
        N_per = 1 << n_bits
        # Build the flat array for the 4D product function
        coords = []
        for v in grid.variables:
            coords.append(jnp.array([_index_to_coord(v, i) for i in range(N_per)]))
        # Outer product: x1 * x2 * x3 * x4
        data = jnp.einsum("i,j,k,l->ijkl", *coords).ravel()
        qtt = fold_to_qtt(data, grid, tol=1e-12)

        integral = qtt.integrate()
        expected = 1.0 / 16.0
        # 4D Riemann sum with 16 points per dim: quadrature error ~O(dx) per dim
        # compounds multiplicatively, so allow ~25% relative error
        rel_err = abs(integral - expected) / expected
        assert rel_err < 0.25, (
            f"4D product integral = {integral}, expected {expected}, "
            f"rel_err = {rel_err:.4e}"
        )


# ===========================================================================
# 9. Discontinuous Functions
# ===========================================================================


@pytest.mark.algorithm
class TestDiscontinuous:
    """Verify QTT behaviour on discontinuous and non-smooth functions."""

    def test_step_function(self):
        """Step function theta(x-0.5): QTT rank should be O(log N), not O(N)."""
        n_bits = 10  # N = 1024
        grid = _make_1d_grid(0, 1, n_bits=n_bits)
        N = grid.variables[0].n_points

        def step(x):
            return 1.0 if x >= 0.5 else 0.0

        data = _sample_on_grid(step, grid)
        qtt = fold_to_qtt(data, grid, tol=1e-12)

        # Bond dimension should be much less than N
        max_chi = max(qtt.bond_dims)
        assert max_chi < N // 4, (
            f"Step function max bond dim {max_chi} is too large (N={N})"
        )

        # Check accuracy: should be exact since step on a dyadic grid
        # maps to a finite-rank QTT
        dense = qtt.to_dense()
        assert jnp.allclose(dense, data, atol=1e-10)

    def test_absolute_value(self):
        """QTT of |x-0.5| on [0,1] should have moderate rank."""
        n_bits = 10
        grid = _make_1d_grid(0, 1, n_bits=n_bits)
        N = grid.variables[0].n_points

        data = _sample_on_grid(lambda x: abs(x - 0.5), grid)
        qtt = fold_to_qtt(data, grid, tol=1e-10)

        # Bond dim should be moderate (much less than N)
        max_chi = max(qtt.bond_dims)
        assert max_chi < N // 4, f"|x-0.5| max bond dim {max_chi} too large (N={N})"

        # Check accuracy
        dense = qtt.to_dense()
        rel_err = float(jnp.max(jnp.abs(dense - data)) / jnp.max(jnp.abs(data)))
        assert rel_err < 1e-6, f"|x-0.5| relative error {rel_err:.4e}"


# ===========================================================================
# 10. Hilbert Matrix (Oseledets 2011)
# ===========================================================================


@pytest.mark.algorithm
class TestHilbertMatrix:
    """Verify QTT representation of the Hilbert matrix."""

    def test_hilbert_low_rank(self):
        """Hilbert matrix H[i,j]=1/(i+j+1) in QTT format has low bond dimension."""
        n_bits = 3  # 8x8 matrix
        grid = _make_1d_grid(0, 1, n_bits=n_bits)
        N = grid.variables[0].n_points  # 8

        # Build 8x8 Hilbert matrix
        H = jnp.zeros((N, N))
        for i in range(N):
            for j in range(N):
                H = H.at[i, j].set(1.0 / (i + j + 1))

        qtt_mat = QTTMatrix.from_dense(H, grid, grid, tol=1e-12)

        # Check max bond dimension is small
        bond_dims = [jnp.array(t).shape[0] for t in qtt_mat.site_tensors[1:]]
        max_chi = max(bond_dims) if bond_dims else 1
        assert max_chi <= 8, f"Hilbert matrix max bond dim {max_chi} > 8"

        # Round-trip: dense -> QTTMatrix -> dense should be accurate
        H_reconstructed = qtt_mat.to_dense()
        assert jnp.allclose(H, H_reconstructed, atol=1e-8), (
            "Hilbert matrix round-trip failed"
        )

    def test_hilbert_apply(self):
        """H @ ones should give [sum_j 1/(i+j+1)] for each row i."""
        n_bits = 3  # 8x8
        grid = _make_1d_grid(0, 1, n_bits=n_bits)
        N = grid.variables[0].n_points

        H = jnp.zeros((N, N))
        for i in range(N):
            for j in range(N):
                H = H.at[i, j].set(1.0 / (i + j + 1))

        qtt_mat = QTTMatrix.from_dense(H, grid, grid, tol=1e-12)

        # Build ones vector as QTT
        ones = QTT.ones(grid)

        # Apply H @ ones
        result = qtt_mat.apply(ones, method="naive", tol=1e-10, max_bond_dim=32)
        result_dense = result.to_dense()

        # Expected: sum_j 1/(i+j+1) for i=0..7
        expected = jnp.array(
            [sum(1.0 / (i + j + 1) for j in range(N)) for i in range(N)]
        )
        assert jnp.allclose(result_dense, expected, atol=1e-4), (
            f"H @ ones mismatch: max err = "
            f"{float(jnp.max(jnp.abs(result_dense - expected))):.4e}"
        )


# ===========================================================================
# 11. Convolution via DFT
# ===========================================================================


@pytest.mark.algorithm
class TestConvolution:
    """Verify circular convolution via DFT in QTT format."""

    def test_convolution_via_dft(self):
        """F_inv(F(f) * F(g)) should approximate circular convolution.

        Using the shift property: convolving with a delta at index 0
        should return the original signal.
        """
        n_bits = 4  # 16 points -- exact DFT at this size
        grid = _make_1d_grid(0, 1, n_bits=n_bits)
        N = grid.variables[0].n_points

        # f = delta at index 0
        f_data = jnp.zeros(N)
        f_data = f_data.at[0].set(1.0)
        qtt_f = fold_to_qtt(f_data, grid, tol=1e-14)

        # g = sin(2*pi*x)
        g_data = _sample_on_grid(lambda x: math.sin(2 * math.pi * x), grid)
        qtt_g = fold_to_qtt(g_data, grid, tol=1e-12)

        # Forward DFT (bond dim 4^2=16 at peak, exact for n_bits=4)
        F = fourier_mpo(grid, max_bond_dim=256, tol=1e-12)
        F_inv = fourier_mpo(grid, inverse=True, max_bond_dim=256, tol=1e-12)

        F_f = F.apply(qtt_f, method="naive", tol=1e-10, max_bond_dim=256)
        F_g = F.apply(qtt_g, method="naive", tol=1e-10, max_bond_dim=256)

        # Hadamard product in Fourier domain (pointwise multiply)
        # For circular convolution: scale by sqrt(N) due to unitary convention
        F_fg = hadamard(F_f, F_g, tol=1e-10, max_bond_dim=256)

        # Inverse DFT, then scale by sqrt(N) for convolution theorem with
        # unitary DFT: conv(f,g) = sqrt(N) * F_inv(F(f) .* F(g))
        conv_result = F_inv.apply(F_fg, method="naive", tol=1e-10, max_bond_dim=256)
        conv_dense = conv_result.to_dense() * math.sqrt(N)

        # Delta at 0 convolving with g should give g (circular)
        assert jnp.allclose(jnp.real(conv_dense), g_data, atol=0.05), (
            f"Convolution with delta failed: max err = "
            f"{float(jnp.max(jnp.abs(jnp.real(conv_dense) - g_data))):.4e}"
        )


# ===========================================================================
# 12. Exponential Convergence
# ===========================================================================


@pytest.mark.algorithm
class TestExponentialConvergence:
    """Verify that QTT error decreases as bond dimension increases."""

    def test_bond_dim_vs_error(self):
        """Error should decrease as bond dimension increases for smooth functions."""
        n_bits = 8  # 256 points
        grid = _make_1d_grid(0, 1, n_bits=n_bits)
        data = _sample_on_grid(lambda x: math.sin(2 * math.pi * x), grid)

        errors = []
        bond_dims = [1, 2, 4, 8]
        for chi in bond_dims:
            qtt = fold_to_qtt(data, grid, max_bond_dim=chi, tol=1e-15)
            dense = qtt.to_dense()
            err = float(jnp.max(jnp.abs(dense - data)))
            errors.append(err)

        # Errors should be monotonically non-increasing
        for i in range(len(errors) - 1):
            assert errors[i + 1] <= errors[i] + 1e-14, (
                f"Error not decreasing: chi={bond_dims[i]} err={errors[i]:.4e}, "
                f"chi={bond_dims[i + 1]} err={errors[i + 1]:.4e}"
            )

        # Error at max bond dim 8 should be much smaller than at bond dim 1
        # sin(2*pi*x) has exact rank 2, so bond dim >= 2 should be exact
        assert errors[-1] < 1e-8, (
            f"Error at chi=8 is {errors[-1]:.4e}, expected near machine precision"
        )
        assert errors[0] > errors[-1], "Error at chi=1 should be larger than at chi=8"


# ===========================================================================
# 13. Layout Comparison
# ===========================================================================


@pytest.mark.algorithm
class TestLayoutComparison:
    """Verify that different grid layouts produce consistent function values."""

    def test_interleaved_vs_grouped_2d(self):
        """Same 2D function should give similar values with different layouts."""
        n_bits = 4  # 16 points per variable

        gx = UniformGrid(0, 2 * math.pi, n_bits)
        gy = UniformGrid(0, 2 * math.pi, n_bits)
        grid_grouped = GridSpec(variables=(gx, gy), layout="grouped")
        grid_interleaved = GridSpec(variables=(gx, gy), layout="interleaved")

        N = gx.n_points

        def f_2d(x, y):
            return math.sin(x) * math.cos(y)

        # Build grouped QTT
        xs = jnp.array([_index_to_coord(gx, i) for i in range(N)])
        ys = jnp.array([_index_to_coord(gy, j) for j in range(N)])
        data_grouped = jnp.array(
            [[f_2d(float(xs[i]), float(ys[j])) for j in range(N)] for i in range(N)]
        ).ravel()
        qtt_grouped = fold_to_qtt(data_grouped, grid_grouped, tol=1e-12)

        # Build interleaved QTT
        # For interleaved layout, the flat array must be ordered by the
        # interleaved bit pattern
        from tenax_qtt.grid import sites_to_grid

        L_interleaved = n_bits * 2  # total sites
        total_points = N * N
        data_interleaved = jnp.zeros(total_points)
        for flat_idx in range(total_points):
            # Decompose flat_idx into site indices (binary digits)
            bits = []
            remaining = flat_idx
            for _ in range(L_interleaved):
                bits.append(remaining % 2)
                remaining //= 2
            bits.reverse()
            sites = tuple(bits)
            coords = sites_to_grid(grid_interleaved, sites)
            val = f_2d(coords[0], coords[1])
            data_interleaved = data_interleaved.at[flat_idx].set(val)
        qtt_interleaved = fold_to_qtt(data_interleaved, grid_interleaved, tol=1e-12)

        # Evaluate at actual grid points to avoid snapping artifacts.
        # Both QTTs should agree at grid-aligned coordinates.
        test_indices = [(1, 2), (5, 10), (7, 3), (14, 0)]
        for ix, iy in test_indices:
            x_val = float(_index_to_coord(gx, ix))
            y_val = float(_index_to_coord(gy, iy))
            val_g = qtt_grouped.evaluate((x_val, y_val))
            val_i = qtt_interleaved.evaluate((x_val, y_val))
            expected = f_2d(x_val, y_val)
            # Both should match the true value at grid points
            assert abs(val_g - expected) < 1e-6, (
                f"Grouped evaluate({x_val:.3f}, {y_val:.3f}) = {val_g}, "
                f"expected {expected}"
            )
            assert abs(val_i - expected) < 1e-6, (
                f"Interleaved evaluate({x_val:.3f}, {y_val:.3f}) = {val_i}, "
                f"expected {expected}"
            )


# ===========================================================================
# 14. Recompression Quality
# ===========================================================================


@pytest.mark.algorithm
class TestRecompressionQuality:
    """Verify that recompression preserves accuracy."""

    def test_recompress_preserves_accuracy(self):
        """Recompressing a QTT should not significantly degrade accuracy."""
        grid = _make_1d_grid(0, 1, n_bits=8)
        data = _sample_on_grid(lambda x: math.sin(2 * math.pi * x), grid)

        # Build with generous tolerance (will have moderate bond dim)
        qtt = fold_to_qtt(data, grid, tol=1e-14)
        original_error = float(jnp.max(jnp.abs(qtt.to_dense() - data)))

        # Recompress -- sin(2*pi*x) has rank 2, so recompression should be lossless
        qtt_rc = recompress(qtt, tol=1e-10, max_bond_dim=4)
        rc_error = float(jnp.max(jnp.abs(qtt_rc.to_dense() - data)))

        # After recompression, error should still be small
        assert rc_error < 1e-6, (
            f"Recompression error {rc_error:.4e} too large "
            f"(original {original_error:.4e})"
        )

    def test_add_then_recompress(self):
        """Adding two QTTs inflates bond dim; recompressing should recover."""
        grid = _make_1d_grid(0, 1, n_bits=8)
        data_sin = _sample_on_grid(lambda x: math.sin(2 * math.pi * x), grid)
        data_cos = _sample_on_grid(lambda x: math.cos(2 * math.pi * x), grid)

        qtt_sin = fold_to_qtt(data_sin, grid, tol=1e-14)
        qtt_cos = fold_to_qtt(data_cos, grid, tol=1e-14)

        # Add without recompression
        qtt_sum = add(qtt_sin, qtt_cos, tol=0)
        max_chi_inflated = max(qtt_sum.bond_dims)

        # Recompress
        qtt_rc = recompress(qtt_sum, tol=1e-10, max_bond_dim=16)
        max_chi_compressed = max(qtt_rc.bond_dims)

        # Compressed bond dim should be smaller than inflated
        assert max_chi_compressed < max_chi_inflated, (
            f"Recompression did not reduce bond dim: "
            f"{max_chi_compressed} >= {max_chi_inflated}"
        )

        # sin(2*pi*x) + cos(2*pi*x) = sqrt(2)*sin(2*pi*x + pi/4)
        # which has QTT rank 2, so compressed bond dim should be small
        assert max_chi_compressed <= 4, (
            f"Compressed bond dim {max_chi_compressed} unexpectedly large"
        )

        # Check accuracy
        expected = data_sin + data_cos
        result = qtt_rc.to_dense()
        err = float(jnp.max(jnp.abs(result - expected)))
        assert err < 1e-6, f"Add+recompress error {err:.4e}"
