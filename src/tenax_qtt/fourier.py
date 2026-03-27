"""DFT operator in QTT format.

Constructs the DFT MPO analytically via binary phase decomposition,
avoiding the O(N^2) dense matrix.  The DFT phase exp(sign * 2*pi*i *
j*k / N) factors over binary digits of j and k, and the site tensors
are built by accumulating twiddle-factor phases left-to-right.  An
SVD compression pass then yields the optimal bond dimensions.

For large grids where the exact bond dimension (4^{R/2}) is too large,
the ``max_precision`` parameter truncates the phase accumulators to *p*
bits, capping the bond dimension at 4^p and introducing a bounded
approximation error of O(R * 2^{-p}).

References
----------
* Oseledets, *SIAM J. Sci. Comput.* 33 (2011) -- QTT format
* Fernandez et al., arXiv:2407.02454 (2024) -- DFT in tensor-train form
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from tenax_qtt.grid import GridSpec
from tenax_qtt.matrix import QTTMatrix

# ---------------------------------------------------------------------------
# Analytical DFT MPO construction
# ---------------------------------------------------------------------------


def _build_fourier_site_tensors(
    R: int,
    sign: float = -1.0,
    max_precision: int | None = None,
) -> list[np.ndarray]:
    """Build DFT MPO site tensors via binary phase decomposition.

    The DFT matrix F[j, k] = exp(sign * 2*pi*i * j*k / N) / sqrt(N)
    with N = 2^R is decomposed using the binary representations
    j = sum_a j_a * 2^{R-1-a} and k = sum_b k_b * 2^{R-1-b} (MSB-first).

    The phase j*k/N factors into pairwise terms exp(sign * 2*pi*i *
    j_a * k_b / 2^{a+b-R+2}) for pairs (a, b) with a+b >= R-1.
    Terms with a+b < R-1 contribute integer multiples of 2*pi and
    therefore vanish.

    The bond state at bond position *a* encodes two phase accumulators
    (J_a, K_a) -- the integers formed by the past j- and k-bits,
    respectively.  As site *a* is processed, phase contributions from
    cross-interactions between the current (j_a, k_a) and accumulated
    past bits are applied, and the accumulators are updated.

    Parameters
    ----------
    R : int
        Number of QTT sites.  Grid has N = 2^R points.
    sign : float
        -1 for forward DFT, +1 for inverse.
    max_precision : int or None
        Maximum bits kept in each accumulator.  ``None`` means full
        precision (bond dimension up to 4^{R/2} after compression).
        Setting ``max_precision = p`` caps the raw bond dimension at
        4^p and introduces approximation error O(R * 2^{-p}).

    Returns
    -------
    list[np.ndarray]
        Site tensors of shape ``(chi_l, 2, 2, chi_r)`` (complex128).
    """
    N = 1 << R
    tensors: list[np.ndarray] = []

    for a in range(R):
        # Number of accumulator bits carried on each bond.
        n_bits_l = (
            0 if a == 0 else (a if max_precision is None else min(a, max_precision))
        )
        n_bits_r = (
            0
            if a == R - 1
            else (a + 1 if max_precision is None else min(a + 1, max_precision))
        )

        chi_l = 1 if n_bits_l == 0 else (1 << n_bits_l) ** 2
        chi_r = 1 if n_bits_r == 0 else (1 << n_bits_r) ** 2
        dim_l = max(1, 1 << n_bits_l)
        dim_r = max(1, 1 << n_bits_r)

        W = np.zeros((chi_l, 2, 2, chi_r), dtype=np.complex128)

        for wl in range(chi_l):
            if a == 0:
                J_prev, K_prev = 0, 0
            else:
                J_prev = wl // dim_l
                K_prev = wl % dim_l

            for ja in range(2):
                for ka in range(2):
                    J_full = 2 * J_prev + ja
                    K_full = 2 * K_prev + ka

                    # Accumulate phase from interactions completed at this site.
                    phase = 0.0

                    # j_a with past k_b (b < a), pair (a, b):
                    if ja == 1:
                        for b in range(a):
                            if a + b >= R - 1:
                                m = a + b - R + 2
                                bit_pos = a - 1 - b
                                if bit_pos < n_bits_l:
                                    phase += ((K_prev >> bit_pos) & 1) / (1 << m)

                    # k_a with past j_{a'} (a' < a), pair (a', a):
                    if ka == 1:
                        for ap in range(a):
                            if ap + a >= R - 1:
                                m = ap + a - R + 2
                                bit_pos = a - 1 - ap
                                if bit_pos < n_bits_l:
                                    phase += ((J_prev >> bit_pos) & 1) / (1 << m)

                    # j_a * k_a (local pair), contributes when 2a >= R-1:
                    if ja == 1 and ka == 1 and 2 * a >= R - 1:
                        m = 2 * a - R + 2
                        phase += 1.0 / (1 << m)

                    total_phase = np.exp(sign * 2j * np.pi * phase)

                    if a == R - 1:
                        wr = 0
                    else:
                        J_bond = J_full % dim_r
                        K_bond = K_full % dim_r
                        wr = J_bond * dim_r + K_bond

                    W[wl, ja, ka, wr] += total_phase

        tensors.append(W)

    # Absorb 1/sqrt(N) normalization into the first site tensor.
    tensors[0] = tensors[0] / np.sqrt(N)
    return tensors


# ---------------------------------------------------------------------------
# SVD compression
# ---------------------------------------------------------------------------


def _svd_compress_site_tensors(
    tensors: list[np.ndarray],
    max_bond_dim: int | None = None,
    tol: float = 1e-12,
) -> list[np.ndarray]:
    """Two-pass (LR then RL) SVD compression of MPO site tensors.

    Parameters
    ----------
    tensors : list[np.ndarray]
        Site tensors, each of shape ``(chi_l, d_out, d_in, chi_r)``.
    max_bond_dim : int or None
        Hard cap on bond dimension.
    tol : float
        Relative singular-value truncation threshold.

    Returns
    -------
    list[np.ndarray]
        Compressed site tensors.
    """
    tensors = [t.copy() for t in tensors]
    R = len(tensors)

    def _keep(s: np.ndarray) -> int:
        k = len(s)
        if tol > 0 and len(s) > 0:
            k = max(1, int(np.sum(s > tol * s[0])))
        if max_bond_dim is not None:
            k = min(k, max_bond_dim)
        return k

    # Left-to-right sweep: put each site in left-canonical form.
    for a in range(R - 1):
        W = tensors[a]
        cl, do, di, cr = W.shape
        mat = W.reshape(cl * do * di, cr)
        u, s, vh = np.linalg.svd(mat, full_matrices=False)
        keep = _keep(s)
        tensors[a] = u[:, :keep].reshape(cl, do, di, keep)
        svh = np.diag(s[:keep]) @ vh[:keep]
        nW = tensors[a + 1]
        tensors[a + 1] = (svh @ nW.reshape(nW.shape[0], -1)).reshape(
            keep,
            nW.shape[1],
            nW.shape[2],
            nW.shape[3],
        )

    # Right-to-left sweep: compress from the right.
    for a in range(R - 1, 0, -1):
        W = tensors[a]
        cl, do, di, cr = W.shape
        mat = W.reshape(cl, do * di * cr)
        u, s, vh = np.linalg.svd(mat, full_matrices=False)
        keep = _keep(s)
        tensors[a] = vh[:keep].reshape(keep, do, di, cr)
        us = u[:, :keep] @ np.diag(s[:keep])
        pW = tensors[a - 1]
        tensors[a - 1] = (pW.reshape(-1, pW.shape[3]) @ us).reshape(
            pW.shape[0],
            pW.shape[1],
            pW.shape[2],
            keep,
        )

    return tensors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Threshold: for R above this, auto-choose a precision cap for the accumulators
# to keep memory and time reasonable.
_FULL_PRECISION_THRESHOLD = 14


def fourier_mpo(
    grid: GridSpec,
    inverse: bool = False,
    max_bond_dim: int = 64,
    tol: float = 1e-12,
) -> QTTMatrix:
    """Build the DFT operator in QTT (MPO) format.

    Uses the unitary convention:

        F[j, k] = exp(sign * 2*pi*i * j * k / N) / sqrt(N)

    where *sign* is -1 for the forward transform and +1 for the inverse.
    This ensures ``F @ F_inv = I``.

    The MPO is constructed analytically from the binary phase decomposition
    of the DFT, then compressed via SVD.  No dense N x N matrix is formed.

    Parameters
    ----------
    grid : GridSpec
        1-D grid specification.  Multi-dimensional grids are not yet
        supported.
    inverse : bool
        If ``True``, build the inverse DFT operator.
    max_bond_dim : int
        Maximum bond dimension kept during SVD compression.
    tol : float
        Truncation tolerance for SVD compression.

    Returns
    -------
    QTTMatrix
        The DFT operator in MPO form.

    Raises
    ------
    NotImplementedError
        If ``grid`` has more than one variable (multi-D Fourier).
    """
    if len(grid.variables) != 1:
        raise NotImplementedError("Multi-D Fourier not yet implemented")

    v = grid.variables[0]
    R = v.n_bits
    sign = 1.0 if inverse else -1.0

    # Choose accumulator precision.  For small R the exact construction
    # is fast; for large R we cap the precision to keep memory bounded.
    if R <= _FULL_PRECISION_THRESHOLD:
        max_precision = None  # full precision
    else:
        # Choose p so that the raw bond dim (4^p) is at most 4 * max_bond_dim,
        # giving the SVD room to compress further.
        target = max(4, 4 * (max_bond_dim or 64))
        max_precision = max(1, math.ceil(math.log2(target) / 2))

    # Build site tensors analytically (numpy).
    site_tensors = _build_fourier_site_tensors(R, sign, max_precision)

    # SVD compress.
    site_tensors = _svd_compress_site_tensors(
        site_tensors,
        max_bond_dim=max_bond_dim,
        tol=tol,
    )

    # Convert numpy arrays to jax arrays for QTTMatrix.
    jax_tensors = [jnp.array(t) for t in site_tensors]

    return QTTMatrix(site_tensors=jax_tensors, grid_in=grid, grid_out=grid)
