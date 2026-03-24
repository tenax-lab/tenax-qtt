"""DFT operator in QTT format.

For v1, builds the full DFT matrix and folds it into MPO form via
``QTTMatrix._from_dense_matrix``.  Future: Chen-Lindsey analytical
construction for O(log N) bond dimension without forming the full matrix.
"""

from __future__ import annotations

import jax.numpy as jnp

from tenax_qtt.grid import GridSpec
from tenax_qtt.matrix import QTTMatrix


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

    Parameters
    ----------
    grid : GridSpec
        1-D grid specification.  Multi-dimensional grids are not yet
        supported.
    inverse : bool
        If ``True``, build the inverse DFT operator.
    max_bond_dim : int
        Maximum bond dimension kept during SVD folding.
    tol : float
        Truncation tolerance for SVD folding.

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
    N = v.n_points
    sign = 1.0 if inverse else -1.0

    # Build full DFT matrix: F[j, k] = exp(sign * 2*pi*i * j*k / N) / sqrt(N)
    j = jnp.arange(N)
    k = jnp.arange(N)
    F = jnp.exp(sign * 2j * jnp.pi * jnp.outer(j, k) / N) / jnp.sqrt(N)

    return QTTMatrix._from_dense_matrix(
        F, grid, grid, max_bond_dim=max_bond_dim, tol=tol
    )
