"""SVD-based QTT construction from dense data."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from tenax import DenseTensor, FlowDirection, TensorIndex, U1Symmetry, svd

from tenax_qtt.grid import GridSpec, local_dim, num_sites
from tenax_qtt.qtt import QTT

_sym = U1Symmetry()


def _trivial_index(dim: int, flow: FlowDirection, label: str) -> TensorIndex:
    return TensorIndex(_sym, np.zeros(dim, dtype=np.int32), flow, label=label)


def fold_to_qtt(
    data: jnp.ndarray,
    grid: GridSpec,
    max_bond_dim: int | None = None,
    tol: float = 1e-8,
) -> QTT:
    """Reshape dense array into QTT via left-to-right SVD sweep."""
    from tenax.core.mps import FiniteMPS

    L = num_sites(grid)
    dims = [local_dim(grid, i) for i in range(L)]

    # Reshape flat array into (d0, d1, ..., dL-1) tensor
    tensor = data.reshape(dims)

    # Left-to-right SVD sweep
    tensors = []
    remainder = tensor
    chi_left = 1
    for i in range(L - 1):
        d = dims[i]
        # Reshape to matrix: (chi_left * d, remaining_dims...)
        mat_shape = (chi_left * d, -1)
        mat = remainder.reshape(mat_shape)

        # Create DenseTensor for SVD
        left_idx = _trivial_index(mat.shape[0], FlowDirection.IN, "left")
        right_idx = _trivial_index(mat.shape[1], FlowDirection.OUT, "right")
        mat_tensor = DenseTensor(mat, (left_idx, right_idx))

        U, s, Vh, _ = svd(
            mat_tensor, ["left"], ["right"],
            new_bond_label="bond",
            max_singular_values=max_bond_dim,
            max_truncation_err=tol,
        )
        chi_new = len(s)

        # Extract U data and reshape to (chi_left, d, chi_new)
        u_data = U.todense()
        site_data = u_data.reshape(chi_left, d, chi_new)

        # Build properly labeled site tensor
        left_label = f"v_{i - 1}_{i}" if i == 0 else f"v{i - 1}_{i}"
        right_label = f"v{i}_{i + 1}"
        site_indices = (
            _trivial_index(chi_left, FlowDirection.IN, left_label),
            _trivial_index(d, FlowDirection.IN, f"p{i}"),
            _trivial_index(chi_new, FlowDirection.OUT, right_label),
        )
        tensors.append(DenseTensor(site_data, site_indices))

        # Remainder = diag(s) @ Vh
        vh_data = Vh.todense()
        remainder = (jnp.diag(s) @ vh_data).reshape(
            chi_new, *[dims[j] for j in range(i + 1, L)]
        )
        chi_left = chi_new

    # Last site: remainder has shape (chi_left, d_last)
    d_last = dims[-1]
    site_data = remainder.reshape(chi_left, d_last, 1)
    left_label = f"v{L - 2}_{L - 1}"
    right_label = f"v{L - 1}_{L}"
    site_indices = (
        _trivial_index(chi_left, FlowDirection.IN, left_label),
        _trivial_index(d_last, FlowDirection.IN, f"p{L - 1}"),
        _trivial_index(1, FlowDirection.OUT, right_label),
    )
    tensors.append(DenseTensor(site_data, site_indices))

    mps = FiniteMPS.from_tensors(tensors)
    return QTT(mps=mps, grid=grid)
