"""tenax-qtt: Quantic Tensor Train algorithms built on Tenax."""

from tenax_qtt.arithmetic import add, hadamard, recompress, scalar_multiply, subtract
from tenax_qtt.cross import QTTResult, cross_interpolation, estimate_error
from tenax_qtt.folding import fold_to_qtt
from tenax_qtt.fourier import fourier_mpo
from tenax_qtt.grid import (
    GridSpec,
    UniformGrid,
    batch_grid_to_sites,
    batch_sites_to_grid,
    grid_to_sites,
    local_dim,
    num_sites,
    site_permutation,
    sites_to_grid,
)
from tenax_qtt.matrix import QTTMatrix
from tenax_qtt.qtt import QTT

__all__ = [
    # Grid
    "UniformGrid",
    "GridSpec",
    "grid_to_sites",
    "sites_to_grid",
    "batch_grid_to_sites",
    "batch_sites_to_grid",
    "site_permutation",
    "num_sites",
    "local_dim",
    # QTT
    "QTT",
    # QTTMatrix
    "QTTMatrix",
    # Cross-interpolation
    "cross_interpolation",
    "QTTResult",
    "estimate_error",
    # Arithmetic
    "add",
    "subtract",
    "scalar_multiply",
    "hadamard",
    "recompress",
    # Fourier
    "fourier_mpo",
    # Folding
    "fold_to_qtt",
]
