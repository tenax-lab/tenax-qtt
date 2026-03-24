"""QTT class wrapping FiniteMPS with grid semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from tenax import DenseTensor, FlowDirection, TensorIndex, U1Symmetry
from tenax.core.mps import FiniteMPS

from tenax_qtt.grid import GridSpec, local_dim, num_sites

if TYPE_CHECKING:
    from tenax import Tensor


def _trivial_index(dim: int, flow: FlowDirection, label: str) -> TensorIndex:
    sym = U1Symmetry()
    charges = np.zeros(dim, dtype=np.int32)
    return TensorIndex(sym, charges, flow, label=label)


def _make_constant_mps(grid: GridSpec, value: float) -> FiniteMPS:
    """Build a bond-dim-1 MPS with constant value at all grid points."""
    L = num_sites(grid)
    tensors = []
    for i in range(L):
        d = local_dim(grid, i)
        # Distribute the constant across sites: first site gets value, rest get 1
        fill = value if i == 0 else 1.0
        data = jnp.full((1, d, 1), fill)
        left_label = f"v_{i - 1}_{i}" if i == 0 else f"v{i - 1}_{i}"
        right_label = f"v{i}_{i + 1}"
        indices = (
            _trivial_index(1, FlowDirection.IN, left_label),
            _trivial_index(d, FlowDirection.IN, f"p{i}"),
            _trivial_index(1, FlowDirection.OUT, right_label),
        )
        tensors.append(DenseTensor(data, indices))
    return FiniteMPS.from_tensors(tensors)


@dataclass(frozen=True)
class QTT:
    """Quantic Tensor Train: a function on a grid stored as an MPS."""

    mps: FiniteMPS
    grid: GridSpec

    # -- MPS delegation --

    @property
    def tensors(self) -> list[Tensor]:
        return self.mps.tensors

    @property
    def bond_dims(self) -> list[int]:
        return self.mps.bond_dims

    @property
    def orth_center(self) -> int | None:
        return self.mps.orth_center

    @property
    def singular_values(self) -> list:
        return self.mps.singular_values

    @property
    def log_norm(self) -> float:
        return self.mps.log_norm

    def canonicalize(self, center: int) -> QTT:
        return QTT(mps=self.mps.canonicalize(center), grid=self.grid)

    def norm(self) -> float:
        return self.mps.norm()

    # -- Constructors --

    @classmethod
    def from_mps(cls, mps: FiniteMPS, grid: GridSpec) -> QTT:
        """Wrap an existing FiniteMPS with grid metadata."""
        L = num_sites(grid)
        if len(mps.tensors) != L:
            raise ValueError(
                f"MPS has {len(mps.tensors)} sites but grid requires {L}"
            )
        return cls(mps=mps, grid=grid)

    @classmethod
    def zeros(cls, grid: GridSpec) -> QTT:
        """Bond-dim-1 QTT representing the zero function."""
        return cls(mps=_make_constant_mps(grid, 0.0), grid=grid)

    @classmethod
    def ones(cls, grid: GridSpec) -> QTT:
        """Bond-dim-1 QTT representing f(x) = 1."""
        return cls(mps=_make_constant_mps(grid, 1.0), grid=grid)

    @classmethod
    def from_cross(
        cls,
        f,
        grid: GridSpec,
        tol: float = 1e-8,
        max_bond_dim: int = 64,
        batch: bool = False,
        **kwargs,
    ):
        """Build a QTT via cross-interpolation (delegates to cross.py).

        Returns a ``QTTResult`` containing the QTT and diagnostics.
        """
        from tenax_qtt.cross import cross_interpolation

        return cross_interpolation(
            f, grid, tol=tol, max_bond_dim=max_bond_dim, batch=batch, **kwargs
        )

    @classmethod
    def from_dense(
        cls,
        data,
        grid: GridSpec,
        max_bond_dim: int | None = None,
        tol: float = 1e-8,
    ) -> QTT:
        """Build a QTT from a dense array via SVD folding."""
        from tenax_qtt.folding import fold_to_qtt

        return fold_to_qtt(data, grid, max_bond_dim=max_bond_dim, tol=tol)

    # -- Evaluation --

    def evaluate(self, x: tuple[float, ...]) -> complex:
        """Evaluate QTT at a single continuous-domain point."""
        from tenax_qtt.grid import grid_to_sites

        sites = grid_to_sites(self.grid, x)
        # Contract MPS by selecting physical index at each site
        result = jnp.array([[1.0]])  # row vector (1, 1)
        for i, t in enumerate(self.tensors):
            # t has shape (chi_left, d_phys, chi_right) as a dense array
            data = t.todense() if hasattr(t, "todense") else t.data
            s = sites[i]
            result = result @ data[:, s, :]
        return complex(result[0, 0])

    def evaluate_batch(self, xs: jax.Array) -> jax.Array:
        """Vectorized evaluation at multiple points."""
        vals = [
            self.evaluate(
                tuple(float(xs[i, j]) for j in range(xs.shape[1]))
            )
            for i in range(xs.shape[0])
        ]
        return jnp.array(vals)

    # -- Dense expansion --

    def to_dense(self) -> jax.Array:
        """Expand QTT to full dense array."""
        result = None
        for i, t in enumerate(self.tensors):
            data = t.todense() if hasattr(t, "todense") else t.data
            if result is None:
                # shape: (1, d, chi_right) -> (d, chi_right)
                result = data[0]
            else:
                # result: (..., chi_left), data: (chi_left, d, chi_right)
                result = jnp.tensordot(result, data, axes=([-1], [0]))
        # Remove trailing dim-1 bond
        return result.reshape(-1)

    # -- Summation and integration --

    def sum(self, variables: list[int] | None = None) -> QTT | complex:
        """Sum over specified variables (or all) without grid spacing."""
        if variables is not None:
            raise NotImplementedError("Partial summation not yet implemented")
        # Full sum: contract each site with all-ones vector
        result = jnp.array([[1.0]])
        for t in self.tensors:
            data = t.todense() if hasattr(t, "todense") else t.data
            d = data.shape[1]
            ones = jnp.ones(d)
            # Contract physical index with ones: (chi_l, d, chi_r) . (d,) -> (chi_l, chi_r)
            contracted = jnp.einsum("ijk,j->ik", data, ones)
            result = result @ contracted
        return complex(result[0, 0])

    def integrate(self, variables: list[int] | None = None) -> QTT | complex:
        """Integrate over specified variables using trapezoidal quadrature."""
        if variables is not None:
            raise NotImplementedError("Partial integration not yet implemented")
        # Full integration: sum * product of dx for each variable
        total_dx = 1.0
        for v in self.grid.variables:
            total_dx *= v.dx
        s = self.sum()
        return s * total_dx

    # -- Arithmetic dunders --

    def __add__(self, other: QTT) -> QTT:
        from tenax_qtt.arithmetic import add

        return add(self, other)

    def __sub__(self, other: QTT) -> QTT:
        from tenax_qtt.arithmetic import subtract

        return subtract(self, other)

    def __mul__(self, scalar: complex) -> QTT:
        from tenax_qtt.arithmetic import scalar_multiply

        return scalar_multiply(self, scalar)

    def __rmul__(self, scalar: complex) -> QTT:
        return self.__mul__(scalar)

    def norm_l2(self) -> float:
        """Continuous L2 norm: sqrt(integral |f(x)|^2 dx)."""
        # For a real-valued QTT, ||f||^2 = <MPS|MPS> * dx
        total_dx = 1.0
        for v in self.grid.variables:
            total_dx *= v.dx
        mps_norm_sq = self.mps.norm() ** 2
        return float(jnp.sqrt(mps_norm_sq * total_dx))
