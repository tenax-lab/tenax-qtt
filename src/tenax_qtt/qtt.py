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

    def left_canonicalize(self) -> QTT:
        return QTT(mps=self.mps.left_canonicalize(), grid=self.grid)

    def right_canonicalize(self) -> QTT:
        return QTT(mps=self.mps.right_canonicalize(), grid=self.grid)

    def overlap(self, other: QTT) -> complex:
        return self.mps.overlap(other.mps)

    def entanglement_entropy(self, bond: int) -> float:
        return self.mps.entanglement_entropy(bond)

    def compute_singular_values(self) -> QTT:
        return QTT(mps=self.mps.compute_singular_values(), grid=self.grid)

    # -- Constructors --

    @classmethod
    def from_mps(cls, mps: FiniteMPS, grid: GridSpec) -> QTT:
        """Wrap an existing FiniteMPS with grid metadata."""
        L = num_sites(grid)
        if len(mps.tensors) != L:
            raise ValueError(f"MPS has {len(mps.tensors)} sites but grid requires {L}")
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
            self.evaluate(tuple(float(xs[i, j]) for j in range(xs.shape[1])))
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
            return self._partial_contract(variables, weighted=False)
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
            return self._partial_contract(variables, weighted=True)
        # Full integration: sum * product of dx for each variable
        total_dx = 1.0
        for v in self.grid.variables:
            total_dx *= v.dx
        s = self.sum()
        return s * total_dx

    def _partial_contract(self, variables: list[int], weighted: bool) -> QTT:
        """Contract out specified variable indices.

        For grouped layout, variables map to contiguous site ranges.
        Contract those sites with weight vectors (ones for sum, dx*ones
        for integrate), absorb pending bond matrices into surviving
        tensors, and build a new QTT with a reduced GridSpec.
        """
        if self.grid.layout != "grouped":
            raise NotImplementedError(
                "Partial contraction only supports grouped layout"
            )

        # Map variable indices to site ranges
        var_sites: dict[int, range] = {}
        offset = 0
        for vi, v in enumerate(self.grid.variables):
            var_sites[vi] = range(offset, offset + v.n_bits)
            offset += v.n_bits

        sites_to_contract = set()
        for vi in variables:
            sites_to_contract.update(var_sites[vi])

        # Contract specified sites, keep others
        L = len(self.tensors)
        new_tensors = []
        pending_matrix = None  # accumulated bond matrix from contracted sites

        for i in range(L):
            data = (
                self.tensors[i].todense()
                if hasattr(self.tensors[i], "todense")
                else self.tensors[i].data
            )

            if i in sites_to_contract:
                d = data.shape[1]
                v_idx = None
                for vi, sr in var_sites.items():
                    if i in sr:
                        v_idx = vi
                        break
                # Apply dx weight only on the first site of each variable
                if weighted and i == var_sites[v_idx].start:
                    weight = jnp.ones(d) * self.grid.variables[v_idx].dx
                else:
                    weight = jnp.ones(d)
                contracted = jnp.einsum("ijk,j->ik", data, weight)
                if pending_matrix is None:
                    pending_matrix = contracted
                else:
                    pending_matrix = pending_matrix @ contracted
            else:
                if pending_matrix is not None:
                    # Absorb pending matrix into this site tensor
                    data = jnp.einsum("pk,kjl->pjl", pending_matrix, data)
                    pending_matrix = None
                new_tensors.append(data)

        if pending_matrix is not None and new_tensors:
            # Absorb trailing contracted sites into last kept tensor
            last = new_tensors[-1]
            new_tensors[-1] = jnp.einsum("ijk,kl->ijl", last, pending_matrix)

        # Build new grid with remaining variables
        remaining_vars = tuple(
            self.grid.variables[vi]
            for vi in range(len(self.grid.variables))
            if vi not in variables
        )
        new_grid = GridSpec(variables=remaining_vars, layout=self.grid.layout)

        # Relabel tensors for new site numbering
        mps_tensors = []
        for i, data in enumerate(new_tensors):
            chi_l, d, chi_r = data.shape
            left_label = f"v_{i - 1}_{i}" if i == 0 else f"v{i - 1}_{i}"
            right_label = f"v{i}_{i + 1}"
            indices = (
                _trivial_index(chi_l, FlowDirection.IN, left_label),
                _trivial_index(d, FlowDirection.IN, f"p{i}"),
                _trivial_index(chi_r, FlowDirection.OUT, right_label),
            )
            mps_tensors.append(DenseTensor(data, indices))

        mps = FiniteMPS.from_tensors(mps_tensors)
        return QTT(mps=mps, grid=new_grid)

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
        from tenax_qtt.arithmetic import hadamard

        f_sq = hadamard(self, self)
        integral = f_sq.integrate()
        return float(jnp.sqrt(jnp.abs(integral)))
